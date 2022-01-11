import os
import shutil
import time
import json
import pause
from datetime import datetime
import numpy as np
import cv2
from vicon_dssdk import ViconDataStream
import dv

from events import *
from frames import *
from poses import *


def get_vicon_network_poses(record_time, address, port, props_markers, poses_file_name):
    poses_file, data = create_pytables_poses_file(poses_file_name, 0, props_markers)

    client = ViconDataStream.Client()
    client.Connect(f'{address}:{port}')
    client.EnableMarkerData()
    client.EnableSegmentData()

    # wait for signal to begin recording
    print('waiting for signal...')
    while True:
        if not client.GetFrame():
            continue

        try:
            prop_quality = client.GetObjectQuality('jpt_wand')
        except ViconDataStream.DataStreamException:
            prop_quality = None

        # if vicon wand signal is present
        if prop_quality is not None:
            break

    # compute start and stop times
    start_time = datetime.now().timestamp() + 3
    stop_time = start_time + record_time

    # wait until start time
    start_timestamp = int(start_time * 1000000)
    print(f'vicon start timestamp (usec): {start_timestamp:,}')
    pause.until(start_time)

    # begin frame collection
    current_time = start_time
    while current_time < stop_time:
        current_time = datetime.now().timestamp()

        if not client.GetFrame():
            continue

        timestamp = int((current_time - start_time) * 1000000)
        data['timestamp'].append([timestamp])

        prop_names = props_markers.keys()
        for prop_name in prop_names:
            marker_names = props_markers[prop_name].keys()

            try:
                prop_quality = client.GetObjectQuality(prop_name)
            except ViconDataStream.DataStreamException:
                prop_quality = None

            if prop_quality is not None:
                root_segment = client.GetSubjectRootSegmentName(prop_name)

                rotation = np.array(client.GetSegmentGlobalRotationMatrix(prop_name, root_segment)[0])
                data['vicon_rotation'][prop_name].append([rotation])

                translation = np.array(client.GetSegmentGlobalTranslation(prop_name, root_segment)[0])
                data['vicon_translation'][prop_name].append([translation[:, np.newaxis]])

                for marker_name in marker_names:
                    marker = np.array(client.GetMarkerGlobalTranslation(prop_name, marker_name)[0])
                    data['vicon_markers'][prop_name][marker_name].append([marker[:, np.newaxis]])

            else:
                rotation = np.full((3, 3), np.nan, dtype='float64')
                data['vicon_rotation'][prop_name].append([rotation])

                translation = np.full(3, np.nan, dtype='float64')
                data['vicon_translation'][prop_name].append([translation[:, np.newaxis]])

                for marker_name in marker_names:
                    marker = np.full(3, np.nan, dtype='float64')
                    data['vicon_markers'][prop_name][marker_name].append([marker[:, np.newaxis]])

    client.Disconnect()

    poses_file.close()
    return


def WITHOUT_NUMPY_get_dvs_aedat_file_events(camera, n_cameras, aedat_file_name, mtx, dist, events_file_name):
    events_file, events_data = create_pytables_events_file(events_file_name, n_cameras)

    print(f'get camera {camera} event data')
    with dv.AedatFile(aedat_file_name) as f:

        if camera == 0:
            events = f['events']
        else:
            events = f[f'events_{camera}']

        for event in events:
            # undistort event
            event_xy_raw = np.array([event.x, event.y], dtype='float64')
            event_xy_undistorted = cv2.undistortPoints(
                event_xy_raw, mtx, dist, None, mtx)[0, 0]

            events_data[f'timestamp_{camera}'].append([event.timestamp])
            events_data[f'polarity_{camera}'].append([event.polarity])
            events_data[f'xy_raw_{camera}'].append([event_xy_raw])
            events_data[f'xy_undistorted_{camera}'].append([event_xy_undistorted])

    events_file.close()
    return


def get_dvs_aedat_file_events(camera, n_cameras, aedat_file_name, mtx, dist, events_file_name):
    events_file, events_data = create_pytables_events_file(events_file_name, n_cameras)

    print(f'get camera {camera} event data')
    with dv.AedatFile(aedat_file_name) as f:

        if camera == 0:
            events = f['events']
        else:
            events = f[f'events_{camera}']
        events = np.hstack([packet for packet in events.numpy()])

        for timestamp, polarity, x, y in zip(events['timestamp'], events['polarity'], events['x'], events['y']):
            # undistort event
            event_xy_raw = np.array([x, y], dtype='float64')
            event_xy_undistorted = cv2.undistortPoints(
                event_xy_raw, mtx, dist, None, mtx)[0, 0]

            events_data[f'timestamp_{camera}'].append([timestamp])
            events_data[f'polarity_{camera}'].append([polarity])
            events_data[f'xy_raw_{camera}'].append([event_xy_raw])
            events_data[f'xy_undistorted_{camera}'].append([event_xy_undistorted])

    events_file.close()
    return


def get_dvs_aedat_file_frames(camera, n_cameras, aedat_file_name, mtx, dist, frames_file_name):
    frames_file, frames_data = create_pytables_frames_file(frames_file_name, n_cameras)

    print(f'get camera {camera} frame data')
    with dv.AedatFile(aedat_file_name) as f:

        if camera == 0:
            frames = f['frames']
        else:
            frames = f[f'frames_{camera}']

        for frame in frames:
            # undistort frame
            frame_image_raw = frame.image
            frame_image_undistorted = cv2.undistort(
                frame_image_raw, mtx, dist, None, mtx)

            frames_data[f'timestamp_{camera}'].append([frame.timestamp])
            frames_data[f'image_raw_{camera}'].append([frame_image_raw])
            frames_data[f'image_undistorted_{camera}'].append([frame_image_undistorted])

    frames_file.close()
    return


def record():

    n_cameras = 2

    vicon_record_time = 30  # in seconds

    test_initials = 'jpt'
    test_scenario = 'floating_kth_hammer'
    test_number = 0

    date_data = time.strftime('%Y%m%d')
    #date_data = 20210929
    path_data = f'./data/{date_data}_{test_initials}_{test_scenario}/{test_number:04}'

    #date_camera = time.strftime('%Y%m%d')
    #date_camera = 20210922
    #path_camera = f'./camera_calibration/{date_camera}'
    path_camera = sorted(os.listdir('./camera_calibration'))[-1]
    path_camera = f'./camera_calibration/{path_camera}'

    #date_projection = time.strftime('%Y%m%d')
    #date_projection = 20210922
    #path_projection = f'./projection_calibration/{date_projection}'
    path_projection = sorted(os.listdir('./projection_calibration'))[-1]
    path_projection = f'./projection_calibration/{path_projection}'

    path_props = './props'

    path_aedat = 'J:/dv_recording'

    # servers
    vicon_address, vicon_port = '127.0.0.1', 801

    # comment out as required
    prop_names = [
        #'kth_hammer',
        'kth_screwdriver',
        #'kth_spanner',
        #'jpt_mallet',
        #'jpt_screwdriver',
    ]

    props_markers = {}
    for prop_name in prop_names:
        with open(f'{path_props}/{prop_name}_markers.json', 'r') as marker_file:
            props_markers[prop_name] = json.load(marker_file)


    ##################################################################

    # === CAMERA CALIBRATION FILES ===

    dv_cam_mtx_file_name = [f'{path_camera}/camera_{i}_matrix.npy' for i in range(n_cameras)]
    dv_cam_mtx = [np.load(file_name) for file_name in dv_cam_mtx_file_name]

    dv_cam_dist_file_name = [f'{path_camera}/camera_{i}_distortion_coefficients.npy' for i in range(n_cameras)]
    dv_cam_dist = [np.load(file_name) for file_name in dv_cam_dist_file_name]


    ##################################################################

    # === DATA FILES ===

    os.makedirs(path_data, exist_ok=True)
    for f in os.listdir(path_data):
        os.remove(f'{path_data}/{f}')
    # for f in os.listdir(path_aedat):
    #     os.remove(f'{path_aedat}/{f}')

    events_file_name = [f'{path_data}/raw_event_{i}.h5' for i in range(n_cameras)]
    frames_file_name = [f'{path_data}/raw_frame_{i}.h5' for i in range(n_cameras)]
    poses_file_name = f'{path_data}/raw_pose.h5'

    ##################################################################


    info_json = {
        'start_time': datetime.now().timestamp(),
        'camera_calibration_path': path_camera,
        'projection_calibration_path': path_projection,
        'prop_marker_files': {},
        'prop_mesh_files': {},
    }

    for prop_name in prop_names:
        info_json['prop_marker_files'][prop_name] = f'{path_props}/{prop_name}_markers.json'
        info_json['prop_mesh_files'][prop_name] = f'{path_props}/{prop_name}_mesh.stl'

    with open(f'{path_data}/info.json', 'w') as info_json_file:
        json.dump(info_json, info_json_file)


    # === BEGIN ===
    print('=== begin recording ===')
    print('start DV recording now and give the wand signal')

    get_vicon_network_poses(vicon_record_time, vicon_address, vicon_port, props_markers, poses_file_name)

    print('\n\n\n\n\n')
    input('stop the dv recording and hit enter')

    aedat_file_name = f'{path_aedat}/{sorted(os.listdir(path_aedat))[-1]}'
    shutil.copy(aedat_file_name, path_data)

    for i in range(n_cameras):
        get_dvs_aedat_file_events(i, n_cameras, aedat_file_name, dv_cam_mtx[i], dv_cam_dist[i], events_file_name[i])
        get_dvs_aedat_file_frames(i, n_cameras, aedat_file_name, dv_cam_mtx[i], dv_cam_dist[i], frames_file_name[i])

    print('\n\n\n\n\n')
    print('=== end recording ===')


if __name__ == '__main__':
    record()
