from datetime import datetime
import numpy as np
import cv2
from vicon_dssdk import ViconDataStream
import dv


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


def get_dvs_network_events(camera, record_time, address, port, mtx, dist, events_file_name):
    events_file, events_data = create_pytables_events_file(events_file_name, n_cameras)

    with dv.NetworkEventInput(address=address, port=port) as f:
        event = next(f)
        stop_timestamp = event.timestamp + int(record_time * 1000000)

        for event in f:
            if event.timestamp >= stop_timestamp:
                break

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


def get_dvs_network_frames(camera, record_time, address, port, mtx, dist, frames_file_name):
    frames_file, frames_data = create_pytables_frames_file(frames_file_name, n_cameras)

    with dv.NetworkFrameInput(address=address, port=port) as f:
        frame = next(f)
        stop_timestamp = frame.timestamp + int(record_time * 1000000)

        for frame in f:
            if frame.timestamp >= stop_timestamp:
                break

            # undistort frame
            frame_image_raw = frame.image
            frame_image_undistorted = cv2.undistort(
                frame_image_raw, mtx, dist, None, mtx)

            frames_data[f'timestamp_{camera}'].append([frame.timestamp])
            frames_data[f'image_raw_{camera}'].append([frame_image_raw])
            frames_data[f'image_undistorted_{camera}'].append([frame_image_undistorted])

    frames_file.close()
    return


def record_network_dvs():

    n_cameras = 2

    vicon_record_time = 30  # in seconds
    dv_record_time = 60     # in seconds (set much higher)

    test_initials = 'jpt'
    test_scenario = 'floating_kth_hammer'
    test_number = 0

    date_data = time.strftime('%Y%m%d')
    #date_data = 20210929
    path_data = f'./data/{date_data}_{test_initials}_{test_scenario}/{test_number:04}'

    date_camera = time.strftime('%Y%m%d')
    #date_camera = 20210922
    path_camera = f'./camera_calibration/{date_camera}'

    date_projection = time.strftime('%Y%m%d')
    #date_projection = 20210922
    path_projection = f'./projection_calibration/{date_projection}'

    path_props = './props'

    # servers
    vicon_address, vicon_port = '127.0.0.1', 801
    dv_address = '127.0.0.1'
    dv_event_port = [36000, 36001]
    dv_frame_port = [36002, 36003]

    # comment out as required
    prop_names = [
        'kth_hammer',
        #'kth_screwdriver',
        #'kth_spanner',
        #'jpt_mallet',
        #'jpt_screwdriver',
    ]

    # props_markers:      contains the translation of each marker, relative to prop origin
    # props_mesh:         contains prop STL meshes (polygon, translation, vertex)
    props_markers = {}
    props_mesh = {}

    for prop_name in prop_names:
        with open(f'{path_props}/{prop_name}_markers.json', 'r') as marker_file:
            props_markers[prop_name] = json.load(marker_file)
        mesh = stl.mesh.Mesh.from_file(f'{path_props}/{prop_name}_mesh.stl').vectors.transpose(0, 2, 1)
        props_mesh[prop_name] = mesh


    ##################################################################

    # === CAMERA CALIBRATION FILES ===

    dv_cam_mtx_file_name = [f'{path_camera}/camera_{i}_matrix.npy' for i in range(n_cameras)]
    dv_cam_mtx = [np.load(file_name) for file_name in dv_cam_mtx_file_name]

    dv_cam_dist_file_name = [f'{path_camera}/camera_{i}_distortion_coefficients.npy' for i in range(n_cameras)]
    dv_cam_dist = [np.load(file_name) for file_name in dv_cam_dist_file_name]


    ##################################################################

    # === DATA FILES ===

    os.makedirs(path_data, exist_ok=True)

    events_file_name = [f'{path_data}/raw_event_{i}.h5' for i in range(n_cameras)]
    frames_file_name = [f'{path_data}/raw_frame_{i}.h5' for i in range(n_cameras)]
    poses_file_name = f'{path_data}/raw_pose.h5'

    ##################################################################


    for f in os.listdir(path_data):
        os.remove(f'{path_data}/{f}')

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
    print('give the wand signal')

    proc = []
    for i in range(n_cameras):
        proc.append(Process(target=get_dvs_network_events, args=(
            i, dv_record_time, dv_address, dv_event_port[i], dv_cam_mtx[i], dv_cam_dist[i], events_file_name[i])))
        proc.append(Process(target=get_dvs_network_frames, args=(
            i, dv_record_time, dv_address, dv_frame_port[i], dv_cam_mtx[i], dv_cam_dist[i], frames_file_name[i])))

    proc.append(Process(target=get_vicon_network_poses, args=(
        vicon_record_time, vicon_address, vicon_port, props_markers, poses_file_name)))

    # start processes
    for p in proc:
        p.start()

    # wait for processes
    for p in proc:
        p.join()

    print('=== end recording ===')
