
import os
from collections import deque
from datetime import datetime
from multiprocessing import Process
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
import numpy as np
import tables
import stl
import cv2
import dv

from calibrate_projection import vicon_to_dv_method_2 as vicon_to_dv
from calibrate_projection import vicon_to_camera_centric_method_2 as vicon_to_camera_centric


def create_event_file(f_name):
    if os.path.exists(f_name):
        os.remove(f_name)

    f = tables.open_file(f_name, mode='a')
    data = {}

    for i in range(2):
        data[f'timestamp_{i}'] = f.create_earray(
            f.root, f'timestamp_{i}', tables.atom.UInt64Atom(), (0,))
        data[f'polarity_{i}'] = f.create_earray(
            f.root, f'polarity_{i}', tables.atom.BoolAtom(), (0,))
        data[f'xy_distorted_{i}'] = f.create_earray(
            f.root, f'xy_distorted_{i}', tables.atom.UInt16Atom(), (0, 2))
        data[f'xy_undistorted_{i}'] = f.create_earray(
            f.root, f'xy_undistorted_{i}', tables.atom.UInt16Atom(), (0, 2))
        data[f'label_{i}'] = f.create_earray(
            f.root, f'label_{i}', tables.atom.Int8Atom(), (0,))

    return f, data


def create_frame_file(f_name):
    if os.path.exists(f_name):
        os.remove(f_name)

    f = tables.open_file(f_name, mode='a')
    data = {}

    for i in range(2):
        data[f'timestamp_{i}'] = f.create_earray(
            f.root, f'timestamp_{i}', tables.atom.UInt64Atom(), (0,))
        data[f'image_distorted_{i}'] = f.create_earray(
            f.root, f'image_distorted_{i}', tables.atom.UInt8Atom(), (0, 260, 346, 3))
        data[f'image_undistorted_{i}'] = f.create_earray(
            f.root, f'image_undistorted_{i}', tables.atom.UInt8Atom(), (0, 260, 346, 3))
        data[f'label_{i}'] = f.create_earray(
            f.root, f'label_{i}', tables.atom.UInt8Atom(), (0, 260, 346))

    return f, data


def create_vicon_file(f_name, props):
    if os.path.exists(f_name):
        os.remove(f_name)

    f = tables.open_file(f_name, mode='a')
    data = {}

    data['timestamp'] = f.create_earray(
        f.root, 'timestamp', tables.atom.UInt64Atom(), (0,))
    data['extrapolated'] = {}
    data['rotation'] = {}
    data['translation'] = {}
    for i in range(2):
        data[f'camera_rotation_{i}'] = {}
        data[f'camera_translation_{i}'] = {}

    g_props = f.create_group(f.root, 'props')
    for prop_name in props.keys():
        g_prop = f.create_group(g_props, prop_name)
        data['extrapolated'] = f.create_earray(
            g_prop, 'extrapolated', tables.atom.BoolAtom(), (0,))
        data['rotation'] = f.create_earray(
            g_prop, 'rotation', tables.atom.Float64Atom(), (0, 3))
        data['translation'][prop_name] = {}
        g_translation = f.create_group(g_prop, 'translation')
        for marker_name in props[prop_name].keys():
            data['translation'][prop_name] = f.create_earray(
                g_translation, marker_name, tables.atom.Float64Atom(), (0, 3))
        for i in range(2):
            data[f'camera_rotation_{i}'] = f.create_earray(
                g_prop, f'camera_rotation_{i}', tables.atom.Float64Atom(), (0, 3))
            data[f'camera_translation_{i}'][prop_name] = {}
            g_camera_translation = f.create_group(g_prop, f'camera_translation_{i}')
            for marker_name in props[prop_name].keys():
                data[f'camera_translation_{i}'][prop_name] = f.create_earray(
                    g_camera_translation, marker_name, tables.atom.Float64Atom(), (0, 3))

    return f, data


def get_events(camera, record_seconds, address, port, mtx, dist, f_name):
    f, data = create_event_file(f_name)

    with dv.NetworkEventInput(address=address, port=port) as event_f:
        event = next(event_f)
        record_time = record_seconds * 1000000
        stop_time = event.timestamp + record_time

        for event in event_f:
            if event.timestamp >= stop_time:
                break

            # undistort event
            event_distorted = np.array([event.x, event.y], dtype='float64')
            event_undistorted = cv2.undistortPoints(
                event_distorted, mtx, dist, None, mtx)[0, 0]

            data[f'timestamp_{camera}'].append([event.timestamp])
            data[f'polarity_{camera}'].append([event.polarity])
            data[f'xy_distorted_{camera}'].append([event_distorted])
            data[f'xy_undistorted_{camera}'].append([event_undistorted])

    f.close()
    return


def get_frames(camera, record_seconds, address, port, mtx, dist, f_name):
    f, data = create_frame_file(f_name)

    with dv.NetworkFrameInput(address=address, port=port) as frame_f:
        frame = next(frame_f)
        record_time = record_seconds * 1000000
        stop_time = frame.timestamp_end_of_frame + record_time

        for frame in frame_f:
            if frame.timestamp_end_of_frame >= stop_time:
                break

            # undistort frame
            frame_distorted = frame.image[:, :, :]
            frame_undistorted = cv2.undistort(
                frame_distorted, mtx, dist, None, mtx)

            data[f'timestamp_{camera}'].append([frame.timestamp])
            data[f'image_distorted_{camera}'].append([frame_distorted])
            data[f'image_undistorted_{camera}'].append([frame_undistorted])

    f.close()
    return


def get_vicon(record_seconds, address, port, props, f_name):

    from vicon_dssdk import ViconDataStream

    f, data = create_vicon_file(f_name, props)

    sanity_check = False

    client = ViconDataStream.Client()
    client.Connect(f'{address}:{port}')
    client.EnableMarkerData()
    client.EnableSegmentData()

    if sanity_check:
        while True:
            if client.GetFrame():
                break

        prop_names = client.GetSubjectNames()
        prop_count = len(prop_names)
        print('prop count:', prop_count)
        assert(prop_count == len(props))

        for prop_i in range(prop_count):
            prop_name = prop_names[prop_i]
            print('prop name:', prop_name)
            assert(prop_name in props.keys())

            marker_names = client.GetMarkerNames(prop_name)
            marker_count = len(marker_names)
            print(' ', prop_name, 'marker count:', marker_count)
            assert(marker_count == len(props[prop_name]))

            for marker_i in range(marker_count):
                marker_name = marker_names[marker_i][0]
                print('   ', prop_name, 'marker', marker_i, 'name:', marker_name)
                assert(marker_name in props[prop_name].keys())


    # begin frame collection
    timestamp = int(datetime.now().timestamp() * 1000000)
    record_time = record_seconds * 1000000
    stop_time = timestamp + record_time

    while timestamp < stop_time:
        if not client.GetFrame():
            continue

        timestamp = int(datetime.now().timestamp() * 1000000)
        data['timestamp'].append([timestamp])

        prop_names = client.GetSubjectNames()
        prop_count = len(prop_names)

        for prop_i in range(prop_count):
            prop_name = prop_names[prop_i]

            marker_names = client.GetMarkerNames(prop_name)
            marker_count = len(marker_names)

            try:
                prop_quality = client.GetObjectQuality(prop_name)
            except ViconDataStream.DataStreamException:
                prop_quality = None

            if prop_quality is not None:
                root_segment = client.GetSubjectRootSegmentName(prop_name)

                rotation = client.GetSegmentGlobalRotationEulerXYZ(prop_name, root_segment)[0]
                rotation = (rotation * 180 / np.pi) + 180
                data['rotation'][prop_name].append([rotation])

                for marker_i in range(marker_count):
                    marker_name = marker_names[marker_i][0]

                    translation = client.GetMarkerGlobalTranslation(prop_name, marker_name)[0]
                    data['translation'][prop_name][marker_name].append([translation])

            else:
                rotation = np.full((1, 3), np.nan)
                data['rotation'][prop_name].append(rotation)

                for marker_i in range(marker_count):
                    marker_name = marker_names[marker_i][0]

                    translation = np.full((1, 3), np.nan)
                    data['translation'][prop_name][marker_name].append(translation)

    client.Disconnect()

    f.close()
    return


def get_next_event(event_iter, camera, usec_offset=0):
    event = {}
    event[f'timestamp_{camera}'] = np.uint64(next(event_iter[f'timestamp_{camera}']) + usec_offset)
    event[f'polarity_{camera}'] = next(event_iter[f'polarity_{camera}'])
    event[f'xy_distorted_{camera}'] = next(event_iter[f'xy_distorted_{camera}'])
    event[f'xy_undistorted_{camera}'] = next(event_iter[f'xy_undistorted_{camera}'])

    return event


def get_next_frame(frame_iter, camera, usec_offset=0):
    frame = {}
    frame[f'timestamp_{camera}'] = np.uint64(next(frame_iter[f'timestamp_{camera}']) + usec_offset)
    frame[f'image_distorted_{camera}'] = next(frame_iter[f'image_distorted_{camera}'])
    frame[f'image_undistorted_{camera}'] = next(frame_iter[f'image_undistorted_{camera}'])

    return frame


def get_next_vicon(vicon_iter, usec_offset=0):
    vicon = {}
    vicon['timestamp'] = np.uint64(next(vicon_iter['timestamp']) + usec_offset)
    vicon['rotation'] = {}
    for prop_name in vicon_iter['rotation'].keys():
        vicon['rotation'][prop_name] = next(vicon_iter['rotation'][prop_name])
    vicon['translation'] = {}
    for prop_name in vicon_iter['translation'].keys():
        vicon['translation'][prop_name] = {}
        for marker_name in vicon_iter['translation'][prop_name].keys():
            vicon['translation'][prop_name][marker_name] = next(
                vicon_iter['translation'][prop_name][marker_name])

    return vicon


def projection():

    record = False
    test_scenario = 'no_human'
    test_number = 0

    path_camera = './camera_calibration'
    path_projection = './projection_calibration'
    path_data = f'./data/{test_scenario}/{test_number:04}'

    record_seconds = 10

    #vicon_usec_offset = -155000
    vicon_usec_offset = -600000

    event_distinguish_polarity = False

    #vicon_translation_error_threshold = np.infty  # millimeters
    vicon_translation_error_threshold = 50.0       # millimeters

    vicon_rotation_error_threshold = np.infty      # degrees
    #vicon_rotation_error_threshold = 30.0         # degrees

    vicon_bad_frame_timeout = 100
    vicon_buffer_length = 300

    prop_mask_dilation_kernel = np.ones((3, 3), 'uint8')

    # servers
    vicon_address, vicon_port = '127.0.0.1', 801
    dv_address = '127.0.0.1'
    dv_event_port = [36000, 36001]
    dv_frame_port = [36002, 36003]

    props = {}
    mesh = {}

    # screwdriver mesh marker coordinates
    props['jt_screwdriver'] = {
        'handle_1':    [ 0.0,  78.0,   13.5],
        'handle_2':    [ 0.0,  78.0,  -13.5],
        'shaft_base':  [ 5.0,  120.0,  0.0 ],
        'shaft_tip':   [-5.0,  164.0,  0.0 ],
    }
    mesh['jt_screwdriver'] = stl.mesh.Mesh.from_file('./props/screwdriver.stl')

    # mallet mesh marker coordinates
    props['jt_mallet'] = {
        'shaft_base':  [ 0.0,   9.0,  164.0],
        'shaft_tip':   [ 0.0,  -9.0,  214.0],
        'head_1':      [-40.0,  0.0,  276.5],
        'head_2':      [ 40.0,  0.0,  276.5],
    }
    mesh['jt_mallet'] = stl.mesh.Mesh.from_file('./props/mallet.stl')



    ##################################################################

    # === CALIBRATION FILES ===

    dv_camera_mtx_file_name = [f'{path_camera}/camera_{i}_matrix.npy' for i in range(2)]
    dv_camera_mtx = [np.load(file_name) for file_name in dv_camera_mtx_file_name]

    dv_camera_dist_file_name = [f'{path_camera}/camera_{i}_distortion_coefficients.npy' for i in range(2)]
    dv_camera_dist = [np.load(file_name) for file_name in dv_camera_dist_file_name]

    dv_space_transform_file_name = [f'{path_projection}/dv_{i}_space_transform.npy' for i in range(2)]
    dv_space_transform = [np.load(file_name) for file_name in dv_space_transform_file_name]


    ##################################################################

    # === DATA FILES ===

    os.makedirs(path_data, exist_ok=True)
    for f in os.listdir(path_data):
        os.remove(path_data + f)

    raw_event_file_name = [f'{path_data}/raw_event_{i}.h5' for i in range(2)]
    raw_frame_file_name = [f'{path_data}/raw_frame_{i}.h5' for i in range(2)]
    raw_vicon_file_name = f'{path_data}/raw_pose.h5'

    final_event_file_name = f'{path_data}/event.h5'
    final_frame_file_name = f'{path_data}/frame.h5'
    final_vicon_file_name = f'{path_data}/pose.h5'

    # event_video_file_name = [f'{path_data}/event_{i}_video.avi' for i in range(2)]
    # frame_video_file_name = [f'{path_data}/frame_{i}_video.avi' for i in range(2)]




    # TODO: JSON FILE AND LINK EXISTING CALIBRATION FILES





    ##################################################################



    if record:
        print('=== begin recording ===')

        processes = []
        for i in range(2):
            processes.append(Process(target=get_events,
                                     args=(i, record_seconds, dv_address, dv_event_port[i],
                                           dv_camera_mtx[i], dv_camera_dist[i], raw_event_file_name[i])))
            processes.append(Process(target=get_frames,
                                     args=(i, record_seconds, dv_address, dv_frame_port[i],
                                           dv_camera_mtx[i], dv_camera_dist[i], raw_frame_file_name[i])))
        processes.append(Process(target=get_vicon,
                                 args=(record_seconds, vicon_address, vicon_port,
                                       props, raw_vicon_file_name)))

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        print('=== end recording ===')

        exit(0)


    ##################################################################



    # load raw Vicon data
    raw_vicon_file = tables.open_file(raw_vicon_file_name, mode='r')
    raw_vicon_iter = {}
    raw_vicon_iter['timestamp'] = raw_vicon_file.root.timestamp.iterrows()
    raw_vicon_iter['rotation'] = {}
    raw_vicon_iter['translation'] = {}
    for prop in raw_vicon_file.root.props:
        prop_name = prop._v_name
        raw_vicon_iter['rotation'][prop_name] = prop.rotation.iterrows()
        raw_vicon_iter['translation'][prop_name] = {}
        for marker in prop.translation:
            marker_name = marker.name
            raw_vicon_iter['translation'][prop_name][marker_name] = marker.iterrows()

    # create final Vicon data file
    final_vicon_file, final_vicon_data = create_vicon_file(final_vicon_file_name, props)

    # initialise good Vicon frame buffer
    vicon_timestamp_buffer = {}
    vicon_rotation_buffer = {}
    vicon_translation_buffer = {}
    for prop_name in props.keys():
        vicon_timestamp_buffer[prop_name] = deque(maxlen=vicon_buffer_length)
        vicon_rotation_buffer[prop_name] = deque(maxlen=vicon_buffer_length)
        vicon_translation_buffer[prop_name] = {}
        for marker_name in props[prop_name].keys():
            vicon_translation_buffer[prop_name][marker_name] = deque(maxlen=vicon_buffer_length)

    # get Vicon frames
    vicon_old = get_next_vicon(raw_vicon_iter, usec_offset=vicon_usec_offset)
    vicon = get_next_vicon(raw_vicon_iter, usec_offset=vicon_usec_offset)

    # append to good Vicon frame buffers
    timestamp = vicon_old['timestamp']
    vicon_timestamp_buffer[prop_name].append(timestamp)
    rotation = vicon_old['rotation'][prop_name]
    vicon_rotation_buffer[prop_name].append(rotation)
    for marker_name in props[prop_name].keys():
        translation = vicon_old['translation'][prop_name][marker_name]
        vicon_translation_buffer[prop_name][marker_name].append(translation)


    # === PREPROCESS VICON DATA ===
    bad_frame_count = 0
    while True:

        try:
            vicon_new = get_next_vicon(raw_vicon_iter, usec_offset=vicon_usec_offset)
        except StopIteration:
            break

        final_vicon_data['timestamp'].append([vicon['timestamp']])

        # for each prop
        for prop_name in props.keys():
            final_vicon_data['extrapolated'][prop_name].append([False])
            rotation = vicon['rotation'][prop_name]
            final_vicon_data['rotation'][prop_name].append([rotation])
            for i in range(2):
                cam_rotation = rotation + dv_space_transform[i][:3]
                final_vicon_data[f'camera_rotation_{i}'][prop_name].append([cam_rotation])
            for marker_name in props[prop_name].keys():
                translation = vicon['translation'][prop_name][marker_name]
                final_vicon_data['translation'][prop_name][marker_name].append([translation])
                for i in range(2):
                    cam_translation = vicon_to_camera_centric(dv_space_transform[i], translation)
                    final_vicon_data[f'camera_translation_{i}'][prop_name][marker_name].append([cam_translation])

            # check current Vicon frame
            frame_is_good = True
            rotation = vicon['rotation'][prop_name]
            rotation_old = vicon_rotation_buffer[prop_name][-1]
            if not all(np.isfinite(rotation)) or any(
                    np.abs(rotation - rotation_old) >= vicon_rotation_error_threshold):
                frame_is_good = False
            for marker_name in props[prop_name].keys():
                translation = vicon['translation'][prop_name][marker_name]
                translation_old = vicon_translation_buffer[prop_name][marker_name][-1]
                if not all(np.isfinite(translation)) or any(
                        np.abs(translation - translation_old) >= vicon_translation_error_threshold):
                    frame_is_good = False

            # extrapolate bad Vicon frame
            if not frame_is_good:
                print('DEBUG: bad frame')
                bad_frame_count += 1

                if bad_frame_count < vicon_bad_frame_timeout:
                    print('DEBUG: extrapolating bad frame')
                    final_vicon_data['extrapolated'][prop_name][-1] = True

                    x = np.array(vicon_timestamp_buffer[prop_name])

                    y = np.array(vicon_rotation_buffer[prop_name])
                    f = interp1d(x, y, axis=0, fill_value='extrapolate', kind='linear')
                    rotation = f(vicon['timestamp'])
                    final_vicon_data['rotation'][prop_name][-1] = rotation
                    for i in range(2):
                        cam_rotation = rotation + dv_space_transform[i][:3]
                        final_vicon_data[f'camera_rotation_{i}'][prop_name][-1] = cam_rotation
                    for marker_name in props[prop_name].keys():
                        y = np.array(vicon_translation_buffer[prop_name][marker_name])
                        f = interp1d(x, y, axis=0, fill_value='extrapolate', kind='linear')
                        translation = f(vicon['timestamp'])
                        final_vicon_data['translation'][prop_name][marker_name][-1] = translation
                        for i in range(2):
                            cam_translation = vicon_to_camera_centric(dv_space_transform[i], translation)
                            final_vicon_data[f'camera_translation_{i}'][prop_name][marker_name][-1] = cam_translation

                else: # bad frame timeout
                    print('DEBUG: bad frame timeout')
                    frame_is_good = True

                    # clear frame buffer
                    vicon_timestamp_buffer[prop_name].clear()
                    vicon_rotation_buffer[prop_name].clear()
                    for marker_name in props[prop_name].keys():
                        vicon_translation_buffer[prop_name][marker_name].clear()

                    # void bad frame data
                    for i in range(-1, -vicon_bad_frame_timeout, -1):
                        rotation = np.full((1, 3), np.nan)
                        final_vicon_data['rotation'][prop_name][i] = rotation
                        final_vicon_data['camera_rotation'][prop_name][i] = rotation
                        for marker_name in props[prop_name].keys():
                            translation = np.full((1, 3), np.nan)
                            final_vicon_data['translation'][prop_name][marker_name][i] = translation
                            final_vicon_data['camera_translation'][prop_name][marker_name][i] = translation

            # append good Vicon frame to buffer
            if frame_is_good:
                print('DEBUG: good frame')
                bad_frame_count = 0

                # append to good Vicon frame buffers
                timestamp = vicon['timestamp']
                vicon_timestamp_buffer[prop_name].append(timestamp)
                rotation = vicon['rotation'][prop_name]
                vicon_rotation_buffer[prop_name].append(rotation)
                for marker_name in props[prop_name].keys():
                    translation = vicon['translation'][prop_name][marker_name]
                    vicon_translation_buffer[prop_name][marker_name].append(translation)

        vicon_old = vicon
        vicon = vicon_new


    raw_vicon_file.close()
    final_vicon_file.close()


    ##################################################################



    # constants
    dv_shape = (260, 346, 3)
    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    grey = (50, 50, 50)

    # initialise temp memory
    event_pos = [np.zeros(dv_shape[:2], dtype='uint64') for i in range(2)]
    event_neg = [np.zeros(dv_shape[:2], dtype='uint64') for i in range(2)]
    event_image = [np.zeros(dv_shape, dtype='uint8') for i in range(2)]
    frame_image = [np.zeros(dv_shape, dtype='uint8') for i in range(2)]
    frame_label = [np.zeros(dv_shape[:2], dtype='int8') for i in range(2)]
    prop_masks = [{name: np.empty(dv_shape[:2], dtype='uint8') for name in props.keys()} for i in range(2)]

    # load raw DV event data
    raw_event_file = []
    raw_event_iter = []
    for i in range(2):
        e_file = tables.open_file(raw_event_file_name[i], mode='r')
        e_iter = {}
        e_iter[f'timestamp_{i}'] = e_file.root.timestamp.iterrows()
        e_iter[f'polarity_{i}'] = e_file.root.polarity.iterrows()
        e_iter[f'xy_distorted_{i}'] = e_file.root.xy_distorted.iterrows()
        e_iter[f'xy_undistorted_{i}'] = e_file.root.xy_undistorted.iterrows()
        raw_event_file.append(e_file)
        raw_event_iter.append(e_iter)

    # load raw DV frame data
    raw_frame_file = []
    raw_frame_iter = []
    for i in range(2):
        f_file = tables.open_file(raw_frame_file_name[i], mode='r')
        f_iter = {}
        f_iter[f'timestamp_{i}'] = f_file.root.timestamp.iterrows()
        f_iter[f'image_distorted_{i}'] = f_file.root.image_distorted.iterrows()
        f_iter[f'image_undistorted_{i}'] = f_file.root.image_undistorted.iterrows()
        raw_frame_file.append(f_file)
        raw_frame_iter.append(f_iter)

    # load final Vicon data file
    final_vicon_file = tables.open_file(final_vicon_file_name, mode='r')
    final_vicon_iter = {}
    final_vicon_iter['timestamp'] = final_vicon_file.root.timestamp.iterrows()
    final_vicon_iter['extrapolated'] = {}
    final_vicon_iter['rotation'] = {}
    final_vicon_iter['translation'] = {}
    for prop in final_vicon_file.root.props:
        prop_name = prop._v_name
        final_vicon_iter['extrapolated'][prop_name] = prop.extrapolated.iterrows()
        final_vicon_iter['rotation'][prop_name] = prop.rotation.iterrows()
        final_vicon_iter['translation'][prop_name] = {}
        for marker in prop.translation:
            marker_name = marker.name
            final_vicon_iter['translation'][prop_name][marker_name] = marker.iterrows()
        for i in range(2):
            final_vicon_iter[f'camera_rotation_{i}'][prop_name] = prop[f'camera_rotation_{i}'].iterrows()
            final_vicon_iter[f'camera_translation_{i}'][prop_name] = {}
            for marker in prop[f'camera_translation_{i}']:
                marker_name = marker.name
                final_vicon_iter[f'camera_translation_{i}'][prop_name][marker_name] = marker.iterrows()

    # create final DV event and frame data files
    final_event_file, final_event_data = create_event_file(final_event_file_name)
    final_frame_file, final_frame_data = create_frame_file(final_frame_file_name)

    # # initialise video recordings
    # event_video_file = [cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'MJPG'), 30, dv_shape[1::-1])
    #                     for file_name in event_video_file_name]
    # frame_video_file = [cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'MJPG'), 30, dv_shape[1::-1])
    #                     for file_name in frame_video_file_name]

    # get Vicon frames
    vicon_old = get_next_vicon(final_vicon_iter)
    vicon = get_next_vicon(final_vicon_iter)
    vicon_midway = vicon_old['timestamp'] / 2 + vicon['timestamp'] / 2

    # catch up DV events
    event = [get_next_event(raw_event_iter[i], i) for i in range(2)]
    for i in range(2):
        while event[i][f'timestamp_{i}'] < vicon_midway:
            event[i] = get_next_event(raw_event_iter[i], i)

    # catch up DV frames
    frame = [get_next_frame(raw_frame_iter[i], i) for i in range(2)]
    for i in range(2):
        while frame[i][f'timestamp_{i}'] <= vicon['timestamp']:
            frame[i] = get_next_frame(raw_frame_iter[i], i)


    # === MAIN LOOP ===
    done_event = [False, False]
    done_frame = [False, False]
    while not all(done_event) or not all(done_frame):

        try:
            vicon_new = get_next_vicon(final_vicon_iter)
            vicon_midway = vicon['timestamp'] / 2 + vicon_new['timestamp'] / 2
        except StopIteration:
            break

        print()
        print('Vicon frame timestamp: ', vicon['timestamp'])

        for prop_name in props.keys():
            print('extrapolated:', next(final_vicon_iter['extrapolated'][prop_name]))

            # compute new prop mask
            for i in range(2):
                prop_masks[i][prop_name].fill(0)

            # get mesh and Vicon marker translations for this prop
            x = np.array(list(props[prop_name].values()))
            y = np.array(list(vicon['translation'][prop_name].values()))

            if not np.isfinite(y).all():
                continue

            # estimate Vicon space transformation
            regressor = MultiOutputRegressor(
                estimator=LinearRegression(),
            ).fit(x, y)

            vicon_space_coefficients = np.array([re.coef_ for re in regressor.estimators_]).T
            vicon_space_constants = np.array([[re.intercept_ for re in regressor.estimators_]])

            # transform STL mesh space to Vicon space, then
            # transform from Vicon space to DV camera space
            vicon_space_p = np.matmul(mesh[prop_name].vectors, vicon_space_coefficients) + vicon_space_constants
            dv_space_p = [vicon_to_dv(dv_space_transform[i], vicon_space_p.T) for i in range(2)]
            dv_space_p_int = [np.rint(p).astype('int32') for p in dv_space_p]

            # compute prop mask
            for i in range(2):
                cv2.fillPoly(prop_masks[i][prop_name], dv_space_p_int[i], 255)
                prop_masks[i][prop_name] = cv2.dilate(prop_masks[i][prop_name], prop_mask_dilation_kernel)



        # process DV frames
        for i in range(2):
            if not done_frame[i]:

                while frame[i][f'timestamp_{i}'] <= vicon['timestamp']:
                    image = frame_image[i]
                    label = frame_label[i]

                    # mask DV frame image
                    image[:, :, :] = frame[i][f'image_undistorted_{i}']
                    for prop_name in props.keys():
                        mask = prop_masks[i][prop_name].astype('bool')
                        image[mask, :] = blue

                    # get frame label
                    label.fill(0)
                    for j in range(len(props)):
                        prop_name = list(props)[j]
                        mask = prop_masks[i][prop_name].astype('bool')
                        label[label != 0 & mask] = -1 # ambiguous label
                        label[label == 0 & mask] = j # prop label

                    # record final frame data
                    final_frame_data[f'timestamp_{i}'].append([frame[i][f'timestamp_{i}']])
                    final_frame_data[f'image_distorted_{i}'].append([frame[i][f'image_distorted_{i}']])
                    final_frame_data[f'image_undistorted_{i}'].append([frame[i][f'image_undistorted_{i}']])
                    final_frame_data[f'label_{i}'].append([label])

                    # # write DV frame video
                    # frame_video_file[i].write(image)

                    # show DV frame image
                    cv2.imshow('frame image', image)
                    k = cv2.waitKey(1)
                    if k == ord('q'):
                        cv2.destroyWindow('frame image')
                        done_frame[i] = True

                    try:
                        frame[i] = get_next_frame(raw_frame_iter[i], i)
                    except StopIteration:
                        done_frame[i] = True
                        break



        # process DV events
        for i in range(2):
            if not done_event[i]:

                image = event_image[i]
                pos = event_pos[i]
                neg = event_neg[i]

                pos.fill(0)
                neg.fill(0)

                while event[i]['timestamp'] < vicon_midway:
                
                    # check DV event is in frame
                    bounded_x = 0 <= event[i][f'xy_undistorted_{i}'][0] < dv_shape[1]
                    bounded_y = 0 <= event[i][f'xy_undistorted_{i}'][1] < dv_shape[0]

                    if bounded_x and bounded_y:
                        if event[i][f'polarity_{i}']:
                            pos[event[f'xy_undistorted_{i}'][1], event[f'xy_undistorted_{i}'][0]] += 1
                        else:
                            neg[event[f'xy_undistorted_{i}'][1], event[f'xy_undistorted_{i}'][0]] += 1

                        # get event label
                        label = 0
                        for j in range(len(props)):
                            prop_name = list(props)[j]
                            mask = prop_masks[prop_name].astype('bool')
                            if mask[event[i][f'xy_undistorted_{i}'][1], event[i][f'xy_undistorted_{i}'][0]]:
                                if label != 0:
                                    label = -1 # ambiguous label
                                    break
                                label = j # prop label

                        # record final event data
                        final_event_data[f'timestamp_{i}'].append([event[f'timestamp_{i}']])
                        final_event_data[f'polarity_{i}'].append([event[f'polarity_{i}']])
                        final_event_data[f'xy_distorted_{i}'].append([event[f'xy_distorted_{i}']])
                        final_event_data[f'xy_undistorted_{i}'].append([event[f'xy_undistorted_{i}']])
                        final_event_data[f'label_{i}'].append([label])

                    try:
                        event[i] = get_next_event(raw_event_iter[i], i)
                    except StopIteration:
                        done_event[i] = True
                        break


                # fill DV event image with events, then mask it
                image.fill(0)
                for prop_name in props.keys():
                    mask = prop_masks[prop_name].astype('bool')
                    image[mask] = grey # show prop mask?
                    if event_distinguish_polarity:
                        mask_neg = neg > pos
                        image[(mask_neg & ~mask)] = green
                        image[(mask_neg & mask)] = blue
                        mask_pos = pos > neg
                        image[(mask_pos & ~mask)] = red
                        image[(mask_pos & mask)] = yellow
                    else:
                        mask_pos_neg = neg.astype('bool') | pos.astype('bool')
                        image[(mask_pos_neg & ~mask)] = green
                        image[(mask_pos_neg & mask)] = red

                # # write DV event video
                # event_video_file[i].write(image)

                # show DV event image
                cv2.imshow('event image', image)
                k = cv2.waitKey(1)
                if k == ord('q'):
                    cv2.destroyWindow('event image')
                    done_event[i] = True

            vicon_old = vicon
            vicon = vicon_new



    # cleanup
    for i in range(2):
        raw_event_file[i].close()
        raw_frame_file[i].close()
    final_vicon_file.close()

    # for i in range(2):
    #     event_video_file[i].release()
    #     frame_video_file[i].release()

    return


if __name__ == '__main__':
    projection()
