
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

from calibrate_projection import vicon_to_dv_2
from calibrate_projection import vicon_to_camera_centric_2


def get_events(address, port, record_time, camera_matrix, distortion_coefficients,
               f_name='./data/event.h5'):
    f = tables.open_file(f_name, mode='w')
    f_timestamp = f.create_earray(f.root, 'timestamp', tables.atom.UInt64Atom(), (0,))
    f_polarity = f.create_earray(f.root, 'polarity', tables.atom.BoolAtom(), (0,))
    f_xy_distorted = f.create_earray(f.root, 'xy_distorted', tables.atom.UInt16Atom(), (0, 2))
    f_xy_undistorted = f.create_earray(f.root, 'xy_undistorted', tables.atom.UInt16Atom(), (0, 2))

    with dv.NetworkEventInput(address=address, port=port) as event_f:
        event = next(event_f)
        stop_time = event.timestamp + record_time

        for event in event_f:
            if event.timestamp >= stop_time:
                break

            # undistort event
            event_distorted = np.array([event.x, event.y], dtype='float64')
            event_undistorted = cv2.undistortPoints(
                event_distorted, camera_matrix, distortion_coefficients, None, camera_matrix)[0, 0]

            f_timestamp.append([event.timestamp])
            f_polarity.append([event.polarity])
            f_xy_distorted.append([event_distorted])
            f_xy_undistorted.append([event_undistorted])

    f.close()
    return


def get_frames(address, port, record_time, camera_matrix, distortion_coefficients,
               f_name='./data/frame.h5'):
    f = tables.open_file(f_name, mode='w')
    f_timestamp = f.create_earray(f.root, 'timestamp', tables.atom.UInt64Atom(), (0,))
    f_image_distorted = f.create_earray(f.root, 'image_distorted', tables.atom.UInt8Atom(), (0, 260, 346, 3))
    f_image_undistorted = f.create_earray(f.root, 'image_undistorted', tables.atom.UInt8Atom(), (0, 260, 346, 3))

    with dv.NetworkFrameInput(address=address, port=port) as frame_f:
        frame = next(frame_f)
        stop_time = frame.timestamp_end_of_frame + record_time

        for frame in frame_f:
            if frame.timestamp_end_of_frame >= stop_time:
                break

            # undistort frame
            frame_distorted = frame.image[:, :, :]
            frame_undistorted = cv2.undistort(
                frame_distorted, camera_matrix, distortion_coefficients, None, camera_matrix)

            f_timestamp.append([frame.timestamp])
            f_image_distorted.append([frame_distorted])
            f_image_undistorted.append([frame_undistorted])

    f.close()
    return


def get_vicon(address, port, record_time, props,
              f_name='./data/pose.h5'):

    from vicon_dssdk import ViconDataStream

    sanity_check = False

    f = tables.open_file(f_name, mode='w')
    f_timestamp = f.create_earray(f.root, 'timestamp', tables.atom.UInt64Atom(), (0,))
    g_props = f.create_group(f.root, 'props')
    for prop_name in props.keys():
        g_prop = f.create_group(g_props, prop_name)
        f.create_earray(g_prop, 'rotation', tables.atom.Float64Atom(), (0, 3))
        g_translation = f.create_group(g_prop, 'translation')
        for marker_name in props[prop_name].keys():
            f.create_earray(g_translation, marker_name, tables.atom.Float64Atom(), (0, 3))

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
    stop_time = timestamp + record_time

    while timestamp < stop_time:
        if not client.GetFrame():
            continue

        timestamp = int(datetime.now().timestamp() * 1000000)
        f_timestamp.append([timestamp])

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
                g_props[prop_name].rotation.append([rotation])

                for marker_i in range(marker_count):
                    marker_name = marker_names[marker_i][0]

                    translation = client.GetMarkerGlobalTranslation(prop_name, marker_name)[0]
                    g_props[prop_name].translation[marker_name].append([translation])

            else:
                rotation = np.full((1, 3), np.nan)
                g_props[prop_name].rotation.append(rotation)

                for marker_i in range(marker_count):
                    marker_name = marker_names[marker_i][0]

                    translation = np.full((1, 3), np.nan)
                    g_props[prop_name].translation[marker_name].append(translation)

    client.Disconnect()

    f.close()
    return


def get_next_event(event_iter, usec_offset=0):
    event = {}
    event['timestamp'] = np.uint64(next(event_iter['timestamp']) + usec_offset)
    event['polarity'] = next(event_iter['polarity'])
    event['xy_distorted'] = next(event_iter['xy_distorted'])
    event['xy_undistorted'] = next(event_iter['xy_undistorted'])

    return event


def get_next_frame(frame_iter, usec_offset=0):
    frame = {}
    frame['timestamp'] = np.uint64(next(frame_iter['timestamp']) + usec_offset)
    frame['image_distorted'] = next(frame_iter['image_distorted'])
    frame['image_undistorted'] = next(frame_iter['image_undistorted'])

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


def create_processed_event_file(f_name='./data/processed_event.h5'):
    if os.path.exists(f_name):
        os.remove(f_name)

    f = tables.open_file(f_name, mode='a')

    data = {}
    data['timestamp'] = f.create_earray(
        f.root, 'timestamp', tables.atom.UInt64Atom(), (0,))
    data['polarity'] = f.create_earray(
        f.root, 'polarity', tables.atom.BoolAtom(), (0,))
    data['xy_distorted'] = f.create_earray(
        f.root, 'xy_distorted', tables.atom.UInt16Atom(), (0, 2))
    data['xy_undistorted'] = f.create_earray(
        f.root, 'xy_undistorted', tables.atom.UInt16Atom(), (0, 2))
    data['label'] = f.create_earray(
        f.root, 'label', tables.atom.Int8Atom(), (0,))

    return data


def create_processed_frame_file(f_name='./data/processed_frame.h5'):
    if os.path.exists(f_name):
        os.remove(f_name)

    f = tables.open_file(f_name, mode='a')

    data = {}
    data['timestamp'] = f.create_earray(
        f.root, 'timestamp', tables.atom.UInt64Atom(), (0,))
    data['image_distorted'] = f.create_earray(
        f.root, 'image_distorted', tables.atom.UInt8Atom(), (0, 260, 346, 3))
    data['image_undistorted'] = f.create_earray(
        f.root, 'image_undistorted', tables.atom.UInt8Atom(), (0, 260, 346, 3))
    data['label'] = f.create_earray(
        f.root, 'label', tables.atom.UInt8Atom(), (0, 260, 346))

    return data


def create_processed_vicon_file(props, f_name='./data/processed_vicon.h5'):
    if os.path.exists(f_name):
        os.remove(f_name)

    f = tables.open_file(f_name, mode='a')

    data = {}
    data['timestamp'] = f.create_earray(
        f.root, 'timestamp', tables.atom.UInt64Atom(), (0,))
    data['extrapolated'] = {}
    data['rotation'] = {}
    data['camera_rotation'] = {}
    data['translation'] = {}
    data['camera_translation'] = {}
    g_props = f.create_group(f.root, 'props')
    for prop_name in props.keys():
        g_prop = f.create_group(g_props, prop_name)
        data['extrapolated'] = f.create_earray(
            g_prop, 'extrapolated', tables.atom.BoolAtom(), (0,))
        data['rotation'] = f.create_earray(
            g_prop, 'rotation', tables.atom.Float64Atom(), (0, 3))
        data['camera_rotation'] = f.create_earray(
            g_prop, 'camera_rotation', tables.atom.Float64Atom(), (0, 3))
        data['translation'][prop_name] = {}
        g_translation = f.create_group(g_prop, 'translation')
        g_camera_translation = f.create_group(g_prop, 'camera_translation')
        for marker_name in props[prop_name].keys():
            data['translation'][prop_name] = f.create_earray(
                g_translation, marker_name, tables.atom.Float64Atom(), (0, 3))
            data['camera_translation'][prop_name] = f.create_earray(
                g_camera_translation, marker_name, tables.atom.Float64Atom(), (0, 3))

    return data


def projection():

    record = False
    test_scenario = 'no_human'
    test_number = 7

    record_seconds = 10
    record_time = record_seconds * 1000000

    #vicon_usec_offset = -155000
    vicon_usec_offset = -600000

    event_distinguish_polarity = False

    #vicon_translation_error_threshold = np.infty  # millimeters
    vicon_translation_error_threshold = 50.0       # millimeters

    vicon_rotation_error_threshold = np.infty      # degrees
    #vicon_rotation_error_threshold = 30.0         # degrees

    vicon_bad_frame_timeout = 100
    vicon_buffer_length = 300

    vicon_address, vicon_port = '127.0.0.1', 801
    dv_address, event_port, frame_port = '127.0.0.1', 36000, 36001

    path_camera_calib = './camera_calibration'
    path_projection_calib = './projection_calibration'
    path_data = './data'


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

    dv_camera_matrix_file_name = f'{path_camera_calib}/camera_matrix.npy'
    dv_camera_matrix = np.load(dv_camera_matrix_file_name)

    dv_distortion_coefficients_file_name = f'{path_camera_calib}/camera_distortion_coefficients.npy'
    dv_distortion_coefficients = np.load(dv_distortion_coefficients_file_name)

    dv_space_transform_file_name = f'{path_projection_calib}/dv_space_transform.npy'
    dv_space_transform = np.load(dv_space_transform_file_name)


    ##################################################################

    # === DATA FILES ===

    os.makedirs(f'{path_data}/{test_scenario}/{test_number:04}', exist_ok=True)

    event_image_video_file_name = f'{path_data}/{test_scenario}/{test_number:04}/event_image_video.avi'
    event_frame_video_file_name = f'{path_data}/{test_scenario}/{test_number:04}/frame_image_video.avi'

    event_file_name = f'{path_data}/{test_scenario}/{test_number:04}/raw_event.h5'
    frame_file_name = f'{path_data}/{test_scenario}/{test_number:04}/raw_frame.h5'
    vicon_file_name = f'{path_data}/{test_scenario}/{test_number:04}/raw_pose.h5'

    processed_event_file_name = f'{path_data}/{test_scenario}/{test_number:04}/event.h5'
    processed_frame_file_name = f'{path_data}/{test_scenario}/{test_number:04}/frame.h5'
    processed_vicon_file_name = f'{path_data}/{test_scenario}/{test_number:04}/pose.h5'




    # TODO: JSON FILE AND LINK EXISTING CALIBRATION FILES


    # TODO: MULTI-CAMERA




    ##################################################################



    if record:
        print('=== begin recording ===')

        processes = []
        processes.append(Process(target=get_events,
                                 args=(dv_address, event_port, record_time,
                                       dv_camera_matrix, dv_distortion_coefficients),
                                 kwargs={'file_name': event_file_name}))
        processes.append(Process(target=get_frames,
                                 args=(dv_address, frame_port, record_time,
                                       dv_camera_matrix, dv_distortion_coefficients),
                                 kwargs={'file_name': frame_file_name}))
        processes.append(Process(target=get_vicon,
                                 args=(vicon_address, vicon_port, record_time,
                                       props),
                                 kwargs={'file_name': vicon_file_name}))

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        print('=== end recording ===')

        exit(0)


    ##################################################################



    # load original Vicon data
    vicon_file = tables.open_file(vicon_file_name, mode='r')
    vicon_iter = {}
    vicon_iter['timestamp'] = vicon_file.root.timestamp.iterrows()
    vicon_iter['rotation'] = {}
    vicon_iter['translation'] = {}
    for prop in vicon_file.root.props:
        prop_name = prop._v_name
        vicon_iter['rotation'][prop_name] = prop.rotation.iterrows()
        vicon_iter['translation'][prop_name] = {}
        for marker in prop.translation:
            marker_name = marker.name
            vicon_iter['translation'][prop_name][marker_name] = marker.iterrows()

    # create processed Vicon data file
    processed_vicon_data = create_processed_vicon_file(props, f_name=processed_vicon_file_name)

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
    vicon_old = get_next_vicon(vicon_iter, usec_offset=vicon_usec_offset)
    vicon = get_next_vicon(vicon_iter, usec_offset=vicon_usec_offset)

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
            vicon_new = get_next_vicon(vicon_iter, usec_offset=vicon_usec_offset)
        except StopIteration:
            break

        processed_vicon_data['timestamp'].append([vicon['timestamp']])

        # for each prop
        for prop_name in props.keys():
            processed_vicon_data['extrapolated'][prop_name].append([False])
            rotation = vicon['rotation'][prop_name]
            processed_vicon_data['rotation'][prop_name].append([rotation])
            camera_rotation = rotation + dv_space_transform[:3]
            processed_vicon_data['camera_rotation'][prop_name].append([camera_rotation])
            for marker_name in props[prop_name].keys():
                translation = vicon['translation'][prop_name][marker_name]
                processed_vicon_data['translation'][prop_name][marker_name].append([translation])
                camera_translation = vicon_to_camera_centric_2(dv_space_transform, translation)
                processed_vicon_data['camera_translation'][prop_name][marker_name].append([camera_translation])

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
                    processed_vicon_data['extrapolated'][prop_name][-1] = True

                    x = np.array(vicon_timestamp_buffer[prop_name])

                    y = np.array(vicon_rotation_buffer[prop_name])
                    f = interp1d(x, y, axis=0, fill_value='extrapolate', kind='linear')
                    rotation = f(vicon['timestamp'])
                    processed_vicon_data['rotation'][prop_name][-1] = rotation
                    camera_rotation = rotation + dv_space_transform[:3]
                    processed_vicon_data['camera_rotation'][prop_name][-1] = camera_rotation
                    for marker_name in props[prop_name].keys():
                        y = np.array(vicon_translation_buffer[prop_name][marker_name])
                        f = interp1d(x, y, axis=0, fill_value='extrapolate', kind='linear')
                        translation = f(vicon['timestamp'])
                        processed_vicon_data['translation'][prop_name][marker_name][-1] = translation
                        camera_translation = vicon_to_camera_centric_2(dv_space_transform, translation)
                        processed_vicon_data['camera_translation'][prop_name][marker_name][-1] = camera_translation

                else: # bad frame timeout
                    print('DEBUG: bad frame timeout')
                    frame_is_good = True

                    # clear frame buffer
                    vicon_timestamp_buffer[prop_name].clear()
                    vicon_rotation_buffer[prop_name].clear()
                    vicon_camera_rotation_buffer[prop_name].clear()
                    for marker_name in props[prop_name].keys():
                        vicon_translation_buffer[prop_name][marker_name].clear()
                        vicon_camera_translation_buffer[prop_name][marker_name].clear()

                    # void bad frame data
                    for i in range(-1, -vicon_bad_frame_timeout, -1):
                        rotation = np.full((1, 3), np.nan)
                        processed_vicon_data['rotation'][prop_name][i] = rotation
                        processed_vicon_data['camera_rotation'][prop_name][i] = rotation
                        for marker_name in props[prop_name].keys():
                            translation = np.full((1, 3), np.nan)
                            processed_vicon_data['translation'][prop_name][marker_name][i] = translation
                            processed_vicon_data['camera_translation'][prop_name][marker_name][i] = translation

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


    vicon_file.close()
    processed_vicon_file.close()


    ##################################################################



    # constants
    dv_shape = (260, 346, 3)
    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    grey = (50, 50, 50)

    # initialise temp memory
    event_pos = np.zeros(dv_shape[:2], dtype='uint64')
    event_neg = np.zeros(dv_shape[:2], dtype='uint64')
    event_image = np.zeros(dv_shape, dtype='uint8')
    frame_image = np.zeros(dv_shape, dtype='uint8')
    frame_label = np.zeros(dv_shape[:2], dtype='int8')
    prop_masks = {name: np.empty(dv_shape[:2], dtype='uint8') for name in props.keys()}

    # initialise video recordings
    event_image_video_file = cv2.VideoWriter(
        event_image_video_file_name, cv2.VideoWriter_fourcc(*'MJPG'),
        30, dv_shape[1::-1])
    frame_image_video_file = cv2.VideoWriter(
        event_frame_video_file_name, cv2.VideoWriter_fourcc(*'MJPG'),
        30, dv_shape[1::-1])

    # load original DV event data
    event_file = tables.open_file(event_file_name, mode='r')
    event_iter = {}
    event_iter['timestamp'] = event_file.root.timestamp.iterrows()
    event_iter['polarity'] = event_file.root.polarity.iterrows()
    event_iter['xy_distorted'] = event_file.root.xy_distorted.iterrows()
    event_iter['xy_undistorted'] = event_file.root.xy_undistorted.iterrows()

    # load original DV frame data
    frame_file = tables.open_file(frame_file_name, mode='r')
    frame_iter = {}
    frame_iter['timestamp'] = frame_file.root.timestamp.iterrows()
    frame_iter['image_distorted'] = frame_file.root.image_distorted.iterrows()
    frame_iter['image_undistorted'] = frame_file.root.image_undistorted.iterrows()

    # load processed Vicon data file
    processed_vicon_file = tables.open_file(processed_vicon_file_name, mode='r')
    processed_vicon_iter = {}
    processed_vicon_iter['timestamp'] = processed_vicon_file.root.timestamp.iterrows()
    processed_vicon_iter['extrapolated'] = {}
    processed_vicon_iter['rotation'] = {}
    processed_vicon_iter['translation'] = {}
    for prop in processed_vicon_file.root.props:
        prop_name = prop._v_name
        processed_vicon_iter['extrapolated'][prop_name] = prop.extrapolated.iterrows()
        processed_vicon_iter['rotation'][prop_name] = prop.rotation.iterrows()
        processed_vicon_iter['translation'][prop_name] = {}
        for marker in prop.translation:
            marker_name = marker.name
            processed_vicon_iter['translation'][prop_name][marker_name] = marker.iterrows()

    # create processed DV event and frame data files
    processed_event_data = create_processed_event_file(f_name=processed_event_file_name)
    processed_frame_data = create_processed_frame_file(f_name=processed_frame_file_name)

    # get Vicon frames
    vicon_old = get_next_vicon(processed_vicon_iter)
    vicon = get_next_vicon(processed_vicon_iter)
    vicon_midway = vicon_old['timestamp'] / 2 + vicon['timestamp'] / 2

    # catch up DV events
    event = get_next_event(event_iter)
    while event['timestamp'] < vicon_midway:
        event = get_next_event(event_iter)

    # catch up DV frames
    frame = get_next_frame(frame_iter)
    while frame['timestamp'] <= vicon['timestamp']:
        frame = get_next_frame(frame_iter)


    # === MAIN LOOP ===
    done_event = False
    done_frame = False
    while not done_event or not done_frame:

        try:
            vicon_new = get_next_vicon(processed_vicon_iter)
            vicon_midway = vicon['timestamp'] / 2 + vicon_new['timestamp'] / 2
        except StopIteration:
            break

        print()
        print('Vicon frame timestamp: ', vicon['timestamp'])

        for prop_name in props.keys():
            print('extrapolated:', next(processed_vicon_iter['extrapolated'][prop_name]))

            # compute new prop mask
            prop_masks[prop_name].fill(0)

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
            dv_space_p = vicon_to_dv_2(dv_space_transform, vicon_space_p.T)
            dv_space_p_int = np.rint(dv_space_p).astype('int32')

            # compute prop mask
            cv2.fillPoly(prop_masks[prop_name], dv_space_p_int, 255)
            prop_mask_dilation_kernel = np.ones((3, 3), 'uint8')
            prop_masks[prop_name] = cv2.dilate(prop_masks[prop_name], prop_mask_dilation_kernel)



        # process DV frames
        if not done_frame:

            if frame['timestamp'] <= vicon['timestamp']:

                # mask DV frame image
                frame_image[:, :, :] = frame['image_undistorted']
                for prop_name in props.keys():
                    prop_mask = prop_masks[prop_name].astype('bool')
                    frame_image[prop_mask, :] = blue

                # get frame label
                frame_label.fill(0)
                for i in range(len(props)):
                    prop_name = list(props)[i]
                    prop_mask = prop_masks[prop_name].astype('bool')
                    frame_label[frame_label != 0 & prop_mask] = -1 # ambiguous label
                    frame_label[frame_label == 0 & prop_mask] = i # prop label

                # record processed frame data
                processed_frame_data['timestamp'].append([frame['timestamp']])
                processed_frame_data['image_distorted'].append([frame['image_distorted']])
                processed_frame_data['image_undistorted'].append([frame['image_undistorted']])
                processed_frame_data['label'].append([frame_label])

                # write and show DV frame image
                frame_image_video_file.write(frame_image)
                cv2.imshow('frame image', frame_image)
                k = cv2.waitKey(1)
                if k == ord('q'):
                    cv2.destroyWindow('frame image')
                    done_frame = True

                while frame['timestamp'] <= vicon['timestamp']:
                    try:
                        frame = get_next_frame(frame_iter)
                    except StopIteration:
                        done_frame = True
                        break



        # process DV events
        if not done_event:
            event_pos.fill(0)
            event_neg.fill(0)

            while event['timestamp'] < vicon_midway:

                # check DV event is in frame
                bounded_x = 0 <= event['xy_undistorted'][0] < dv_shape[1]
                bounded_y = 0 <= event['xy_undistorted'][1] < dv_shape[0]

                if bounded_x and bounded_y:
                    if event['polarity']:
                        event_pos[event['xy_undistorted'][1], event['xy_undistorted'][0]] += 1
                    else:
                        event_neg[event['xy_undistorted'][1], event['xy_undistorted'][0]] += 1

                    # get event label
                    label = 0
                    for i in range(len(props)):
                        prop_name = list(props)[i]
                        prop_mask = prop_masks[prop_name].astype('bool')
                        if prop_mask[event['xy_undistorted'][1], event['xy_undistorted'][0]]:
                            if label != 0:
                                label = -1 # ambiguous label
                                break
                            label = i # prop label

                    # record processed event data
                    processed_event_data['timestamp'].append([event['timestamp']])
                    processed_event_data['polarity'].append([event['polarity']])
                    processed_event_data['xy_distorted'].append([event['xy_distorted']])
                    processed_event_data['xy_undistorted'].append([event['xy_undistorted']])
                    processed_event_data['label'].append([label])

                try:
                    event = get_next_event(event_iter)
                except StopIteration:
                    done_event = True
                    break


            # fill DV event image with events, then mask it
            event_image.fill(0)
            for prop_name in props.keys():
                prop_mask = prop_masks[prop_name].astype('bool')
                event_image[prop_mask] = grey # show prop mask?
                if event_distinguish_polarity:
                    event_mask_neg = event_neg > event_pos
                    event_image[(event_mask_neg & ~prop_mask)] = green
                    event_image[(event_mask_neg & prop_mask)] = blue
                    event_mask_pos = event_pos > event_neg
                    event_image[(event_mask_pos & ~prop_mask)] = red
                    event_image[(event_mask_pos & prop_mask)] = yellow
                else:
                    event_mask = event_neg.astype('bool') | event_pos.astype('bool')
                    event_image[(event_mask & ~prop_mask)] = green
                    event_image[(event_mask & prop_mask)] = red

            # write and show DV event image
            event_image_video_file.write(event_image)
            cv2.imshow('event image', event_image)
            k = cv2.waitKey(1)
            if k == ord('q'):
                cv2.destroyWindow('event image')
                done_event = True

        vicon_old = vicon
        vicon = vicon_new


    event_image_video_file.release()
    frame_image_video_file.release()

    event_file.close()
    frame_file.close()
    processed_event_file.close()
    processed_frame_file.close()
    processed_vicon_file.close()

    return


if __name__ == '__main__':
    projection()
