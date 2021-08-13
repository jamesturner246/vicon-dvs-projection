
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


def create_processed_event_file(f_name='./data/processed_event.h5'):
    f = tables.open_file(f_name, mode='w')
    f.create_earray(f.root, 'timestamp', tables.atom.UInt64Atom(), (0,))
    f.create_earray(f.root, 'polarity', tables.atom.BoolAtom(), (0,))
    f.create_earray(f.root, 'x', tables.atom.UInt16Atom(), (0,))
    f.create_earray(f.root, 'y', tables.atom.UInt16Atom(), (0,))
    f.create_earray(f.root, 'label', tables.atom.Int8Atom(), (0,))
    f.close()


def create_processed_frame_file(f_name='./data/processed_frame.h5'):
    f = tables.open_file(f_name, mode='w')
    f.create_earray(f.root, 'timestamp', tables.atom.UInt64Atom(), (0,))
    f.create_earray(f.root, 'timestamp_a', tables.atom.UInt64Atom(), (0,))
    f.create_earray(f.root, 'timestamp_b', tables.atom.UInt64Atom(), (0,))
    f.create_earray(f.root, 'image', tables.atom.UInt8Atom(), (0, 260, 346, 3))
    f.create_earray(f.root, 'label', tables.atom.UInt8Atom(), (0, 260, 346))
    f.close()


def create_processed_vicon_file(props, f_name='./data/processed_vicon.h5'):
    f = tables.open_file(f_name, mode='w')
    f.create_earray(f.root, 'timestamp', tables.atom.UInt64Atom(), (0,))
    g_props = f.create_group(f.root, 'props')
    for prop_name in props.keys():
        g_prop = f.create_group(g_props, prop_name)
        f.create_earray(g_prop, 'extrapolated', tables.atom.BoolAtom(), (0,))
        f.create_earray(g_prop, 'quality', tables.atom.Float64Atom(), (0,))
        f.create_earray(g_prop, 'rotation', tables.atom.Float64Atom(), (0, 3))
        g_translation = f.create_group(g_prop, 'translation')
        for marker_name in props[prop_name].keys():
            f.create_earray(g_translation, marker_name, tables.atom.Float64Atom(), (0, 3))
    f.close()


def get_events(address, port, record_time, camera_matrix, distortion_coefficients,
               f_name='./data/event.h5'):
    f = tables.open_file(f_name, mode='w')
    f_timestamp = f.create_earray(f.root, 'timestamp', tables.atom.UInt64Atom(), (0,))
    f_polarity = f.create_earray(f.root, 'polarity', tables.atom.BoolAtom(), (0,))
    f_x = f.create_earray(f.root, 'x', tables.atom.UInt16Atom(), (0,))
    f_y = f.create_earray(f.root, 'y', tables.atom.UInt16Atom(), (0,))

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
            f_x.append([event_undistorted[0]])
            f_y.append([event_undistorted[1]])

    f.close()
    return


def get_frames(address, port, record_time, camera_matrix, distortion_coefficients,
               f_name='./data/frame.h5'):
    f = tables.open_file(f_name, mode='w')
    f_timestamp_a = f.create_earray(f.root, 'timestamp_a', tables.atom.UInt64Atom(), (0,))
    f_timestamp_b = f.create_earray(f.root, 'timestamp_b', tables.atom.UInt64Atom(), (0,))
    f_image = f.create_earray(f.root, 'image', tables.atom.UInt8Atom(), (0, 260, 346, 3))

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

            f_timestamp_a.append([frame.timestamp_start_of_frame])
            f_timestamp_b.append([frame.timestamp_end_of_frame])
            f_image.append([frame_undistorted])

    f.close()
    return


def get_vicon_pyvicon(address, port, record_time, props,
                      f_name='./data/vicon.h5'):

    import pyvicon as pv

    sanity_check = False

    f = tables.open_file(f_name, mode='w')
    f_timestamp = f.create_earray(f.root, 'timestamp', tables.atom.UInt64Atom(), (0,))
    g_props = f.create_group(f.root, 'props')
    for prop_name in props.keys():
        g_prop = f.create_group(g_props, prop_name)
        f.create_earray(g_prop, 'quality', tables.atom.Float64Atom(), (0,))
        f.create_earray(g_prop, 'rotation', tables.atom.Float64Atom(), (0, 3))
        g_translation = f.create_group(g_prop, 'translation')
        for marker_name in props[prop_name].keys():
            f.create_earray(g_translation, marker_name, tables.atom.Float64Atom(), (0, 3))

    client = pv.PyVicon()
    result = client.connect(f'{address}:{port}')
    result = client.enable_marker_data()
    result = client.enable_segment_data()

    if sanity_check:
        while True:
            result = client.get_frame()
            print('get_frame:', result)

            if result != pv.Result.NoFrame:
                break

        prop_count = client.get_subject_count()
        print('prop count:', prop_count)
        assert(prop_count == len(props))

        for prop_i in range(prop_count):
            prop_name = client.get_subject_name(prop_i)
            print('prop name:', prop_name)
            assert(prop_name in props.keys())

            marker_count = client.get_marker_count(prop_name)
            print(' ', prop_name, 'marker count:', marker_count)
            assert(marker_count == len(props[prop_name]))

            for marker_i in range(marker_count):
                marker_name = client.get_marker_name(prop_name, marker_i)
                print('   ', prop_name, 'marker', marker_i, 'name:', marker_name)
                assert(marker_name in props[prop_name].keys())


    # begin frame collection
    timestamp = int(datetime.now().timestamp() * 1000000)
    stop_time = timestamp + record_time

    while timestamp < stop_time:
        result = client.get_frame()

        if result == pv.Result.NoFrame:
            continue

        timestamp = int(datetime.now().timestamp() * 1000000)
        f_timestamp.append([timestamp])

        prop_count = client.get_subject_count()

        for prop_i in range(prop_count):
            prop_name = client.get_subject_name(prop_i)

            marker_count = client.get_marker_count(prop_name)

            prop_quality = client.get_subject_quality(prop_name)
            g_props[prop_name].quality.append([prop_quality])

            if prop_quality is not None:
                root_segment = client.get_subject_root_segment_name(prop_name)

                rotation = client.get_segment_global_rotation_euler_xyz(prop_name, root_segment)
                g_props[prop_name].rotation.append([rotation])

                for marker_i in range(marker_count):
                    marker_name = client.get_marker_name(prop_name, marker_i)

                    translation = client.get_marker_global_translation(prop_name, marker_name)
                    g_props[prop_name].translation[marker_name].append([translation])

            else:
                rotation = np.full((1, 3), np.nan)
                g_props[prop_name].rotation.append(rotation)

                for marker_i in range(marker_count):
                    marker_name = client.get_marker_name(prop_name, marker_i)

                    translation = np.full((1, 3), np.nan)
                    g_props[prop_name].translation[marker_name].append(translation)

    result = client.disconnect()

    f.close()
    return


def get_vicon(address, port, record_time, props,
              f_name='./data/vicon.h5'):

    from vicon_dssdk import ViconDataStream

    sanity_check = False

    f = tables.open_file(f_name, mode='w')
    f_timestamp = f.create_earray(f.root, 'timestamp', tables.atom.UInt64Atom(), (0,))
    g_props = f.create_group(f.root, 'props')
    for prop_name in props.keys():
        g_prop = f.create_group(g_props, prop_name)
        f.create_earray(g_prop, 'quality', tables.atom.Float64Atom(), (0,))
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
            g_props[prop_name].quality.append([prop_quality])

            if prop_quality is not None:
                root_segment = client.GetSubjectRootSegmentName(prop_name)

                rotation = client.GetSegmentGlobalRotationEulerXYZ(prop_name, root_segment)[0]
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


def projection():

    prop_mesh = stl.mesh.Mesh.from_file('./screwdriver-decimated.stl')

    vicon_address, vicon_port = '127.0.0.1', 801
    dv_address, event_port, frame_port = '127.0.0.1', 36000, 36001

    props = {}

    # # screwdriver mesh marker coordinates
    # props['jt_screwdriver'] = {
    #     'handle_1':    [ 0.0,  78.0,   13.5],
    #     'handle_2':    [ 0.0,  78.0,  -13.5],
    #     'shaft_base':  [ 5.0,  120.0,  0.0 ],
    #     'shaft_tip':   [-5.0,  164.0,  0.0 ],
    # }

    # PROTOTYPE: screwdriver mesh marker coordinates
    props['jt_screwdriver'] = {
        'handle_1':    [ 0.0,  78.0,   13.5],
        'handle_2':    [ 0.0,  78.0,  -13.5],
        'shaft_base':  [ 7.5,  100.0,  0.0 ], # alternate position
        'shaft_tip':   [-5.0,  164.0,  0.0 ],
    }

    # # mallet mesh marker coordinates
    # props['jt_mallet'] = {
    #     'shaft_base':  [ 0.0,  0.0,  0.0 ],
    #     'shaft_tip':   [ 0.0,  0.0,  0.0 ],
    #     'head_1':      [ 0.0,  0.0,  0.0 ],
    #     'head_2':      [ 0.0,  0.0,  0.0 ],
    # }

    # # Vicon wand mesh marker coordinates
    # props['jt_wand'] = {
    #     'top_left':    [ 0.0,  0.0,  0.0 ],
    #     'top_centre':  [ 0.0,  0.0,  0.0 ],
    #     'top_right':   [ 0.0,  0.0,  0.0 ],
    #     'middle':      [ 0.0,  0.0,  0.0 ],
    #     'bottom':      [ 0.0,  0.0,  0.0 ],
    # }

    record = False
    test_name = 'no_human'
    test_number = 7
    record_seconds = 10

    record_time = record_seconds * 1000000

    #vicon_usec_offset = -155000
    vicon_usec_offset = -600000

    event_distinguish_polarity = False

    #vicon_translation_error_threshold = np.infty  # millimeters
    #vicon_rotation_error_threshold = np.infty     # radians

    vicon_translation_error_threshold = 50.0  # millimeters
    vicon_rotation_error_threshold = 0.5      # radians

    vicon_bad_frame_timeout = 100
    vicon_buffer_length = 300


    ##################################################################



    dv_space_coefficients_file_name = './calibration/dv_space_coefficients.npy'
    dv_space_constants_file_name = './calibration/dv_space_constants.npy'

    dv_camera_matrix_file_name = './calibration/camera_matrix.npy'
    dv_distortion_coefficients_file_name = './calibration/camera_distortion_coefficients.npy'

    processed_event_file_name = f'./data/processed_event_{test_name}_{test_number:04}.h5'
    processed_frame_file_name = f'./data/processed_frame_{test_name}_{test_number:04}.h5'
    processed_vicon_file_name = f'./data/processed_vicon_{test_name}_{test_number:04}.h5'
    event_file_name = f'./data/event_{test_name}_{test_number:04}.h5'
    frame_file_name = f'./data/frame_{test_name}_{test_number:04}.h5'
    vicon_file_name = f'./data/vicon_{test_name}_{test_number:04}.h5'

    event_image_video_file_name = f'./data/event_image_video_{test_name}_{test_number:04}.avi'
    event_frame_video_file_name = f'./data/frame_image_video_{test_name}_{test_number:04}.avi'


    dv_space_coefficients = np.load(dv_space_coefficients_file_name)
    dv_space_constants = np.load(dv_space_constants_file_name)

    dv_camera_matrix = np.load(dv_camera_matrix_file_name)
    dv_distortion_coefficients = np.load(dv_distortion_coefficients_file_name)


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



    def get_next_event(de_iter, usec_offset=0):
        de = {}
        de['timestamp'] = np.uint64(next(de_iter['timestamp']) + usec_offset)
        de['polarity'] = next(de_iter['polarity'])
        de['x'] = next(de_iter['x'])
        de['y'] = next(de_iter['y'])

        return de


    def get_next_frame(df_iter, usec_offset=0):
        df = {}
        df['timestamp_a'] = np.uint64(next(df_iter['timestamp_a']) + usec_offset)
        df['timestamp_b'] = np.uint64(next(df_iter['timestamp_b']) + usec_offset)
        df['image'] = next(df_iter['image'])

        return df


    def get_next_vicon(vf_iter, usec_offset=0):
        vf = {}
        vf['timestamp'] = np.uint64(next(vf_iter['timestamp']) + usec_offset)
        vf['quality'] = {}
        for prop_name in vf_iter['quality'].keys():
            vf['quality'][prop_name] = next(vf_iter['quality'][prop_name])
        vf['rotation'] = {}
        for prop_name in vf_iter['rotation'].keys():
            vf['rotation'][prop_name] = next(vf_iter['rotation'][prop_name])
        vf['translation'] = {}
        for prop_name in vf_iter['translation'].keys():
            vf['translation'][prop_name] = {}
            for marker_name in vf_iter['translation'][prop_name].keys():
                vf['translation'][prop_name][marker_name] = next(
                    vf_iter['translation'][prop_name][marker_name])

        return vf


    ##################################################################



    # load original Vicon data
    vicon_file = tables.open_file(vicon_file_name, mode='r')
    vicon_iter = {}
    vicon_iter['timestamp'] = vicon_file.root.timestamp.iterrows()
    vicon_iter['quality'] = {}
    vicon_iter['rotation'] = {}
    vicon_iter['translation'] = {}
    for prop in vicon_file.root.props:
        prop_name = prop._v_name
        vicon_iter['quality'][prop_name] = prop.quality.iterrows()
        vicon_iter['rotation'][prop_name] = prop.rotation.iterrows()
        vicon_iter['translation'][prop_name] = {}
        for marker in prop.translation:
            marker_name = marker.name
            vicon_iter['translation'][prop_name][marker_name] = marker.iterrows()

    # create processed Vicon data file
    create_processed_vicon_file(props, f_name=processed_vicon_file_name)
    processed_vicon_file = tables.open_file(processed_vicon_file_name, mode='a')
    processed_vicon_data = {}
    processed_vicon_data['timestamp'] = processed_vicon_file.root.timestamp
    processed_vicon_data['extrapolated'] = {}
    processed_vicon_data['quality'] = {}
    processed_vicon_data['rotation'] = {}
    processed_vicon_data['translation'] = {}
    for prop in processed_vicon_file.root.props:
        prop_name = prop._v_name
        processed_vicon_data['extrapolated'][prop_name] = prop.extrapolated
        processed_vicon_data['quality'][prop_name] = prop.quality
        processed_vicon_data['rotation'][prop_name] = prop.rotation
        processed_vicon_data['translation'][prop_name] = {}
        for marker in prop.translation:
            marker_name = marker.name
            processed_vicon_data['translation'][prop_name][marker_name] = marker

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
            quality = vicon['quality'][prop_name]
            processed_vicon_data['extrapolated'][prop_name].append([False])
            processed_vicon_data['quality'][prop_name].append([quality])
            rotation = vicon['rotation'][prop_name]
            processed_vicon_data['rotation'][prop_name].append([rotation])
            for marker_name in props[prop_name].keys():
                translation = vicon['translation'][prop_name][marker_name]
                processed_vicon_data['translation'][prop_name][marker_name].append([translation])

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
                    processed_vicon_data['rotation'][prop_name][-1] = f(vicon['timestamp'])
                    for marker_name in props[prop_name].keys():
                        y = np.array(vicon_translation_buffer[prop_name][marker_name])
                        f = interp1d(x, y, axis=0, fill_value='extrapolate', kind='linear')
                        processed_vicon_data['translation'][prop_name][marker_name][-1] = f(vicon['timestamp'])

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
                        processed_vicon_data['rotation'][prop_name][i] = rotation
                        for marker_name in props[prop_name].keys():
                            translation = np.full((1, 3), np.nan)
                            processed_vicon_data['translation'][prop_name][marker_name][i] = translation

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
    event_iter['x'] = event_file.root.x.iterrows()
    event_iter['y'] = event_file.root.y.iterrows()

    # load original DV frame data
    frame_file = tables.open_file(frame_file_name, mode='r')
    frame_iter = {}
    frame_iter['timestamp_a'] = frame_file.root.timestamp_a.iterrows()
    frame_iter['timestamp_b'] = frame_file.root.timestamp_b.iterrows()
    frame_iter['image'] = frame_file.root.image.iterrows()

    # load processed Vicon data file
    processed_vicon_file = tables.open_file(processed_vicon_file_name, mode='r')
    processed_vicon_iter = {}
    processed_vicon_iter['timestamp'] = processed_vicon_file.root.timestamp.iterrows()
    processed_vicon_iter['extrapolated'] = {}
    processed_vicon_iter['quality'] = {}
    processed_vicon_iter['rotation'] = {}
    processed_vicon_iter['translation'] = {}
    for prop in processed_vicon_file.root.props:
        prop_name = prop._v_name
        processed_vicon_iter['extrapolated'][prop_name] = prop.extrapolated.iterrows()
        processed_vicon_iter['quality'][prop_name] = prop.quality.iterrows()
        processed_vicon_iter['rotation'][prop_name] = prop.rotation.iterrows()
        processed_vicon_iter['translation'][prop_name] = {}
        for marker in prop.translation:
            marker_name = marker.name
            processed_vicon_iter['translation'][prop_name][marker_name] = marker.iterrows()

    # create processed DV event data file
    create_processed_event_file(f_name=processed_event_file_name)
    processed_event_file = tables.open_file(processed_event_file_name, mode='a')
    processed_event_data = {}
    processed_event_data['timestamp'] = processed_event_file.root.timestamp
    processed_event_data['polarity'] = processed_event_file.root.polarity
    processed_event_data['x'] = processed_event_file.root.x
    processed_event_data['y'] = processed_event_file.root.y
    processed_event_data['label'] = processed_event_file.root.label

    # create processed DV frame data file
    create_processed_frame_file(f_name=processed_frame_file_name)
    processed_frame_file = tables.open_file(processed_frame_file_name, mode='a')
    processed_frame_data = {}
    processed_frame_data['timestamp'] = processed_frame_file.root.timestamp
    processed_frame_data['timestamp_a'] = processed_frame_file.root.timestamp_a
    processed_frame_data['timestamp_b'] = processed_frame_file.root.timestamp_b
    processed_frame_data['image'] = processed_frame_file.root.image
    processed_frame_data['label'] = processed_frame_file.root.label

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
    frame_timestamp = frame['timestamp_a'] / 2 + frame['timestamp_b'] / 2
    while frame_timestamp < vicon['timestamp']:
        frame = get_next_frame(frame_iter)
        frame_timestamp = frame['timestamp_a'] / 2 + frame['timestamp_b'] / 2


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
            vicon_space_p = np.matmul(prop_mesh.vectors, vicon_space_coefficients) + vicon_space_constants
            dv_space_p = np.matmul(vicon_space_p, dv_space_coefficients) + dv_space_constants
            dv_space_p = dv_space_p[:, :, :2] * (1.0 / dv_space_p[:, :, 2, np.newaxis])
            dv_space_p_int = np.rint(dv_space_p).astype('int32')

            # compute prop mask
            cv2.fillPoly(prop_masks[prop_name], dv_space_p_int, 255)
            prop_mask_dilation_kernel = np.ones((3, 3), 'uint8')
            prop_masks[prop_name] = cv2.dilate(prop_masks[prop_name], prop_mask_dilation_kernel)



        # process DV frames
        if not done_frame:

            if vicon['timestamp'] >= frame_timestamp:

                # mask DV frame image
                frame_image[:, :, :] = frame['image']
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
                processed_frame_data['timestamp'].append([frame_timestamp])
                processed_frame_data['timestamp_a'].append([frame['timestamp_a']])
                processed_frame_data['timestamp_b'].append([frame['timestamp_b']])
                processed_frame_data['image'].append([frame['image']])
                processed_frame_data['label'].append([frame_label])

                # write and show DV frame image
                frame_image_video_file.write(frame_image)
                cv2.imshow('frame image', frame_image)
                k = cv2.waitKey(1)
                if k == ord('q'):
                    cv2.destroyWindow('frame image')
                    done_frame = True

                while frame_timestamp < vicon['timestamp']:
                    try:
                        frame = get_next_frame(frame_iter)
                        frame_timestamp = frame['timestamp_a'] / 2 + frame['timestamp_b'] / 2
                    except StopIteration:
                        done_frame = True
                        break



        # process DV events
        if not done_event:
            event_pos.fill(0)
            event_neg.fill(0)

            while event['timestamp'] < vicon_midway:

                # check DV event is in frame
                bounded_x = 0 <= event['x'] < dv_shape[1]
                bounded_y = 0 <= event['y'] < dv_shape[0]

                if bounded_x and bounded_y:
                    if event['polarity']:
                        event_pos[event['y'], event['x']] += 1
                    else:
                        event_neg[event['y'], event['x']] += 1

                    # get event label
                    label = 0
                    for i in range(len(props)):
                        prop_name = list(props)[i]
                        prop_mask = prop_masks[prop_name].astype('bool')
                        if prop_mask[event['y'], event['x']]:
                            if label != 0:
                                label = -1 # ambiguous label
                                break
                            label = i # prop label

                    # record processed event data
                    processed_event_data['timestamp'].append([event['timestamp']])
                    processed_event_data['polarity'].append([event['polarity']])
                    processed_event_data['x'].append([event['x']])
                    processed_event_data['y'].append([event['y']])
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
