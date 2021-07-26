
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


def create_labelled_data_file(f_name='./data/labelled.h5'):
    f = tables.open_file(f_name, mode='w')
    f_timestamp = f.create_earray(f.root, 'timestamp', tables.atom.UInt64Atom(), (0,))
    f_polarity = f.create_earray(f.root, 'polarity', tables.atom.BoolAtom(), (0,))
    f_x = f.create_earray(f.root, 'x', tables.atom.UInt16Atom(), (0,))
    f_y = f.create_earray(f.root, 'y', tables.atom.UInt16Atom(), (0,))
    f_label = f.create_earray(f.root, 'label', tables.atom.StringAtom(32), (0,))
    f.close()


def get_dv_events(address, port, record_time, camera_matrix, distortion_coefficients,
                  f_name='./data/dv_event.h5'):
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


def get_dv_frames(address, port, record_time, camera_matrix, distortion_coefficients,
                  f_name='./data/dv_frame.h5'):
    f = tables.open_file(f_name, mode='w')
    f_timestamp = f.create_earray(f.root, 'timestamp', tables.atom.UInt64Atom(), (0,))
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

            f_timestamp.append([frame.timestamp])
            f_timestamp_a.append([frame.timestamp_start_of_frame])
            f_timestamp_b.append([frame.timestamp_end_of_frame])
            f_image.append([frame_undistorted])

    f.close()
    return


def get_vicon_frame_pyvicon(address, port, record_time, props,
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


def get_vicon_frame(address, port, record_time, props,
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
    dv_address, dv_event_port, dv_frame_port = '127.0.0.1', 36000, 36001

    dv_space_coefficients_file = './calibration/dv_space_coefficients.npy'
    dv_space_constants_file = './calibration/dv_space_constants.npy'

    dv_space_coefficients = np.load(dv_space_coefficients_file)
    dv_space_constants = np.load(dv_space_constants_file)

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


    #record = True
    record = False

    test_name = 'no_human'
    test_number = 7

    #record_seconds = 3
    record_seconds = 10
    record_time = record_seconds * 1000000

    #vicon_usec_offset = -155000
    vicon_usec_offset = -600000

    dv_event_distinguish_polarity = False

    #vicon_translation_error_threshold = np.infty # mm
    vicon_translation_error_threshold = 50.0 # mm
    vicon_rotation_error_threshold = np.infty # degrees
    #vicon_rotation_error_threshold = 30.0 # degrees
    vicon_bad_frame_timeout = 50
    #vicon_frame_buffer_length = 1000
    #vicon_frame_buffer_length = 100
    vicon_frame_buffer_length = 10



    dv_camera_matrix = np.load('./calibration/camera_matrix.npy')
    dv_distortion_coefficients = np.load('./calibration/camera_distortion_coefficients.npy')

    labelled_file_name = f'./data/labelled_{test_name}_{test_number:04}.h5'
    dv_event_file_name = f'./data/dv_event_{test_name}_{test_number:04}.h5'
    dv_frame_file_name = f'./data/dv_frame_{test_name}_{test_number:04}.h5'
    vicon_frame_file_name = f'./data/vicon_{test_name}_{test_number:04}.h5'

    event_image_video_file_name = f'./data/event_image_video_{test_name}_{test_number:04}.avi'
    event_frame_video_file_name = f'./data/frame_image_video_{test_name}_{test_number:04}.avi'



    ##################################################################


    if record:
        print('=== begin recording ===')

        processes = []
        processes.append(Process(target=get_dv_events,
                                 args=(dv_address, dv_event_port, record_time,
                                       dv_camera_matrix, dv_distortion_coefficients),
                                 kwargs={'file_name': dv_event_file_name}))
        processes.append(Process(target=get_dv_frames,
                                 args=(dv_address, dv_frame_port, record_time,
                                       dv_camera_matrix, dv_distortion_coefficients),
                                 kwargs={'file_name': dv_frame_file_name}))
        processes.append(Process(target=get_vicon_frame,
                                 args=(vicon_address, vicon_port, record_time,
                                       props),
                                 kwargs={'file_name': vicon_frame_file_name}))

        #processes.append(Process(target=get_vicon_frame_pyvicon,
        #                         args=(vicon_address, vicon_port, record_time,
        #                               props),
        #                         kwargs={'file_name': vicon_frame_file_name}))

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        print('=== end recording ===')

        exit(0)


    ##################################################################



    create_labelled_data_file(f_name=labelled_file_name)

    labelled_file = tables.open_file(labelled_file_name, mode='a')
    labelled_iter = {}
    labelled_iter['timestamp'] = labelled_file.root.timestamp
    labelled_iter['polarity'] = labelled_file.root.polarity
    labelled_iter['x'] = labelled_file.root.x
    labelled_iter['y'] = labelled_file.root.y
    labelled_iter['label'] = labelled_file.root.label

    dv_event_file = tables.open_file(dv_event_file_name, mode='r')
    dv_event_iter = {}
    dv_event_iter['timestamp'] = dv_event_file.root.timestamp.iterrows()
    dv_event_iter['polarity'] = dv_event_file.root.polarity.iterrows()
    dv_event_iter['x'] = dv_event_file.root.x.iterrows()
    dv_event_iter['y'] = dv_event_file.root.y.iterrows()

    dv_frame_file = tables.open_file(dv_frame_file_name, mode='r')
    dv_frame_iter = {}
    dv_frame_iter['timestamp_a'] = dv_frame_file.root.timestamp_a.iterrows()
    dv_frame_iter['timestamp_b'] = dv_frame_file.root.timestamp_b.iterrows()
    dv_frame_iter['image'] = dv_frame_file.root.image.iterrows()

    vicon_frame_file = tables.open_file(vicon_frame_file_name, mode='r')
    vicon_frame_iter = {}
    vicon_frame_iter['timestamp'] = vicon_frame_file.root.timestamp.iterrows()
    vicon_frame_iter['quality'] = {}
    vicon_frame_iter['rotation'] = {}
    vicon_frame_iter['translation'] = {}
    for prop in vicon_frame_file.root.props:
        prop_name = prop._v_name
        vicon_frame_iter['quality'][prop_name] = prop.quality.iterrows()
        vicon_frame_iter['rotation'][prop_name] = prop.rotation.iterrows()
        vicon_frame_iter['translation'][prop_name] = {}
        for marker in prop.translation:
            marker_name = marker.name
            vicon_frame_iter['translation'][prop_name][marker_name] = marker.iterrows()


    def get_next_dv_event(de_iter, usec_offset=0):
        de = {}
        de['timestamp'] = np.uint64(next(de_iter['timestamp']) + usec_offset)
        de['polarity'] = next(de_iter['polarity'])
        de['x'] = next(de_iter['x'])
        de['y'] = next(de_iter['y'])

        return de


    def get_next_dv_frame(df_iter, usec_offset=0):
        df = {}
        df['timestamp_a'] = np.uint64(next(df_iter['timestamp_a']) + usec_offset)
        df['timestamp_b'] = np.uint64(next(df_iter['timestamp_b']) + usec_offset)
        df['image'] = next(df_iter['image'])

        return df


    def get_next_vicon_frame(vf_iter, usec_offset=0):
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


    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    grey = (50, 50, 50)

    dv_shape = (260, 346, 3)

    event_image_video_file = cv2.VideoWriter(
        event_image_video_file_name, cv2.VideoWriter_fourcc(*'MJPG'),
        30, dv_shape[1::-1])
    frame_image_video_file = cv2.VideoWriter(
        event_frame_video_file_name, cv2.VideoWriter_fourcc(*'MJPG'),
        30, dv_shape[1::-1])

    event_pos = np.zeros(dv_shape[:2], dtype='uint64')
    event_neg = np.zeros(dv_shape[:2], dtype='uint64')
    event_image = np.zeros(dv_shape, dtype='uint8')
    frame_image = np.zeros(dv_shape, dtype='uint8')


    # initialise good Vicon frame buffer
    vicon_timestamp_buffer = deque(maxlen=vicon_frame_buffer_length)
    vicon_rotation_buffer = {}
    vicon_translation_buffer = {}
    for prop_name in props.keys():
        vicon_rotation_buffer[prop_name] = deque(maxlen=vicon_frame_buffer_length)
        vicon_translation_buffer[prop_name] = {}
        for marker_name in props[prop_name].keys():
            vicon_translation_buffer[prop_name][marker_name] = deque(maxlen=vicon_frame_buffer_length)

    # initialise prop masks
    prop_masks = {name: np.empty(dv_shape[:2], dtype='uint8') for name in props.keys()}

    # get Vicon frames
    vicon_frame_old = get_next_vicon_frame(vicon_frame_iter, usec_offset=vicon_usec_offset)
    vicon_frame = get_next_vicon_frame(vicon_frame_iter, usec_offset=vicon_usec_offset)
    vicon_frame_midway = vicon_frame_old['timestamp'] / 2 + vicon_frame['timestamp'] / 2

    # catch up DV events
    dv_event = get_next_dv_event(dv_event_iter)
    while dv_event['timestamp'] < vicon_frame_midway:
        dv_event = get_next_dv_event(dv_event_iter)

    # catch up DV frames
    dv_frame = get_next_dv_frame(dv_frame_iter)
    dv_frame_timestamp = dv_frame['timestamp_a'] / 2 + dv_frame['timestamp_b'] / 2
    while dv_frame_timestamp < vicon_frame['timestamp']:
        dv_frame = get_next_dv_frame(dv_frame_iter)
        dv_frame_timestamp = dv_frame['timestamp_a'] / 2 + dv_frame['timestamp_b'] / 2



    # === MAIN LOOP ===
    done_dv_event = False
    done_dv_frame = False
    bad_frame_count = 0

    while not done_dv_event or not done_dv_frame:
        print('Vicon frame timestamp: ', vicon_frame['timestamp'])
        print()

        try:
            vicon_frame_new = get_next_vicon_frame(vicon_frame_iter, usec_offset=vicon_usec_offset)
            vicon_frame_midway = vicon_frame['timestamp'] / 2 + vicon_frame_new['timestamp'] / 2
        except StopIteration:
            break

        # check current Vicon frame
        frame_is_good = True
        for prop_name in props.keys():
            rotation = vicon_frame['rotation'][prop_name]
            rotation_old = vicon_frame_old['rotation'][prop_name]
            if not all(np.isfinite(rotation)) or any(
                    np.abs(rotation - rotation_old) >= vicon_rotation_error_threshold):
                frame_is_good = False
            for marker_name in props[prop_name].keys():
                translation = vicon_frame['translation'][prop_name][marker_name]
                translation_old = vicon_frame_old['translation'][prop_name][marker_name]
                if not all(np.isfinite(translation)) or any(
                        np.abs(translation - translation_old) >= vicon_translation_error_threshold):
                    frame_is_good = False


        # extrapolate bad Vicon frame
        if not frame_is_good:
            print('DEBUG: bad frame')
            bad_frame_count += 1

            if bad_frame_count < vicon_bad_frame_timeout:
                #print('DEBUG: interpolating bad frame')

                x = np.array(vicon_timestamp_buffer)
                for prop_name in props.keys():
                    y = np.array(vicon_rotation_buffer[prop_name])
                    f = interp1d(x, y, axis=0, fill_value='extrapolate', kind='linear')
                    vicon_frame['rotation'][prop_name][:] = f(vicon_frame['timestamp'])
                    for marker_name in props[prop_name].keys():
                        y = np.array(vicon_translation_buffer[prop_name][marker_name])
                        f = interp1d(x, y, axis=0, fill_value='extrapolate', kind='linear')
                        vicon_frame['translation'][prop_name][marker_name][:] = f(vicon_frame['timestamp'])

            else: # bad frame timeout
                print('DEBUG: bad frame timeout')
                frame_is_good = True


        # append good Vicon frame to buffer
        if frame_is_good:
            #print('DEBUG: good frame')
            bad_frame_count = 0

            # append to good Vicon frame buffers
            timestamp = vicon_frame['timestamp']
            vicon_timestamp_buffer.append(timestamp)
            for prop_name in props.keys():
                rotation = vicon_frame['rotation'][prop_name]
                vicon_rotation_buffer[prop_name].append(rotation)
                for marker_name in props[prop_name].keys():
                    translation = vicon_frame['translation'][prop_name][marker_name]
                    vicon_translation_buffer[prop_name][marker_name].append(translation)


        # compute new prop masks
        for prop_name in props.keys():

            # clear prop mask
            prop_masks[prop_name].fill(0)

            # get mesh and Vicon marker translations for this prop
            x = np.array(list(props[prop_name].values()))
            y = np.array(list(vicon_frame['translation'][prop_name].values()))

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
        if not done_dv_frame:

            if vicon_frame['timestamp'] >= dv_frame_timestamp:

                # mask DV frame image
                frame_image[:, :, :] = dv_frame['image']
                for prop_name in props.keys():
                    prop_mask = prop_masks[prop_name].astype('bool')
                    frame_image[prop_mask, :] = blue

                # write and show DV frame image
                frame_image_video_file.write(frame_image)
                cv2.imshow('frame image', frame_image)
                k = cv2.waitKey(1)
                if k == ord('q'):
                    cv2.destroyWindow('frame image')
                    done_dv_frame = True

                while dv_frame_timestamp < vicon_frame['timestamp']:
                    try:
                        dv_frame = get_next_dv_frame(dv_frame_iter)
                        dv_frame_timestamp = dv_frame['timestamp_a'] / 2 + dv_frame['timestamp_b'] / 2
                    except StopIteration:
                        done_dv_frame = True
                        break



        # process DV events
        if not done_dv_event:
            event_pos.fill(0)
            event_neg.fill(0)

            while dv_event['timestamp'] < vicon_frame_midway:

                # check DV event is in frame
                bounded_x = 0 <= dv_event['x'] < dv_shape[1]
                bounded_y = 0 <= dv_event['y'] < dv_shape[0]

                if bounded_x and bounded_y:
                    if dv_event['polarity']:
                        event_pos[dv_event['y'], dv_event['x']] += 1
                    else:
                        event_neg[dv_event['y'], dv_event['x']] += 1

                    label = ''
                    for prop_name in props.keys():
                        prop_mask = prop_masks[prop_name].astype('bool')
                        if prop_mask[dv_event['y'], dv_event['x']]:
                            if label:
                                label = 'UNKNOWN'
                                break
                            label = prop_name

                    labelled_iter['timestamp'].append([dv_event['timestamp']])
                    labelled_iter['polarity'].append([dv_event['polarity']])
                    labelled_iter['x'].append([dv_event['x']])
                    labelled_iter['y'].append([dv_event['y']])
                    labelled_iter['label'].append([label])

                try:
                    dv_event = get_next_dv_event(dv_event_iter)
                except StopIteration:
                    done_dv_event = True
                    break


            # fill DV event image with events, then mask it
            event_image.fill(0)
            for prop_name in props.keys():
                prop_mask = prop_masks[prop_name].astype('bool')
                event_image[prop_mask] = grey # show prop mask?
                if dv_event_distinguish_polarity:
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
                done_dv_event = True



        vicon_frame_old = vicon_frame
        vicon_frame = vicon_frame_new



    event_image_video_file.release()
    frame_image_video_file.release()

    labelled_file.close()
    dv_event_file.close()
    dv_frame_file.close()
    vicon_frame_file.close()
    return


if __name__ == '__main__':
    projection()
