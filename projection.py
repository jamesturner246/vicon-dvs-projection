
import os
import json
from collections import deque
from datetime import datetime
import time
import pause
from multiprocessing import Process
from scipy.optimize import minimize
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
import numpy as np
import tables
import stl
import cv2
import dv

from calibrate_projection import euler_angles_to_rotation_matrix
from calibrate_projection import rotation_matrix_to_euler_angles
from calibrate_projection import tait_bryan_angles_to_rotation_matrix

n_camera = 1


def create_event_file(f_name):
    if os.path.exists(f_name):
        os.remove(f_name)

    f = tables.open_file(f_name, mode='a')
    data = {}

    for i in range(n_camera):
        data[f'timestamp_{i}'] = f.create_earray(
            f.root, f'timestamp_{i}', tables.atom.UInt64Atom(), (0,))
        data[f'polarity_{i}'] = f.create_earray(
            f.root, f'polarity_{i}', tables.atom.BoolAtom(), (0,))
        data[f'xy_raw_{i}'] = f.create_earray(
            f.root, f'xy_raw_{i}', tables.atom.UInt16Atom(), (0, 2))
        data[f'xy_undistorted_{i}'] = f.create_earray(
            f.root, f'xy_undistorted_{i}', tables.atom.Float64Atom(), (0, 2))
        data[f'label_{i}'] = f.create_earray(
            f.root, f'label_{i}', tables.atom.Int8Atom(), (0,))

    return f, data


def create_frame_file(f_name):
    if os.path.exists(f_name):
        os.remove(f_name)

    f = tables.open_file(f_name, mode='a')
    data = {}

    for i in range(n_camera):
        data[f'timestamp_{i}'] = f.create_earray(
            f.root, f'timestamp_{i}', tables.atom.UInt64Atom(), (0,))
        data[f'image_raw_{i}'] = f.create_earray(
            f.root, f'image_raw_{i}', tables.atom.UInt8Atom(), (0, 260, 346, 3))
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
    for i in range(n_camera):
        data[f'camera_rotation_{i}'] = {}
        data[f'camera_translation_{i}'] = {}

    g_props = f.create_group(f.root, 'props')
    for prop_name in props.keys():
        g_prop = f.create_group(g_props, prop_name)
        data['extrapolated'][prop_name] = f.create_earray(
            g_prop, 'extrapolated', tables.atom.BoolAtom(), (0,))
        data['rotation'][prop_name] = f.create_earray(
            g_prop, 'rotation', tables.atom.Float64Atom(), (0, 3))
        data['translation'][prop_name] = {}
        g_translation = f.create_group(g_prop, 'translation')
        for marker_name in props[prop_name].keys():
            data['translation'][prop_name][marker_name] = f.create_earray(
                g_translation, marker_name, tables.atom.Float64Atom(), (0, 3))
        for i in range(n_camera):
            data[f'camera_rotation_{i}'][prop_name] = f.create_earray(
                g_prop, f'camera_rotation_{i}', tables.atom.Float64Atom(), (0, 3))
            data[f'camera_translation_{i}'][prop_name] = {}
            g_camera_translation = f.create_group(g_prop, f'camera_translation_{i}')
            for marker_name in props[prop_name].keys():
                data[f'camera_translation_{i}'][prop_name][marker_name] = f.create_earray(
                    g_camera_translation, marker_name, tables.atom.Float64Atom(), (0, 3))

    return f, data


def get_event(camera, record_time, address, port, mtx, dist, f_name):
    f, data = create_event_file(f_name)

    with dv.NetworkEventInput(address=address, port=port) as event_f:
        event = next(event_f)
        stop_timestamp = event.timestamp + int(record_time * 1000000)

        for event in event_f:
            if event.timestamp >= stop_timestamp:
                break

            # undistort event
            event_xy_raw = np.array([event.x, event.y], dtype='float64')
            event_xy_undistorted = cv2.undistortPoints(
                event_xy_raw, mtx, dist, None, mtx)[0, 0]

            data[f'timestamp_{camera}'].append([event.timestamp])
            data[f'polarity_{camera}'].append([event.polarity])
            data[f'xy_raw_{camera}'].append([event_xy_raw])
            data[f'xy_undistorted_{camera}'].append([event_xy_undistorted])

    f.close()
    return


def get_frame(camera, record_time, address, port, mtx, dist, f_name):
    f, data = create_frame_file(f_name)

    with dv.NetworkFrameInput(address=address, port=port) as frame_f:
        frame = next(frame_f)
        stop_timestamp = frame.timestamp + int(record_time * 1000000)

        for frame in frame_f:
            if frame.timestamp >= stop_timestamp:
                break

            # undistort frame
            frame_image_raw = frame.image
            frame_image_undistorted = cv2.undistort(
                frame_image_raw, mtx, dist, None, mtx)

            data[f'timestamp_{camera}'].append([frame.timestamp])
            data[f'image_raw_{camera}'].append([frame_image_raw])
            data[f'image_undistorted_{camera}'].append([frame_image_undistorted])

    f.close()
    return


def get_vicon(record_time, address, port, props, f_name):

    from vicon_dssdk import ViconDataStream

    f, data = create_vicon_file(f_name, props)

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
            prop_quality = client.GetObjectQuality('jt_wand')
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

        prop_names = props.keys()
        for prop_name in prop_names:
            marker_names = props[prop_name].keys()

            try:
                prop_quality = client.GetObjectQuality(prop_name)
            except ViconDataStream.DataStreamException:
                prop_quality = None

            if prop_quality is not None:
                root_segment = client.GetSubjectRootSegmentName(prop_name)

                rotation = client.GetSegmentGlobalRotationEulerXYZ(prop_name, root_segment)[0]
                rotation = tait_bryan_angles_to_rotation_matrix(rotation)
                rotation = rotation_matrix_to_euler_angles(rotation)
                data['rotation'][prop_name].append([rotation])

                for marker_name in marker_names:
                    translation = client.GetMarkerGlobalTranslation(prop_name, marker_name)[0]
                    data['translation'][prop_name][marker_name].append([translation])

            else:
                rotation = np.full(3, np.nan)
                data['rotation'][prop_name].append([rotation])

                for marker_name in marker_names:
                    translation = np.full(3, np.nan)
                    data['translation'][prop_name][marker_name].append([translation])

    client.Disconnect()

    f.close()
    return


def get_next_event(camera, event_iter):
    event = {}
    event[f'timestamp_{camera}'] = next(event_iter[f'timestamp_{camera}'])
    event[f'polarity_{camera}'] = next(event_iter[f'polarity_{camera}'])
    event[f'xy_raw_{camera}'] = next(event_iter[f'xy_raw_{camera}'])
    event[f'xy_undistorted_{camera}'] = next(event_iter[f'xy_undistorted_{camera}'])

    return event


def get_next_frame(camera, frame_iter):
    frame = {}
    frame[f'timestamp_{camera}'] = next(frame_iter[f'timestamp_{camera}'])
    frame[f'image_raw_{camera}'] = next(frame_iter[f'image_raw_{camera}'])
    frame[f'image_undistorted_{camera}'] = next(frame_iter[f'image_undistorted_{camera}'])

    return frame


def get_next_vicon(vicon_iter):
    vicon = {}

    if 'timestamp' in vicon_iter:
        vicon['timestamp'] = next(vicon_iter['timestamp'])

    if 'extrapolated' in vicon_iter:
        vicon['extrapolated'] = {}
        for prop_name in vicon_iter['extrapolated'].keys():
            vicon['extrapolated'][prop_name] = next(vicon_iter['extrapolated'][prop_name])

    if 'rotation' in vicon_iter:
        vicon['rotation'] = {}
        for prop_name in vicon_iter['rotation'].keys():
            vicon['rotation'][prop_name] = next(vicon_iter['rotation'][prop_name])

    for i in range(n_camera):
        if f'camera_rotation_{i}' in vicon_iter:
            vicon[f'camera_rotation_{i}'] = {}
            for prop_name in vicon_iter[f'camera_rotation_{i}'].keys():
                vicon[f'camera_rotation_{i}'][prop_name] = next(vicon_iter[f'camera_rotation_{i}'][prop_name])

    if 'translation' in vicon_iter:
        vicon['translation'] = {}
        for prop_name in vicon_iter['translation'].keys():
            vicon['translation'][prop_name] = {}
            for marker_name in vicon_iter['translation'][prop_name].keys():
                vicon['translation'][prop_name][marker_name] = next(
                    vicon_iter['translation'][prop_name][marker_name])

    for i in range(n_camera):
        if f'camera_translation_{i}' in vicon_iter:
            vicon[f'camera_translation_{i}'] = {}
            for prop_name in vicon_iter[f'camera_translation_{i}'].keys():
                vicon[f'camera_translation_{i}'][prop_name] = {}
                for marker_name in vicon_iter[f'camera_translation_{i}'][prop_name].keys():
                    vicon[f'camera_translation_{i}'][prop_name][marker_name] = next(
                        vicon_iter[f'camera_translation_{i}'][prop_name][marker_name])

    return vicon


def projection():

    record = False
    test_scenario = 'no_human'
    test_number = 0

    #date = time.strftime('%Y%m%d')
    date = 20210914
    initials = 'jt'

    path_camera = './camera_calibration'
    path_projection = './projection_calibration'
    path_data = f'./data/{date}_{initials}_{test_scenario}/{test_number:04}'

    vicon_record_time = 10  # in seconds
    dv_record_time = 60     # in seconds (set much higher)

    event_distinguish_polarity = False

    #vicon_translation_error_threshold = np.inf    # millimeters
    vicon_translation_error_threshold = 30.0       # millimeters

    vicon_rotation_error_threshold = np.inf        # radians
    #vicon_rotation_error_threshold = 1.5          # radians

    vicon_bad_frame_timeout = 100
    vicon_buffer_length = 300

    prop_mask_dilation_kernel = np.ones((3, 3), 'uint8')

    # servers
    vicon_address, vicon_port = '127.0.0.1', 801
    dv_address = '127.0.0.1'
    dv_event_port = [36000, 36001]
    dv_frame_port = [36002, 36003]

    dv_camera_shape = [np.array([260, 346]) for i in range(n_camera)]
    dv_camera_origin_offset = [dv_camera_shape[i][::-1] / 2 for i in range(n_camera)]
    dv_camera_nominal_focal_length = [4.0 for i in range(n_camera)]
    dv_camera_pixel_mm = [1.8e-2 for i in range(n_camera)]

    props = {}
    mesh = {}

    # # dummy wand coordinates (comment out when not debugging)
    # props['jt_wand'] = {
    #     'top_left':    [ 0.0,  0.0,  0.0 ],
    #     'top_centre':  [ 0.0,  0.0,  0.0 ],
    #     'top_right':   [ 0.0,  0.0,  0.0 ],
    #     'middle':      [ 0.0,  0.0,  0.0 ],
    #     'bottom':      [ 0.0,  0.0,  0.0 ],
    # }

    # screwdriver mesh marker coordinates
    props['jt_screwdriver'] = {
        'handle_1':    [ 0.0,  78.0,   13.5 ],
        'handle_2':    [ 0.0,  78.0,  -13.5 ],
        'shaft_base':  [ 5.0,  120.0,  0.0  ],
        'shaft_tip':   [-5.0,  164.0,  0.0  ],
    }
    mesh['jt_screwdriver'] = stl.mesh.Mesh.from_file('./props/screwdriver.stl').vectors

    # # mallet mesh marker coordinates
    # props['jt_mallet'] = {
    #     'shaft_base':  [ 0.0,   9.0,  164.0 ],
    #     'shaft_tip':   [ 0.0,  -9.0,  214.0 ],
    #     'head_1':      [-40.0,  0.0,  276.5 ],
    #     'head_2':      [ 40.0,  0.0,  276.5 ],
    # }
    # mesh['jt_mallet'] = stl.mesh.Mesh.from_file('./props/mallet.stl').vectors



    ##################################################################

    # === CALIBRATION FILES ===

    dv_camera_mtx_file_name = [f'{path_camera}/camera_{i}_matrix.npy' for i in range(n_camera)]
    dv_camera_mtx = [np.load(file_name) for file_name in dv_camera_mtx_file_name]

    dv_camera_dist_file_name = [f'{path_camera}/camera_{i}_distortion_coefficients.npy' for i in range(n_camera)]
    dv_camera_dist = [np.load(file_name) for file_name in dv_camera_dist_file_name]

    dv_space_transform_file_name = [f'{path_projection}/dv_{i}_space_transform.npy' for i in range(n_camera)]
    dv_space_transform = [np.load(file_name) for file_name in dv_space_transform_file_name]
    v_to_dv_rotation = [euler_angles_to_rotation_matrix(dv_space_transform[i][0:3]) for i in range(n_camera)]
    v_to_dv_translation = [dv_space_transform[i][3:6] for i in range(n_camera)]
    dv_camera_focal_length = [dv_camera_nominal_focal_length[i] * dv_space_transform[i][6] for i in range(n_camera)]
    dv_camera_x_scale = [dv_space_transform[i][7] for i in range(n_camera)]


    ##################################################################

    # === DATA FILES ===

    os.makedirs(path_data, exist_ok=True)

    raw_event_file_name = [f'{path_data}/raw_event_{i}.h5' for i in range(n_camera)]
    raw_frame_file_name = [f'{path_data}/raw_frame_{i}.h5' for i in range(n_camera)]
    raw_vicon_file_name = f'{path_data}/raw_pose.h5'

    final_event_file_name = f'{path_data}/event.h5'
    final_frame_file_name = f'{path_data}/frame.h5'
    final_vicon_file_name = f'{path_data}/pose.h5'

    # event_video_file_name = [f'{path_data}/event_{i}_video.avi' for i in range(n_camera)]
    # frame_video_file_name = [f'{path_data}/frame_{i}_video.avi' for i in range(n_camera)]


    ##################################################################



    if record:
        print('=== begin recording ===')

        for f in os.listdir(path_data):
            os.remove(f'{path_data}/{f}')

        info_json = {
            'start_time': datetime.now().timestamp(),
            'camera_calibration_path': path_camera,
            'projection_calibration_path': path_projection}
        with open(f'{path_data}/info.json', 'w') as info_json_file:
            json.dump(info_json, info_json_file)

        proc = []
        for i in range(n_camera):
            proc.append(Process(target=get_event, args=(
                i, dv_record_time, dv_address, dv_event_port[i],
                dv_camera_mtx[i], dv_camera_dist[i], raw_event_file_name[i])))
            proc.append(Process(target=get_frame, args=(
                i, dv_record_time, dv_address, dv_frame_port[i],
                dv_camera_mtx[i], dv_camera_dist[i], raw_frame_file_name[i])))

        proc.append(Process(target=get_vicon, args=(
            vicon_record_time, vicon_address, vicon_port,
            props, raw_vicon_file_name)))

        # start processes
        for p in proc:
            p.start()

        # wait for processes
        for p in proc:
            p.join()

        print('=== end recording ===')

        exit(0)


    ##################################################################



    print('begin preprocessing')

    # load raw Vicon data
    raw_vicon_file = tables.open_file(raw_vicon_file_name, mode='r')
    raw_vicon_iter = {}
    timestamp = raw_vicon_file.root.timestamp
    raw_vicon_iter['timestamp'] = timestamp.iterrows()
    raw_vicon_iter['rotation'] = {}
    raw_vicon_iter['translation'] = {}
    for prop_name in props.keys():
        rotation = raw_vicon_file.root.props[prop_name].rotation
        raw_vicon_iter['rotation'][prop_name] = rotation.iterrows()
        raw_vicon_iter['translation'][prop_name] = {}
        for marker_name in props[prop_name].keys():
            translation = raw_vicon_file.root.props[prop_name].translation[marker_name]
            raw_vicon_iter['translation'][prop_name][marker_name] = translation.iterrows()

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

    vicon = get_next_vicon(raw_vicon_iter)

    # append to good Vicon frame buffers
    for prop_name in props.keys():
        timestamp = vicon['timestamp']
        vicon_timestamp_buffer[prop_name].append(timestamp)
        rotation = vicon['rotation'][prop_name]
        vicon_rotation_buffer[prop_name].append(rotation)
        for marker_name in props[prop_name].keys():
            translation = vicon['translation'][prop_name][marker_name]
            vicon_translation_buffer[prop_name][marker_name].append(translation)

    vicon = get_next_vicon(raw_vicon_iter)


    # get transforms from mesh space to Vicon space zero
    mesh_to_v0_rotation = {}
    mesh_to_v0_translation = {}
    mesh_v0 = {}

    method = 'nelder-mead'
    options = {'disp': True, 'maxiter': 50000, 'maxfev': 100000, 'xatol': 1e-10, 'fatol': 1e-10}

    def err_fun_vicon_to_mesh(params, vicon_ps, mesh_ps, v_to_v0_translations, v_to_v0_rotations):
        error= 0
        for vicon_p, mesh_p, v_to_v0_translation, v_to_v0_rotation in zip(vicon_ps, mesh_ps, v_to_v0_translations, v_to_v0_rotations):
            v0_p = np.dot(vicon_p + v_to_v0_translation, v_to_v0_rotation)

            v0_to_mesh_rotation = euler_angles_to_rotation_matrix(params[0:3])
            v0_to_mesh_translation = params[3:6]

            output = np.dot(v0_p + v0_to_mesh_translation, v0_to_mesh_rotation)
            difference = output - mesh_p
            error += np.sqrt(np.mean(difference ** 2))

        return error

    for prop_name in props.keys():
        mesh_markers = []
        vicon_markers = []
        #params = np.zeros(6, dtype='float64')
        #params = np.random.rand(6) * 2 * np.pi - np.pi
        params = np.array([-0.23134114, -2.83301699,  2.81133181, -1.54930157, -2.35502374,  2.1578688 ])
        print(params)
        vicon_rotation= []
        vicon_translation= []
        v_to_v0_rotation= []
        v_to_v0_translation= []
        for i_vicon in range(0,1000,100):
            i_marker = 0
            mesh_marker= np.empty((len(props[prop_name]), 3), dtype='float64')
            vicon_marker= np.empty(mesh_marker.shape, dtype='float64')
            for marker_name in props[prop_name].keys():
                mesh_marker[i_marker] = props[prop_name][marker_name]
                vicon_marker[i_marker] = raw_vicon_file.root.props[prop_name].translation[marker_name][i_vicon]
                i_marker += 1
            mesh_markers.append(mesh_marker)
            vicon_markers.append(vicon_marker)
            vicon_rotation.append(raw_vicon_file.root.props[prop_name].rotation[i_vicon])
            vicon_translation.append(vicon_marker.mean(0))

            v_to_v0_rotation.append(euler_angles_to_rotation_matrix(vicon_rotation[-1]))
            v_to_v0_translation.append(-vicon_translation[-1])

        for v_to_v0_rot, v_to_v0_trans, vicon_m in zip(v_to_v0_rotation,v_to_v0_translation,vicon_markers):
            print(np.matmul(vicon_m+v_to_v0_trans,v_to_v0_rot))
        xxx= input()
        err = err_fun_vicon_to_mesh(
            params, vicon_markers, mesh_markers, v_to_v0_translation, v_to_v0_rotation)
        print(f'{prop_name} mesh to vicon transform: original guess has error: {err}')

        result = minimize(
            err_fun_vicon_to_mesh, params, method=method, options=options,
            args=(vicon_markers, mesh_markers, v_to_v0_translation, v_to_v0_rotation))
        params = result['x']

        err = err_fun_vicon_to_mesh(
            params, vicon_markers, mesh_markers, v_to_v0_translation, v_to_v0_rotation)
        print(f'{prop_name} mesh to vicon transform: final result has error: {err}')

        mesh_to_v0_rotation[prop_name] = euler_angles_to_rotation_matrix(params[0:3]).T
        mesh_to_v0_translation[prop_name] = -params[3:6]
        mesh_v0[prop_name] = np.matmul(mesh[prop_name], mesh_to_v0_rotation[prop_name])
        mesh_v0[prop_name] += mesh_to_v0_translation[prop_name]
        mesh_p = np.array(list(props[prop_name].values()))
        mesh_p_v0= np.matmul(mesh_p, mesh_to_v0_rotation[prop_name])
        mesh_p_v0 += mesh_to_v0_translation[prop_name]

        #reconstructed= np.matmul(vicon_markers+ v_to_v0_translation, v_to_v0_rotation)
        #reconstructed= np.matmul(reconstructed-mesh_to_v0_translation[prop_name], mesh_to_v0_rotation[prop_name].T)
        #print(params)
        #print(mesh_markers-reconstructed)
        #xxx= input()




    # === PREPROCESS VICON DATA ===
    bad_frame_count = 0
    while True:

        try:
            vicon_new = get_next_vicon(raw_vicon_iter)
        except StopIteration:
            break

        final_vicon_data['timestamp'].append([vicon['timestamp']])

        # for each prop
        for prop_name in props.keys():
            final_vicon_data['extrapolated'][prop_name].append([False])
            rotation = vicon['rotation'][prop_name]
            final_vicon_data['rotation'][prop_name].append([rotation])
            for i in range(n_camera):
                cam_rotation = rotation_matrix_to_euler_angles(
                    np.dot(euler_angles_to_rotation_matrix(rotation),
                           v_to_dv_rotation[i]))
                final_vicon_data[f'camera_rotation_{i}'][prop_name].append([cam_rotation])
            for marker_name in props[prop_name].keys():
                translation = vicon['translation'][prop_name][marker_name]
                final_vicon_data['translation'][prop_name][marker_name].append([translation])
                for i in range(n_camera):
                    cam_translation = np.matmul(translation, v_to_dv_rotation[i]) + v_to_dv_translation[i] * 10
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
                #print('DEBUG: bad frame')
                bad_frame_count += 1

                if bad_frame_count < vicon_bad_frame_timeout and len(vicon_timestamp_buffer[prop_name]) > 1:
                    #print('DEBUG: extrapolating bad frame')
                    final_vicon_data['extrapolated'][prop_name][-1] = True

                    x = np.array(vicon_timestamp_buffer[prop_name])
                    y = np.array(vicon_rotation_buffer[prop_name])
                    f = interp1d(x, y, axis=0, fill_value='extrapolate', kind='linear')
                    rotation = f(vicon['timestamp'])
                    final_vicon_data['rotation'][prop_name][-1] = rotation
                    for i in range(n_camera):
                        cam_rotation = rotation_matrix_to_euler_angles(
                            np.dot(euler_angles_to_rotation_matrix(rotation),
                                   v_to_dv_rotation[i]))
                        final_vicon_data[f'camera_rotation_{i}'][prop_name][-1] = cam_rotation
                    for marker_name in props[prop_name].keys():
                        y = np.array(vicon_translation_buffer[prop_name][marker_name])
                        f = interp1d(x, y, axis=0, fill_value='extrapolate', kind='linear')
                        translation = f(vicon['timestamp'])
                        final_vicon_data['translation'][prop_name][marker_name][-1] = translation
                        for i in range(n_camera):
                            cam_translation = np.matmul(translation, v_to_dv_rotation[i]) + v_to_dv_translation[i] * 10
                            final_vicon_data[f'camera_translation_{i}'][prop_name][marker_name][-1] = cam_translation

                else: # bad frame timeout
                    #print('DEBUG: bad frame timeout')
                    frame_is_good = True

                    # clear frame buffer
                    vicon_timestamp_buffer[prop_name].clear()
                    vicon_rotation_buffer[prop_name].clear()
                    for marker_name in props[prop_name].keys():
                        vicon_translation_buffer[prop_name][marker_name].clear()

                    # void bad frame data
                    bad_data = np.full(3, np.nan)
                    a = max(0, len(final_vicon_data['timestamp']) - vicon_bad_frame_timeout + 1)
                    b = len(final_vicon_data['timestamp'])
                    final_vicon_data['rotation'][prop_name][a:b] = bad_data
                    for i in range(n_camera):
                        final_vicon_data[f'camera_rotation_{i}'][prop_name][a:b] = bad_data
                    for marker_name in props[prop_name].keys():
                        final_vicon_data['translation'][prop_name][marker_name][a:b] = bad_data
                        for i in range(n_camera):
                            final_vicon_data[f'camera_translation_{i}'][prop_name][marker_name][a:b] = bad_data

            # append good Vicon frame to buffer
            if frame_is_good:
                #print('DEBUG: good frame')
                bad_frame_count = 0

                # append to good Vicon frame buffers
                timestamp = vicon['timestamp']
                vicon_timestamp_buffer[prop_name].append(timestamp)
                rotation = vicon['rotation'][prop_name]
                vicon_rotation_buffer[prop_name].append(rotation)
                for marker_name in props[prop_name].keys():
                    translation = vicon['translation'][prop_name][marker_name]
                    vicon_translation_buffer[prop_name][marker_name].append(translation)


        vicon = vicon_new


    raw_vicon_file.close()
    final_vicon_file.close()

    print('finished preprocessing')


    ##################################################################



    # constants
    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    grey = (50, 50, 50)

    # initialise temp memory
    event_pos = [np.zeros(dv_camera_shape[i], dtype='uint64') for i in range(n_camera)]
    event_neg = [np.zeros(dv_camera_shape[i], dtype='uint64') for i in range(n_camera)]
    event_image = [np.zeros(np.hstack((dv_camera_shape[i], [3])), dtype='uint8') for i in range(n_camera)]
    frame_image = [np.zeros(np.hstack((dv_camera_shape[i], [3])), dtype='uint8') for i in range(n_camera)]
    frame_label = [np.zeros(dv_camera_shape[i], dtype='int8') for i in range(n_camera)]
    frame_label_depth = [np.zeros(dv_camera_shape[i], dtype='float64') for i in range(n_camera)]
    prop_masks = [{name: np.empty(dv_camera_shape[i], dtype='uint8') for name in props.keys()}
                  for i in range(n_camera)]

    # load raw DV event data
    raw_event_file = []
    raw_event_iter = []
    for i in range(n_camera):
        e_file = tables.open_file(raw_event_file_name[i], mode='r')
        e_iter = {}
        e_iter[f'timestamp_{i}'] = e_file.root[f'timestamp_{i}'].iterrows()
        e_iter[f'polarity_{i}'] = e_file.root[f'polarity_{i}'].iterrows()
        e_iter[f'xy_raw_{i}'] = e_file.root[f'xy_raw_{i}'].iterrows()
        e_iter[f'xy_undistorted_{i}'] = e_file.root[f'xy_undistorted_{i}'].iterrows()
        raw_event_file.append(e_file)
        raw_event_iter.append(e_iter)

    # load raw DV frame data
    raw_frame_file = []
    raw_frame_iter = []
    for i in range(n_camera):
        f_file = tables.open_file(raw_frame_file_name[i], mode='r')
        f_iter = {}
        f_iter[f'timestamp_{i}'] = f_file.root[f'timestamp_{i}'].iterrows()
        f_iter[f'image_raw_{i}'] = f_file.root[f'image_raw_{i}'].iterrows()
        f_iter[f'image_undistorted_{i}'] = f_file.root[f'image_undistorted_{i}'].iterrows()
        raw_frame_file.append(f_file)
        raw_frame_iter.append(f_iter)

    # load final Vicon data file
    final_vicon_file = tables.open_file(final_vicon_file_name, mode='r')
    final_vicon_iter = {}
    timestamp = final_vicon_file.root.timestamp
    final_vicon_iter['timestamp'] = timestamp.iterrows()
    final_vicon_iter['extrapolated'] = {}
    final_vicon_iter['rotation'] = {}
    final_vicon_iter['translation'] = {}
    for i in range(n_camera):
        final_vicon_iter[f'camera_rotation_{i}'] = {}
        final_vicon_iter[f'camera_translation_{i}'] = {}
    for prop_name in props.keys():
        extrapolated = final_vicon_file.root.props[prop_name].extrapolated
        final_vicon_iter['extrapolated'][prop_name] = extrapolated.iterrows()
        rotation = final_vicon_file.root.props[prop_name].rotation
        final_vicon_iter['rotation'][prop_name] = rotation.iterrows()
        final_vicon_iter['translation'][prop_name] = {}
        for marker_name in props[prop_name].keys():
            translation = final_vicon_file.root.props[prop_name].translation[marker_name]
            final_vicon_iter['translation'][prop_name][marker_name] = translation.iterrows()
        for i in range(n_camera):
            camera_rotation = final_vicon_file.root.props[prop_name][f'camera_rotation_{i}']
            final_vicon_iter[f'camera_rotation_{i}'][prop_name] = camera_rotation.iterrows()
            final_vicon_iter[f'camera_translation_{i}'][prop_name] = {}
            for marker_name in props[prop_name].keys():
                camera_translation = final_vicon_file.root.props[prop_name][f'camera_translation_{i}'][marker_name]
                final_vicon_iter[f'camera_translation_{i}'][prop_name][marker_name] = camera_translation.iterrows()

    # create final DV event and frame data files
    final_event_file, final_event_data = create_event_file(final_event_file_name)
    final_frame_file, final_frame_data = create_frame_file(final_frame_file_name)

    # # initialise video recordings
    # event_video_file = [cv2.VideoWriter(
    #     event_video_file_name[i], cv2.VideoWriter_fourcc(*'MJPG'),
    #     30, dv_camera_shape[i][::-1]) for i in range(n_camera)]
    # frame_video_file = [cv2.VideoWriter(
    #     frame_video_file_name[i], cv2.VideoWriter_fourcc(*'MJPG'),
    #     30, dv_camera_shape[i][::-1]) for i in range(n_camera)]


    # manually find first DV event and frame
    event_start_timestamp = [None for i in range(n_camera)]
    event = [None for i in range(n_camera)]
    frame_start_timestamp = [None for i in range(n_camera)]
    frame = [None for i in range(n_camera)]
    for i in range(n_camera):
        search_image_shape = (dv_camera_shape[i][0], dv_camera_shape[i][1] * 2, 3)
        search_image = np.empty(search_image_shape, dtype='uint8')
        event_search_image = search_image[:, :dv_camera_shape[i][1]]
        frame_search_image = search_image[:, dv_camera_shape[i][1]:]

        batch = 30
        i_event = 0
        n_event = len(raw_event_file[i].root[f'timestamp_{i}'])
        event_timestamp = raw_event_file[i].root[f'timestamp_{i}']
        event_xy_undistorted = raw_event_file[i].root[f'xy_undistorted_{i}']

        i_frame = 0
        n_frame = len(raw_frame_file[i].root[f'timestamp_{i}'])
        frame_timestamp = raw_frame_file[i].root[f'timestamp_{i}']
        frame_image_undistorted = raw_frame_file[i].root[f'image_undistorted_{i}']

        while True:
            # search events
            event_search_image.fill(0)
            for xy in event_xy_undistorted[i_event:(i_event + batch)]:
                xy_int = np.rint(xy).astype('int32')
                xy_bounded = all(xy_int >= 0) and all(xy_int < dv_camera_shape[i][::-1])
                if xy_bounded:
                    event_search_image[xy_int[1], xy_int[0]] = 255

            # search frames
            if frame_timestamp[i_frame] < event_timestamp[i_event]:
                while True:
                    i_frame += 1
                    if i_frame == n_frame:
                        i_frame -= 1
                        break
                    if frame_timestamp[i_frame] > event_timestamp[i_event]:
                        i_frame -= 1
                        break
            elif frame_timestamp[i_frame] > event_timestamp[i_event]:
                while True:
                    i_frame -= 1
                    if i_frame < 0:
                        i_frame = 0
                        break
                    if frame_timestamp[i_frame] <= event_timestamp[i_event]:
                        break
            frame_search_image[:] = frame_image_undistorted[i_frame]

            print(f'dv {i} timestamps (usec):'
                  f'  event: {event_timestamp[i_event]:,}:'
                  f'  frame: {frame_timestamp[i_frame]:,}', end='\r')

            cv2.imshow(f'find first dv {i} event and frame', search_image)
            k = cv2.waitKey(0)
            if k == ord(' '):
                cv2.destroyWindow(f'find first dv {i} event and frame')
                break
            elif k == ord(','):
                i_event = max(i_event - batch, 0)
            elif k == ord('<'):
                i_event = max(i_event - (50 * batch), 0)
            elif k == ord('.'):
                i_event = min(i_event + batch, n_event - 1)
            elif k == ord('>'):
                i_event = min(i_event + (50 * batch), n_event - 1)

        print()

        event_start_timestamp[i] = np.uint64(event_timestamp[i_event] + 3000000) # plus 3 seconds
        frame_start_timestamp[i] = np.uint64(frame_timestamp[i_frame] + 3000000) # plus 3 seconds
        event[i] = get_next_event(i, raw_event_iter[i])
        while event[i][f'timestamp_{i}'] < event_start_timestamp[i]:
            event[i] = get_next_event(i, raw_event_iter[i])
        frame[i] = get_next_frame(i, raw_frame_iter[i])
        while frame[i][f'timestamp_{i}'] < frame_start_timestamp[i]:
            frame[i] = get_next_frame(i, raw_frame_iter[i])


    # get first Vicon pose
    vicon = get_next_vicon(final_vicon_iter)



    # === MAIN LOOP ===
    done_event = [False for i in range(n_camera)]
    done_frame = [False for i in range(n_camera)]
    while not all(done_event) or not all(done_frame):

        try:
            vicon_new = get_next_vicon(final_vicon_iter)
            vicon_midway = vicon['timestamp'] / 2 + vicon_new['timestamp'] / 2
        except StopIteration:
            #print('DEBUG: out of vicon poses')
            break

        print()
        print('Vicon frame timestamp: ', vicon['timestamp'])

        for prop_name in props.keys():
            #print(f'DEBUG: extrapolated {prop_name}:', vicon['extrapolated'][prop_name])







            # TODO: fix below for new transform

            # get mesh and Vicon marker translations for this prop
            mesh_p = np.array(list(props[prop_name].values()))
            vicon_p = np.array(list(vicon['translation'][prop_name].values()))

            if not np.isfinite(vicon_p).all():
                prop_masks[i][prop_name].fill(0)
                continue

            # TODO: remove old transform

            # transform to Vicon space
            regressor = MultiOutputRegressor(estimator=LinearRegression()).fit(mesh_p_v0, vicon_p)
            mesh_to_v_coefficients = np.array([re.coef_ for re in regressor.estimators_]).T
            mesh_to_v_constants = np.array([[re.intercept_ for re in regressor.estimators_]])
            #vicon_space_p = np.matmul(mesh_v0[prop_name], mesh_to_v_coefficients) + mesh_to_v_constants

            # TODO: use new transform

            # transform to Vicon space
            v0_to_v_rotation = euler_angles_to_rotation_matrix(vicon['rotation'][prop_name])
            v0_to_v_translation = np.mean(list(vicon['translation'][prop_name].values()), 0)
            vicon_space_p = np.matmul(mesh_v0[prop_name], v0_to_v_rotation.T) + v0_to_v_translation



            

            # transform to DV camera space
            for i in range(n_camera):
                dv_space_p = np.matmul(vicon_space_p, v_to_dv_rotation[i]) + v_to_dv_translation[i] * 10
                dv_space_p[:, :, :2] *= (1 / dv_space_p[:, :, 2, np.newaxis])
                dv_space_p = dv_space_p[:, :, :2]
                dv_space_p *= dv_camera_focal_length[i]
                dv_space_p /= dv_camera_pixel_mm[i]
                dv_space_p *= dv_camera_x_scale[i]
                dv_space_p += dv_camera_origin_offset[i]
                dv_space_p_int = np.rint(dv_space_p).astype('int32')

                # compute prop mask
                prop_masks[i][prop_name].fill(0)
                cv2.fillPoly(prop_masks[i][prop_name], dv_space_p_int, 255)
                prop_masks[i][prop_name] = cv2.dilate(prop_masks[i][prop_name], prop_mask_dilation_kernel)



        # process DV frames
        for i in range(n_camera):
            if not done_frame[i]:

                timestamp = frame[i][f'timestamp_{i}'] - frame_start_timestamp[i]
                while timestamp <= vicon['timestamp']:
                    image = frame_image[i]
                    label = frame_label[i]
                    label_depth = frame_label_depth[i]

                    # mask DV frame image
                    image[:] = frame[i][f'image_undistorted_{i}']
                    for prop_name in props.keys():
                        mask = prop_masks[i][prop_name].astype('bool')
                        image[mask, :] = blue

                    # get frame label
                    label.fill(0)
                    label_depth.fill(np.inf)
                    for j in range(len(props)):
                        prop_name = list(props)[j]
                        mask = prop_masks[i][prop_name].astype('bool')

                        prop_depth = 0.0
                        for marker_name in props[prop_name].keys():
                            marker_depth = vicon[f'camera_translation_{i}'][prop_name][marker_name][2]
                            prop_depth += marker_depth / len(props[prop_name])

                        label[mask & (prop_depth < label_depth)] = j
                        label_depth[mask & (prop_depth < label_depth)] = prop_depth

                    # record final frame data
                    final_frame_data[f'timestamp_{i}'].append([timestamp])
                    final_frame_data[f'image_raw_{i}'].append([frame[i][f'image_raw_{i}']])
                    final_frame_data[f'image_undistorted_{i}'].append([frame[i][f'image_undistorted_{i}']])
                    final_frame_data[f'label_{i}'].append([label])

                    # # write DV frame video
                    # frame_video_file[i].write(image)

                    # show DV frame image
                    cv2.imshow(f'frame {i} image', image)
                    k = cv2.waitKey(1)
                    if k == ord('q'):
                        cv2.destroyWindow(f'frame {i} image')
                        done_frame[i] = True

                    try:
                        frame[i] = get_next_frame(i, raw_frame_iter[i])
                    except StopIteration:
                        #print(f'DEBUG: out of dv {i} frames')
                        done_frame[i] = True
                        break

                    timestamp = frame[i][f'timestamp_{i}'] - frame_start_timestamp[i]



        # process DV events
        for i in range(n_camera):
            if not done_event[i]:

                image = event_image[i]
                pos = event_pos[i]
                neg = event_neg[i]

                image.fill(0)
                pos.fill(0)
                neg.fill(0)

                timestamp = event[i][f'timestamp_{i}'] - event_start_timestamp[i]
                while timestamp < vicon_midway:

                    # check DV event is in frame
                    xy_int = np.rint(event[i][f'xy_undistorted_{i}']).astype('int32')
                    xy_bounded = all(xy_int >= 0) and all(xy_int < dv_camera_shape[i][::-1])

                    if xy_bounded:
                        if event[i][f'polarity_{i}']:
                            pos[xy_int[1], xy_int[0]] += 1
                        else:
                            neg[xy_int[1], xy_int[0]] += 1

                        if event_distinguish_polarity:
                            if event[i][f'polarity_{i}']:
                                image[xy_int[1], xy_int[0]] = red
                            else:
                                image[xy_int[1], xy_int[0]] = green
                        else:
                            image[xy_int[1], xy_int[0]] = green

                        # get event label
                        label = 0
                        label_depth = np.inf
                        for j in range(len(props)):
                            prop_name = list(props)[j]
                            mask = prop_masks[i][prop_name].astype('bool')

                            prop_depth = 0.0
                            for marker_name in props[prop_name].keys():
                                marker_depth = vicon[f'camera_translation_{i}'][prop_name][marker_name][2]
                                prop_depth += marker_depth / len(props[prop_name])

                            if mask[xy_int[1], xy_int[0]]:
                                if prop_depth < label_depth:
                                    label = j
                                    label_depth = prop_depth

                        # record final event data
                        final_event_data[f'timestamp_{i}'].append([timestamp])
                        final_event_data[f'polarity_{i}'].append([event[i][f'polarity_{i}']])
                        final_event_data[f'xy_raw_{i}'].append([event[i][f'xy_raw_{i}']])
                        final_event_data[f'xy_undistorted_{i}'].append([event[i][f'xy_undistorted_{i}']])
                        final_event_data[f'label_{i}'].append([label])

                    try:
                        event[i] = get_next_event(i, raw_event_iter[i])
                    except StopIteration:
                        #print(f'DEBUG: out of dv {i} events')
                        done_event[i] = True
                        break

                    timestamp = event[i][f'timestamp_{i}'] - event_start_timestamp[i]


                # fill DV event image with events, then mask it
                for prop_name in props.keys():
                    mask = prop_masks[i][prop_name].astype('bool')
                    image[mask] = grey # show prop mask?
                    if event_distinguish_polarity:
                        mask_neg = neg > pos
                        image[(mask_neg & mask)] = blue
                        mask_pos = pos > neg
                        image[(mask_pos & mask)] = yellow
                    else:
                        mask_pos_neg = neg.astype('bool') | pos.astype('bool')
                        image[(mask_pos_neg & mask)] = red

                # # write DV event video
                # event_video_file[i].write(image)

                # show DV event image
                cv2.imshow(f'event {i} image', image)
                k = cv2.waitKey(1)
                if k == ord('q'):
                    cv2.destroyWindow(f'event {i} image')
                    done_event[i] = True


        vicon = vicon_new



    # cleanup
    for i in range(n_camera):
        raw_event_file[i].close()
        raw_frame_file[i].close()
    final_event_file.close()
    final_frame_file.close()
    final_vicon_file.close()

    # for i in range(n_camera):
    #     event_video_file[i].release()
    #     frame_video_file[i].release()

    return


if __name__ == '__main__':
    projection()
