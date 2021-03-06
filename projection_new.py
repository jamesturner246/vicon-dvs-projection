
import os
import json
from collections import deque
import time
import pause
from argparse import ArgumentParser
from multiprocessing import Process
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import numpy as np
import tables
import stl
import cv2
import dv

from calibrate_projection import euler_angles_to_rotation_matrix
from seek_dvs_start import seek_dvs_start
from events import *
from frames import *
from poses import *

# OpenCV colours
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
yellow = (0, 255, 255)
grey = (50, 50, 50)

n_cameras = 2


def get_next_event(camera, events_iter):
    event = {}
    event[f'timestamp_{camera}'] = next(events_iter[f'timestamp_{camera}'])
    event[f'polarity_{camera}'] = next(events_iter[f'polarity_{camera}'])
    event[f'xy_undistorted_{camera}'] = next(events_iter[f'xy_undistorted_{camera}'])

    return event


def get_next_frame(camera, frames_iter):
    frame = {}
    frame[f'timestamp_{camera}'] = next(frames_iter[f'timestamp_{camera}'])
    frame[f'image_undistorted_{camera}'] = next(frames_iter[f'image_undistorted_{camera}'])

    return frame


def get_next_pose(poses_iter):
    pose = {}

    if 'timestamp' in poses_iter:
        pose['timestamp'] = next(
            poses_iter['timestamp'])

    if 'vicon_rotation' in poses_iter:
        pose['vicon_rotation'] = {}
        for prop_name in poses_iter['vicon_rotation'].keys():
            pose['vicon_rotation'][prop_name] = next(
                poses_iter['vicon_rotation'][prop_name])

    if 'rotation' in poses_iter:
        pose['rotation'] = {}
        for prop_name in poses_iter['rotation'].keys():
            pose['rotation'][prop_name] = next(
                poses_iter['rotation'][prop_name])

    for i in range(n_cameras):
        if f'camera_{i}_rotation' in poses_iter:
            pose[f'camera_{i}_rotation'] = {}
            for prop_name in poses_iter[f'camera_{i}_rotation'].keys():
                pose[f'camera_{i}_rotation'][prop_name] = next(
                    poses_iter[f'camera_{i}_rotation'][prop_name])

    if 'vicon_translation' in poses_iter:
        pose['vicon_translation'] = {}
        for prop_name in poses_iter['vicon_translation'].keys():
            pose['vicon_translation'][prop_name] = next(
                poses_iter['vicon_translation'][prop_name])

    if 'translation' in poses_iter:
        pose['translation'] = {}
        for prop_name in poses_iter['translation'].keys():
            pose['translation'][prop_name] = next(
                poses_iter['translation'][prop_name])

    for i in range(n_cameras):
        if f'camera_{i}_translation' in poses_iter:
            pose[f'camera_{i}_translation'] = {}
            for prop_name in poses_iter[f'camera_{i}_translation'].keys():
                pose[f'camera_{i}_translation'][prop_name] = next(
                    poses_iter[f'camera_{i}_translation'][prop_name])

    if 'vicon_markers' in poses_iter:
        pose['vicon_markers'] = {}
        for prop_name in poses_iter['vicon_markers'].keys():
            pose['vicon_markers'][prop_name] = {}
            for marker_name in poses_iter['vicon_markers'][prop_name].keys():
                pose['vicon_markers'][prop_name][marker_name] = next(
                    poses_iter['vicon_markers'][prop_name][marker_name])

    return pose


def create_pytables_poses_events_file(
        data_file_name, n_cameras, props_markers, dvs_cam_height, dvs_cam_width):
    if os.path.exists(data_file_name):
        os.remove(data_file_name)

    data_file = tables.open_file(data_file_name, mode='a')
    data = {}

    data['timestamp'] = data_file.create_earray(
        data_file.root, 'timestamp', tables.atom.UInt64Atom(), (0,))

    for i in range(n_cameras):
        data[f'camera_{i}_events'] = data_file.create_earray(
            data_file.root, f'camera_{i}_events', tables.atom.UInt8Atom(),
            (0, dvs_cam_height[i], dvs_cam_width[i], 2))

    for i in range(n_cameras):
        data[f'camera_{i}_rotation'] = {}

    for i in range(n_cameras):
        data[f'camera_{i}_translation'] = {}

    for i in range(n_cameras):
        data[f'camera_{i}_xy'] = {}

    g_props = data_file.create_group(data_file.root, 'props')
    for prop_name in props_markers.keys():
        g_prop = data_file.create_group(g_props, prop_name)

        for i in range(n_cameras):
            data[f'camera_{i}_rotation'][prop_name] = data_file.create_earray(
                g_prop, f'camera_{i}_rotation', tables.atom.Float64Atom(), (0, 3, 3))

        for i in range(n_cameras):
            data[f'camera_{i}_translation'][prop_name] = data_file.create_earray(
                g_prop, f'camera_{i}_translation', tables.atom.Float64Atom(), (0, 3, 1))

        for i in range(n_cameras):
            data[f'camera_{i}_xy'][prop_name] = data_file.create_earray(
                g_prop, f'camera_{i}_xy', tables.atom.Float64Atom(), (0, 2))

    return data_file, data


def projection(path_data):

    if not os.path.exists(path_data):
        raise ValueError('path does not exist')

    vicon_address, vicon_port = '127.0.0.1', 801

    event_distinguish_polarity = False

    #vicon_marker_error_threshold = np.inf    # millimeters
    vicon_marker_error_threshold = 30.0       # millimeters

    vicon_bad_pose_timeout = 100
    vicon_buffer_length = 300

    dvs_cam_height = [np.uint32(260) for i in range(n_cameras)]
    dvs_cam_width = [np.uint32(346) for i in range(n_cameras)]
    dvs_cam_origin_x_offset = [dvs_cam_width[i] / 2 for i in range(n_cameras)]
    dvs_cam_origin_y_offset = [dvs_cam_height[i] / 2 for i in range(n_cameras)]
    dvs_cam_nominal_f_len = [4.0 for i in range(n_cameras)]
    dvs_cam_pixel_mm = [1.8e-2 for i in range(n_cameras)]

    # Read recording info from JSON
    with open(f'{path_data}/info.json', 'r') as info_json_file:
        info_json = json.load(info_json_file)


    ##################################################################

    # === READ PROPS DATA ===
    props_markers = {}    # contains the translation of each marker, relative to prop origin
    props_meshes = {}     # contains prop STL meshes (polygon, translation, vertex)
    props_labels = {}     # contains integer > 0 class labels of the props
    props_dilation = {}   # contains dilation kernels for the mask of each prop

    props_names = list(info_json['prop_marker_files'].keys())
    for prop_name in props_names:
        with open(info_json['prop_marker_files'][prop_name], 'r') as marker_file:
            markers = json.load(marker_file)
        props_markers[prop_name] = markers
        mesh = stl.mesh.Mesh.from_file(info_json['prop_mesh_files'][prop_name]).vectors.transpose(0, 2, 1)
        props_meshes[prop_name] = mesh
        props_labels[prop_name] = info_json['prop_labels'][prop_name]
        props_dilation[prop_name] = np.ones((3, 3), 'uint8')

    #props_dilation['kth_hammer'] = np.ones((4, 4), 'uint8')
    #props_dilation['kth_screwdriver'] = np.ones((4, 4), 'uint8')
    #props_dilation['kth_spanner'] = np.ones((4, 4), 'uint8')


    ##################################################################

    # === READ CALIBRATION FILES ===

    path_projection = info_json['projection_calibration_path']

    v_to_dvs_rotation_file = [f'{path_projection}/v_to_dv_{i}_rotation.npy' for i in range(n_cameras)]
    v_to_dvs_rotation = [np.load(name) for name in v_to_dvs_rotation_file]

    v_to_dvs_translation_file = [f'{path_projection}/v_to_dv_{i}_translation.npy' for i in range(n_cameras)]
    v_to_dvs_translation = [np.load(name) for name in v_to_dvs_translation_file]

    v_to_dvs_f_len_scale_file = [f'{path_projection}/v_to_dv_{i}_focal_length_scale.npy' for i in range(n_cameras)]
    v_to_dvs_f_len_scale = [np.load(name) for name in v_to_dvs_f_len_scale_file]
    v_to_dvs_f_len = [dvs_cam_nominal_f_len[i] * v_to_dvs_f_len_scale[i] for i in range(n_cameras)]

    v_to_dvs_x_scale_file = [f'{path_projection}/v_to_dv_{i}_x_scale.npy' for i in range(n_cameras)]
    v_to_dvs_x_scale = [np.load(name) for name in v_to_dvs_x_scale_file]


    ##################################################################

    # === DATA FILE NAMES ===

    raw_events_file_name = [f'{path_data}/raw_event_{i}.h5' for i in range(n_cameras)]
    raw_frames_file_name = [f'{path_data}/raw_frame_{i}.h5' for i in range(n_cameras)]
    raw_poses_file_name = f'{path_data}/raw_pose.h5'

    final_events_file_name = f'{path_data}/event.h5'
    # final_frames_file_name = f'{path_data}/frame.h5'
    final_poses_file_name = f'{path_data}/pose.h5'

    # events_video_file_name = [f'{path_data}/event_{i}_video.avi' for i in range(n_cameras)]
    # frames_video_file_name = [f'{path_data}/frame_{i}_video.avi' for i in range(n_cameras)]


    ##################################################################

    # === COMPUTE MESH TO V0 TRANSFORM ===

    # load raw Vicon pose data
    raw_poses_file = tables.open_file(raw_poses_file_name, mode='r')

    # optimisation
    method = 'nelder-mead'
    options = {'disp': True, 'maxiter': 50000, 'maxfev': 100000, 'xatol': 1e-10, 'fatol': 1e-10}

    def err_fun_mesh_to_v0(m, mesh_ps, vicon_ps, v0_to_v_rotations, v0_to_v_translations):
        error= 0

        for mesh_p, vicon_p, v0_to_v_translation, v0_to_v_rotation in zip(
                mesh_ps, vicon_ps, v0_to_v_translations, v0_to_v_rotations):

            mesh_to_v0_rotation = euler_angles_to_rotation_matrix(m[0:3])
            mesh_to_v0_translation = m[3:6, np.newaxis]

            v0_p = np.dot(mesh_to_v0_rotation, mesh_p) + mesh_to_v0_translation

            output = np.dot(v0_to_v_rotation, v0_p) + v0_to_v_translation
            difference = output - vicon_p
            error += np.sqrt(np.mean(difference ** 2))

        return error

    # get transforms from mesh space to Vicon space zero
    mesh_to_v0_rotation = {}
    mesh_to_v0_translation = {}

    for prop_name in props_names:
        mesh_markers = []
        vicon_markers = []
        v0_to_v_rotations = []
        v0_to_v_translations = []

        #m = np.zeros(6, dtype='float64')
        #m = np.random.rand(6) * 2 * np.pi - np.pi
        m = np.array([-0.23134114, -2.83301699,  2.81133181, -1.54930157, -2.35502374,  2.1578688 ])

        indices = np.linspace(0, len(raw_poses_file.root.timestamp) - 1, 50, dtype='uint64')
        for i in indices:
            rotation = raw_poses_file.root.props[prop_name].vicon_rotation[i]
            translation = raw_poses_file.root.props[prop_name].vicon_translation[i]

            if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
                continue

            mesh_marker = np.empty((3, len(props_markers[prop_name])), dtype='float64')
            vicon_marker = np.empty(mesh_marker.shape, dtype='float64')
            for i_marker, marker_name in enumerate(props_markers[prop_name].keys()):
                mesh_marker[:, i_marker] = props_markers[prop_name][marker_name]
                vicon_marker[:, i_marker] = raw_poses_file.root.props[prop_name].vicon_markers[marker_name][i][:, 0]

            mesh_markers.append(mesh_marker)
            vicon_markers.append(vicon_marker)
            v0_to_v_rotations.append(rotation)
            v0_to_v_translations.append(translation)

        print(m)
        err = err_fun_mesh_to_v0(
            m, mesh_markers, vicon_markers, v0_to_v_rotations, v0_to_v_translations)
        print(f'{prop_name} mesh to vicon transform: original guess has error: {err}')

        result = minimize(
            err_fun_mesh_to_v0, m, method=method, options=options,
            args=(mesh_markers, vicon_markers, v0_to_v_rotations, v0_to_v_translations))
        m = result['x']

        print(m)
        err = err_fun_mesh_to_v0(
            m, mesh_markers, vicon_markers, v0_to_v_rotations, v0_to_v_translations)
        print(f'{prop_name} mesh to vicon transform: final result has error: {err}')

        mesh_to_v0_rotation[prop_name] = euler_angles_to_rotation_matrix(m[0:3])
        mesh_to_v0_translation[prop_name] = m[3:6, np.newaxis]


    ##################################################################

    # === PREPROCESS VICON POSE DATA ===
    print('begin preprocessing')

    # create final Vicon pose data file
    final_poses_file, final_poses_data = create_pytables_final_poses_file(
        final_poses_file_name, n_cameras, props_markers)

    # get raw Vicon pose data iterators
    raw_poses_iter = {}
    timestamp = raw_poses_file.root.timestamp
    raw_poses_iter['timestamp'] = timestamp.iterrows()
    raw_poses_iter['vicon_rotation'] = {}
    raw_poses_iter['vicon_translation'] = {}
    raw_poses_iter['vicon_markers'] = {}
    for prop_name in props_names:
        rotation = raw_poses_file.root.props[prop_name].vicon_rotation
        raw_poses_iter['vicon_rotation'][prop_name] = rotation.iterrows()
        translation = raw_poses_file.root.props[prop_name].vicon_translation
        raw_poses_iter['vicon_translation'][prop_name] = translation.iterrows()
        raw_poses_iter['vicon_markers'][prop_name] = {}
        for marker_name in props_markers[prop_name].keys():
            marker = raw_poses_file.root.props[prop_name].vicon_markers[marker_name]
            raw_poses_iter['vicon_markers'][prop_name][marker_name] = marker.iterrows()

    # initialise good Vicon pose buffer
    vicon_timestamp_buffer = {}
    vicon_rotation_buffer = {}
    vicon_translation_buffer = {}
    vicon_markers_buffer = {}
    for prop_name in props_names:
        vicon_timestamp_buffer[prop_name] = deque(maxlen=vicon_buffer_length)
        vicon_rotation_buffer[prop_name] = deque(maxlen=vicon_buffer_length)
        vicon_translation_buffer[prop_name] = deque(maxlen=vicon_buffer_length)
        vicon_markers_buffer[prop_name] = {}
        for marker_name in props_markers[prop_name].keys():
            vicon_markers_buffer[prop_name][marker_name] = deque(maxlen=vicon_buffer_length)

    pose = get_next_pose(raw_poses_iter)

    # append to good Vicon pose buffers
    for prop_name in props_names:
        timestamp = pose['timestamp']
        vicon_timestamp_buffer[prop_name].append(timestamp)
        rotation = pose['vicon_rotation'][prop_name]
        vicon_rotation_buffer[prop_name].append(rotation)
        translation = pose['vicon_translation'][prop_name]
        vicon_translation_buffer[prop_name].append(translation)
        for marker_name in props_markers[prop_name].keys():
            marker = pose['vicon_markers'][prop_name][marker_name]
            vicon_markers_buffer[prop_name][marker_name].append(marker)

    pose = get_next_pose(raw_poses_iter)

    bad_pose_count = 0
    while True:

        try:
            pose_new = get_next_pose(raw_poses_iter)
        except StopIteration:
            break

        final_poses_data['timestamp'].append([pose['timestamp']])

        # for each prop
        for prop_name in props_names:
            v0_to_v_rotation = pose['vicon_rotation'][prop_name]
            rotation = mesh_to_v0_rotation[prop_name]
            rotation = np.dot(v0_to_v_rotation, rotation)
            final_poses_data['rotation'][prop_name].append([rotation])
            for i in range(n_cameras):
                cam_rotation = np.dot(v_to_dvs_rotation[i], rotation)
                final_poses_data[f'camera_{i}_rotation'][prop_name].append([cam_rotation])

            v0_to_v_translation = pose['vicon_translation'][prop_name]
            translation = mesh_to_v0_translation[prop_name]
            translation = np.dot(v0_to_v_rotation, translation) + v0_to_v_translation
            final_poses_data['translation'][prop_name].append([translation])
            for i in range(n_cameras):
                cam_translation = np.dot(v_to_dvs_rotation[i], translation) + v_to_dvs_translation[i]
                final_poses_data[f'camera_{i}_translation'][prop_name].append([cam_translation])

            # check current Vicon pose
            pose_is_good = True
            for marker_name in props_markers[prop_name].keys():
                marker = pose['vicon_markers'][prop_name][marker_name]
                marker_old = vicon_markers_buffer[prop_name][marker_name][-1]
                if not all(np.isfinite(marker)) or any(
                        np.abs(marker - marker_old) >= vicon_marker_error_threshold):
                    pose_is_good = False

            # extrapolate bad Vicon pose
            if not pose_is_good:
                #print('DEBUG: bad pose')
                bad_pose_count += 1

                if bad_pose_count < vicon_bad_pose_timeout and len(vicon_timestamp_buffer[prop_name]) > 1:
                    #print('DEBUG: extrapolating bad pose')
                    x = np.array(vicon_timestamp_buffer[prop_name])

                    y = np.array(vicon_rotation_buffer[prop_name])
                    f = interp1d(x, y, axis=0, fill_value='extrapolate', kind='linear')
                    v0_to_v_rotation = f(pose['timestamp'])
                    rotation = mesh_to_v0_rotation[prop_name]
                    rotation = np.dot(v0_to_v_rotation, rotation)
                    final_poses_data['rotation'][prop_name][-1] = rotation
                    for i in range(n_cameras):
                        cam_rotation = np.dot(v_to_dvs_rotation[i], rotation)
                        final_poses_data[f'camera_{i}_rotation'][prop_name][-1] = cam_rotation

                    y = np.array(vicon_translation_buffer[prop_name])
                    f = interp1d(x, y, axis=0, fill_value='extrapolate', kind='linear')
                    v0_to_v_translation = f(pose['timestamp'])
                    translation = mesh_to_v0_translation[prop_name]
                    translation = np.dot(v0_to_v_rotation, translation) + v0_to_v_translation
                    final_poses_data['translation'][prop_name][-1] = translation
                    for i in range(n_cameras):
                        cam_translation = np.dot(v_to_dvs_rotation[i], translation) + v_to_dvs_translation[i]
                        final_poses_data[f'camera_{i}_translation'][prop_name][-1] = cam_translation

                else: # bad pose timeout
                    #print('DEBUG: bad pose timeout')
                    pose_is_good = True

                    # clear pose buffer
                    vicon_timestamp_buffer[prop_name].clear()
                    vicon_rotation_buffer[prop_name].clear()
                    vicon_translation_buffer[prop_name].clear()
                    for marker_name in props_markers[prop_name].keys():
                        vicon_markers_buffer[prop_name][marker_name].clear()

                    # void bad pose data
                    a = max(0, len(final_poses_data['timestamp']) - vicon_bad_pose_timeout + 1)
                    b = len(final_poses_data['timestamp'])

                    final_poses_data['rotation'][prop_name][a:b] = np.nan
                    for i in range(n_cameras):
                        final_poses_data[f'camera_{i}_rotation'][prop_name][a:b] = np.nan

                    final_poses_data['translation'][prop_name][a:b] = np.nan
                    for i in range(n_cameras):
                        final_poses_data[f'camera_{i}_translation'][prop_name][a:b] = np.nan

            # append good Vicon pose to buffer
            if pose_is_good:
                #print('DEBUG: good pose')
                bad_pose_count = 0

                # append to good Vicon pose buffers
                timestamp = pose['timestamp']
                vicon_timestamp_buffer[prop_name].append(timestamp)

                rotation = pose['vicon_rotation'][prop_name]
                vicon_rotation_buffer[prop_name].append(rotation)

                translation = pose['vicon_translation'][prop_name]
                vicon_translation_buffer[prop_name].append(translation)

                for marker_name in props_markers[prop_name].keys():
                    marker = pose['vicon_markers'][prop_name][marker_name]
                    vicon_markers_buffer[prop_name][marker_name].append(marker)


        pose = pose_new


    raw_poses_file.close()
    final_poses_file.close()

    print('finished preprocessing')


    ##################################################################


    # load raw DVS event data
    raw_events_file = []
    raw_events_iter = []
    for i in range(n_cameras):
        e_file = tables.open_file(raw_events_file_name[i], mode='r')
        e_iter = {}
        e_iter[f'timestamp_{i}'] = e_file.root[f'timestamp_{i}'].iterrows()
        e_iter[f'polarity_{i}'] = e_file.root[f'polarity_{i}'].iterrows()
        e_iter[f'xy_undistorted_{i}'] = e_file.root[f'xy_undistorted_{i}'].iterrows()
        raw_events_file.append(e_file)
        raw_events_iter.append(e_iter)

    # load raw DVS frame data
    raw_frames_file = []
    raw_frames_iter = []
    for i in range(n_cameras):
        f_file = tables.open_file(raw_frames_file_name[i], mode='r')
        f_iter = {}
        f_iter[f'timestamp_{i}'] = f_file.root[f'timestamp_{i}'].iterrows()
        f_iter[f'image_undistorted_{i}'] = f_file.root[f'image_undistorted_{i}'].iterrows()
        raw_frames_file.append(f_file)
        raw_frames_iter.append(f_iter)


    # Seek initial DVS event and frame
    dvs_start_timestamp = seek_dvs_start(
        n_cameras, dvs_cam_height, dvs_cam_width,
        raw_events_file, raw_frames_file, start_signal_delay_secs=3)

    event = [None for i in range(n_cameras)]
    frame = [None for i in range(n_cameras)]
    for i in range(n_cameras):
        event[i] = get_next_event(i, raw_events_iter[i])
        while event[i][f'timestamp_{i}'] < dvs_start_timestamp[i]:
            event[i] = get_next_event(i, raw_events_iter[i])
        frame[i] = get_next_frame(i, raw_frames_iter[i])
        while frame[i][f'timestamp_{i}'] < dvs_start_timestamp[i]:
            frame[i] = get_next_frame(i, raw_frames_iter[i])


    # initialise temp memory
    pose_xy = [np.zeros((2,), dtype='float64') for i in range(n_cameras)]
    event_pos = [np.zeros((dvs_cam_height[i], dvs_cam_width[i]), dtype='uint64') for i in range(n_cameras)]
    event_neg = [np.zeros((dvs_cam_height[i], dvs_cam_width[i]), dtype='uint64') for i in range(n_cameras)]
    event_image = [np.zeros((dvs_cam_height[i], dvs_cam_width[i], 3), dtype='uint8') for i in range(n_cameras)]
    event_accumulated = [np.zeros((dvs_cam_height[i], dvs_cam_width[i], 2), dtype='uint8') for i in range(n_cameras)]
    # frame_image = [np.zeros((dvs_cam_height[i], dvs_cam_width[i], 3), dtype='uint8') for i in range(n_cameras)]
    # frame_label = [np.zeros((dvs_cam_height[i], dvs_cam_width[i]), dtype='int8') for i in range(n_cameras)]
    # frame_label_depth = [np.zeros((dvs_cam_height[i], dvs_cam_width[i]), dtype='float64') for i in range(n_cameras)]
    prop_masks = [{prop_name: np.empty((dvs_cam_height[i], dvs_cam_width[i]), dtype='uint8')
                   for prop_name in props_names} for i in range(n_cameras)]

    # load final Vicon pose data file
    final_poses_file = tables.open_file(final_poses_file_name, mode='r')
    final_poses_iter = {}

    timestamp = final_poses_file.root.timestamp
    final_poses_iter['timestamp'] = timestamp.iterrows()

    final_poses_iter['rotation'] = {}
    for i in range(n_cameras):
        final_poses_iter[f'camera_{i}_rotation'] = {}

    final_poses_iter['translation'] = {}
    for i in range(n_cameras):
        final_poses_iter[f'camera_{i}_translation'] = {}

    for prop_name in props_names:
        rotation = final_poses_file.root.props[prop_name].rotation
        final_poses_iter['rotation'][prop_name] = rotation.iterrows()
        for i in range(n_cameras):
            cam_rotation = final_poses_file.root.props[prop_name][f'camera_{i}_rotation']
            final_poses_iter[f'camera_{i}_rotation'][prop_name] = cam_rotation.iterrows()

        translation = final_poses_file.root.props[prop_name].translation
        final_poses_iter['translation'][prop_name] = translation.iterrows()
        for i in range(n_cameras):
            cam_translation = final_poses_file.root.props[prop_name][f'camera_{i}_translation']
            final_poses_iter[f'camera_{i}_translation'][prop_name] = cam_translation.iterrows()

    # create final DVS event and frame data files
    # final_events_file, final_events_data = create_pytables_final_events_file(final_events_file_name, n_cameras)
    # final_frames_file, final_frames_data = create_pytables_final_frames_file(final_frames_file_name, n_cameras)


    poses_events_file_name = f'{path_data}/data_pose_event.h5'
    poses_events_file, poses_events_data = create_pytables_poses_events_file(
        poses_events_file_name, n_cameras, props_markers, dvs_cam_height, dvs_cam_width)


    # # initialise video recordings
    # events_video_file = [cv2.VideoWriter(
    #     events_video_file_name[i], cv2.VideoWriter_fourcc(*'MJPG'),
    #     330, (dvs_cam_width[i], dvs_cam_height[i])) for i in range(n_cameras)]
    # frames_video_file = [cv2.VideoWriter(
    #     frames_video_file_name[i], cv2.VideoWriter_fourcc(*'MJPG'),
    #     330, (dvs_cam_width[i], dvs_cam_height[i])) for i in range(n_cameras)]

    # get first Vicon pose
    pose = get_next_pose(final_poses_iter)



    # === MAIN LOOP ===
    done_event = [False for i in range(n_cameras)]
    done_frame = [True for i in range(n_cameras)]
    while not all(done_event) or not all(done_frame):

        try:
            pose_new = get_next_pose(final_poses_iter)
            pose_midway = pose['timestamp'] / 2 + pose_new['timestamp'] / 2
        except StopIteration:
            #print('DEBUG: out of vicon poses')
            break

        print()
        print('Vicon pose timestamp: ', pose['timestamp'])

        # write timestamp to h5 file
        poses_events_data['timestamp'].append([pose['timestamp']])

        for prop_name in props_names:

            # compute prop mask for each camera
            for i in range(n_cameras):
                prop_masks[i][prop_name].fill(0)

                xy = pose_xy[i]

                # write pose to h5 file
                poses_events_data[f'camera_{i}_rotation'][prop_name].append(
                    [pose[f'camera_{i}_rotation'][prop_name]])
                poses_events_data[f'camera_{i}_translation'][prop_name].append(
                    [pose[f'camera_{i}_translation'][prop_name]])

                mesh_to_dvs_rotation = pose[f'camera_{i}_rotation'][prop_name]
                mesh_to_dvs_translation = pose[f'camera_{i}_translation'][prop_name]

                if not np.isfinite(mesh_to_dvs_rotation).all() or not np.isfinite(mesh_to_dvs_translation).all():
                    # write xy to h5 file
                    xy[:] = np.nan
                    poses_events_data[f'camera_{i}_xy'][prop_name].append([xy])
                    continue

                # write xy to h5 file
                dvs_space_p = pose[f'camera_{i}_translation'][prop_name].T[0]
                xy[:] = dvs_space_p[:2] * (1 / dvs_space_p[2])
                xy *= v_to_dvs_f_len[i]
                xy /= dvs_cam_pixel_mm[i]
                xy *= v_to_dvs_x_scale[i]
                xy += [dvs_cam_origin_x_offset[i], dvs_cam_origin_y_offset[i]]
                if not (0 <= xy[0] < dvs_cam_width[i]) or not (0 <= xy[1] < dvs_cam_height[i]):
                    xy[:] = np.nan
                poses_events_data[f'camera_{i}_xy'][prop_name].append([xy])

                # transform to DVS camera space
                dvs_space_p = np.matmul(mesh_to_dvs_rotation, props_meshes[prop_name]) + mesh_to_dvs_translation
                dvs_space_p[:, :2, :] *= (1 / dvs_space_p[:, np.newaxis, 2, :])
                dvs_space_p = dvs_space_p[:, :2, :]
                dvs_space_p *= v_to_dvs_f_len[i]
                dvs_space_p /= dvs_cam_pixel_mm[i]
                dvs_space_p *= v_to_dvs_x_scale[i]
                dvs_space_p += [[dvs_cam_origin_x_offset[i]], [dvs_cam_origin_y_offset[i]]]
                dvs_space_p_int = np.rint(dvs_space_p).astype('int32')

                # transpose points for OpenCV
                dvs_space_p_int = dvs_space_p_int.transpose(0, 2, 1)

                # compute prop mask
                cv2.fillPoly(prop_masks[i][prop_name], dvs_space_p_int, 255)
                prop_masks[i][prop_name] = cv2.dilate(prop_masks[i][prop_name], props_dilation[prop_name])



        # process DVS events
        for i in range(n_cameras):
            if not done_event[i]:

                pos = event_pos[i]
                neg = event_neg[i]
                image = event_image[i]
                accumulated = event_accumulated[i]

                pos.fill(0)
                neg.fill(0)
                image.fill(0)
                accumulated.fill(0)

                timestamp = event[i][f'timestamp_{i}'] - dvs_start_timestamp[i]
                while timestamp < pose_midway:

                    # check DVS event is in frame
                    xy_int = np.rint(event[i][f'xy_undistorted_{i}']).astype('int32')
                    xy_bounded = all(xy_int >= 0) and all(xy_int < [dvs_cam_width[i], dvs_cam_height[i]])

                    if xy_bounded:
                        if event[i][f'polarity_{i}']:
                            accumulated[xy_int[1], xy_int[0], 1] += 1
                            pos[xy_int[1], xy_int[0]] += 1
                        else:
                            accumulated[xy_int[1], xy_int[0], 0] += 1
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
                        for prop_name in props_names:
                            mask = prop_masks[i][prop_name].astype('bool')
                            prop_depth = pose[f'camera_{i}_translation'][prop_name][2, 0]
                            if mask[xy_int[1], xy_int[0]]:
                                if prop_depth < label_depth:
                                    label = props_labels[prop_name]
                                    label_depth = prop_depth

                        # record final event data
                        # final_events_data[f'timestamp_{i}'].append([timestamp])
                        # final_events_data[f'polarity_{i}'].append([event[i][f'polarity_{i}']])
                        # final_events_data[f'xy_undistorted_{i}'].append([event[i][f'xy_undistorted_{i}']])
                        # final_events_data[f'label_{i}'].append([label])

                    try:
                        event[i] = get_next_event(i, raw_events_iter[i])
                    except StopIteration:
                        #print(f'DEBUG: out of dv {i} events')
                        done_event[i] = True
                        break

                    timestamp = event[i][f'timestamp_{i}'] - dvs_start_timestamp[i]


                # write accumulated events to h5 file
                poses_events_data[f'camera_{i}_events'].append([event_accumulated[i]])


                # fill DVS event image with events, then mask it
                for prop_name in props_names:
                    # FIXME: image[mask] = grey will overwrite events from other props
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

                # # write DVS event video
                # events_video_file[i].write(image)

                # show DVS event image
                cv2.imshow(f'event {i} image', image)
                k = cv2.waitKey(1)
                if k == ord('q'):
                    cv2.destroyWindow(f'event {i} image')
                    done_event[i] = True



        # # process DVS frames
        # for i in range(n_cameras):
        #     if not done_frame[i]:

        #         timestamp = frame[i][f'timestamp_{i}'] - dvs_start_timestamp[i]
        #         while timestamp <= pose['timestamp']:
        #             image = frame_image[i]
        #             label = frame_label[i]
        #             label_depth = frame_label_depth[i]

        #             # get frame label
        #             label.fill(0)
        #             label_depth.fill(np.inf)
        #             for prop_name in props_names:
        #                 mask = prop_masks[i][prop_name].astype('bool')
        #                 prop_depth = pose[f'camera_{i}_translation'][prop_name][2, 0]
        #                 label[mask & (prop_depth < label_depth)] = props_labels[prop_name]
        #                 label_depth[mask & (prop_depth < label_depth)] = prop_depth

        #             # record final frame data
        #             final_frames_data[f'timestamp_{i}'].append([timestamp])
        #             final_frames_data[f'image_undistorted_{i}'].append([frame[i][f'image_undistorted_{i}']])
        #             final_frames_data[f'label_{i}'].append([label])


        #             # mask DVS frame image
        #             image[:] = frame[i][f'image_undistorted_{i}']
        #             for prop_name in props_names:
        #                 mask = prop_masks[i][prop_name].astype('bool')
        #                 image[mask, :] = blue

        #             # # uncomment to plot prop centre translation
        #             # dvs_space_p = pose[f'camera_{i}_translation'][prop_name].T[0]
        #             # dvs_space_p[:2] *= (1 / dvs_space_p[2])
        #             # dvs_space_p = dvs_space_p[:2]
        #             # dvs_space_p *= v_to_dvs_f_len[i]
        #             # dvs_space_p /= dvs_cam_pixel_mm[i]
        #             # dvs_space_p *= v_to_dvs_x_scale[i]
        #             # dvs_space_p += [dvs_cam_origin_x_offset[i], dvs_cam_origin_y_offset[i]]
        #             # dvs_space_p_int = np.rint(dvs_space_p).astype('int32')
        #             # cv2.circle(image, (dvs_space_p_int[0], dvs_space_p_int[1]), 3, (0, 255, 0), -1)

        #             # # write DVS frame video
        #             # frames_video_file[i].write(image)

        #             # show DVS frame image
        #             cv2.imshow(f'frame {i} image', image)
        #             k = cv2.waitKey(1)
        #             if k == ord('q'):
        #                 cv2.destroyWindow(f'frame {i} image')
        #                 done_frame[i] = True

        #             try:
        #                 frame[i] = get_next_frame(i, raw_frames_iter[i])
        #             except StopIteration:
        #                 #print(f'DEBUG: out of dv {i} frames')
        #                 done_frame[i] = True
        #                 break

        #             timestamp = frame[i][f'timestamp_{i}'] - dvs_start_timestamp[i]


        pose = pose_new



    # cleanup
    for i in range(n_cameras):
        raw_events_file[i].close()
        raw_frames_file[i].close()
    # final_events_file.close()
    # final_frames_file.close()
    final_poses_file.close()
    poses_events_file.close()

    # for i in range(n_cameras):
    #     events_video_file[i].release()
    #     frames_video_file[i].release()

    return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('path_data')
    args = parser.parse_args()
    projection(args.path_data)
