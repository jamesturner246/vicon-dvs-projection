import os
import tables


def create_pytables_poses_file(poses_file_name, n_cameras, props_markers):
    if os.path.exists(poses_file_name):
        os.remove(poses_file_name)

    poses_file = tables.open_file(poses_file_name, mode='a')
    data = {}

    data['timestamp'] = poses_file.create_earray(
        poses_file.root, 'timestamp', tables.atom.UInt64Atom(), (0,))

    data['vicon_rotation'] = {}
    data['rotation'] = {}
    for i in range(n_cameras):
        data[f'camera_{i}_rotation'] = {}

    data['vicon_translation'] = {}
    data['translation'] = {}
    for i in range(n_cameras):
        data[f'camera_{i}_translation'] = {}

    data['vicon_markers'] = {}

    g_props = poses_file.create_group(poses_file.root, 'props')
    for prop_name in props_markers.keys():
        g_prop = poses_file.create_group(g_props, prop_name)

        data['vicon_rotation'][prop_name] = poses_file.create_earray(
            g_prop, 'vicon_rotation', tables.atom.Float64Atom(), (0, 3, 3))
        data['rotation'][prop_name] = poses_file.create_earray(
            g_prop, 'rotation', tables.atom.Float64Atom(), (0, 3, 3))
        for i in range(n_cameras):
            data[f'camera_{i}_rotation'][prop_name] = poses_file.create_earray(
                g_prop, f'camera_{i}_rotation', tables.atom.Float64Atom(), (0, 3, 3))

        data['vicon_translation'][prop_name] = poses_file.create_earray(
            g_prop, 'vicon_translation', tables.atom.Float64Atom(), (0, 3, 1))
        data['translation'][prop_name] = poses_file.create_earray(
            g_prop, 'translation', tables.atom.Float64Atom(), (0, 3, 1))
        for i in range(n_cameras):
            data[f'camera_{i}_translation'][prop_name] = poses_file.create_earray(
                g_prop, f'camera_{i}_translation', tables.atom.Float64Atom(), (0, 3, 1))

        data['vicon_markers'][prop_name] = {}
        g_vicon_markers = poses_file.create_group(g_prop, 'vicon_markers')
        for marker_name in props_markers[prop_name].keys():
            data['vicon_markers'][prop_name][marker_name] = poses_file.create_earray(
                g_vicon_markers, marker_name, tables.atom.Float64Atom(), (0, 3, 1))

    return poses_file, data
