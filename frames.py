import os
import tables


def create_pytables_frames_file(frames_file_name, n_cameras):
    if os.path.exists(frames_file_name):
        os.remove(frames_file_name)

    frames_file = tables.open_file(frames_file_name, mode='a')
    data = {}

    for i in range(n_cameras):
        data[f'timestamp_{i}'] = frames_file.create_earray(
            frames_file.root, f'timestamp_{i}', tables.atom.UInt64Atom(), (0,))
        data[f'image_raw_{i}'] = frames_file.create_earray(
            frames_file.root, f'image_raw_{i}', tables.atom.UInt8Atom(), (0, 260, 346, 3))
        data[f'image_undistorted_{i}'] = frames_file.create_earray(
            frames_file.root, f'image_undistorted_{i}', tables.atom.UInt8Atom(), (0, 260, 346, 3))
        data[f'label_{i}'] = frames_file.create_earray(
            frames_file.root, f'label_{i}', tables.atom.UInt8Atom(), (0, 260, 346))

    return frames_file, data
