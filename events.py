import os
import tables


def create_pytables_raw_events_file(events_file_name, n_cameras):
    if os.path.exists(events_file_name):
        os.remove(events_file_name)

    events_file = tables.open_file(events_file_name, mode='a')
    data = {}

    for i in range(n_cameras):
        data[f'timestamp_{i}'] = events_file.create_earray(
            events_file.root, f'timestamp_{i}', tables.atom.UInt64Atom(), (0,))
        data[f'polarity_{i}'] = events_file.create_earray(
            events_file.root, f'polarity_{i}', tables.atom.BoolAtom(), (0,))
        data[f'xy_raw_{i}'] = events_file.create_earray(
            events_file.root, f'xy_raw_{i}', tables.atom.UInt16Atom(), (0, 2))
        data[f'xy_undistorted_{i}'] = events_file.create_earray(
            events_file.root, f'xy_undistorted_{i}', tables.atom.Float64Atom(), (0, 2))

    return events_file, data


def create_pytables_final_events_file(events_file_name, n_cameras):
    if os.path.exists(events_file_name):
        os.remove(events_file_name)

    events_file = tables.open_file(events_file_name, mode='a')
    data = {}

    for i in range(n_cameras):
        data[f'timestamp_{i}'] = events_file.create_earray(
            events_file.root, f'timestamp_{i}', tables.atom.UInt64Atom(), (0,))
        data[f'polarity_{i}'] = events_file.create_earray(
            events_file.root, f'polarity_{i}', tables.atom.BoolAtom(), (0,))
        data[f'xy_undistorted_{i}'] = events_file.create_earray(
            events_file.root, f'xy_undistorted_{i}', tables.atom.Float64Atom(), (0, 2))
        data[f'label_{i}'] = events_file.create_earray(
            events_file.root, f'label_{i}', tables.atom.Int8Atom(), (0,))

    return events_file, data
