import pathlib
import json
import tables
import numpy as np
import tensorflow as tf
import cv2


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _serialize_example(shape, events, xy):
    feature = {
        'shape': _bytes_feature(shape),
        'events': _bytes_feature(events),
        'xy': _bytes_feature(xy),
    }
    proto_example = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized_example = proto_example.SerializeToString()
    return serialized_example


def tfrecord_from_h5(in_directory, h5_file_name, out_directory=None, tfr_file_name=None):
    i = 0 # camera index

    if out_directory is None:
        out_directory = in_directory

    if tfr_file_name is None:
        tfr_file_name = h5_file_name.split('.')[0] + '.tfrecord'

    h5_file_path = f'{in_directory}/{h5_file_name}'
    tfr_file_path = f'{out_directory}/{tfr_file_name}'
    print(h5_file_path)
    print(tfr_file_path)

    # xy matrix (all props)
    all_xy = np.empty((2, 3), dtype='float32')

    # Read JSON recording info
    with open(f'{in_directory}/info.json', 'r') as info_json_file:
        info_json = json.load(info_json_file)
    props_labels = info_json['prop_labels']

    # Read HDF5 data file
    f = tables.open_file(h5_file_path, 'r')
    props_names = [item[0] for item in f.root.props._v_children.items()]

    # HDF5 data iterators
    iter_timestamp = f.root.timestamp.iterrows()
    iter_events = f.root[f'camera_{i}_events'].iterrows()
    iter_rotation = {}
    iter_translation = {}
    iter_xy = {}
    for prop_name in props_names:
        iter_rotation[prop_name] = f.root.props[prop_name][f'camera_{i}_rotation'].iterrows()
        iter_translation[prop_name] = f.root.props[prop_name][f'camera_{i}_translation'].iterrows()
        iter_xy[prop_name] = f.root.props[prop_name][f'camera_{i}_xy'].iterrows()

    # Write TFRecords
    with tf.io.TFRecordWriter(tfr_file_path) as tfr_writer:

        done = False
        while not done:

            try:
                timestamp = next(iter_timestamp)
                events = next(iter_events)
                rotation = {}
                translation = {}
                xy = {}
                for prop_name in props_names:
                    rotation[prop_name] = next(iter_rotation[prop_name])
                    translation[prop_name] = next(iter_translation[prop_name])
                    xy[prop_name] = next(iter_xy[prop_name])

            except StopIteration:
                done = True
                continue

            all_xy.fill(np.nan)
            for prop_name in props_names:
                all_xy[:, props_labels[prop_name] - 1] = xy[prop_name]

            tensor_events = tf.convert_to_tensor(events, dtype=tf.uint8)
            tensor_all_xy = tf.convert_to_tensor(all_xy, dtype=tf.float32)

            s_shape = tf.io.serialize_tensor(tf.shape(tensor_events))
            s_events = tf.io.serialize_tensor(tensor_events)
            s_xy = tf.io.serialize_tensor(tensor_all_xy)

            example = _serialize_example(s_shape, s_events, s_xy)
            tfr_writer.write(example)

    f.close()
    return


def convert_all():
    out_files = [
        "screwdriver",
        "hammer",
        "spanner",
        "hammer_screwdriver",
        "screwdriver_spanner",
    ]

    dirs = [
        "./data/20220112_jpt_floating_kth_screwdriver",
        "./data/20220113_jpt_floating_kth_hammer",
        "./data/20220113_jpt_floating_kth_spanner",
        "./data/20220114_jpt_floating_kth_hammer_kth_screwdriver",
        "./data/20220117_jpt_floating_kth_screwdriver_kth_spanner",
    ]

    for d, out_file in zip(dirs, out_files):
        for i in range(3):
            tfrecord_from_h5(
                d + "/000" + str(i),
                "data_pose_event.h5",
                "./tfrecord",
                out_file + "_000" + str(i) + ".tfrecord",
            )


def tfrecord_parse(serialized_example):
    feature = {
	'shape': tf.io.FixedLenFeature((), tf.string),
	'events': tf.io.FixedLenFeature((), tf.string),
	'xy': tf.io.FixedLenFeature((), tf.string),
    }
    example = tf.io.parse_single_example(serialized_example, feature)

    shape = tf.io.parse_tensor(example['shape'], out_type=tf.int32)
    events = tf.io.parse_tensor(example['events'], out_type=tf.uint8)
    events = tf.reshape(events, shape)

    n_class = 3
    xy = tf.io.parse_tensor(example['xy'], out_type=tf.float32)
    xy = tf.reshape(xy, (2, n_class))

    return {'inputs_events': events, 'inputs_xy': xy}


def tfrecord_dataset(path, batch_size=None, shuffle_data=False, shuffle_files=False, shuffle_files_seed=None):
    files_ds = tf.data.Dataset.list_files(path + '/*.tfrecord', shuffle=shuffle_files, seed=shuffle_files_seed)

    files_ds = files_ds.cache()

    if shuffle_files:
        files_ds = files_ds.shuffle(len(files_ds))

    data_ds = files_ds.interleave(tf.data.TFRecordDataset, cycle_length=8, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle_data:
        data_ds = data_ds.shuffle(buffer_size=8192)

    # n_data_ds = 0
    # for x in data_ds:
    #     n_data_ds += 1
    # print('number of samples:', n_data_ds)
    # exit(0)

    data_ds = data_ds.map(tfrecord_parse, num_parallel_calls=tf.data.AUTOTUNE)

    if not batch_size is None:
        data_ds = data_ds.batch(batch_size)

    data_ds = data_ds.prefetch(tf.data.AUTOTUNE)

    return data_ds


def tfrecord_show(path):
    data_ds = tfrecord_dataset(path)
    data_iter = iter(data_ds)
    image = np.zeros((260, 346, 3), dtype='uint8')

    done = False
    while not done:

        try:
            x = next(data_iter)
        except StopIteration:
            cv2.destroyAllWindows()
            done = True
            continue

        events = x['inputs_events'].numpy()
        xy = x['outputs_xy'].numpy()
        xy_int = np.rint(xy).astype('int32')

        image.fill(0)
        image[:, :, :2] = events
        image[image > 0] = 255
        cv2.circle(image, (xy_int[0, 0], xy_int[1, 0]), 3, (0, 0, 255), -1)
        cv2.circle(image, (xy_int[0, 1], xy_int[1, 1]), 3, (0, 0, 255), -1)
        cv2.circle(image, (xy_int[0, 2], xy_int[1, 2]), 3, (0, 0, 255), -1)

        # show event image
        cv2.imshow(f'camera image', image)
        k = cv2.waitKey(10)
        if k == ord('q'):
            cv2.destroyAllWindows()
            done = True

    return


if __name__ == '__main__':
    tfrecord_show('./data/test')
