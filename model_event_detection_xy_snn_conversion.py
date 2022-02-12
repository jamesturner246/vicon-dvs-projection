import cv2
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np

from tfrecord import *


def parse_event_tfrecord(serialized_example):
    feature = {
        'shape': tf.io.FixedLenFeature((), tf.string),
        'frame': tf.io.FixedLenFeature((), tf.string),
        'object_pose': tf.io.FixedLenFeature((), tf.string),
        'object_class': tf.io.FixedLenFeature((), tf.string),
    }
    example = tf.io.parse_single_example(serialized_example, feature)

    n_class = 3

    shape = tf.io.parse_tensor(example['shape'], out_type=tf.int32)
    frame = tf.io.parse_tensor(example['frame'], out_type=tf.uint8)
    frame = tf.reshape(frame, shape)
    #frame = tf.image.random_flip_left_right(frame) # NOTE: can't flip for pose training without modifying labels

    object_detection = tf.io.parse_tensor(example['object_class'], out_type=tf.float32)
    object_detection = tf.math.ceil(object_detection)
    object_detection = tf.reshape(object_detection, (n_class,))
    object_pose = tf.io.parse_tensor(example['object_pose'], out_type=tf.float32)
    object_pose = tf.reshape(object_pose, (n_class, 6))[:, :3] # take translation, since rotation is missing

    # Normalise pose to within [-1, 1]
    # NOTE: object valid (x, y) coordinates (pixels) are initially in ([0, 640], [0, 480])
    # NOTE: object depth (metres) is initially in [0, 3.5]
    object_pose_mid = tf.constant([320.0, 240.0, 1.75])
    object_pose = (object_pose - object_pose_mid) / object_pose_mid

    return {'inputs_frame': frame, 'inputs_detection': object_detection, 'inputs_pose': object_pose}








def parse_tfrecord(serialized_example):
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
    xy = tf.reshape(xy, (n_class, 2))

    return {'inputs_events': events, 'inputs_xy': xy}








if __name__ == '__main__':

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    model_name = 'event_detection_xy'

    train_path = './tfrecord_train'
    val_path = './tfrecord_validate'

    n_class = 3
    use_bias = False

    batch_size = 64
    epochs = 100
    #epochs = 40
    initial_epoch = 0
    momentum = 0.9
    dropout_rate = 0.25

    record_tensorboard = False

    filters_1 = 16
    filters_2 = 32
    filters_3 = 64
    filters_detection = 128
    filters_pose = 128

    # Learning rate scheduler
    learning_rate =  0.0005
    lr_time = [25, 50, 75]
    #lr_time = [10, 20, 30]
    def lr_schedule(epoch):
        if epoch < lr_time[0]:
            return learning_rate
        elif epoch < lr_time[1]:
            return learning_rate / 10
        elif epoch < lr_time[2]:
            return learning_rate / 100
        else:
            return learning_rate / 1000

    # Regularizer
    weight_decay = 0.0001
    regularizer = tf.keras.regularizers.L2(weight_decay)

    # Datasets
    train_batched_ds = tfrecord_dataset(
        train_path, batch_size=batch_size, shuffle_data=True, shuffle_files=True, shuffle_files_seed=111)
    val_batched_ds = tfrecord_dataset(val_path, batch_size=batch_size)







    class XYEndpoint(layers.Layer):

        def __init__(self, name='xy_endpoint'):
            super(XYEndpoint, self).__init__(name=name)

        def call(self, xy_frame, xy_targets, loss_weight=1.0):

            # xy_frame should be [batch_i, row_i, col_i, prop_i]
            assert(len(xy_frame.get_shape()) == 4)

            # xy_targets should be [batch_i, (x, y), prop_i]
            assert(len(xy_targets.get_shape()) == 3)

            # convert prediction frame to predicted xy coordinates
            frame_flat = tf.reshape(xy_frame, (tf.shape(xy_frame)[0], -1, tf.shape(xy_frame)[3]))
            amax_frame_flat = tf.cast(tf.argmax(frame_flat, axis=1), tf.int32)
            amax_frame_row = amax_frame_flat // tf.shape(xy_frame)[2]
            amax_frame_col = amax_frame_flat % tf.shape(xy_frame)[2]
            xy_predictions = tf.cast(tf.stack((amax_frame_row, amax_frame_col), axis=1), tf.float32)

            # Loss: MSE with missing objects masked
            visible = tf.math.is_finite(xy_targets)[:, 0, :] & tf.math.is_finite(xy_targets)[:, 1, :]
            loss = tf.reduce_mean(tf.square(xy_predictions - xy_targets), 1) # MSE (for each prop)
            loss_mask = tf.where(~visible, tf.zeros_like(loss), loss)
            loss_mean = tf.reduce_mean(loss_mask)
            loss_mean_weighted = loss_mean * loss_weight
            self.add_loss(loss_mean_weighted)

            # Accuracy: MSE with missing objects masked
            self.add_metric(loss_mean, name='xy_mse')

            return xy_frame


    class DetectionEndpoint(layers.Layer):

        def __init__(self, name='detection_endpoint'):
            super(DetectionEndpoint, self).__init__(name=name)

        def call(self, detection, xy_targets, loss_weight=1.0):

            # detection should be [batch_i, (invisible_1, visible_1, invisible_2, visible_2, ... )]
            assert(len(detection.get_shape()) == 2)

            # xy_targets should be [batch_i, (x, y), prop_i]
            assert(len(xy_targets.get_shape()) == 3)

            # convert one-hot detections to sparse
            detection_reshaped = tf.reshape(detection, (tf.shape(detection)[0], -1, 2))
            detection_reshaped = tf.transpose(detection_reshaped, perm=(0, 2, 1))
            detection_predictions = tf.argmax(detection_reshaped, axis=1)
            detection_predictions = tf.cast(detection_predictions, tf.float32)

            # convert xy_targets to detection targets
            detection_targets = tf.math.is_finite(xy_targets)[:, 0, :] & tf.math.is_finite(xy_targets)[:, 1, :]
            detection_targets = tf.cast(detection_targets, tf.float32)

            # Loss: binary crossentropy
            loss = tf.keras.losses.binary_crossentropy(detection_targets, detection_predictions)
            loss_mean = tf.reduce_mean(loss)
            loss_mean_weighted = loss_mean * loss_weight
            self.add_loss(loss_mean_weighted)

            # Accuracy: binary accuracy
            accuracy = tf.keras.metrics.binary_accuracy(detection_targets, detection_predictions)
            accuracy_mean = tf.reduce_mean(accuracy)
            self.add_metric(accuracy_mean, name='detection_binary_accuracy')

            return detection












    if True:
    #with tf.device('/CPU:0'):
        # Create TF model
        inputs_events = layers.Input(shape=(260, 346, 2), name='inputs_events')
        inputs_xy = layers.Input(shape=(2, n_class), name='inputs_xy')

        # Block 1 (Conv)
        conv_1_1 = layers.Conv2D(
            filters_1, 7, padding='same', use_bias=use_bias, activation='relu',
            kernel_regularizer=regularizer, name='conv_1_1')(inputs_events)
        dropout_1_1 = layers.Dropout(
            dropout_rate, name='dropout_1_1')(conv_1_1)
        conv_1_2 = layers.Conv2D(
            filters_1, 7, padding='same', use_bias=use_bias, activation='relu',
            kernel_regularizer=regularizer, name='conv_1_2')(dropout_1_1)
        pool_1 = layers.AveragePooling2D(
            2, name='pool_1')(conv_1_2)

        # Block 2 (Conv)
        conv_2_1 = layers.Conv2D(
            filters_2, 5, padding='same', use_bias=use_bias, activation='relu',
            kernel_regularizer=regularizer, name='conv_2_1')(pool_1)
        dropout_2_1 = layers.Dropout(
            dropout_rate, name='dropout_2_1')(conv_2_1)
        conv_2_2 = layers.Conv2D(
            filters_2, 5, padding='same', use_bias=use_bias, activation='relu',
            kernel_regularizer=regularizer, name='conv_2_2')(dropout_2_1)
        pool_2 = layers.AveragePooling2D(
            2, name='pool_2')(conv_2_2)

        # Block 3 (Conv)
        conv_3_1 = layers.Conv2D(
            filters_3, 3, padding='same', use_bias=use_bias, activation='relu',
            kernel_regularizer=regularizer, name='conv_3_1')(pool_2)
        dropout_3_1 = layers.Dropout(
            dropout_rate, name='dropout_3_1')(conv_3_1)
        conv_3_2 = layers.Conv2D(
            filters_3, 3, padding='same', use_bias=use_bias, activation='relu',
            kernel_regularizer=regularizer, name='conv_3_2')(dropout_3_1)
        pool_3 = layers.AveragePooling2D(
            2, name='pool_3')(conv_3_2)
        dropout_3_2 = layers.Dropout(
            dropout_rate, name='dropout_3_2')(conv_3_2)
        conv_3_3 = layers.Conv2D(
            filters_3, 3, padding='same', use_bias=use_bias, activation='relu',
            kernel_regularizer=regularizer, name='conv_3_3')(dropout_3_2)
        pool_3 = layers.AveragePooling2D(
            2, name='pool_3')(conv_3_3)

        # Dense (detection)
        flatten_detection = layers.Flatten(
            name='flatten_detection')(pool_3)
        dense_detection_1 = layers.Dense(
            filters_detection, use_bias=use_bias, activation='relu',
            kernel_regularizer=regularizer, name='dense_detection_1')(flatten_detection)
        dropout_detection_1 = layers.Dropout(
            dropout_rate, name='dropout_detection_1')(dense_detection_1)
        dense_detection_2 = layers.Dense(
            2 * n_class, use_bias=use_bias, activation='relu',
            kernel_regularizer=regularizer, name='detection_out')(dropout_detection_1)
        detection_endpoint = DetectionEndpoint(
            name='detection_endpoint')(dense_detection_2, inputs_xy, 1.0)



        # Dense (xy)
        conv_transpose_xy = layers.Conv2DTranspose(
            3, 8, strides=4, padding='valid', use_bias=False, activation='relu',
            kernel_regularizer=regularizer, name='conv_transpose_xy')(conv_3_3) # TODO: BEFORE OR AFTER POOLING
        xy_endpoint = XYEndpoint(name='xy_endpoint')(conv_transpose_xy, inputs_xy, 1.0)




        # TODO: FIXME: XY MSE LOSS IS NAN




        # Model definition
        model = tf.keras.models.Model(
            inputs=[
                inputs_events,
                inputs_xy,
            ],
            outputs=[
                detection_endpoint,
                xy_endpoint,
            ],
            name=model_name)
        
        model.summary()

        # Callbacks
        callbacks = [tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=True)]
        if record_tensorboard:
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1))
        
        # Compile model
        #optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer)

        # # Load weights
        # #model.load_weights(model_name + '_weights.h5')
        # #model.load_weights(model_name + '_DEPTH_weights.h5')
        # for layer in model.layers:
        #     weights = layer.get_weights()
        #     if weights:
        #         new_weights = [np.load('model_event_detection_pose_depth_weights/' + layer.name + '_weights.npy')]
        #         layer.set_weights(new_weights)

        # Fit model
        model.fit(train_batched_ds, validation_data=val_batched_ds, epochs=epochs, initial_epoch=initial_epoch, callbacks=callbacks)

        # Evaluate model
        model.evaluate(val_batched_ds)

        # Save weights
        #model.save_weights(model_name + '_weights.h5')
        #model.save_weights(model_name + '_DEPTH_weights.h5')
        for layer in model.layers:
            weights = layer.get_weights()
            if weights:
                np.save(layer.name + '_weights', weights[0])


        exit(0)







        # Model to convert
        model_to_convert = tf.keras.models.Model(
            inputs=[
                inputs_frame,
            ],
            outputs=[
                detection_out,
            ],
            name=model_name)
        model_to_convert.get_layer('detection_out').activation = tf.keras.activations.relu




    # # COMPARE MODEL AND MODEL_TO_CONVERT
    # for l2 in model_to_convert.layers:
    #     l1 = model.get_layer(l2.name)
    #     print(l1.name, l2.name)
    #     w1 = l1.get_weights()
    #     w2 = l2.get_weights()
    #     if w1 and w2:
    #         print(np.equal(w1, w2).all())
    #     print()




    # MLG CONVERSION

    #mlg_batch_size = 60
    #mlg_batch_size = 5
    mlg_batch_size = 1

    #mlg_eval_time = 100
    #mlg_eval_time = 50
    mlg_eval_time = 40

    def get_frame_detection(x):
        return (x['inputs_frame'], x['inputs_detection'])

    n_val_ds = 5888 # TODO: FIXME: manually adjust
    val_mlg_ds = val_ds.map(get_frame_detection, num_parallel_calls=tf.data.AUTOTUNE)
    val_mlg_ds = val_mlg_ds.batch(mlg_batch_size)
    val_mlg_ds = val_mlg_ds.prefetch(tf.data.AUTOTUNE)
    val_mlg_ds = val_mlg_ds.as_numpy_iterator()






    import ml_genn

    # norm data
    def get_frame(x):
        return (x['inputs_frame'])
    norm_ds = train_ds.map(get_frame, num_parallel_calls=tf.data.AUTOTUNE)
    norm_ds = norm_ds.take(256)
    norm_ds = norm_ds.as_numpy_iterator()
    norm_ds = np.array([x for x in norm_ds])

    #converter = ml_genn.converters.Simple(signed_input=True, input_type='poisson')
    converter = ml_genn.converters.SpikeNorm([norm_ds], mlg_eval_time, signed_input=True, input_type='poisson')

    mlg_model = ml_genn.Model.convert_tf_model(
        model_to_convert, converter=converter, connectivity_type='toeplitz',
        dt=1.0, batch_size=mlg_batch_size, rng_seed=0)





    # DATA-NORM

    # norm_layer_names = [
    #     'conv_1_1',
    #     'conv_1_2',
    #     'conv_2_1',
    #     'conv_2_2',
    #     'conv_3_1',
    #     'conv_3_2',
    #     'conv_3_3',
    #     'dense_detection_1',
    #     #'dense_detection_out',
    # ]

    # mlg_i = 1
    # previous_factor = 1
    # for layer_name in norm_layer_names:
    #     tf_layer = model_to_convert.get_layer(layer_name)
    #     mlg_layer = mlg_model.layers[mlg_i]
    #     print('mlg', mlg_layer.name, 'tf', tf_layer.name)
    #     assert(mlg_layer.name == tf_layer.name)

    #     tf_layer_fn = tf.keras.backend.function(model_to_convert.inputs, tf_layer.output)
    #     max_activation = np.max(tf_layer_fn(norm_ds))
    #     max_weight = np.max(tf_layer.get_weights()[0])
    #     scale_factor = np.maximum(max_activation, max_weight)
    #     threshold = scale_factor / previous_factor
    #     previous_factor = scale_factor

    #     mlg_layer.neurons.set_threshold(threshold)

    #     mlg_i += 1


    print()
    print()
    print()
    print()
    print()






    acc, spk_i, spk_t = mlg_model.evaluate_iterator(val_mlg_ds, n_val_ds, mlg_eval_time)
    print('Accuracy of mlGeNN model: {}%'.format(acc[0]))



    exit(0)









    np.set_printoptions(precision=24, floatmode='fixed', linewidth=120, suppress=True)







    # import matplotlib.pyplot as plt

    # layer_name = 'conv_1_1'
    # #layer_name = 'conv_1_2'
    # #layer_name = 'conv_2_1'
    # #layer_name = 'conv_2_2'

    # ic_i = [0, 8]
    # #oc_i = [0, 8]
    # oc_i = [8, 16]

    # layer = model.get_layer(layer_name)
    # weights = layer.get_weights()[0]
    # ic_i[1] = min(ic_i[1], weights.shape[2])
    # oc_i[1] = min(oc_i[1], weights.shape[3])
    # weights = weights[:, :, ic_i[0]:ic_i[1], oc_i[0]:oc_i[1]]
    # print(weights.shape)
    # print(f'{layer_name} weights:  min {np.min(weights)}  max {np.max(weights)}  mean {np.mean(weights)}  weights std {np.std(weights)}')
    # print()

    # #w_clip_lo = np.min(weights)
    # #w_clip_hi = np.max(weights)
    # w_clip_lo = np.mean(weights) - 2 * np.std(weights)
    # w_clip_hi = np.mean(weights) + 2 * np.std(weights)

    # fig, axes = plt.subplots(ic_i[1] - ic_i[0], oc_i[1] - oc_i[0])
    # for ic in range(ic_i[1] - ic_i[0]):
    #     for oc in range(oc_i[1] - oc_i[0]):
    #         ax = axes[ic, oc]
    #         w = weights[:, :, ic, oc]
    #         print(f'ic {ic_i[0] + ic}  oc {oc_i[0] + oc}  w mean {np.mean(w)}  w std {np.std(w)}')

    #         im = ax.imshow(w, vmin=w_clip_lo, vmax=w_clip_hi)
    #         ax.set_xticks([])
    #         ax.set_yticks([])

    # axes[(ic_i[1] - ic_i[0]) // 2, 0].set_ylabel('input channel')
    # axes[-1, (oc_i[1] - oc_i[0]) // 2].set_xlabel('output channel')

    # fig.suptitle(f'{layer_name} kernels: in channels {ic_i[0]}-{ic_i[1]-1}, out channels {oc_i[0]}-{oc_i[1]-1}')
    # fig.tight_layout()
    # fig.colorbar(im, ax=axes.ravel().tolist())

    # plt.show()







    prop_names = ['hammer', 'wrench', 'screwdriver']
    pose_mid = np.array([320.0, 240.0, 1.75])

    #ds = train_ds
    ds = val_ds
    for x in ds.batch(1):
        prediction = model(x)

        frame = x['inputs_frame'].numpy()[0]
        print(f'FRAME:  dtype: {frame.dtype}  shape: {frame.shape}')
        print()

        detection_target = x['inputs_detection'].numpy()[0]
        print(f'DETECTION TARGET:  dtype: {detection_target.dtype}  shape: {detection_target.shape}')
        print(detection_target)
        detection_prediction = prediction[0].numpy()[0]
        print(f'DETECTION PREDICTION:  dtype: {detection_prediction.dtype}  shape: {detection_prediction.shape}')
        print(detection_prediction)
        print()

        pose_target = x['inputs_pose'].numpy()[0]
        print(f'POSE TARGET:  dtype: {pose_target.dtype}  shape: {pose_target.shape}')
        print(pose_target)
        pose_prediction = prediction[1].numpy()[0]
        print(f'POSE PREDICTION:  dtype: {pose_prediction.dtype}  shape: {pose_prediction.shape}')
        print(pose_prediction)
        print()

        frame_pose = frame.copy() + 0.5
        for prop, dp, pp, dt, pt in zip(prop_names, detection_prediction, pose_prediction, detection_target, pose_target):

            if dp > 0.5:
                p = (pp * pose_mid) + pose_mid
                p_int = np.rint(p).astype('int32')
                cv2.circle(frame_pose, (p_int[0], p_int[1]), 3, (0, 255, 0), -1)
                print(f'  PREDICTION:  prop: {prop}  coordinates: ({p[0]}, {p[1]})  depth: {p[2]} metres')

            if dt > 0.5:
                p = (pt * pose_mid) + pose_mid
                p_int = np.rint(p).astype('int32')
                cv2.circle(frame_pose, (p_int[0], p_int[1]), 3, (0, 0, 255), -1)
                print(f'  TARGET:      prop: {prop}  coordinates: ({p[0]}, {p[1]})  depth: {p[2]} metres')

        print()

        cv2.imshow(f'frame', frame_pose)
        k = cv2.waitKey(0)
        if k == ord('q'):
            break

    cv2.destroyWindow(f'frame')
