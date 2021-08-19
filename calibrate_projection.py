#!/bin/env python3

import time
import cv2
import dv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from tqdm import tqdm


def identify_wand_markers(dv):
    Np = dv.shape[0]

    # calculate the abs(cos(angle)) for all pairs of difference vectors
    # those pairs who lie on the same acis will have (close to) 1
    cand = []
    ang = []
    for p1 in range(Np):
        for p2 in range(p1 + 1, Np):
            d1 = dv[p1, :] - dv[p2, :]
            for p3 in range(p2 + 1, Np):
                d2 = dv[p1, :] - dv[p3, :]
                ctheta = np.abs(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2)))
                cand.append([p1, p2, p3])
                ang.append(ctheta)

    # select the two largest ones
    idx = np.argsort(ang)
    ax = [cand[idx[-2]], cand[idx[-1]]]

    # count how often each LED appears in the two axes
    # the one that appears twice is the "centre point"
    cnt = np.zeros(Np)
    for i in range(2):
        for j in range(3):
            cnt[ax[i][j]] += 1
            if cnt[ax[i][j]] == 2:
                c = ax[i][j]

    # c is the index of the centre point
    #print(c)

    # calculate the cos(angle) between the two segments on each axis
    # the one where the segments are in the same direction (cos approx 1)
    # is the long one
    d = np.zeros((2, 2))
    k = 0
    for j in range(3):
        if ax[0][j] != c:
            d[k, :] = dv[ax[0][j], :] - dv[c, :]
            k += 1
    ctheta = np.dot(d[0], d[1]) / (np.linalg.norm(d[0]) * np.linalg.norm(d[1]))
    if ctheta > 0:
        long = 0
    else:
        long = 1

    #print("long axis is: {}".format(ax[long]))
    #print("short axis is: {}".format(ax[1 - long]))

    # Assign the LED IDs within each identified axis depending on whether they are
    # closer or farther away from the centre
    top_centre = int(c)
    ln = np.zeros(2)
    id = np.zeros(2)
    k = 0
    for i in ax[1 - long]:
        if i != c:
            ln[k] = np.linalg.norm(dv[i, :] - dv[c, :])
            id[k] = i
            k += 1

    if ln[0] < ln[1]:
        top_right = int(id[0])
        top_left = int(id[1])
    else:
        top_right = int(id[1])
        top_left = int(id[0])

    k = 0
    for i in ax[long]:
        if i != c:
            ln[k] = np.linalg.norm(dv[i, :] - dv[c, :])
            id[k] = i
            k += 1

    if ln[0] < ln[1]:
        middle = int(id[0])
        bottom = int(id[1])
    else:
        middle = int(id[1])
        bottom = int(id[0])

    dv = dv[(top_left, top_centre, top_right, middle, bottom), :]
    return dv


def get_vicon_coordinates(i_epoch, address, port, prop_name, marker_names,
                          n_frame=100, t_frame=0.001,
                          reuse=False, debug=False):

    print('Vicon coordinates')

    marker_count = len(marker_names)
    marker_coordinates_file = f'./calibration/vicon_coordinates_{i_epoch}.npy'
    mean_coordinates = np.empty((marker_count, 3), dtype='float64')
    median_coordinates = np.empty((marker_count, 3), dtype='float64')

    if reuse:
        marker_coordinates = np.load(marker_coordinates_file)

    else:
        marker_coordinates = np.empty((n_frame, marker_count, 3), dtype='float64')

        from vicon_dssdk import ViconDataStream

        client = ViconDataStream.Client()
        client.Connect(f'{address}:{port}')
        client.EnableMarkerData()

        progress = tqdm(total=n_frame, leave=False)
        i_frame = 0

        while i_frame < n_frame:
            if not client.GetFrame():
                continue

            try:
                prop_quality = client.GetObjectQuality(prop_name)
            except ViconDataStream.DataStreamException:
                prop_quality = None

            if prop_quality is not None:
                actual_marker_names = client.GetMarkerNames(prop_name)
                actual_marker_count = len(actual_marker_names)

                for i_marker in range(actual_marker_count):
                    marker_name = marker_names[i_marker]
                    actual_marker_name = actual_marker_names[i_marker][0]
                    assert(marker_name == actual_marker_name)

                    translation = client.GetMarkerGlobalTranslation(prop_name, actual_marker_name)[0]
                    marker_coordinates[i_frame, i_marker, :] = translation

                i_frame += 1
                progress.update(1)

            time.sleep(t_frame)

        progress.close()
        
        client.Disconnect()

    if not reuse:
        np.save(marker_coordinates_file, marker_coordinates)

    # compute median Vicon coordinates
    median_coordinates[:, :] = np.median(marker_coordinates, 0)
    for i in range(marker_count):
        print(f'name: {marker_names[i]:<15s} coordinates: {median_coordinates[i]}')

    return median_coordinates


def get_dv_wand_coordinates(i_epoch, address, event_port, frame_port, prop_name, marker_names,
                            camera_matrix, distortion_coefficients,
                            n_event=10000, frame_shape=(260, 346, 3),
                            reuse=False, debug=False):

    print('DV coordinates')

    n_marker = len(marker_names)
    coordinates_file = f'./calibration/dv_coordinates_{i_epoch}.npy'

    if reuse:
        coordinates = np.load(coordinates_file)

    else:
        coordinates = np.empty((len(marker_names), 2), dtype='float64')

        if debug:
            coordinates_int = np.empty((len(marker_names), 2), dtype='int32')
            coordinates_image = np.empty(frame_shape[:2], dtype='uint8')

        erode_dv_kernel = np.ones((3, 3), 'uint8')
        dilate_dv_kernel = np.ones((10, 10), 'uint8')

        event_t = np.empty((n_event), dtype='uint64')
        event_xy = np.empty((n_event, 2), dtype='int32')
        event_image = np.zeros(frame_shape[:2], dtype='int64')

        with dv.NetworkEventInput(address=address, port=event_port) as f:
            progress = tqdm(total=n_event, leave=False)
            i_event = 0

            for event in f:
                if i_event == n_event:
                    break

                # undistort event
                event_distorted = np.array([event.x, event.y], dtype='float64')
                event_undistorted = cv2.undistortPoints(
                    event_distorted, camera_matrix, distortion_coefficients, None, camera_matrix)[0, 0]
                event_undistorted_int = np.rint(event_undistorted).astype('int32')

                try:
                    event_image[event_undistorted_int[1], event_undistorted_int[0]] += 1.0
                except IndexError:
                    continue

                event_t[i_event] = event.timestamp
                event_xy[i_event, :] = event_undistorted

                i_event += 1
                progress.update(1)

            progress.close()

        if debug:
            # plot unmasked event image
            cv2.imshow('unmasked event image', (event_image / event_image.max()).astype('float32'))
            cv2.waitKey()

        # event mask threshold
        event_image_mask = (event_image > 0).astype('float32')

        if debug:
            # plot event image mask (threshold)
            cv2.imshow('event image mask (threshold)', event_image_mask)
            cv2.waitKey()

        # event mask erode and dilate
        event_image_mask = cv2.erode(event_image_mask, erode_dv_kernel)
        event_image_mask = cv2.dilate(event_image_mask, dilate_dv_kernel)
        event_image[~(event_image_mask.astype('bool'))] = 0
        event_xy_masked = np.array(
            [xy for xy in event_xy if event_image_mask[xy[1], xy[0]] > 0], dtype='int32')

        if debug:
            # plot event image mask (erode and dilate)
            cv2.imshow('event image mask (erode and dilate)', event_image_mask)
            cv2.waitKey()

            # plot masked event image
            cv2.imshow('masked event image', (event_image / event_image.max()).astype('float32'))
            cv2.waitKey()

        # k-means of masked events
        k_means = KMeans(
            n_clusters=n_marker,
            init='k-means++',
            n_init=100,
            max_iter=300,
            #tol=1e-4,
            #random_state=0,
        ).fit(event_xy_masked)
        coordinates[:, :] = k_means.cluster_centers_
        coordinates = identify_wand_markers(coordinates)

        if debug:
            coordinates_int[:, :] = np.rint(coordinates).astype('int32')
            coordinates_image.fill(0)
            coordinates_image[coordinates_int[:, 1], coordinates_int[:, 0]] = 255

            # plot DV coordinates
            cv2.imshow('event coordinates', coordinates_image)
            cv2.waitKey()

            # close debug plots
            cv2.destroyAllWindows()

        with dv.NetworkFrameInput(address=address, port=frame_port) as f:
            frame = next(f)

        frame_image = cv2.undistort(
                frame.image, camera_matrix, distortion_coefficients, None, camera_matrix)

        # plot markers
        fig, ax = plt.subplots()
        ax.imshow(frame_image[:, :, (2, 1, 0)])
        ax.plot(coordinates[:, 0], coordinates[:, 1], '*b')
        ax.text(coordinates[0, 0], coordinates[0, 1], 'top_left', color='blue')
        ax.text(coordinates[1, 0], coordinates[1, 1], 'top_centre', color='blue')
        ax.text(coordinates[2, 0], coordinates[2, 1], 'top_right', color='blue')
        ax.text(coordinates[3, 0], coordinates[3, 1], 'middle', color='blue')
        ax.text(coordinates[4, 0], coordinates[4, 1], 'bottom', color='blue')
        ax.set_ylim([0, frame_shape[0]])
        ax.set_xlim([0, frame_shape[1]])
        ax.invert_yaxis()
        plt.show()

    if not reuse:
        np.save(coordinates_file, coordinates)

    # print DV coordinates
    for i in range(n_marker):
        print(f'name: {marker_names[i]:<15s} coordinates: {coordinates[i]}')

    return coordinates


def calibrate():

    n_epoch = 20

    debug = False
    test = True

    reuse = True
    reuse_vicon = reuse
    reuse_dv = reuse

    method = 2

    prop_name = 'jt_wand'
    marker_names = [
        'top_left',
        'top_centre',
        'top_right',
        'middle',
        'bottom',
    ]
    n_marker = len(marker_names)

    # DV
    dv_address, dv_event_port, dv_frame_port = '127.0.0.1', 36000, 36001
    dv_frame_shape = (260, 346, 3)
    dv_n_event = 10000
    dv_wand_coordinates = np.empty((n_epoch, n_marker, 2), dtype='float64')
    dv_camera_matrix = np.load('./calibration/camera_matrix.npy')
    dv_distortion_coefficients = np.load('./calibration/camera_distortion_coefficients.npy')

    # Vicon
    vicon_address, vicon_port = '127.0.0.1', 801
    vicon_n_frame = 100
    vicon_t_frame = 0.001
    vicon_wand_coordinates = np.empty((n_epoch, n_marker, 3), dtype='float64')

    i_epoch = 0
    while i_epoch < n_epoch:
        if not (reuse_vicon and reuse_dv):
            print(f'calibration epoch {i_epoch}')
            input('relocate prop and press enter...')

        vicon_wand_coordinates[i_epoch, :, :] = get_vicon_coordinates(
            i_epoch, vicon_address, vicon_port, prop_name, marker_names,
            n_frame=vicon_n_frame, t_frame=vicon_t_frame,
            reuse=reuse_vicon, debug=debug)

        dv_wand_coordinates[i_epoch, :, :] = get_dv_wand_coordinates(
            i_epoch, dv_address, dv_event_port, dv_frame_port, prop_name, marker_names,
            dv_camera_matrix, dv_distortion_coefficients,
            n_event=dv_n_event, frame_shape=dv_frame_shape,
            reuse=reuse_dv, debug=debug)

        if not (reuse_vicon and reuse_dv):
            accept = ''
            while accept != 'y' and accept != 'n':
                accept = input('accept epoch? (y/n): ')
                if accept == 'y':
                    i_epoch += 1
        else:
            i_epoch += 1



    #########################################################################

    if method == 1:

        def vicon_to_dv(m, v):
            z = vicon_to_camera_centric(m, v)
            z *= (1.0 / z[2]) * 4  # focal length 4 mm
            z = z[:2]
            z /= 1.8e-2            # 18 micrometer/pixel = 1.8e-2 mm/pixel
            z += [173, 130]        # add the origin offset from image centre to top left corner explicitly
            return z

        def vicon_to_camera_centric(m, v):
            M = np.reshape(m[:9], (3, 3))
            z = np.dot(M, v)       # apply the rotation to get into the camera orientation frame
            z += m[9:12] * 10      # add the translation (using cm for a better scale of fitting)
            return z

        def err_fun(m, vicon_p, dv_p):
            assert dv_p.shape[0] == vicon_p.shape[0]

            error = 0.0
            for v, d in zip(vicon_p, dv_p):
                output = vicon_to_dv(m, v)
                difference = output - d
                error += np.dot(difference, difference)

            return np.sqrt(error)

        # Vicon to DV transformation
        m_file = './calibration/bootstrap_dv_space_transform.npy'
        m = np.load(m_file)
        x = np.vstack(vicon_wand_coordinates)
        y = np.vstack(dv_wand_coordinates)

        print('Original guess has error: ', err_fun(m, x, y))

        method = 'nelder-mead'
        options = {'disp': True, 'maxiter': 50000, 'maxfev': 100000, 'xatol': 1e-10, 'fatol': 1e-10}
        result = minimize(err_fun, m, args=(x, y), method=method, options=options)

        m = result['x']
        print('DV space transform error: ', err_fun(m, x, y))

        m_file = './calibration/dv_space_transform.npy'
        np.save(m_file, m)

        print('DV space coefficients')
        print(np.reshape(m[:9], (3, 3)))
        print('DV space constants')
        print(m[9:])
        print()



    #########################################################################

    elif method == 2:

        def vicon_to_dv(m, v):
            z = vicon_to_camera_centric(m, v)
            z *= (1.0 / z[2]) * 4 * m[6]  # focal length 4 mm
            z = z[:2]
            z /= 1.8e-2                   # 18 micrometer/pixel = 1.8e-2 mm/pixel
            z[0] *= m[7]                  # allow for some rescaling of x due to the camera  undistortion method
            z += [173, 130]               # add the origin offset from image centre to top left corner explicitly
            return z

        def vicon_to_camera_centric(m, v):
            M = the_matrix(m)
            z = np.dot(M, v)              # apply the rotation to get into the camera orientation frame
            z += m[3:6] * 10              # add the translation (using cm for a better scale of fitting)
            return z

        def the_matrix(m):
            M = np.array([
                [np.cos(m[0]), -np.sin(m[0]), 0],
                [np.sin(m[0]), np.cos(m[0]), 0],
                [0, 0, 1]])
            M = np.dot(np.array([
                [1, 0, 0],
                [0, np.cos(m[1]), -np.sin(m[1])],
                [0, np.sin(m[1]), np.cos(m[1])]]), M)
            M = np.dot(np.array([
                [np.cos(m[2]), -np.sin(m[2]), 0],
                [np.sin(m[2]), np.cos(m[2]), 0],
                [0, 0, 1]]), M)
            return M

        def err_fun(m, vicon_p, dv_p):
            # the meaning of m is as follows:
            # 0-2 Euler angles of transform into camera oriented space
            # 3-5 translation of vector to be relative to camera origin (chosen as pinhole position)
            # 6 scale factor for stretch in x-direction due to camera calibration/undistortion

            assert dv_p.shape[0] == vicon_p.shape[0]

            error = 0.0
            for v, d in zip(vicon_p, dv_p):
                output = vicon_to_dv(m, v)
                difference = output - d
                error += np.dot(difference, difference)

            return np.sqrt(error)


        # Vicon to DV transformation
        m = np.empty(8)
        m[:3] = np.array([3.14, 1.6, 0.0])  # initial guess for angles - turn x to -x and rotate around x to get z pointing in -z direction
        m[3:6] = np.array([22, 93, 231])  # initial guess for translation from pinhole to vicon origin
        m[6:8]= [ 1.0, 1.0 ]  # initial guess for the deviation of focal length and x-stretching from camera undistortion
        x = np.vstack(vicon_wand_coordinates)
        y = np.vstack(dv_wand_coordinates)

        print('Original guess has error: ', err_fun(m, x, y))

        method = 'nelder-mead'
        options = {'disp': True, 'maxiter': 50000, 'maxfev': 100000, 'xatol': 1e-10, 'fatol': 1e-10}
        result = minimize(err_fun, m, args=(x, y), method=method, options=options)

        m = result['x']
        print('DV space transform error: ', err_fun(m, x, y))

        m_file = './calibration/dv_space_transform.npy'
        np.save(m_file, m)

        print("Euler angles: {}".format(m[:3]))
        print("Translation: {}".format(m[3:6]))
        print("focal length and x rescales: {}".format(m[6:]))
        print("The matrix: {}".format(the_matrix(m)))
        print()



    #########################################################################

    else:
        raise RuntimeError('invalid method')


    plt.figure()
    for i in range(y.shape[0]//5):
        plt.scatter(y[5*i:5*(i+1),0],y[5*i:5*(i+1),1])
    z= np.zeros(y.shape)
    for i in range(x.shape[0]):
        z[i,:]= vicon_to_dv(m, x[i,:])
    for i in range(z.shape[0]//5):
        plt.scatter(z[5*i:5*(i+1),0],z[5*i:5*(i+1),1],marker='x')
    ax= np.array([[[ 0, 0, 0], [1000, 0, 0]],
         [[ 0, 0, 0], [0, 1000, 0]],
         [[ 0, 0, 0], [0, 0, 1000]]])
    ax2= np.empty((3,2,2))
    for i in range(3):
        for j in range(2):
            ax2[i,j,:]= vicon_to_dv(m,ax[i,j,:])
    for i in range(3):
        plt.plot(ax2[i,:,0].flatten(),ax2[i,:,1].flatten())
    plt.xlim([ 0, 346])
    plt.ylim([ 0, 260])
    plt.gca().invert_yaxis()
    plt.show()


    if test:

        from vicon_dssdk import ViconDataStream

        image = np.empty(dv_frame_shape, dtype='uint8')

        vicon_client = ViconDataStream.Client()
        vicon_client.Connect(f'{vicon_address}:{vicon_port}')
        vicon_client.EnableMarkerData()

        for dv_frame in dv.NetworkFrameInput(address=dv_address, port=dv_frame_port):
            image[:, :, :] = cv2.undistort(
                dv_frame.image, dv_camera_matrix, dv_distortion_coefficients, None, dv_camera_matrix)

            if not vicon_client.GetFrame():
                continue

            prop_names = vicon_client.GetSubjectNames()
            prop_count = len(prop_names)

            for i_prop in range(prop_count):
                prop_name = prop_names[i_prop]

                marker_names = vicon_client.GetMarkerNames(prop_name)
                marker_count = len(marker_names)

                try:
                    prop_quality = vicon_client.GetObjectQuality(prop_name)
                except ViconDataStream.DataStreamException:
                    prop_quality = None

                if prop_quality is not None:

                    for i_marker in range(marker_count):
                        marker_name = marker_names[i_marker][0]

                        translation = vicon_client.GetMarkerGlobalTranslation(prop_name, marker_name)[0]
                        translation = np.array([translation])

                        # transform from Vicon space to DV camera space
                        xy = vicon_to_dv(m, translation)
                        xy_int = np.rint(xy).astype('int32')

                        cv2.circle(image, (xy_int[0], xy_int[1]), 3, (0, 255, 0), -1)

            cv2.imshow('image', image)
            k = cv2.waitKey(1)
            if k == 27 or k == ord('q'):
                exit(0)

        vicon_client.Disconnect()


    return


if __name__ == '__main__':
    calibrate()
