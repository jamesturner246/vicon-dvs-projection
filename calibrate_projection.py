
import os
import time
import cv2
import dv
import contextlib
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


def collect_vicon_data(address, port, prop_name, marker_names, n_frame, t_frame, debug):

    from vicon_dssdk import ViconDataStream

    marker_translation = np.empty((n_frame, len(marker_names), 3), dtype='float64')

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
            assert(len(marker_names) == len(actual_marker_names))

            for i_marker in range(len(actual_marker_names)):
                marker_name = marker_names[i_marker]
                actual_marker_name = actual_marker_names[i_marker][0]
                assert(marker_name == actual_marker_name)

                translation = client.GetMarkerGlobalTranslation(prop_name, actual_marker_name)[0]
                marker_translation[i_frame, i_marker, :] = translation

            i_frame += 1
            progress.update(1)

        time.sleep(t_frame)

    progress.close()        
    client.Disconnect()

    return marker_translation


def process_vicon_data(marker_translation, debug):
    coordinates = np.median(marker_translation, 0)

    return coordinates


def get_vicon_coordinates(i_epoch, address, port, prop_name, marker_names,
                          n_frame=100, t_frame=0.001,
                          reuse=False, debug=False,
                          path='./calibration'):

    print('get Vicon coordinates')

    marker_translation_file = f'{path}/vicon_marker_translation_{i_epoch}.npy'
    coordinates_file = f'{path}/vicon_coordinates_{i_epoch}.npy'

    if reuse:
        marker_translation = np.load(marker_translation_file)

        coordinates = process_vicon_data(
            marker_translation, debug)
        np.save(coordinates_file, coordinates)

    else:
        marker_translation = collect_vicon_data(
            address, port, prop_name, marker_names, n_frame, t_frame, debug)
        np.save(marker_translation_file, marker_translation)

        coordinates = process_vicon_data(
            marker_translation, debug)
        np.save(coordinates_file, coordinates)

    # print Vicon coordinates
    for i in range(len(marker_names)):
        print(f'name: {marker_names[i]:<15s} coordinates: {coordinates[i]}')

    return coordinates


def collect_dv_data(address, event_port, n_event, debug):
    event_xy = np.empty((n_event, 2), dtype='int32')

    with dv.NetworkEventInput(address=address, port=event_port) as f:
        progress = tqdm(total=n_event, leave=False)
        i_event = 0

        for event in f:
            if i_event == n_event:
                break

            # record and undistort event
            event_xy[i_event] = [event.x, event.y]

            i_event += 1
            progress.update(1)

        progress.close()

    return event_xy


def process_dv_data(event_xy, marker_count, camera_shape, camera_mtx, camera_dist, debug):
    erode_dv_kernel = np.ones((2, 2), 'uint8')
    dilate_dv_kernel = np.ones((10, 10), 'uint8')

    event_image = np.zeros(camera_shape, dtype='int64')
    event_xy_masked = np.empty(event_xy.shape, dtype='int32')

    # accumulate events
    for xy in event_xy:
        xy_undistorted = cv2.undistortPoints(
            xy.astype('float64'), camera_mtx, camera_dist, None, camera_mtx)[0, 0]
        xy_int = np.rint(xy_undistorted).astype('int32')
        xy_bounded = all(xy_int >= 0) and all(xy_int < camera_shape[::-1])

        if xy_bounded:
            event_image[xy_int[1], xy_int[0]] += 1.0

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

    # mask events
    i_event = 0
    for xy in event_xy:
        xy_undistorted = cv2.undistortPoints(
            xy.astype('float64'), camera_mtx, camera_dist, None, camera_mtx)[0, 0]
        xy_int = np.rint(xy_undistorted).astype('int32')
        xy_bounded = all(xy_int >= 0) and all(xy_int < camera_shape[::-1])

        if xy_bounded:
            if event_image_mask[xy_int[1], xy_int[0]] > 0:
                event_xy_masked[i_event] = xy_undistorted
                i_event += 1

    event_xy_masked.resize((i_event, 2))

    if debug:
        # plot event image mask (erode and dilate)
        cv2.imshow('event image mask (erode and dilate)', event_image_mask)
        cv2.waitKey()

        # plot masked event image
        cv2.imshow('masked event image', (event_image / event_image.max()).astype('float32'))
        cv2.waitKey()

    # k-means of masked events
    k_means = KMeans(
        n_clusters=marker_count,
        init='k-means++',
        n_init=100,
        max_iter=300,
        #tol=1e-4,
        #random_state=0,
    )
    result = k_means.fit(event_xy_masked)
    coordinates = result.cluster_centers_
    coordinates = identify_wand_markers(coordinates)

    if debug:
        coordinates_int = np.rint(coordinates).astype('int32')
        coordinates_image = np.zeros(camera_shape, dtype='uint8')
        coordinates_image[coordinates_int[:, 1], coordinates_int[:, 0]] = 255

        # plot DV coordinates
        cv2.imshow('event coordinates', coordinates_image)
        cv2.waitKey()

        # close debug plots
        cv2.destroyAllWindows()

    return coordinates


def get_dv_coordinates(i_epoch, address, event_port, frame_port, prop_name, marker_names,
                       camera, camera_shape, camera_mtx, camera_dist, n_event=10000,
                       path='./calibration', reuse=False, debug=False):

    print(f'get DV {camera} coordinates')

    event_xy_file = f'{path}/dv_{camera}_event_xy_{i_epoch}.npy'
    coordinates_file = f'{path}/dv_{camera}_coordinates_{i_epoch}.npy'

    if reuse:
        event_xy = np.load(event_xy_file)

        coordinates = process_dv_data(
            event_xy, len(marker_names), camera_shape, camera_mtx, camera_dist, debug)
        np.save(coordinates_file, coordinates)

    else:
        done = False
        while not done:
            event_xy = collect_dv_data(
                address, event_port, n_event, debug)
            np.save(event_xy_file, event_xy)

            coordinates = process_dv_data(
                event_xy, len(marker_names), camera_shape, camera_mtx, camera_dist, debug)
            np.save(coordinates_file, coordinates)

            # plot coordinates
            with dv.NetworkFrameInput(address=address, port=frame_port) as f:
                frame = next(f)

            frame_image = cv2.undistort(
                frame.image, camera_mtx, camera_dist, None, camera_mtx)

            # plot markers
            fig, ax = plt.subplots()
            ax.imshow(frame_image[:, :, (2, 1, 0)])
            ax.plot(coordinates[:, 0], coordinates[:, 1], '*b')
            ax.text(coordinates[0, 0], coordinates[0, 1], 'top_left', color='blue')
            ax.text(coordinates[1, 0], coordinates[1, 1], 'top_centre', color='blue')
            ax.text(coordinates[2, 0], coordinates[2, 1], 'top_right', color='blue')
            ax.text(coordinates[3, 0], coordinates[3, 1], 'middle', color='blue')
            ax.text(coordinates[4, 0], coordinates[4, 1], 'bottom', color='blue')
            ax.set_title(f'camera {camera}')
            ax.set_ylim([0, camera_shape[0]])
            ax.set_xlim([0, camera_shape[1]])
            ax.invert_yaxis()
            plt.show()

            while True:
                accept = input('accept coordinates? (y/n/q): ')
                if accept == 'y':
                    done = True
                    break
                elif accept == 'n':
                    break
                elif accept == 'q':
                    exit(0)

    # print DV coordinates
    for i in range(len(marker_names)):
        print(f'camera: {camera} name: {marker_names[i]:<15s} coordinates: {coordinates[i]}')

    return coordinates


def vicon_to_dv_method_1(v, m, origin_offset, nominal_focal_length, pixel_mm):
    z = vicon_to_camera_centric_method_1(v, m)
    z[:2] *= (1 / z[2])
    z = z[:2]
    z *= nominal_focal_length
    z /= pixel_mm       # millimetres per pixel
    z += origin_offset  # add the origin offset from image centre to top left corner explicitly
    return z


def vicon_to_camera_centric_method_1(v, m):
    M = np.reshape(m[:9], (3, 3))
    z = np.dot(v, M)    # apply the rotation to get into the camera orientation frame
    z += m[9:12] * 10   # add the translation (using cm for a better scale of fitting)
    return z


def vicon_to_dv_method_2(v, m, origin_offset, nominal_focal_length, pixel_mm):
    focal_length = nominal_focal_length * m[6]
    z = vicon_to_camera_centric_method_2(v, m)
    z[:2] *= (1 / z[2])
    z = z[:2]
    z *= focal_length
    z /= pixel_mm       # millimetres per pixel
    z[0] *= m[7]        # allow for some rescaling of x due to the camera undistortion method
    z += origin_offset  # add the origin offset from image centre to top left corner explicitly
    return z


def vicon_to_camera_centric_method_2(v, m):
    M = euler_angles_to_rotation_matrix(m)
    z = np.dot(v, M)    # apply the rotation to get into the camera orientation frame
    z += m[3:6] * 10    # add the translation (using cm for a better scale of fitting)
    return z


def euler_angles_to_rotation_matrix(m):
    M = np.array([
        [ np.cos(m[0]), np.sin(m[0]), 0],
        [-np.sin(m[0]), np.cos(m[0]), 0],
        [ 0,            0,            1]])
    M = np.dot(M, np.array([
        [1,  0,            0],
        [0,  np.cos(m[1]), np.sin(m[1])],
        [0, -np.sin(m[1]), np.cos(m[1])]]))
    M = np.dot(M, np.array([
        [ np.cos(m[2]), np.sin(m[2]), 0],
        [-np.sin(m[2]), np.cos(m[2]), 0],
        [ 0,            0,            1]]))
    return M

def rotation_matrix_to_euler_angles(M):
    tolerance= 1e-10
    m= np.empty(3)   
    m[0]= np.arctan(M[0,2]/M[1,2])
    m[1]= np.arccos(M[2,2])
    m[2]= np.arctan(-M[2,0]/M[2,1])
    # problem: Euler angles alpha and gamma are [-pi, pi] not just [-pi/2, pi/2] as produced by arctan. Need to find out whether we have the right angle for alpha and gamma:
    # M[0,2] == sin(m[0])*sin(m[1]) can be used to check m[0] - it needs to produce the right sign
    if M[0,2]*np.sin(m[0])*np.sin(m[1]) < 0.0:
        if m[0] > 0.0:
            m[0]= m[0]-np.pi
        else:
            m[0]= m[0]+np.pi
    # M[2,0] == sin(m[1])*sin(m[2]) can be used to check m[2] - it needs to produce the right sign
    if M[2,0]*np.sin(m[1])*np.sin(m[2]) < 0.0:
        if m[2] > 0.0:
            m[2]= m[2]-np.pi
        else:
            m[2]= m[2]+np.pi
    return m
        
def err_fun(m, vicon_p, dv_p, vicon_to_dv, origin_offset, nominal_focal_length, pixel_mm):
    assert dv_p.shape[0] == vicon_p.shape[0]

    error = 0.0
    for v, d in zip(vicon_p, dv_p):
        output = vicon_to_dv(v, m, origin_offset, nominal_focal_length, pixel_mm)
        difference = output - d
        error += np.dot(difference, difference)

    return np.sqrt(error)


def calibrate():

    n_epoch = 20

    calibrate_projection_method = 2

    reuse = True
    debug = False
    test = True

    path_camera = './camera_calibration'
    path_projection = './projection_calibration'
    path = './calibration'

    # make calibration directories
    os.makedirs(path_projection, exist_ok=True)
    if not reuse:
        os.makedirs(path, exist_ok=True)

    prop_name = 'jt_wand'
    marker_names = [
        'top_left',
        'top_centre',
        'top_right',
        'middle',
        'bottom',
    ]
    n_marker = len(marker_names)

    dv_n_event = 10000
    vicon_n_frame = 100
    vicon_t_frame = 0.001

    # servers
    vicon_address, vicon_port = '127.0.0.1', 801
    dv_address = '127.0.0.1'
    dv_event_port = [36000, 36001]
    dv_frame_port = [36002, 36003]

    # DV camera
    dv_camera_shape = [np.array([260, 346]) for i in range(2)]
    dv_camera_mtx_file_name = [f'{path_camera}/camera_{i}_matrix.npy' for i in range(2)]
    dv_camera_mtx = [np.load(file_name) for file_name in dv_camera_mtx_file_name]
    dv_camera_dist_file_name = [f'{path_camera}/camera_{i}_distortion_coefficients.npy' for i in range(2)]
    dv_camera_dist = [np.load(file_name) for file_name in dv_camera_dist_file_name]
    dv_camera_origin_offset = [dv_camera_shape[i] / 2 for i in range(2)]
    dv_camera_nominal_focal_length = [4.0 for i in range(2)]
    dv_camera_pixel_mm = [1.8e-2 for i in range(2)]

    # allocate temp memory
    vicon_wand_coordinates = np.empty((n_epoch, n_marker, 3), dtype='float64')
    dv_wand_coordinates = [np.empty((n_epoch, n_marker, 2), dtype='float64') for i in range(2)]

    i_epoch = 0
    while i_epoch < n_epoch:
        print(f'calibration epoch {i_epoch}')
        if not reuse:
            input('relocate prop and press enter...')

        vicon_wand_coordinates[i_epoch, :, :] = get_vicon_coordinates(
            i_epoch, vicon_address, vicon_port, prop_name, marker_names,
            n_frame=vicon_n_frame, t_frame=vicon_t_frame,
            reuse=reuse, debug=debug, path=path)

        for i in range(2):
            dv_wand_coordinates[i][i_epoch, :, :] = get_dv_coordinates(
                i_epoch, dv_address, dv_event_port[i], dv_frame_port[i], prop_name, marker_names,
                i, dv_camera_shape[i], dv_camera_mtx[i], dv_camera_dist[i], n_event=dv_n_event,
                path=path, reuse=reuse, debug=debug)

        if not reuse:
            while True:
                accept = input('accept epoch? (y/n/q): ')
                if accept == 'y':
                    i_epoch += 1
                    break
                elif accept == 'n':
                    break
                elif accept == 'q':
                    exit(0)
        else:
            i_epoch += 1


    #########################################################################

    if calibrate_projection_method == 1:
        vicon_to_dv = vicon_to_dv_method_1

        # Vicon to DV transformation
        m_file = [f'{path_projection}/dv_{i}_space_transform.npy' for i in range(2)]
        m = [np.load('./bootstrap_calibration/bootstrap_dv_space_transform.npy') for i in range(2)]
        x = np.vstack(vicon_wand_coordinates)
        y = [np.vstack(coordinates) for coordinates in dv_wand_coordinates]

        method = 'nelder-mead'
        options = {'disp': True, 'maxiter': 50000, 'maxfev': 100000, 'xatol': 1e-10, 'fatol': 1e-10}

        for i in range(2):
            err = err_fun(m[i], x, y[i], vicon_to_dv, dv_camera_origin_offset[i],
                          dv_camera_nominal_focal_length[i], dv_camera_pixel_mm[i])
            print(f'camera {i} transform: original guess has error: {err}')

            result = minimize(err_fun, m[i], method=method, options=options,
                              args=(x, y[i], vicon_to_dv, dv_camera_origin_offset[i],
                                    dv_camera_nominal_focal_length[i], dv_camera_pixel_mm[i]))
            m[i] = result['x']

            err = err_fun(m[i], x, y[i], vicon_to_dv, dv_camera_origin_offset[i],
                          dv_camera_nominal_focal_length[i], dv_camera_pixel_mm[i])
            print(f'camera {i} transform: final result has error: {err}')

            np.save(m_file[i], m[i])

            print('DV space coefficients')
            print(np.reshape(m[i][:9], (3, 3)))
            print('DV space constants')
            print(m[i][9:12])
            print()

    #########################################################################

    elif calibrate_projection_method == 2:
        vicon_to_dv = vicon_to_dv_method_2

        # the meaning of m is as follows:
        # 0-2 Euler angles of transform into camera oriented space
        # 3-5 translation of vector to be relative to camera origin (chosen as pinhole position)
        # 6 scale factor for stretch in x-direction due to camera calibration/undistortion

        # Vicon to DV transformation
        m_file = [f'{path_projection}/dv_{i}_space_transform.npy' for i in range(2)]
        m = [np.empty(8) for i in range(2)]
        for i in range(2):
            # initial guess for angles - turn x to -x and rotate around x to get z pointing in -z direction
            m[i][:3] = [3.14, 1.6, 0.0]
            # initial guess for translation from pinhole to vicon origin (in cm)
            m[i][3:6] = [22.0, 93.0, 231.0]
            # initial guess for the deviation of focal length and x-stretching from camera undistortion
            m[i][6:8] = [1.0, 1.0]
        x = np.vstack(vicon_wand_coordinates)
        y = [np.vstack(coordinates) for coordinates in dv_wand_coordinates]

        method = 'nelder-mead'
        options = {'disp': True, 'maxiter': 50000, 'maxfev': 100000, 'xatol': 1e-10, 'fatol': 1e-10}

        for i in range(2):
            err = err_fun(m[i], x, y[i], vicon_to_dv, dv_camera_origin_offset[i],
                          dv_camera_nominal_focal_length[i], dv_camera_pixel_mm[i])
            print(f'camera {i} transform: original guess has error: {err}')

            result = minimize(err_fun, m[i], method=method, options=options,
                              args=(x, y[i], vicon_to_dv, dv_camera_origin_offset[i],
                                    dv_camera_nominal_focal_length[i], dv_camera_pixel_mm[i]))
            m[i] = result['x']

            err = err_fun(m[i], x, y[i], vicon_to_dv, dv_camera_origin_offset[i],
                          dv_camera_nominal_focal_length[i], dv_camera_pixel_mm[i])
            print(f'camera {i} transform: final result has error: {err}')

            np.save(m_file[i], m[i])

            print("Euler angles: {}".format(m[i][:3]))
            print("Translation: {}".format(m[i][3:6]))
            print("focal length and x rescales: {}".format(m[i][6:]))
            print("The matrix: {}".format(euler_angles_to_rotation_matrix(m[i])))
            print()

    #########################################################################

    else:
        raise RuntimeError('invalid projection calibration method')

    #########################################################################


    for i in range(2):
        plt.figure()
        for j in range(y[i].shape[0] // 5):
            plt.scatter(y[i][5*j : 5*(j+1), 0], y[i][5*j : 5*(j+1), 1])
        z = np.zeros(y[i].shape)
        for j in range(x.shape[0]):
            z[j, :] = vicon_to_dv(x[j, :], m[i], dv_camera_origin_offset[i],
                                  dv_camera_nominal_focal_length[i], dv_camera_pixel_mm[i])
        for j in range(z.shape[0] // 5):
            plt.scatter(z[5*j : 5*(j+1), 0], z[5*j : 5*(j+1), 1], marker='x')
        ax = np.array([[[ 0, 0, 0], [1000, 0, 0]],
                       [[ 0, 0, 0], [0, 1000, 0]],
                       [[ 0, 0, 0], [0, 0, 1000]]])
        ax2 = np.empty((3, 2, 2))
        for j in range(3):
            for k in range(2):
                ax2[j, k, :] = vicon_to_dv(ax[j, k, :], m[i], dv_camera_origin_offset[i],
                                           dv_camera_nominal_focal_length[i], dv_camera_pixel_mm[i])
        for j in range(3):
            plt.plot(ax2[j, :, 0].flatten(), ax2[j, :, 1].flatten())
        plt.xlim([0, 346])
        plt.ylim([0, 260])
        plt.gca().invert_yaxis()
        plt.show()



    if test:

        from vicon_dssdk import ViconDataStream

        vicon_client = ViconDataStream.Client()
        vicon_client.Connect(f'{vicon_address}:{vicon_port}')
        vicon_client.EnableMarkerData()

        frame_image = [np.empty(np.hstack((dv_camera_shape[i], [3])), dtype='uint8') for i in range(2)]

        with contextlib.ExitStack() as stack:
            frame_servers = [stack.enter_context(dv.NetworkFrameInput(
                address=dv_address, port=dv_frame_port[i])) for i in range(2)]

            while True:
                dv_frame = [next(frame_servers[i]) for i in range(2)]

                for i in range(2):
                    frame_image[i][:, :, :] = cv2.undistort(
                        dv_frame[i].image, dv_camera_mtx[i], dv_camera_dist[i], None, dv_camera_mtx[i])

                if not vicon_client.GetFrame():
                    continue

                prop_names = vicon_client.GetSubjectNames()
                prop_count = len(prop_names)

                for i_prop in range(prop_count):
                    prop_name = prop_names[i_prop]
                    marker_names = vicon_client.GetMarkerNames(prop_name)

                    try:
                        prop_quality = vicon_client.GetObjectQuality(prop_name)
                    except ViconDataStream.DataStreamException:
                        prop_quality = None

                    if prop_quality is not None:

                        for i_marker in range(len(marker_names)):
                            marker_name = marker_names[i_marker][0]

                            translation = vicon_client.GetMarkerGlobalTranslation(prop_name, marker_name)[0]
                            translation = np.array(translation)

                            # transform from Vicon space to DV camera space
                            for i in range(2):
                                xy = vicon_to_dv(translation, m[i], dv_camera_origin_offset[i],
                                                 dv_camera_nominal_focal_length[i], dv_camera_pixel_mm[i])
                                xy_int = np.rint(xy).astype('int32')
                                cv2.circle(frame_image[i], (xy_int[0], xy_int[1]), 3, (0, 255, 0), -1)

                for i in range(2):
                    cv2.imshow(f'camera {i}', frame_image[i])
                    k = cv2.waitKey(1)
                    if k == 27 or k == ord('q'):
                        exit(0)

        vicon_client.Disconnect()

    return


if __name__ == '__main__':
    calibrate()
