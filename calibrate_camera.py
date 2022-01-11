
import matplotlib.pyplot as plt
import numpy as np
import cv2
import dv
import os
import time


def calibrate():

    # config
    image_shape = (260, 346)
    cb_rows = 9
    cb_cols = 6

    # files
    date = time.strftime('%Y%m%d')
    path = f'./camera_calibration/{date}'
    os.makedirs(path, exist_ok=True)
    mtx_file_name = [f'{path}/camera_{i}_matrix.npy' for i in range(2)]
    dist_file_name = [f'{path}/camera_{i}_distortion_coefficients.npy' for i in range(2)]

    # servers
    dv_address = '127.0.0.1'
    dv_frame_port = [36002, 36003]

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    # === CALIBRATE ===
    for i in range(2):
        print(f'calibrating camera {i}')

        # prepare object points, like (0, 0, 0), (1, 0, 0), (2, 0, 0) ...., (6, 5, 0)
        objp = np.zeros((cb_cols * cb_rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cb_rows, 0:cb_cols].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        sample_i = 0

        with dv.NetworkFrameInput(address=dv_address, port=dv_frame_port[i]) as f:

            for frame in f:
                image_colour = frame.image.copy()
                image_grey = cv2.cvtColor(image_colour, cv2.COLOR_BGR2GRAY)

                cv2.imshow('frame image', image_colour)
                k = cv2.waitKey(1)

                if k == ord('q'):
                    break
                elif k == ord(' '):
                    ret, corners = cv2.findChessboardCorners(image_grey, (cb_rows, cb_cols), None)

                    # If found, add object points, image points (after refining them)
                    if ret == True:
                        corners = cv2.cornerSubPix(image_grey, corners, (5, 5), (-1, -1), criteria)

                        # Draw and display the corners
                        cv2.drawChessboardCorners(image_colour, (cb_rows, cb_cols), corners, ret)
                        cv2.imshow('chessboard corners', image_colour)
                        k = cv2.waitKey(0)

                        if k == ord('q'):
                            break
                        elif k == ord(' '):
                            objpoints.append(objp)
                            imgpoints.append(corners)

                            print(sample_i)
                            sample_i += 1

                        cv2.destroyWindow('chessboard corners')

        cv2.destroyAllWindows()

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)

        np.save(mtx_file_name[i], mtx)
        np.save(dist_file_name[i], dist)

    return


def test():

    # files
    date = time.strftime('%Y%m%d')
    path = f'./camera_calibration/{date}'
    mtx_file_name = [f'{path}/camera_{i}_matrix.npy' for i in range(2)]
    dist_file_name = [f'{path}/camera_{i}_distortion_coefficients.npy' for i in range(2)]

    # servers
    dv_address = '127.0.0.1'
    dv_frame_port = [36002, 36003]

    mtx = [np.load(name) for name in mtx_file_name]
    dist = [np.load(name) for name in dist_file_name]

    # === TEST ===
    for i in range(2):
        print(f'testing camera {i}')

        with dv.NetworkFrameInput(address=dv_address, port=dv_frame_port[i]) as f:
            frame = next(f)
            image_distorted = frame.image.copy()

        point_distorted = []
        for j in range(33):
            for k in range(24):
                point = np.array([10 * j + 10, 10 * k + 10])
                point_distorted.append(point.astype('float64'))
                image_distorted[point[1], point[0], :] = np.array([0, 0, 0])
        point_distorted = np.vstack(point_distorted)

        image_undistorted = cv2.undistort(image_distorted, mtx[i], dist[i], None, mtx[i])

        point_undistorted = []
        for x in point_distorted:
            y = cv2.undistortPoints(x, mtx[i], dist[i], None, mtx[i])[0]
            point_undistorted.append(y)
        point_undistorted = np.vstack(point_undistorted)

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image_distorted[:, :, (2, 1, 0)])
        ax[1].imshow(image_undistorted[:, :, (2, 1, 0)])
        ax[1].plot(point_undistorted[:, 0], point_undistorted[:, 1], 'b.', markerfacecolor=None)
        plt.show()

    return


if __name__ == '__main__':
    calibrate()
    test()
