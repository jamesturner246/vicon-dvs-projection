
import matplotlib.pyplot as plt
import numpy as np
import cv2
import dv




def calibrate():
    image_shape = (260, 346)
    cb_rows = 9
    cb_cols = 6

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0, 0, 0), (1, 0, 0), (2, 0, 0) ...., (6, 5, 0)
    objp = np.zeros((cb_cols * cb_rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cb_rows, 0:cb_cols].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    with dv.NetworkFrameInput(address='192.168.1.100', port=36001) as f:

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

                    cv2.destroyWindow('chessboard corners')

    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)

    np.save('./calib/mtx.npy', mtx)
    np.save('./calib/dist.npy', dist)
    np.save('./calib/rvecs.npy', rvecs)
    np.save('./calib/tvecs.npy', tvecs)

    return





def test():
    mtx = np.load('./calib/mtx.npy')
    dist = np.load('./calib/dist.npy')

    image_distorted = np.load('./distorted.npy')

    point_distorted = []
    for i in range(33):
        for j in range(24):
            point = np.array([10 * i + 10, 10 * j + 10])
            point_distorted.append(point.astype('float64'))
            image_distorted[point[1], point[0], :] = np.array([0, 0, 0])
    point_distorted = np.vstack(point_distorted)


    # h, w = image_distorted.shape[:2]
    # #newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))


    image_undistorted = cv2.undistort(image_distorted, mtx, dist, None, mtx)
    #image_undistorted = cv2.undistort(image_distorted, mtx, dist, None, newcameramtx)

    point_undistorted = []
    for x in point_distorted: 
        y = cv2.undistortPoints(x, mtx, dist, None, mtx)[0]
        #y = cv2.undistortPoints(x, mtx, dist, None, newcameramtx)[0]
        point_undistorted.append(y)
    point_undistorted = np.vstack(point_undistorted)

    #x, y, w, h = roi
    #image_undistorted = image_undistorted[y:y+h, x:x+w]


    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image_distorted[:, :, (2, 1, 0)])
    ax[1].imshow(image_undistorted[:, :, (2, 1, 0)])
    ax[1].plot(point_undistorted[:, 0], point_undistorted[:, 1], 'b.', markerfacecolor=None)
    plt.show()

    return


if __name__ == '__main__':
    #calibrate()
    test()
