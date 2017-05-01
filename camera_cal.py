import cv2
import glob
import numpy as np


def get_camera_distortion(img_dir, nx=9, ny=6):
    """
    Given a directory containing chessboard calibration images with nx corners in the x-axis and ny corners in the
    y-axis, calculate the camera matrix and distortion coefficients. Note: all calibration images must be the same
    size or this method may produce erroneous results.
     
    :param img_dir: The directory containing calibration images
    :param nx: The number of chessboard corners along the x-axis 
    :param ny: The number of chessboard corners along the y-axis
    :return: Camera matrix and distortion coefficients that can be used to undistort images
    """

    # Setup the object and image points
    obj_points = []
    img_points = []
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Get corners from all of the calibration images
    for img_file in glob.iglob(img_dir + '/calibration*.jpg'):
        img = cv2.imread(img_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret:
            obj_points.append(objp)
            img_points.append(corners)

    # Calibrate the camera - we are assuming here that all images have the same shape (1280x720)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    # Return the distortion matrix if we were able to calculate one
    if ret:
        return mtx, dist
    else:
        return None


if __name__ == '__main__':
    mtx, dist = get_camera_distortion("camera_cal")
    print(mtx)
    print(dist)
