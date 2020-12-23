import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


# import matplotlib.image as mpimg


class Camera():
    def __init__(self):
        self.calibrated = None
        self.mtx = None
        self.dist = None

        self.objpoints, self.imgpoints = self.get_calibration_points()

    def get_calibration_points(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob('camera_cal/calibration*.jpg')

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            # img = mpimg.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # If found, add object points, image points
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                # cv2.imshow('img',img)

        return objpoints, imgpoints

    def calibrate(self, img, force=False):
        if (not self.calibrated or force):
            ret, self.mtx, self.dist, rvecs, tvecs =\
                cv2.calibrateCamera(self.objpoints, self.imgpoints, img.shape[1:], None, None)
            self.calibrated = True

    def cal_undistort(self, img):
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return undist
