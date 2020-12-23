import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import *
from collections import deque

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

VISUALIZATION = False

# Class to receive the characteristics of each line detection
class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False

        # buffer of the last fits.
        self.prev_fits = deque(maxlen=16)

        # current fit
        self.fit = None

    def get_curvature_real(self):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        ploty = np.linspace(0, 719, num=720)

        # Calculate x values using polynomial coeffs
        plotx = self.fit[0] * ploty ** 2 + self.fit[1] * ploty + self.fit[2]

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)

        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(ploty * ym_per_pix, plotx * xm_per_pix, 2)

        # curve_fit = np.polyfit(ploty * ym_per_pix, fitx * xm_per_pix, 2)
        # x = mx / (my ** 2) * a * (y ** 2) + (mx / my) * b * y + c

        # Calculate the radius of curvature
        curve_rad = ((1 + (2 * fit_cr[0] * y_eval + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])

        return curve_rad

    def get_position(self):
        # Get the x position at the bottom of the screen
        y = 719
        return self.fit[0] * y ** 2 + self.fit[1] * y + self.fit[2]


left = Line()
right = Line()


def avg_fit(fits):
    avg = sum(fits)/len(fits)
    return avg


def fit_poly(leftx, lefty, rightx, righty):
    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 11
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        if (VISUALIZATION):
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), GREEN, 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), GREEN, 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        print("Error")
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit polynomial based on pixels found
    left_fit, right_fit = fit_poly(leftx, lefty, rightx, righty)

    if (VISUALIZATION):
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = RED
        out_img[righty, rightx] = BLUE

        display_img(out_img)

    return left_fit, right_fit


def search_around_poly(binary_warped):
    left_fit = left.fit
    right_fit = right.fit

    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set the area of search based on activated x-values
    # within the +/- margin of our polynomial function
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fit, right_fit = fit_poly(leftx, lefty, rightx, righty)

    if (VISUALIZATION):
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = RED
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = BLUE

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), GREEN)
        cv2.fillPoly(window_img, np.int_([right_line_pts]), GREEN)
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # Plot the polynomial lines onto the image
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        ## End visualization steps ##

    return left_fit, right_fit


def get_lane_pts(warped):

    if (left.detected and right.detected):
        left_fit, right_fit = search_around_poly(warped)
        left.detected = False
        right.detected = False
    else:
        # Find our lane pixels first
        left_fit, right_fit = find_lane_pixels(warped)
        left.detected = True
        right.detected = True

    left.prev_fits.append(left_fit)
    right.prev_fits.append(right_fit)
    avg_left = avg_fit(left.prev_fits)
    avg_right = avg_fit(right.prev_fits)
    #left.prev_fits.pop()
    #right.prev_fits.pop()
    #left.prev_fits.append(avg_left)
    #right.prev_fits.append(avg_right)

    left.fit = avg_left
    right.fit = avg_right


    # Generate y values
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])

    # Calculate x values using polynomial coeffs
    left_fitx = left.fit[0] * ploty ** 2 + left.fit[1] * ploty + left.fit[2]
    right_fitx = right.fit[0] * ploty ** 2 + right.fit[1] * ploty + right.fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    return pts

def measure_curvature():
    curvature = (left.get_curvature_real() + right.get_curvature_real()) // 2
    return curvature

def measure_vehicle_position():
    m_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Vehicle position is the offset of the midpoint with regards to lane center
    midpoint = 1280 / 2
    center_of_lanes = (right.get_position() + left.get_position()) / 2
    position_meters = (midpoint - center_of_lanes) * m_per_pix

    return position_meters

