import numpy as np
import cv2
import lane
import camera as cam
import utils

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

TRANSFORM = (np.float32([[190, 720], [590, 450], [690, 450], [1120, 720]]),
             np.float32([[350, 720], [350, 0], [850, 0], [850, 720]]))

ROI = [[(150, 720), (500, 450), (780, 450), (1130, 720)]]


class Pipeline():
    def __init__(self, img, camera):
        self.image = img
        # self.orig = img
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        self.undist_img = None
        self.camera = camera
        self.camera.calibrate(img)

    def undistort(self):
        self.undist_img = self.camera.cal_undistort(self.image)
        self.image = np.copy(self.undist_img)
        # utils.display_before_after(image, undistorted, "Original", "Undistorted")
        return self

    def gradient_binary(self, s_thresh=(170, 255), sx_thresh=(20, 100)):
        img = np.copy(self.image)
        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        # Stack each channel
        # color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
        # return color_binary
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        combined_binary = cv2.convertScaleAbs(combined_binary, alpha=(255.0))
        #cv2.imwrite('binary.jpg', combined_binary)
        self.image = combined_binary
        return self

    def mask_region_of_interest(self):
        mask = np.zeros_like(self.image)
        vertices = np.array(ROI, dtype=np.int32)

        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image\n",
        if len(self.image.shape) > 2:
            channel_count = self.image.shape[2]  # i.e. 3 or 4 depending on your image\n",
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_image = cv2.bitwise_and(self.image, mask)

        #cv2.imwrite('masked.jpg', masked_image)
        self.image = masked_image
        return self

    def warp(self):
        img_size = (self.image.shape[1], self.image.shape[0])
        M = cv2.getPerspectiveTransform(*TRANSFORM)
        warped = cv2.warpPerspective(self.image, M, img_size, flags=cv2.INTER_LINEAR)
        self.image = warped

        #cv2.imwrite('warped.jpg', warped)
        return self

    def unwarp(self):
        img_size = (self.image.shape[1], self.image.shape[0])
        Minv = cv2.getPerspectiveTransform(*reversed(TRANSFORM))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        unwarped = cv2.warpPerspective(self.image, Minv, img_size)
        self.image = unwarped
        return self

    def add_weighted(self):
        # Combine the result with the original image
        weighted = cv2.addWeighted(self.undist_img, 0.9, self.image, 0.3, 0)
        self.image = weighted
        return self

    def draw_lanes(self):
        # Create image to draw the lines
        warp_zero = np.zeros_like(self.image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts = lane.get_lane_pts(self.image)

        # Fill the detected lane with color
        cv2.fillPoly(color_warp, np.int_([pts]), GREEN)
        self.image = color_warp
        return self

    def display_params(self):
        curvature = lane.measure_curvature()
        position = lane.measure_vehicle_position()

        curv_str = "Radius of Curvature = {:.1f} Km".format(curvature / 1000)
        pos_str = "Vehicle is {:.2f} m {} of center".format(abs(position), ("right", "left")[position < 0])

        # Add text to image
        cv2.putText(self.image, curv_str, (30, 50), cv2.FONT_HERSHEY_PLAIN, 3, WHITE, 2)
        cv2.putText(self.image, pos_str, (30, 100), cv2.FONT_HERSHEY_PLAIN, 3, WHITE, 2)

        return self
