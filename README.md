## Advanced Lane Finding


The steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The implementation was done in python, split in the following files:
 pipeline.py, lane.py, camera.py, utils.py and main.py

[//]: # (Image References)

[image1]: ./output_images/undistorted.png "Undistorted"
[image3]: ./output_images/gradient.jpg "Binary"
[image4]: ./output_images/masked.jpg "ROI"
[image5]: ./output_images/warped.jpg  "Warp Example"
[image6]: ./output_images/lane.jpg "Fit Visual"
[image7]: ./output_images/final.jpg "Output"
[video1]: ./out.mp4 "Video"


### Camera Calibration

  The code for camera calibration is located in  camera.py. It provides a class Camera with 3 methods:
    - `get_calibration_points()`. Using a set of chessboard images, it returns a list of 3d (objpoints) points  which are the  (x, y, z) coordinates of the chessboard corners in the world and and 2d (imgpoints) points in image plane. It uses the OpenCV `findChessboardCorners()` function.
   - `calibrate()` It uses the previously saved `objpoints` and `imgpoints` in order to compute the camera calibration and distortion coefficients. It uses the `cv2.calibrateCamera()` function.
- `cal_undistort()` This is called for every image and uses the the saved camera calibration and distortion coefficients to undistort the image

### Pipeline (single images)

The pipeline.py file contains the main steps in the video processing pipeline. The API was created so its usage is simple and descriptiive like this:

	 camera = cam.Camera()
	 pipeline = Pipeline(image, camera)
	 out = pipeline \
	        .undistort() \
	        .gradient_binary() \
	        .mask_region_of_interest() \
	        .warp() \
	        .draw_lanes() \
	        .unwarp() \
	        .add_weighted() \
	        .display_params() \
	        .image

Each of these steps takes an image from the previous step and outputs a modified image to the next step, in a transparent manner.

#### 1. Distortion correction

Using the camera matrix and the distortion coefficients calculated at the during the calibration, the pipeline undistorts the image using the OpenCV `undistort()` function.
An comparative example can be seen in the following picture:

![alt text][image1]{height=350}

#### 2. Threshold using color transforms and gradients

For this step I used a combination of HLS colour threshold (using the S channel) and Sobel x-gradient threshold to generate a binary image. This can be found in function gradient_binary in pipeline.py.
Here's an example of the output for this step.

![alt text][image3]{height=300}

#### 3. Mask region of interest
In order to improve the accuracy by filtering out the noise of the lane finding algorithm, a ROI masking step is added here. This is implemented in `mask_region_of_interest()` in pipeline.py

![alt text][image4]{height=300}

#### 4. Perspective transform (warp)

The code for my perspective transform includes 2 pair functions `warp()` and `unwarp()`,  in the file `pipeline.py`. The second funtion will be used later in the pipeline, for drawing the lanes on the road.

The source points of the transform matrix were measured on the test image.
This resulted in the following source and destination points:

| Source      | Destination   |
|:-------------:|:-------------:|
| 190, 720   | 350, 720  |
| 590, 450   | 350, 0      |
| 690, 450   | 850, 0      |
| 1120, 720 | 850,720   |

The transformed image can be seen here:

![alt text][image5]{height=300}

#### 5. Lane drawing

In my pipeline drawing the lanes is accomplished with 3 pipeline steps (as per pipeline.py).
  - `draw_lanes()`
  - `unwarp()`
  - `add_weighted()`

 The first step is the only one doing lane/line specific processing. Since this is the most complex part of the  pipeline, all the lane specific functions were added in a separate file, lane.py

For computing the points which describe the lane a sliding window algorithm was used (`find_lane_pixels()` in lane.py)
The algorithm works like this: the image is split vertically in several zones, each using a window (per line) for searching the line pixels. Starting from the buttom up, detect where most of the pixels are. Initially use a histogram and the maximum of it will indicate where the first window is. Then, for the upper part, use the previous window center as a starting point. If line pixels are found, shift the window arround them.

A visualization of this algorithm can be seen in this picture:

![alt text][image6]{height=300}

Then, use numpy polyfit function to fit a 2nd order polynomial into these line pixels. It returs 3 values  which will be the definition of the line (used for drawing or for measuring curvature or position)..

Since this was computed on a binary warped image, we can now discard the image and use only the fit parameters to draw the lane (see `get_lane_pts()` in lane.py). Thus draw_lanes() will get a list of points and will use the `cv2.fillPolly()` to draw the lane on a blank image. The next steps will map this lane on the original (undistorted) image.

Sine we are still in an warped space, the drawn lane need to be mapped onto the unwarped image.
This is done with the next step in the pipeline - `unwarp()` from pipeline.py. This is the counter part of the warp() and uses the inverse matrix.

Next, the add_weighted() step will overwite the lane onto the original (undistorted) image.


#### 5. Curvature and vehicle position

Curvature and vehicle position are computed based on the left and right line fit parameters, using the functions `measure_curvature()` and `measure vehicle position()` in lane.py.

For curvature measurement I used the formula
 `Curvature = ((1 + (2*A*y + B)^2)^3/2)/abs(2*A)`, where A, B, C are the fit parameters.
Since the measurement is done per line (Line() class method), the lane curvatures is the average of left and right.

The vehicle position is computed as the distance between center of the lane (median of left and right lines) and the midpoint at the bottom of the screen (since the camera is placed in the in the middle).

Plotting these parameters is done with the `display_params()`, as the last step of the pipeline.

#### 6. Output image

After displaying the curvature and vehicle position, nothing remains to be done. The image field of the pipeline class contains the wanted output:

![alt text][image7]{height=300}

---

### Pipeline (video)

Here's a [link to my video result](./out.mp4)

---

### Discussion

The pipeline was created mostly in line with the training material.

I had some problems with the left line detection for some parts of the video, but I succeeded to fix them by slightly changing the transform matrix and by adding ROI.
Quality wise, averaging the previous lines made a big improvement. However, there is still room for improvement by ignoring the lines which are not like the previous ones (need a sanity check), since averaging will just level their effects.

