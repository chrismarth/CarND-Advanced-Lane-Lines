
# Advanced Lane Finding Project

The goals of this project were the following:

* Compute a camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a binary image that can be used to find lane lines.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/undistorted_straight1.jpg "Road Undistorted"
[image3]: ./output_images/test6_threshold.jpg "Binary Example"
[image4]: ./test_images/test6.jpg "Binary Example Original"
[image5]: ./test_images/straight_lines1.jpg "Warp Example Original"
[image6]: ./output_images/straight_lines1_warped.jpg "Warp Example"
[image7]: ./output_images/test2_warpedlines.jpg "Warped Fit Visual"
[image8]: ./output_images/test2_warpedthreshold.jpg "Warped Threshold Visual"
[image9]: ./output_images/test2_warpedoverlay.jpg "Unwarped Fit Overlay"
[image10]: ./output_images/camera_calibration.jpg "Camera Calibration"
[video8]: ./project_video.mp4 "Video"

### Files included in GitHub Repository

* `camera_cal.py` - Contains the code to perform camera calibration.
* `curvature.py` - Contains the code to calculate radius of curvature and offset.
* `run_model.ipynb` - Jupyter notebook Showing results from each pipeline stage and how final video was created.
* `threshold.py` - Contains the code apply various thresholds to produce binary image.
* `transform.py` - Contains the code to apply perspective transform to warp image.
* `window.py` - Contains the code to find lane lines in warped binary images.
* `writeup.py` - The file you are currently reading.

### Camera Calibration

The code for this step is contained in both the Jupyter notebook  `run_model.ipynb`, under the section titled "Camera Calibration" and in the file `camera_cal.py`.  

Calibration was performed by identifying the "object points" on the calibration images, and in each image these are the (x, y, z) coordinates of the chessboard corners. It is assumed that the chessboard is fixed on the (x, y) plane at z = 0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

`obj_points` and `img_points` were then used to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained the following result for the first calibraion image: 

![alt text][image10]

The `cv2.undistort()` function was subsequently used to undistort all input camera images in all subsequent modeling steps.


### Pipeline Description

The processing pipeline used to identify the lane and lane lines consisted of the following steps:

1. Correct for input image distortion using the calculated camera matrix and distortion coefficients
2. Apply image gradient and color space thresholds to produce a binary image that isolates lane lines as best possible
3. Apply a perspective transform to warp the binary image to a 2D flat plane
4. Identify the lane using a convolutional sliding window approach
5. Fit a polynomial curve to the identified left and right lane lines
6. Calculate the radius of curvature and offset using the polynomial fits
7. Draw an overlay of the left and right lanes and lane area
8. Apply the inverse perspective transform to warp the image back to 3D space
9. Add Radius of curvature and offset annotations to image
10. Save the output image

The main aspects of these steps are summarized in the following sections.

#### 1. Distortion-corrected image

As mentioned above, the `cv2.undistort()` function was used to undistort all input camera images in all subsequent modeling steps. An example input image is shown here:

![alt text][image2]

#### 2. Produce Binary images

A combination of color and gradient thresholds were applied to each image in order to produce a binary image. The code for this step is contained in both the Jupyter notebook  `run_model.ipynb`, under the section titled "Thresholding" and in the file `threshold.py`. Much experimentation was performed to find a combination of color space transformations and gradients that produced an image that isolated the lane lines as best possible. Ultimately, the input BGR images were tranformed to the HLS color space and separate binary images for thresholds applied to both the L and S channels were produced. Likewise, the Sobel gradient was applied in the x-direction to produce an additional binary image. These three binary images were then combined to produce the final binary image used for analysis. An example of the original images and the resulting binary image are shown here:

![alt text][image4] 

![alt text][image3]

Despite our best efforts to produce impeccable binary images, image frames that included bright asphalt or concrete as well as frames that included dark shadows proved to be problematic. Subsequent sections will explain how these problem frames were dealt with.

#### 3. Perspective Transform

The code for this step is contained in both the Jupyter notebook  `run_model.ipynb`, under the section titled "Thresholding" and in the file `transform.py`.   I inspected the test images depicting straight road travel and manually chose appropriate source and destination points. The points chosen were as follows:

```python
src = np.float32([[265, 690], [1055, 690], [595, 450], [685, 450]])
dst = np.float32([[265, 720], [1055, 720], [265, 0], [1055, 0]])
```
With these points we calculate the perspective tranform matrix M and it's inverse Minv (in the method `get_perspective_transform`). These matrices were used to transform between 2D and 3D space during processing.

Verification that the perspective transform was working as expected by subjectively examining the transformed image for the test images on straight roads. For these images, the transformed image should show vertical parallel lane lines. An example of the original and transformed image for one of the straight line test images is shown here:

![alt text][image5]

![alt text][image6]

It should also be noted that the selection of src and dest points seemed to have an impact on radius of curvature and offset calculations. This makes intuitive sense as src points picked at different y-locations in the image will result in a different vertical length in the transformed image, which means the conversion from pixels to linear length (in meters) may be different. Some effort was made to correct for this in radius of curvature and offset calculations.

#### 4. Identify Lane Lines 

The code for this step is contained in both the Jupyter notebook  `run_model.ipynb`, under the section titled "Finding Lane Lines" and in the file `window.py`.
Once we had a binary image warped to the 2-D plane a convolutional sliding window search was performed to identify left and right lane lines. Code for this part of the pipeline was heavily influenced by that detailed in the lecture notes. Once the sliding window search identified window centroids (in the method `find_window_centroids`), these centroids were converted to (X,Y) points and a second order polynomial was fit was used to generate the left and right lane lines (in the method `fit_polynomial_blind`). Using the polynomial, we were then able to generate points to draw the lane lines and lane area using the cv2.fillPoly and cv2.polylines functions (in the method `draw_lane_lines`). An example of a 2D binary image and the lane lines that were drawn are shown here:

![alt text][image8]

![alt text][image7]

#### 5. Radius of Curvature of the lane and Vehicle Offset.

The code for this step is contained in both the Jupyter notebook  `run_model.ipynb`, under the section titled "Radius of Curvature" and in the file `curvature.py`. With the polynomial fit for both the left and right lane, a simple calculation for radius of curvature and offset were performed in the functions `radius_of_curvature` and `offset`, respectively. The output of these calculations was overlayed as a text annotation on the final output images.

#### 6. Lane Overlay Applied to Original 3D image

With all of the processing steps above completed all that is left to do is apply the calculated overlay to the original 3D image. As mentioned above, this step is really nothing more that applying the inverse perspective transform to the lane overlay calculated in the 2D image. An example of the final result is shown here:

![alt text][image9]


### Video Pipeline

#### 1. Pipeline Setup
The code for this step is contained in the Jupyter notebook  `run_model.ipynb` under the section titled "Build the Complete Lane Finding Pipeline". The two key functions here are `find_lanes` and `check_fit`. `find_lanes` encapsulates all of the pipeline steps described above to produce the final output image frames. `check_fit` is a "sanity check" function that we used to help the pipeline deal with cases where the lane lines were difficult to find. As mentioned above, lane lines were difficult to find in those image frames with bright asphalt or concrete and images with dark shadows. Instead of letting the pipeline produce erroneous lane lines we used the `check_fit` function to make sure the lanes that were found in each frame were valid. This function works by checking the coefficients of the proposed polynomial fit for the lane lines against the average of those same coefficients across a configurable number of previous image frames. If the the proposed fit deviates from the average beyond a certain tolerance, the proposed image lines are thrown out and the average is used instead. If the proposed fit does not deviate beyond the tolerance, the proposed fit is used for the lane lines and the average is updated. How many previous frames to look back and the tolerances on the coefficients were chosen empirically, and the final results helped the processing pipeline deal with problematic image frames quite well.
 
The output video can be found here: [Output Video](./output_video.mp4)


### Discussion

The most obvious problem with my pipeline is how woefully inefficient it is. Given enough time, however, this problem could probably be addressed with the most glaring inefficiencies contained within the code doing the sliding window search. There is much duplicate processing across the image space that could be condensed into a single pass in order to greatly improve performance.

Another obvious issue was gross inaccuracies in the radius of curvature and offset calculations. While the pipeline generates values that are most likely correct to an order of magnitude, it would seem that these values are not anywhere near reliable enough to effect any control actions to steer the vehicle. As mentioned above, I feel this is in large part to how the transform from the 3D images to 2D space is performed. It would seem that the linear distance per pixel would need to be configured exactly for the specific transform. Assumptions that were made regarding the negligible slope of the road were also probably somewhat incorrect resulting in inaccurate measurements.

Beyond these issues, the other main issue with this project, as with the original lane lines project, is dealing with obscured lane lines or "unfriendly" image frames where combinations of color and shadow push pure computer vision techniques past their point of applicability. It would seem that ultimately we would want to be able to use something beyond pure computer vision to solve the lane line problem. This could be perhaps a deep learning model or another classification model that can classify the likelihood of any given pixel in the image frame as being a lane line (or something else).
 
