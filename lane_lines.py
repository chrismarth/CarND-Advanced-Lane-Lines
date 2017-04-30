import transform
import threshold
import cv2
import matplotlib.pyplot as plt

# Warp and threshold the image
img = cv2.cvtColor(cv2.imread("test_images/straight_lines1.jpg"), cv2.COLOR_BGR2RGB)
M, Minv = transform.get_perspective_transform()
img_warped = transform.warp(img, M)
img_hls_thresh = threshold.hls_threshold_s_img(img_warped)
img_sobel_thresh = threshold.sobel_threshold_x_img(img_warped)
img_thresh = threshold.combine_binary(img_hls_thresh, img_sobel_thresh)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 10))
ax1.set_title("Source")
ax1.imshow(img)
ax2.set_title("Warped and Threshold")
ax2.imshow(img_thresh, cmap='gray')