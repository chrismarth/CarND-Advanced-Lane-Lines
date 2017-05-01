import cv2
import numpy as np
import matplotlib.pyplot as plt


def hls_threshold_s_img(image, lower=90, upper=255):
    """
    Given an RGB image, this function converts the images to the HLS color space and then applies the lower and upper 
    bounds to the S channel to return a binary image
    :param image: RGB image to threshold 
    :param lower: The lower bound of the S channel
    :param upper: The upper bound of the S channel
    :return: 
    """
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    binary = np.zeros_like(S)
    binary[(S > lower) & (S <= upper)] = 1
    return binary


def hls_threshold_l_img(image, lower=90, upper=255):
    """
    Given an RGB image, this function converts the images to the HLS color space and then applies the lower and upper 
    bounds to the L channel to return a binary image
    :param image: RGB image to threshold 
    :param lower: The lower bound of the L channel
    :param upper: The upper bound of the L channel
    :return: 
    """
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    L = hls[:,:,1]
    binary = np.zeros_like(L)
    binary[(L > lower) & (L <= upper)] = 1
    return binary


def sobel_threshold_x_img(img, lower=20, upper=100):
    """
    Given an RGB image, this function converts the images to the Gray color space and then applies the lower and upper 
    bounds to the Sobel Operator in the X direction to return a binary image 
    :param image: RGB image to threshold 
    :param lower: The lower bound of the Sobel X gradient
    :param upper: The upper bound of the Sobel X gradient
    :return: 
    """
    # TODO: Explore other color spaces besides Gray here
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= lower) & (scaled_sobel <= upper)] = 1
    return sxbinary


def sobel_threshold_y_img(img, lower=20, upper=100):
    """
    Given an RGB image, this function converts the images to the Gray color space and then applies the lower and upper 
    bounds to the Sobel Operator in the Y direction to return a binary image 
    :param image: RGB image to threshold 
    :param lower: The lower bound of the Sobel X gradient
    :param upper: The upper bound of the Sobel X gradient
    :return: 
    """
    # TODO: Explore other color spaces besides Gray here
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)  # Take the derivative in x
    abs_sobely = np.absolute(sobely)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobely / np.max(abs_sobely))

    # Threshold x gradient
    sybinary = np.zeros_like(scaled_sobel)
    sybinary[(scaled_sobel >= lower) & (scaled_sobel <= upper)] = 1
    return sybinary


def combine_binary(img1, img2):
    """
    Given two binary images, returns their combination 
    :param img1: First binary image
    :param img2: Second binary image 
    :return: 
    """
    combined_binary = np.zeros_like(img1)
    combined_binary[(img1 == 1) | (img2 == 1)] = 1
    return combined_binary


if __name__ == '__main__':
    image = cv2.cvtColor(cv2.imread("test_images/straight_lines1.jpg"), cv2.COLOR_BGR2RGB)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 10))
    ax1.set_title("Source")
    ax1.imshow(image)
    ax2.set_title("Threshold")
    ax2.imshow(hls_threshold_s_img(image), cmap='gray')
