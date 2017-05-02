import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_perspective_transform():
    """
    Retuns the perspective transform matrix and the inverse perspective transform matrix. Note that this function is
    specific to the images from the CarND-Advanced-Lane-Lines test images and the specific matrices were derived from
    the straight_lines1.jpg and straight_lines2.jpg images.
    :return: M and Minv. The perspective transform matrix and the inverse matrix, respectively
    """
    # The src and dst points were derived experimentally from the supplied test images
    src = np.float32([[265, 690], [1055, 690], [595, 450], [685, 450]])
    dst = np.float32([[265, 720], [1055, 720], [265, 0], [1055, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


def warp(img, M):
    """
    Given an image img, apply the perspective transform contained in the supplied matrix M
    :param img: Image to warp
    :param M: Perspective transform matrix
    :return: The warped image
    """
    return cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

if __name__ == '__main__':
    img = cv2.imread("test_images/straight_lines1.jpg")
    M, Minv = get_perspective_transform()
    img_warped = warp(img, M)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 10))
    ax1.set_title("Source")
    ax1.imshow(img)
    ax2.set_title("Warped")
    ax2.imshow(img_warped)

