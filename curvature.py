import numpy as np

# These were adjusted empirically since they will be somewhat dependent on how the image is warped
ym_per_pix = 40/720   # meters per pixel in y dimension
xm_per_pix = 3.0/700  # meters per pixel in x dimension


def radius_of_curvature(left_fit, right_fit):
    """
    This calculation assumes that the projected, bird's-eye-view lane image is about 30 meters long and 3.7 meters wide
    :param left_fit: Polynomial fit for the left lane line 
    :param right_fit: Polynomial fit for the right lane line
    :return: tuple of radius of curvature (left, right)
    """
    y_eval = 719
    left_curverad = ((1 + (2 * left_fit[0] * y_eval*ym_per_pix + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval*ym_per_pix + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
    return left_curverad, right_curverad


def offset(left_fit, right_fit):
    """
    Given the polynomial fits for the left and right lane lines, returns the vehicle offset from the center of the image
    :param left_fit: Polynomial fit for the left lane line 
    :param right_fit: Polynomial fit for the right lane line
    :return: offset from the center of the image in (m)
    """
    y_eval = 719
    width = 1280
    left_fitx = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
    right_fitx = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
    center = int((left_fitx + right_fitx)/2)
    offset = abs(int(width/2) - center)
    return offset*xm_per_pix


def format_curvature(left_curve, right_curve):
    """
    Returns a formatted string indicating the average radius of curvature between the left and right lane lines
    :param left_curve: radius of curvature for left lane line 
    :param right_curve: radius of curvature for right lane line
    :return: formatted string indicating the average radius of curvature between the left and right lane lines
    """
    return str(round((left_curve + right_curve) / 2)) + " (m)"


def format_offset(offset):
    """
    Returns a formatted string indicating the offset left or right from the center of the lane
    :param offset: offset from center
    :return: formatted string indicating the offset left or right from the center of the lane
    """
    if offset <= 0:
        return str(round(offset, 3)) + " (m) Left"
    else:
        return str(round(offset, 3)) + " (m) Right"
