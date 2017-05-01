import numpy as np

ym_per_pix = 30/720   # meters per pixel in y dimension
xm_per_pix = 3.7/700  # meters per pixel in x dimension

def radius_of_curvature(left_fit, right_fit):
    """
    This calculation assumes that the projected, bird's-eye-view lane image is about 30 meters long and 3.7 meters wide
    :param left_fit: Polynomial fit for the left lane line 
    :param right_fit: Polynomial fit for the right lane line
    :return: 
    """
    y_eval = 719
    left_curverad = ((1 + (2 * left_fit[0] * y_eval*ym_per_pix + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval*ym_per_pix + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
    return left_curverad, right_curverad


# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)