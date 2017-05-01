import numpy as np
import cv2
import curvature


def fit_polynomial(image, window_width=50, window_height=80, margin=100, minpix=50):
    """
    TODO: Combine this with find_window_centroids
    
    :param image: 
    :param window_width: 
    :param window_height: 
    :param margin: 
    :return: 
    """
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    centroids = find_window_centroids(image, window_width, window_height, margin)
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Step through the windows one by one
    for window_i in range(len(centroids)):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = image.shape[0] - (window_i + 1) * window_height
        win_y_high = image.shape[0] - window_i * window_height
        win_xleft_low = centroids[window_i][0] - window_width
        win_xleft_high = centroids[window_i][0] + window_width
        win_xright_low = centroids[window_i][1] - window_width
        win_xright_high = centroids[window_i][1] + window_width
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each - We do this in both pixel and real-world space since we need both
    left_fit_px = np.polyfit(lefty, leftx, 2)
    right_fit_px = np.polyfit(righty, rightx, 2)
    left_fit_m = np.polyfit(lefty*curvature.ym_per_pix, leftx*curvature.xm_per_pix, 2)
    right_fit_m = np.polyfit(righty*curvature.ym_per_pix, rightx*curvature.xm_per_pix, 2)

    return left_fit_px, right_fit_px, left_fit_m, right_fit_m


def find_window_centroids(image, window_width=50, window_height=80, margin=100):
    """
    
    :param image: 
    :param window_width: 
    :param window_height: 
    :param margin: 
    :return: 
    """
    window_centroids = []
    window = np.ones(window_width)

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    r_sum = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(image.shape[1] / 2)

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(image.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(image[int(image.shape[0] - (level + 1) * window_height):int(image.shape[0] - level * window_height), :], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        # Add what we found for that layer
        window_centroids.append((l_center, r_center))

    return window_centroids


def window_mask(img_ref, center, level, width=50, height=80):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output


def draw_mask(image, centroids):
    # If we found any window centers
    if len(centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(image)
        r_points = np.zeros_like(image)

        # Go through each level and draw the windows
        for level in range(0, len(centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(image, centroids[level][0], level)
            r_mask = window_mask(image, centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
        zero_channel = np.zeros_like(template)  # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
        warpage = np.array(cv2.merge((image, image, image)), np.uint8)  # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results

    # If no window centers found, just display original road image
    else:
        output = np.array(cv2.merge((image, image, image)), np.uint8)

    return output


def draw_lane_lines(image, left_fit, right_fit):
    out_img = np.dstack((image, image, image)) * 255
    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    poly_pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(out_img, np.int_([poly_pts]), (0, 255, 0))
    cv2.polylines(out_img, np.int_([pts_left]), 0, (255, 0, 0), thickness=40)
    cv2.polylines(out_img, np.int_([pts_right]), 0, (0, 0, 255), thickness=40)

    return out_img
