import cv2
import numpy as np

def get_perspective_matrices(image):
    height, width = image.shape[:2]
    src = np.float32([[width*0.1, height*0.95], [width*0.9, height*0.95], 
                      [width*0.6, height*0.65], [width*0.4, height*0.65]])
    dst = np.float32([[width*0.2, height], [width*0.8, height], 
                      [width*0.8, 0], [width*0.2, 0]])
    return cv2.getPerspectiveTransform(src, dst), cv2.getPerspectiveTransform(dst, src)

def find_lane_pixels(warped):
    # Sliding Window Search
    histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)
    midpoint = int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows, margin = 9, 50
    window_height = int(warped.shape[0]//nwindows)
    nonzero = warped.nonzero()
    nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
    
    left_lane_inds, right_lane_inds = [], []
    lx, rx = leftx_base, rightx_base

    for window in range(nwindows):
        y_low, y_high = warped.shape[0]-(window+1)*window_height, warped.shape[0]-window*window_height
        l_inds = ((nonzeroy >= y_low) & (nonzeroy < y_high) & (nonzerox >= lx-margin) & (nonzerox < lx+margin)).nonzero()[0]
        r_inds = ((nonzeroy >= y_low) & (nonzeroy < y_high) & (nonzerox >= rx-margin) & (nonzerox < rx+margin)).nonzero()[0]
        left_lane_inds.append(l_inds)
        right_lane_inds.append(r_inds)
        if len(l_inds) > 50: lx = int(np.mean(nonzerox[l_inds]))
        if len(r_inds) > 50: rx = int(np.mean(nonzerox[r_inds]))

    return np.concatenate(left_lane_inds), np.concatenate(right_lane_inds), nonzerox, nonzeroy

def draw_lane_overlay(image, warped, left_lane_inds, right_lane_inds, nonzerox, nonzeroy, Minv):
    # 1. Fit the polynomials
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
    left_fit = np.polyfit(nonzeroy[left_lane_inds], nonzerox[left_lane_inds], 2)
    right_fit = np.polyfit(nonzeroy[right_lane_inds], nonzerox[right_lane_inds], 2)
    
    # 2. Generate x and y values for plotting
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # 3. Create a blank image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = cv2.merge((warp_zero, warp_zero, warp_zero))

    # 4. Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # 5. Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # 6. Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    
    # 7. Combine the result with the original image
    return cv2.addWeighted(image, 1, newwarp, 0.3, 0)