"""Previous image-space lane detector (hand-picked trapezoid bird's-eye view, fits in
warped pixels). Kept unchanged as the baseline that eval_lane.py compares the
calibrated metric detector in lane_utils.py against."""
import cv2
import numpy as np

import config

# --- Colour / brightness segmentation -----------------------------------------

def lane_color_mask(image):
    """Binary (0/255) mask of white and yellow lane-marking candidates.

    * white  : bright pixels, plus bright *narrow* stripes found with a top-hat
               filter so markings in shadow (lower absolute lightness) still pass.
    * yellow : hue/saturation window in HLS.
    Everything above the horizon region is discarded.
    """
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    hue, light, sat = cv2.split(hls)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    tophat = cv2.morphologyEx(light, cv2.MORPH_TOPHAT, kernel)
    white = (light >= 200) | ((tophat >= 35) & (light >= 120))
    yellow = (hue >= 15) & (hue <= 40) & (sat >= 80) & (light >= 80)

    mask = ((white | yellow).astype(np.uint8)) * 255
    mask[:int(image.shape[0] * 0.55), :] = 0
    return mask


# --- Bird's-eye view -----------------------------------------------------------

def get_perspective_matrices(image):
    height, width = image.shape[:2]
    src = np.float32([[width*0.1, height*0.95], [width*0.9, height*0.95],
                      [width*0.6, height*0.65], [width*0.4, height*0.65]])
    dst = np.float32([[width*0.2, height], [width*0.8, height],
                      [width*0.8, 0], [width*0.2, 0]])
    return cv2.getPerspectiveTransform(src, dst), cv2.getPerspectiveTransform(dst, src)


# --- Sliding window search + polynomial fit --------------------------------------

def find_lane_pixels(warped):
    """Sliding-window search. Returns (left_inds, right_inds, nonzerox, nonzeroy)
    or None when the bird's-eye mask has no usable base on either side."""
    histogram = np.sum(warped[warped.shape[0]//2:, :], axis=0)
    midpoint = int(histogram.shape[0]//2)
    if histogram[:midpoint].max() == 0 or histogram[midpoint:].max() == 0:
        return None
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


def fit_lane(warped):
    """Fit left/right quadratics x = f(y) in bird's-eye space.

    Returns (left_fit, right_fit) or None if the evidence is too weak or the
    geometry is implausible (crossing lines, absurd lane width).
    """
    found = find_lane_pixels(warped)
    if found is None:
        return None
    l_inds, r_inds, nx, ny = found
    if len(l_inds) < config.LANE_MIN_PIXELS or len(r_inds) < config.LANE_MIN_PIXELS:
        return None
    try:
        left_fit = np.polyfit(ny[l_inds], nx[l_inds], 2)
        right_fit = np.polyfit(ny[r_inds], nx[r_inds], 2)
    except (np.linalg.LinAlgError, ValueError, TypeError):
        return None

    h, w = warped.shape[:2]
    ys = np.linspace(0, h - 1, 10)
    width = np.polyval(right_fit, ys) - np.polyval(left_fit, ys)
    if (width.min() < config.LANE_MIN_WIDTH_FRAC * w) or (width.max() > config.LANE_MAX_WIDTH_FRAC * w):
        return None

    # The ego vehicle must be inside its own lane: at the bottom of the bird's-eye
    # view the vehicle sits at the image centre. This rejects curbs / sidewalk
    # edges that happen to look like a lane pair.
    tol = config.LANE_EGO_TOLERANCE_FRAC * w
    left_bottom, right_bottom = np.polyval(left_fit, h - 1), np.polyval(right_fit, h - 1)
    if not (left_bottom - tol < w / 2 < right_bottom + tol):
        return None

    # ... roughly centred in it, and with roughly parallel edges (a real lane
    # does not change width by more than ~60% over the visible stretch).
    bottom_width = right_bottom - left_bottom
    offset = abs((left_bottom + right_bottom) / 2 - w / 2)
    if offset > config.LANE_MAX_OFFSET_FRAC * bottom_width:
        return None
    if width.max() > config.LANE_MAX_WIDTH_RATIO * width.min():
        return None
    return left_fit, right_fit


class LaneTracker:
    """Temporal smoothing and outlier rejection around ``fit_lane``.

    * EMA over the polynomial coefficients.
    * A fit whose base position jumps too far is treated as an outlier.
    * The previous fit is carried for up to LANE_MAX_MISSING frames, then dropped
      (``update`` then returns ``(None, 'none')`` and the caller falls back).
    """

    def __init__(self):
        self.fit = None
        self.missing = 0

    def update(self, warped):
        h, w = warped.shape[:2]
        new = fit_lane(warped)
        fresh = False

        if new is not None:
            if self.fit is None:
                self.fit, self.missing, fresh = new, 0, True
            else:
                jump = max(abs(np.polyval(new[0], h) - np.polyval(self.fit[0], h)),
                           abs(np.polyval(new[1], h) - np.polyval(self.fit[1], h)))
                if jump <= config.LANE_MAX_JUMP_FRAC * w:
                    a = config.LANE_SMOOTH_ALPHA
                    self.fit = tuple((1 - a) * old + a * nw for old, nw in zip(self.fit, new))
                    self.missing, fresh = 0, True
                else:
                    self.missing += 1
        else:
            self.missing += 1

        if self.fit is not None and self.missing > config.LANE_MAX_MISSING:
            self.fit = None
            self.missing = 0
        if self.fit is None:
            return None, 'none'
        return self.fit, ('detected' if fresh else 'coasting')


def lane_mask_from_fit(image_shape, warped_shape, fit, Minv):
    """Polygon between the two fits, warped back into the camera image.
    Returns a uint8 {0,1} mask the size of the camera image."""
    h = warped_shape[0]
    ploty = np.linspace(0, h - 1, h)
    left_fitx = np.polyval(fit[0], ploty)
    right_fitx = np.polyval(fit[1], ploty)

    color_warp = np.zeros(warped_shape[:2], dtype=np.uint8)
    pts_left = np.transpose(np.vstack([left_fitx, ploty]))
    pts_right = np.flipud(np.transpose(np.vstack([right_fitx, ploty])))
    pts = np.vstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, [np.int32(pts)], 1)

    back = cv2.warpPerspective(color_warp, Minv, (image_shape[1], image_shape[0]),
                               flags=cv2.INTER_NEAREST)
    return (back > 0).astype(np.uint8)


def lane_edge_polylines(warped_shape, fit, Minv):
    """Left and right lane edges as Nx2 int32 pixel polylines in the camera image."""
    h = warped_shape[0]
    ploty = np.linspace(0, h - 1, 40)
    edges = []
    for coeffs in fit:
        pts = np.column_stack((np.polyval(coeffs, ploty), ploty)).astype(np.float32)
        img_pts = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), Minv).reshape(-1, 2)
        edges.append(img_pts.astype(np.int32))
    return edges


def draw_polylines(image, polylines, color, thickness=3):
    for pts in polylines:
        if len(pts) > 1:
            cv2.polylines(image, [pts.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)
    return image


def draw_mask_overlay(image, mask, color=(0, 255, 0), alpha=0.3):
    """Blend a coloured region (mask > 0) into the image."""
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
