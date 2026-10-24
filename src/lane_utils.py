"""Lane detection in a calibrated, metric bird's-eye view (BEV).

The BEV is a ground-plane homography built from the camera calibration (not
hand-picked image fractions): one BEV pixel = 1/scale metres, BEV axes are the
vehicle's (x forward, y left) in metres. Lane lines are fitted as
``y = c2*x^2 + c1*x + c0`` *in metres*, so lane width, ego position and curvature
can be checked against real-world limits (a 3.5 m lane, not "30% of the image").

The previous image-space detector is kept in ``lane_legacy.py`` as an eval baseline.
"""
from dataclasses import dataclass

import cv2
import numpy as np

import config
import fusion_utils as fs


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


# --- Metric bird's-eye view ---------------------------------------------------------

@dataclass
class Bev:
    M: np.ndarray           # 3x3 image -> BEV pixels
    Minv: np.ndarray        # 3x3 BEV pixels -> image
    size: tuple             # (width, height) of the BEV image
    scale: float            # BEV pixels per metre
    x_near: float           # m, bottom edge of the BEV (closest ground visible)
    x_far: float            # m, top edge
    half_width: float       # m, lateral half extent

    def to_meters(self, u, v):
        """BEV pixel (u, v) -> ground (x forward, y left) in metres."""
        return self.x_far - np.asarray(v) / self.scale, self.half_width - np.asarray(u) / self.scale

    def to_pixels(self, x, y):
        """Ground (x, y) in metres -> BEV pixel (u, v)."""
        return (self.half_width - np.asarray(y)) * self.scale, (self.x_far - np.asarray(x)) * self.scale

    def image_to_ground(self, uv):
        """Image pixels (N, 2) -> ground (x, y) metres via the BEV homography."""
        pts = cv2.perspectiveTransform(np.asarray(uv, np.float32).reshape(-1, 1, 2), self.M).reshape(-1, 2)
        return self.to_meters(pts[:, 0], pts[:, 1])


def make_bev(calib, image_shape, x_near=None, x_far=None, half_width=None, scale=None):
    """Ground-plane BEV from the calibration (flat ground at z = GROUND_Z)."""
    x_near = x_near or config.BEV_X_NEAR
    x_far = x_far or config.BEV_X_FAR
    half_width = half_width or config.BEV_HALF_WIDTH
    scale = scale or config.BEV_SCALE

    corners = np.array([[x_near, half_width], [x_near, -half_width],
                        [x_far, half_width], [x_far, -half_width]], dtype=np.float64)
    ground = np.column_stack((corners, np.full(4, config.GROUND_Z)))
    uv, _ = fs.project_points(ground, image_shape, calib)       # raw pixels (may lie outside the frame)

    bev = Bev(np.eye(3), np.eye(3), (int(2 * half_width * scale), int((x_far - x_near) * scale)),
              scale, x_near, x_far, half_width)
    dst = np.column_stack(bev.to_pixels(corners[:, 0], corners[:, 1])).astype(np.float32)
    bev.M = cv2.getPerspectiveTransform(uv.astype(np.float32), dst)
    bev.Minv = cv2.getPerspectiveTransform(dst, uv.astype(np.float32))
    return bev


# --- Lane fit (metres) ----------------------------------------------------------------

@dataclass
class LaneFit:
    left: np.ndarray        # polyval coefficients: y_left(x)
    right: np.ndarray
    x_start: float
    x_end: float

    def y_left(self, x):
        return np.polyval(self.left, x)

    def y_right(self, x):
        return np.polyval(self.right, x)

    def center(self, x):
        return 0.5 * (self.y_left(x) + self.y_right(x))

    def width(self, x):
        return self.y_left(x) - self.y_right(x)

    def blend(self, new, alpha):
        return LaneFit((1 - alpha) * self.left + alpha * new.left,
                       (1 - alpha) * self.right + alpha * new.right,
                       (1 - alpha) * self.x_start + alpha * new.x_start,
                       (1 - alpha) * self.x_end + alpha * new.x_end)


def _side_points(x, y, bev, band, prior_poly):
    """Sliding windows along x for one lane line; returns (x, y) of its pixels (metres)."""
    nwin = config.LANE_M_WINDOWS
    edges = np.linspace(bev.x_near, bev.x_far, nwin + 1)
    margin, minpix = config.LANE_M_MARGIN, config.LANE_M_MINPIX
    lo, hi = band

    if prior_poly is not None:
        center_of = lambda w, cur: float(np.polyval(prior_poly, 0.5 * (edges[w] + edges[w + 1])))
        cur = None
    else:
        near = (x < bev.x_near + config.LANE_M_BASE_RANGE) & (y > lo) & (y < hi)
        if near.sum() < minpix:
            return None
        hist, bins = np.histogram(y[near], bins=np.arange(lo, hi + 0.05, 0.05))
        cur = float(bins[np.argmax(hist)] + 0.025)
        center_of = lambda w, c: c

    xs, ys = [], []
    for w in range(nwin):
        c = center_of(w, cur)
        sel = (x >= edges[w]) & (x < edges[w + 1]) & (np.abs(y - c) < margin)
        if sel.sum() >= minpix:
            xs.append(x[sel]); ys.append(y[sel])
            if prior_poly is None:
                cur = float(np.mean(y[sel]))
    if not xs:
        return None
    return np.concatenate(xs), np.concatenate(ys)


def _fit_side(pts):
    """Quadratic y(x) for one side, or None if support is too thin / too short."""
    if pts is None or len(pts[0]) < config.LANE_M_MIN_POINTS:
        return None
    x, y = pts
    if x.max() - x.min() < config.LANE_M_MIN_SPAN:
        return None
    try:
        coeffs = np.polyfit(x, y, 2)
    except (np.linalg.LinAlgError, ValueError, TypeError):
        return None
    if abs(coeffs[0]) > config.LANE_M_MAX_C2:
        return None
    return coeffs, float(x.max())


def _geometry_ok(left, right, x_start, x_end):
    """Real-world plausibility: lane width, ego inside / centred, parallel edges."""
    xs = np.linspace(x_start, x_end, 10)
    width = np.polyval(left, xs) - np.polyval(right, xs)
    if width.min() < config.LANE_M_MIN_WIDTH or width.max() > config.LANE_M_MAX_WIDTH:
        return False
    if width.max() > config.LANE_M_MAX_WIDTH_RATIO * width.min():
        return False
    yl, yr = np.polyval(left, x_start), np.polyval(right, x_start)
    tol = config.LANE_M_EGO_TOL
    if not (yl > -tol and yr < tol):                       # ego inside its lane
        return False
    return abs(0.5 * (yl + yr)) <= config.LANE_M_MAX_CENTER_OFFSET


def fit_lane(warped, bev, prior=None, single_width=None):
    """Fit a metric lane from a BEV marking mask.

    Returns (LaneFit, 'detected' | 'single') or (None, None). 'single' means only
    one line had support and the other was inferred one lane-width away.
    """
    ys_px, xs_px = warped.nonzero()
    if len(ys_px) == 0:
        return None, None
    x, y = bev.to_meters(xs_px, ys_px)

    left_pts = _side_points(x, y, bev, (config.LANE_M_MIN_OFFSET, config.LANE_M_MAX_OFFSET),
                            None if prior is None else prior.left)
    right_pts = _side_points(x, y, bev, (-config.LANE_M_MAX_OFFSET, -config.LANE_M_MIN_OFFSET),
                             None if prior is None else prior.right)
    left, right = _fit_side(left_pts), _fit_side(right_pts)
    x_start = bev.x_near

    if left and right:
        x_end = min(left[1], right[1]) + config.LANE_M_EXTRAPOLATE
        if x_end - x_start >= config.LANE_M_MIN_SPAN and _geometry_ok(left[0], right[0], x_start, x_end):
            return LaneFit(left[0], right[0], x_start, min(x_end, bev.x_far)), 'detected'

    if config.LANE_M_SINGLE_LINE and (bool(left) != bool(right)):
        w = single_width or config.LANE_M_DEFAULT_WIDTH
        x_end = (left or right)[1] + config.LANE_M_EXTRAPOLATE
        if left:
            fit_l, fit_r = left[0], left[0] - np.array([0.0, 0.0, w])
        else:
            fit_r, fit_l = right[0], right[0] + np.array([0.0, 0.0, w])
        if x_end - x_start >= config.LANE_M_MIN_SPAN and _geometry_ok(fit_l, fit_r, x_start, x_end):
            return LaneFit(fit_l, fit_r, x_start, min(x_end, bev.x_far)), 'single'
    return None, None


class LaneTracker:
    """Temporal smoothing and outlier rejection around ``fit_lane``.

    * Windows follow the previous fit (robust against distractors); if that fails
      a fresh search is tried.
    * EMA over the coefficients; a fit whose lateral position jumps by more than
      LANE_M_MAX_JUMP metres between frames is treated as an outlier.
    * The previous fit is carried for up to LANE_MAX_MISSING frames ('coasting'),
      then dropped (``update`` returns ``(None, 'none')``).
    """

    def __init__(self, bev):
        self.bev = bev
        self.fit = None
        self.missing = 0
        self.width = None          # EMA of the measured lane width (m), full detections only

    def update(self, warped):
        new, kind = fit_lane(warped, self.bev, self.fit, self.width)
        if new is None and self.fit is not None:
            new, kind = fit_lane(warped, self.bev, None, self.width)
        fresh = None

        if new is not None:
            if self.fit is None:
                self.fit, self.missing, fresh = new, 0, kind
            else:
                xr = self.bev.x_near + 3.0
                jump = max(abs(new.y_left(xr) - self.fit.y_left(xr)),
                           abs(new.y_right(xr) - self.fit.y_right(xr)))
                if jump <= config.LANE_M_MAX_JUMP:
                    self.fit = self.fit.blend(new, config.LANE_SMOOTH_ALPHA)
                    self.missing, fresh = 0, kind
                else:
                    self.missing += 1
            if fresh == 'detected':
                w = float(new.width(self.bev.x_near + 3.0))
                self.width = w if self.width is None else 0.9 * self.width + 0.1 * w
        else:
            self.missing += 1

        if self.fit is not None and self.missing > config.LANE_MAX_MISSING:
            self.fit, self.missing = None, 0
        if self.fit is None:
            return None, 'none'
        return self.fit, (fresh or 'coasting')


def consistent_with_path(fit, curvature, tol=None):
    """Is the lane where the vehicle is heading?

    Marking-based detection also fires on crosswalks, intersection markings, road
    edges... The ego-motion path (constant curvature from the yaw rate) is an
    independent, causal estimate of where the car is going; a lane whose centre
    disagrees with it near the vehicle is rejected.
    """
    tol = tol or config.LANE_M_PATH_TOL
    xs = np.linspace(fit.x_start, min(fit.x_end, fit.x_start + config.LANE_M_GATE_RANGE), 6)
    return bool(np.max(np.abs(fit.center(xs) - fs.corridor_centerline(xs, curvature))) <= tol)


# --- Masks / drawing ------------------------------------------------------------------

def _bev_polylines(bev, fit, n=40):
    xs = np.linspace(fit.x_start, fit.x_end, n)
    left = np.column_stack(bev.to_pixels(xs, fit.y_left(xs)))
    right = np.column_stack(bev.to_pixels(xs, fit.y_right(xs)))
    return left.astype(np.float32), right.astype(np.float32)


def lane_mask_from_fit(image_shape, bev, fit):
    """Polygon between the lane lines warped back into the camera image.
    Returns a uint8 {0, 1} mask the size of the camera image."""
    left, right = _bev_polylines(bev, fit)
    canvas = np.zeros((bev.size[1], bev.size[0]), dtype=np.uint8)
    cv2.fillPoly(canvas, [np.int32(np.vstack((left, right[::-1])))], 1)
    back = cv2.warpPerspective(canvas, bev.Minv, (image_shape[1], image_shape[0]), flags=cv2.INTER_NEAREST)
    return (back > 0).astype(np.uint8)


def lane_edge_polylines(bev, fit):
    """Left and right lane edges as Nx2 int32 pixel polylines in the camera image."""
    edges = []
    for pts in _bev_polylines(bev, fit):
        img_pts = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), bev.Minv).reshape(-1, 2)
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
