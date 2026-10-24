"""LiDAR <-> camera fusion helpers.

* project_points / project_lidar_to_camera : Velodyne -> image pixels
* box_lidar_distance                       : robust LiDAR range inside a 2D box
* corridor_mask / build_path_mask          : image-space "ego path" masks
* in_path_lidar                            : LiDAR points whose ground footprint is on the path
* box_in_mask                              : is a tracked box standing on the path?
"""
import cv2
import numpy as np

import config
from calibration import Calibration, load_calibration

_DEFAULT_CALIB = None


def _calib(calib):
    global _DEFAULT_CALIB
    if calib is not None:
        return calib
    if _DEFAULT_CALIB is None:
        _DEFAULT_CALIB = load_calibration()
    return _DEFAULT_CALIB


def obstacle_points(scan):
    """Drop ego-vehicle and ground returns from an Nx4 scan; keep obstacles.

    A fixed "ground is at z = const" filter fails on slopes (embankments, hills),
    leaving terrain that looks like obstacles. Instead the ground height is estimated
    locally: the lowest return in each 1 m x 1 m cell, minimised over the 3x3
    neighbourhood (so a cell fully covered by a car body borrows its neighbours'
    ground). Points more than OBSTACLE_MIN_HEIGHT above that are obstacles.
    """
    near = scan[(scan[:, 0] > config.OBSTACLE_MIN_X) & (scan[:, 0] < config.GRID_MAX_X) &
                (np.abs(scan[:, 1]) < config.GRID_MAX_Y)]
    if len(near) == 0:
        return near
    ix = near[:, 0].astype(int)
    iy = (near[:, 1] + config.GRID_MAX_Y).astype(int)
    nx, ny = int(config.GRID_MAX_X) + 2, int(2 * config.GRID_MAX_Y) + 2

    cell_min = np.full((nx, ny), np.inf, dtype=np.float32)
    np.minimum.at(cell_min, (ix, iy), near[:, 2])
    # a sub-road outlier must not lower the local ground by more than 0.3 m
    cell_min = np.where(np.isfinite(cell_min), np.maximum(cell_min, config.GROUND_Z - 0.3), np.inf)

    padded = np.pad(cell_min, 1, constant_values=np.inf)
    local = np.min([padded[dx:dx + nx, dy:dy + ny] for dx in range(3) for dy in range(3)], axis=0)
    height = near[:, 2] - local[ix, iy]
    return near[height > config.OBSTACLE_MIN_HEIGHT]


def project_points(xyz, image_shape, calib=None):
    """Project Nx3 Velodyne points to pixels.

    Returns (uv Nx2 float, valid N bool). ``valid`` is True for points in front
    of the camera and inside the image.
    """
    calib = _calib(calib)
    n = len(xyz)
    if n == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=bool)

    pts = np.column_stack((xyz[:, :3], np.ones(n)))
    pts_cam = pts @ calib.Tr_velo_to_cam.T
    pts_cam = pts_cam @ calib.R_rect.T
    proj = pts_cam @ calib.P_rect.T

    in_front = proj[:, 2] > 0.1
    z = np.where(in_front, proj[:, 2], 1.0)
    uv = np.column_stack((proj[:, 0] / z, proj[:, 1] / z))

    h, w = image_shape[:2]
    valid = in_front & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv, valid


def project_lidar_to_camera(points, image_shape, calib=None):
    """Backwards-compatible helper: (pixels, forward distance) of visible points."""
    points = points[points[:, 0] > 0]
    uv, valid = project_points(points, image_shape, calib)
    return uv[valid], points[valid, 0]


def box_lidar_distance(uv, depths, box):
    """Robust forward distance of the object inside ``box`` from projected LiDAR.

    Only the central part of the box is used (box edges are mostly background),
    and the nearest surface wins so road/buildings behind the object are ignored.
    Returns None when there is not enough LiDAR support.
    """
    x1, y1, x2, y2 = box
    sx = (x2 - x1) * config.BOX_SHRINK
    sy = (y2 - y1) * config.BOX_SHRINK
    inside = ((uv[:, 0] >= x1 + sx) & (uv[:, 0] <= x2 - sx) &
              (uv[:, 1] >= y1 + sy) & (uv[:, 1] <= y2 - sy))
    d = depths[inside]
    if len(d) < config.BOX_MIN_POINTS:
        return None
    near = np.percentile(d, config.BOX_NEAR_PERCENTILE)
    return float(np.median(d[d <= near + config.BOX_NEAR_BAND]))


def corridor_centerline(xs, curvature=0.0):
    """Lateral position (m, + = left) of a constant-curvature path at ranges xs."""
    return 0.5 * curvature * np.asarray(xs) ** 2


def corridor_mask(image_shape, calib=None, half_width=None, curvature=0.0):
    """Binary image mask of the ground corridor ahead (straight, or curved by 1/m)."""
    half_width = half_width or config.FALLBACK_CORRIDOR_HALF_WIDTH
    xs = np.linspace(config.CORRIDOR_X_RANGE[0], config.CORRIDOR_X_RANGE[1], 80)
    yc = corridor_centerline(xs, curvature)
    ground = np.full_like(xs, config.GROUND_Z)
    left = np.column_stack((xs, yc + half_width, ground))
    right = np.column_stack((xs, yc - half_width, ground))
    ring = np.vstack((left, right[::-1]))

    uv, _ = project_points(ring, image_shape, calib)
    # The polygon may legally extend past the frame, so the in-image flag is ignored
    # (every point is ahead of the camera, x >= 2 m).
    uv = np.clip(uv, -5000, 5000).astype(np.int32)
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [uv], 1)
    return mask


def corridor_edge_polylines(image_shape, calib=None, half_width=None, curvature=0.0):
    """Left/right edges of the corridor as pixel polylines (for drawing)."""
    half_width = half_width or config.FALLBACK_CORRIDOR_HALF_WIDTH
    xs = np.linspace(config.CORRIDOR_X_RANGE[0], config.CORRIDOR_X_RANGE[1], 80)
    yc = corridor_centerline(xs, curvature)
    h = image_shape[0]
    edges = []
    for sign in (1.0, -1.0):
        pts = np.column_stack((xs, yc + sign * half_width, np.full_like(xs, config.GROUND_Z)))
        uv, _ = project_points(pts, image_shape, calib)
        uv = np.clip(uv, -5000, 5000)
        uv = uv[uv[:, 1] > 0]          # keep from the horizon down
        uv[:, 1] = np.minimum(uv[:, 1], h + 50)
        edges.append(uv.astype(np.int32))
    return edges


def build_path_mask(image_shape, lane_mask=None, calib=None, dilate_px=9, curvature=0.0):
    """Ego path = detected lane UNION the fixed corridor.

    * The detected lane follows curves and handles wide lanes, but lane detection
      can be wrong (no markings, curbs mistaken for lines) and only reaches the
      top of the bird's-eye-view ROI (~16 m).
    * The straight corridor (vehicle width + margin) is always a real hazard zone
      and covers far range.
    The union means a bad lane fit can never hide something directly ahead.
    Returns (mask uint8 {0,1}, source str: 'lane' or 'corridor').
    """
    path = corridor_mask(image_shape, calib, curvature=curvature)
    source = 'corridor'
    if lane_mask is not None and lane_mask.any():
        path = np.maximum(path, lane_mask)
        source = 'lane'
    if dilate_px > 1:
        path = cv2.dilate(path, np.ones((dilate_px, dilate_px), np.uint8))
    return path, source


def in_path_lidar(xyz, image_shape, path_mask, calib=None, curvature=0.0):
    """Boolean mask over ``xyz``: True if the point's ground footprint is on the path.

    Using the footprint (same x/y, z = ground) instead of the point's own pixel
    means a car's roof counts if the car stands in the lane, regardless of how
    tall it is. Close points must also satisfy a lateral limit (PATH_MAX_LATERAL).
    """
    if len(xyz) == 0:
        return np.zeros(0, dtype=bool)
    foot = xyz[:, :3].copy()
    foot[:, 2] = config.GROUND_Z
    uv, valid = project_points(foot, image_shape, calib)
    out = np.zeros(len(xyz), dtype=bool)
    if valid.any():
        u = uv[valid, 0].astype(int)
        v = uv[valid, 1].astype(int)
        out[valid] = path_mask[v, u] > 0
    # A wide lane polygon (bike lane, shoulder) must not pull kerbs and verges into
    # the path: nearby points also need to be within vehicle width + margin.
    near = xyz[:, 0] < config.PATH_LATERAL_RANGE
    lateral = np.abs(xyz[:, 1] - corridor_centerline(xyz[:, 0], curvature))
    out &= ~near | (lateral < config.PATH_MAX_LATERAL)
    return out


def in_path_distance(path_x):
    """Distance of the nearest *supported* in-path obstacle, or 99 if none.

    Needs LIDAR_MIN_CLUSTER points within LIDAR_CLUSTER_BAND metres of depth, so
    stray near-field returns (noise, foliage, reflections) can't trigger braking
    while real obstacles (cars, people: dozens of points) pass easily.
    """
    k = config.LIDAR_MIN_CLUSTER
    x = np.sort(np.asarray(path_x))
    if len(x) < k:
        return 99.0
    close = x[k - 1:] - x[:len(x) - k + 1] <= config.LIDAR_CLUSTER_BAND
    return float(x[:len(x) - k + 1][close][0]) if close.any() else 99.0


def box_in_mask(path_mask, box, min_fraction=0.3):
    """True if the central span of a box's bottom edge lies on the path mask."""
    h, w = path_mask.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    y = int(np.clip(y2 - 1, 0, h - 1))
    cx, half = (x1 + x2) / 2.0, (x2 - x1) / 4.0
    xs = np.clip(np.arange(int(cx - half), int(cx + half) + 1), 0, w - 1)
    if len(xs) == 0:
        return False
    return float(np.mean(path_mask[y, xs] > 0)) >= min_fraction
