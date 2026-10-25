"""Short-horizon motion prediction for tracked objects and ego-path conflict checks.

Objects are predicted in the world frame from the Kalman state of ``Track3D``:

  cv   : constant velocity
  ctrv : constant turn rate and velocity (turn rate from the smoothed heading change)
  static: zero velocity (baseline)

``time_to_conflict`` rolls the ego car forward on a constant-curvature arc at its
current speed and returns the first time its footprint would overlap the object's.
That sees a vehicle cutting in *before* it is inside the ego corridor, which the
in-path test (current position only) cannot.
"""
import math

import numpy as np

import config

EGO_HALF_WIDTH = 1.0        # m
EGO_REAR = 4.0              # m behind the reference point (the ego frame origin is the sensor)
BUMPER_MARGIN = 1.0         # m, conflict once the gap to the object is below this
LATERAL_MARGIN = 0.3        # m
MODELS = ('static', 'cv', 'ctrv')


def predict_xy(track, ts, model='cv'):
    """World-frame positions (len(ts), 2) of ``track`` at times ``ts`` (s, from now)."""
    ts = np.asarray(ts, float)
    x0, y0, vx, vy = track.state
    if model == 'static':
        return np.tile([x0, y0], (len(ts), 1))
    w = track.heading_rate if model == 'ctrv' else 0.0
    if abs(w) < 1e-3:
        return np.column_stack((x0 + vx * ts, y0 + vy * ts))
    # exact integration of a constant-turn-rate motion
    th0 = math.atan2(vy, vx)
    sp = math.hypot(vx, vy)
    return np.column_stack((x0 + sp / w * (np.sin(th0 + w * ts) - math.sin(th0)),
                            y0 - sp / w * (np.cos(th0 + w * ts) - math.cos(th0))))


def ego_future(pose_ego_world, speed, curvature, ts):
    """Ego position (len(ts), 2) and heading (len(ts),) in the world, on a constant-curvature arc."""
    ts = np.asarray(ts, float)
    s = max(speed, 0.0) * ts
    if abs(curvature) < 1e-6:
        lx, ly, lth = s, np.zeros_like(s), np.zeros_like(s)
    else:
        lx, ly, lth = np.sin(curvature * s) / curvature, (1 - np.cos(curvature * s)) / curvature, curvature * s
    R, t = pose_ego_world[:2, :2], pose_ego_world[:2, 3]
    pos = np.column_stack((lx, ly)) @ R.T + t
    yaw0 = math.atan2(R[1, 0], R[0, 0])
    return pos, yaw0 + lth


def time_to_conflict(track, pose_ego_world, ego_speed, curvature=0.0, model='cv',
                     horizon=4.0, step=0.1):
    """First time within ``horizon`` s at which ego and object footprints overlap (inf if none)."""
    ts = np.arange(0.0, horizon + 1e-9, step)
    obj = predict_xy(track, ts, model)
    ego, yaw = ego_future(pose_ego_world, ego_speed, curvature, ts)
    d = obj - ego
    c, s = np.cos(yaw), np.sin(yaw)
    xr = c * d[:, 0] + s * d[:, 1]                 # object centre in the future ego frame
    yr = -s * d[:, 0] + c * d[:, 1]
    half_len = max(track.x_off, 0.5)
    half_wid = 0.5 * max(track.ext_y, 0.5)
    hit = ((xr - half_len < BUMPER_MARGIN) & (xr + half_len > -EGO_REAR) &
           (np.abs(yr) < EGO_HALF_WIDTH + half_wid + LATERAL_MARGIN))
    if hit.any():
        return float(ts[np.argmax(hit)])
    return math.inf


def displacement_errors(history, horizons, model='cv'):
    """ADE / FDE style errors of one track history.

    history: list of (t, x, y, vx, vy, heading_rate) world-frame filter states (one per frame,
             t in seconds, uniformly increasing); ground truth for a prediction made at
             index i is the *filtered position* at i + h (the best position estimate we have;
             KITTI raw has no object labels).
    Returns {h: [errors]} for every starting index with enough future.
    """
    from types import SimpleNamespace
    arr = np.array(history, float)
    out = {h: [] for h in horizons}
    for i in range(len(arr)):
        tr = SimpleNamespace(state=arr[i, 1:5], heading_rate=arr[i, 5])
        for h in horizons:
            j = int(np.argmin(np.abs(arr[:, 0] - (arr[i, 0] + h))))      # nearest frame to t + h
            if j <= i or abs(arr[j, 0] - (arr[i, 0] + h)) > 0.06:
                continue
            pred = predict_xy(tr, [arr[j, 0] - arr[i, 0]], model)[0]
            out[h].append(float(np.linalg.norm(pred - arr[j, 1:3])))
    return out
