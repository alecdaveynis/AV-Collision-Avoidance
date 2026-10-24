"""Timestamps and OXTS (ego-motion) helpers for KITTI raw drives."""
from datetime import datetime
from pathlib import Path

import numpy as np

import config


def load_timestamps(path):
    """Timestamps in seconds (float array) from a KITTI timestamps.txt, or None."""
    path = Path(path)
    if not path.exists():
        return None
    times = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # KITTI stores nanoseconds; datetime only handles microseconds.
            base, _, frac = line.partition('.')
            dt = datetime.strptime(f"{base}.{frac[:6].ljust(6, '0')}", '%Y-%m-%d %H:%M:%S.%f')
            times.append(dt.timestamp())
    return np.array(times)


def frame_dts(timestamps, n_frames, default=config.DEFAULT_DT):
    """Per-frame dt (s). dt[0] uses the default."""
    dts = np.full(n_frames, default, dtype=float)
    if timestamps is not None and len(timestamps) >= n_frames:
        diffs = np.diff(timestamps[:n_frames])
        valid = (diffs > 1e-3) & (diffs < 1.0)
        dts[1:][valid] = diffs[valid]
    return dts


def nearest_index(timestamps, t):
    """(index, |dt|) of the timestamp closest to t."""
    i = int(np.searchsorted(timestamps, t))
    cands = [j for j in (i - 1, i) if 0 <= j < len(timestamps)]
    best = min(cands, key=lambda j: abs(timestamps[j] - t))
    return best, float(abs(timestamps[best] - t))


def read_forward_speed(path):
    """Forward speed (m/s) from one OXTS packet file, or None if missing."""
    path = Path(path)
    if not path.exists():
        return None
    vals = np.loadtxt(path)
    # OXTS fields: lat lon alt roll pitch yaw vn ve vf vl vu ...
    return float(vals[8])


# --- full OXTS packets and poses (for driven-path ground truth / yaw rate) ----------
EARTH_RADIUS = 6378137.0


def read_packets(files):
    """Stack OXTS packet files into an (N, 25+) float array."""
    return np.array([np.loadtxt(f) for f in files])


def poses_from_packets(packets):
    """World poses (N, 4, 4) of the IMU, KITTI-devkit convention (Mercator + RPY)."""
    lat0 = packets[0, 0]
    scale = np.cos(lat0 * np.pi / 180.0)
    poses = np.zeros((len(packets), 4, 4))
    for k, p in enumerate(packets):
        lat, lon, alt, roll, pitch, yaw = p[:6]
        t = np.array([scale * lon * np.pi * EARTH_RADIUS / 180.0,
                      scale * EARTH_RADIUS * np.log(np.tan((90.0 + lat) * np.pi / 360.0)),
                      alt])
        rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        poses[k] = np.eye(4)
        poses[k, :3, :3] = rz @ ry @ rx
        poses[k, :3, 3] = t
    return poses
