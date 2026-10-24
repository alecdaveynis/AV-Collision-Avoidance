"""Build (features, hindsight-TTC-label) samples from pipeline tracks.csv files.

For a sample at frame t of one tracked object:

  input  : camera/ego features of frames t-T+1 .. t            (past only)
  label  : TTC = d(t) / closing_speed, where d and closing_speed come from a
           linear fit of LiDAR range over frames t .. t+W      (the future)
  baselines (past only, what a non-learned system could compute at time t):
           the same fit over frames t-W .. t, on LiDAR range and on monocular range

TTC is clipped to [TTC_MIN, TTC_MAX]; "not closing" means TTC_MAX.
Real collisions never occur in KITTI, so labels are *kinematic* TTC: how long
until the range would reach zero at the speed actually observed next.
"""
import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config
import ttc_model

MIN_CLOSING = config.MIN_CLOSING_SPEED


def ttc_from_range_history(times, dists, at_end):
    """TTC from a linear fit of range over a window.

    times, dists: 1-D arrays (same length >= 3, no NaNs).
    at_end: evaluate the fitted range at the last sample (past window) or the
            first sample (future window).
    Returns TTC in seconds, clipped to [TTC_MIN, TTC_MAX].
    """
    slope, intercept = np.polyfit(times - times[0], dists, 1)
    closing = -slope
    t_eval = (times[-1] - times[0]) if at_end else 0.0
    d = slope * t_eval + intercept
    if closing > MIN_CLOSING and d > 0:
        return float(np.clip(d / closing, config.TTC_MIN, config.TTC_MAX))
    return float(config.TTC_MAX)


@dataclass
class Samples:
    X: np.ndarray                 # (N, T, FEATURE_DIM)
    ttc: np.ndarray               # (N,) hindsight label, seconds
    base_lidar: np.ndarray        # (N,) past-window LiDAR baseline TTC (TTC_MAX if unavailable)
    base_mono: np.ndarray         # (N,) past-window monocular baseline TTC
    dist: np.ndarray              # (N,) current LiDAR range
    drive: list = field(default_factory=list)
    track: list = field(default_factory=list)
    frame: list = field(default_factory=list)

    def __len__(self):
        return len(self.ttc)

    def subset(self, mask):
        idx = np.nonzero(mask)[0]
        return Samples(self.X[idx], self.ttc[idx], self.base_lidar[idx], self.base_mono[idx],
                       self.dist[idx], [self.drive[i] for i in idx],
                       [self.track[i] for i in idx], [self.frame[i] for i in idx])

    @staticmethod
    def concat(parts):
        parts = [p for p in parts if len(p)]
        return Samples(np.concatenate([p.X for p in parts]),
                       np.concatenate([p.ttc for p in parts]),
                       np.concatenate([p.base_lidar for p in parts]),
                       np.concatenate([p.base_mono for p in parts]),
                       np.concatenate([p.dist for p in parts]),
                       sum([p.drive for p in parts], []),
                       sum([p.track for p in parts], []),
                       sum([p.frame for p in parts], []))


def read_tracks(path):
    """tracks.csv -> {track_id: [row dict sorted by frame]} with numeric fields."""
    by_track = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            row = {
                'frame': int(r['frame']), 'time': float(r['time']), 'dt': float(r['dt']),
                'cls': int(r['cls']),
                'box': (float(r['x1']), float(r['y1']), float(r['x2']), float(r['y2'])),
                'img_shape': (int(r['img_h']), int(r['img_w'])),
                'ego': float(r['ego_speed']) if r['ego_speed'] != '' else 0.0,
                'mono': float(r['mono_dist']),
                'lidar': float(r['lidar_dist']) if r['lidar_dist'] != '' else np.nan,
            }
            by_track.setdefault(int(r['id']), []).append(row)
    for rows in by_track.values():
        rows.sort(key=lambda x: x['frame'])
    return by_track


def contiguous_runs(rows):
    run = [rows[0]]
    for r in rows[1:]:
        if r['frame'] - run[-1]['frame'] == 1:
            run.append(r)
        else:
            yield run
            run = [r]
    yield run


def build_samples(tracks_csv, drive_name=None):
    """All valid samples of one drive. Samples need a full input window, a full
    future LiDAR window (for the label) and a full past LiDAR window (baseline)."""
    T, W = ttc_model.SEQ_LEN, config.TTC_LABEL_WINDOW
    drive_name = drive_name or Path(tracks_csv).parent.name
    X, ttc, bl, bm, dist, drv, trk, frm = [], [], [], [], [], [], [], []

    for tid, rows in read_tracks(tracks_csv).items():
        for run in contiguous_runs(rows):
            n = len(run)
            base = np.stack([ttc_model.frame_features(r['box'], r['img_shape'], r['cls'],
                                                      r['mono'], r['ego']) for r in run])
            dts = np.array([r['dt'] for r in run])
            times = np.array([r['time'] for r in run])
            lid = np.array([r['lidar'] for r in run])
            mono = np.array([r['mono'] for r in run])

            for j in range(T - 1, n - W):                  # needs W future frames
                fut = slice(j, j + W + 1)
                past = slice(j - W, j + 1)
                if np.isnan(lid[fut]).any() or lid[j] <= 0:
                    continue
                label = ttc_from_range_history(times[fut], lid[fut], at_end=False)

                if j - W >= 0 and not np.isnan(lid[past]).any():
                    b_l = ttc_from_range_history(times[past], lid[past], at_end=True)
                else:
                    b_l = float(config.TTC_MAX)
                b_m = (ttc_from_range_history(times[past], mono[past], at_end=True)
                       if j - W >= 0 else float(config.TTC_MAX))

                X.append(ttc_model.build_sequence(base[j - T + 1:j + 1], dts[j - T + 1:j + 1]))
                ttc.append(label); bl.append(b_l); bm.append(b_m); dist.append(lid[j])
                drv.append(drive_name); trk.append(tid); frm.append(run[j]['frame'])

    if not X:
        return Samples(np.zeros((0, T, ttc_model.FEATURE_DIM), np.float32), np.zeros(0),
                       np.zeros(0), np.zeros(0), np.zeros(0))
    return Samples(np.stack(X).astype(np.float32), np.array(ttc), np.array(bl), np.array(bm),
                   np.array(dist), drv, trk, frm)
