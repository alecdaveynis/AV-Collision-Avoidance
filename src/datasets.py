"""Access to KITTI raw drives, synced ('*_sync') or unsynced ('*_extract').

Camera frames are the master clock. LiDAR scans and OXTS packets are matched to
each frame by nearest timestamp, which works for both layouts:

* *_sync    : scans / packets already line up one-to-one with the images
* *_extract : images at 10 Hz, LiDAR scans at 10 Hz with an arbitrary phase
              (a scan can be up to ~50 ms away), OXTS at 100 Hz
"""
import os
from pathlib import Path

import numpy as np

import config
import oxts_utils


class Drive:
    def __init__(self, root):
        self.root = Path(root)
        self.name = self.root.name
        self.image_dir = self.root / 'image_02' / 'data'
        self.frames = sorted(self.image_dir.glob('*.png'))

        self.image_ts = oxts_utils.load_timestamps(self.root / 'image_02' / 'timestamps.txt')
        self.lidar_dir = self.root / 'velodyne_points' / 'data'
        self.lidar_files = sorted(list(self.lidar_dir.glob('*.bin')) + list(self.lidar_dir.glob('*.txt')))
        self.lidar_ts = oxts_utils.load_timestamps(self.root / 'velodyne_points' / 'timestamps.txt')
        self.oxts_dir = self.root / 'oxts' / 'data'
        self.oxts_files = sorted(self.oxts_dir.glob('*.txt'))
        self.oxts_ts = oxts_utils.load_timestamps(self.root / 'oxts' / 'timestamps.txt')
        self._packets = None
        self._poses = None

    def __len__(self):
        return len(self.frames)

    @property
    def times(self):
        """Frame times in seconds relative to the first frame."""
        if self.image_ts is None or len(self.image_ts) < len(self.frames):
            return np.arange(len(self.frames)) * config.DEFAULT_DT
        return self.image_ts[:len(self.frames)] - self.image_ts[0]

    def dts(self):
        return oxts_utils.frame_dts(self.image_ts, len(self.frames))

    # -- LiDAR ---------------------------------------------------------------------
    def _match(self, files, file_ts, i):
        """Index of the file matching frame i (nearest timestamp), or None."""
        if not files:
            return None
        if self.image_ts is None or file_ts is None or len(file_ts) != len(files):
            return i if i < len(files) else None
        j, delta = oxts_utils.nearest_index(file_ts, self.image_ts[i])
        return j, delta

    def lidar(self, i, tolerance=config.LIDAR_SYNC_TOLERANCE):
        """Nx4 [x, y, z, reflectance] scan nearest in time to frame i, or None."""
        m = self._match(self.lidar_files, self.lidar_ts, i)
        if m is None:
            return None
        j = m if isinstance(m, int) else (m[0] if m[1] <= tolerance else None)
        if j is None:
            return None
        path = self.lidar_files[j]
        if path.suffix == '.bin':
            return np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        # *_extract drives store scans as text: "x y z r" per line
        return np.fromfile(path, dtype=np.float32, sep=' ').reshape(-1, 4)

    # -- Ego motion --------------------------------------------------------------
    def ego_speed(self, i):
        m = self._match(self.oxts_files, self.oxts_ts, i)
        if m is None:
            return None
        j = m if isinstance(m, int) else m[0]
        return oxts_utils.read_forward_speed(self.oxts_files[j])

    # -- OXTS packets / poses -----------------------------------------------------
    def _oxts_index(self, i):
        m = self._match(self.oxts_files, self.oxts_ts, i)
        if m is None:
            return None
        return m if isinstance(m, int) else m[0]

    @property
    def packets(self):
        if self._packets is None:
            self._packets = oxts_utils.read_packets(self.oxts_files)
        return self._packets

    @property
    def poses(self):
        if self._poses is None:
            self._poses = oxts_utils.poses_from_packets(self.packets)
        return self._poses

    def yaw_rate(self, i):
        """Yaw rate (rad/s, + = turning left) at frame i, or None."""
        j = self._oxts_index(i)
        return None if j is None else float(self.packets[j, 19])

    def driven_path(self, i, calib, length=40.0, min_length=20.0):
        """Where the car actually drove next, in the Velodyne frame at frame i.

        Returns (N, 2) array of (x forward, y left), x increasing, or None if the
        recording ends before ``min_length`` metres of future path exist. This is
        future information (ground truth for evaluation), not available online.
        """
        j0 = self._oxts_index(i)
        if j0 is None:
            return None
        poses = self.poses
        inv0 = np.linalg.inv(poses[j0])
        fut = poses[j0:, :3, 3]
        pts_imu = (inv0[:3, :3] @ fut.T).T + inv0[:3, 3]
        T = calib.T_imu_to_velo
        pts = (T[:3, :3] @ pts_imu.T).T + T[:3, 3]
        steps = np.linalg.norm(np.diff(pts_imu[:, :2], axis=0), axis=1)
        travelled = np.concatenate([[0.0], np.cumsum(steps)])
        pts = pts[travelled <= length + 5.0]
        if len(pts) < 2 or travelled[len(pts) - 1] < min_length:
            return None
        xy = pts[:, :2]
        keep = np.concatenate([[True], np.diff(xy[:, 0]) > 1e-3])      # strictly increasing x
        xy = xy[keep]
        keep = np.maximum.accumulate(xy[:, 0]) == xy[:, 0]
        return xy[keep]


def discover_drives(data_dir=None):
    """Find every drive (a folder containing image_02/data) below data_dir."""
    data_dir = Path(data_dir or config.DATA_DIR)
    found = []
    for dirpath, dirnames, _ in os.walk(data_dir):
        if 'image_02' in dirnames and (Path(dirpath) / 'image_02' / 'data').is_dir():
            found.append(Drive(dirpath))
            dirnames.clear()
    return sorted(found, key=lambda d: d.name)
