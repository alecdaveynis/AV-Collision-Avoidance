"""KITTI calibration loading.

If the official calibration files are present in ``config.CALIB_DIR`` they are
parsed; otherwise the values for the 2011_09_26 recording (the ones that were
previously hard-coded) are used.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config


@dataclass
class Calibration:
    P_rect: np.ndarray      # 3x4 projection of rectified camera 02
    R_rect: np.ndarray      # 4x4 rectification rotation (cam 00)
    Tr_velo_to_cam: np.ndarray  # 4x4 velodyne -> cam 00
    T_imu_to_velo: np.ndarray = None  # 4x4 OXTS IMU -> velodyne (None -> default)

    def __post_init__(self):
        if self.T_imu_to_velo is None:
            self.T_imu_to_velo = _DEFAULT_IMU_TO_VELO.copy()

    @property
    def fx(self):
        return float(self.P_rect[0, 0])

    @property
    def fy(self):
        return float(self.P_rect[1, 1])


_DEFAULT_P_RECT = np.array([[7.21e+02, 0.00e+00, 6.09e+02, 4.48e+01],
                            [0.00e+00, 7.21e+02, 1.72e+02, 2.16e-01],
                            [0.00e+00, 0.00e+00, 1.00e+00, 2.74e-03]])
_DEFAULT_TR = np.array([[7.53e-03, -9.99e-01, -3.22e-04, -4.06e-03],
                        [1.48e-02, 7.28e-04, -9.99e-01, -7.63e-02],
                        [9.99e-01, 7.50e-03, 1.48e-02, -2.71e-01],
                        [0, 0, 0, 1]])


# 2011_09_26 IMU -> Velodyne (calib_imu_to_velo.txt). The rotation is ~identity
# (|angles| < 1 degree), so only the translation matters in practice.
_DEFAULT_IMU_TO_VELO = np.array([[1.0, 0.0, 0.0, -8.087e-01],
                                 [0.0, 1.0, 0.0, 3.196e-01],
                                 [0.0, 0.0, 1.0, -7.997e-01],
                                 [0, 0, 0, 1]])


def _parse_calib_file(path):
    out = {}
    with open(path) as f:
        for line in f:
            if ':' not in line:
                continue
            key, val = line.split(':', 1)
            try:
                out[key.strip()] = np.array([float(x) for x in val.split()])
            except ValueError:
                pass  # e.g. calib_time
    return out


def load_calibration(calib_dir=None):
    calib_dir = Path(calib_dir or config.CALIB_DIR)
    cam_file = calib_dir / 'calib_cam_to_cam.txt'
    velo_file = calib_dir / 'calib_velo_to_cam.txt'
    if cam_file.exists() and velo_file.exists():
        cam = _parse_calib_file(cam_file)
        velo = _parse_calib_file(velo_file)
        P_rect = cam['P_rect_02'].reshape(3, 4)
        R_rect = np.eye(4)
        R_rect[:3, :3] = cam['R_rect_00'].reshape(3, 3)
        Tr = np.eye(4)
        Tr[:3, :3] = velo['R'].reshape(3, 3)
        Tr[:3, 3] = velo['T']
        imu_file = calib_dir / 'calib_imu_to_velo.txt'
        T_iv = None
        if imu_file.exists():
            imu = _parse_calib_file(imu_file)
            T_iv = np.eye(4)
            T_iv[:3, :3] = imu['R'].reshape(3, 3)
            T_iv[:3, 3] = imu['T']
        return Calibration(P_rect, R_rect, Tr, T_iv)
    return Calibration(_DEFAULT_P_RECT.copy(), np.eye(4), _DEFAULT_TR.copy())
