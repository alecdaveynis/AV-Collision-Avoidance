"""LiDAR 3D object proposals + Kalman tracking with velocity, in a world frame.

Pipeline per frame
  1. ``cluster_obstacles``  : obstacle points (fusion_utils.obstacle_points) -> bird's-eye
                              connected components -> one ``Cluster`` per object
  2. ``Tracker3D.update``   : cluster centres are moved into a *world* frame with the ego
                              pose, associated to tracks (Hungarian, gated), and filtered
                              with a constant-velocity Kalman filter [x, y, vx, vy]
  3. ``Track3D`` properties : position / velocity relative to the ego car, bumper range,
                              closing speed and kinematic TTC *with an uncertainty
                              interval* from the filter covariance

Working in the world frame means a parked car has zero velocity however the ego car
moves, so the velocity estimate is the object's own motion (not ego-motion leaking
into the range rate). Everything is causal.
"""
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

import config

# --- tunables (kept here: they belong to this module only) ---------------------------
CELL = 0.4                  # m, bird's-eye occupancy cell
CLUSTER_MAX_X = 60.0
CLUSTER_MAX_Y = 20.0
CLUSTER_MIN_POINTS = 8
CLUSTER_MAX_EXTENT = 12.0   # m, longer/wider blobs are walls, hedges, rails: not tracked as objects
MEAS_STD = 0.35             # m, cluster-centre measurement noise (viewpoint changes shift it)
ACCEL_STD = 2.5             # m/s^2, process noise of the constant-velocity model
INIT_VEL_STD = 6.0          # m/s, velocity prior of a new track
GATE_BASE = 2.0             # m, association gate = GATE_BASE + speed_allowance * dt
GATE_SPEED = 12.0           # m/s
CONFIRM_HITS = 3
MAX_MISSES = 4
STATIC_SPEED = 0.8          # m/s, below this an object counts as stationary
Z_95 = 1.645                # one-sided 95 %
TTC_BAND_SCALE = 3.2        # inflation of the filter sigma so the 90 % band is actually ~90 % (eval_tracking3d.py)


@dataclass
class Cluster:
    """One LiDAR object proposal, in the Velodyne frame (x fwd, y left, z up)."""
    x: float                # centroid
    y: float
    z: float                # mid height
    x_near: float           # nearest surface along x
    ext_x: float            # extent along x / y (m)
    ext_y: float
    n: int
    z_top: float = 0.0


def cluster_obstacles(obs_pts, cell=CELL, min_points=CLUSTER_MIN_POINTS):
    """Group obstacle points into object clusters (bird's-eye connected components)."""
    if len(obs_pts) == 0:
        return []
    m = obs_pts[(obs_pts[:, 0] > 0) & (obs_pts[:, 0] < CLUSTER_MAX_X) & (np.abs(obs_pts[:, 1]) < CLUSTER_MAX_Y)]
    if len(m) < min_points:
        return []
    nx = int(CLUSTER_MAX_X / cell) + 1
    ny = int(2 * CLUSTER_MAX_Y / cell) + 1
    ix = (m[:, 0] / cell).astype(int)
    iy = ((m[:, 1] + CLUSTER_MAX_Y) / cell).astype(int)
    grid = np.zeros((nx, ny), np.uint8)
    grid[ix, iy] = 1
    # bridge the gaps between LiDAR rings (spacing grows with range)
    grid = cv2.dilate(grid, np.ones((3, 3), np.uint8))
    n_lab, labels = cv2.connectedComponents(grid, connectivity=8)
    point_label = labels[ix, iy]
    out = []
    for lab in range(1, n_lab):
        p = m[point_label == lab]
        if len(p) < min_points:
            continue
        ext_x = float(p[:, 0].max() - p[:, 0].min())
        ext_y = float(p[:, 1].max() - p[:, 1].min())
        if max(ext_x, ext_y) > CLUSTER_MAX_EXTENT:
            continue
        out.append(Cluster(x=float(p[:, 0].mean()), y=float(p[:, 1].mean()),
                           z=float(0.5 * (p[:, 2].min() + p[:, 2].max())),
                           x_near=float(np.percentile(p[:, 0], 5)),
                           ext_x=ext_x, ext_y=ext_y, n=len(p), z_top=float(p[:, 2].max())))
    return out


def world_from_ego(pose_ego_world, xy):
    """Ego-frame (N, 2) ground-plane points -> world-frame (N, 2)."""
    xy = np.atleast_2d(xy)
    R, t = pose_ego_world[:2, :2], pose_ego_world[:2, 3]
    return xy @ R.T + t


@dataclass
class Track3D:
    id: int
    state: np.ndarray                 # [x, y, vx, vy] world frame
    P: np.ndarray                     # 4x4 covariance
    hits: int = 1
    misses: int = 0
    age: int = 1
    x_off: float = 0.0                # centroid - nearest-surface offset (EMA), m
    ext_x: float = 0.0
    ext_y: float = 0.0
    z: float = 0.0
    z_top: float = 0.0
    heading_rate: float = 0.0         # rad/s of the velocity vector (EMA), for CTRV prediction
    _last_heading: float = None
    # filled by Tracker3D after each update (ego-relative quantities)
    rel_pos: np.ndarray = field(default_factory=lambda: np.zeros(2))       # ego frame (x fwd, y left)
    rel_vel: np.ndarray = field(default_factory=lambda: np.zeros(2))       # ego frame, object minus ego
    ego_R: np.ndarray = field(default_factory=lambda: np.eye(2))           # world<-ego rotation

    @property
    def confirmed(self):
        return self.hits >= CONFIRM_HITS

    @property
    def speed_world(self):
        return float(np.hypot(self.state[2], self.state[3]))

    @property
    def is_static(self):
        return self.speed_world < STATIC_SPEED

    @property
    def bumper_range(self):
        """Longitudinal distance to the nearest surface (m)."""
        return max(float(self.rel_pos[0]) - self.x_off, 0.3)

    @property
    def lateral(self):
        return float(self.rel_pos[1])

    @property
    def closing_speed(self):
        """m/s along the ego x axis, > 0 when the gap shrinks."""
        return float(-self.rel_vel[0])

    def closing_std(self):
        """Std of the closing speed (m/s), from the velocity covariance rotated into the ego frame."""
        Pv = self.P[2:, 2:]
        ux = self.ego_R[:, 0]            # ego x axis in world coordinates
        return float(math.sqrt(max(ux @ Pv @ ux, 0.0)))

    def ttc(self):
        """(ttc, ttc_low, ttc_high) in seconds. inf when not closing. low/high = calibrated 90 % band.

        The raw filter covariance under-covers (it ignores centroid drift and label noise), so the
        sigma is scaled by TTC_BAND_SCALE, fitted leave-one-drive-out by eval_tracking3d.py.
        """
        c, s, r = self.closing_speed, TTC_BAND_SCALE * self.closing_std(), self.bumper_range

        def one(closing):
            return r / closing if closing > config.MIN_CLOSING_SPEED else math.inf
        return one(c), one(c + Z_95 * s), one(c - Z_95 * s)


class Tracker3D:
    def __init__(self):
        self.tracks = {}
        self._next_id = 1

    # -- Kalman primitives ----------------------------------------------------------
    @staticmethod
    def _F_Q(dt):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = ACCEL_STD ** 2
        Q = np.zeros((4, 4))
        for i in (0, 1):
            Q[i, i] = q * dt ** 4 / 4
            Q[i, i + 2] = Q[i + 2, i] = q * dt ** 3 / 2
            Q[i + 2, i + 2] = q * dt ** 2
        return F, Q

    def _predict(self, dt):
        F, Q = self._F_Q(dt)
        for t in self.tracks.values():
            t.state = F @ t.state
            t.P = F @ t.P @ F.T + Q

    def _new(self, pos_w, c):
        P = np.diag([MEAS_STD ** 2, MEAS_STD ** 2, INIT_VEL_STD ** 2, INIT_VEL_STD ** 2])
        t = Track3D(self._next_id, np.array([pos_w[0], pos_w[1], 0.0, 0.0]), P,
                    x_off=c.x - c.x_near, ext_x=c.ext_x, ext_y=c.ext_y, z=c.z, z_top=c.z_top)
        self.tracks[t.id] = t
        self._next_id += 1

    # -- main entry -----------------------------------------------------------------
    def update(self, clusters, pose_ego_world, dt, ego_vel_world=None):
        """Advance one frame.

        clusters       : list of Cluster in the ego (Velodyne) frame
        pose_ego_world : 4x4 transform ego -> world at this frame
        dt             : seconds since the previous update
        ego_vel_world  : (2,) ego velocity in the world frame (m/s); estimated from
                         consecutive poses when omitted
        Returns the list of *confirmed* tracks (ego-relative fields refreshed).
        """
        dt = dt if dt and dt > 1e-3 else config.DEFAULT_DT
        R_ego = pose_ego_world[:2, :2]
        pos_ego_w = pose_ego_world[:2, 3]
        if ego_vel_world is None:
            prev = getattr(self, '_prev_ego_pos', None)
            ego_vel_world = (pos_ego_w - prev) / dt if prev is not None else R_ego[:, 0] * 0.0
        self._prev_ego_pos = pos_ego_w.copy()
        ego_vel_world = np.asarray(ego_vel_world, float)

        self._predict(dt)
        meas = (world_from_ego(pose_ego_world, np.array([[c.x, c.y] for c in clusters]))
                if clusters else np.zeros((0, 2)))

        ids = list(self.tracks)
        matched_t, matched_m = set(), set()
        if ids and len(meas):
            cost = np.full((len(ids), len(meas)), 1e6)
            for i, tid in enumerate(ids):
                t = self.tracks[tid]
                gate = GATE_BASE + GATE_SPEED * dt
                d = np.linalg.norm(meas - t.state[:2], axis=1)
                cost[i] = np.where(d < gate, d, 1e6)
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 1e5:
                    self._correct(self.tracks[ids[i]], meas[j], clusters[j], dt)
                    matched_t.add(ids[i])
                    matched_m.add(j)

        for tid in ids:
            if tid not in matched_t:
                self.tracks[tid].misses += 1
                self.tracks[tid].age += 1
        for tid in [k for k, t in self.tracks.items() if t.misses > MAX_MISSES]:
            del self.tracks[tid]
        for j, c in enumerate(clusters):
            if j not in matched_m:
                self._new(meas[j], c)

        out = []
        for t in self.tracks.values():
            if not t.confirmed or t.misses > 1:
                continue
            t.ego_R = R_ego
            t.rel_pos = R_ego.T @ (t.state[:2] - pos_ego_w)
            t.rel_vel = R_ego.T @ (t.state[2:] - ego_vel_world)
            out.append(t)
        return out

    def _correct(self, t, z, c, dt):
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        R = np.eye(2) * MEAS_STD ** 2
        y = z - H @ t.state
        S = H @ t.P @ H.T + R
        K = t.P @ H.T @ np.linalg.inv(S)
        t.state = t.state + K @ y
        t.P = (np.eye(4) - K @ H) @ t.P
        t.hits += 1
        t.misses = 0
        t.age += 1
        a = 0.3
        t.x_off = (1 - a) * t.x_off + a * (c.x - c.x_near)
        t.ext_x = (1 - a) * t.ext_x + a * c.ext_x
        t.ext_y = (1 - a) * t.ext_y + a * c.ext_y
        t.z, t.z_top = c.z, c.z_top
        sp = t.speed_world
        if sp > 1.0:
            h = math.atan2(t.state[3], t.state[2])
            if t._last_heading is not None:
                dh = (h - t._last_heading + math.pi) % (2 * math.pi) - math.pi
                t.heading_rate = 0.7 * t.heading_rate + 0.3 * float(np.clip(dh / dt, -0.8, 0.8))
            t._last_heading = h
        else:
            t._last_heading = None
            t.heading_rate *= 0.8


def in_ego_path(track, curvature=0.0, half_width=None):
    """True if the track's centre lies inside the (curved) ego corridor."""
    half_width = config.PATH_MAX_LATERAL if half_width is None else half_width
    x = track.rel_pos[0]
    centre = 0.5 * curvature * x * x
    return 0.0 < x < config.CORRIDOR_X_RANGE[1] and abs(track.rel_pos[1] - centre) < half_width + 0.5 * track.ext_y


def match_to_boxes(tracks3d, yolo_tracks, uv_fn):
    """Attach 3D tracks to camera tracks: the 3D track whose centre falls in a box.

    uv_fn(track3d) -> (u, v) pixel of the track's centre or None. Boxes are matched
    to the nearest 3D track whose centre is inside; each 3D track is used once.
    Returns {yolo_track_id: Track3D}.
    """
    cands = []
    for t3 in tracks3d:
        uv = uv_fn(t3)
        if uv is None:
            continue
        for yt in yolo_tracks:
            x1, y1, x2, y2 = yt['box']
            if x1 <= uv[0] <= x2 and y1 <= uv[1] <= y2:
                cands.append((t3.rel_pos[0], t3.id, yt['id'], t3))
    cands.sort(key=lambda c: c[0])
    used3, usedy, out = set(), set(), {}
    for _, i3, iy, t3 in cands:
        if i3 in used3 or iy in usedy:
            continue
        used3.add(i3)
        usedy.add(iy)
        out[iy] = t3
    return out
