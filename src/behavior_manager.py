# behavior_manager.py
"""Behaviour planner: GO / SLOW / STOP from in-path obstacles.

Improvements over the first version:
* real frame dt (from KITTI timestamps) instead of a constant
* per-track EMA smoothing of distance and closing speed
* only obstacles flagged ``in_path`` (ego lane) influence the decision
* thresholds scale with ego speed (OXTS), so 6 m means something different at
  10 km/h and at 50 km/h
* hysteresis (confirm to escalate, hold to release) so the state doesn't flicker
* stale tracks are forgotten
* optional learned TTC (see ttc_model.py): ``ttc_mode`` selects analytic only,
  learned only, or a geometric-mean fusion. Distance thresholds stay rule-based
  in every mode, so the network can only change *when* the TTC rule fires.
"""
import math
from dataclasses import dataclass
from typing import Optional

import config


class BehaviorState:
    GO = "DRIVE"
    SLOW = "SLOWING"
    STOP = "STOPPED"


_SEVERITY = {BehaviorState.GO: 0, BehaviorState.SLOW: 1, BehaviorState.STOP: 2}
_BRAKE = {BehaviorState.GO: 0.0, BehaviorState.SLOW: 0.3, BehaviorState.STOP: 1.0}


@dataclass
class Decision:
    state: str
    text: str
    brake: float
    ttc: float                       # s, TTC actually used (math.inf if not closing)
    min_dist: float                  # m, closest in-path obstacle (99 if none)
    stop_dist: float
    slow_dist: float
    ttc_analytic: float = math.inf   # s, range-rate TTC (worst in-path entity)
    ttc_learned: Optional[float] = None  # s, network TTC (worst in-path track), None if unavailable
    ego_speed: Optional[float] = None
    lead_id: Optional[int] = None
    lead_dist: Optional[float] = None
    lead_closing: Optional[float] = None   # m/s, > 0 means closing
    lead_speed: Optional[float] = None     # m/s, estimated object forward speed
    raw_state: str = BehaviorState.GO      # state before hysteresis
    demand_decel: float = 0.0              # m/s^2, braking the safe-stop rule asked for (improved planner; 0 for baseline)


class _Smoother:
    """EMA distance + closing speed for one tracked entity."""

    def __init__(self, dist, frame):
        self.dist = dist
        self.last_raw = dist      # previous raw measurement (for the jump gate)
        self.closing = 0.0
        self.has_closing = False
        self.frame = frame

    def update(self, dist, dt, frame):
        # A range jump no real object could make in one frame means the measurement
        # now refers to something else (new nearest object, track ID swap, ...):
        # restart the estimate instead of reporting a huge closing speed. The gate
        # compares consecutive RAW measurements (the physical change per frame), not
        # the lagging smoothed value, whose lag alone grows with closing speed.
        jump = abs(dist - self.last_raw) > config.MAX_RANGE_RATE * dt
        self.last_raw = dist
        if jump:
            self.dist, self.closing, self.has_closing, self.frame = dist, 0.0, False, frame
            return
        alpha = config.DIST_EMA_ALPHA
        new_dist = (1 - alpha) * self.dist + alpha * dist
        inst_closing = (self.dist - new_dist) / dt
        inst_closing = max(-40.0, min(40.0, inst_closing))
        if self.has_closing:
            self.closing = (1 - alpha) * self.closing + alpha * inst_closing
        else:
            self.closing, self.has_closing = inst_closing, True
        self.dist, self.frame = new_dist, frame


def _ttc(dist, closing):
    if closing > config.MIN_CLOSING_SPEED:
        return dist / closing
    return math.inf


TTC_MODES = ('analytic', 'learned', 'fused', 'kalman')


def combine_ttc(analytic, learned, mode, kinematic=None):
    """TTC used for one object under the selected mode.

    analytic : range-rate TTC (inf if not closing)
    learned  : network TTC or None (track too young)
    fused    : geometric mean of both (analytic capped at TTC_MAX so 'not closing'
               doesn't dominate); falls back to analytic when no network output.
    learned  : network only; falls back to analytic when no network output.
    kalman   : TTC from the 3D Kalman track (``kinematic``, uses the object's own velocity);
               falls back to analytic when the object has no 3D track.
    """
    if mode == 'kalman':
        return analytic if kinematic is None else kinematic
    if mode == 'analytic' or learned is None:
        return analytic
    if mode == 'learned':
        return learned
    return math.sqrt(min(analytic, config.TTC_MAX) * learned)


class BehaviorManager:
    def __init__(self, ttc_mode='analytic'):
        if ttc_mode not in TTC_MODES:
            raise ValueError(f"ttc_mode must be one of {TTC_MODES}")
        self.ttc_mode = ttc_mode
        self.state = BehaviorState.GO
        self.dt = config.DEFAULT_DT
        self.frame = 0
        self._tracks = {}        # id -> _Smoother
        self._lidar = None       # _Smoother for the in-path LiDAR minimum
        self._esc = 0
        self._rel = 0
        self._demand = 0.0       # m/s^2, set by _classify (used by subclasses that brake proportionally)

    # -- policy hooks: the baseline implementations below are the original behaviour, unchanged ---------
    def _classify(self, entities, slow_dist, v):
        """Raw (pre-hysteresis) state for the in-path entities.

        entities: list of (distance, stop_zone, ttc, closing_speed or None if not yet observed). Returns (raw_state, immediate).
        """
        raw = BehaviorState.GO
        immediate = False
        for dist, zone, e_ttc, _closing in entities:
            if dist < zone or e_ttc < config.TTC_STOP:
                raw = BehaviorState.STOP
            elif (dist < slow_dist or e_ttc < config.TTC_SLOW) and raw == BehaviorState.GO:
                raw = BehaviorState.SLOW
            if dist < 0.5 * zone or e_ttc < 0.5 * config.TTC_STOP:
                immediate = True
        return raw, immediate

    def _brake_command(self, state):
        """Brake fraction (0..1 of the vehicle's maximum deceleration) for the current state."""
        return _BRAKE[state]

    # -- helpers ---------------------------------------------------------------
    def _forget_stale(self):
        stale = [k for k, s in self._tracks.items()
                 if self.frame - s.frame > config.TRACK_MAX_AGE]
        for k in stale:
            del self._tracks[k]

    def _apply_hysteresis(self, raw, immediate):
        cur, new = _SEVERITY[self.state], _SEVERITY[raw]
        if new > cur:
            self._rel = 0
            self._esc += 1
            if immediate or self._esc >= config.ESCALATE_FRAMES:
                self.state, self._esc = raw, 0
        elif new < cur:
            self._esc = 0
            self._rel += 1
            if self._rel >= config.RELEASE_FRAMES:
                self.state, self._rel = raw, 0
        else:
            self._esc = self._rel = 0

    # -- main entry ------------------------------------------------------------
    def update_state(self, obstacles, lidar_dist, dt=None, ego_speed=None):
        """Update the FSM.

        obstacles : list of dicts with 'id', 'dist' and optionally 'in_path'
        lidar_dist: closest LiDAR return on the ego path (>= 99 means none)
        dt        : seconds since the previous frame
        ego_speed : forward speed in m/s (None -> treated as 0)
        """
        self.frame += 1
        self.dt = dt if dt and dt > 1e-3 else config.DEFAULT_DT

        # --- smooth per-track distance, derive closing speed -------------------
        in_path = []
        advisories = []          # predicted (not yet in-path) conflicts: may slow the car, never stop it
        for obs in obstacles:
            if obs.get('advisory'):
                advisories.append(obs)
                continue
            sm = self._tracks.get(obs['id'])
            if sm is None:
                sm = self._tracks[obs['id']] = _Smoother(obs['dist'], self.frame)
            else:
                sm.update(obs['dist'], self.dt, self.frame)
            if obs.get('in_path', True):
                in_path.append((obs['id'], sm, obs.get('ttc_learned'), obs.get('ttc_3d'), obs.get('closing_3d'), obs.get('track_hits')))
        self._forget_stale()

        # --- LiDAR in-path minimum, handled like one more track -----------------
        if lidar_dist < 99.0:
            if self._lidar is None:
                self._lidar = _Smoother(lidar_dist, self.frame)
            else:
                self._lidar.update(lidar_dist, self.dt, self.frame)
        else:
            self._lidar = None

        # --- speed-aware zones ------------------------------------------------------
        # Stop zone depends on how fast the obstacle is *closing* (a lead car at our
        # speed is not a wall); unknown closing speed is treated as a static obstacle.
        # Slow zone is a time-headway rule on our own speed.
        v = max(ego_speed, 0.0) if ego_speed is not None else 0.0
        slow_dist = config.SLOW_BASE_DIST + v * config.SLOW_HEADWAY

        def stop_zone(closing):
            c = max(closing, 0.0)
            return config.STOP_BASE_DIST + c * c / (2.0 * config.EMERGENCY_DECEL)

        # --- per-entity checks over in-path obstacles --------------------------------
        # The in-path LiDAR range-rate TTC always applies (in every mode) as the
        # rule-based safety layer; per-track TTC follows ``ttc_mode``.
        entities = []        # (distance, stop_zone, ttc, closing speed)
        ttc = ttc_analytic = math.inf
        ttc_learned = None
        lead = None
        for oid, sm, learned, kin, kin_closing, hits in in_path:
            analytic = _ttc(sm.dist, sm.closing)
            ttc_analytic = min(ttc_analytic, analytic)
            if learned is not None:
                ttc_learned = learned if ttc_learned is None else min(ttc_learned, learned)
            used = combine_ttc(analytic, learned, self.ttc_mode, kin)
            closing = sm.closing if sm.has_closing else v
            zone = stop_zone(closing)                      # baseline zone: smoothed range-rate closing, as before
            # closing speed offered to planners that opt in: Kalman estimate, else smoothed range-rate, else None (unknown)
            if self.ttc_mode == 'kalman' and kin_closing is not None:
                closing = kin_closing
                if hits is not None and hits < config.SAFE_MIN_TRACK_HITS:
                    closing = None                         # young track: Kalman velocity still dominated by its prior
            elif not sm.has_closing:
                closing = None
            entities.append((sm.dist, zone, used, closing))
            if lead is None or sm.dist < lead[1].dist:
                lead = (oid, sm)
        if lidar_dist < 99.0:
            if self._lidar is not None:
                lidar_ttc = _ttc(self._lidar.dist, self._lidar.closing)
                closing = self._lidar.closing if self._lidar.has_closing else v
                known = self._lidar.has_closing
            else:
                lidar_ttc, closing, known = math.inf, v, False
            ttc_analytic = min(ttc_analytic, lidar_ttc)
            # raw LiDAR-minimum range-rate is noisier than the Kalman track's velocity: in kalman mode it is not offered
            entities.append((lidar_dist, stop_zone(closing), lidar_ttc,
                             closing if (known and self.ttc_mode != 'kalman') else None))
        ttc = min([e[2] for e in entities], default=math.inf)
        min_dist = min([e[0] for e in entities], default=99.0)

        self._demand = 0.0
        raw, immediate = self._classify(entities, slow_dist, v)
        if raw == BehaviorState.GO:
            for adv in advisories:
                if adv['dist'] < slow_dist or adv.get('ttc_3d', math.inf) < config.TTC_SLOW:
                    raw = BehaviorState.SLOW
                    ttc = min(ttc, adv.get('ttc_3d', math.inf))
                    break
        self._apply_hysteresis(raw, immediate)

        # Zone shown on the dashboard: the lead's, or the static-obstacle zone if none.
        if lead is not None and lead[1].has_closing:
            stop_dist = stop_zone(lead[1].closing)
        else:
            stop_dist = stop_zone(v)

        ttc_txt = f"{ttc:.1f}s" if math.isfinite(ttc) else "--"
        if self.state == BehaviorState.STOP:
            text = f"BRAKE 100% (TTC: {ttc_txt})"
        elif self.state == BehaviorState.SLOW:
            text = f"SLOWING (TTC: {ttc_txt})"
        else:
            text = "DRIVING"

        dec = Decision(state=self.state, text=text, brake=self._brake_command(self.state),
                       ttc=ttc, min_dist=min_dist, stop_dist=stop_dist,
                       slow_dist=slow_dist, ttc_analytic=ttc_analytic,
                       ttc_learned=ttc_learned, ego_speed=ego_speed, raw_state=raw,
                       demand_decel=self._demand)
        if lead is not None:
            dec.lead_id, dec.lead_dist, dec.lead_closing = lead[0], lead[1].dist, lead[1].closing
            if ego_speed is not None:
                dec.lead_speed = ego_speed - lead[1].closing
        return dec
