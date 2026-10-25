"""Adapter: ``src/safety_sim.py`` actors/dynamics + ``Tracker3D`` + ``BehaviorManager`` as a SystemUnderTest.

What is reused unchanged from the repository
  * ``safety_sim``: Actor/car, DT, MAX_DECEL, LAG, EGO_*, RSSPolicy (the reference braking rule)
  * ``tracker3d``: Cluster, Tracker3D, in_ego_path          (the production tracker)
  * ``behavior_manager.BehaviorManager``                     (the production planner, FSM + TTC rules)

What is new here (``safety_sim.run_episode`` is a closed loop with a hardcoded sensor model, so it
cannot be stepped or perturbed): a stepping loop with the *same* order of operations and dynamics,
and a parameterised sensor model with the fault hooks of ``Scenario``:

  nominal sensor (as in safety_sim.measure) : 0.1 m + 0.2 % of range cluster noise, 55 m range,
                                              lateral noise 0.1 m
  dropout_prob / ghost_prob                 : per-frame Bernoulli, nominal 0.10 / 0.04
  distance_noise_std_m                      : extra range noise, added in quadrature
  detection loss window                     : the lead produces no cluster for [start, start + duration)
  perception_latency_s                      : the tracker and planner receive data that is
                                              round(latency / DT) steps old (stale ranges, no
                                              ego-motion compensation: the worst case for a planner)
  ttc_error_factor                          : multiplies the per-track Kalman TTC given to the planner
                                              (only ``kalman`` planner mode reads it)

Determinism: all randomness is one ``default_rng(seed)``; every step draws the same number of
variates regardless of which faults are enabled, so removing a fault in a counterfactual run does
not shift the noise sequence of the others.
"""
from __future__ import annotations

import contextlib
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np

_SRC = str(Path(__file__).resolve().parents[2] / 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import config                      # noqa: E402  (repo module)
import safety_sim as ss            # noqa: E402
import tracker3d as t3             # noqa: E402
from behavior_manager import BehaviorManager, BehaviorState  # noqa: E402
from safe_braking import PLANNERS, make_behavior_manager  # noqa: E402

from ..scenario import FAULT_PARAMS, PARAMS, Scenario  # noqa: E402
from ..sut import (Episode, GroundTruthState, Interventions, PerceptionTelemetry,  # noqa: E402
                   PlannerTelemetry, SafetyEvent, StepRecord, SystemUnderTest, VehicleLimits)

DETECTION_RANGE = 55.0     # m, cars (safety_sim.measure)
POLICIES = ('planner', 'reference', 'max_brake', 'none')


@contextlib.contextmanager
def patched_config(overrides: Optional[Dict[str, float]]):
    """Temporarily override ``config.*`` planner thresholds (used for planner-variant / regression experiments)."""
    old = {}
    try:
        for k, v in (overrides or {}).items():
            if not hasattr(config, k):
                raise KeyError(f'unknown config key {k!r}')
            old[k] = getattr(config, k)
            setattr(config, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(config, k, v)


class SafetySimSUT(SystemUnderTest):
    name = 'safety_sim'
    supported_faults = frozenset(FAULT_PARAMS)
    supported_interventions = frozenset({'oracle_distance', 'oracle_ttc', 'reference', 'max_brake', 'none'})

    def __init__(self, planner_mode: str = 'kalman', config_overrides: Optional[Dict[str, float]] = None,
                 planner: str = 'baseline'):
        if planner_mode not in ('kalman', 'analytic'):
            raise ValueError("planner_mode must be 'kalman' or 'analytic'")
        if planner not in PLANNERS:
            raise ValueError(f'planner must be one of {PLANNERS}')
        self.planner = planner                  # braking policy: 'baseline' (original BehaviorManager) | 'improved'
        self.planner_mode = planner_mode
        self.config_overrides = dict(config_overrides or {})
        self._scenario: Optional[Scenario] = None
        self._iv = Interventions()
        self._events: List[SafetyEvent] = []

    # -- configuration ---------------------------------------------------------------------------
    def describe(self) -> Dict:
        return {'name': self.name, 'planner': self.planner, 'planner_mode': self.planner_mode,
                'config_overrides': dict(sorted(self.config_overrides.items())),
                'dt': ss.DT, 'max_decel': ss.MAX_DECEL, 'actuator_lag': ss.LAG}

    def limits(self) -> VehicleLimits:
        return VehicleLimits(dt=ss.DT, max_decel=ss.MAX_DECEL, actuator_lag=ss.LAG, lead_length=4.5)

    def check_supported(self, scenario: Scenario, interventions: Interventions) -> List[str]:
        missing = []
        if self.planner_mode == 'analytic':
            # the analytic planner computes its own range-rate TTC inside BehaviorManager: there is no input to perturb
            if abs(scenario.ttc_error_factor - 1.0) > 1e-9:
                missing.append('ttc_error_factor (analytic planner mode has no TTC input)')
            if interventions.oracle_ttc:
                missing.append('oracle_ttc (analytic planner mode has no TTC input)')
        if interventions.policy not in POLICIES:
            missing.append(f'policy {interventions.policy!r}')
        return missing

    # -- lifecycle ------------------------------------------------------------------------------
    def initialize(self, scenario: Scenario, interventions: Optional[Interventions] = None) -> None:
        scenario.validate()
        iv = interventions or Interventions()
        missing = self.check_supported(scenario, iv)
        if missing:
            raise ValueError('not supported by this SUT: ' + '; '.join(missing))
        self._scenario, self._iv = scenario, iv
        self.reset()

    def reset(self) -> None:
        if self._scenario is None:
            raise RuntimeError('initialize() first')
        s = self._scenario
        self._rng = np.random.default_rng(s.seed)
        self._k = 0
        self._ego_x, self._v, self._a_act = 0.0, s.ego_speed, 0.0
        lead = ss.car(s.initial_gap + 4.5 / 2, 0.0, s.lead_speed)
        decel, onset = s.lead_decel, s.brake_onset_s

        def script(a, t, ego_x, dt):
            if decel > 0 and t >= onset:
                a.vx = max(a.vx - decel * dt, 0.0)
        lead.script = script
        self._lead = lead
        self._tracker = t3.Tracker3D()
        self._pm = make_behavior_manager(self.planner, self.planner_mode)
        self._rss = ss.RSSPolicy()
        self._delay = int(round(s.perception_latency_s / ss.DT))
        self._buf: List[tuple] = []
        self._events = []
        self._done = False
        self._collided = False
        self._prev_state = BehaviorState.GO
        self._braking_started = False
        self._loss_on = False
        self._onset_logged = False
        self._last = None
        self._gt = self._make_gt(0.0)
        self._perc = PerceptionTelemetry(False, 0, False)
        self._plan = PlannerTelemetry('DRIVE', 'DRIVE', 0.0)

    @property
    def done(self) -> bool:
        return self._done

    # -- ground truth ---------------------------------------------------------------------------------
    def _make_gt(self, t: float) -> GroundTruthState:
        a = self._lead
        gap = a.x - a.length / 2 - self._ego_x
        closing = self._v - a.vx
        ttc = gap / closing if closing > 0.1 and gap > 0 else math.inf
        return GroundTruthState(t=t, ego_x=self._ego_x, ego_v=self._v, ego_decel=self._a_act,
                                lead_x=a.x - a.length / 2, lead_v=a.vx, gap=gap, closing_speed=closing,
                                ttc=ttc, collided=self._collided)

    def ground_truth(self) -> GroundTruthState:
        return self._gt

    def perception_telemetry(self) -> PerceptionTelemetry:
        return self._perc

    def planner_telemetry(self) -> PlannerTelemetry:
        return self._plan

    def events(self) -> List[SafetyEvent]:
        return list(self._events)

    # -- sensor model ---------------------------------------------------------------------------------------
    def _measure(self, t: float):
        """Clusters produced at time t (before latency). Returns (clusters, lead_cluster_present)."""
        s, rng = self._scenario, self._rng
        # fixed-length draw sequence (see module docstring)
        u_drop, n_rng, n_lat = rng.random(), rng.normal(), rng.normal()
        u_ghost, g_x, g_y = rng.random(), rng.uniform(10, 40), rng.uniform(-1, 1)
        a, ego_x = self._lead, self._ego_x
        clusters, lead_present = [], False
        rear = a.x - a.length / 2 - ego_x
        in_loss = (s.detection_loss_duration_s > 0 and
                   s.detection_loss_start_s <= t < s.detection_loss_start_s + s.detection_loss_duration_s)
        behind = a.x + a.length / 2 < ego_x - 1.0 or (rear < 0.3 and a.x < ego_x)
        r = max(rear, 0.0)
        if not behind and r <= DETECTION_RANGE and u_drop >= s.dropout_prob and not in_loss:
            if self._iv.oracle_distance:
                sx, sy = 0.0, 0.0
            else:
                sx = math.hypot(0.1 + 0.002 * r, s.distance_noise_std_m)
                sy = 0.1
            near = max(rear + n_rng * sx, 0.1)
            depth = 0.8
            clusters.append(t3.Cluster(x=near + depth / 2, y=a.y + n_lat * sy, z=0.8, x_near=near,
                                       ext_x=depth, ext_y=a.width, n=40, z_top=1.5))
            lead_present = True
        if u_ghost < s.ghost_prob:
            clusters.append(t3.Cluster(x=g_x, y=g_y, z=0.8, x_near=g_x - 0.2, ext_x=0.4, ext_y=0.4,
                                       n=12, z_top=1.2))
        return clusters, lead_present, in_loss

    # -- decision -------------------------------------------------------------------------------------------
    def _plan_step(self, tracks, clusters, pose, v, dt, lead_world_x, gt_ttc):
        """Mirror of safety_sim.PlannerPolicy.decide with fault / oracle hooks and Decision capture."""
        s = self._scenario
        obs = []
        for tr in tracks:
            raw = [c.x_near for c in clusters
                   if abs(c.x - tr.rel_pos[0]) < 1.5 and abs(c.y - tr.rel_pos[1]) < 1.5]
            ttc3 = tr.ttc()[0]
            if self._iv.oracle_ttc:
                is_lead = lead_world_x is not None and abs(tr.state[0] - lead_world_x) < 3.0
                ttc3 = gt_ttc if is_lead else math.inf
            elif math.isfinite(ttc3):
                ttc3 = ttc3 * s.ttc_error_factor
            obs.append({'id': tr.id, 'dist': min(raw) if raw else tr.bumper_range,
                        'in_path': t3.in_ego_path(tr), 'ttc_3d': ttc3, 'closing_3d': tr.closing_speed, 'track_hits': tr.hits})
        lid = min([c.x_near for c in clusters
                   if c.x_near > 0 and abs(c.y) < config.PATH_MAX_LATERAL + 0.5 * c.ext_y], default=99.0)
        dec = self._pm.update_state(obs, lid, dt=dt, ego_speed=v)
        return ss.MAX_DECEL * dec.brake, dec

    # -- step ------------------------------------------------------------------------------------------------
    def step(self) -> StepRecord:
        if self._scenario is None:
            raise RuntimeError('initialize() first')
        if self._done:
            raise RuntimeError('episode finished')
        with patched_config(self.config_overrides):
            return self._step()

    def _step(self) -> StepRecord:
        s, dt = self._scenario, ss.DT
        k = self._k
        t = k * dt
        a = self._lead
        events: List[SafetyEvent] = []
        self._gt = gt = self._make_gt(t)
        pose = np.eye(4)
        pose[0, 3] = self._ego_x

        clusters, lead_present, in_loss = self._measure(t)
        self._buf.append((clusters, lead_present, pose, self._v, a.x, gt.ttc))
        if in_loss != self._loss_on:
            events.append(SafetyEvent(t, 'detection_loss_start' if in_loss else 'detection_loss_end'))
            self._loss_on = in_loss
        if s.lead_decel > 0 and t >= s.brake_onset_s and not self._onset_logged:
            events.append(SafetyEvent(t, 'lead_brake_onset', f'{s.lead_decel:.1f} m/s^2'))
            self._onset_logged = True

        # perception output the stack sees now (delay steps old)
        if len(self._buf) > self._delay:
            d_clusters, d_lead, d_pose, d_v, d_lead_x, d_gt_ttc = self._buf.pop(0)
            tracks = self._tracker.update(d_clusters, d_pose, dt, np.array([d_v, 0.0]))
            have_data = True
            lead_world_x = d_lead_x
        else:
            d_clusters, d_lead, tracks, have_data, lead_world_x, d_gt_ttc = [], False, [], False, None, math.inf

        # policy
        pol = self._iv.policy
        dec = None
        if pol == 'planner':
            if have_data:
                cmd, dec = self._plan_step(tracks, d_clusters, d_pose, self._v, dt, lead_world_x, d_gt_ttc)
            else:
                cmd, dec = self._plan_step([], [], pose, self._v, dt, None, math.inf)
        elif pol == 'reference':
            cmd = self._rss.decide(tracks, d_clusters, pose, self._v, dt) if have_data else 0.0
        elif pol == 'max_brake':
            cmd = ss.MAX_DECEL
        else:
            cmd = 0.0
        cmd = float(np.clip(cmd, 0.0, ss.MAX_DECEL))

        # perception telemetry: nearest in-path track = what the planner treats as lead
        raw_in = [c.x_near for c in d_clusters
                  if c.x_near > 0 and abs(c.y) < config.PATH_MAX_LATERAL + 0.5 * c.ext_y]
        raw_range = min(raw_in) if raw_in else math.nan
        inpath = [tr for tr in tracks if t3.in_ego_path(tr)]
        if inpath:
            lead_tr = min(inpath, key=lambda tr: tr.bumper_range)
            est_gap, est_closing = lead_tr.bumper_range, lead_tr.closing_speed
            ttc_raw = lead_tr.ttc()[0]
            est_ttc = d_gt_ttc if self._iv.oracle_ttc else ttc_raw * s.ttc_error_factor
            perc = PerceptionTelemetry(d_lead, len(d_clusters), True, est_gap, est_closing, est_ttc,
                                       est_gap - gt.gap, raw_range)
        else:
            perc = PerceptionTelemetry(d_lead, len(d_clusters), False, raw_range=raw_range)
        if dec is not None:
            plan = PlannerTelemetry(dec.state, dec.raw_state, cmd, dec.ttc, dec.min_dist)
        else:
            st = 'STOPPED' if cmd >= 0.9 * ss.MAX_DECEL else ('SLOWING' if cmd > 0.5 else 'DRIVE')
            plan = PlannerTelemetry(st, st, cmd)
        if plan.state != self._prev_state:
            events.append(SafetyEvent(t, 'state_change', f'{self._prev_state} -> {plan.state}'))
            self._prev_state = plan.state
        if cmd >= 0.5 and not self._braking_started:
            events.append(SafetyEvent(t, 'first_brake_command', f'{cmd:.1f} m/s^2'))
            self._braking_started = True

        # dynamics (same order as safety_sim.run_episode)
        self._a_act += (cmd - self._a_act) * dt / ss.LAG
        self._v = max(self._v - self._a_act * dt, 0.0)
        self._ego_x += self._v * dt
        a.step(t, self._ego_x, dt)
        gap_after = a.x - a.length / 2 - self._ego_x
        if gap_after <= 0 and not self._collided:
            self._collided = True
            events.append(SafetyEvent(t + dt, 'collision',
                                      f'relative impact speed {max(self._v - a.vx, 0.0):.2f} m/s'))
        self._k += 1
        if self._collided or (self._v < 0.05 and t > 3.0) or self._k >= int(s.duration_s / dt):
            self._done = True
            if not self._collided and self._v < 0.05:
                events.append(SafetyEvent(t + dt, 'ego_stopped', f'gap {gap_after:.2f} m'))
        self._perc, self._plan = perc, plan
        self._events.extend(events)
        rec = StepRecord(t=t, gt=gt, perception=perc, planner=plan,
                         gap_after=max(gap_after, 0.0), collided=self._collided, events=events)
        rec.gt.collided = self._collided
        return rec


def default_sut_factory(planner_mode: str = 'kalman', config_overrides: Optional[Dict[str, float]] = None,
                        planner: str = 'baseline'):
    """Returns a zero-argument callable creating a fresh SafetySimSUT (used by search workers / tests)."""
    def make():
        return SafetySimSUT(planner_mode=planner_mode, config_overrides=config_overrides, planner=planner)
    return make
