"""Feasibility oracles: is a collision physically unavoidable, and could a competent policy avoid it?

Both oracles are pure kinematic integrations of the scenario's *physical* parameters (ego speed, lead
speed / braking, gap) with the SUT's own actuator model, so they are cheap and need no tracker.

``max_brake``  : perfect information, brakes at the actuator limit from t = 0. If even this collides, the
                 collision is **physically unavoidable** under the model (longitudinal braking only; no
                 steering, no acceleration away from the lead). No planner can be blamed for it.
``reference``  : perfect information, the longitudinal RSS rule of ``safety_sim.RSSPolicy`` (worst-case lead
                 braking 8 m/s^2), but acting on the world as it was ``delay_s`` ago (default 0.2 s = the production
                 tracker's minimum confirmation time, ``(CONFIRM_HITS - 1) * dt``). A *competent* avoider that is not
                 granted clairvoyance; if it avoids a collision the planner did not, the failure is attributed to the
                 planner stack.

Classification used by the framework (see ``metrics.collision_class``)
    unavoidable : max_brake collides
    hard        : max_brake avoids, reference (with its confirmation delay) collides  (excluded by default)
    avoidable   : reference avoids
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace

import numpy as np

from .scenario import Scenario
from .sut import VehicleLimits

_PHYS = ('ego_speed', 'lead_speed', 'initial_gap', 'lead_decel', 'brake_onset_s', 'duration_s')


@dataclass(frozen=True)
class OracleResult:
    policy: str
    collided: bool
    min_gap: float
    impact_speed: float       # relative speed at impact (0 if no collision)


def _rss_cmd(v, gap, lead_v, limits: VehicleLimits, rss) -> float:
    """safety_sim.RSSPolicy applied to ground truth (reuses that class so the rule cannot drift)."""
    tr = SimpleNamespace(rel_pos=np.array([gap + 0.4, 0.0]), ext_y=1.8, bumper_range=gap,
                         closing_speed=v - lead_v)
    return rss.decide([tr], [], None, v, limits.dt)


@lru_cache(maxsize=65536)
def _simulate(phys: tuple, policy: str, dt: float, max_decel: float, lag: float, lead_len: float,
              delay_s: float = 0.0) -> OracleResult:
    ego_speed, lead_speed, gap0, decel, onset, duration = phys
    import safety_sim as ss     # imported lazily: adapters/ puts src/ on sys.path
    rss = ss.RSSPolicy()
    limits = VehicleLimits(dt, max_decel, lag, lead_len)
    ego_x, v, a_act = 0.0, ego_speed, 0.0
    lead_rear, lead_v = gap0, lead_speed
    min_gap, collided, impact = gap0, False, 0.0
    delay = int(round(delay_s / dt))
    hist = []                                   # (gap, lead speed) as measured at each past step
    for k in range(int(duration / dt)):
        t = k * dt
        gap = lead_rear - ego_x
        hist.append((gap, lead_v))
        if policy == 'max_brake':
            cmd = max_decel
        elif policy == 'reference':
            if len(hist) > delay:
                g_old, lv_old = hist[-1 - delay]
                cmd = _rss_cmd(v, g_old, lv_old, limits, rss)
            else:
                cmd = 0.0                       # nothing observed yet
        else:
            cmd = 0.0
        cmd = float(np.clip(cmd, 0.0, max_decel))
        a_act += (cmd - a_act) * dt / lag
        v = max(v - a_act * dt, 0.0)
        ego_x += v * dt
        if decel > 0 and t >= onset:
            lead_v = max(lead_v - decel * dt, 0.0)
        lead_rear += lead_v * dt
        gap_after = lead_rear - ego_x
        min_gap = min(min_gap, max(gap_after, 0.0))
        if gap_after <= 0:
            collided, impact = True, max(v - lead_v, 0.0)
            break
        if v < 0.05 and t > 3.0:
            break
    return OracleResult(policy, collided, min_gap, impact)


def run_oracle(scenario: Scenario, policy: str, limits: VehicleLimits, delay_s: float = 0.0) -> OracleResult:
    phys = tuple(round(float(getattr(scenario, k)), 9) for k in _PHYS)
    return _simulate(phys, policy, limits.dt, limits.max_decel, limits.actuator_lag, limits.lead_length,
                     delay_s if policy == 'reference' else 0.0)


def required_decel(gap: float, closing: float, margin: float = 2.0) -> float:
    """Constant deceleration needed to remove the closing speed while keeping ``margin`` metres of gap,
    assuming the lead holds its current speed. 0 if not closing; inf if the margin is already gone."""
    if closing <= 0.1:
        return 0.0
    room = gap - margin
    if room <= 1e-6:
        return math.inf
    return closing * closing / (2.0 * room)
