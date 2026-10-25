"""Deterministic safety metrics and failure classification. No LLM, no randomness.

Every definition is programmatic (see ``FailureCriteria`` for the thresholds).

collision          the lead's rear bumper reached the ego front bumper (gap <= 0).
impact_speed       relative speed ego - lead at that step (m/s).
min_gap            minimum bumper gap over the run, clipped at 0 (m).
min_ttc            minimum ground-truth TTC = gap / closing_speed over steps with closing_speed > 0.1 (s).
required decel     a_req = closing^2 / (2 (gap - margin)), margin = 2 m; the constant deceleration that
                   removes the closing speed while keeping the margin, lead assumed to hold its speed.
hazard             maximal run of steps with a_req >= hazard_decel (3 m/s^2): braking harder than "comfortable"
                   is needed to stay safe.
brake_reaction_time  from the first step of the first hazard to the first step whose commanded deceleration
                   is >= 0.5 m/s^2 (0 if already braking); null if there was no hazard or no response.
missed_hazard      a hazard that lasted at least response_window (1.5 s) or ended in a collision, during which
                   the planner never commanded >= 0.5 m/s^2.
unnecessary_braking  hard braking = a maximal run of steps with commanded deceleration > 3 m/s^2 (STOPPED-level). A hard-braking
                   run is *unnecessary* if, at its first step (before braking changed anything), ground truth said it was
                   not needed: not closing, or a_req < 1 m/s^2. Seconds = total length of unnecessary runs; flagged if
                   >= 0.3 s. (Judging necessity at the onset avoids counting the tail of a justified stop, when braking
                   has already made a_req small.) Note this is relative to a constant-speed lead: earlier braking that
                   anticipates a lead's *future* braking counts as unnecessary.
near_miss          no collision but min_gap < 0.5 m.
perception anomaly lead not tracked (or not detected) while closing at < 40 m, and range errors of the tracked
                   lead: an observable symptom, NOT a causal claim (causality is the job of diagnose.py).

Failure kinds (``assess``): collision (only if physically avoidable and a competent reference policy avoids it:
see oracle.py), near_miss (only if the reference policy keeps >= 0.5 m), missed_hazard, unnecessary_braking.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .oracle import OracleResult, required_decel, run_oracle
from .sut import Episode, STATES

KIND_ORDER = ('collision', 'near_miss', 'missed_hazard', 'unnecessary_braking')
EXCLUDED_COLLISION_PENALTY = 25.0


@dataclass(frozen=True)
class FailureCriteria:
    near_miss_gap: float = 0.5
    hazard_decel: float = 3.0
    hazard_margin: float = 2.0
    response_cmd: float = 0.5
    response_window: float = 1.5
    unnecessary_cmd: float = 3.0
    unnecessary_req: float = 1.0
    unnecessary_min_s: float = 0.3
    perception_range: float = 40.0
    kinds: tuple = KIND_ORDER                  # which kinds count as a failure
    include_hard_collisions: bool = False      # count collisions the reference policy also cannot avoid
    anomaly_lost_steps: int = 3
    anomaly_range_error: float = 2.0
    reference_delay_s: float = 0.2             # = (tracker3d.CONFIRM_HITS - 1) * DT, see oracle.py


DEFAULT_CRITERIA = FailureCriteria()


def _f(x: float) -> Optional[float]:
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else float(x)


@dataclass
class Assessment:
    # raw metrics
    collision: bool
    initial_required_decel: float
    impact_speed: float
    min_gap: float
    min_ttc: float
    brake_reaction_time: Optional[float]
    hazard_count: int
    hazard_onset: Optional[float]
    missed_hazards: int
    unnecessary_braking_s: float
    near_miss: bool
    states_visited: List[str]
    max_state: str
    peak_cmd_decel: float
    first_brake_t: Optional[float]
    lead_lost_steps: int
    max_range_error: float
    mean_abs_range_error: float
    perception_anomaly: bool
    # oracle
    oracle_max_brake_collides: bool
    oracle_reference_collides: bool
    oracle_reference_min_gap: float
    collision_class: str                 # none | unavoidable | hard | avoidable
    # verdict
    failure_kinds: List[str] = field(default_factory=list)
    primary_kind: Optional[str] = None
    severity: float = 0.0
    robustness: float = 0.0              # lower = closer to / deeper into failure (collision: -impact speed)
    objective: float = 0.0               # robustness with excluded (unavoidable/hard) collisions pushed to a plateau

    @property
    def failed(self) -> bool:
        return bool(self.failure_kinds)

    def to_dict(self) -> dict:
        d = {k: (_f(v) if isinstance(v, float) else v) for k, v in asdict(self).items()}
        d['failed'] = self.failed
        return d


def _runs(mask: np.ndarray):
    """Maximal [start, end) runs of True."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def assess(ep: Episode, criteria: FailureCriteria = DEFAULT_CRITERIA) -> Assessment:
    c, dt = criteria, ep.limits.dt
    recs = ep.records
    gap = np.array([r.gt.gap for r in recs])
    closing = np.array([r.gt.closing_speed for r in recs])
    cmd = np.array([r.planner.cmd_decel for r in recs])
    t = np.array([r.t for r in recs])
    collided = any(r.collided for r in recs)
    impact = 0.0
    for e in ep.events:
        if e.kind == 'collision':
            try:
                impact = float(e.detail.split('speed')[1].split('m/s')[0])
            except (IndexError, ValueError):
                impact = 0.0
    min_gap = float(min([r.gap_after for r in recs] + [gap[0]])) if len(recs) else math.inf
    ttcs = [r.gt.ttc for r in recs if math.isfinite(r.gt.ttc)]
    min_ttc = min(ttcs) if ttcs else math.inf

    a_req = np.array([required_decel(g, v, c.hazard_margin) for g, v in zip(gap, closing)])
    hazard = a_req >= c.hazard_decel
    runs = _runs(hazard)
    reaction, missed, onset = None, 0, None
    win = int(round(c.response_window / dt))
    for n, (a, b) in enumerate(runs):
        if n == 0:
            onset = float(t[a])
        resp = np.nonzero(cmd[a:b] >= c.response_cmd)[0]
        if n == 0 and len(resp):
            reaction = float(t[a + resp[0]] - t[a])
        ended_in_collision = collided and b == len(recs)
        if not len(resp) and ((b - a) >= win or ended_in_collision):
            missed += 1
    unnecessary_s = 0.0
    for a, b in _runs(cmd > c.unnecessary_cmd):
        if closing[a] <= 0.1 or a_req[a] < c.unnecessary_req:
            unnecessary_s += (b - a) * dt

    states = [r.planner.state for r in recs]
    visited = [s for s in STATES if s in states]
    first_brake = next((float(r.t) for r in recs if r.planner.cmd_decel >= c.response_cmd), None)

    near = (closing > 0.1) & (gap < c.perception_range)
    lost = int(sum(1 for r, n in zip(recs, near) if n and not r.perception.tracked))
    errs = np.array([r.perception.range_error for r, n in zip(recs, near) if n and r.perception.tracked])
    max_err = float(np.max(np.abs(errs))) if len(errs) else 0.0
    mean_err = float(np.mean(np.abs(errs))) if len(errs) else 0.0
    anomaly = lost >= c.anomaly_lost_steps or max_err >= c.anomaly_range_error

    mb = run_oracle(ep.scenario, 'max_brake', ep.limits)
    ref = run_oracle(ep.scenario, 'reference', ep.limits, c.reference_delay_s)
    if not collided:
        ccls = 'none'
    elif mb.collided:
        ccls = 'unavoidable'
    elif ref.collided:
        ccls = 'hard'
    else:
        ccls = 'avoidable'

    kinds = []
    if collided and (ccls == 'avoidable' or (ccls == 'hard' and c.include_hard_collisions)):
        kinds.append('collision')
    near_miss = (not collided) and min_gap < c.near_miss_gap
    if near_miss and ref.min_gap >= c.near_miss_gap and not mb.collided:
        kinds.append('near_miss')
    if missed > 0 and not (collided and ccls == 'unavoidable'):
        kinds.append('missed_hazard')
    if unnecessary_s >= c.unnecessary_min_s:
        kinds.append('unnecessary_braking')
    kinds = [k for k in KIND_ORDER if k in kinds and k in c.kinds]
    primary = kinds[0] if kinds else None

    severity = 0.0
    if 'collision' in kinds:
        severity = 100.0 + 10.0 * impact
    elif 'near_miss' in kinds:
        severity = 50.0 + 20.0 * (c.near_miss_gap - min_gap)
    elif 'missed_hazard' in kinds:
        severity = 30.0 + 5.0 * missed
    elif 'unnecessary_braking' in kinds:
        severity = 10.0 + unnecessary_s

    robustness = -impact if collided else min_gap
    objective = robustness
    if collided and 'collision' not in kinds:
        objective = EXCLUDED_COLLISION_PENALTY            # do not let the search chase impossible collisions
    return Assessment(
        collision=collided, initial_required_decel=ep.scenario.initial_required_decel(), impact_speed=impact, min_gap=min_gap, min_ttc=min_ttc,
        brake_reaction_time=reaction, hazard_count=len(runs), hazard_onset=onset, missed_hazards=missed,
        unnecessary_braking_s=unnecessary_s, near_miss=near_miss, states_visited=visited,
        max_state=visited[-1] if visited else 'DRIVE', peak_cmd_decel=float(cmd.max()) if len(cmd) else 0.0,
        first_brake_t=first_brake, lead_lost_steps=lost, max_range_error=max_err,
        mean_abs_range_error=mean_err, perception_anomaly=anomaly,
        oracle_max_brake_collides=mb.collided, oracle_reference_collides=ref.collided,
        oracle_reference_min_gap=ref.min_gap, collision_class=ccls, failure_kinds=kinds, primary_kind=primary,
        severity=severity, robustness=robustness, objective=objective)
