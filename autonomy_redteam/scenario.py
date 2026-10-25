"""Structured, validated, JSON-serialisable scenario description.

A :class:`Scenario` describes one longitudinal lead-vehicle encounter that the existing
closed-loop simulator (``src/safety_sim.py``) can actually model: an ego car approaches a lead
car in the same lane; the lead car may brake; the perception stack can be degraded.

Only parameters the simulator models are included. There are deliberately no lane-change,
pedestrian or lateral-motion parameters: the framework does not claim to test them.

Replayability: a scenario plus its ``seed`` fully determines a run (all randomness comes from
one ``numpy`` generator seeded with ``seed``; see ``adapters/safety_sim_adapter.py``).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

SCHEMA_VERSION = 1


class ScenarioValidationError(ValueError):
    """Raised for out-of-bounds or inconsistent scenario parameters."""


@dataclass(frozen=True)
class ParamSpec:
    """Bounds and minimisation semantics of one continuous scenario parameter.

    group      'physical' (defines the encounter) or 'fault' (a perception perturbation that can be
               removed without changing what physically happens).
    simplest   value of a fault parameter meaning "no perturbation" (the minimiser moves towards it).
    benign     for physical parameters, the bound that makes the encounter *milder*
               ('low' or 'high'); used by the minimiser to find the mildest failing conditions.
    tol        resolution below which two values are treated as equal by the minimiser.
    """
    low: float
    high: float
    default: float
    group: str
    unit: str
    tol: float
    simplest: Optional[float] = None
    benign: Optional[str] = None
    doc: str = ''


PARAMS: Dict[str, ParamSpec] = {
    'ego_speed': ParamSpec(6.0, 20.0, 14.0, 'physical', 'm/s', 0.25, benign='low',
                           doc='ego initial speed'),
    'lead_speed': ParamSpec(0.0, 20.0, 14.0, 'physical', 'm/s', 0.25, benign='high',
                            doc='lead vehicle initial speed (0 = parked)'),
    'initial_gap': ParamSpec(6.0, 80.0, 30.0, 'physical', 'm', 0.5, benign='high',
                             doc='bumper-to-bumper distance at t = 0'),
    'lead_decel': ParamSpec(0.0, 9.0, 0.0, 'physical', 'm/s^2', 0.25, benign='low',
                            doc='lead braking deceleration (0 = never brakes)'),
    'brake_onset_s': ParamSpec(0.0, 6.0, 2.0, 'physical', 's', 0.25, benign=None,
                               doc='time at which the lead starts braking'),
    'perception_latency_s': ParamSpec(0.0, 0.6, 0.0, 'fault', 's', 0.1, simplest=0.0,
                                      doc='age of the perception output the planner acts on '
                                          '(quantised to the 0.1 s simulation step)'),
    'distance_noise_std_m': ParamSpec(0.0, 2.0, 0.0, 'fault', 'm', 0.1, simplest=0.0,
                                      doc='extra Gaussian range-noise std added to the nominal sensor noise'),
    'dropout_prob': ParamSpec(0.0, 0.7, 0.10, 'fault', '', 0.05, simplest=0.0,
                              doc='per-frame probability the lead produces no LiDAR cluster '
                                  '(0.10 = the simulator\'s nominal sensor)'),
    'detection_loss_start_s': ParamSpec(0.0, 8.0, 0.0, 'fault', 's', 0.25, simplest=0.0,
                                        doc='start of a temporary total loss of the lead detection'),
    'detection_loss_duration_s': ParamSpec(0.0, 4.0, 0.0, 'fault', 's', 0.25, simplest=0.0,
                                           doc='duration of that loss (0 = none)'),
    'ttc_error_factor': ParamSpec(0.5, 3.0, 1.0, 'fault', 'x', 0.05, simplest=1.0,
                                  doc='multiplicative error on the TTC the planner receives '
                                      '(> 1 optimistic, < 1 pessimistic); kalman planner mode only'),
    'ghost_prob': ParamSpec(0.0, 0.15, 0.04, 'fault', '', 0.02, simplest=0.0,
                            doc='per-frame probability of a one-frame false in-lane cluster '
                                '(0.04 = the simulator\'s nominal sensor)'),
}

PHYSICAL_PARAMS = tuple(k for k, v in PARAMS.items() if v.group == 'physical')
FAULT_PARAMS = tuple(k for k, v in PARAMS.items() if v.group == 'fault')

# A "fault" in the minimiser's sense can span several parameters (the detection loss is a window).
FAULT_GROUPS: Dict[str, Tuple[str, ...]] = {
    'perception_latency': ('perception_latency_s',),
    'distance_noise': ('distance_noise_std_m',),
    'sensor_dropout': ('dropout_prob',),
    'detection_loss': ('detection_loss_start_s', 'detection_loss_duration_s'),
    'ttc_error': ('ttc_error_factor',),
    'ghost_clusters': ('ghost_prob',),
}

DURATION_BOUNDS = (4.0, 30.0)
PLANNER_MODES = ('kalman', 'analytic')


@dataclass(frozen=True)
class Scenario:
    ego_speed: float = PARAMS['ego_speed'].default
    lead_speed: float = PARAMS['lead_speed'].default
    initial_gap: float = PARAMS['initial_gap'].default
    lead_decel: float = PARAMS['lead_decel'].default
    brake_onset_s: float = PARAMS['brake_onset_s'].default
    perception_latency_s: float = PARAMS['perception_latency_s'].default
    distance_noise_std_m: float = PARAMS['distance_noise_std_m'].default
    dropout_prob: float = PARAMS['dropout_prob'].default
    detection_loss_start_s: float = PARAMS['detection_loss_start_s'].default
    detection_loss_duration_s: float = PARAMS['detection_loss_duration_s'].default
    ttc_error_factor: float = PARAMS['ttc_error_factor'].default
    ghost_prob: float = PARAMS['ghost_prob'].default
    duration_s: float = 14.0
    seed: int = 0
    name: str = ''
    schema_version: int = SCHEMA_VERSION

    # -- validation ---------------------------------------------------------------------
    def problems(self, bounds: Optional[Mapping[str, Tuple[float, float]]] = None) -> List[str]:
        out: List[str] = []
        bounds = bounds or {}
        for k, spec in PARAMS.items():
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                out.append(f'{k} must be a finite number, got {v!r}')
                continue
            lo, hi = bounds.get(k, (spec.low, spec.high))
            lo, hi = max(lo, spec.low), min(hi, spec.high)
            if not lo - 1e-9 <= v <= hi + 1e-9:
                out.append(f'{k}={v} outside [{lo}, {hi}] {spec.unit}')
        d = self.duration_s
        if not isinstance(d, (int, float)) or not math.isfinite(d) or not DURATION_BOUNDS[0] <= d <= DURATION_BOUNDS[1]:
            out.append(f'duration_s={d!r} outside {DURATION_BOUNDS}')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            out.append(f'seed must be a non-negative int, got {self.seed!r}')
        if self.schema_version != SCHEMA_VERSION:
            out.append(f'unsupported schema_version {self.schema_version} (expected {SCHEMA_VERSION})')
        if not out:
            if self.lead_decel > 0 and self.brake_onset_s >= self.duration_s:
                out.append('brake_onset_s must be earlier than duration_s when lead_decel > 0')
            if self.detection_loss_duration_s > 0 and self.detection_loss_start_s >= self.duration_s:
                out.append('detection_loss_start_s must be earlier than duration_s')
        return out

    def validate(self, bounds: Optional[Mapping[str, Tuple[float, float]]] = None) -> 'Scenario':
        p = self.problems(bounds)
        if p:
            raise ScenarioValidationError('; '.join(p))
        return self

    # -- helpers --------------------------------------------------------------------------
    def replace(self, **kw) -> 'Scenario':
        return replace(self, **kw)

    def params(self) -> Dict[str, float]:
        return {k: getattr(self, k) for k in PARAMS}

    def active_faults(self) -> List[str]:
        """Fault groups whose parameters differ from 'no perturbation'."""
        out = []
        for g, keys in FAULT_GROUPS.items():
            if g == 'detection_loss':
                active = self.detection_loss_duration_s > 1e-9
            else:
                active = any(abs(getattr(self, k) - PARAMS[k].simplest) > 1e-9 for k in keys)
            if active:
                out.append(g)
        return out

    def without_faults(self, groups: Iterable[str]) -> 'Scenario':
        kw = {}
        for g in groups:
            for k in FAULT_GROUPS[g]:
                kw[k] = PARAMS[k].simplest
        return self.replace(**kw)

    def initial_required_decel(self, margin: float = 2.0) -> float:
        """Deceleration the ego would need at t = 0 to stop ``margin`` metres behind a lead holding its speed
        (0 if not closing). Scenarios that start with a large value begin *inside* a hazard."""
        closing = self.ego_speed - self.lead_speed
        if closing <= 0.1:
            return 0.0
        room = self.initial_gap - margin
        return math.inf if room <= 1e-6 else closing * closing / (2.0 * room)

    def fingerprint(self) -> str:
        """Stable hash of everything that affects the simulation (the name does not)."""
        d = asdict(self)
        d.pop('name', None)
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    # -- serialisation ----------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, d: Mapping, validate: bool = True) -> 'Scenario':
        if not isinstance(d, Mapping):
            raise ScenarioValidationError('scenario must be a JSON object')
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ScenarioValidationError(f'unknown scenario fields: {sorted(unknown)}')
        s = cls(**d)
        return s.validate() if validate else s

    @classmethod
    def from_json(cls, text: str, validate: bool = True) -> 'Scenario':
        try:
            return cls.from_dict(json.loads(text), validate)
        except json.JSONDecodeError as e:
            raise ScenarioValidationError(f'invalid JSON: {e}') from e

    @classmethod
    def load(cls, path) -> 'Scenario':
        path = Path(path)
        try:
            text = path.read_text()
        except OSError as e:
            raise ScenarioValidationError(f'cannot read {path}: {e}') from e
        d = json.loads(text) if text.strip() else {}
        if isinstance(d, dict) and 'scenario' in d and isinstance(d['scenario'], dict):
            d = d['scenario']              # accept result files that wrap the scenario
        return cls.from_dict(d)

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + '\n')
        return path

    def describe(self) -> str:
        f = self.active_faults()
        return (f'ego {self.ego_speed:.1f} m/s, lead {self.lead_speed:.1f} m/s, gap {self.initial_gap:.1f} m, '
                f'lead decel {self.lead_decel:.1f} m/s^2 @ {self.brake_onset_s:.1f}s, '
                f'faults: {", ".join(f) if f else "none"}, seed {self.seed}')


INITIAL_DECEL_LIMIT = 2.5     # m/s^2: generated scenarios must not *start* in an imminent hazard (see repair below)


# --- deterministic generation ------------------------------------------------------------------
class ScenarioGenerator:
    """Deterministic scenario sampler with configurable (validated) bounds.

    Fault parameters are sampled "spike and slab": each fault group is active with probability
    ``fault_prob`` (otherwise exactly 'no perturbation'), so the search sees both clean runs and
    combinations of faults and the minimiser has real subsets to reduce.
    """

    def __init__(self, seed: int = 0, bounds: Optional[Mapping[str, Tuple[float, float]]] = None,
                 fault_prob: float = 0.5, max_initial_decel: Optional[float] = INITIAL_DECEL_LIMIT):
        self.bounds = dict(self._check_bounds(bounds or {}))
        self.seed = int(seed)
        self.fault_prob = float(fault_prob)
        self.max_initial_decel = max_initial_decel
        self._rng = np.random.default_rng(np.random.SeedSequence([self.seed, 0xA17]))
        self._n = 0

    @staticmethod
    def _check_bounds(bounds):
        for k, (lo, hi) in bounds.items():
            if k not in PARAMS:
                raise ScenarioValidationError(f'unknown parameter in bounds: {k}')
            s = PARAMS[k]
            if not (s.low <= lo <= hi <= s.high):
                raise ScenarioValidationError(f'bounds for {k} must satisfy {s.low} <= lo <= hi <= {s.high}')
        return bounds

    def bound(self, k: str) -> Tuple[float, float]:
        s = PARAMS[k]
        lo, hi = self.bounds.get(k, (s.low, s.high))
        return lo, hi

    def sample(self, name_prefix: str = 'gen') -> Scenario:
        r = self._rng
        vals = {}
        for k in PHYSICAL_PARAMS:
            lo, hi = self.bound(k)
            vals[k] = float(r.uniform(lo, hi))
        # lead braking is a discrete choice: half the encounters have a steady lead
        if r.random() < 0.25:
            vals['lead_decel'] = 0.0
        for g, keys in FAULT_GROUPS.items():
            on = r.random() < self.fault_prob
            for k in keys:
                lo, hi = self.bound(k)
                draw = float(r.uniform(lo, hi))
                vals[k] = draw if on else PARAMS[k].simplest
        if vals['detection_loss_duration_s'] <= 1e-9:
            vals['detection_loss_start_s'] = 0.0
        vals = {k: float(np.clip(round(v, 3), *self._lohi(k))) for k, v in vals.items()}
        vals = self._repair(vals)
        seed = int(self._rng.integers(0, 2 ** 31 - 1))
        self._n += 1
        return Scenario(**vals, seed=seed, name=f'{name_prefix}-{self._n:04d}').validate(self.bounds)

    def _lohi(self, k):
        return self.bound(k)

    def _repair(self, vals: Dict[str, float]) -> Dict[str, float]:
        """Make the encounter *develop* during the run instead of starting inside an imminent hazard.

        A parked car 15 m ahead of a 15 m/s ego is a physical set-up problem, not a planner test; real AEB test
        protocols start the approach with the target far away. If the required deceleration at t = 0 exceeds
        ``max_initial_decel``, first increase the gap, then (gap at its bound) lower the ego speed.
        """
        lim = self.max_initial_decel
        if lim is None:
            return vals
        v = dict(vals)
        closing = v['ego_speed'] - v['lead_speed']
        if closing <= 0.1:
            return v
        margin = 2.0
        need_gap = margin + closing * closing / (2.0 * lim)
        glo, ghi = self.bound('initial_gap')
        if need_gap <= ghi:
            v['initial_gap'] = round(min(ghi, max(v['initial_gap'], need_gap) + 1e-3), 3)
            return v
        v['initial_gap'] = ghi
        max_closing = math.sqrt(2.0 * lim * (ghi - margin))
        elo, ehi = self.bound('ego_speed')
        v['ego_speed'] = round(max(elo, min(v['ego_speed'], v['lead_speed'] + max_closing) - 1e-3), 3)
        if v['ego_speed'] - v['lead_speed'] > max_closing:        # ego already at its lower bound
            llo, lhi = self.bound('lead_speed')
            v['lead_speed'] = round(min(lhi, v['ego_speed'] - max_closing + 1e-3), 3)
        return v

    # -- vector view (used by the adaptive search) ----------------------------------------------
    @staticmethod
    def vector_keys() -> Tuple[str, ...]:
        return tuple(PARAMS)

    def to_unit(self, s: Scenario) -> np.ndarray:
        out = []
        for k in PARAMS:
            lo, hi = self.bound(k)
            out.append(0.0 if hi <= lo else (getattr(s, k) - lo) / (hi - lo))
        return np.array(out)

    def from_unit(self, u: np.ndarray, seed: int, name: str = '') -> Scenario:
        vals = {}
        for x, k in zip(np.clip(u, 0.0, 1.0), PARAMS):
            lo, hi = self.bound(k)
            vals[k] = float(round(lo + x * (hi - lo), 3))
        # a fault that is "almost off" is off: avoids meaningless 0.001 s detection windows etc.
        for k in FAULT_PARAMS:
            sp = PARAMS[k]
            if abs(vals[k] - sp.simplest) < 0.5 * sp.tol:
                vals[k] = sp.simplest
        if vals['detection_loss_duration_s'] <= 1e-9:
            vals['detection_loss_start_s'] = 0.0
        vals = self._repair(vals)
        return Scenario(**vals, seed=int(seed), name=name).validate(self.bounds)
