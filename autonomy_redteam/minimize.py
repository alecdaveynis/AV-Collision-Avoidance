"""Failure minimisation inspired by delta debugging (Zeller & Hildebrandt, 2002).

Goal: keep the failure, drop everything that is not needed to reproduce it.

Predicate   ``fails(s)``: the scenario still exhibits the *same failure kind* as the original (e.g. an
            avoidable collision) when replayed with its seed. With ``seeds=K`` > 1 the predicate requires the
            kind in at least half of K derived noise seeds (guards against fitting one noise draw).

Phases      1. fault removal   : ``ddmin`` over the set of active perturbation groups (latency, noise, dropout,
                                  detection loss, TTC error, ghost clusters) -> a 1-minimal necessary set.
            2. fault severity  : per remaining fault parameter, bisection towards "no perturbation".
            3. physical simplification: drop lead braking if it is not needed, then bisect each physical parameter
                                  towards its *benign* bound (lower ego speed, higher lead speed, larger gap,
                                  gentler braking) to find the mildest failing conditions.
            4. severity again   : phase 3 can change what is necessary.
            5. snap to round numbers if the failure survives.

Objective   Phases 2-4 minimise ``distance(s)`` = sum over fault parameters of |p - none| / range
            + sum over physical parameters with a benign bound of |p - benign| / range.
            It is a weighted L1 distance to the "boring" scenario, with tolerance ``ParamSpec.tol`` per
            parameter. The result is NOT claimed to be a unique minimum: failure regions are not monotone,
            bisection finds *a* boundary point along each axis, and coordinates interact.

Every candidate is replayed through the SUT; the result records every change and whether the minimised
scenario reproduces (deterministic replay + failure rate over fresh noise seeds).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .runner import Evaluator
from .scenario import FAULT_GROUPS, FAULT_PARAMS, PARAMS, PHYSICAL_PARAMS, Scenario

ROUND_TO = {'ego_speed': 0.5, 'lead_speed': 0.5, 'initial_gap': 1.0, 'lead_decel': 0.5,
            'brake_onset_s': 0.5, 'perception_latency_s': 0.1, 'distance_noise_std_m': 0.1,
            'dropout_prob': 0.05, 'detection_loss_start_s': 0.5, 'detection_loss_duration_s': 0.5,
            'ttc_error_factor': 0.05, 'ghost_prob': 0.01}


def derived_seeds(base: int, k: int) -> List[int]:
    return [int(base)] + [int((base + 7919 * i + 104729) % (2 ** 31 - 1)) for i in range(1, k)]


def distance(s: Scenario) -> float:
    """Weighted L1 distance to the 'boring' scenario (see module docstring)."""
    d = 0.0
    for k in FAULT_PARAMS:
        sp = PARAMS[k]
        d += abs(getattr(s, k) - sp.simplest) / (sp.high - sp.low)
    for k in PHYSICAL_PARAMS:
        sp = PARAMS[k]
        if sp.benign:
            bound = sp.low if sp.benign == 'low' else sp.high
            d += abs(getattr(s, k) - bound) / (sp.high - sp.low)
    return d


@dataclass
class Change:
    phase: str
    param: str
    old: object
    new: object
    note: str = ''

    def to_dict(self):
        return {'phase': self.phase, 'param': self.param, 'old': self.old, 'new': self.new, 'note': self.note}


@dataclass
class MinimizationResult:
    original: Scenario
    minimized: Scenario
    kind: str
    changes: List[Change] = field(default_factory=list)
    necessary_faults: List[str] = field(default_factory=list)
    removed_faults: List[str] = field(default_factory=list)
    replays: int = 0
    distance_before: float = 0.0
    distance_after: float = 0.0
    reproducible: Dict = field(default_factory=dict)
    seeds: int = 1
    predicate_fallback: bool = False     # multi-seed predicate not satisfied by the original; used its own seed only

    def to_dict(self) -> dict:
        return {'kind': self.kind, 'original': self.original.to_dict(), 'minimized': self.minimized.to_dict(),
                'necessary_faults': self.necessary_faults, 'removed_faults': self.removed_faults,
                'changes': [c.to_dict() for c in self.changes], 'replays': self.replays,
                'distance_before': round(self.distance_before, 4), 'distance_after': round(self.distance_after, 4),
                'reproducibility': self.reproducible, 'predicate_seeds': self.seeds,
                'predicate_fallback_to_single_seed': self.predicate_fallback}


class Minimizer:
    def __init__(self, ev: Evaluator, kind: str, seeds: int = 1):
        self.ev, self.kind, self.seeds = ev, kind, max(1, int(seeds))
        self.replays = 0
        self._memo: Dict[str, bool] = {}
        self.initial_decel_limit: Optional[float] = None     # set by run(): minimised scenarios must not start more imminent

    # -- predicate -------------------------------------------------------------------------------
    def fails(self, s: Scenario) -> bool:
        if s.problems():
            return False
        if self.initial_decel_limit is not None and s.initial_required_decel() > self.initial_decel_limit + 1e-9:
            return False
        fp = s.fingerprint()
        if fp in self._memo:
            return self._memo[fp]
        seeds = derived_seeds(s.seed, self.seeds)
        need = math.ceil(len(seeds) / 2)
        hits, ok = 0, False
        for i, sd in enumerate(seeds):
            self.replays += 1
            hit = self.kind in self.ev.evaluate(s.replace(seed=sd)).failure_kinds
            if i == 0 and not hit:
                break                       # the scenario's own seed must reproduce it
            hits += hit
            if hits >= need:
                ok = True
                break
            if hits + (len(seeds) - i - 1) < need:
                break
        self._memo[fp] = ok
        return ok

    # -- phase 1: ddmin over fault groups -------------------------------------------------------------
    def ddmin_faults(self, s: Scenario) -> List[str]:
        active = s.active_faults()

        def test(keep: Sequence[str]) -> bool:
            return self.fails(s.without_faults([g for g in active if g not in keep]))

        if not test(active):
            raise ValueError('the original scenario does not exhibit the failure')
        if test([]):
            return []
        items = list(active)
        n = 2
        while len(items) >= 2:
            chunk = max(1, math.ceil(len(items) / n))
            subsets = [items[i:i + chunk] for i in range(0, len(items), chunk)]
            reduced = False
            for sub in subsets:
                if len(sub) < len(items) and test(sub):
                    items, n, reduced = sub, 2, True
                    break
            if not reduced:
                for sub in subsets:
                    comp = [x for x in items if x not in sub]
                    if comp and len(comp) < len(items) and test(comp):
                        items, n, reduced = comp, max(n - 1, 2), True
                        break
            if not reduced:
                if n >= len(items):
                    break
                n = min(len(items), 2 * n)
        return items

    # -- bisection towards a target value ------------------------------------------------------------------
    def bisect(self, s: Scenario, param: str, target: float, extra: Optional[Dict[str, float]] = None) -> Scenario:
        """Move ``param`` from its current (failing) value towards ``target`` as far as the failure persists."""
        tol = PARAMS[param].tol
        cur = getattr(s, param)
        if abs(cur - target) <= tol:
            return s
        extra = extra or {}
        # try the target outright (the common case: the parameter is irrelevant)
        cand = s.replace(**{param: target}, **extra)
        if self.fails(cand):
            return cand
        good, bad = cur, target            # good fails, bad does not
        while abs(good - bad) > tol:
            mid = round(0.5 * (good + bad), 6)
            c = s.replace(**{param: mid}, **extra)
            if self.fails(c):
                good = mid
            else:
                bad = mid
        return s.replace(**{param: good}, **extra)

    # -- driver -----------------------------------------------------------------------------------------------
    def run(self, scenario: Scenario, reduce_physical: bool = True) -> MinimizationResult:
        s = scenario.validate()
        self.initial_decel_limit = max(s.initial_required_decel(), 0.0)
        res = MinimizationResult(original=s, minimized=s, kind=self.kind, seeds=self.seeds,
                                 distance_before=distance(s))

        def log(phase, param, old, new, note=''):
            res.changes.append(Change(phase, param, _r(old), _r(new), note))

        # phase 1
        keep = self.ddmin_faults(s)
        removed = [g for g in s.active_faults() if g not in keep]
        for g in removed:
            for k in FAULT_GROUPS[g]:
                if getattr(s, k) != PARAMS[k].simplest:
                    log('fault_removal', k, getattr(s, k), PARAMS[k].simplest, f'{g} not needed')
        s = s.without_faults(removed)
        res.removed_faults, res.necessary_faults = removed, list(keep)

        def reduce_faults(s, phase):
            for g in keep:
                for k in FAULT_GROUPS[g]:
                    if g == 'detection_loss' and k == 'detection_loss_start_s':
                        continue
                    old = getattr(s, k)
                    s = self.bisect(s, k, PARAMS[k].simplest)
                    if getattr(s, k) != old:
                        log(phase, k, old, getattr(s, k), 'severity reduced (bisection)')
            return s

        s = reduce_faults(s, 'fault_severity')

        # phase 3
        if reduce_physical:
            if s.lead_decel > 0:
                c = s.replace(lead_decel=0.0)
                if self.fails(c):
                    log('physical', 'lead_decel', s.lead_decel, 0.0, 'lead braking not needed')
                    s = c
            for k in ('lead_decel', 'ego_speed', 'initial_gap', 'lead_speed'):
                sp = PARAMS[k]
                if not sp.benign or (k == 'lead_decel' and s.lead_decel == 0.0):
                    continue
                target = sp.low if sp.benign == 'low' else sp.high
                old = getattr(s, k)
                s = self.bisect(s, k, target)
                if getattr(s, k) != old:
                    log('physical', k, old, getattr(s, k), f'moved towards benign bound {target}')
            s = reduce_faults(s, 'fault_severity')

        # phase 5: round to readable values
        for k, step in ROUND_TO.items():
            old = getattr(s, k)
            new = round(round(old / step) * step, 6)
            if abs(new - old) < 1e-9 or not (PARAMS[k].low <= new <= PARAMS[k].high):
                continue
            c = s.replace(**{k: new})
            if not c.problems() and self.fails(c):
                log('rounding', k, old, new, 'snapped to a round value')
                s = c

        res.minimized = s.replace(name=(scenario.name or 'scenario') + '-min')
        res.replays = self.replays
        res.distance_after = distance(s)
        return res


def _r(x):
    return round(x, 4) if isinstance(x, float) else x


# --- reproducibility ---------------------------------------------------------------------------------------
def reproducibility(ev: Evaluator, original: Scenario, minimized: Scenario, kind: str, n_seeds: int = 20) -> dict:
    """Deterministic-replay check and failure rate over fresh noise seeds (counts as simulations)."""
    from .adapters import default_sut_factory

    def fresh_trace(sc):
        sut = ev.sut_factory()
        ep = sut.run(sc)
        return [round(r.gt.gap, 9) for r in ep.records], [r.planner.cmd_decel for r in ep.records]

    t1, t2 = fresh_trace(minimized), fresh_trace(minimized)
    seeds = derived_seeds(minimized.seed, n_seeds)
    rate_min = float(np.mean([kind in ev.evaluate(minimized.replace(seed=sd)).failure_kinds for sd in seeds]))
    rate_org = float(np.mean([kind in ev.evaluate(original.replace(seed=sd)).failure_kinds for sd in seeds]))
    deterministic = t1 == t2
    fails_with_seed = kind in ev.evaluate(minimized).failure_kinds
    if not (deterministic and fails_with_seed):
        status = 'NOT reproducible'
    elif rate_min < 0.25:
        status = 'reproducible for its seed only (seed-fragile)'
    else:
        status = 'reproducible'
    return {'status': status, 'deterministic_replay': deterministic, 'fails_with_its_seed': fails_with_seed,
            'seed_failure_rate_minimized': rate_min, 'seed_failure_rate_original': rate_org, 'n_seeds': n_seeds}


def minimize(ev: Evaluator, scenario: Scenario, kind: Optional[str] = None, seeds: int = 1,
             reduce_physical: bool = True, check_reproducibility: bool = True) -> MinimizationResult:
    a = ev.evaluate(scenario)
    if not a.failed:
        raise ValueError('scenario does not fail; nothing to minimise '
                         f'(min gap {a.min_gap:.2f} m, collision class {a.collision_class})')
    kind = kind or a.primary_kind
    if kind not in a.failure_kinds:
        raise ValueError(f'scenario does not exhibit failure kind {kind!r} (has {a.failure_kinds})')
    m = Minimizer(ev, kind, seeds)
    fallback = False
    if seeds > 1 and not m.fails(scenario):
        # the original only fails on a minority of noise seeds: a majority predicate would reject it
        m, fallback = Minimizer(ev, kind, 1), True
    res = m.run(scenario, reduce_physical)
    res.predicate_fallback = fallback
    if check_reproducibility:
        res.reproducible = reproducibility(ev, scenario, res.minimized, kind)
    return res
