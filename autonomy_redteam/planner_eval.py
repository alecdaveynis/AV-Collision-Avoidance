"""Before/after evaluation of braking planners under identical conditions.

Protocol (pre-registered here, in code, before the held-out sets were ever run):

* Both planners run the *same* scenario objects (same seeds) in the *same* simulator with the *same* metric and oracle
  code. Nothing in the physics, the sensor model, the scenario generator or the failure criteria differs between rows;
  only ``SafetySimSUT(planner=...)`` does. Paired comparisons therefore isolate the planner.
* Scenario sets are split into **dev** (allowed to inform design decisions) and **heldout** (generated from disjoint
  seeds / a parameter region the dev sets never contain; run once after the planner configuration was frozen).
  ``planner_fingerprint()`` hashes the improved planner's source and constants; the held-out report records it, so a
  later change to the planner is visible as a mismatch with ``docs/redteam/heldout_freeze.json``.
* Collision classes (unavoidable / hard / avoidable) come from kinematic oracles that depend on the scenario only, so
  the unavoidable-collision rate is a property of the scenario set, not of the planner.
* Extra reference rows: ``baseline+EMERGENCY_DECEL=4`` (the simplest threshold-only change that also removes the known
  failure, to show what the improved policy adds beyond a retuned constant) and ``reference_rss`` (the RSS rule on
  ground-truth-free perception; a comparison point, not a claim of correctness).

Metrics per set and planner: failure rate, avoidable / unavoidable collision rate, minimum separation, collision
relative speed, unnecessary-braking rate, braking reaction time, peak commanded deceleration, state transitions,
safety-margin violations (min gap < 1 m). Rates carry Wilson 95 % intervals; paired differences carry an exact McNemar
p-value and a paired bootstrap interval.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .adapters import default_sut_factory
from .metrics import DEFAULT_CRITERIA, assess
from .oracle import run_oracle
from .scenario import Scenario, ScenarioGenerator
from .sut import Interventions

MARGIN_VIOLATION_GAP = 1.0        # m: half the planner's 2 m standstill margin
NOISE_SEEDS = 20
REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------------------------------------------------
# planners under comparison
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PlannerSpec:
    name: str
    planner: str = 'baseline'
    overrides: Tuple[Tuple[str, float], ...] = ()
    policy: str = 'planner'
    note: str = ''

    def factory(self, ttc_mode: str = 'kalman'):
        return default_sut_factory(ttc_mode, dict(self.overrides), self.planner)

    def interventions(self) -> Interventions:
        return Interventions(policy=self.policy)


PLANNERS: Dict[str, PlannerSpec] = {
    'baseline': PlannerSpec('baseline', 'baseline', note='original BehaviorManager'),
    'improved': PlannerSpec('improved', 'improved', note='SafeBehaviorManager (src/safe_braking.py)'),
    'baseline+EMERGENCY_DECEL=4': PlannerSpec('baseline+EMERGENCY_DECEL=4', 'baseline', (('EMERGENCY_DECEL', 4.0),),
                                              note='original planner, stop zone sized for 4 m/s^2 (threshold-only control)'),
    'reference_rss': PlannerSpec('reference_rss', 'baseline', policy='reference', note='RSS longitudinal rule (comparison point)'),
}
DEFAULT_COMPARE = ('baseline', 'improved')


def planner_fingerprint() -> Dict[str, object]:
    """Hash of everything that defines the improved planner (source + constants)."""
    import sys
    src = str(REPO / 'src')
    if src not in sys.path:
        sys.path.insert(0, src)
    import config
    consts = {k: getattr(config, k) for k in sorted(dir(config)) if k.startswith('SAFE_')}
    consts.update({k: getattr(config, k) for k in ('STOP_BASE_DIST', 'SLOW_BASE_DIST', 'SLOW_HEADWAY', 'TTC_SLOW', 'TTC_STOP',
                                                   'EMERGENCY_DECEL', 'ESCALATE_FRAMES', 'RELEASE_FRAMES', 'DIST_EMA_ALPHA',
                                                   'MIN_CLOSING_SPEED')})
    h = hashlib.sha256()
    h.update((REPO / 'src' / 'safe_braking.py').read_bytes())
    h.update((REPO / 'src' / 'behavior_manager.py').read_bytes())
    h.update(json.dumps(consts, sort_keys=True).encode())
    return {'sha256': h.hexdigest()[:16], 'constants': consts}


# --------------------------------------------------------------------------------------------------------------------
# scenario sets
# --------------------------------------------------------------------------------------------------------------------
def _noise_seeds(sc: Scenario, k: int, offset: int = 0) -> List[Scenario]:
    return [sc.replace(seed=int((sc.seed + 7919 * (i + offset) + 104729) % (2 ** 31 - 1)), name=f'{sc.name}#n{i + offset}')
            for i in range(k)]


def _draw(gen: ScenarioGenerator, n: int, prefix: str) -> List[Scenario]:
    return [gen.sample().replace(name=f'{prefix}-{i:03d}') for i in range(n)]


def _reference_min_gap(sc: Scenario) -> Tuple[bool, float]:
    from .adapters import SafetySimSUT
    lim = SafetySimSUT().limits()
    r = run_oracle(sc, 'reference', lim, DEFAULT_CRITERIA.reference_delay_s)
    return r.collided, r.min_gap


def _near_boundary(seed: int, n: int, prefix: str, bounds=None, lo=0.5, hi=3.0, max_tries=20000) -> List[Scenario]:
    """Fault-free scenarios an independent reference controller (RSS, perfect information) only just handles.

    Selection uses neither planner under test: candidates are kept when the reference policy avoids the collision and
    its minimum gap lies in [lo, hi] m. These are tight but demonstrably avoidable."""
    gen = ScenarioGenerator(seed, bounds=bounds, fault_prob=0.0)
    out: List[Scenario] = []
    for _ in range(max_tries):
        sc = gen.sample()
        collided, mg = _reference_min_gap(sc)
        if not collided and lo <= mg <= hi:
            out.append(sc.replace(name=f'{prefix}-{len(out):03d}'))
            if len(out) >= n:
                break
    return out


def _fault_sets(seed: int, n: int, prefix: str, bounds=None) -> List[Scenario]:
    """Sensor dropout and perception latency forced on; every other fault off."""
    gen = ScenarioGenerator(seed, bounds=bounds, fault_prob=0.0)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xFA17]))
    out = []
    for i in range(n):
        sc = gen.sample()
        out.append(sc.replace(dropout_prob=round(float(rng.uniform(0.2, 0.5)), 3),
                              perception_latency_s=round(float(rng.uniform(0.0, 0.4)), 3), name=f'{prefix}-{i:03d}'))
    return out


@dataclass(frozen=True)
class SetSpec:
    name: str
    role: str                    # dev | heldout
    description: str
    build: Callable[[], List[Scenario]]


def _known(which: str, name: str) -> List[Scenario]:
    sc = Scenario.load(REPO / 'docs' / 'redteam' / 'known_failure' / f'{which}_scenario.json')
    return _noise_seeds(sc.replace(name=name), NOISE_SEEDS)


def _regression_suite() -> List[Scenario]:
    out = []
    for f in sorted((REPO / 'scenarios' / 'regression').glob('*.json')):
        sc = Scenario.load(f)
        out += _noise_seeds(sc.replace(name=f.stem), NOISE_SEEDS)
    return out


SETS: Dict[str, SetSpec] = {s.name: s for s in [
    SetSpec('known_failure_minimized', 'dev', 'the discovered failure after minimization (no faults) x 20 noise seeds',
            lambda: _known('minimized', 'known-min')),
    SetSpec('known_failure_original', 'dev', 'the discovered failure as found by the search (latency 0.3 s, distance noise, 1.4 s detection loss) x 20 noise seeds',
            lambda: _known('original', 'known-orig')),
    SetSpec('regression_suite', 'dev', 'committed scenarios/regression/*.json x 20 noise seeds', _regression_suite),
    SetSpec('dev_random', 'dev', '300 random scenarios, faults on with p=0.5, ego_speed 6-16 m/s, generator seed 11',
            lambda: _draw(ScenarioGenerator(11, bounds={'ego_speed': (6.0, 16.0)}), 300, 'dev-random')),
    SetSpec('dev_near_boundary', 'dev', '100 fault-free scenarios the RSS reference only just handles (min gap 0.5-3 m), ego 6-16 m/s, seed 12',
            lambda: _near_boundary(12, 100, 'dev-near', {'ego_speed': (6.0, 16.0)})),
    SetSpec('dev_faults', 'dev', '150 scenarios with dropout 0.2-0.5 and latency 0-0.4 s, ego 6-16 m/s, seed 13',
            lambda: _fault_sets(13, 150, 'dev-faults', {'ego_speed': (6.0, 16.0)})),
    SetSpec('heldout_random', 'heldout', '500 random scenarios, full parameter box, generator seed 90001 (unseen seeds)',
            lambda: _draw(ScenarioGenerator(90001), 500, 'ho-random')),
    SetSpec('heldout_speed_range', 'heldout', '300 random scenarios with ego_speed 16-20 m/s, a range no dev set contains, seed 90002',
            lambda: _draw(ScenarioGenerator(90002, bounds={'ego_speed': (16.0, 20.0)}), 300, 'ho-fast')),
    SetSpec('heldout_near_boundary', 'heldout', '150 fault-free scenarios the RSS reference only just handles, full box, seed 90003',
            lambda: _near_boundary(90003, 150, 'ho-near')),
    SetSpec('heldout_faults', 'heldout', '200 scenarios with dropout 0.2-0.5 and latency 0-0.4 s, full box, seed 90004',
            lambda: _fault_sets(90004, 200, 'ho-faults')),
]}


# --------------------------------------------------------------------------------------------------------------------
# running and metrics
# --------------------------------------------------------------------------------------------------------------------
def run_set(spec: PlannerSpec, scenarios: List[Scenario], set_name: str, ttc_mode: str = 'kalman') -> List[dict]:
    sut = spec.factory(ttc_mode)()
    iv = spec.interventions()
    rows = []
    for sc in scenarios:
        ep = sut.run(sc, iv)
        a = assess(ep, DEFAULT_CRITERIA)
        states = [r.planner.state for r in ep.records]
        trans = sum(1 for i in range(1, len(states)) if states[i] != states[i - 1])
        dur = len(ep.records) * ep.limits.dt
        rows.append({
            'set': set_name, 'scenario': sc.name, 'planner': spec.name,
            'collision': bool(a.collision), 'collision_class': a.collision_class, 'impact_speed': float(a.impact_speed),
            'min_gap': float(a.min_gap), 'reaction_time': a.brake_reaction_time, 'hazard': a.hazard_count > 0,
            'unnecessary': 'unnecessary_braking' in a.failure_kinds, 'unnecessary_s': float(a.unnecessary_braking_s),
            'peak_cmd': float(a.peak_cmd_decel), 'transitions': int(trans), 'duration_s': float(dur),
            'margin_violation': bool(a.min_gap < MARGIN_VIOLATION_GAP or a.collision),
            'failed': bool(a.failed), 'kinds': list(a.failure_kinds),
        })
    return rows


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _rate(rows, pred) -> dict:
    k = sum(1 for r in rows if pred(r))
    lo, hi = wilson(k, len(rows))
    return {'k': k, 'n': len(rows), 'rate': k / len(rows) if rows else float('nan'), 'ci95': [round(lo, 4), round(hi, 4)]}


def _q(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float('nan')


def summarize(rows: List[dict]) -> dict:
    coll = [r for r in rows if r['collision']]
    haz = [r['reaction_time'] for r in rows if r['reaction_time'] is not None]
    mg = [r['min_gap'] for r in rows]
    total_s = sum(r['duration_s'] for r in rows)
    return {
        'n': len(rows),
        'failure': _rate(rows, lambda r: r['failed']),
        'collision': _rate(rows, lambda r: r['collision']),
        'avoidable_collision': _rate(rows, lambda r: r['collision'] and r['collision_class'] == 'avoidable'),
        'hard_collision': _rate(rows, lambda r: r['collision'] and r['collision_class'] == 'hard'),
        'unavoidable_collision': _rate(rows, lambda r: r['collision'] and r['collision_class'] == 'unavoidable'),
        'unnecessary_braking': _rate(rows, lambda r: r['unnecessary']),
        'margin_violation': _rate(rows, lambda r: r['margin_violation']),
        'min_gap_m': {'mean': float(np.mean(mg)) if mg else float('nan'), 'median': _q(mg, 50), 'p5': _q(mg, 5)},
        'impact_speed_given_collision_mps': {'mean': float(np.mean([r['impact_speed'] for r in coll])) if coll else 0.0,
                                             'max': float(max([r['impact_speed'] for r in coll], default=0.0)), 'n': len(coll)},
        'mean_impact_speed_all_mps': float(np.mean([r['impact_speed'] for r in rows])) if rows else float('nan'),
        'reaction_time_s': {'mean': float(np.mean(haz)) if haz else float('nan'), 'p95': _q(haz, 95), 'n': len(haz)},
        'peak_cmd_decel': {'mean': float(np.mean([r['peak_cmd'] for r in rows])) if rows else float('nan'),
                           'p95': _q([r['peak_cmd'] for r in rows], 95)},
        'transitions_per_episode': float(np.mean([r['transitions'] for r in rows])) if rows else float('nan'),
        'transitions_per_min': 60.0 * sum(r['transitions'] for r in rows) / total_s if total_s else float('nan'),
    }


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts (b: only A, c: only B)."""
    n = b + c
    if n == 0:
        return 1.0
    from scipy.stats import binomtest
    return float(binomtest(min(b, c), n, 0.5).pvalue)


def paired_binary(rows_a: List[dict], rows_b: List[dict], key: Callable[[dict], bool], n_boot: int = 2000, seed: int = 0) -> dict:
    """A vs B on the same scenarios: discordant counts, exact McNemar p, bootstrap CI of rate(B) - rate(A)."""
    assert [r['scenario'] for r in rows_a] == [r['scenario'] for r in rows_b]
    a = np.array([key(r) for r in rows_a], dtype=float)
    b = np.array([key(r) for r in rows_b], dtype=float)
    n = len(a)
    only_a, only_b = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = (b[idx].mean(axis=1) - a[idx].mean(axis=1))
    return {'n': n, 'rate_a': float(a.mean()), 'rate_b': float(b.mean()), 'diff_b_minus_a': float(b.mean() - a.mean()),
            'ci95': [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)],
            'only_a': only_a, 'only_b': only_b, 'mcnemar_p': mcnemar_exact(only_a, only_b)}


COMPARED = {
    'avoidable collision': lambda r: r['collision'] and r['collision_class'] == 'avoidable',
    'any collision': lambda r: r['collision'],
    'unnecessary braking': lambda r: r['unnecessary'],
    'margin violation (<1 m)': lambda r: r['margin_violation'],
    'any failure': lambda r: r['failed'],
}


def run_sensitivity(out_dir: Path, set_name: str = 'heldout_random', lags=(0.3,), planners=DEFAULT_COMPARE, verbose=True) -> dict:
    """Post-hoc robustness check, *not* part of the pre-registered comparison: the simulated brake actuator is slower than
    both planners assume (the improved planner assumes ``SAFE_ACTUATOR_LAG`` = 0.25 s). The physics model is changed
    explicitly and only here; the same scenarios and metrics are used."""
    import sys
    sys.path.insert(0, str(REPO / 'src'))
    import safety_sim as ss
    scs = SETS[set_name].build()
    res = {'set': set_name, 'n': len(scs), 'planners': list(planners), 'lags': {}}
    old = ss.LAG
    try:
        for lag in lags:
            ss.LAG = lag
            per = {p: summarize(run_set(PLANNERS[p], scs, set_name)) for p in planners}
            res['lags'][str(lag)] = per
            if verbose:
                print(f'[{set_name} @ actuator lag {lag}s] ' + ' | '.join(
                    f"{p}: avoidable coll {per[p]['avoidable_collision']['k']}/{per[p]['n']}, unnecessary {per[p]['unnecessary_braking']['k']}"
                    for p in planners), flush=True)
    finally:
        ss.LAG = old
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'planner_sensitivity.json').write_text(json.dumps(res, indent=1, default=float))
    return res


# --------------------------------------------------------------------------------------------------------------------
# orchestration + report
# --------------------------------------------------------------------------------------------------------------------
def run_comparison(out_dir: Path, split: str = 'dev', planners=DEFAULT_COMPARE, sets: Optional[List[str]] = None,
                   ttc_mode: str = 'kalman', verbose: bool = True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = sets or [s.name for s in SETS.values() if split in ('all', s.role)]
    fp = planner_fingerprint()
    res: dict = {'protocol_version': 1, 'split': split, 'planners': {p: PLANNERS[p].__dict__ | {'overrides': dict(PLANNERS[p].overrides)} for p in planners},
                 'improved_planner_fingerprint': fp, 'sets': {}, 'created_unix': int(time.time()),
                 'margin_violation_gap_m': MARGIN_VIOLATION_GAP}
    t0 = time.perf_counter()
    for sname in names:
        spec = SETS[sname]
        scs = spec.build()
        per = {}
        all_rows = {}
        for p in planners:
            rows = run_set(PLANNERS[p], scs, sname, ttc_mode)
            all_rows[p] = rows
            per[p] = summarize(rows)
        pair = {}
        if len(planners) >= 2:
            a, b = planners[0], planners[1]
            pair = {lbl: paired_binary(all_rows[a], all_rows[b], fn) for lbl, fn in COMPARED.items()}
        res['sets'][sname] = {'role': spec.role, 'description': spec.description, 'n': len(scs), 'summary': per,
                              'paired': pair, 'paired_planners': list(planners[:2])}
        (out_dir / f'rows_{sname}.json').write_text(json.dumps(all_rows))
        if verbose:
            print(f'[{sname}] n={len(scs)} ' + ' | '.join(
                f"{p}: coll {per[p]['collision']['k']} (avoid {per[p]['avoidable_collision']['k']}) unnec {per[p]['unnecessary_braking']['k']}"
                for p in planners), flush=True)
    res['seconds'] = round(time.perf_counter() - t0, 1)
    (out_dir / f'planner_comparison_{split}.json').write_text(json.dumps(res, indent=1, default=float))
    (out_dir / f'planner_comparison_{split}.md').write_text(to_markdown(res))
    return res


def _pct(x: dict) -> str:
    return f"{100 * x['rate']:.1f}% ({x['k']}/{x['n']}) [{100 * x['ci95'][0]:.1f}, {100 * x['ci95'][1]:.1f}]"


def to_markdown(res: dict) -> str:
    planners = list(res['planners'])
    L = [f"# Planner comparison ({res['split']} sets)", '',
         f"Improved-planner fingerprint `{res['improved_planner_fingerprint']['sha256']}`; failure criteria and simulator identical for every row. "
         f"Rates show `% (k/n) [Wilson 95% CI]`. Scenario-with-noise-seed rows are not independent samples: the n for "
         f"`known_failure_*` and `regression_suite` counts (scenario, seed) pairs.", '']
    for sname, s in res['sets'].items():
        L += [f"## {sname} ({s['role']}, n={s['n']})", f"{s['description']}", '',
              '| metric | ' + ' | '.join(planners) + ' |', '|---|' + '---|' * len(planners)]

        def row(label, fn):
            L.append(f'| {label} | ' + ' | '.join(fn(s['summary'][p]) for p in planners) + ' |')
        row('avoidable collisions', lambda m: _pct(m['avoidable_collision']))
        row('hard collisions (reference also fails)', lambda m: _pct(m['hard_collision']))
        row('unavoidable collisions', lambda m: _pct(m['unavoidable_collision']))
        row('unnecessary braking', lambda m: _pct(m['unnecessary_braking']))
        row('margin violation (<1 m)', lambda m: _pct(m['margin_violation']))
        row('any failure', lambda m: _pct(m['failure']))
        row('min gap median / p5 (m)', lambda m: f"{m['min_gap_m']['median']:.2f} / {m['min_gap_m']['p5']:.2f}")
        row('impact speed given collision, mean / max (m/s)',
            lambda m: f"{m['impact_speed_given_collision_mps']['mean']:.2f} / {m['impact_speed_given_collision_mps']['max']:.2f} (n={m['impact_speed_given_collision_mps']['n']})")
        row('brake reaction time mean / p95 (s)',
            lambda m: f"{m['reaction_time_s']['mean']:.2f} / {m['reaction_time_s']['p95']:.2f} (n={m['reaction_time_s']['n']})")
        row('peak commanded decel mean / p95 (m/s^2)', lambda m: f"{m['peak_cmd_decel']['mean']:.2f} / {m['peak_cmd_decel']['p95']:.2f}")
        row('state transitions per minute', lambda m: f"{m['transitions_per_min']:.1f}")
        if s['paired']:
            a, b = s['paired_planners']
            L += ['', f'Paired difference {b} minus {a} (same scenarios):', '',
                  '| outcome | rate ' + a + ' | rate ' + b + ' | difference [bootstrap 95% CI] | only ' + a + ' / only ' + b + ' | McNemar p |',
                  '|---|---|---|---|---|---|']
            for lbl, d in s['paired'].items():
                L.append(f"| {lbl} | {100 * d['rate_a']:.1f}% | {100 * d['rate_b']:.1f}% | {100 * d['diff_b_minus_a']:+.1f} pp "
                         f"[{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}] | {d['only_a']} / {d['only_b']} | {d['mcnemar_p']:.3g} |")
        L.append('')
    return '\n'.join(L)
