"""Evaluation of the diagnostic engine on planted faults and no-fault controls, before and after the attribution rework.

Every case is diagnosed once; the *legacy* labels (Shapley / supported) and the *new* verdict
(``attribution.attribute``) are scored on the same simulations, so the two engines are compared on identical evidence.

Cases
  planted   a scenario the planner survives (<= 1 of 8 noise seeds fails) with exactly one perturbation added at the mildest
            severity of its ladder that makes it fail in >= 75 % of seeds. Expected cause = the planted component.
  null      a scenario with no injected fault that the planner fails (>= 75 % of seeds): the cause is the planner itself,
            so any perception attribution is a false positive.

The label of the planted fault is used only for scoring, never by the diagnosis (it receives the scenario to replay and
nothing else; ``attribution.py`` is checked by a test not to read fault parameters).

``dev`` (seed 11) is the set the original evaluation used, so the legacy numbers can be reproduced. ``heldout`` uses
different seeds and different scenarios and was run once, after the attribution thresholds were frozen.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .adapters import default_sut_factory
from .attribution import PERCEPTION, attribute
from .diagnose import diagnose
from .evaluation import PLANTED
from .metrics import DEFAULT_CRITERIA
from .minimize import derived_seeds
from .runner import Evaluator
from .scenario import ScenarioGenerator

SPLITS = {'dev': {'seed': 11, 'per_type': 12, 'null': 20},
          'heldout': {'seed': 9011, 'per_type': 12, 'null': 30}}


def build_cases(ev: Evaluator, seed: int, per_type: int, n_null: int) -> List[dict]:
    rng = np.random.default_rng(seed)
    cases: List[dict] = []
    for fault, (component, param, ladder) in PLANTED.items():
        gen = ScenarioGenerator(seed + sum(map(ord, fault)), fault_prob=0.0)       # stable (str hash() is salted)
        got, tries = 0, 0
        while got < per_type and tries < 400:
            tries += 1
            base = gen.sample(f'plant-{fault}')
            if sum(ev.evaluate(base.replace(seed=sd)).failed for sd in derived_seeds(base.seed, 8)) > 1:
                continue
            start = float(np.round(rng.uniform(0.0, 3.0), 2))
            planted = None
            for sev in ladder:
                kw = {param: sev}
                if fault == 'detection_loss':
                    kw['detection_loss_start_s'] = start
                s = base.replace(**kw)
                rate = np.mean([ev.evaluate(s.replace(seed=sd)).failed for sd in derived_seeds(s.seed, 8)])
                if rate >= 0.75 and ev.evaluate(s).failed:
                    planted = s
                    break
            if planted is None:
                continue
            got += 1
            cases.append({'type': 'planted', 'fault': fault, 'expected': component, 'severity': planted.params()[param],
                          'scenario': planted, 'kind': ev.evaluate(planted).primary_kind})
        for _ in range(per_type - got):
            cases.append({'type': 'unconstructed', 'fault': fault, 'expected': component})
    gen = ScenarioGenerator(seed + 7, fault_prob=0.0)
    got, tries = 0, 0
    while got < n_null and tries < 800:
        tries += 1
        s = gen.sample('null')
        if ev.evaluate(s).primary_kind != 'collision':
            continue
        if np.mean([ev.evaluate(s.replace(seed=sd)).failed for sd in derived_seeds(s.seed, 8)]) < 0.75:
            continue
        got += 1
        cases.append({'type': 'null', 'fault': None, 'expected': 'planner_logic', 'scenario': s, 'kind': 'collision'})
    return cases


def _score_new(case: dict, attr: dict) -> dict:
    exp = case['expected']
    flagged = set(attr['primary']) | set(attr['contributing'])
    v = attr['verdict']
    out = {'verdict': v, 'primary': attr['primary'], 'contributing': attr['contributing'],
           'outcome_improved_by': attr['outcome_improved_by']}
    if v == 'inconclusive':
        out['outcome'] = 'inconclusive'
    elif v == 'primary' and attr['primary'] == [exp]:
        out['outcome'] = 'correct_unique'
    elif exp in attr['primary']:
        out['outcome'] = 'correct_but_joint'
    else:
        out['outcome'] = 'wrong'
    out['extra_flagged'] = sorted(flagged - {exp})
    out['perception_flagged'] = sorted(c for c in flagged if c in PERCEPTION)
    return out


def _cached_cases(ev: Evaluator, split: str, cache: Optional[Path]) -> List[dict]:
    """Case construction (a few minutes) is deterministic and independent of the attribution code, so it can be cached."""
    from .scenario import Scenario
    cfg = SPLITS[split]
    if cache is not None and Path(cache).exists():
        raw = json.loads(Path(cache).read_text())
        return [dict(c, scenario=Scenario.from_dict(c['scenario'])) if 'scenario' in c else c for c in raw]
    cases = build_cases(ev, cfg['seed'], cfg['per_type'], cfg['null'])
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps([dict(c, scenario=c['scenario'].to_dict()) if 'scenario' in c else c for c in cases], default=float))
    return cases


def evaluate_split(split: str, ev: Optional[Evaluator] = None, verbose: bool = True, cache: Optional[Path] = None) -> dict:
    cfg = SPLITS[split]
    ev = ev or Evaluator(default_sut_factory('kalman'), DEFAULT_CRITERIA)
    t0 = time.perf_counter()
    cases = _cached_cases(ev, split, cache)
    rows = []
    for i, c in enumerate(cases):
        if c['type'] == 'unconstructed':
            continue
        sc = c['scenario']
        d = diagnose(ev, sc, kind=c['kind'], n_seeds=8)
        attr = attribute(ev, sc, d)
        sup = [h['component'] for h in d.hypotheses if h['status'] == 'supported']
        shap = {k: v['value'] for k, v in d.shapley.items()}
        top = max(shap, key=shap.get) if shap else None
        row = {'type': c['type'], 'fault': c['fault'], 'expected': c['expected'], 'severity': c.get('severity'),
               'scenario': sc.to_dict(),
               'legacy': {'top_shapley': top, 'supported': sup, 'top1_correct': top == c['expected'],
                          'expected_supported': c['expected'] in sup},
               'new': _score_new(c, attr), 'attribution': {k: attr[k] for k in ('window', 'inconclusive_reasons', 'components')}}
        rows.append(row)
        if verbose and (len(rows) % 10 == 0):
            print(f'  {split}: {len(rows)} cases diagnosed ({time.perf_counter() - t0:.0f}s)', flush=True)
    return {'split': split, 'config': cfg, 'unconstructed': dict(Counter(c['fault'] for c in cases if c['type'] == 'unconstructed')),
            'rows': rows, 'summary': summarize(rows), 'seconds': round(time.perf_counter() - t0, 1)}


def _mean(xs):
    return round(float(np.mean(xs)), 3) if len(xs) else None


def summarize(rows: List[dict]) -> dict:
    pl = [r for r in rows if r['type'] == 'planted']
    nu = [r for r in rows if r['type'] == 'null']
    out: Dict = {'planted_cases': len(pl), 'null_cases': len(nu)}
    # legacy (Shapley / supported) engine
    n_sup = sum(len(r['legacy']['supported']) for r in pl)
    out['legacy'] = {
        'top1_accuracy': _mean([r['legacy']['top1_correct'] for r in pl]),
        'planted_labelled_supported': _mean([r['legacy']['expected_supported'] for r in pl]),
        'precision_of_supported': round(sum(r['expected'] in r['legacy']['supported'] for r in pl) / n_sup, 3) if n_sup else None,
        'planner_logic_false_flag_planted': _mean(['planner_logic' in r['legacy']['supported'] for r in pl]),
        'null_planner_supported': _mean(['planner_logic' in r['legacy']['supported'] for r in nu]),
        'null_perception_false_support': _mean([any(c != 'planner_logic' for c in r['legacy']['supported']) for r in nu]),
        'by_fault': {f: {'n': len(g), 'top1': _mean([r['legacy']['top1_correct'] for r in g]),
                         'supported': _mean([r['legacy']['expected_supported'] for r in g])}
                     for f in PLANTED for g in [[r for r in pl if r['fault'] == f]]},
    }
    cnt = Counter(r['new']['outcome'] for r in pl)
    out['new'] = {
        'accuracy_unique': _mean([r['new']['outcome'] == 'correct_unique' for r in pl]),
        'accuracy_including_joint': _mean([r['new']['outcome'] in ('correct_unique', 'correct_but_joint') for r in pl]),
        'wrong_rate': _mean([r['new']['outcome'] == 'wrong' for r in pl]),
        'inconclusive_rate': _mean([r['new']['outcome'] == 'inconclusive' for r in pl]),
        'false_positive_rate_any_extra_component': _mean([bool(r['new']['extra_flagged']) for r in pl]),
        'false_positive_rate_perception_extra': _mean([bool(set(r['new']['perception_flagged']) - {r['expected']}) for r in pl]),
        'planner_logic_false_flag_planted': _mean(['planner_logic' in r['new']['extra_flagged'] for r in pl]),
        'outcome_counts': dict(cnt),
        'by_fault': {f: {'n': len(g), 'correct_unique': _mean([r['new']['outcome'] == 'correct_unique' for r in g]),
                         'correct_but_joint': _mean([r['new']['outcome'] == 'correct_but_joint' for r in g]),
                         'wrong': _mean([r['new']['outcome'] == 'wrong' for r in g]),
                         'inconclusive': _mean([r['new']['outcome'] == 'inconclusive' for r in g])}
                     for f in PLANTED for g in [[r for r in pl if r['fault'] == f]]},
        'null': {'planner_logic_primary': _mean([r['new']['primary'] == ['planner_logic'] for r in nu]),
                 'perception_falsely_flagged': _mean([bool(r['new']['perception_flagged']) for r in nu]),
                 'inconclusive': _mean([r['new']['outcome'] == 'inconclusive' for r in nu]),
                 'joint_or_other': _mean([r['new']['verdict'] in ('joint',) for r in nu])},
    }
    return out


def to_markdown(res: dict) -> str:
    s, L = res['summary'], []
    L += [f"# Diagnosis evaluation ({res['split']}, seed {res['config']['seed']})", '',
          f"{s['planted_cases']} planted-fault cases, {s['null_cases']} no-fault controls; could not construct: {res['unconstructed']}. "
          f"All scoring uses the planted label only after the fact.", '']
    lg, nw = s['legacy'], s['new']
    L += ['## Legacy engine (Shapley share, status *supported*)', '',
          f"- top-1 Shapley accuracy {lg['top1_accuracy']}; planted component labelled *supported* in {lg['planted_labelled_supported']}; "
          f"precision of *supported* labels {lg['precision_of_supported']}; planner-logic false flags (planted cases) {lg['planner_logic_false_flag_planted']}",
          f"- no-fault controls: planner logic *supported* in {lg['null_planner_supported']}; a perception component falsely *supported* in {lg['null_perception_false_support']}", '',
          '| planted fault | n | top-1 | planted supported |', '|---|---|---|---|']
    for f, v in lg['by_fault'].items():
        L.append(f"| {f} | {v['n']} | {v['top1']} | {v['supported']} |")
    L += ['', '## New engine (temporal evidence + controlled replacement, explicit joint / inconclusive)', '',
          f"- planted component is the unique primary root cause: **{nw['accuracy_unique']}**; correct but reported jointly with another component: "
          f"{round(nw['accuracy_including_joint'] - nw['accuracy_unique'], 3)}; wrong: {nw['wrong_rate']}; inconclusive: {nw['inconclusive_rate']}",
          f"- false positives: any component other than the planted one flagged primary/contributing in {nw['false_positive_rate_any_extra_component']} of planted cases "
          f"(a perception component in {nw['false_positive_rate_perception_extra']}; planner logic in {nw['planner_logic_false_flag_planted']})",
          f"- no-fault controls: planner logic the unique primary cause in {nw['null']['planner_logic_primary']}; a perception component flagged in "
          f"**{nw['null']['perception_falsely_flagged']}**; inconclusive {nw['null']['inconclusive']}", '',
          '| planted fault | n | correct (unique) | correct, joint | wrong | inconclusive |', '|---|---|---|---|---|---|']
    for f, v in nw['by_fault'].items():
        L.append(f"| {f} | {v['n']} | {v['correct_unique']} | {v['correct_but_joint']} | {v['wrong']} | {v['inconclusive']} |")
    return '\n'.join(L)


def run(split: str, out_dir: Path, verbose: bool = True, cache: Optional[Path] = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = evaluate_split(split, verbose=verbose, cache=cache)
    (out_dir / f'diagnosis_eval_{split}.json').write_text(json.dumps(res, indent=1, default=float))
    (out_dir / f'diagnosis_eval_{split}.md').write_text(to_markdown(res))
    return res


def calibrate_nominal(out: Optional[Path] = None, seed: int = 500, n: int = 300) -> dict:
    """Distribution of every attribution statistic over fault-free scenarios (seed 500, disjoint from dev/held-out).
    The detection thresholds in ``attribution.py`` are 1.5 x the nominal p99 (continuous statistics) or the first value above the
    nominal maximum (lost-track run, lag, missing-return fraction); no failing or planted case is involved."""
    from . import attribution as at
    gen = ScenarioGenerator(seed, fault_prob=0.0)
    sut = default_sut_factory('kalman')()
    stats = at.nominal_statistics([sut.run(gen.sample()) for _ in range(n)])
    res = {'seed': seed, 'n_scenarios': n, 'statistics': stats,
           'thresholds': {'LOST_RUN_STEPS': at.LOST_RUN_STEPS, 'MISS_FRACTION': at.MISS_FRACTION, 'LAG_MIN_S': at.LAG_MIN_S,
                          'DIST_RMS_M': at.DIST_RMS_M, 'RAW_JITTER_M': at.RAW_JITTER_M, 'TTC_LOG_RATIO': round(at.TTC_LOG_RATIO, 3)}}
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, indent=1))
    return res
