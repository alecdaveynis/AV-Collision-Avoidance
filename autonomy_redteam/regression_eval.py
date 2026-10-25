"""False-positive / false-negative evaluation of regression detectors on controlled software mutations.

Two detectors are compared on identical suites and identical mutated planners:

  old   ``regress.run_regression``   single noise seed per scenario, fixed tolerances (baseline format 1)
  new   ``contracts.run_regression_v2``   contracts + paired-seed statistics (baseline format 2)

A *mutation* is a change of one planner constant. Ground truth for "did the change really hurt safety" is **not** the mutation's
intent and **not** the detector's suite: it is measured on a large separate set of random scenarios (the *label set*) with the
mutated and the original planner on the same scenarios and seeds, and an exact one-sided McNemar test on the paired outcomes:

  harmful      p < 0.01 that more scenarios fail (and at least 1 % of the label set)
  improvement  the same in the other direction
  neutral      otherwise

The original benchmark used a cruder label (net change of >= 2 % of 300 scenarios); it is reported too (``legacy_label``) so the
old numbers (8 of 15 non-harmful flagged, 4 of 4 harmful detected) are reproduced exactly.

Splits
  dev      the 18 pre-registered mutations and suite of the original evaluation (``evaluation.MUTANTS``, search seed 123).
           The detector design was done after having seen these results, so dev numbers are *not* held-out evidence.
  heldout  24 new mutations (``HELDOUT_MUTANTS``, fixed in this file before they were run), a different suite (search seed 321) and a
           different label set (seed 4242). Run once, after the detector was frozen.

Also measured: the noise-only false-alarm rate, i.e. the same unmodified planner compared against a baseline recorded on *different*
noise seeds (nothing changed, so every flag is a false positive).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from .adapters import default_sut_factory
from .contracts import (Policy, _role, _runs, aggregate, create_baseline_v2, decide, mcnemar_one_sided,
                        run_regression_v2)
from .evaluation import MUTANTS
from .metrics import DEFAULT_CRITERIA
from .minimize import derived_seeds
from .regress import RegressionCriteria, create_baseline, load_suite, promote, run_regression
from .runner import Evaluator
from .scenario import ScenarioGenerator
from .search import run_search
from .suite import build_suite

HELDOUT_MUTANTS = [   # fixed before any held-out run; intent is only a rough design label, truth comes from the label set
    ('identity', {}, 'none'),
    ('TTC_SLOW=1.2', {'TTC_SLOW': 1.2}, 'harmful'), ('TTC_STOP=0.6', {'TTC_STOP': 0.6}, 'harmful'),
    ('STOP_BASE_DIST=0.8', {'STOP_BASE_DIST': 0.8}, 'harmful'), ('SLOW_HEADWAY=0.4', {'SLOW_HEADWAY': 0.4}, 'harmful'),
    ('ESCALATE_FRAMES=4', {'ESCALATE_FRAMES': 4}, 'harmful'), ('EMERGENCY_DECEL=14', {'EMERGENCY_DECEL': 14.0}, 'harmful'),
    ('SLOW_BASE_DIST=3', {'SLOW_BASE_DIST': 3.0}, 'harmful'), ('DIST_EMA_ALPHA=0.2', {'DIST_EMA_ALPHA': 0.2}, 'harmful'),
    ('MIN_CLOSING_SPEED=2', {'MIN_CLOSING_SPEED': 2.0}, 'harmful'),
    ('DIST_EMA_ALPHA=0.55', {'DIST_EMA_ALPHA': 0.55}, 'benign'), ('RELEASE_FRAMES=7', {'RELEASE_FRAMES': 7}, 'benign'),
    ('TRACK_MAX_AGE=7', {'TRACK_MAX_AGE': 7}, 'benign'), ('TTC_STOP=0.95', {'TTC_STOP': 0.95}, 'benign'),
    ('STOP_BASE_DIST=2.2', {'STOP_BASE_DIST': 2.2}, 'benign'), ('SLOW_BASE_DIST=5.8', {'SLOW_BASE_DIST': 5.8}, 'benign'),
    ('SLOW_HEADWAY=1.05', {'SLOW_HEADWAY': 1.05}, 'benign'), ('MIN_CLOSING_SPEED=0.6', {'MIN_CLOSING_SPEED': 0.6}, 'benign'),
    ('ESCALATE_FRAMES=3', {'ESCALATE_FRAMES': 3}, 'benign'),
    ('TTC_SLOW=3.8', {'TTC_SLOW': 3.8}, 'improvement'), ('STOP_BASE_DIST=3.5', {'STOP_BASE_DIST': 3.5}, 'improvement'),
    ('SLOW_HEADWAY=1.4', {'SLOW_HEADWAY': 1.4}, 'improvement'), ('TTC_STOP=1.4', {'TTC_STOP': 1.4}, 'improvement'),
    ('EMERGENCY_DECEL=6', {'EMERGENCY_DECEL': 6.0}, 'improvement'),
]

SPLITS = {
    'dev': {'mutants': MUTANTS, 'suite_search_seed': 123, 'suite_budget': 300, 'safe_seed': 55,
            'label_seed': 888, 'label_n': 600, 'legacy_seed': 777, 'legacy_n': 300},
    'heldout': {'mutants': HELDOUT_MUTANTS, 'suite_search_seed': 321, 'suite_budget': 300, 'safe_seed': 56,
                'label_seed': 4242, 'label_n': 600, 'legacy_seed': None, 'legacy_n': 0},
}
N_SAFE = 6


def _ev(overrides=None) -> Evaluator:
    return Evaluator(default_sut_factory('kalman', overrides or {}), DEFAULT_CRITERIA)


def _label_set(seed: int, n: int):
    gen = ScenarioGenerator(seed)
    return [gen.sample('label') for _ in range(n)]


def _failed(ev: Evaluator, scs) -> np.ndarray:
    return np.array([ev.evaluate(s).failed for s in scs])


def label_mutation(base_fail: np.ndarray, mut_fail: np.ndarray, alpha: float = 0.01, min_frac: float = 0.01) -> dict:
    worse = int((mut_fail & ~base_fail).sum())
    better = int((base_fail & ~mut_fail).sum())
    n = len(base_fail)
    thr = max(3, int(round(min_frac * n)))
    p_h, p_i = mcnemar_one_sided(worse, better), mcnemar_one_sided(better, worse)
    lab = 'harmful' if (p_h < alpha and worse - better >= thr) else 'improvement' if (p_i < alpha and better - worse >= thr) else 'neutral'
    return {'label': lab, 'worse': worse, 'better': better, 'net_failing': int(mut_fail.sum() - base_fail.sum()),
            'p_harm': round(p_h, 5), 'p_improve': round(p_i, 5)}


def _safe_scenarios(seed: int, n: int, pol: Policy):
    """Scenarios the unmodified planner survives with clear margin in every seed (candidates for contracts)."""
    ev = _ev()
    gen = ScenarioGenerator(seed, fault_prob=0.0)
    out = []
    for _ in range(2000):
        sc = gen.sample('safe')
        if _role(_runs(ev, sc, derived_seeds(sc.seed, pol.n_seeds)), pol) == 'safe_contract':
            out.append(sc)
            if len(out) == n:
                break
    return out


def _summary_flags(rows: List[dict], key: str, label_key: str) -> dict:
    harm = [r for r in rows if r[label_key] == 'harmful']
    non = [r for r in rows if r[label_key] != 'harmful']
    fl = lambda r: r[key]['flagged']  # noqa: E731
    return {'harmful_n': len(harm), 'harmful_detected': sum(fl(r) for r in harm),
            'false_negatives': [r['mutant'] for r in harm if not fl(r)],
            'nonharmful_n': len(non), 'nonharmful_flagged': sum(fl(r) for r in non),
            'false_positives': [r['mutant'] for r in non if fl(r)],
            'neutral_n': sum(r[label_key] == 'neutral' for r in rows),
            'neutral_flagged': sum(fl(r) for r in rows if r[label_key] == 'neutral'),
            'improvement_n': sum(r[label_key] == 'improvement' for r in rows),
            'improvement_flagged': sum(fl(r) for r in rows if r[label_key] == 'improvement'),
            'identity_flagged': any(fl(r) for r in rows if r['mutant'] == 'identity')}


def _trigger(res: dict) -> str:
    if res['passed']:
        return '-'
    t = set()
    for r in res['rows']:
        if r['status'] == 'hard_violation':
            t.add('contract')
        elif r['status'] == 'degraded':
            t.add('statistical')
        elif r['status'] == 'stale_baseline':
            t.add('stale')
    sl = res['suite_level']
    if sl['pairs_worse'] - sl['pairs_better'] >= 3 and sl['p_one_sided'] < 0.005:
        t.add('suite-level')
    if any(d['degraded'] for d in sl.get('pooled', {}).values()):
        t.add('pooled')
    return '+'.join(sorted(t))


def noise_only(suite_dir: Path, pol: Policy, trials: int, seed: int = 99) -> dict:
    """Same planner, baseline recorded on seed set A, current run on seed set B: every flag is a false alarm."""
    ev = _ev()
    items = load_suite(suite_dir)
    rng = np.random.default_rng(seed)
    flags_new, flags_old, tiers = 0, 0, {}
    for _t in range(trials):
        rows_in = []
        for name, sc, _m in items:
            sa = [int(x) for x in rng.integers(1, 2 ** 31 - 1, pol.n_seeds)]
            sb = [int(x) for x in rng.integers(1, 2 ** 31 - 1, pol.n_seeds)]
            ra, rb = _runs(ev, sc, sa), _runs(ev, sc, sb)
            rows_in.append((name, _role(ra, pol), ra, rb))
        rows, suite = aggregate(rows_in, pol)
        tier, passed = decide(rows, suite, pol)
        flags_new += not passed
        tiers[tier] = tiers.get(tier, 0) + 1
        fl = False   # old detector: one seed per scenario (the first of each set)
        for name, _role_, ra, rb in rows_in:
            a0, b0 = ra[0], rb[0]
            fl |= bool(b0['safety_failure'] and not a0['safety_failure'])
            fl |= bool(b0['min_gap'] is not None and a0['min_gap'] is not None and b0['min_gap'] < a0['min_gap'] - 0.5)
            fl |= bool(b0['impact_speed'] > a0['impact_speed'] + 1.0)
            fl |= bool(b0['unnecessary_braking_s'] > a0['unnecessary_braking_s'] + 0.5)
        flags_old += fl
    return {'trials': trials, 'scenarios': len(items), 'seeds_per_scenario': pol.n_seeds, 'new_flagged': flags_new,
            'new_false_alarm_rate': round(flags_new / trials, 3), 'new_tiers': tiers,
            'old_flagged_approx': flags_old, 'old_false_alarm_rate_approx': round(flags_old / trials, 3),
            'note': 'old = the same checks as regress.run_regression restricted to the metrics available per seed (new failure, min gap, '
                    'impact speed, unnecessary braking), using one seed per scenario; an approximation of the exact old code path'}


def evaluate_split(split: str, out: Path, verbose: bool = True, pol: Policy = Policy(), noise_trials: int = 40,
                   work: Path = None) -> dict:
    cfg = SPLITS[split]
    out = Path(out)
    t0 = time.perf_counter()
    base_ev = _ev()
    sres = run_search('random', base_ev, cfg['suite_budget'], cfg['suite_search_seed'])
    work = Path(work) if work else out / 'work'
    suite_dir = work / split / 'suite'
    info = build_suite(base_ev, sres, suite_dir, n_fail=8, n_pass=8)
    ext_dir = work / split / 'suite_plus_safe'
    ext_dir.mkdir(parents=True, exist_ok=True)
    for name, sc, meta in load_suite(suite_dir):
        promote(sc, ext_dir, name, meta)
    for i, sc in enumerate(_safe_scenarios(cfg['safe_seed'], N_SAFE, pol)):
        promote(sc, ext_dir, f'safe_{i + 1:02d}', {'role': 'robust safe scenario (contract)'})
    detectors = {}
    for dname, d in (('suite', suite_dir), ('suite_plus_safe', ext_dir)):
        detectors[dname] = {'dir': d, 'v1': create_baseline(base_ev, d, 'base', 'unmodified'),
                            'v2': create_baseline_v2(base_ev, d, 'base', 'unmodified', pol)}
    label_scs = _label_set(cfg['label_seed'], cfg['label_n'])
    base_label_fail = _failed(base_ev, label_scs)
    legacy = None
    if cfg['legacy_seed'] is not None:
        leg_scs = _label_set(cfg['legacy_seed'], cfg['legacy_n'])
        legacy = {'scs': leg_scs, 'base': int(_failed(base_ev, leg_scs).sum()), 'thr': max(3, int(round(0.02 * cfg['legacy_n'])))}
    rows = []
    for i, (name, ov, intent) in enumerate(cfg['mutants']):
        ev = _ev(ov)
        lab = label_mutation(base_label_fail, _failed(ev, label_scs))
        row = {'mutant': name, 'intent': intent, 'label': lab['label'], 'label_detail': lab}
        if legacy is not None:
            delta = int(_failed(ev, legacy['scs']).sum()) - legacy['base']
            row['legacy_label'] = ('harmful' if delta >= legacy['thr'] else 'improvement' if delta <= -legacy['thr'] else 'neutral')
            row['legacy_delta'] = delta
        for dname, dd in detectors.items():
            old = run_regression(_ev(ov), dd['dir'], dd['v1'], RegressionCriteria())
            new = run_regression_v2(_ev(ov), dd['dir'], dd['v2'], pol)
            row[f'old_{dname}'] = {'flagged': not old['passed'], 'violations': old['violations'], 'counts': old['counts']}
            row[f'new_{dname}'] = {'flagged': not new['passed'], 'tier': new['tier'], 'trigger': _trigger(new), 'counts': new['counts'],
                                   'violations': new['violations'], 'suite_level': new['suite_level'], 'tradeoff': new['tradeoff_note']}
        rows.append(row)
        if verbose:
            print(f'  [{split}] {i + 1}/{len(cfg["mutants"])} {name}: label={lab["label"]} old={row["old_suite"]["flagged"]} '
                  f'new={row["new_suite"]["flagged"]} ({time.perf_counter() - t0:.0f}s)', flush=True)
    summary: Dict = {}
    for dname in detectors:
        for det in ('old', 'new'):
            summary[f'{det}_{dname}'] = _summary_flags(rows, f'{det}_{dname}', 'label')
            if legacy is not None:
                summary[f'{det}_{dname}_legacy_label'] = _summary_flags(rows, f'{det}_{dname}', 'legacy_label')
    res = {'split': split, 'config': {k: v for k, v in cfg.items() if k != 'mutants'}, 'policy': pol.__dict__, 'suite': info,
           'n_safe_added': N_SAFE, 'rows': rows, 'summary': summary,
           'noise_only': {'suite': noise_only(suite_dir, pol, noise_trials), 'suite_plus_safe': noise_only(ext_dir, pol, noise_trials)},
           'seconds': round(time.perf_counter() - t0, 1)}
    out.mkdir(parents=True, exist_ok=True)
    (out / f'regression_eval_{split}.json').write_text(json.dumps(res, indent=1, default=str))
    (out / f'regression_eval_{split}.md').write_text(to_markdown(res))
    return res


def freeze_record(path: Path) -> dict:
    """Hash of the detector and of the held-out definitions, written before the held-out split is run."""
    import hashlib
    here = Path(__file__).resolve().parent
    rec = {'purpose': 'regression detector frozen before the held-out regression evaluation was run',
           'contracts_py_sha256': hashlib.sha256((here / 'contracts.py').read_bytes()).hexdigest()[:16],
           'policy': Policy().__dict__, 'heldout_mutants': [m[0] for m in HELDOUT_MUTANTS], 'heldout_config': {k: v for k, v in SPLITS['heldout'].items() if k != 'mutants'},
           'frozen_unix': int(time.time())}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rec, indent=1))
    return rec


def _fmt(s: dict) -> str:
    return (f'{s["harmful_detected"]}/{s["harmful_n"]} harmful detected; {s["nonharmful_flagged"]}/{s["nonharmful_n"]} non-harmful flagged '
            f'(neutral {s["neutral_flagged"]}/{s["neutral_n"]}, improvements {s["improvement_flagged"]}/{s["improvement_n"]}); '
            f'identity flagged: {s["identity_flagged"]}')


def to_markdown(res: dict) -> str:
    L = [f'# Regression detector evaluation ({res["split"]})', '',
         f'Label set: {res["config"]["label_n"]} random scenarios (seed {res["config"]["label_seed"]}), paired exact McNemar, p<0.01 and >=1 %. '
         f'Suite: {res["suite"]["n_failures"]} failing + {res["suite"]["n_passing"]} near-boundary passing scenarios'
         f' (+{res["n_safe_added"]} robust safe scenarios in the extended suite).', '']
    for dname in ('suite', 'suite_plus_safe'):
        L += [f'## Suite: {dname}', '']
        for det in ('old', 'new'):
            L.append(f'- **{det}** (measured label): {_fmt(res["summary"][f"{det}_{dname}"])}')
            ll = res['summary'].get(f'{det}_{dname}_legacy_label')
            if ll:
                L.append(f'  - legacy label (net >= 2 % of 300): {_fmt(ll)}')
        for det in ('old', 'new'):
            s = res['summary'][f'{det}_{dname}']
            L.append(f'- {det} false negatives: {s["false_negatives"] or "none"}; false positives: {s["false_positives"] or "none"}')
        L.append('')
    L += ['## Noise-only false alarms (unchanged planner, different seeds)', '']
    for k, v in res['noise_only'].items():
        L.append(f'- {k}: new detector flagged {v["new_flagged"]}/{v["trials"]} ({v["new_false_alarm_rate"]}); '
                 f'old checks (one seed) flagged {v["old_flagged_approx"]}/{v["trials"]} ({v["old_false_alarm_rate_approx"]}); tiers {v["new_tiers"]}')
    L += ['', '## Per mutation', '',
          '| mutation | intent | measured label (worse/better/net, p_harm) | old | new (suite) | trigger | new (+safe) | trigger |',
          '|---|---|---|---|---|---|---|---|']
    for r in res['rows']:
        d = r['label_detail']
        L.append(f'| {r["mutant"]} | {r["intent"]} | {r["label"]} ({d["worse"]}/{d["better"]}/{d["net_failing"]:+d}, {d["p_harm"]}) | '
                 f'{"FLAG" if r["old_suite"]["flagged"] else "-"} | {"FLAG" if r["new_suite"]["flagged"] else "-"} | {r["new_suite"]["trigger"]} | '
                 f'{"FLAG" if r["new_suite_plus_safe"]["flagged"] else "-"} | {r["new_suite_plus_safe"]["trigger"]} |')
    return '\n'.join(L) + '\n'
