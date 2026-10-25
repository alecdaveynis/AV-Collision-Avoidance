"""Evaluation of the framework itself (``python -m autonomy_redteam evaluate``).

A. Search: random vs CMA-ES at equal simulation budgets, repeated over trials (same seeds for both methods), in two
   regimes: the full parameter box and a "rare" box where the nominal planner passes more often. Metrics: failing
   scenarios, simulations to first failure, *unique reproducible findings* (signature = kind + necessary fault set +
   lead regime, ddmin-reduced; reproducible = fails in >= 3 of 5 derived noise seeds), worst impact speed.
B. Minimisation: on distinct findings, does the minimised scenario still fail, with how many perturbations, replays, and
   how seed-fragile is it?
C. Diagnosis: planted-fault recovery. A single known perturbation is injected into a scenario the planner survives without
   it; the diagnosis should name the matching component. Also scenarios that fail with *no* injected fault (cause: the
   planner itself).
D. Regression detection: a suite from a search, a baseline of the unmodified planner, then planner "mutants" (threshold
   changes, standing in for new software versions). Ground truth for "harmful" comes from an independent paired holdout of
   random scenarios, not from the mutation's intent.
E. Cost: simulations, wall-clock time.
F. Agent-guided investigation: only reported if an API key is configured; it is not run otherwise.

Nothing in this file is hard-coded as a result; the README quotes ``evaluation_results.json`` from an actual run.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .adapters import default_sut_factory
from .diagnose import diagnose
from .findings import group_failures, lead_regime, sig_str
from .metrics import DEFAULT_CRITERIA
from .minimize import derived_seeds, minimize
from .regress import RegressionCriteria, create_baseline, run_regression
from .runner import Evaluator
from .scenario import Scenario, ScenarioGenerator
from .search import run_search
from .suite import build_suite

ACTIVE_PLANNER = 'baseline'          # set by run_evaluation(planner=...); 'baseline' reproduces the committed results


def _sut(overrides=None):
    return default_sut_factory('kalman', overrides, ACTIVE_PLANNER)


RARE_BOUNDS = {'ego_speed': (6.0, 16.0), 'initial_gap': (35.0, 80.0), 'lead_decel': (0.0, 4.0),
               'lead_speed': (4.0, 20.0)}

MUTANTS = [   # pre-registered before the evaluation was run; labels come from the holdout, not from 'intent'
    ('identity', {}, 'none'),
    ('TTC_SLOW=1.5', {'TTC_SLOW': 1.5}, 'harmful'), ('TTC_STOP=0.5', {'TTC_STOP': 0.5}, 'harmful'),
    ('STOP_BASE_DIST=0.5', {'STOP_BASE_DIST': 0.5}, 'harmful'), ('SLOW_HEADWAY=0.3', {'SLOW_HEADWAY': 0.3}, 'harmful'),
    ('ESCALATE_FRAMES=5', {'ESCALATE_FRAMES': 5}, 'harmful'), ('EMERGENCY_DECEL=20', {'EMERGENCY_DECEL': 20.0}, 'harmful'),
    ('SLOW_BASE_DIST=2', {'SLOW_BASE_DIST': 2.0}, 'harmful'), ('MIN_CLOSING_SPEED=3', {'MIN_CLOSING_SPEED': 3.0}, 'harmful'),
    ('DIST_EMA_ALPHA=0.45', {'DIST_EMA_ALPHA': 0.45}, 'benign'), ('RELEASE_FRAMES=6', {'RELEASE_FRAMES': 6}, 'benign'),
    ('TRACK_MAX_AGE=6', {'TRACK_MAX_AGE': 6}, 'benign'), ('TTC_STOP=1.05', {'TTC_STOP': 1.05}, 'benign'),
    ('STOP_BASE_DIST=2.1', {'STOP_BASE_DIST': 2.1}, 'benign'), ('SLOW_BASE_DIST=6.2', {'SLOW_BASE_DIST': 6.2}, 'benign'),
    ('TTC_SLOW=4.0', {'TTC_SLOW': 4.0}, 'improvement'), ('STOP_BASE_DIST=4', {'STOP_BASE_DIST': 4.0}, 'improvement'),
    ('SLOW_HEADWAY=1.5', {'SLOW_HEADWAY': 1.5}, 'improvement'), ('TTC_STOP=1.5', {'TTC_STOP': 1.5}, 'improvement'),
]
PLANTED = {   # fault group -> component a correct diagnosis names, severity ladder (mildest first)
    'perception_latency': ('reaction_latency', 'perception_latency_s', [0.2, 0.3, 0.4, 0.5, 0.6]),
    'detection_loss': ('object_detection', 'detection_loss_duration_s', [0.3, 0.6, 1.0, 1.5, 2.5]),
    'sensor_dropout': ('object_detection', 'dropout_prob', [0.3, 0.5, 0.7]),
    'distance_noise': ('distance_estimation', 'distance_noise_std_m', [0.5, 1.0, 1.5, 2.0]),
    'ttc_error': ('ttc_estimation', 'ttc_error_factor', [1.5, 2.0, 2.5, 3.0]),
}


def _stats(xs) -> dict:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return {'n': 0, 'mean': None, 'std': None, 'median': None}
    return {'n': len(xs), 'mean': round(float(np.mean(xs)), 3), 'std': round(float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0, 3),
            'median': round(float(np.median(xs)), 3)}


def _paired(a: List[float], b: List[float]) -> dict:
    d = np.array(a, float) - np.array(b, float)
    rng = np.random.default_rng(0)
    boots = [float(np.mean(rng.choice(d, len(d)))) for _ in range(2000)] if len(d) else [0.0]
    return {'mean_diff': round(float(np.mean(d)), 3) if len(d) else None,
            'ci95': [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
            'wins_first': int((d > 0).sum()), 'ties': int((d == 0).sum()), 'wins_second': int((d < 0).sum())}


def _is_reproducible(ev: Evaluator, s: Scenario, kind: str, k: int = 5) -> bool:
    hits = sum(kind in ev.evaluate(s.replace(seed=sd)).failure_kinds for sd in derived_seeds(s.seed, k))
    return hits >= 3


# --- A. search comparison -------------------------------------------------------------------------------------------------------
def eval_search(budget: int, trials: int) -> dict:
    out: Dict = {'budget': budget, 'trials': trials, 'regimes': {}}
    for regime, bounds in (('full_box', None), ('rare_box', RARE_BOUNDS)):
        per: Dict[str, List[dict]] = {'random': [], 'cmaes': []}
        reps: Dict[str, Dict[tuple, Scenario]] = {'random': {}, 'cmaes': {}}
        for method in per:
            for seed in range(trials):
                ev = Evaluator(_sut(), DEFAULT_CRITERIA)
                t0 = time.perf_counter()
                res = run_search(method, ev, budget, seed, bounds)
                search_s = time.perf_counter() - t0
                n_search = ev.n_sims
                findings = group_failures(ev, res.evaluations)
                repro = [f for f in findings if _is_reproducible(ev, f.representative, f.kind)]
                for f in repro:
                    reps[method].setdefault(f.signature, f.representative)
                per[method].append({
                    'seed': seed, 'sims': n_search, 'failing': len(res.failures), 'first_failure_sim': res.first_failure_sim,
                    'unique_findings': len(findings), 'unique_reproducible': len(repro),
                    'worst_impact_speed': round(max([e.assessment.impact_speed for e in res.failures] or [0.0]), 2),
                    'search_seconds': round(search_s, 3), 'post_processing_sims': ev.n_sims - n_search,
                    'signatures': sorted(sig_str(f.signature) for f in repro)})
        summ = {}
        for m, rows in per.items():
            summ[m] = {'failing': _stats([r['failing'] for r in rows]),
                       'first_failure_sim': _stats([r['first_failure_sim'] for r in rows]),
                       'trials_without_failure': sum(r['first_failure_sim'] is None for r in rows),
                       'unique_reproducible': _stats([r['unique_reproducible'] for r in rows]),
                       'worst_impact_speed': _stats([r['worst_impact_speed'] for r in rows]),
                       'search_seconds': _stats([r['search_seconds'] for r in rows]),
                       'distinct_signatures_over_all_trials': len(reps[m])}
        out['regimes'][regime] = {
            'bounds': bounds, 'summary': summ,
            'paired_cmaes_minus_random_unique_reproducible': _paired([r['unique_reproducible'] for r in per['cmaes']],
                                                                      [r['unique_reproducible'] for r in per['random']]),
            'paired_cmaes_minus_random_failing': _paired([r['failing'] for r in per['cmaes']], [r['failing'] for r in per['random']]),
            'union_signatures': {m: len(reps[m]) for m in reps},
            'union_both': len(set(reps['random']) | set(reps['cmaes'])), 'trial_rows': per,
            'collected_scenarios': [s.to_dict() for m in reps for s in reps[m].values()]}
    return out


# --- B. minimisation ----------------------------------------------------------------------------------------------------------
def eval_minimization(scenarios: List[dict], limit: int = 40) -> dict:
    ev = Evaluator(_sut(), DEFAULT_CRITERIA)
    seen, rows = set(), []
    for d in scenarios:
        s = Scenario.from_dict(d)
        if s.fingerprint() in seen:
            continue
        seen.add(s.fingerprint())
        if len(rows) >= limit:
            break
        a = ev.evaluate(s)
        if not a.failed:
            continue
        t0 = time.perf_counter()
        try:
            m = minimize(ev, s, kind=a.primary_kind, seeds=3, check_reproducibility=True)
        except ValueError:
            continue
        # independent verification with a fresh SUT instance
        ver = Evaluator(_sut(), DEFAULT_CRITERIA).evaluate(m.minimized)
        r = m.reproducible
        rows.append({'kind': m.kind, 'faults_before': len(s.active_faults()), 'faults_after': len(m.necessary_faults),
                     'distance_before': round(m.distance_before, 3), 'distance_after': round(m.distance_after, 3),
                     'replays': m.replays, 'still_fails_fresh_sut': m.kind in ver.failure_kinds,
                     'deterministic': r['deterministic_replay'], 'seed_rate_original': r['seed_failure_rate_original'],
                     'seed_rate_minimized': r['seed_failure_rate_minimized'], 'status': r['status'],
                     'predicate_fallback': m.predicate_fallback, 'seconds': round(time.perf_counter() - t0, 3)})
    n = len(rows)
    if not n:
        return {'n': 0}
    return {'n': n, 'still_fails_fresh_sut': sum(r['still_fails_fresh_sut'] for r in rows) / n,
            'deterministic_replay': sum(r['deterministic'] for r in rows) / n,
            'faults_before': _stats([r['faults_before'] for r in rows]), 'faults_after': _stats([r['faults_after'] for r in rows]),
            'distance_before': _stats([r['distance_before'] for r in rows]), 'distance_after': _stats([r['distance_after'] for r in rows]),
            'replays': _stats([r['replays'] for r in rows]), 'seconds': _stats([r['seconds'] for r in rows]),
            'seed_rate_original': _stats([r['seed_rate_original'] for r in rows]),
            'seed_rate_minimized': _stats([r['seed_rate_minimized'] for r in rows]),
            'seed_fragile_after': sum(r['status'].startswith('reproducible for its seed only') for r in rows) / n,
            'seed_fragile_before': sum(r['seed_rate_original'] < 0.25 for r in rows) / n,
            'status_counts': dict(Counter(r['status'] for r in rows)),
            'kinds': dict(Counter(r['kind'] for r in rows)), 'rows': rows}


# --- C. diagnosis: planted faults ---------------------------------------------------------------------------------------------
def eval_diagnosis(per_type: int, n_null: int, seed: int = 11) -> dict:
    ev = Evaluator(_sut(), DEFAULT_CRITERIA)
    rng = np.random.default_rng(seed)
    cases, skipped = [], Counter()
    for fault, (component, param, ladder) in PLANTED.items():
        gen = ScenarioGenerator(seed + sum(map(ord, fault)), fault_prob=0.0)      # stable (str hash() is salted)
        got, tries = 0, 0
        while got < per_type and tries < 400:
            tries += 1
            base = gen.sample(f'plant-{fault}')
            clean_fail = sum(ev.evaluate(base.replace(seed=sd)).failed for sd in derived_seeds(base.seed, 8))
            if clean_fail > 1:
                continue                                   # planner already fails without the fault: not a clean plant
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
            kind = ev.evaluate(planted).primary_kind
            d = diagnose(ev, planted, kind=kind, n_seeds=8)
            sup = [h['component'] for h in d.hypotheses if h['status'] == 'supported']
            weak = [h['component'] for h in d.hypotheses if h['status'] == 'weak']
            shap = {c: v['value'] for c, v in d.shapley.items()}
            top = max(shap, key=shap.get) if shap else None
            cases.append({'planted': fault, 'expected': component, 'severity': planted.params()[param], 'kind': kind,
                          'top_shapley': top, 'supported': sup, 'weak': weak,
                          'top1_correct': top == component, 'expected_supported': component in sup,
                          'planner_logic_supported': 'planner_logic' in sup})
        skipped[fault] = per_type - got
    tp = sum(c['expected_supported'] for c in cases)
    n_sup = sum(len(c['supported']) for c in cases)
    n_correct_sup = sum(1 for c in cases if c['expected'] in c['supported'])
    by = {}
    for fault in PLANTED:
        rows = [c for c in cases if c['planted'] == fault]
        by[fault] = {'cases': len(rows), 'top1_accuracy': round(float(np.mean([c['top1_correct'] for c in rows])), 3) if rows else None,
                     'expected_supported_rate': round(float(np.mean([c['expected_supported'] for c in rows])), 3) if rows else None,
                     'planner_logic_false_flag_rate': round(float(np.mean([c['planner_logic_supported'] for c in rows])), 3) if rows else None,
                     'could_not_construct': skipped[fault]}
    # null cases: failures with no injected fault
    gen = ScenarioGenerator(seed + 7, fault_prob=0.0)
    null_rows, tries = [], 0
    while len(null_rows) < n_null and tries < 600:
        tries += 1
        s = gen.sample('null')
        a = ev.evaluate(s)
        if a.primary_kind != 'collision' or not np.mean([ev.evaluate(s.replace(seed=sd)).failed for sd in derived_seeds(s.seed, 8)]) >= 0.75:
            continue
        d = diagnose(ev, s, kind='collision', n_seeds=8)
        sup = [h['component'] for h in d.hypotheses if h['status'] == 'supported']
        null_rows.append({'supported': sup, 'planner_logic_supported': 'planner_logic' in sup,
                          'perception_falsely_supported': [c for c in sup if c != 'planner_logic']})
    n = len(cases)
    return {'planted_cases': n, 'by_fault': by,
            'top1_accuracy': round(float(np.mean([c['top1_correct'] for c in cases])), 3) if n else None,
            'expected_component_supported_rate': round(tp / n, 3) if n else None,
            'precision_of_supported_labels': round(n_correct_sup / n_sup, 3) if n_sup else None,
            'planner_logic_false_flag_rate': round(float(np.mean([c['planner_logic_supported'] for c in cases])), 3) if n else None,
            'null_cases': len(null_rows),
            'null_planner_logic_supported_rate': round(float(np.mean([r['planner_logic_supported'] for r in null_rows])), 3) if null_rows else None,
            'null_perception_false_support_rate': round(float(np.mean([bool(r['perception_falsely_supported']) for r in null_rows])), 3) if null_rows else None,
            'cases': cases, 'null_rows': null_rows}


# --- D. regression detection --------------------------------------------------------------------------------------------------------
def eval_regression(suite_budget: int, holdout_n: int, workdir: Path) -> dict:
    base_ev = Evaluator(_sut(), DEFAULT_CRITERIA)
    sres = run_search('random', base_ev, suite_budget, 123)
    suite_dir = workdir / 'suite'
    info = build_suite(base_ev, sres, suite_dir, n_fail=8, n_pass=8)
    baseline = create_baseline(base_ev, suite_dir, 'eval-base', 'unmodified planner')
    gen = ScenarioGenerator(777)
    holdout = [gen.sample('holdout') for _ in range(holdout_n)]

    def hold_stats(ev):
        r = [ev.evaluate(s) for s in holdout]
        return {'failing': sum(a.failed for a in r), 'collisions': sum(a.collision for a in r),
                'avoidable_collisions': sum(a.collision_class == 'avoidable' for a in r)}
    base_h = hold_stats(Evaluator(_sut(), DEFAULT_CRITERIA))
    rows = []
    for name, ov, intent in MUTANTS:
        ev = Evaluator(_sut(ov), DEFAULT_CRITERIA)
        h = hold_stats(ev)
        delta = h['failing'] - base_h['failing']
        thr = max(3, int(round(0.02 * holdout_n)))
        label = 'harmful' if delta >= thr else ('improvement' if delta <= -thr else 'neutral')
        reg = run_regression(Evaluator(_sut(ov), DEFAULT_CRITERIA), suite_dir, baseline, RegressionCriteria())
        rows.append({'mutant': name, 'intent': intent, 'holdout_failing': h['failing'], 'holdout_delta': delta, 'label': label,
                     'flagged': not reg['passed'], 'violations': reg['violations'], 'counts': reg['counts']})
    harmful = [r for r in rows if r['label'] == 'harmful']
    nonharm = [r for r in rows if r['label'] != 'harmful']
    return {'suite': info, 'holdout_n': holdout_n, 'holdout_baseline': base_h,
            'label_threshold_scenarios': max(3, int(round(0.02 * holdout_n))), 'mutants': rows,
            'harmful_n': len(harmful), 'harmful_detected': sum(r['flagged'] for r in harmful),
            'recall_on_harmful': round(sum(r['flagged'] for r in harmful) / len(harmful), 3) if harmful else None,
            'nonharmful_n': len(nonharm), 'nonharmful_flagged': sum(r['flagged'] for r in nonharm),
            'false_alarm_rate': round(sum(r['flagged'] for r in nonharm) / len(nonharm), 3) if nonharm else None,
            'identity_flagged': next(r['flagged'] for r in rows if r['mutant'] == 'identity')}


# --- driver ------------------------------------------------------------------------------------------------------------------------------
def run_evaluation(out: Path, budget: int = 150, trials: int = 10, quick: bool = False, planner: str = 'baseline') -> int:
    global ACTIVE_PLANNER
    ACTIVE_PLANNER = planner
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(budget=budget, trials=trials, min_limit=40, plant=12, null=20, suite_budget=300, holdout=300)
    if quick:
        cfg = dict(budget=30, trials=2, min_limit=5, plant=1, null=2, suite_budget=60, holdout=40)
    t0 = time.perf_counter()
    results: Dict = {'config': cfg, 'sut': _sut()().describe()}
    print(f'[A] search comparison: budget {cfg["budget"]}, {cfg["trials"]} trials, 2 regimes ...', flush=True)
    t = time.perf_counter()
    results['search'] = eval_search(cfg['budget'], cfg['trials'])
    results['search']['seconds'] = round(time.perf_counter() - t, 1)
    pool = results['search']['regimes']['full_box']['collected_scenarios'] + results['search']['regimes']['rare_box']['collected_scenarios']
    print('[B] minimisation ...', flush=True)
    t = time.perf_counter()
    results['minimization'] = eval_minimization(pool, cfg['min_limit'])
    results['minimization']['phase_seconds'] = round(time.perf_counter() - t, 1)
    print('[C] diagnosis (planted faults) ...', flush=True)
    t = time.perf_counter()
    results['diagnosis'] = eval_diagnosis(cfg['plant'], cfg['null'])
    results['diagnosis']['phase_seconds'] = round(time.perf_counter() - t, 1)
    print('[D] regression detection ...', flush=True)
    t = time.perf_counter()
    results['regression'] = eval_regression(cfg['suite_budget'], cfg['holdout'], out / 'regression_eval')
    results['regression']['phase_seconds'] = round(time.perf_counter() - t, 1)
    have_key = bool(os.environ.get('ANTHROPIC_API_KEY') or os.environ.get('OPENAI_API_KEY') or os.environ.get('REDTEAM_LLM_API_KEY'))
    results['agent_guided'] = {'status': 'not run',
                               'reason': ('a key is configured but agent-guided search is not implemented as a budgeted search driver; '
                                          'agent investigation is only unit-tested with a scripted model' if have_key else
                                          'no API key configured; the deterministic investigator is exercised in tests, not compared as a search method')}
    results['total_seconds'] = round(time.perf_counter() - t0, 1)
    for r in results['search']['regimes'].values():
        r.pop('collected_scenarios', None)
    (out / 'evaluation_results.json').write_text(json.dumps(results, indent=2, default=str))
    (out / 'evaluation_summary.md').write_text(summary_markdown(results))
    print(summary_markdown(results))
    print(f'wrote {out / "evaluation_results.json"} ({results["total_seconds"]}s)')
    return 0


def summary_markdown(r: dict) -> str:
    L = [f'# Autonomy Red Team evaluation ({r["config"]})\n']
    L.append('## A. Search at equal budget (mean +- std over trials; same seeds for both methods)\n')
    L.append('| regime | method | failing / budget | sims to first failure | unique reproducible findings | worst impact (m/s) |\n|---|---|---|---|---|---|')
    for reg, d in r['search']['regimes'].items():
        for m, s in d['summary'].items():
            f = lambda k: f'{s[k]["mean"]} +- {s[k]["std"]}'
            L.append(f'| {reg} | {m} | {f("failing")} / {r["search"]["budget"]} | {f("first_failure_sim")} | {f("unique_reproducible")} | {f("worst_impact_speed")} |')
    for reg, d in r['search']['regimes'].items():
        p = d['paired_cmaes_minus_random_unique_reproducible']
        L.append(f'\n{reg}: CMA-ES minus random, unique reproducible findings per trial: mean {p["mean_diff"]} (95% bootstrap CI {p["ci95"]}); '
                 f'cmaes better in {p["wins_first"]}, tie {p["ties"]}, random better {p["wins_second"]}. '
                 f'Distinct signatures over all trials: random {d["union_signatures"]["random"]}, cmaes {d["union_signatures"]["cmaes"]}, both {d["union_both"]}.')
    m = r['minimization']
    L.append('\n## B. Minimisation\n')
    if m.get('n'):
        L.append(f'- {m["n"]} distinct failures minimised; still fail on a fresh SUT: {m["still_fails_fresh_sut"]:.0%}; deterministic replay: {m["deterministic_replay"]:.0%}')
        L.append(f'- active perturbations {m["faults_before"]["mean"]} -> {m["faults_after"]["mean"]} (mean); distance to boring scenario {m["distance_before"]["mean"]} -> {m["distance_after"]["mean"]}; replays {m["replays"]["mean"]} per failure')
        L.append(f'- failure rate over 20 fresh noise seeds: original {m["seed_rate_original"]["mean"]}, minimised {m["seed_rate_minimized"]["mean"]}; seed-fragile (<25%): before {m["seed_fragile_before"]:.0%}, after {m["seed_fragile_after"]:.0%}')
    d = r['diagnosis']
    L.append('\n## C. Diagnosis (planted-fault recovery)\n')
    L.append(f'- {d["planted_cases"]} planted cases: top-1 Shapley accuracy {d["top1_accuracy"]}, planted component labelled *supported* in {d["expected_component_supported_rate"]}, precision of *supported* labels {d["precision_of_supported_labels"]}, planner-logic false flags {d["planner_logic_false_flag_rate"]}')
    L.append('\n| planted fault | cases | top-1 | expected supported | planner-logic false flag | could not construct |\n|---|---|---|---|---|---|')
    for k, v in d['by_fault'].items():
        L.append(f'| {k} | {v["cases"]} | {v["top1_accuracy"]} | {v["expected_supported_rate"]} | {v["planner_logic_false_flag_rate"]} | {v["could_not_construct"]} |')
    L.append(f'\nNo injected fault ({d["null_cases"]} failing scenarios): planner logic labelled supported in {d["null_planner_logic_supported_rate"]}, a perception component falsely labelled supported in {d["null_perception_false_support_rate"]}.')
    g = r['regression']
    L.append('\n## D. Regression detection\n')
    L.append(f'- suite: {g["suite"]["n_failures"]} known failures + {g["suite"]["n_passing"]} near-boundary passes; holdout {g["holdout_n"]} scenarios (label threshold +-{g["label_threshold_scenarios"]} failing scenarios)')
    L.append(f'- harmful mutants detected: {g["harmful_detected"]}/{g["harmful_n"]}; non-harmful flagged: {g["nonharmful_flagged"]}/{g["nonharmful_n"]}; identity flagged: {g["identity_flagged"]}\n')
    L.append('| mutant | intent | holdout delta | label | flagged | violations |\n|---|---|---|---|---|---|')
    for x in g['mutants']:
        L.append(f'| {x["mutant"]} | {x["intent"]} | {x["holdout_delta"]:+d} | {x["label"]} | {x["flagged"]} | {", ".join(x["violations"][:4]) or "-"} |')
    L.append(f'\n## E. Cost\n\nTotal {r["total_seconds"]}s on one CPU core (search {r["search"]["seconds"]}s, minimisation {m.get("phase_seconds")}s, '
             f'diagnosis {d["phase_seconds"]}s, regression {g["phase_seconds"]}s).')
    L.append(f'\n## F. Agent-guided\n\n{r["agent_guided"]["status"]}: {r["agent_guided"]["reason"]}')
    return '\n'.join(L) + '\n'
