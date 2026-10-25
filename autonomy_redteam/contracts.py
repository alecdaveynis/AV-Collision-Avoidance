"""Tiered regression detection: safety contracts for robust scenarios, statistics for noisy ones.

The original detector (``regress.py``, baseline format 1) compares ONE noise seed per scenario against fixed tolerances. On the
benchmark mutations it flagged 8 of 15 non-harmful changes, because a scenario close to the failure boundary flips between pass
and fail under harmless perturbations, and because any single worse metric counted as a violation however small.

This module (baseline format 2) runs every suite scenario on ``n_seeds`` paired noise seeds in both versions and separates four
kinds of change, only the first two of which fail CI:

1. **Hard safety violation.** A scenario carries a *contract* derived from the baseline, never from the new version:
     * ``safe_contract``  the baseline had no safety failure (collision, near miss, missed hazard) in any seed AND kept at least
                          ``contract_margin_m`` of worst-seed clearance. Contract: no safety failure in any seed.
     * ``known_failure``  the baseline failed in >= 75 % of seeds (a promoted failure, a tripwire). Contract: it must not get
                          materially worse: mean impact speed up by more than ``known_worse_abs`` m/s AND ``known_worse_rel`` AND
                          worse in at least 75 % of seeds. (The max over seeds is far too noisy to be a contract.) Getting better is
                          reported, never punished.
   Marginal scenarios (neither) carry no contract: a pass/fail flip there is evidence about a rate, not a violation.
2. **Statistically meaningful degradation.** Exact tests whose false-alarm budget is controlled across the whole suite
   (Bonferroni over scenarios for per-scenario tests; a small fixed alpha for each pooled test):
     * per scenario, one-sided Fisher test on the safety-failure count (at least ``min_extra_failures`` more failing seeds);
     * pooled over all (scenario, seed) pairs: exact McNemar on safety failures, and exact one-sided sign tests on the paired
       change of minimum gap, impact speed and unnecessary-braking time, counting only pairs that moved by more than the
       tolerance. Pooling gives power against diffuse degradation that no single scenario can show.
   Sign tests replace bootstrap-of-the-mean: minimum gap and impact speed are heavy-tailed (a collision is a jump), the
   bootstrap was anticonservative with 8 seeds and produced most of the false alarms in the noise-only experiment.
3. **Informational.** Significant but non-safety changes (reaction time, peak deceleration, state transitions), improvements,
   and safety changes that are not statistically distinguishable from noise.
4. **Within tolerance.** Everything else.

Contracts are an explicit, reviewable statement ("this scenario must stay collision-free") and are kept deliberately strict, so
a change that fixes many failures but breaks a safe scenario is still flagged; it is annotated with the net suite effect so that a
reviewer can see the trade-off.

Seeds are paired: the baseline and the current run use the same noise seeds, so identical software gives identical results and the
exit code is deterministic. The false-alarm rate of the statistical tier under pure seed noise is measured in ``regression_eval.py``.
"""
from __future__ import annotations

import datetime as _dt
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import __version__
from .metrics import Assessment
from .minimize import derived_seeds
from .regress import _git_commit, load_suite
from .runner import Evaluator
from .scenario import Scenario, ScenarioValidationError

BASELINE_FORMAT_V2 = 2
SAFETY_KINDS = ('collision', 'near_miss', 'missed_hazard')
ROLES = ('safe_contract', 'known_failure', 'marginal')


@dataclass(frozen=True)
class Policy:
    n_seeds: int = 12
    contract_margin_m: float = 1.5          # baseline worst-seed min gap required to promote a passing scenario to a contract
    known_failure_rate: float = 0.75
    known_worse_abs: float = 3.0            # m/s, mean impact speed increase that makes a known failure materially worse
    known_worse_rel: float = 0.25
    alpha_scenario_family: float = 0.02     # Bonferroni-split over the scenarios of the suite
    alpha_suite: float = 0.005              # each pooled test
    min_extra_failures: int = 2
    suite_min_net: int = 3
    pooled_min_net: int = 4                 # pooled sign tests: at least this many more worse than better pairs
    gap_tol: float = 0.5                    # m
    impact_tol: float = 1.0                 # m/s
    unnecessary_tol: float = 0.5            # s
    reaction_tol: float = 0.2               # s
    decel_tol: float = 1.0                  # m/s^2 (informational)
    bootstrap: int = 2000
    strict: bool = False                    # a scenario missing from the baseline also fails


# --- statistics (small, dependency free, exact) -------------------------------------------------------------------------------
def fisher_one_sided(k_new: int, n_new: int, k_old: int, n_old: int) -> float:
    """P(at least k_new failures in the new run | the pooled failure count is spread at random over both runs)."""
    total_fail = k_new + k_old
    n = n_new + n_old
    denom = math.comb(n, total_fail)
    hi = min(n_new, total_fail)
    return float(sum(math.comb(n_new, i) * math.comb(n_old, total_fail - i) for i in range(k_new, hi + 1)) / denom)


def mcnemar_one_sided(worse: int, better: int) -> float:
    """Exact one-sided sign test: P(>= ``worse`` of the discordant pairs went the bad way | no effect)."""
    m = worse + better
    if m == 0:
        return 1.0
    return float(sum(math.comb(m, i) for i in range(worse, m + 1)) / 2 ** m)


def paired_bootstrap(diffs: Sequence[float], n: int = 2000, seed: int = 0) -> Tuple[float, float, float]:
    d = np.asarray(diffs, float)
    if len(d) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# --- per-seed records ---------------------------------------------------------------------------------------------------------
def _rec(a: Assessment) -> dict:
    d = a.to_dict()
    fin = lambda x: None if x is None or (isinstance(x, float) and not math.isfinite(x)) else float(x)  # noqa: E731
    return {'safety_failure': any(k in SAFETY_KINDS for k in d['failure_kinds']), 'failure_kinds': d['failure_kinds'],
            'collision': bool(d['collision']), 'min_gap': fin(d['min_gap']), 'impact_speed': float(d['impact_speed']),
            'min_ttc': fin(d['min_ttc']), 'brake_reaction_time': fin(d['brake_reaction_time']),
            'unnecessary_braking_s': float(d['unnecessary_braking_s']), 'peak_cmd_decel': float(d['peak_cmd_decel']),
            'n_states': len(d['states_visited']), 'collision_class': d['collision_class']}


def _runs(ev: Evaluator, sc: Scenario, seeds: Sequence[int]) -> List[dict]:
    return [_rec(ev.evaluate(sc.replace(seed=s))) for s in seeds]


def _role(runs: List[dict], pol: Policy) -> str:
    n = len(runs)
    k = sum(r['safety_failure'] for r in runs)
    if k == 0 and min((r['min_gap'] if r['min_gap'] is not None else 99.0) for r in runs) >= pol.contract_margin_m:
        return 'safe_contract'
    if k >= pol.known_failure_rate * n and any(r['collision'] for r in runs):
        return 'known_failure'
    return 'marginal'


def create_baseline_v2(ev: Evaluator, suite_dir: Path, version: str, note: str = '', pol: Policy = Policy()) -> dict:
    scen: Dict[str, dict] = {}
    for name, sc, _meta in load_suite(suite_dir):
        seeds = derived_seeds(sc.seed, pol.n_seeds)
        runs = _runs(ev, sc, seeds)
        scen[name] = {'fingerprint': sc.fingerprint(), 'role': _role(runs, pol), 'seeds': list(seeds), 'runs': runs,
                      'summary': {'safety_failures': sum(r['safety_failure'] for r in runs),
                                  'collisions': sum(r['collision'] for r in runs),
                                  'worst_min_gap': min((r['min_gap'] if r['min_gap'] is not None else 99.0) for r in runs),
                                  'worst_impact_speed': max(r['impact_speed'] for r in runs)}}
    return {'baseline_format': BASELINE_FORMAT_V2, 'baseline_version': version, 'note': note,
            'created_utc': _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds'), 'git_commit': _git_commit(),
            'framework_version': __version__, 'sut': ev.sut_info, 'policy': asdict(pol), 'scenarios': scen}


# --- comparison ---------------------------------------------------------------------------------------------------------------
def _paired_metric(base: List[dict], cur: List[dict], key: str, fill: Optional[float] = None) -> List[float]:
    out = []
    for b, c in zip(base, cur):
        x, y = b[key], c[key]
        x = fill if x is None else x
        y = fill if y is None else y
        if x is not None and y is not None:
            out.append(float(y) - float(x))
    return out


def compare_scenario(name: str, role: str, base: List[dict], cur: List[dict], pol: Policy, n_scenarios: int = 1) -> dict:
    n = len(base)
    kb = sum(r['safety_failure'] for r in base)
    kc = sum(r['safety_failure'] for r in cur)
    cb = sum(r['collision'] for r in base)
    cc = sum(r['collision'] for r in cur)
    hard: List[str] = []
    degr: List[str] = []
    info: List[str] = []
    improved = False

    # 1. contracts (derived from the baseline only)
    if role == 'safe_contract' and kc > 0:
        kinds = sorted({k for r in cur for k in r['failure_kinds'] if k in SAFETY_KINDS})
        hard.append(f'contract broken: baseline had no safety failure in {n} seeds (worst clearance '
                    f'{min((r["min_gap"] if r["min_gap"] is not None else 99.0) for r in base):.1f} m); now {", ".join(kinds)} in {kc}/{n} seeds')
    if role == 'known_failure':
        ib = [r['impact_speed'] for r in base]
        ic = [r['impact_speed'] for r in cur]
        mb, mc = float(np.mean(ib)), float(np.mean(ic))
        worse_seeds = sum(c > b + 1e-9 for b, c in zip(ib, ic))
        if mc - mb > max(pol.known_worse_abs, pol.known_worse_rel * mb) and worse_seeds >= 0.75 * n:
            hard.append(f'known failure materially worse: mean impact speed {mb:.1f} -> {mc:.1f} m/s, worse in {worse_seeds}/{n} seeds')
        if cc < cb or kc < kb:
            improved = True
            info.append(f'known failure reduced: collisions {cb}/{n} -> {cc}/{n}, safety failures {kb}/{n} -> {kc}/{n}')

    # 2. per-scenario statistics: failure counts only (Bonferroni over the scenarios); continuous metrics are pooled in aggregate()
    alpha = pol.alpha_scenario_family / max(n_scenarios, 1)
    p = fisher_one_sided(kc, n, kb, n)
    if kc - kb >= pol.min_extra_failures and p < alpha:
        degr.append(f'safety-failure rate {kb}/{n} -> {kc}/{n} (one-sided Fisher p={p:.4f} < {alpha:.4f})')
    elif kc > kb:
        info.append(f'safety failures {kb}/{n} -> {kc}/{n}: not significant after correction for {n_scenarios} scenarios (p={p:.3f})')
    elif kc < kb and not role == 'known_failure':
        improved = True
        info.append(f'safety failures {kb}/{n} -> {kc}/{n}')

    # 3. informational: effect sizes with bootstrap intervals (descriptive, never a failure criterion on their own)
    for key, tol, label, unit, fill in (('min_gap', pol.gap_tol, 'min gap', 'm', 99.0),
                                        ('impact_speed', pol.impact_tol, 'impact speed', 'm/s', 0.0),
                                        ('unnecessary_braking_s', pol.unnecessary_tol, 'unnecessary braking', 's', 0.0),
                                        ('brake_reaction_time', pol.reaction_tol, 'brake reaction time', 's', None),
                                        ('peak_cmd_decel', pol.decel_tol, 'peak commanded deceleration', 'm/s^2', None),
                                        ('n_states', 0.5, 'planner states visited', '', None)):
        d = _paired_metric(base, cur, key, fill=fill)
        if len(d) >= 4:
            m, lo, hi = paired_bootstrap(d, pol.bootstrap)
            if abs(m) > tol and (lo > 0 or hi < 0):
                info.append(f'{label} {m:+.2f} {unit} (95% CI {lo:+.2f}..{hi:+.2f})'.replace('  ', ' '))
                if key == 'min_gap' and m > 0:
                    improved = True

    if hard:
        status = 'hard_violation'
    elif degr:
        status = 'degraded'
    elif improved:
        status = 'improved'
    elif info:
        status = 'informational'
    else:
        status = 'within_tolerance'
    return {'scenario': name, 'role': role, 'status': status, 'hard': hard, 'degraded': degr, 'info': info,
            'safety_failures': [kb, kc], 'collisions': [cb, cc], 'n_seeds': n, 'p_fisher': round(p, 5)}


def sign_test_pooled(items, key: str, tol: float, fill: float, sign: int) -> Tuple[int, int, float]:
    """Pooled over all (scenario, seed) pairs that moved by more than ``tol``: (#worse, #better, one-sided exact p of 'worse').
    ``sign`` = +1 when an increase is worse (impact speed, unnecessary braking), -1 when a decrease is worse (minimum gap)."""
    worse = better = 0
    for _name, _role_, base, cur in items:
        for d in _paired_metric(base, cur, key, fill=fill):
            if sign * d > tol:
                worse += 1
            elif sign * d < -tol:
                better += 1
    return worse, better, mcnemar_one_sided(worse, better)


POOLED = (('min_gap', 'gap_tol', 99.0, -1, 'minimum gap'), ('impact_speed', 'impact_tol', 0.0, +1, 'impact speed'),
          ('unnecessary_braking_s', 'unnecessary_tol', 0.0, +1, 'unnecessary braking time'))


def aggregate(items: List[Tuple[str, str, List[dict], List[dict]]], pol: Policy) -> Tuple[List[dict], dict]:
    """Per-scenario verdicts and the pooled suite-level tests from (name, role, baseline_runs, current_runs)."""
    rows, worse, better = [], 0, 0
    for name, role, base, cur in items:
        rows.append(compare_scenario(name, role, base, cur, pol, n_scenarios=len(items)))
        for rb, rc in zip(base, cur):
            worse += int(rc['safety_failure'] and not rb['safety_failure'])
            better += int(rb['safety_failure'] and not rc['safety_failure'])
    p_suite = mcnemar_one_sided(worse, better)
    suite = {'pairs_worse': worse, 'pairs_better': better, 'p_one_sided': round(p_suite, 6),
             'p_one_sided_improvement': round(mcnemar_one_sided(better, worse), 6),
             'degraded': bool(worse - better >= pol.suite_min_net and p_suite < pol.alpha_suite),
             'improved': bool(better - worse >= pol.suite_min_net and mcnemar_one_sided(better, worse) < pol.alpha_suite),
             'pooled': {}}
    for key, tol_name, fill, sign, label in POOLED:
        w, b, p = sign_test_pooled(items, key, getattr(pol, tol_name), fill, sign)
        sig = bool(w - b >= pol.pooled_min_net and p < pol.alpha_suite)
        suite['pooled'][key] = {'label': label, 'worse': w, 'better': b, 'p_one_sided': round(p, 6), 'degraded': sig}
        suite['degraded'] = bool(suite['degraded'] or sig)
    return rows, suite


def decide(rows: List[dict], suite: dict, pol: Policy) -> Tuple[str, bool]:
    tier = ('hard_violation' if any(r['status'] == 'hard_violation' for r in rows)
            else 'degraded' if (suite['degraded'] or any(r['status'] == 'degraded' for r in rows))
            else 'stale_baseline' if any(r['status'] == 'stale_baseline' for r in rows) else 'ok')
    passed = tier == 'ok' and not (pol.strict and any(r['status'] == 'missing_baseline' for r in rows))
    return tier, passed


def run_regression_v2(ev: Evaluator, suite_dir: Path, baseline: dict, pol: Optional[Policy] = None) -> dict:
    if baseline.get('baseline_format') != BASELINE_FORMAT_V2:
        raise ScenarioValidationError(f'not a format-2 baseline: {baseline.get("baseline_format")!r}')
    pol = pol or Policy(**baseline.get('policy', {}))
    bs = baseline['scenarios']
    special: List[dict] = []
    items = []
    suite_items = load_suite(suite_dir)
    for name, sc, _meta in suite_items:
        b = bs.get(name)
        if b is None:
            special.append({'scenario': name, 'role': None, 'status': 'missing_baseline', 'hard': [], 'degraded': [],
                            'info': ['scenario is not in the baseline'], 'safety_failures': None, 'collisions': None, 'n_seeds': 0})
        elif b['fingerprint'] != sc.fingerprint():
            special.append({'scenario': name, 'role': b['role'], 'status': 'stale_baseline', 'hard': [], 'degraded': [],
                            'info': ['scenario parameters differ from the baseline recording: regenerate the baseline'],
                            'safety_failures': None, 'collisions': None, 'n_seeds': 0})
        else:
            items.append((name, b['role'], b['runs'], _runs(ev, sc, b['seeds'])))
    rows, suite = aggregate(items, pol)
    rows += special
    names = {r['scenario'] for r in rows}
    for n in bs:
        if n not in names:
            rows.append({'scenario': n, 'role': bs[n]['role'], 'status': 'removed_from_suite', 'hard': [], 'degraded': [],
                         'info': ['in baseline, not in suite'], 'safety_failures': None, 'collisions': None, 'n_seeds': 0})
    fail_status = {'hard_violation', 'degraded', 'stale_baseline'} | ({'missing_baseline'} if pol.strict else set())
    violations = [r['scenario'] for r in rows if r['status'] in fail_status]
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r['status']] = counts.get(r['status'], 0) + 1
    tier, passed = decide(rows, suite, pol)
    tradeoff = None
    if not passed and suite['improved']:
        tradeoff = (f'net suite effect is positive ({suite["pairs_better"]} (scenario, seed) outcomes fixed vs {suite["pairs_worse"]} broken, '
                    f'p={suite["p_one_sided_improvement"]:.4f}) but the violation(s) above are still real and need a decision')
    return {'baseline_format': 2, 'baseline_version': baseline.get('baseline_version'), 'baseline_commit': baseline.get('git_commit'),
            'baseline_sut': baseline.get('sut'), 'current_sut': ev.sut_info, 'suite': str(suite_dir), 'policy': asdict(pol),
            'counts': counts, 'n_scenarios': len(suite_items), 'violations': violations, 'tier': tier, 'passed': passed,
            'suite_level': suite, 'tradeoff_note': tradeoff, 'rows': rows, 'simulations': ev.n_sims}


def to_markdown_v2(res: dict) -> str:
    L = [f'# Safety regression report: {"PASS" if res["passed"] else "FAIL"} ({res["tier"]})\n',
         f'- Baseline: `{res["baseline_version"]}` (commit {res["baseline_commit"]}), suite `{res["suite"]}`, '
         f'{res["n_scenarios"]} scenarios x {res["policy"]["n_seeds"]} paired seeds, {res["simulations"]} simulations',
         f'- Baseline SUT: `{res["baseline_sut"]}`', f'- Current SUT:  `{res["current_sut"]}`',
         '- Counts: ' + ', '.join(f'{k}: {v}' for k, v in sorted(res['counts'].items()))]
    s = res['suite_level']
    L.append(f'- Suite level (exact McNemar over paired scenario/seed outcomes): {s["pairs_worse"]} newly failing, '
             f'{s["pairs_better"]} newly passing, one-sided p={s["p_one_sided"]}; '
             + ('**statistically meaningful degradation**' if s['degraded'] else 'improvement' if s['improved'] else 'no meaningful suite-level change'))
    for key, d in s.get('pooled', {}).items():
        L.append(f'  - pooled sign test, {d["label"]}: {d["worse"]} pairs worse vs {d["better"]} better (beyond tolerance), p={d["p_one_sided"]}'
                 + (' **degraded**' if d['degraded'] else ''))
    if res.get('tradeoff_note'):
        L.append(f'- Trade-off: {res["tradeoff_note"]}')
    L.append('')
    L.append('| scenario | role | tier | safety failures (base -> now) | details |\n|---|---|---|---|---|')
    order = {'hard_violation': 0, 'degraded': 1, 'stale_baseline': 2, 'missing_baseline': 3, 'removed_from_suite': 4,
             'improved': 5, 'informational': 6, 'within_tolerance': 7}
    for r in sorted(res['rows'], key=lambda r: (order.get(r['status'], 9), r['scenario'])):
        sf = f'{r["safety_failures"][0]}/{r["n_seeds"]} -> {r["safety_failures"][1]}/{r["n_seeds"]}' if r['safety_failures'] else '-'
        det = '; '.join(r['hard'] + r['degraded'] + r['info'])
        L.append(f'| `{r["scenario"]}` | {r["role"] or "-"} | {r["status"]} | {sf} | {det} |')
    return '\n'.join(L) + '\n'
