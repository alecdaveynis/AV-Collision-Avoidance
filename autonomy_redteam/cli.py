"""Command line interface: ``python -m autonomy_redteam <command>``.

search      find failures (random or CMA-ES) under a simulation budget, group and rank them
minimize    delta-debugging style minimisation of one failing scenario
diagnose    counterfactual root-cause experiments for one failing scenario
investigate minimise + diagnose + full report (plots, GIF, Markdown) for one scenario
promote     copy a scenario into a regression suite
suite       build a regression suite from a search
baseline    record the current software's results on a suite as a versioned baseline
regress     compare the current software against a baseline (nonzero exit on regression)
demo        small deterministic end-to-end campaign
evaluate    framework evaluation (random vs CMA-ES, minimisation, diagnosis, regression detection)
agent       optional LLM investigation (works without a key: deterministic investigator)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from . import __version__
from .adapters import SafetySimSUT, default_sut_factory
from .attribution import attribute, summary_lines
from .diagnose import diagnose
from .findings import group_failures, sig_str
from .investigate import investigate
from .metrics import DEFAULT_CRITERIA
from .minimize import minimize
from .regress import (RegressionCriteria, create_baseline, promote, run_regression, to_markdown)
from .runner import Evaluator
from .scenario import Scenario, ScenarioValidationError
from .search import METHODS, run_search
from .suite import build_suite

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'output' / 'redteam'


def _parse_overrides(items: Optional[List[str]]) -> Dict[str, float]:
    out = {}
    for it in items or []:
        if '=' not in it:
            raise ScenarioValidationError(f'--override expects KEY=VALUE, got {it!r}')
        k, v = it.split('=', 1)
        try:
            out[k.strip()] = float(v)
        except ValueError:
            raise ScenarioValidationError(f'--override value must be a number: {it!r}')
    return out


def resolve_planner(args):
    """(braking policy, TTC mode). ``--planner kalman|analytic`` is accepted as the legacy spelling of ``--ttc-mode``."""
    planner = getattr(args, 'planner', None) or 'baseline'
    ttc_mode = getattr(args, 'ttc_mode', None) or 'kalman'
    if planner in ('kalman', 'analytic'):
        planner, ttc_mode = 'baseline', planner
    return planner, ttc_mode


def make_evaluator(args, **kw) -> Evaluator:
    overrides = _parse_overrides(getattr(args, 'override', None))
    planner, ttc_mode = resolve_planner(args)
    factory = default_sut_factory(ttc_mode, overrides, planner)
    try:
        factory()                              # validates the override keys early
        from .adapters.safety_sim_adapter import patched_config
        with patched_config(overrides):
            pass
    except KeyError as e:
        raise ScenarioValidationError(str(e))
    return Evaluator(factory, DEFAULT_CRITERIA, **kw)


def _add_sut_args(p):
    p.add_argument('--planner', choices=('baseline', 'improved', 'kalman', 'analytic'), default='baseline',
                   help="braking policy of the system under test: 'baseline' (original BehaviorManager) or 'improved' "
                        "(src/safe_braking.py). 'kalman'/'analytic' are accepted as legacy aliases of --ttc-mode")
    p.add_argument('--ttc-mode', choices=('kalman', 'analytic'), default=None,
                   help='TTC source of the planner (default kalman; TTC fault injection needs kalman)')
    p.add_argument('--override', action='append', metavar='KEY=VALUE',
                   help='override a planner threshold in src/config.py (e.g. TTC_SLOW=2.0); repeatable. '
                        'Emulates a new software version for regression experiments')


def _print(*a):
    print(*a, flush=True)


def _write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n')
    return path


# --- commands --------------------------------------------------------------------------------------------------------
def cmd_search(args) -> int:
    ev = make_evaluator(args)
    out = Path(args.out or DEFAULT_OUT / f'search_{args.method}_seed{args.seed}')
    _print(f'search: method={args.method} budget={args.budget} seed={args.seed} sut={ev.sut_info}')
    res = run_search(args.method, ev, args.budget, args.seed, fault_prob=args.fault_prob)
    findings = group_failures(ev, res.evaluations)
    data = res.to_dict()
    data['unique_findings'] = len(findings)
    data['post_processing_simulations'] = ev.n_sims - res.sims_used
    _write_json(out / 'search_results.json', data)
    _write_json(out / 'findings.json', [dict(rank=i + 1, **f.to_dict()) for i, f in enumerate(findings)])
    for i, f in enumerate(findings[:args.top], 1):
        f.representative.replace(name=f'finding_{i:02d}').save(out / 'failures' / f'finding_{i:02d}.json')
    _print(f'{res.sims_used} simulations, {len(res.failures)} failing ({len(res.failures) / max(res.sims_used, 1):.0%}); '
           f'first failure at simulation #{res.first_failure_sim}; {len(findings)} distinct findings '
           f'(equivalence: kind + necessary faults + lead regime). {res.wall_seconds:.1f}s')
    if not findings:
        _print('no failure found within the budget (nothing is fabricated).')
    else:
        _print(f'\n{"#":>2} {"severity":>8} {"count":>5}  signature')
        for i, f in enumerate(findings[:args.top], 1):
            _print(f'{i:>2} {f.severity:8.1f} {f.members:5d}  {sig_str(f.signature)}')
            _print(f'     {f.representative.describe()}')
    _print(f'\nresults: {out}')
    if args.investigate and findings:
        for i, f in enumerate(findings[:args.investigate], 1):
            inv = investigate(ev, f.representative.replace(name=f'finding_{i:02d}'), out / f'investigation_{i:02d}',
                              gif=not args.no_gif)
            _print(f'investigated finding {i}: {inv.files["report"]}')
    return 0


def _load_scenario(path) -> Scenario:
    return Scenario.load(path)


def cmd_minimize(args) -> int:
    ev = make_evaluator(args)
    sc = _load_scenario(args.scenario)
    out = Path(args.out or Path(args.scenario).with_suffix('').as_posix() + '_minimized')
    res = minimize(ev, sc, seeds=args.seeds, reduce_physical=not args.no_physical)
    out.mkdir(parents=True, exist_ok=True)
    res.minimized.save(out / 'minimized_scenario.json')
    sc.save(out / 'original_scenario.json')
    _write_json(out / 'minimization.json', res.to_dict())
    _print(f'failure kind: {res.kind}')
    _print(f'original : {sc.describe()}')
    _print(f'minimized: {res.minimized.describe()}')
    _print(f'necessary faults: {", ".join(res.necessary_faults) or "none"}   removed: {", ".join(res.removed_faults) or "none"}')
    _print(f'reproducibility: {res.reproducible["status"]} (deterministic replay: {res.reproducible["deterministic_replay"]}; '
           f'failure rate over {res.reproducible["n_seeds"]} fresh seeds: minimized {res.reproducible["seed_failure_rate_minimized"]:.0%}, '
           f'original {res.reproducible["seed_failure_rate_original"]:.0%})')
    if res.predicate_fallback:
        _print('note: original fails on a minority of noise seeds; minimised with its own seed only.')
    _print('changes:')
    for c in res.changes:
        _print(f'  [{c.phase}] {c.param}: {c.old} -> {c.new}  ({c.note})')
    _print(f'replays: {res.replays}; distance to boring scenario {res.distance_before:.2f} -> {res.distance_after:.2f}')
    _print(f'saved: {out}')
    return 0


def cmd_diagnose(args) -> int:
    ev = make_evaluator(args)
    sc = _load_scenario(args.scenario)
    d = diagnose(ev, sc, n_seeds=args.seeds)
    d.attribution = attribute(ev, sc, d)
    out = Path(args.out or Path(args.scenario).with_suffix('').as_posix() + '_diagnosis.json')
    _write_json(out, d.to_dict())
    _print(f'failure kind: {d.kind}; {d.sims} simulations; original fails in {d.variants[0].k}/{d.variants[0].n} seeds')
    for line in summary_lines(d.attribution):
        _print(line)
    _print('legacy Shapley view (outcome sensitivity, not root cause):')
    for h in d.ranked_hypotheses():
        _print(f'  {h["status"]:>13}  {h["label"]}  (share {h["share"]:+.2f})')
        for line in h['evidence']:
            _print(f'                 - {line}')
    for n in d.notes:
        _print(f'  note: {n}')
    _print(f'saved: {out}')
    return 0


def cmd_investigate(args) -> int:
    ev = make_evaluator(args)
    sc = _load_scenario(args.scenario)
    out = Path(args.out or DEFAULT_OUT / f'investigation_{sc.name or sc.fingerprint()}')
    inv = investigate(ev, sc, out, minimize_seeds=args.min_seeds, diag_seeds=args.seeds, gif=not args.no_gif)
    _print(json.dumps(inv.summary(), indent=2))
    return 0


def cmd_promote(args) -> int:
    sc = _load_scenario(args.scenario)
    p = promote(sc, Path(args.suite), args.name, meta={'origin': 'promote', 'source': str(args.scenario)})
    _print(f'promoted to {p}')
    return 0


def cmd_suite(args) -> int:
    ev = make_evaluator(args)
    res = run_search(args.method, ev, args.budget, args.seed)
    info = build_suite(ev, res, Path(args.out), args.n_fail, args.n_pass)
    _print(json.dumps(info, indent=2))
    return 0


def cmd_baseline(args) -> int:
    ev = make_evaluator(args)
    if args.seeds and args.seeds >= 2:
        from .contracts import Policy, create_baseline_v2
        b = create_baseline_v2(ev, Path(args.suite), args.version, args.note or '', Policy(n_seeds=args.seeds))
        _write_json(Path(args.out), b)
        roles: dict = {}
        for v in b['scenarios'].values():
            roles[v['role']] = roles.get(v['role'], 0) + 1
        _print(f'baseline {args.version} (format 2, {args.seeds} paired seeds): {len(b["scenarios"])} scenarios, roles {roles} -> {args.out}')
        return 0
    b = create_baseline(ev, Path(args.suite), args.version, args.note or '')
    _write_json(Path(args.out), b)
    nfail = sum(1 for s in b['scenarios'].values() if s['result']['failure_kinds'])
    _print(f'baseline {args.version}: {len(b["scenarios"])} scenarios ({nfail} failing) -> {args.out}')
    return 0


def cmd_regress(args) -> int:
    try:
        baseline = json.loads(Path(args.baseline).read_text())
    except (OSError, json.JSONDecodeError) as e:
        _print(f'error: cannot read baseline {args.baseline}: {e}')
        return 2
    ev = make_evaluator(args)
    if baseline.get('baseline_format') == 2:
        from .contracts import run_regression_v2, to_markdown_v2
        res = run_regression_v2(ev, Path(args.suite), baseline)
        out = Path(args.out or DEFAULT_OUT / 'regress')
        _write_json(out / 'regression_results.json', res)
        md = to_markdown_v2(res)
        (out / 'regression_summary.md').write_text(md)
        _print(md)
        _print(f'wrote {out / "regression_results.json"} and {out / "regression_summary.md"}')
        return 0 if res['passed'] else 1
    crit = RegressionCriteria(gap_tol=args.gap_tol, impact_tol=args.impact_tol, ttc_tol=args.ttc_tol,
                              reaction_tol=args.reaction_tol, strict=args.strict)
    res = run_regression(ev, Path(args.suite), baseline, crit)
    out = Path(args.out or DEFAULT_OUT / 'regress')
    _write_json(out / 'regression_results.json', res)
    md = to_markdown(res)
    (out / 'regression_summary.md').write_text(md)
    _print(md)
    _print(f'wrote {out / "regression_results.json"} and {out / "regression_summary.md"}')
    return 0 if res['passed'] else 1


def cmd_demo(args) -> int:
    from .demo import run_demo
    return run_demo(Path(args.out or DEFAULT_OUT / 'demo'), budget=args.budget, seed=args.seed, gif=not args.no_gif)


def cmd_fix_demo(args) -> int:
    from .fix_demo import run_fix_demo
    return run_fix_demo(Path(args.out), seed=args.seed, budget=args.budget, n_seeds=args.seeds,
                        validation_n=args.validation, gif=not args.no_gif)


def cmd_regression_eval(args) -> int:
    from . import regression_eval as re_
    res = re_.evaluate_split(args.split, Path(args.out), work=Path(args.work) if args.work else None, noise_trials=args.noise_trials)
    _print((Path(args.out) / f'regression_eval_{args.split}.md').read_text())
    return 0


def cmd_diag_eval(args) -> int:
    from . import diagnosis_eval as de
    out = Path(args.out)
    if args.split == 'calibrate':
        r = de.calibrate_nominal(out / 'nominal_calibration.json')
        _print(json.dumps(r['statistics'], indent=1))
        return 0
    res = de.run(args.split, out, cache=Path(args.cache) if args.cache else None)
    _print(de.to_markdown(res))
    _print(f'saved: {out}/diagnosis_eval_{args.split}.json|md')
    return 0


def cmd_compare(args) -> int:
    from .planner_eval import PLANNERS, SETS, run_comparison, run_sensitivity
    out = Path(args.out or DEFAULT_OUT / 'planner_comparison')
    res = run_comparison(out, args.split, tuple(args.planners), args.set or None, args.ttc_mode or 'kalman')
    _print(f"\nwrote {out}/planner_comparison_{args.split}.md and .json ({res['seconds']} s)")
    if args.sensitivity:
        run_sensitivity(out, args.sensitivity_set, tuple(args.lag), tuple(args.planners))
    return 0


def cmd_evaluate(args) -> int:
    from .evaluation import run_evaluation
    return run_evaluation(Path(args.out or DEFAULT_OUT / 'evaluation'), budget=args.budget, trials=args.trials,
                          quick=args.quick, planner=args.planner)


def cmd_agent(args) -> int:
    from .agent import run_agent_cli
    return run_agent_cli(args, make_evaluator)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='python -m autonomy_redteam', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--version', action='version', version=f'autonomy_redteam {__version__}')
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('search', help='search for failures')
    s.add_argument('--budget', type=int, default=100, help='simulation budget')
    s.add_argument('--seed', type=int, default=42)
    s.add_argument('--method', choices=METHODS, default='random')
    s.add_argument('--fault-prob', type=float, default=0.5, help='probability each perturbation is active in a sample')
    s.add_argument('--top', type=int, default=10, help='how many ranked findings to print / save')
    s.add_argument('--investigate', type=int, default=0, metavar='N', help='also investigate the top N findings')
    s.add_argument('--no-gif', action='store_true')
    s.add_argument('--out')
    _add_sut_args(s)
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser('minimize', help='minimise a failing scenario')
    s.add_argument('--scenario', required=True)
    s.add_argument('--seeds', type=int, default=3, help='noise seeds in the failure predicate (majority, own seed must fail)')
    s.add_argument('--no-physical', action='store_true', help='only remove / reduce perturbations; keep physical parameters')
    s.add_argument('--out')
    _add_sut_args(s)
    s.set_defaults(fn=cmd_minimize)

    s = sub.add_parser('diagnose', help='counterfactual root-cause experiments')
    s.add_argument('--scenario', required=True)
    s.add_argument('--seeds', type=int, default=8, help='noise seeds per variant')
    s.add_argument('--out')
    _add_sut_args(s)
    s.set_defaults(fn=cmd_diagnose)

    s = sub.add_parser('investigate', help='minimise + diagnose + report')
    s.add_argument('--scenario', required=True)
    s.add_argument('--seeds', type=int, default=8)
    s.add_argument('--min-seeds', type=int, default=3)
    s.add_argument('--no-gif', action='store_true')
    s.add_argument('--out')
    _add_sut_args(s)
    s.set_defaults(fn=cmd_investigate)

    s = sub.add_parser('promote', help='add a scenario to a regression suite')
    s.add_argument('--scenario', required=True)
    s.add_argument('--suite', required=True)
    s.add_argument('--name')
    s.set_defaults(fn=cmd_promote)

    s = sub.add_parser('suite', help='build a regression suite from a search')
    s.add_argument('--out', required=True)
    s.add_argument('--budget', type=int, default=300)
    s.add_argument('--seed', type=int, default=0)
    s.add_argument('--method', choices=METHODS, default='random')
    s.add_argument('--n-fail', type=int, default=6)
    s.add_argument('--n-pass', type=int, default=6)
    _add_sut_args(s)
    s.set_defaults(fn=cmd_suite)

    s = sub.add_parser('baseline', help='record a versioned baseline for a suite')
    s.add_argument('--suite', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--version', required=True, help='label of the software version being recorded')
    s.add_argument('--note')
    s.add_argument('--seeds', type=int, default=0,
                   help='>= 2: record a format-2 baseline with this many paired noise seeds per scenario (contracts + statistics); default: legacy single-seed format 1')
    _add_sut_args(s)
    s.set_defaults(fn=cmd_baseline)

    s = sub.add_parser('regress', help='compare against a baseline; exit 1 on regression')
    s.add_argument('--suite', required=True)
    s.add_argument('--baseline', required=True, help='explicit versioned baseline file (no default on purpose)')
    s.add_argument('--out')
    s.add_argument('--strict', action='store_true', help='scenarios missing from the baseline are violations')
    s.add_argument('--gap-tol', type=float, default=RegressionCriteria.gap_tol)
    s.add_argument('--impact-tol', type=float, default=RegressionCriteria.impact_tol)
    s.add_argument('--ttc-tol', type=float, default=RegressionCriteria.ttc_tol)
    s.add_argument('--reaction-tol', type=float, default=RegressionCriteria.reaction_tol)
    _add_sut_args(s)
    s.set_defaults(fn=cmd_regress)

    s = sub.add_parser('demo', help='deterministic end-to-end demonstration')
    s.add_argument('--budget', type=int, default=100)
    s.add_argument('--seed', type=int, default=42)
    s.add_argument('--no-gif', action='store_true')
    s.add_argument('--out')
    s.set_defaults(fn=cmd_demo)

    from .planner_eval import PLANNERS as _PL, SETS as _SETS
    s = sub.add_parser('fix-demo', help='one complete failure-to-fix cycle: discover, minimise, diagnose, fix, validate, protect')
    s.add_argument('--out', default=str(DEFAULT_OUT / 'fix_demo'))
    s.add_argument('--seed', type=int, default=42)
    s.add_argument('--budget', type=int, default=100)
    s.add_argument('--seeds', type=int, default=16, help='noise seeds for the paired before/after runs on the failing scenarios')
    s.add_argument('--validation', type=int, default=300, help='fresh random scenarios for the before/after validation')
    s.add_argument('--no-gif', action='store_true')
    s.set_defaults(fn=cmd_fix_demo)

    s = sub.add_parser('regression-eval', help='false-positive / false-negative evaluation of the regression detectors on planner mutations')
    s.add_argument('--split', choices=['dev', 'heldout'], default='dev')
    s.add_argument('--out', default=str(DEFAULT_OUT / 'regression_eval'))
    s.add_argument('--work', default=None, help='scratch directory for the generated suites')
    s.add_argument('--noise-trials', type=int, default=40)
    s.set_defaults(fn=cmd_regression_eval)

    s = sub.add_parser('diag-eval', help='evaluate the diagnostic attribution on planted faults and no-fault controls')
    s.add_argument('--split', choices=['dev', 'heldout', 'calibrate'], default='dev')
    s.add_argument('--out', default=str(DEFAULT_OUT / 'diagnosis_eval'))
    s.add_argument('--cache', default=None, help='cache file for the constructed cases (construction takes minutes)')
    s.set_defaults(fn=cmd_diag_eval)

    s = sub.add_parser('compare', help='before/after comparison of braking planners on identical scenario sets')
    s.add_argument('--planners', nargs='+', default=['baseline', 'improved'], choices=sorted(_PL),
                   help='first two are compared pairwise (paired statistics); others are reported as extra rows')
    s.add_argument('--split', choices=('dev', 'heldout', 'all'), default='dev',
                   help='dev: sets the planner design may use; heldout: frozen sets, run after the freeze')
    s.add_argument('--set', action='append', choices=sorted(_SETS), help='run only these sets (repeatable)')
    s.add_argument('--ttc-mode', choices=('kalman', 'analytic'), default=None)
    s.add_argument('--sensitivity', action='store_true', help='also run the post-hoc slower-actuator robustness check')
    s.add_argument('--sensitivity-set', default='heldout_random', choices=sorted(_SETS))
    s.add_argument('--lag', type=float, nargs='+', default=[0.3], help='actual actuator lag(s) for --sensitivity (s)')
    s.add_argument('--out')
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser('evaluate', help='evaluate the framework itself')
    s.add_argument('--budget', type=int, default=150, help='simulations per search trial')
    s.add_argument('--trials', type=int, default=10)
    s.add_argument('--quick', action='store_true', help='tiny configuration for smoke tests')
    s.add_argument('--planner', choices=['baseline', 'improved'], default='baseline',
                   help='braking planner of the system under test (default: the original planner, as in the committed results)')
    s.add_argument('--out')
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser('agent', help='LLM-assisted investigation (deterministic fallback without an API key)')
    s.add_argument('--scenario', required=True)
    s.add_argument('--out')
    s.add_argument('--max-tool-calls', type=int, default=12)
    s.add_argument('--max-sims', type=int, default=600)
    s.add_argument('--max-seconds', type=float, default=120.0)
    s.add_argument('--max-tokens', type=int, default=60000)
    s.add_argument('--no-llm', action='store_true', help='force the deterministic investigator even if a key is set')
    _add_sut_args(s)
    s.set_defaults(fn=cmd_agent)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except ScenarioValidationError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    except ValueError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f'error: file not found: {e.filename}', file=sys.stderr)
        return 2
