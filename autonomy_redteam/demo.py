"""``python -m autonomy_redteam demo``: search -> pick top finding -> minimise -> diagnose -> report -> regression gate.

Everything is executed for real against the repository's planner; if the campaign finds no failure it says so.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .adapters import default_sut_factory
from .findings import group_failures, sig_str
from .investigate import investigate
from .metrics import DEFAULT_CRITERIA
from .regress import RegressionCriteria, create_baseline, promote, run_regression, to_markdown
from .runner import Evaluator
from .search import run_search


def run_demo(out: Path, budget: int = 100, seed: int = 42, gif: bool = True) -> int:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ev = Evaluator(default_sut_factory('kalman'), DEFAULT_CRITERIA)
    print(f'[1/5] random search: budget {budget} simulations, seed {seed}, SUT {ev.sut_info}')
    res = run_search('random', ev, budget, seed)
    findings = group_failures(ev, res.evaluations)
    (out / 'search_results.json').write_text(json.dumps(res.to_dict(), indent=2))
    print(f'      {len(res.failures)}/{res.sims_used} scenarios failed; first failure at simulation #{res.first_failure_sim}; '
          f'{len(findings)} distinct findings')
    if not findings:
        print('      No failure found within this budget. Nothing to minimise or diagnose (no failure is fabricated).')
        return 0
    for i, f in enumerate(findings[:5], 1):
        print(f'      #{i} severity {f.severity:6.1f} x{f.members:<3d} {sig_str(f.signature)}')
    top = findings[0]
    sc = top.representative.replace(name='demo_top_finding')
    print(f'[2/5] investigating the top finding: {sc.describe()}')
    inv = investigate(ev, sc, out / 'investigation', gif=gif,
                      repro_hint=f'python -m autonomy_redteam demo --seed {seed} --budget {budget}\n'
                                 f'python -m autonomy_redteam minimize --scenario {out / "investigation" / "original_scenario.json"}\n'
                                 f'python -m autonomy_redteam regress --suite {out / "regression_suite"} '
                                 f'--baseline {out / "baseline_v1.json"}')
    m = inv.minimization
    print(f'[3/5] minimised: {m.minimized.describe()}')
    print(f'      necessary faults: {", ".join(m.necessary_faults) or "none"}; reproducibility: {m.reproducible["status"]}')
    from .attribution import summary_lines
    for line in summary_lines(inv.diagnosis.attribution):
        print('      ' + line)
    for h in inv.diagnosis.ranked_hypotheses():
        print(f'      shapley view: {h["label"]}: {h["status"]} (share {h["share"]:+.2f})')
    suite = out / 'regression_suite'
    promote(m.minimized.replace(name='demo_minimized_failure'), suite, 'demo_minimized_failure',
            meta={'origin': 'demo', 'role': 'known failure (tripwire)'})
    base = create_baseline(ev, suite, 'v1-demo', 'baseline of the unmodified planner')
    (out / 'baseline_v1.json').write_text(json.dumps(base, indent=2))
    print(f'[4/5] promoted the minimised failure to {suite} and recorded baseline v1-demo')
    ev2 = Evaluator(default_sut_factory('kalman'), DEFAULT_CRITERIA)
    reg = run_regression(ev2, suite, base, RegressionCriteria())
    (out / 'regression_summary.md').write_text(to_markdown(reg))
    print(f'[5/5] regression check against the baseline: {"PASS" if reg["passed"] else "FAIL"} ({reg["counts"]})')
    print(f'\nreport: {inv.files["report"]}\nartifacts: {out}   ({time.perf_counter() - t0:.1f}s, {ev.n_sims} + {ev2.n_sims} simulations)')
    return 0
