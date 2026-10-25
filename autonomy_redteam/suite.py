"""Build a regression suite from a search: minimised distinct failures + passing scenarios closest to failure."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .findings import Finding, group_failures, lead_regime, sig_str
from .minimize import minimize
from .regress import promote
from .runner import Evaluator
from .search import SearchResult


def build_suite(ev: Evaluator, result: SearchResult, out_dir: Path, n_fail: int = 6, n_pass: int = 6,
                findings: Optional[List[Finding]] = None, minimize_seeds: int = 3) -> Dict:
    out_dir = Path(out_dir)
    findings = findings if findings is not None else group_failures(ev, result.evaluations)
    written: List[str] = []
    seen_kinds: Dict[str, int] = {}
    for i, f in enumerate(findings[:n_fail], 1):
        try:
            m = minimize(ev, f.representative, kind=f.kind, seeds=minimize_seeds, check_reproducibility=False)
        except ValueError:
            continue
        name = f'fail_{i:02d}_{f.kind}_{lead_regime(m.minimized)}'
        p = promote(m.minimized.replace(name=name), out_dir, name,
                    meta={'origin': 'search', 'method': result.method, 'search_seed': result.seed,
                          'signature': sig_str(f.signature), 'role': 'known failure (tripwire)'})
        written.append(p.name)
    # passing scenarios that are nearest to failing: smallest ground-truth gap without a collision
    cands = [e for e in result.evaluations if not e.assessment.failed and not e.assessment.collision]
    cands.sort(key=lambda e: e.assessment.min_gap)
    seen_regimes: Dict[str, int] = {}
    k = 0
    for e in cands:
        reg = lead_regime(e.scenario)
        if seen_regimes.get(reg, 0) >= max(1, n_pass // 3):
            continue
        k += 1
        seen_regimes[reg] = seen_regimes.get(reg, 0) + 1
        name = f'pass_{k:02d}_{reg}'
        p = promote(e.scenario.replace(name=name), out_dir, name,
                    meta={'origin': 'search', 'method': result.method, 'search_seed': result.seed,
                          'role': 'passing, near the failure boundary', 'min_gap_at_creation': round(e.assessment.min_gap, 3)})
        written.append(p.name)
        if k >= n_pass:
            break
    return {'suite': str(out_dir), 'files': written, 'n_failures': sum(1 for w in written if w.startswith('fail')),
            'n_passing': sum(1 for w in written if w.startswith('pass'))}
