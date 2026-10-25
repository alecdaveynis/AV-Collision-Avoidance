"""Regression testing of safety performance across software versions.

A *suite* is a directory of scenario JSON files (promoted minimised failures, plus passing scenarios that sit close
to failure). A *baseline* is an explicitly selected, versioned JSON file produced by ``baseline`` from some earlier
version of the software; ``regress`` re-runs the suite and compares.

Findings (per scenario)
  new_failure      baseline had no failure, now fails (or fails with an additional kind)      -> violation
  worsened         a safety metric degraded beyond its tolerance                              -> violation
  stale_baseline   scenario file differs from what the baseline was recorded on               -> violation
  missing_baseline scenario not in the baseline                                               -> violation only with --strict
  fixed            baseline failed, now passes                                                -> reported (update the baseline)
  unchanged / improved

Tolerances (``RegressionCriteria``) absorb the small numeric differences between machines; all runs are deterministic
on one machine. Exit code: 0 clean, 1 violations, 2 usage / data error.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import __version__
from .metrics import Assessment
from .runner import Evaluator
from .scenario import Scenario, ScenarioValidationError

BASELINE_FORMAT = 1
TRACKED = ('collision', 'impact_speed', 'min_gap', 'min_ttc', 'brake_reaction_time', 'missed_hazards',
           'unnecessary_braking_s', 'max_state', 'peak_cmd_decel', 'first_brake_t', 'collision_class')


@dataclass(frozen=True)
class RegressionCriteria:
    gap_tol: float = 0.5               # m, decrease of min_gap that counts as worse
    impact_tol: float = 1.0            # m/s, increase of impact speed (~10 % of typical impact speeds)
    ttc_tol: float = 0.3               # s, decrease of min ground-truth TTC
    reaction_tol: float = 0.2          # s, increase of brake reaction time
    unnecessary_tol: float = 0.5       # s, increase of unnecessary hard braking
    fail_on: Tuple[str, ...] = ('new_failure', 'worsened', 'stale_baseline')
    strict: bool = False               # also fail on scenarios missing from the baseline


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, timeout=5,
                             cwd=Path(__file__).resolve().parents[1])
        return out.stdout.strip() or None
    except Exception:
        return None


def load_suite(suite_dir: Path) -> List[Tuple[str, Scenario, dict]]:
    suite_dir = Path(suite_dir)
    if not suite_dir.is_dir():
        raise ScenarioValidationError(f'suite directory not found: {suite_dir}')
    out = []
    for p in sorted(suite_dir.glob('*.json')):
        raw = json.loads(p.read_text())
        meta = raw.get('meta', {}) if isinstance(raw, dict) else {}
        sc = Scenario.load(p)
        out.append((p.stem, sc, meta))
    if not out:
        raise ScenarioValidationError(f'no *.json scenarios in {suite_dir}')
    return out


def _record(a: Assessment) -> dict:
    d = a.to_dict()
    rec = {k: d[k] for k in TRACKED}
    rec['failure_kinds'] = d['failure_kinds']
    return rec


def create_baseline(ev: Evaluator, suite_dir: Path, version: str, note: str = '') -> dict:
    scen: Dict[str, dict] = {}
    for name, sc, _ in load_suite(suite_dir):
        scen[name] = {'fingerprint': sc.fingerprint(), 'result': _record(ev.evaluate(sc))}
    return {'baseline_format': BASELINE_FORMAT, 'baseline_version': version, 'note': note,
            'created_utc': _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds'),
            'git_commit': _git_commit(), 'framework_version': __version__, 'sut': ev.sut_info,
            'criteria': asdict(ev.criteria), 'scenarios': scen}


def _cmp_worse(name, cur, base, crit: RegressionCriteria) -> List[str]:
    out = []
    if cur['min_gap'] < base['min_gap'] - crit.gap_tol:
        out.append(f'min_gap {base["min_gap"]:.2f} -> {cur["min_gap"]:.2f} m')
    if cur['impact_speed'] > base['impact_speed'] + crit.impact_tol:
        out.append(f'impact_speed {base["impact_speed"]:.2f} -> {cur["impact_speed"]:.2f} m/s')
    bt, ct = base['min_ttc'], cur['min_ttc']
    if ct is not None and (bt is None or ct < bt - crit.ttc_tol):
        out.append(f'min_ttc {"inf" if bt is None else f"{bt:.2f}"} -> {ct:.2f} s')
    br, cr = base['brake_reaction_time'], cur['brake_reaction_time']
    if cr is not None and br is not None and cr > br + crit.reaction_tol:
        out.append(f'brake_reaction_time {br:.2f} -> {cr:.2f} s')
    if cr is None and br is not None and cur['missed_hazards'] > base['missed_hazards']:
        out.append('hazard no longer answered by the planner')
    if cur['missed_hazards'] > base['missed_hazards']:
        out.append(f'missed_hazards {base["missed_hazards"]} -> {cur["missed_hazards"]}')
    if cur['unnecessary_braking_s'] > base['unnecessary_braking_s'] + crit.unnecessary_tol:
        out.append(f'unnecessary_braking_s {base["unnecessary_braking_s"]:.2f} -> {cur["unnecessary_braking_s"]:.2f}')
    if cur['collision'] and not base['collision']:
        out.append('now collides')
    return out


def run_regression(ev: Evaluator, suite_dir: Path, baseline: dict, crit: RegressionCriteria = RegressionCriteria()) -> dict:
    if baseline.get('baseline_format') != BASELINE_FORMAT:
        raise ScenarioValidationError(f'unsupported baseline format {baseline.get("baseline_format")!r}')
    items, rows = load_suite(suite_dir), []
    bs = baseline['scenarios']
    for name, sc, meta in items:
        a = ev.evaluate(sc)
        cur = _record(a)
        row = {'scenario': name, 'status': 'unchanged', 'details': [], 'current': cur,
               'baseline': bs.get(name, {}).get('result'), 'failure_kinds': cur['failure_kinds']}
        b = bs.get(name)
        if b is None:
            row['status'] = 'missing_baseline'
            row['details'].append('scenario is not in the baseline')
        elif b['fingerprint'] != sc.fingerprint():
            row['status'] = 'stale_baseline'
            row['details'].append('scenario parameters differ from the baseline recording: regenerate the baseline')
        else:
            base = b['result']
            new_kinds = [k for k in cur['failure_kinds'] if k not in base['failure_kinds']]
            lost = [k for k in base['failure_kinds'] if k not in cur['failure_kinds']]
            if new_kinds:
                row['status'] = 'new_failure'
                row['details'].append('new failure kind(s): ' + ', '.join(new_kinds))
            worse = _cmp_worse(name, cur, base, crit)
            if worse and row['status'] == 'unchanged':
                row['status'] = 'worsened'
            row['details'].extend(worse)
            if row['status'] == 'unchanged' and lost and not cur['failure_kinds']:
                row['status'] = 'fixed'
                row['details'].append('failure(s) gone: ' + ', '.join(lost))
            elif row['status'] == 'unchanged' and (cur['min_gap'] > base['min_gap'] + crit.gap_tol):
                row['status'] = 'improved'
                row['details'].append(f'min_gap {base["min_gap"]:.2f} -> {cur["min_gap"]:.2f} m')
        rows.append(row)
    names = {r['scenario'] for r in rows}
    for n in bs:
        if n not in names:
            rows.append({'scenario': n, 'status': 'removed_from_suite', 'details': ['in baseline, not in suite'],
                         'current': None, 'baseline': bs[n]['result'], 'failure_kinds': []})
    fail_on = set(crit.fail_on) | ({'missing_baseline'} if crit.strict else set())
    violations = [r for r in rows if r['status'] in fail_on]
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r['status']] = counts.get(r['status'], 0) + 1
    return {'baseline_version': baseline.get('baseline_version'), 'baseline_commit': baseline.get('git_commit'),
            'baseline_sut': baseline.get('sut'), 'current_sut': ev.sut_info, 'suite': str(suite_dir),
            'criteria': asdict(crit), 'counts': counts, 'n_scenarios': len(items),
            'violations': [r['scenario'] for r in violations], 'passed': not violations, 'rows': rows,
            'simulations': ev.n_sims}


def to_markdown(res: dict) -> str:
    L = [f'# Safety regression report: {"PASS" if res["passed"] else "FAIL"}\n',
         f'- Baseline: `{res["baseline_version"]}` (commit {res["baseline_commit"]}), suite `{res["suite"]}`, '
         f'{res["n_scenarios"]} scenarios, {res["simulations"]} simulations',
         f'- Baseline SUT: `{json.dumps(res["baseline_sut"])}`',
         f'- Current SUT:  `{json.dumps(res["current_sut"])}`',
         '- Counts: ' + ', '.join(f'{k}: {v}' for k, v in sorted(res['counts'].items())), '']
    if res['violations']:
        L.append(f'**Violations ({len(res["violations"])}):** ' + ', '.join(f'`{v}`' for v in res['violations']) + '\n')
    L.append('| scenario | status | failure kinds | min gap (m) | impact (m/s) | details |\n|---|---|---|---|---|---|')
    order = {'new_failure': 0, 'worsened': 1, 'stale_baseline': 2, 'missing_baseline': 3, 'removed_from_suite': 4,
             'fixed': 5, 'improved': 6, 'unchanged': 7}
    for r in sorted(res['rows'], key=lambda r: (order.get(r['status'], 9), r['scenario'])):
        cur = r['current']
        L.append(f'| `{r["scenario"]}` | {r["status"]} | {", ".join(r["failure_kinds"]) or "-"} | '
                 f'{cur["min_gap"]:.2f} | {cur["impact_speed"]:.2f} | {"; ".join(r["details"])} |'
                 if cur else f'| `{r["scenario"]}` | {r["status"]} | - | - | - | {"; ".join(r["details"])} |')
    return '\n'.join(L) + '\n'


def promote(scenario: Scenario, suite_dir: Path, name: Optional[str] = None, meta: Optional[dict] = None) -> Path:
    """Save a scenario into a suite (as a regression test)."""
    suite_dir = Path(suite_dir)
    suite_dir.mkdir(parents=True, exist_ok=True)
    stem = name or scenario.name or scenario.fingerprint()
    stem = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in stem)
    path = suite_dir / f'{stem}.json'
    path.write_text(json.dumps({'scenario': scenario.to_dict(), 'meta': meta or {}}, indent=2) + '\n')
    return path
