"""End-to-end investigation of one failing scenario: minimise -> diagnose -> artifacts -> report."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from . import report as rep
from .attribution import attribute
from .diagnose import Diagnosis, diagnose
from .metrics import Assessment
from .minimize import MinimizationResult, minimize
from .runner import Evaluator
from .scenario import Scenario


@dataclass
class Investigation:
    scenario: Scenario
    minimization: MinimizationResult
    diagnosis: Diagnosis
    assessment_original: Assessment
    assessment_minimized: Assessment
    files: Dict[str, str]
    out_dir: Path

    def summary(self) -> dict:
        m = self.minimization
        return {'kind': self.diagnosis.kind, 'scenario': self.scenario.name or self.scenario.fingerprint(),
                'necessary_faults': m.necessary_faults, 'reproducibility': m.reproducible.get('status'),
                'hypotheses': [{'component': h['component'], 'status': h['status'], 'share': h['share']}
                               for h in self.diagnosis.ranked_hypotheses()],
                'report': self.files.get('report')}


def investigate(ev: Evaluator, scenario: Scenario, out_dir: Path, minimize_seeds: int = 3, diag_seeds: int = 8,
                gif: bool = True, llm_text: Optional[str] = None, repro_hint: Optional[str] = None,
                reduce_physical: bool = True) -> Investigation:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    a0 = ev.evaluate(scenario)
    if not a0.failed:
        raise ValueError('scenario does not fail under the current criteria; nothing to investigate')
    mres = minimize(ev, scenario, kind=a0.primary_kind, seeds=minimize_seeds, reduce_physical=reduce_physical)
    smin = mres.minimized
    a_min = ev.evaluate(smin)
    ep_min = ev.episode_with(smin)
    diag = diagnose(ev, smin, kind=mres.kind, n_seeds=diag_seeds)
    diag.attribution = attribute(ev, smin, diag)

    files: Dict[str, str] = {}
    scenario.save(out_dir / 'original_scenario.json')
    smin.save(out_dir / 'minimized_scenario.json')
    (out_dir / 'minimization.json').write_text(json.dumps(mres.to_dict(), indent=2))
    (out_dir / 'diagnosis.json').write_text(json.dumps(diag.to_dict(), indent=2))
    rep.write_telemetry_csv(ep_min, out_dir / 'telemetry.csv')
    (out_dir / 'telemetry.json').write_text(json.dumps(ep_min.to_dict()))
    rep.plot_telemetry(ep_min, out_dir / 'telemetry.png', title=f'{smin.name}\n{smin.describe()}')
    rep.plot_counterfactuals(diag, out_dir / 'counterfactuals.png')
    files.update(telemetry_png='telemetry.png', counterfactual_png='counterfactuals.png')
    if gif:
        rep.make_gif(ep_min, out_dir / 'animation.gif')
        files['gif'] = 'animation.gif'
    repro = repro_hint or (f'python -m autonomy_redteam minimize --scenario {out_dir / "original_scenario.json"}\n'
                           f'python -m autonomy_redteam diagnose --scenario {out_dir / "minimized_scenario.json"}\n'
                           f'# keep it as a regression test:\n'
                           f'python -m autonomy_redteam promote --scenario {out_dir / "minimized_scenario.json"} '
                           f'--suite scenarios/regression/')
    md = rep.build_markdown(scenario, a0, mres, a_min, ep_min, diag, files, repro, llm_text, ev.sut_info)
    (out_dir / 'report.md').write_text(md)
    files['report'] = str(out_dir / 'report.md')
    return Investigation(scenario, mres, diag, a0, a_min, files, out_dir)
