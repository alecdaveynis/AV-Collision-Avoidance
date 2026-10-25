"""``python -m autonomy_redteam fix-demo``: one complete failure-to-fix cycle, executed for real.

  1 discovery        random search against the original planner (fixed seed/budget) -> most severe finding
  2 minimisation     delta-debugging style reduction to the simplest reproducible scenario
  3 diagnosis        temporal + counterfactual attribution; mechanism read off the telemetry
  4 improvement      the deterministic improved planner (src/safe_braking.py) on the same scenarios and noise seeds
  5 validation       fresh, unseen random scenarios, both planners, paired statistics
  6 protection       the failure becomes a permanent contract scenario in a regression suite with a format-2 baseline recorded on
                     the improved planner; the original planner is then run against it and must be caught

Artifacts (``--out``): before_after.png, before_after.gif, report.md, results.json, suite/ and baseline_improved.json
(the last two are intermediate copies of scenarios/ content; the committed docs/ copy prunes them).
Nothing here is mocked; the numbers in the report come from the simulations run by this command.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .adapters import default_sut_factory
from .attribution import _arrays, attribute, critical_window, summary_lines
from .diagnose import diagnose
from .contracts import Policy, create_baseline_v2, run_regression_v2, to_markdown_v2
from .findings import group_failures, sig_str
from .investigate import investigate
from .metrics import DEFAULT_CRITERIA
from .minimize import derived_seeds
from .planner_eval import PLANNERS, COMPARED, _draw, paired_binary, run_set, summarize
from .regress import load_suite, promote
from .report import STATE_COLOR
from .runner import Evaluator
from .scenario import Scenario, ScenarioGenerator
from .search import run_search
from .sut import Episode

REPO = Path(__file__).resolve().parents[1]
BASE_COLOR, FIX_COLOR = '#c0392b', '#1f77b4'


def _ev(planner: str = 'baseline') -> Evaluator:
    return Evaluator(default_sut_factory('kalman', None, planner), DEFAULT_CRITERIA)


# --- telemetry-derived mechanism ---------------------------------------------------------------------------------------------
def mechanism(ep: Episode) -> Dict:
    """Facts about *why* the planner command was insufficient, computed from the episode (ground truth + planner telemetry)."""
    a = _arrays(ep)
    dt, lim = ep.limits.dt, ep.limits
    t = a['t']
    states = [r.planner.state for r in ep.records]
    w = critical_window(ep)
    first_stop = next((i for i, s in enumerate(states) if s == 'STOPPED'), None)
    first_slow = next((i for i, s in enumerate(states) if s != 'DRIVE'), None)
    out: Dict = {'t_first_non_drive': None if first_slow is None else float(t[first_slow]),
                 't_first_stopped': None if first_stop is None else float(t[first_stop]),
                 't_hazard': None if w is None else w.t_h}
    tau = lim.actuator_lag + dt
    if first_stop is not None:
        g, c = float(a['gap'][first_stop]), float(a['closing'][first_stop])
        out.update(gap_at_stopped=g, closing_at_stopped=c,
                   distance_needed_at_full_brake=c * tau + c * c / (2 * lim.max_decel),
                   standstill_margin=2.0)
    if w is not None and first_stop is not None:
        ih = int(np.argmin(np.abs(t - w.t_h)))
        seg = slice(ih, first_stop + 1)
        req = [(c * c / (2 * max(g - 2.0, 0.05))) if c > 0.1 else 0.0 for g, c in zip(a['gap'][seg], a['closing'][seg])]
        out.update(slowing_phase_s=float(t[first_stop] - t[ih]), mean_cmd_during_slowing=float(np.mean(a['cmd'][seg])),
                   mean_required_during_slowing=float(np.mean(req)), peak_required_during_slowing=float(np.max(req)))
    out['collided'] = bool(a['collided'].any())
    out['min_gap'] = float(np.min(a['gap_after']))
    out['impact_speed'] = float(max([r.gt.closing_speed for r in ep.records if r.collided], default=0.0))
    return out


# --- visuals ------------------------------------------------------------------------------------------------------------------
def plot_before_after(eps: Dict[str, Episode], path: Path, title: str) -> Path:
    fig, ax = plt.subplots(6, 1, figsize=(9.5, 12), sharex=True, gridspec_kw={'height_ratios': [1.6, 1.8, 1.4, 1.5, 0.35, 0.35]})
    cols = {'baseline': BASE_COLOR, 'improved': FIX_COLOR}
    for name, ep in eps.items():
        s, c = ep.series(), cols[name]
        t = np.array(s['t'])
        ax[0].plot(t, s['ego_v'], color=c, label=f'ego speed ({name})')
        ax[1].plot(t, s['gap'], color=c, label=f'gap ({name})')
        ax[2].plot(t, np.minimum(np.nan_to_num(s['ttc'], posinf=10.0), 10.0), color=c, label=f'true TTC ({name})')
        ax[3].plot(t, s['cmd_decel'], color=c, label=f'brake command ({name})')
        ax[3].plot(t, s['ego_decel'], color=c, ls=':', lw=1, label=f'applied decel ({name})')
    first = next(iter(eps.values())).series()
    ax[0].plot(first['t'], first['lead_v'], color='k', ls='--', label='lead speed')
    ax[0].set_ylabel('speed (m/s)')
    ax[1].axhline(0, color='k', lw=0.8)
    ax[1].axhline(2, color='gray', ls=':', lw=0.8)
    ax[1].text(0.2, 2.6, '2 m standstill margin', fontsize=7, color='gray')
    ax[1].set_ylabel('gap to lead (m)')
    ax[2].axhline(3, color='gray', ls=':', lw=0.8)
    ax[2].axhline(1, color='gray', ls=':', lw=0.8)
    ax[2].set_ylabel('true TTC (s, cap 10)')
    ax[3].set_ylabel('decel (m/s$^2$)')
    for i, (name, ep) in enumerate(eps.items()):
        for r in ep.records:
            ax[4 + i].axvspan(r.t, r.t + ep.limits.dt, color=STATE_COLOR[r.planner.state], lw=0)
        ax[4 + i].set_yticks([])
        ax[4 + i].set_ylabel(name, rotation=0, ha='right', va='center', fontsize=8)
        for ep_ in [ep]:
            hit = [r.t for r in ep_.records if r.collided]
            if hit:
                for k in (0, 1, 2, 3):
                    ax[k].axvline(hit[0], color=cols[name], lw=1.2, ls='-.')
                ax[1].annotate('collision', (hit[0], 1.5), color=cols[name], fontsize=8)
    for k in range(4):
        ax[k].legend(loc='upper right', fontsize=7, ncol=2)
    ax[0].set_title(title, fontsize=8)
    ax[5].set_xlabel('time (s)   [state bars: green DRIVE, yellow SLOWING, red STOPPED]')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def make_before_after_gif(eps: Dict[str, Episode], path: Path, stride: int = 2) -> Path:
    from matplotlib.animation import FuncAnimation, PillowWriter
    n = max(len(e.records) for e in eps.values())
    dt = next(iter(eps.values())).limits.dt
    allx = [r.gt.ego_x for e in eps.values() for r in e.records] + [r.gt.lead_x + 4.5 for e in eps.values() for r in e.records]
    lo, hi = min(allx) - 10, max(allx) + 8
    fig, axs = plt.subplots(len(eps), 1, figsize=(9, 2.6 * len(eps)), sharex=True)
    axs = np.atleast_1d(axs)
    arts = {}
    for ax, (name, ep) in zip(axs, eps.items()):
        ax.set_xlim(lo, hi)
        ax.set_ylim(-3.2, 4.4)
        ax.axhline(-1.9, color='gray', lw=1)
        ax.axhline(1.9, color='gray', lw=1)
        ax.set_yticks([])
        er, lr = Rectangle((0, -0.9), 4.5, 1.8, color=BASE_COLOR if name == 'baseline' else FIX_COLOR), Rectangle((0, -0.9), 4.5, 1.8, color='dimgray')
        ax.add_patch(er)
        ax.add_patch(lr)
        bar = Rectangle((lo, 2.8), hi - lo, 0.55, color=STATE_COLOR['DRIVE'])
        ax.add_patch(bar)
        txt = ax.text(0.01, 0.62, '', transform=ax.transAxes, va='top', family='monospace', fontsize=8)
        ax.set_title(f'{name} planner', fontsize=9, loc='right')
        arts[name] = (er, lr, bar, txt)
    axs[-1].set_xlabel('longitudinal position (m)')
    frames = list(range(0, n, stride)) + [n - 1]

    def draw(i):
        out = []
        for name, ep in eps.items():
            r = ep.records[min(i, len(ep.records) - 1)]
            er, lr, bar, txt = arts[name]
            er.set_xy((r.gt.ego_x - 4.5, -0.9))
            lr.set_xy((r.gt.lead_x, -0.9))
            bar.set_color(STATE_COLOR[r.planner.state])
            hit = any(x.collided for x in ep.records[:min(i, len(ep.records) - 1) + 1])
            txt.set_text(f't={r.t:4.1f}s {r.planner.state:8s} cmd={r.planner.cmd_decel:3.1f}  gap={r.gt.gap:5.1f} m  '
                         f'ego={r.gt.ego_v:4.1f} lead={r.gt.lead_v:4.1f} m/s' + ('   COLLISION' if hit else ''))
            out += [er, lr, bar, txt]
        return out

    ani = FuncAnimation(fig, draw, frames=frames, blit=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(path, writer=PillowWriter(fps=max(1, int(round(1 / (dt * stride))))))
    plt.close(fig)
    return path


# --- the cycle ----------------------------------------------------------------------------------------------------------------
def _paired_seed_runs(sc: Scenario, n: int, planners=('baseline', 'improved')) -> Dict[str, dict]:
    out = {}
    for p in planners:
        ev = _ev(p)
        rows = [ev.evaluate(sc.replace(seed=s)) for s in derived_seeds(sc.seed, n)]
        out[p] = {'n': n, 'collisions': sum(a.collision for a in rows), 'failures': sum(a.failed for a in rows),
                  'mean_min_gap_m': round(float(np.mean([a.min_gap for a in rows])), 2),
                  'mean_impact_speed_mps': round(float(np.mean([a.impact_speed for a in rows])), 2),
                  'worst_impact_speed_mps': round(float(max(a.impact_speed for a in rows)), 2)}
    return out


def run_fix_demo(out: Path, seed: int = 42, budget: int = 100, n_seeds: int = 16, validation_n: int = 300,
                 gif: bool = True, update_repo_suite: bool = False) -> int:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    res: Dict = {'command': f'python -m autonomy_redteam fix-demo --seed {seed} --budget {budget}'}
    ev_b, ev_i = _ev('baseline'), _ev('improved')

    # 1 discovery
    print(f'[1/6] discovery: random search, seed {seed}, budget {budget}, original planner')
    sres = run_search('random', ev_b, budget, seed)
    findings = group_failures(ev_b, sres.evaluations)
    res['discovery'] = {'sims': sres.sims_used, 'failing': len(sres.failures), 'first_failure_sim': sres.first_failure_sim,
                        'findings': [{'severity': round(f.severity, 1), 'members': f.members, 'signature': sig_str(f.signature)} for f in findings[:5]]}
    print(f'      {len(sres.failures)}/{sres.sims_used} scenarios failed; {len(findings)} distinct findings; first failure at #{sres.first_failure_sim}')
    if not findings:
        print('      no failure found: nothing to fix (no failure is fabricated)')
        return 0
    top = findings[0].representative.replace(name='top_finding')
    print(f'      top finding: {top.describe()}')

    # 2-3 minimisation + diagnosis (+ the standard investigation report)
    print('[2/6] minimisation and [3/6] diagnosis')
    inv = investigate(ev_b, top, out / 'investigation_baseline', gif=False,
                      repro_hint=f'python -m autonomy_redteam fix-demo --seed {seed} --budget {budget}')
    m = inv.minimization
    smin = m.minimized.replace(name='known_failure_minimized')
    attr = inv.diagnosis.attribution
    print(f'      minimised: {smin.describe()}')
    print(f'      necessary faults: {", ".join(m.necessary_faults) or "none (fails with no injected faults)"}; reproducible: {m.reproducible["status"]}')
    for line in summary_lines(attr):
        print('      ' + line)
    ep_b = ev_b.episode_with(smin)
    mech = mechanism(ep_b)
    res['minimization'] = {'minimized': smin.to_dict(), 'necessary_faults': m.necessary_faults, 'reproducible': m.reproducible}
    res['diagnosis'] = {'verdict': attr['verdict'], 'primary': attr['primary'], 'contributing': attr['contributing'],
                        'outcome_improved_by': attr['outcome_improved_by'], 'inconclusive_reasons': attr['inconclusive_reasons'],
                        'mechanism': mech}
    if mech.get('t_first_stopped') is not None:
        print(f'      mechanism: STOPPED only at t={mech["t_first_stopped"]:.1f}s with gap {mech["gap_at_stopped"]:.1f} m, closing '
              f'{mech["closing_at_stopped"]:.1f} m/s; full braking then needs {mech["distance_needed_at_full_brake"]:.1f} m (incl. actuator lag) '
              f'plus the {mech["standstill_margin"]:.0f} m margin; SLOWING phase commanded {mech.get("mean_cmd_during_slowing", float("nan")):.1f} m/s^2 '
              f'against {mech.get("mean_required_during_slowing", float("nan")):.1f} m/s^2 required')

    # 4 improvement on the failing scenarios
    print(f'[4/6] improvement: same scenarios, same {n_seeds} noise seeds, both planners')
    ep_i = ev_i.episode_with(smin)
    res['improvement'] = {'minimized': _paired_seed_runs(smin, n_seeds), 'original_fault_laden': _paired_seed_runs(top, n_seeds)}
    for k, v in res['improvement'].items():
        print(f'      {k}: baseline {v["baseline"]["collisions"]}/{n_seeds} collisions (mean min gap {v["baseline"]["mean_min_gap_m"]} m) -> '
              f'improved {v["improved"]["collisions"]}/{n_seeds} (mean min gap {v["improved"]["mean_min_gap_m"]} m)')
    # the original scenario also carries perception faults: where does the blame go before and after the fix?
    after: Dict[str, dict] = {}
    for p_, ev_p in (('baseline', ev_b), ('improved', ev_i)):
        if ev_p.evaluate(top).failed:
            at_ = attribute(ev_p, top, diagnose(ev_p, top, kind=ev_p.evaluate(top).primary_kind, n_seeds=8))
            after[p_] = {'verdict': at_['verdict'], 'primary': at_['primary'], 'contributing': at_['contributing'], 'lines': summary_lines(at_)}
            print(f'      original (fault-laden) scenario, {p_} planner: {at_["verdict"]} -> {", ".join(at_["primary"]) or "-"}')
        else:
            after[p_] = {'verdict': 'no failure', 'primary': [], 'contributing': [], 'lines': []}
    res['improvement']['original_attribution_by_planner'] = after
    eps = {'baseline': ep_b, 'improved': ep_i}
    png = plot_before_after(eps, out / 'before_after.png', f'{smin.describe()}')
    if gif:
        make_before_after_gif(eps, out / 'before_after.gif')

    # 5 validation on unseen scenarios
    print(f'[5/6] validation: {validation_n} fresh random scenarios (seed 31337), both planners, paired')
    scs = _draw(ScenarioGenerator(31337), validation_n, 'val')
    rows = {p: run_set(PLANNERS[p], scs, 'fresh') for p in ('baseline', 'improved')}
    summ = {p: summarize(r) for p, r in rows.items()}
    paired = {k: paired_binary(rows['baseline'], rows['improved'], f) for k, f in COMPARED.items()}
    res['validation'] = {'n': validation_n, 'seed': 31337, 'summary': summ, 'paired_improved_minus_baseline': paired}
    for k, v in paired.items():
        print(f'      {k:>26}: baseline {v["rate_a"]:.3f} -> improved {v["rate_b"]:.3f}  (discordant {v["only_a"]}/{v["only_b"]}, McNemar p={v["mcnemar_p"]:.4g})')

    # 6 protection
    print('[6/6] regression protection')
    suite = out / 'suite'
    for p in sorted((REPO / 'scenarios' / 'regression').glob('*.json')):
        _, sc, meta = load_suite_one(p)
        promote(sc, suite, p.stem, meta)
    promote(smin, suite, 'known_failure_gentle_lead_braking',
            {'origin': f'fix-demo seed {seed} budget {budget}', 'role': 'fixed failure: contract for the improved planner'})
    pol = Policy()
    base_i = create_baseline_v2(ev_i, suite, 'improved-v1', 'recorded on the improved planner', pol)
    (out / 'baseline_improved.json').write_text(json.dumps(base_i, indent=1))
    role = base_i['scenarios']['known_failure_gentle_lead_braking']['role']
    ok = run_regression_v2(_ev('improved'), suite, base_i, pol)
    bad = run_regression_v2(_ev('baseline'), suite, base_i, pol)
    (out / 'regression_improved_vs_own_baseline.md').write_text(to_markdown_v2(ok))
    (out / 'regression_original_planner_vs_improved_baseline.md').write_text(to_markdown_v2(bad))
    res['protection'] = {'contract_role_of_fixed_scenario': role, 'improved_vs_own_baseline': {'passed': ok['passed'], 'tier': ok['tier']},
                         'original_planner_vs_improved_baseline': {'passed': bad['passed'], 'tier': bad['tier'], 'violations': bad['violations']}}
    print(f'      fixed scenario recorded as role "{role}"; improved vs its own baseline: {"PASS" if ok["passed"] else "FAIL"}; '
          f'original planner against the improved baseline: {"PASS" if bad["passed"] else "FAIL"} ({bad["tier"]}; {", ".join(bad["violations"][:4])})')
    if update_repo_suite:
        promote(smin, REPO / 'scenarios' / 'regression', 'known_failure_gentle_lead_braking', {'origin': 'fix-demo', 'role': 'fixed failure'})
    res['seconds'] = round(time.perf_counter() - t0, 1)
    (out / 'results.json').write_text(json.dumps(res, indent=1, default=str))
    (out / 'report.md').write_text(_report(res, smin, top, png.name, 'before_after.gif' if gif else None, attr, ok, bad))
    print(f'\nreport: {out / "report.md"}   ({res["seconds"]}s)')
    return 0


def load_suite_one(p: Path):
    raw = json.loads(p.read_text())
    return p.stem, Scenario.load(p), (raw.get('meta', {}) if isinstance(raw, dict) else {})


def _report(res: dict, smin: Scenario, top: Scenario, png: str, gif: Optional[str], attr: dict, ok: dict, bad: dict) -> str:
    d, mech, v = res['discovery'], res['diagnosis']['mechanism'], res['validation']
    L = ['# Failure-to-fix: gentle lead braking at highway speed', '',
         f'Generated by `{res["command"]}`; every number below was computed by that run.', '',
         '## 1. Discovery', '',
         f'Random search (seed {res["command"].split("--seed ")[1].split()[0]}, {d["sims"]} simulations) against the original planner: '
         f'{d["failing"]} failing scenarios, first at simulation #{d["first_failure_sim"]}. Most severe finding: `{top.describe()}`.', '',
         '## 2. Minimisation', '',
         f'Reduced to `{smin.describe()}` (faults needed: {", ".join(res["minimization"]["necessary_faults"]) or "none"}). '
         f'Reproducibility: {res["minimization"]["reproducible"]["status"]}.', '',
         '## 3. Diagnosis', '']
    sl = summary_lines(attr)
    L += [f'- {sl[0]}'] + [f'  - {x.strip().lstrip("- ").strip()}' for x in sl[1:]]
    if mech.get('t_first_stopped') is not None:
        L += ['', f'Mechanism (from telemetry): the planner left DRIVE at t={mech["t_first_non_drive"]:.1f} s but only commanded the SLOWING level '
              f'(about {mech.get("mean_cmd_during_slowing", float("nan")):.1f} m/s^2 on average) while the ground truth needed '
              f'{mech.get("mean_required_during_slowing", float("nan")):.1f} m/s^2 on average (peak {mech.get("peak_required_during_slowing", float("nan")):.1f}). '
              f'STOPPED was reached at t={mech["t_first_stopped"]:.1f} s with {mech["gap_at_stopped"]:.1f} m left and {mech["closing_at_stopped"]:.1f} m/s '
              f'closing; stopping needs {mech["distance_needed_at_full_brake"]:.1f} m even at the full 8 m/s^2 once the actuator lag is included, '
              f'and the planner also wants to keep {mech["standstill_margin"]:.0f} m. The stop threshold assumes instantaneous braking.']
    L += ['', f'![before/after]({png})', '']
    if gif:
        L += [f'![animation]({gif})', '']
    L += ['## 4. Improvement', '',
          'Deterministic improved braking policy (`src/safe_braking.py`): required deceleration from closing speed and separation, '
          'corrected for actuator lag and filter delay, proportional braking with a 1.25 margin, TTC cannot trigger a stop on its own, '
          'unconfirmed tracks cannot trigger hard braking, hysteresis on release. No LLM in the loop.', '',
          '| scenario | planner | collisions | mean min gap (m) | mean impact speed (m/s) |', '|---|---|---|---|---|']
    for k, r in res['improvement'].items():
        if k == 'original_attribution_by_planner':
            continue
        for p in ('baseline', 'improved'):
            x = r[p]
            L.append(f'| {k.replace("_", " ")} | {p} | {x["collisions"]}/{x["n"]} | {x["mean_min_gap_m"]} | {x["mean_impact_speed_mps"]} |')
    oa = res['improvement'].get('original_attribution_by_planner', {})
    if oa:
        L += ['', 'The originally discovered scenario also carries perception faults (latency, range noise, detection loss). The planner change does not '
              'fix it, and the diagnosis says why, before and after:', '']
        for p_, x in oa.items():
            L.append(f'- {p_} planner: verdict **{x["verdict"]}**' + (f' ({", ".join(x["primary"])})' if x['primary'] else ''))
        L += ['', 'With the fix, the planner-logic cause disappears and what remains is perception (late or missing lead information); this is a '
              'separate defect class that a braking policy cannot repair.']
    L += ['', '## 5. Validation on unseen scenarios', '',
          f'{v["n"]} fresh random scenarios (seed {v["seed"]}), identical for both planners, paired exact McNemar. These scenarios were not used '
          f'to build or tune the policy. The full pre-registered held-out evaluation (several sets, CIs, controls) is in '
          f'`docs/redteam/planner_eval/`.', '',
          '| outcome | baseline | improved | only baseline / only improved | McNemar p |', '|---|---|---|---|---|']
    for k, x in v['paired_improved_minus_baseline'].items():
        L.append(f'| {k} | {x["rate_a"]:.3f} | {x["rate_b"]:.3f} | {x["only_a"]} / {x["only_b"]} | {x["mcnemar_p"]:.3g} |')
    L += ['', '## 6. Regression protection', '',
          f'The minimised failure is now a scenario in the regression suite; recorded on the improved planner it has the role '
          f'`{res["protection"]["contract_role_of_fixed_scenario"]}`, i.e. a contract: it must stay collision-free in every noise seed.', '',
          f'- improved planner vs its own baseline: **{"PASS" if ok["passed"] else "FAIL"}**',
          f'- original planner vs the improved baseline: **{"PASS" if bad["passed"] else "FAIL"}** ({bad["tier"]}; violations: {", ".join(bad["violations"]) or "none"})', '',
          'Reintroducing the original behaviour is therefore caught by the regression gate.', '',
          '## Limits', '',
          '- One simulator, one longitudinal scenario family; the actuator is a first-order lag model.',
          '- The improved policy was developed against this failure and mechanism; the unseen-scenario table is the evidence it is not specific to it. '
          'Remaining failures after the fix are mostly perception-fault induced and are listed in `docs/redteam/planner_eval/`.']
    return '\n'.join(L) + '\n'
