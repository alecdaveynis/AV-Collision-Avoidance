"""Plots, GIF and Markdown report for one investigated failure. All numbers come from executed simulations."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .diagnose import Diagnosis
from .metrics import Assessment
from .minimize import MinimizationResult
from .scenario import PARAMS, Scenario
from .sut import Episode

STATE_COLOR = {'DRIVE': '#6bbf59', 'SLOWING': '#f2c14e', 'STOPPED': '#d9534f'}


def _f(x, nd=2):
    return 'n/a' if x is None or (isinstance(x, float) and not math.isfinite(x)) else f'{x:.{nd}f}'


# --- plots ---------------------------------------------------------------------------------------------------
def plot_telemetry(ep: Episode, path: Path, title: str = '') -> Path:
    s = ep.series()
    t = np.array(s['t'])
    fig, ax = plt.subplots(5, 1, figsize=(9, 10.5), sharex=True,
                           gridspec_kw={'height_ratios': [2.2, 1.6, 1.6, 1.6, 0.5]})
    ax[0].plot(t, s['gap'], label='ground-truth gap', color='k')
    ax[0].plot(t, s['est_gap'], label='estimated gap (planner input)', color='tab:orange', ls='--')
    ax[0].set_ylabel('gap to lead (m)')
    ax[0].legend(loc='upper right')
    ax[0].set_title(title or ep.scenario.name or 'episode', fontsize=8)
    ax[1].plot(t, s['ego_v'], label='ego speed', color='tab:blue')
    ax[1].plot(t, s['lead_v'], label='lead speed', color='tab:red')
    ax[1].set_ylabel('speed (m/s)')
    ax[1].legend(loc='upper right')
    ax[2].plot(t, s['cmd_decel'], label='commanded', color='tab:purple')
    ax[2].plot(t, s['ego_decel'], label='applied (after lag)', color='tab:green')
    ax[2].set_ylabel('decel (m/s^2)')
    ax[2].legend(loc='upper right')
    cap = 10.0
    ax[3].plot(t, np.minimum(s['ttc'], cap), label='ground-truth TTC', color='k')
    ax[3].plot(t, np.minimum(np.nan_to_num(s['est_ttc'], nan=cap, posinf=cap), cap), label='estimated TTC',
               color='tab:orange', ls='--')
    ax[3].axhline(3.0, color='gray', ls=':', lw=0.8)
    ax[3].axhline(1.0, color='gray', ls=':', lw=0.8)
    ax[3].set_ylabel(f'TTC (s, cap {cap:.0f})')
    ax[3].legend(loc='upper right')
    for i, r in enumerate(ep.records):
        ax[4].axvspan(r.t, r.t + ep.limits.dt, color=STATE_COLOR[r.planner.state], lw=0)
    ax[4].set_yticks([])
    ax[4].set_ylabel('state')
    ax[4].set_xlabel('time (s)')
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in STATE_COLOR.values()]
    ax[4].legend(handles, STATE_COLOR.keys(), ncol=3, loc='upper center', bbox_to_anchor=(0.5, -1.0), fontsize=8)
    for e in ep.events:
        if e.kind in ('lead_brake_onset', 'collision', 'detection_loss_start', 'detection_loss_end'):
            for a in ax[:4]:
                a.axvline(e.t, color={'collision': 'r'}.get(e.kind, 'gray'), ls='-.', lw=0.8, alpha=0.7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_counterfactuals(d: Diagnosis, path: Path) -> Path:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 0.45 * len(d.variants) + 2.2),
                                 gridspec_kw={'width_ratios': [2.2, 1]})
    names = [v.name for v in d.variants][::-1]
    rates = [v.rate for v in d.variants][::-1]
    from .diagnose import wilson
    err = np.array([[v.rate - wilson(v.k, v.n)[0], wilson(v.k, v.n)[1] - v.rate] for v in d.variants][::-1]).T
    colors = ['tab:gray' if v.name == 'original' else ('tab:green' if v.rate < d.variants[0].rate else 'tab:red')
              for v in d.variants][::-1]
    a1.barh(names, rates, xerr=err, color=colors, ecolor='k', capsize=2)
    a1.set_xlim(0, 1.05)
    a1.set_xlabel(f'failure rate over {d.n_seeds} noise seeds (90% Wilson CI)')
    a1.set_title('counterfactual variants')
    comps = list(d.shapley)
    vals = [d.shapley[c]['value'] for c in comps]
    lo = [d.shapley[c]['value'] - d.shapley[c]['ci90'][0] for c in comps]
    hi = [d.shapley[c]['ci90'][1] - d.shapley[c]['value'] for c in comps]
    a2.barh(comps, vals, xerr=[lo, hi], color='tab:blue', ecolor='k', capsize=2)
    a2.axvline(0, color='k', lw=0.8)
    a2.set_title('Shapley share of failure prob.\n(90% bootstrap CI)', fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def make_gif(ep: Episode, path: Path, stride: int = 2) -> Path:
    """Top-down longitudinal view: ego (blue), lead (red), planner state colour, gap and speeds."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    recs = ep.records
    ego = np.array([r.gt.ego_x for r in recs])
    lead_rear = np.array([r.gt.lead_x for r in recs])
    lo = min(ego.min(), 0) - 8
    hi = max((lead_rear + 4.5).max(), (ego + 5).max()) + 8
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.set_xlim(lo, hi)
    ax.set_ylim(-3.2, 4.2)
    ax.axhline(-1.9, color='gray', lw=1)
    ax.axhline(1.9, color='gray', lw=1)
    ax.set_yticks([])
    ax.set_xlabel('longitudinal position (m)')
    ego_r = Rectangle((0, -0.9), 4.5, 1.8, color='tab:blue')
    lead_r = Rectangle((0, -0.9), 4.5, 1.8, color='tab:red')
    ax.add_patch(ego_r)
    ax.add_patch(lead_r)
    txt = ax.text(0.01, 0.95, '', transform=ax.transAxes, va='top', family='monospace', fontsize=8)
    bar = Rectangle((lo, 2.6), hi - lo, 0.5, color=STATE_COLOR['DRIVE'])
    ax.add_patch(bar)
    idx = list(range(0, len(recs), stride)) + [len(recs) - 1]

    def draw(i):
        r = recs[i]
        ego_r.set_xy((r.gt.ego_x - 4.5, -0.9))
        lead_r.set_xy((r.gt.lead_x, -0.9))
        bar.set_color(STATE_COLOR[r.planner.state])
        ev = ' COLLISION' if r.collided else ''
        txt.set_text(f't={r.t:4.1f}s  state={r.planner.state:8s} cmd={r.planner.cmd_decel:3.1f} m/s^2\n'
                     f'gap={r.gt.gap:5.1f} m  ego={r.gt.ego_v:4.1f} m/s  lead={r.gt.lead_v:4.1f} m/s{ev}')
        return ego_r, lead_r, bar, txt

    ani = FuncAnimation(fig, draw, frames=idx, blit=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(path, writer=PillowWriter(fps=max(1, int(round(1 / (ep.limits.dt * stride))))))
    plt.close(fig)
    return path


def write_telemetry_csv(ep: Episode, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'ego_v', 'lead_v', 'gap', 'est_gap', 'range_error', 'gt_ttc', 'est_ttc', 'state',
                    'raw_state', 'cmd_decel', 'applied_decel', 'lead_detected', 'tracked', 'collided'])
        for r in ep.records:
            w.writerow([f'{r.t:.1f}', f'{r.gt.ego_v:.3f}', f'{r.gt.lead_v:.3f}', f'{r.gt.gap:.3f}',
                        _f(r.perception.est_gap, 3), _f(r.perception.range_error, 3), _f(r.gt.ttc, 3),
                        _f(r.perception.est_ttc, 3), r.planner.state, r.planner.raw_state,
                        f'{r.planner.cmd_decel:.3f}', f'{r.gt.ego_decel:.3f}', int(r.perception.lead_detected),
                        int(r.perception.tracked), int(r.collided)])
    return path


# --- timeline & recommendations ----------------------------------------------------------------------------------------
def key_timeline(ep: Episode, a: Assessment) -> List[Dict]:
    rows = []
    for e in ep.events:
        rows.append({'t': e.t, 'event': e.kind, 'detail': e.detail})
    if a.hazard_onset is not None:
        r = next(r for r in ep.records if abs(r.t - a.hazard_onset) < 1e-9)
        rows.append({'t': a.hazard_onset, 'event': 'hazard_onset',
                     'detail': f'ground truth needs >= 3 m/s^2 to stay safe (gap {r.gt.gap:.1f} m, closing {r.gt.closing_speed:.1f} m/s, '
                               f'planner state {r.planner.state})'})
    rows.sort(key=lambda x: (x['t'], x['event']))
    return rows


def recommendations(d: Diagnosis, ep: Episode, a: Assessment) -> List[str]:
    out = []
    hyp = {h['component']: h for h in d.hypotheses}

    def on(c):
        return hyp.get(c, {}).get('status') in ('supported', 'weak')
    recs = ep.records
    if on('planner_logic'):
        first_stop = next((r for r in recs if r.planner.state == 'STOPPED'), None)
        first_slow = next((r for r in recs if r.planner.state != 'DRIVE'), None)
        peak_req = 0.0
        from .oracle import required_decel
        for r in recs:
            q = required_decel(r.gt.gap, r.gt.closing_speed)
            if math.isfinite(q):
                peak_req = max(peak_req, q)
        s = ('Planner logic: failure persists with ideal perception. '
             f'Peak ground-truth required deceleration was {_f(peak_req, 1)} m/s^2 while the SLOWING state commands '
             f'only {0.3 * ep.limits.max_decel:.1f} m/s^2')
        if first_stop is not None:
            s += (f'; STOPPED was first commanded at t={first_stop.t:.1f}s with a ground-truth gap of '
                  f'{first_stop.gt.gap:.1f} m at {first_stop.gt.ego_v:.1f} m/s '
                  f'(braking distance at full decel: {first_stop.gt.ego_v ** 2 / (2 * ep.limits.max_decel):.1f} m + '
                  f'{first_stop.gt.ego_v * ep.limits.actuator_lag:.1f} m travelled during the {ep.limits.actuator_lag:.1f} s actuator lag)')
        elif first_slow is not None:
            s += f'; the planner never reached STOPPED (first non-DRIVE at t={first_slow.t:.1f}s)'
        out.append(s + '. Candidate change: size the stop zone with the actuator lag and escalation hysteresis '
                       '(extra term closing * (lag + hysteresis time)), and scale SLOWING brake with required deceleration. '
                       'Not validated: re-run `regress` and `search` after any change.')
    if on('object_detection'):
        out.append('Object detection: failure disappears with ideal detection. Candidate change: keep the last confirmed '
                   'in-path track alive (coasting with its estimated velocity) across short detection gaps and treat an '
                   'abruptly vanishing close object as an obstacle for a few frames; consider redundancy from a second range source.')
    if on('distance_estimation'):
        out.append('Distance estimation: failure disappears with ground-truth ranges. Candidate change: inflate zone '
                   'thresholds by the range uncertainty (e.g. +2 sigma) or filter the range with the Kalman state before the zone test.')
    if on('ttc_estimation'):
        out.append('TTC estimation: failure disappears with ground-truth TTC. Candidate change: do not let an optimistic TTC '
                   'override the distance zones; use the minimum of independent TTC estimates (the LiDAR range-rate layer already does this).')
    if on('reaction_latency'):
        out.append('Perception latency: failure disappears without latency. Candidate change: extrapolate tracks to the '
                   'current time with ego motion before the planner uses them, and add v * latency to the stop-zone distance.')
    if not out:
        out.append('No component reached the evidence threshold; do not change the planner on this evidence. Increase '
                   'n_seeds (`diagnose --seeds 20`) or inspect the telemetry plot.')
    return out


# --- markdown ----------------------------------------------------------------------------------------------------------------
def _param_table(sc: Scenario, other: Optional[Scenario] = None) -> str:
    rows = ['| parameter | value | unit |' + (' minimized |' if other else ''), '|---|---|---|' + ('---|' if other else '')]
    for k, sp in PARAMS.items():
        v = getattr(sc, k)
        line = f'| `{k}` | {v:g} | {sp.unit} |'
        if other is not None:
            w = getattr(other, k)
            line += f' {w:g}{" **(changed)**" if abs(w - v) > 1e-9 else ""} |'
        rows.append(line)
    rows.append(f'| `seed` | {sc.seed} | |' + (f' {other.seed} |' if other else ''))
    return '\n'.join(rows)


def build_markdown(sc: Scenario, a_orig: Assessment, minimization: Optional[MinimizationResult],
                   a_min: Assessment, ep_min: Episode, diag: Diagnosis, files: Dict[str, str],
                   repro_cmd: str, llm_text: Optional[str] = None, sut_info: Optional[dict] = None) -> str:
    L = []
    kind = diag.kind
    title = {'collision': 'Collision', 'near_miss': 'Near miss', 'missed_hazard': 'Missed hazard',
             'unnecessary_braking': 'Unnecessary hard braking'}[kind]
    L.append(f'# Autonomy Red Team failure report: {title} ({sc.name or sc.fingerprint()})\n')
    L.append('_All numbers below come from executed simulations. Attribution is evidence, not proof._\n')
    L.append('## 1. Failure summary\n')
    L.append(f'- **Failure kind:** `{kind}` (all kinds exhibited: {", ".join(a_min.failure_kinds)})')
    L.append(f'- **System under test:** {json.dumps(sut_info or {})}')
    L.append(f'- **Outcome (minimized scenario):** min gap {_f(a_min.min_gap)} m, impact speed {_f(a_min.impact_speed)} m/s, '
             f'min ground-truth TTC {_f(a_min.min_ttc)} s, states visited {", ".join(a_min.states_visited)}')
    L.append(f'- **Collision class:** `{a_min.collision_class}` (physically unavoidable = even full braking from t=0 collides; '
             f'hard = only avoidable with instant reaction; avoidable = a perfect-information reference rule with the '
             f'production confirmation delay avoids it)')
    L.append(f'- **Brake reaction time:** {_f(a_min.brake_reaction_time)} s from hazard onset '
             f'(onset t={_f(a_min.hazard_onset)} s); missed hazards: {a_min.missed_hazards}')
    L.append(f'- **Perception symptoms:** lead lost for {a_min.lead_lost_steps} closing steps, max range error '
             f'{_f(a_min.max_range_error)} m (symptoms only; causality is tested in section 6)\n')
    L.append('## 2. Severity and reproducibility\n')
    L.append(f'- **Severity score:** {_f(a_min.severity, 1)} (collision: 100 + 10 x impact speed; near miss: 50+; missed hazard: 30+; '
             f'unnecessary braking: 10+)')
    if minimization is not None and minimization.reproducible:
        r = minimization.reproducible
        L.append(f'- **Reproducibility:** {r["status"]}. Deterministic replay identical: {r["deterministic_replay"]}. '
                 f'Failure rate over {r["n_seeds"]} fresh noise seeds: minimized {r["seed_failure_rate_minimized"]:.0%}, '
                 f'original {r["seed_failure_rate_original"]:.0%}.')
    L.append(f'- **Noise-seed robustness inside the diagnosis:** original conditions fail in '
             f'{diag.variants[0].k}/{diag.variants[0].n} seeds.\n')
    L.append('## 3. Original scenario conditions\n')
    L.append(f'`{sc.describe()}`\n')
    L.append(_param_table(sc, minimization.minimized if minimization else None) + '\n')
    L.append('## 4. Minimized failure conditions\n')
    if minimization is not None:
        m = minimization
        L.append(f'- Faults needed to reproduce: **{", ".join(m.necessary_faults) or "none (fails with no injected faults)"}**; '
                 f'removed as unnecessary: {", ".join(m.removed_faults) or "none"}.')
        L.append(f'- Replays used: {m.replays}. Distance to the "boring" scenario (weighted L1, see `minimize.py`): '
                 f'{m.distance_before:.2f} -> {m.distance_after:.2f}.')
        L.append('- The minimized values are *a* boundary point found by bisection along each axis, not a unique minimum.\n')
        if m.changes:
            L.append('| phase | parameter | from | to | note |\n|---|---|---|---|---|')
            for c in m.changes:
                L.append(f'| {c.phase} | `{c.param}` | {c.old} | {c.new} | {c.note} |')
            L.append('')
        L.append(f'Minimized scenario: `{m.minimized.describe()}`\n')
    L.append('## 5. Timeline of key events (minimized scenario)\n')
    L.append('| t (s) | event | detail |\n|---|---|---|')
    for r in key_timeline(ep_min, a_min):
        L.append(f'| {r["t"]:.1f} | {r["event"]} | {r["detail"]} |')
    L.append('')
    if files.get('telemetry_png'):
        L.append(f'![telemetry]({files["telemetry_png"]})\n')
    if files.get('gif'):
        L.append(f'![animation]({files["gif"]})\n')
    L.append('## 6. Evidence from controlled interventions\n')
    L.append(f'Each variant replays the minimized scenario under identical initial conditions and the same {diag.n_seeds} noise seeds '
             f'({diag.sims} simulations). "Fixed" components get ideal behaviour (see `interventions.py`).\n')
    L.append('| variant | failure rate | 90% CI | mean min gap (m) | collision rate |\n|---|---|---|---|---|')
    for v in diag.variants:
        d = v.to_dict()
        L.append(f'| {v.name} | {d["failure_rate"]:.2f} ({d["failures"]}/{d["n_seeds"]}) | {d["ci90"][0]:.2f}-{d["ci90"][1]:.2f} | '
                 f'{d["mean_min_gap"]:.2f} | {d["collision_rate"]:.2f} |')
    L.append('')
    if files.get('counterfactual_png'):
        L.append(f'![counterfactuals]({files["counterfactual_png"]})\n')
    L.append('Single interventions on the scenario\'s own seed (changed metrics):\n')
    L.append('| intervention | failure rate original -> with | paired seeds fixed / broken | changed metrics (seed 0) |\n|---|---|---|---|')
    for e in diag.evidence:
        ch = '; '.join(f'{k}: {v[0]} -> {v[1]}' for k, v in e['changed_metrics_seed0'].items()) or 'none'
        L.append(f'| `{e["intervention"]}` | {e["failure_rate_original"]:.2f} -> {e["failure_rate_with_intervention"]:.2f} | '
                 f'{e["paired_fixed_by_intervention"]} / {e["paired_broken_by_intervention"]} | {ch} |')
    L.append('')
    if diag.minimal_fixing_sets:
        L.append('Smallest component sets whose repair removes (>= 80 %) of the failures: ' +
                 '; '.join(f'{{{", ".join(f["components"])}}} ({f["from"]:.2f} -> {f["failure_rate"]:.2f})'
                           for f in diag.minimal_fixing_sets) + '\n')
    L.append('## 7. Suspected contributing components\n')
    if diag.attribution:
        from .attribution import render_markdown
        L.extend(render_markdown(diag.attribution))
        L.append('### Outcome-sensitivity view (Shapley share; answers "which repair helps", not "which component failed")\n')
    L.append('Status rule: *supported* if the lower 90 % bound of the Shapley share exceeds 0.10; *weak* if only the point '
             'estimate does; planner logic is *supported* if the failure persists with ideal perception (lower bound > 0.5) '
             'and the reference rule fails at least 0.25 less often.\n')
    for h in diag.ranked_hypotheses():
        L.append(f'- **{h["label"]}: {h["status"]}**')
        for line in h['evidence']:
            L.append(f'  - {line}')
    L.append('')
    for n in diag.notes:
        L.append(f'> {n}')
    L.append('')
    L.append('## 8. Recommended engineering changes\n')
    L.append('Candidate changes derived from the evidence above; none has been validated.\n')
    for r in recommendations(diag, ep_min, a_min):
        L.append(f'- {r}')
    L.append('')
    if llm_text:
        L.append('## LLM investigation notes\n')
        L.append('_Written by the optional LLM agent from tool outputs; metrics above are deterministic and authoritative._\n')
        L.append(llm_text + '\n')
    L.append('## 9. Reproduce and guard against regression\n')
    L.append('```bash')
    L.append(repro_cmd)
    L.append('```')
    return '\n'.join(L) + '\n'
