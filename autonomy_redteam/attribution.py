"""Evidence-based attribution: separates *contribution*, *outcome improvement* and *primary root cause*.

The Shapley view in ``diagnose.py`` measures how much the failure probability moves when a component is replaced by
its ideal. That answers "which intervention improves the outcome", which is not the same as "which component
failed first". In a scenario the planner survives only narrowly, fixing almost anything helps (including the 10 %
nominal dropout every episode has), so outcome sensitivity alone ranks innocent components highly. This module adds
what was missing and keeps the three questions separate:

1. **Contributed** - the component was *observably wrong* in this failing episode, *before* the unsafe decision, and a
   controlled replacement with ground truth changes the outcome. Both are required.
2. **Intervention improved the outcome** - fixing the component alone removes most failures (``fixes_outcome``). Reported
   on its own; it does not make the component the cause.
3. **Primary root cause** - a contributor whose counterfactual effect is large and not matched by another contributor.
   When two contributors have comparable effect the verdict is *joint* (ambiguous), and when no component meets the
   bar the verdict is *inconclusive* with reasons, never a forced guess.

Observed errors come from telemetry compared with ground truth. No knowledge of which fault was injected is read: this
module never touches the scenario's fault parameters (enforced by a test). Controlled replacement is a scenario edit that
is a no-op for a component that was never degraded, so it needs no label either.

    object_detection   the lead is not tracked for >= 3 consecutive steps, or >= 25 % of steps have no lead return
    reaction_latency   the perceived range matches the *earlier* true range better than the current one (lag >= 0.2 s)
    distance_estimation   zero-mean range residual (after removing that lag) with RMS above the nominal-sensor bound, or jitter of the raw LiDAR return above its nominal bound
    ttc_estimation     median |log(est TTC / true TTC)| above the nominal-estimator bound   (kalman mode only)

Thresholds come from *nominal* (fault-free) runs only (``diagnosis_eval.calibrate_nominal``, results in
``docs/redteam/diagnosis_eval/nominal_calibration.json``): discrete ones sit just above the nominal maximum, continuous ones are about
1.5 x the nominal p99 (rounded). No failing or planted case sets a threshold; the dev set was inspected while building the
detectors (documented in the README), the held-out set was run once after the thresholds were frozen.

Temporal ordering. ``t_h`` = first step where the ground truth needs >= 3 m/s^2 to stay safe. ``t_pnr`` (point of no return) = first
step where even maximum braking, delayed by the actuator lag and one control step, cannot stop in the remaining gap. An error is a
candidate cause only if it begins before ``t_pnr`` (plus one perception frame) and inside the window that opens one second before
``t_h``; it could not have caused a collision it started after. ``t_u`` (first demonstrably insufficient command) is reported for the
narrative. For unwarranted braking the deadline is the first unnecessary hard-braking step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .diagnose import Diagnosis, wilson
from .metrics import DEFAULT_CRITERIA
from .oracle import required_decel
from .sut import Episode

# --- thresholds (set from nominal runs, see calibrate_nominal) ---------------------------------------------------------------
LOST_RUN_STEPS = 3                 # consecutive steps without a confirmed in-path track
MISS_FRACTION = 0.25               # share of window steps without any lead return (nominal sensor: 0.10)
LAG_MIN_S = 0.2
LAG_GAIN = 0.7                     # residual at the fitted lag must be < 0.7 x residual at zero lag
RAW_JITTER_M = 0.3                 # m, raw-return noise (1.5 x nominal p99 of 0.19; see nominal_calibration.json)
DIST_RMS_M = 0.6                   # m, zero-mean range residual (1.5 x nominal p99 of 0.40)
TTC_LOG_RATIO = math.log(1.8)      # 0.59 (1.5 x nominal p99 of 0.40)
WINDOW_PRE_S = 1.0
FIX_FRACTION = 0.5                 # fixing the component alone must remove at least half of the original failures
ALONE_MIN_RATE = 0.5               # component alone (everything else ideal) must fail at least half of the seeds
SHAPLEY_FLOOR = 0.10
DOMINANCE = 0.5                    # a second contributor with effect >= 0.5 x the first makes the verdict joint
MAX_LAG_STEPS = 8

PERCEPTION = ('object_detection', 'reaction_latency', 'distance_estimation', 'ttc_estimation')
LABELS = {'object_detection': 'Object detection', 'reaction_latency': 'Reaction latency (perception)',
          'distance_estimation': 'Distance estimation', 'ttc_estimation': 'TTC estimation',
          'planner_logic': 'Planner decision logic / thresholds'}


@dataclass
class Window:
    t_h: Optional[float]
    t_u: Optional[float]
    t_end: float
    start: int
    stop: int
    note: str
    t_deadline: float = 0.0          # an error must start no later than this to be a candidate cause
    closing_only: bool = True        # restrict observations to closing steps (False for unwarranted braking)


def _arrays(ep: Episode) -> Dict[str, np.ndarray]:
    r = ep.records
    return {
        't': np.array([x.t for x in r]), 'gap': np.array([x.gt.gap for x in r]),
        'closing': np.array([x.gt.closing_speed for x in r]), 'ttc': np.array([x.gt.ttc for x in r]),
        'cmd': np.array([x.planner.cmd_decel for x in r]),
        'tracked': np.array([x.perception.tracked for x in r]), 'detected': np.array([x.perception.lead_detected for x in r]),
        'est_gap': np.array([x.perception.est_gap for x in r], float),
        'est_ttc': np.array([x.perception.est_ttc for x in r], float),
        'raw': np.array([x.perception.raw_range for x in r], float),
        'collided': np.array([x.collided for x in r]), 'gap_after': np.array([x.gap_after for x in r], float),
    }


def critical_window(ep: Episode, kind: str = 'collision') -> Optional[Window]:
    """Time window in which perception errors can matter, from ground truth and the planner command only.

    Hazard failures: ``t_h`` opens the window (one second earlier). An error is a candidate cause only if it starts before the
    *point of no return* ``t_pnr``: the first step from which even instantaneous-onset maximum braking, delayed by the actuator
    lag and one control step, cannot stop in the remaining gap. An error that begins after that could not have caused the
    collision. (Without a point of no return, e.g. a near miss, the deadline is the minimum-gap time.) ``t_u``, the first
    demonstrably insufficient command, is reported for the narrative but is not the deadline: the baseline planner legitimately
    needs a few frames after ``t_h`` to escalate, which is planner behaviour, not a perception error.

    Unwarranted braking: the window ends at the first unnecessary hard-braking run and the deadline is its first step.
    """
    c, dt = DEFAULT_CRITERIA, ep.limits.dt
    a = _arrays(ep)
    n = len(a['t'])
    a_req = np.array([required_decel(g, v, c.hazard_margin) for g, v in zip(a['gap'], a['closing'])])
    if kind == 'unnecessary_braking':
        for s0, e0 in _runs(a['cmd'] > c.unnecessary_cmd):
            if a['closing'][s0] <= 0.1 or a_req[s0] < c.unnecessary_req:
                start = max(s0 - int(round(WINDOW_PRE_S / dt)), 0)
                return Window(float(a['t'][s0]), float(a['t'][s0]), float(a['t'][min(e0, n - 1)]), start, e0, 'first unwarranted hard-braking run',
                              t_deadline=float(a['t'][s0]), closing_only=False)
        return None
    hz = np.nonzero(a_req >= c.hazard_decel)[0]
    if not len(hz):
        return None
    ih = int(hz[0])
    t_end_i = int(np.argmax(a['collided'])) if a['collided'].any() else int(np.argmin(a['gap_after']))
    iu = None
    for i in range(max(ih - int(round(0.5 / dt)), 0), n):
        if a_req[i] >= 1.5 and a['cmd'][i] < 0.8 * min(a_req[i], ep.limits.max_decel):
            iu = i
            break
    note = 'first insufficient command' if iu is not None else 'no insufficient command found; using the end of the episode'
    if iu is None:
        iu = t_end_i
    tau = ep.limits.actuator_lag + dt
    cl = np.maximum(a['closing'], 0.0)
    need = np.where(cl > 0.1, cl ** 2 / (2.0 * np.maximum(a['gap'] - cl * tau, 1e-3)), 0.0)
    pnr = np.nonzero(need >= ep.limits.max_decel)[0]
    ip = int(pnr[0]) if len(pnr) else t_end_i
    ip = min(ip, t_end_i)
    start = max(ih - int(round(WINDOW_PRE_S / dt)), 0)
    return Window(float(a['t'][ih]), float(a['t'][iu]), float(a['t'][t_end_i]), start, max(t_end_i + 1, start + 1), note,
                  t_deadline=float(a['t'][ip]))


# --- observations ------------------------------------------------------------------------------------------------------------
def _runs(mask: np.ndarray):
    out, s = [], None
    for i, m in enumerate(mask):
        if m and s is None:
            s = i
        if not m and s is not None:
            out.append((s, i))
            s = None
    if s is not None:
        out.append((s, len(mask)))
    return out


def raw_jitter(raw: np.ndarray, idx: np.ndarray) -> Optional[float]:
    """High-frequency noise of the raw LiDAR range (m): std of the second difference over consecutive returns / sqrt(6).
    Independent of any constant offset and of smooth motion, so it needs no calibration of the sensor frame."""
    d2 = []
    ok = np.isfinite(raw[idx])
    for i in range(1, len(idx) - 1):
        if ok[i - 1] and ok[i] and ok[i + 1] and idx[i + 1] - idx[i] == 1 and idx[i] - idx[i - 1] == 1:
            d2.append(raw[idx[i + 1]] - 2 * raw[idx[i]] + raw[idx[i - 1]])
    if len(d2) < 6:
        return None
    return float(np.std(d2) / math.sqrt(6.0))


def fit_lag(est: np.ndarray, gap: np.ndarray, idx: np.ndarray, max_steps: int = MAX_LAG_STEPS):
    """Lag (steps) at which the perceived range best matches the earlier true range; (lag, rms0, rms_lag, residuals)."""
    best, rms0 = None, None
    for L in range(0, max_steps + 1):
        sel = idx[idx - L >= 0]
        sel = sel[np.isfinite(est[sel])]
        if len(sel) < 4:
            continue
        res = est[sel] - gap[sel - L]
        rms = float(np.sqrt(np.mean(res ** 2)))
        if L == 0:
            rms0 = rms
        if best is None or rms < best[2] - 1e-9:
            best = (L, rms, rms, res)
    if best is None:
        return None
    return (best[0], rms0 if rms0 is not None else best[2], best[2], best[3])


def observe(ep: Episode, window: Optional[Window] = None, kind: str = 'collision') -> Dict[str, Dict]:
    """Observed anomaly per perception component over the critical window (or an explicit ``window``)."""
    w = window or critical_window(ep, kind)
    empty = lambda msg: {c: {'observed': False, 'detail': msg, 'start_t': None} for c in PERCEPTION}  # noqa: E731
    if w is None:
        return empty('no hazard window')
    a = _arrays(ep)
    dt = ep.limits.dt
    idx = np.arange(w.start, w.stop)
    keep = a['gap'][idx] < DEFAULT_CRITERIA.perception_range
    if w.closing_only:
        keep &= a['closing'][idx] > 0.1
    idx = idx[keep]
    if len(idx) < 4:
        return empty('too few closing steps in the window')
    return _features(a, idx, dt)


def _features(a: Dict[str, np.ndarray], idx: np.ndarray, dt: float) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    lost = ~a['tracked'][idx]
    runs = [(s, e) for s, e in _runs(lost) if e - s >= LOST_RUN_STEPS]
    miss_frac = float(np.mean(~a['detected'][idx]))
    det_obs = bool(runs) or (len(idx) >= 8 and miss_frac >= MISS_FRACTION)
    start = float(a['t'][idx[runs[0][0]]]) if runs else (float(a['t'][idx[0]]) if det_obs else None)
    out['object_detection'] = {'observed': det_obs, 'start_t': start,
                               'detail': f'longest lost-track run {max([e - s for s, e in runs], default=0)} steps, '
                                         f'no lead return in {100 * miss_frac:.0f}% of window steps'}
    tr = idx[a['tracked'][idx] & np.isfinite(a['est_gap'][idx])]
    fit = fit_lag(a['est_gap'], a['gap'], tr) if len(tr) >= 6 else None
    if fit is None:
        out['reaction_latency'] = {'observed': False, 'start_t': None, 'detail': 'too few tracked steps to fit a lag'}
        out['distance_estimation'] = {'observed': False, 'start_t': None, 'detail': 'too few tracked steps'}
    else:
        L, rms0, rmsL, res = fit
        lag_s = L * dt
        lat_obs = lag_s >= LAG_MIN_S - 1e-9 and rmsL < LAG_GAIN * rms0
        out['reaction_latency'] = {'observed': bool(lat_obs), 'start_t': float(a['t'][tr[0]]) if lat_obs else None,
                                   'detail': f'perceived range best matches the true range {lag_s:.1f} s earlier '
                                             f'(rms {rms0:.2f} m at zero lag -> {rmsL:.2f} m at the fitted lag)'}
        rms_noise = float(np.sqrt(np.mean((res - float(np.mean(res))) ** 2)))
        jit = raw_jitter(a['raw'], idx)
        dist_obs = rms_noise > DIST_RMS_M or (jit is not None and jit > RAW_JITTER_M)
        out['distance_estimation'] = {'observed': bool(dist_obs), 'start_t': float(a['t'][tr[0]]) if dist_obs else None,
                                      'detail': f'zero-mean range residual rms {rms_noise:.2f} m after removing the {lag_s:.1f} s lag; '
                                                f'raw-return jitter {"n/a" if jit is None else f"{jit:.2f} m"}'}
    ok = (np.isfinite(a['est_ttc'][idx]) & np.isfinite(a['ttc'][idx]) & (a['ttc'][idx] < 8.0)
          & (a['est_ttc'][idx] > 0) & (a['ttc'][idx] > 0))
    if ok.sum() >= 4:
        med = float(np.median(np.abs(np.log(a['est_ttc'][idx][ok] / a['ttc'][idx][ok]))))
        obs = med > TTC_LOG_RATIO
        out['ttc_estimation'] = {'observed': bool(obs), 'start_t': float(a['t'][idx[ok][0]]) if obs else None,
                                 'detail': f'median |log(est/true TTC)| {med:.2f} (bound {TTC_LOG_RATIO:.2f}) over {int(ok.sum())} steps'}
    else:
        out['ttc_estimation'] = {'observed': False, 'start_t': None, 'detail': 'no comparable TTC samples in the window'}
    return out


def nominal_statistics(episodes: Sequence[Episode]) -> Dict[str, Dict[str, float]]:
    """Distribution of every observation statistic over fault-free episodes (the basis of the thresholds above)."""
    rows = {'miss_frac': [], 'lag_s': [], 'dist_rms': [], 'raw_jitter': [], 'ttc_log': [], 'lost_run': []}
    for ep in episodes:
        a = _arrays(ep)
        idx = np.nonzero((a['closing'] > 0.5) & (a['gap'] < DEFAULT_CRITERIA.perception_range))[0]
        if len(idx) < 8:
            continue
        rows['miss_frac'].append(float(np.mean(~a['detected'][idx])))
        rows['lost_run'].append(max([e - s for s, e in _runs(~a['tracked'][idx])], default=0))
        tr = idx[a['tracked'][idx] & np.isfinite(a['est_gap'][idx])]
        fit = fit_lag(a['est_gap'], a['gap'], tr) if len(tr) >= 6 else None
        if fit:
            L, _r0, _rl, res = fit
            rows['lag_s'].append(L * ep.limits.dt)
            rows['dist_rms'].append(float(np.sqrt(np.mean((res - res.mean()) ** 2))))
        j = raw_jitter(a['raw'], idx)
        if j is not None:
            rows['raw_jitter'].append(j)
        ok = (np.isfinite(a['est_ttc'][idx]) & np.isfinite(a['ttc'][idx]) & (a['ttc'][idx] < 8.0)
              & (a['est_ttc'][idx] > 0) & (a['ttc'][idx] > 0))
        if ok.sum() >= 4:
            rows['ttc_log'].append(float(np.median(np.abs(np.log(a['est_ttc'][idx][ok] / a['ttc'][idx][ok])))))
    return {k: {'n': len(v), 'p50': float(np.percentile(v, 50)) if v else float('nan'),
                'p95': float(np.percentile(v, 95)) if v else float('nan'),
                'p99': float(np.percentile(v, 99)) if v else float('nan'), 'max': float(max(v)) if v else float('nan')}
            for k, v in rows.items()}


# --- verdict -----------------------------------------------------------------------------------------------------------------
def _paired_counts(base: List[bool], var: List[bool]):
    fixed = sum(1 for b, w in zip(base, var) if b and not w)
    broken = sum(1 for b, w in zip(base, var) if (not b) and w)
    return fixed, broken


def attribute(ev, scenario, diag: Diagnosis) -> Dict:
    """Verdict for a diagnosed failure. Reuses the variants already simulated by ``diagnose``; one extra episode is run for telemetry."""
    kind = diag.kind
    variants = {frozenset(v.fixed): v for v in diag.variants if v.policy == 'planner'
                and (v.name == 'original' or v.name.startswith('fix ') or v.name.startswith('ideal'))}
    testable = list(diag.testable)
    full = frozenset(testable)
    base, ideal = variants[frozenset()], variants[full]
    p0, p_ideal = base.rate, ideal.rate
    ep = ev.run(scenario)
    win = critical_window(ep, kind)
    obs = observe(ep, win, kind) if win is not None else {c: {'observed': False, 'start_t': None, 'detail': 'no hazard window'} for c in PERCEPTION}
    reasons: List[str] = []
    if win is None:
        reasons.append('no unwarranted hard-braking run' if kind == 'unnecessary_braking'
                       else 'the ground truth never required hard braking in this episode, so there is no hazard window')
    if p0 < 0.5:
        reasons.append(f'seed-fragile: the failure occurs in only {base.k}/{base.n} noise seeds')

    per: Dict[str, Dict] = {}
    for c in testable:
        v = variants[frozenset([c])]
        fixed, broken = _paired_counts(base.outcomes, v.outcomes)
        k0 = max(base.k, 1)
        fix_frac = fixed / k0
        alone = variants[full - frozenset([c])]
        alone_excess = alone.rate - p_ideal
        shap = diag.shapley[c]
        fixes = fixed >= FIX_FRACTION * k0 and broken <= fixed / 4
        sufficient_alone = alone.rate >= ALONE_MIN_RATE and alone_excess >= 0.375
        shap_ok = shap['ci90'][0] > SHAPLEY_FLOOR
        o = obs[c]
        timely = bool(o['observed'] and win is not None and o['start_t'] is not None and o['start_t'] <= win.t_deadline + 0.1 + 1e-9)
        cf = bool(fixes or sufficient_alone or shap_ok)
        per[c] = {'label': LABELS[c], 'error_observed': bool(o['observed']), 'error_before_unsafe_decision': timely,
                  'observation': o['detail'], 'fixes_outcome': bool(fixes), 'fix_fraction': round(fix_frac, 3),
                  'paired_fixed': fixed, 'paired_broken': broken, 'sufficient_alone': bool(sufficient_alone),
                  'failure_rate_alone': round(alone.rate, 3), 'shapley_share': shap['value'],
                  'counterfactual_evidence': cf,
                  'effect': round(max(fix_frac if fixes else 0.0, alone_excess if sufficient_alone else 0.0,
                                      shap['value'] if shap_ok else 0.0), 3),
                  'contributed': bool(timely and cf)}
    for c in diag.untestable:
        per[c] = {'label': LABELS[c], 'class': 'not testable on this SUT', 'contributed': False, 'effect': 0.0,
                  'error_observed': False, 'fixes_outcome': False, 'observation': diag.untestable[c]}

    contributors = [c for c in testable if per[c]['contributed']]
    lo_i, hi_i = wilson(ideal.k, ideal.n)
    planner_fails = lo_i > 0.5
    ref = [v for v in diag.variants if v.name == 'reference policy (ideal perception)']
    ref_txt = 'n/a' if not ref else f'{ref[0].rate:.2f}'
    per['planner_logic'] = {
        'label': LABELS['planner_logic'], 'contributed': bool(planner_fails), 'effect': round(p_ideal, 3),
        'error_observed': bool(planner_fails), 'fixes_outcome': False,
        'observation': f'with every perception component ideal the planner still fails in {ideal.k}/{ideal.n} seeds '
                       f'(90% CI {lo_i:.2f}..{hi_i:.2f}); the RSS reference on the same inputs fails in {ref_txt} of seeds '
                       f'(a comparison point, not ground truth)'}

    for c, d in per.items():
        if 'class' in d:
            continue
        if d['contributed']:
            d['class'] = 'contributing'
        elif d.get('error_observed') and not d.get('counterfactual_evidence'):
            d['class'] = 'error observed, ground-truth replacement does not change the outcome'
        elif d.get('error_observed') and d.get('counterfactual_evidence'):
            d['class'] = 'error observed only after the unsafe decision'
        elif d.get('fixes_outcome'):
            d['class'] = 'fixing it helps, but no error was observed in the critical window'
        else:
            d['class'] = 'not implicated'

    cands = sorted(contributors + (['planner_logic'] if planner_fails else []), key=lambda c: -per[c]['effect'])
    primary: List[str] = []
    verdict = 'inconclusive'
    if cands:
        top = per[cands[0]]['effect']
        if top >= 0.5:
            rivals = [c for c in cands[1:] if per[c]['effect'] >= DOMINANCE * top]
            primary = [cands[0]] + rivals
            verdict = 'primary' if not rivals else 'joint'
        else:
            reasons.append(f'no component has a counterfactual effect of at least 0.5 (largest {top:.2f})')
    else:
        reasons.append('no perception component was both observed wrong before the unsafe decision and counterfactually '
                       'relevant, and the failure does not persist with ideal perception')
    for c in primary:
        per[c]['class'] = 'primary root cause' if verdict == 'primary' else 'joint root cause (ambiguous split)'
    return {'verdict': verdict, 'primary': primary, 'contributing': [c for c in cands if c not in primary],
            'outcome_improved_by': [c for c in testable if per[c]['fixes_outcome']],
            'error_observed': [c for c in testable if per[c]['error_observed']],
            'window': None if win is None else {'t_hazard': win.t_h, 't_unsafe_decision': win.t_u, 't_deadline': win.t_deadline, 't_end': win.t_end, 'note': win.note},
            'components': per, 'inconclusive_reasons': reasons,
            'failure_rate_original': round(p0, 3), 'failure_rate_ideal_perception': round(p_ideal, 3), 'n_seeds': base.n,
            'definitions': {'contributing': 'error observed before the unsafe decision AND ground-truth replacement changes the outcome',
                            'fixes_outcome': 'fixing only this component removes >= half of the failures (paired seeds)',
                            'primary root cause': 'contributor with effect >= 0.5 and no contributor of comparable effect'}}


def summary_lines(attr: Dict) -> List[str]:
    """Plain-text lines for the CLI / demo."""
    comps = attr['components']
    v = attr['verdict']
    head = {'primary': 'primary root cause: ' + ', '.join(comps[c]['label'] for c in attr['primary']),
            'joint': 'joint / ambiguous root cause between: ' + ', '.join(comps[c]['label'] for c in attr['primary']),
            'inconclusive': 'inconclusive'}[v]
    out = [f'attribution: {head}']
    if v == 'inconclusive':
        out += [f'  reason: {r}' for r in attr['inconclusive_reasons']]
    if attr['contributing']:
        out.append('  also contributing: ' + ', '.join(comps[c]['label'] for c in attr['contributing']))
    if attr['outcome_improved_by']:
        out.append('  fixing alone removes most failures: ' + ', '.join(comps[c]['label'] for c in attr['outcome_improved_by'])
                   + ' (an intervention result, not a root-cause claim)')
    for c, d in comps.items():
        if d.get('error_observed') or d.get('contributed'):
            out.append(f'  - {d["label"]}: {d["class"]}; {d["observation"]}')
    return out


def render_markdown(attr: Dict) -> List[str]:
    comps = attr['components']
    L = ['### Evidence-based attribution\n',
         'Three questions are answered separately. *Contributed*: the component was observably wrong before the point of no return '
         'AND ground-truth replacement changes the outcome. *Intervention improves outcome*: fixing it alone removes at least half the '
         'failures. *Primary root cause*: a contributor with a large effect and no comparable rival. Anything else is reported as '
         '*joint* or *inconclusive*, never forced.\n']
    w = attr.get('window')
    if w:
        L.append(f'Critical window: hazard from t={w["t_hazard"]:.1f} s, first insufficient command t={w["t_unsafe_decision"]:.1f} s, '
                 f'errors count if they start before t={w["t_deadline"]:.1f} s ({w["note"]}).\n')
    L.append(f'**Verdict: {attr["verdict"]}**' + (f' ({", ".join(comps[c]["label"] for c in attr["primary"])})' if attr['primary'] else '')
             + f'. Failure rate {attr["failure_rate_original"]:.2f} as given, {attr["failure_rate_ideal_perception"]:.2f} with ideal perception ({attr["n_seeds"]} seeds).\n')
    if attr['inconclusive_reasons']:
        L += [f'- Not concluded because: {r}' for r in attr['inconclusive_reasons']]
        L.append('')
    L.append('| component | observed error in window | before deadline | fixing alone helps | counterfactual effect | classification |\n|---|---|---|---|---|---|')
    for c, d in comps.items():
        L.append(f'| {d["label"]} | {d.get("error_observed", False)} | {d.get("error_before_unsafe_decision", "-")} | '
                 f'{d.get("fixes_outcome", False)} | {d.get("effect", 0):.2f} | {d.get("class", "")} |')
    L.append('')
    for c, d in comps.items():
        if d.get('observation'):
            L.append(f'- {d["label"]}: {d["observation"]}')
    L.append('')
    return L
