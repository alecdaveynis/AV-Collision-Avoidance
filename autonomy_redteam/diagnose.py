"""Counterfactual root-cause isolation by controlled experiments.

For a failing scenario, the engine re-runs it under *identical initial conditions and noise seeds* while giving
individual perception components their ideal behaviour, and measures how the failure changes.

Components (each has one ideal-behaviour intervention, see ``interventions.py``)
    object_detection      no dropout, no detection-loss window, no ghost clusters
    distance_estimation   noise-free LiDAR ranges (ground-truth distance)
    ttc_estimation        ground-truth TTC replaces the per-track TTC            (kalman planner mode only)
    reaction_latency      no perception latency

Design (why this is more than "one intervention flipped the outcome")
  * Full factorial: every subset of the components is fixed in turn (2^k variants), each replicated over ``n_seeds``
    noise seeds, so interactions between components are observed, not assumed away.
  * Outcome = failure rate over seeds with a Wilson interval, not a single run.
  * Attribution: Shapley values of the failure *probability* over the components (they sum exactly to
    p(failure) - p(failure with ideal perception)), with a bootstrap-over-seeds interval. The remainder,
    p(failure with ideal perception), is the part perception cannot explain.
  * Planner logic: if failure persists with ideal perception, and a reference braking rule (RSS) avoids it on
    the same inputs, the evidence points at the planner's decision rule / thresholds / actuation margins.
  * Labels are "supported / weak / not supported" with the rule stated in the report: *evidence consistent with*,
    never a proof of a unique root cause. The kinematic oracle says whether the collision was physically avoidable.

Everything here is deterministic given the scenario; no LLM involved.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from .interventions import COMPONENTS, INTERVENTIONS, apply, names_for_components
from .metrics import Assessment
from .minimize import derived_seeds
from .runner import Evaluator
from .scenario import Scenario

SUPPORTED_MARGIN = 0.10          # Shapley share (probability points) that counts as a real contribution
HYPOTHESIS_LABEL = {'object_detection': 'Object detection', 'distance_estimation': 'Distance estimation',
                    'ttc_estimation': 'TTC estimation', 'reaction_latency': 'Reaction latency (perception)',
                    'planner_logic': 'Planner decision logic / thresholds'}


def wilson(k: int, n: int, z: float = 1.645) -> Tuple[float, float]:
    """Wilson score interval (default ~90 %)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def shapley(players: Sequence[str], v: Dict[FrozenSet[str], float]) -> Dict[str, float]:
    """Shapley values for a game v(set of 'broken' players) -> failure probability."""
    n = len(players)
    out = {}
    for c in players:
        others = [p for p in players if p != c]
        tot = 0.0
        for r in range(len(others) + 1):
            w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for B in itertools.combinations(others, r):
                tot += w * (v[frozenset(B) | {c}] - v[frozenset(B)])
        out[c] = tot
    return out


@dataclass
class VariantResult:
    name: str
    fixed: Tuple[str, ...]                 # components given ideal behaviour
    interventions: Tuple[str, ...]
    policy: str
    outcomes: List[bool]                   # failure of the target kind, per seed (paired across variants)
    assessments: List[Assessment]

    @property
    def n(self):
        return len(self.outcomes)

    @property
    def k(self):
        return int(sum(self.outcomes))

    @property
    def rate(self):
        return self.k / self.n if self.n else 0.0

    def to_dict(self) -> dict:
        lo, hi = wilson(self.k, self.n)
        return {'name': self.name, 'fixed_components': list(self.fixed), 'interventions': list(self.interventions),
                'policy': self.policy, 'n_seeds': self.n, 'failures': self.k, 'failure_rate': round(self.rate, 4),
                'ci90': [round(lo, 4), round(hi, 4)],
                'mean_min_gap': round(float(np.mean([a.min_gap for a in self.assessments])), 3),
                'collision_rate': round(float(np.mean([a.collision for a in self.assessments])), 4),
                'mean_impact_speed': round(float(np.mean([a.impact_speed for a in self.assessments if a.collision]
                                                          or [0.0])), 3)}


@dataclass
class Diagnosis:
    scenario: Scenario
    kind: str
    n_seeds: int
    testable: List[str]
    untestable: Dict[str, str]
    original: Dict
    variants: List[VariantResult]
    shapley: Dict[str, Dict]
    hypotheses: List[Dict]
    evidence: List[Dict]
    minimal_fixing_sets: List[Dict]
    planner: Dict
    notes: List[str]
    sims: int = 0
    attribution: Optional[Dict] = None      # evidence-based verdict, attached by attribution.attribute (see attribution.py)

    def to_dict(self) -> dict:
        return {'scenario': self.scenario.to_dict(), 'failure_kind': self.kind, 'n_seeds': self.n_seeds,
                'testable_components': self.testable, 'untestable_components': self.untestable,
                'original_outcome': self.original, 'variants': [v.to_dict() for v in self.variants],
                'shapley_failure_probability': self.shapley, 'hypotheses': self.hypotheses,
                'evidence': self.evidence, 'minimal_fixing_sets': self.minimal_fixing_sets,
                'planner_logic_check': self.planner, 'notes': self.notes, 'simulations': self.sims,
                'attribution': self.attribution}

    def ranked_hypotheses(self) -> List[Dict]:
        order = {'supported': 0, 'weak': 1, 'not supported': 2, 'not testable': 3}
        return sorted(self.hypotheses, key=lambda h: (order[h['status']], -h.get('share', 0.0)))


_KEY_METRICS = ('collision', 'impact_speed', 'min_gap', 'min_ttc', 'brake_reaction_time', 'missed_hazards',
                'unnecessary_braking_s', 'max_state', 'first_brake_t')


def _metrics(a: Assessment) -> dict:
    d = a.to_dict()
    return {k: d[k] for k in _KEY_METRICS}


def _changed(a0: dict, a1: dict) -> dict:
    out = {}
    for k in _KEY_METRICS:
        x, y = a0[k], a1[k]
        if isinstance(x, float) and isinstance(y, float):
            if abs(x - y) > 1e-6:
                out[k] = [round(x, 3), round(y, 3)]
        elif x != y:
            out[k] = [x if not isinstance(x, float) else round(x, 3), y if not isinstance(y, float) else round(y, 3)]
    return out


def diagnose(ev: Evaluator, scenario: Scenario, kind: Optional[str] = None, n_seeds: int = 8,
             components: Optional[Sequence[str]] = None) -> Diagnosis:
    sims0 = ev.n_sims
    a0 = ev.evaluate(scenario)
    kind = kind or a0.primary_kind
    if kind is None:
        raise ValueError('scenario does not fail; nothing to diagnose')
    seeds = derived_seeds(scenario.seed, n_seeds)
    comps_all = list(components or COMPONENTS)
    testable, untestable = [], {}
    for c in comps_all:
        sc, iv = apply(scenario, names_for_components([c]))
        missing = ev.sut.check_supported(sc, iv)
        if missing and c == 'ttc_estimation':
            untestable[c] = '; '.join(missing)
        elif missing:
            untestable[c] = '; '.join(missing)
        else:
            testable.append(c)
    if not testable:
        raise ValueError('no testable component for this SUT')

    def run_variant(name, fixed, extra_ints=(), policy=None) -> VariantResult:
        names = tuple(names_for_components(fixed)) + tuple(extra_ints)
        outs, assess_ = [], []
        for sd in seeds:
            sc, iv = apply(scenario.replace(seed=sd), names)
            if policy:
                from dataclasses import replace as _r
                iv = _r(iv, policy=policy)
            a = ev.evaluate(sc, iv)
            outs.append(kind in a.failure_kinds)
            assess_.append(a)
        return VariantResult(name, tuple(fixed), names, policy or 'planner', outs, assess_)

    variants: Dict[FrozenSet[str], VariantResult] = {}
    for r in range(len(testable) + 1):
        for S in itertools.combinations(testable, r):
            nm = 'original' if r == 0 else ('ideal perception (all)' if r == len(testable) else
                                            'fix ' + '+'.join(S))
            variants[frozenset(S)] = run_variant(nm, S)
    full = frozenset(testable)
    base = variants[frozenset()]
    ideal = variants[full]
    # extra variants: reference policy, fine-grained detection ablations
    extras: List[VariantResult] = []
    ref_orig = run_variant('reference policy (original perception)', (), policy='reference')
    ref_ideal = run_variant('reference policy (ideal perception)', testable, policy='reference')
    extras += [ref_orig, ref_ideal]
    fine = []
    for nm, active in (('no_dropout', scenario.dropout_prob > 0), ('no_detection_loss', scenario.detection_loss_duration_s > 0),
                       ('no_ghosts', scenario.ghost_prob > 0)):
        if active and 'object_detection' in testable:
            fine.append(run_variant(nm, (), extra_ints=(nm,)))
    extras += fine

    # --- attribution ------------------------------------------------------------------------------
    players = list(testable)
    n = base.n

    def game_from(outcome_by_set: Dict[FrozenSet[str], np.ndarray]) -> Dict[FrozenSet[str], float]:
        return {frozenset(B): float(outcome_by_set[full - frozenset(B)].mean()) for B in
                itertools.chain.from_iterable(itertools.combinations(players, r) for r in range(len(players) + 1))}

    outcome = {S: np.array(v.outcomes, float) for S, v in variants.items()}
    phi = shapley(players, game_from(outcome))
    rng = np.random.default_rng(12345)
    boots = {c: [] for c in players}
    for _ in range(400):
        idx = rng.integers(0, n, n)
        ph = shapley(players, game_from({S: o[idx] for S, o in outcome.items()}))
        for c in players:
            boots[c].append(ph[c])
    shap = {c: {'value': round(phi[c], 4),
                'ci90': [round(float(np.percentile(boots[c], 5)), 4), round(float(np.percentile(boots[c], 95)), 4)]}
            for c in players}
    p0, p_ideal = base.rate, ideal.rate

    # --- evidence per intervention (original seed + rate over seeds) ----------------------------------------
    evidence = []
    seed0_base = _metrics(base.assessments[0])
    for c in players:
        v = variants[frozenset([c])]
        met = _metrics(v.assessments[0])
        fixed = int(sum(1 for b, w in zip(base.outcomes, v.outcomes) if b and not w))
        broken = int(sum(1 for b, w in zip(base.outcomes, v.outcomes) if (not b) and w))
        evidence.append({'component': c, 'intervention': v.interventions[0] if v.interventions else '',
                         'description': INTERVENTIONS[v.interventions[0]].description,
                         'original_outcome_seed0': seed0_base, 'with_intervention_seed0': met,
                         'changed_metrics_seed0': _changed(seed0_base, met),
                         'failure_rate_original': round(base.rate, 4), 'failure_rate_with_intervention': round(v.rate, 4),
                         'paired_fixed_by_intervention': fixed, 'paired_broken_by_intervention': broken,
                         'n_seeds': n})
    for v in extras:
        met = _metrics(v.assessments[0])
        evidence.append({'component': 'planner_logic' if 'reference' in v.name else 'object_detection',
                         'intervention': ','.join(v.interventions) or f'policy={v.policy}', 'description': v.name,
                         'original_outcome_seed0': seed0_base, 'with_intervention_seed0': met,
                         'changed_metrics_seed0': _changed(seed0_base, met),
                         'failure_rate_original': round(base.rate, 4), 'failure_rate_with_intervention': round(v.rate, 4),
                         'paired_fixed_by_intervention': int(sum(1 for b, w in zip(base.outcomes, v.outcomes) if b and not w)),
                         'paired_broken_by_intervention': int(sum(1 for b, w in zip(base.outcomes, v.outcomes) if (not b) and w)),
                         'n_seeds': n})

    # --- hypotheses ----------------------------------------------------------------------------------------------
    hyps = []
    notes = []
    for c in comps_all:
        if c in untestable:
            hyps.append({'component': c, 'label': HYPOTHESIS_LABEL[c], 'status': 'not testable',
                         'share': 0.0, 'evidence': [f'not testable on this SUT: {untestable[c]}']})
            continue
        s = shap[c]
        single = p0 - variants[frozenset([c])].rate
        alone = variants[full - frozenset([c])].rate - p_ideal      # failure caused by this component alone (others ideal)
        ev_lines = [f'Shapley share of failure probability {s["value"]:+.2f} (90% CI {s["ci90"][0]:+.2f}..{s["ci90"][1]:+.2f})',
                    f'fixing only this component changes the failure rate by {-single:+.2f} '
                    f'({p0:.2f} -> {variants[frozenset([c])].rate:.2f})',
                    f'with every other component ideal, this component alone yields a failure rate of '
                    f'{variants[full - frozenset([c])].rate:.2f} (ideal-perception baseline {p_ideal:.2f})']
        if s['ci90'][0] > SUPPORTED_MARGIN:
            status = 'supported'
        elif s['value'] > SUPPORTED_MARGIN:
            status = 'weak'
        else:
            status = 'not supported'
        hyps.append({'component': c, 'label': HYPOTHESIS_LABEL[c], 'status': status, 'share': s['value'],
                     'evidence': ev_lines})
    lo_i, hi_i = wilson(ideal.k, ideal.n)
    ref_ideal_rate = ref_ideal.rate
    ac = a0.collision_class
    persists = lo_i > 0.5
    plan_lines = [f'with ideal perception the production planner still fails in {ideal.k}/{ideal.n} seeds '
                  f'(rate {p_ideal:.2f}, 90% CI {lo_i:.2f}..{hi_i:.2f})',
                  f'the reference RSS rule on ideal perception fails in {ref_ideal.k}/{ref_ideal.n} seeds',
                  f'kinematic oracle: collision class = {ac} (max-brake from t=0 collides: {a0.oracle_max_brake_collides}; '
                  f'reference policy with perfect information collides: {a0.oracle_reference_collides})']
    if persists and ref_ideal_rate < p_ideal - 0.25:
        pstatus = 'supported'
    elif p_ideal > SUPPORTED_MARGIN and ref_ideal_rate < p_ideal:
        pstatus = 'weak'
    else:
        pstatus = 'not supported'
    hyps.append({'component': 'planner_logic', 'label': HYPOTHESIS_LABEL['planner_logic'], 'status': pstatus,
                 'share': round(p_ideal, 4), 'evidence': plan_lines})
    planner = {'failure_rate_ideal_perception': round(p_ideal, 4), 'ci90': [round(lo_i, 4), round(hi_i, 4)],
               'reference_rate_original_perception': round(ref_orig.rate, 4),
               'reference_rate_ideal_perception': round(ref_ideal_rate, 4), 'collision_class': ac,
               'status': pstatus}

    # --- minimal fixing sets ------------------------------------------------------------------------------------------
    fixing = []
    if p0 > 0:
        for r in range(0, len(testable) + 1):
            for S in itertools.combinations(testable, r):
                v = variants[frozenset(S)]
                if v.rate <= 0.2 * p0 and r > 0:
                    if not any(set(f['components']) <= set(S) for f in fixing):
                        fixing.append({'components': list(S), 'failure_rate': round(v.rate, 4), 'from': round(p0, 4)})
    if p0 < 0.5:
        notes.append(f'the original scenario fails in only {base.k}/{base.n} noise seeds: the failure is seed-fragile, '
                     'so the attribution below has low power')
    if persists:
        notes.append('failure persists with ideal perception: perception cannot be the whole explanation')
    if n < 8:
        notes.append(f'only {n} seeds per variant: intervals are wide')
    notes.append('Labels are evidence consistent with a cause (Shapley share and 90 % intervals over noise seeds), '
                 'not a proof of a unique root cause; components interact and the attribution is conditional on this '
                 'scenario, planner configuration and noise model.')
    if 'ttc_estimation' in untestable:
        notes.append('TTC estimation was not tested (analytic planner mode computes TTC internally).')
    return Diagnosis(scenario=scenario, kind=kind, n_seeds=n, testable=testable, untestable=untestable,
                     original=a0.to_dict(), variants=list(variants.values()) + extras, shapley=shap,
                     hypotheses=hyps, evidence=evidence, minimal_fixing_sets=fixing, planner=planner, notes=notes,
                     sims=ev.n_sims - sims0)
