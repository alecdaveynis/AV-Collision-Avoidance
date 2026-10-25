"""Named counterfactual interventions used by the diagnostic engine (and available to the agent).

An intervention edits the scenario (removing a perturbation) and/or switches the SUT into an oracle
mode (ground-truth distance, ground-truth TTC, reference braking policy). Every intervention is applied
to the *same scenario and seed*, so differences in outcome are attributable to the intervention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from .scenario import PARAMS, Scenario
from .sut import Interventions


@dataclass(frozen=True)
class Intervention:
    name: str
    description: str
    component: str                       # hypothesis the intervention speaks to
    scenario_overrides: Tuple[Tuple[str, float], ...] = ()
    sut: Tuple[Tuple[str, object], ...] = ()


def _ov(**kw):
    return tuple(kw.items())


INTERVENTIONS: Dict[str, Intervention] = {i.name: i for i in [
    Intervention('no_dropout', 'remove random sensor dropout (dropout_prob = 0)', 'object_detection',
                 _ov(dropout_prob=0.0)),
    Intervention('no_detection_loss', 'remove the temporary detection-loss window', 'object_detection',
                 _ov(detection_loss_duration_s=0.0, detection_loss_start_s=0.0)),
    Intervention('no_ghosts', 'remove false in-lane clusters', 'object_detection', _ov(ghost_prob=0.0)),
    Intervention('ideal_detection', 'restore ideal object detection: no dropout, no detection loss, no ghosts',
                 'object_detection',
                 _ov(dropout_prob=0.0, detection_loss_duration_s=0.0, detection_loss_start_s=0.0, ghost_prob=0.0)),
    Intervention('oracle_distance', 'replace measured LiDAR ranges by ground truth (no range noise at all)',
                 'distance_estimation', _ov(distance_noise_std_m=0.0), (('oracle_distance', True),)),
    Intervention('oracle_ttc', 'replace the per-track TTC by the ground-truth TTC', 'ttc_estimation',
                 _ov(ttc_error_factor=1.0), (('oracle_ttc', True),)),
    Intervention('no_latency', 'remove perception latency', 'reaction_latency', _ov(perception_latency_s=0.0)),
    Intervention('reference_policy',
                 'same perception, but decisions by the reference longitudinal RSS rule instead of the planner',
                 'planner_logic', (), (('policy', 'reference'),)),
]}

# interventions that jointly give each perception component its ideal behaviour
COMPONENT_FIX: Dict[str, Tuple[str, ...]] = {
    'object_detection': ('ideal_detection',),
    'distance_estimation': ('oracle_distance',),
    'ttc_estimation': ('oracle_ttc',),
    'reaction_latency': ('no_latency',),
}
COMPONENTS = tuple(COMPONENT_FIX)


def apply(scenario: Scenario, names: Iterable[str]) -> Tuple[Scenario, Interventions]:
    """Scenario + SUT switches for a set of named interventions (order-independent)."""
    sc_kw: Dict[str, float] = {}
    sut_kw: Dict[str, object] = {}
    for n in names:
        if n not in INTERVENTIONS:
            raise KeyError(f'unknown intervention {n!r}; known: {sorted(INTERVENTIONS)}')
        iv = INTERVENTIONS[n]
        sc_kw.update(dict(iv.scenario_overrides))
        sut_kw.update(dict(iv.sut))
    new = scenario.replace(**sc_kw) if sc_kw else scenario
    return new, Interventions(**sut_kw)


def names_for_components(components: Iterable[str]) -> List[str]:
    out: List[str] = []
    for c in components:
        out.extend(COMPONENT_FIX[c])
    return out
