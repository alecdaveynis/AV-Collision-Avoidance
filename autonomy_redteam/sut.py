"""The ``SystemUnderTest`` interface: everything the red-team framework needs from an AV stack.

The framework (search, minimisation, diagnosis, regression, reporting) only talks to this
interface and to the telemetry dataclasses below, never to ``safety_sim`` directly. A CARLA or
ROS 2 adapter would implement the same few methods; none is provided yet (see ``adapters/``).

Conventions
  * ``step()`` advances simulated time by ``limits.dt`` and returns one :class:`StepRecord`.
  * Ground-truth fields in a record describe the world at the *start* of the step (the instant
    the planner decided); ``gap_after`` / ``collided`` describe the world after the step.
  * Missing numbers are ``nan``; "not closing" TTC is ``inf``. Serialisation maps both to null.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Dict, FrozenSet, List, Optional

from .scenario import Scenario

NAN = math.nan
STATES = ('DRIVE', 'SLOWING', 'STOPPED')       # same labels as behavior_manager.BehaviorState


def _clean(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


@dataclass(frozen=True)
class VehicleLimits:
    """Actuation limits and time step of the SUT; the feasibility oracle uses the same numbers."""
    dt: float
    max_decel: float          # m/s^2
    actuator_lag: float       # s, first-order lag between commanded and applied deceleration
    lead_length: float = 4.5  # m


@dataclass(frozen=True)
class Interventions:
    """SUT-level counterfactual switches (scenario-level ones are applied by editing the Scenario)."""
    oracle_distance: bool = False     # perception receives noise-free LiDAR clusters
    oracle_ttc: bool = False          # per-track TTC replaced by the true TTC
    policy: str = 'planner'           # 'planner' | 'reference' | 'max_brake' | 'none'

    def key(self) -> str:
        return f'od={int(self.oracle_distance)},ot={int(self.oracle_ttc)},pol={self.policy}'

    def is_default(self) -> bool:
        return self == Interventions()


@dataclass
class GroundTruthState:
    t: float
    ego_x: float
    ego_v: float
    ego_decel: float          # applied deceleration (after actuator lag)
    lead_x: float             # lead rear bumper position
    lead_v: float
    gap: float                # bumper-to-bumper
    closing_speed: float      # ego_v - lead_v (> 0 : gap shrinking)
    ttc: float                # gap / closing_speed (inf if not closing)
    collided: bool = False


@dataclass
class PerceptionTelemetry:
    lead_detected: bool       # a cluster of the lead reached the stack this step (after latency)
    n_clusters: int
    tracked: bool             # the stack reports a confirmed in-path track
    est_gap: float = NAN      # range of the nearest in-path track as the planner sees it
    est_closing: float = NAN
    est_ttc: float = NAN      # track TTC after fault injection / oracle substitution
    range_error: float = NAN  # est_gap - true gap now (includes staleness from latency)
    raw_range: float = NAN    # nearest in-path LiDAR return before tracking/filtering (a raw sensor output, any stack has it)


@dataclass
class PlannerTelemetry:
    state: str                # DRIVE | SLOWING | STOPPED
    raw_state: str            # before hysteresis
    cmd_decel: float          # m/s^2 commanded
    ttc: float = math.inf     # TTC the planner's decision used (min over entities)
    min_dist: float = 99.0


@dataclass
class SafetyEvent:
    t: float
    kind: str
    detail: str = ''


@dataclass
class StepRecord:
    t: float
    gt: GroundTruthState
    perception: PerceptionTelemetry
    planner: PlannerTelemetry
    gap_after: float
    collided: bool
    events: List[SafetyEvent] = field(default_factory=list)


@dataclass
class Episode:
    scenario: Scenario
    records: List[StepRecord]
    limits: VehicleLimits
    interventions: Interventions = field(default_factory=Interventions)
    sut_info: Dict = field(default_factory=dict)

    @property
    def events(self) -> List[SafetyEvent]:
        return [e for r in self.records for e in r.events]

    def series(self) -> Dict[str, List[float]]:
        """Flat time series for plotting / CSV."""
        rows = {k: [] for k in ('t', 'ego_v', 'lead_v', 'gap', 'est_gap', 'ttc', 'est_ttc', 'cmd_decel',
                                'ego_decel', 'closing')}
        for r in self.records:
            rows['t'].append(r.t)
            rows['ego_v'].append(r.gt.ego_v)
            rows['lead_v'].append(r.gt.lead_v)
            rows['gap'].append(r.gt.gap)
            rows['est_gap'].append(r.perception.est_gap)
            rows['ttc'].append(r.gt.ttc)
            rows['est_ttc'].append(r.perception.est_ttc)
            rows['cmd_decel'].append(r.planner.cmd_decel)
            rows['ego_decel'].append(r.gt.ego_decel)
            rows['closing'].append(r.gt.closing_speed)
        return rows

    def states(self) -> List[str]:
        return [r.planner.state for r in self.records]

    def to_dict(self) -> dict:
        recs = []
        for r in self.records:
            d = asdict(r)
            recs.append({'t': r.t, 'gt': _walk(d['gt']), 'perception': _walk(d['perception']),
                         'planner': _walk(d['planner']), 'gap_after': _clean(r.gap_after),
                         'collided': r.collided, 'events': d['events']})
        return {'scenario': self.scenario.to_dict(), 'interventions': asdict(self.interventions),
                'limits': asdict(self.limits), 'sut': self.sut_info, 'records': recs}


def _walk(d):
    return {k: _clean(v) for k, v in d.items()}


class SystemUnderTest(ABC):
    """Abstract AV stack + simulator pair driven by the framework."""

    name: str = 'sut'
    #: scenario fault parameters this SUT can actually model (others are rejected, not silently ignored)
    supported_faults: FrozenSet[str] = frozenset()
    #: SUT-level interventions it supports (subset of Interventions field names / policy names)
    supported_interventions: FrozenSet[str] = frozenset()

    @abstractmethod
    def initialize(self, scenario: Scenario, interventions: Optional[Interventions] = None) -> None:
        """Load a scenario and put the simulation into its initial state."""

    @abstractmethod
    def reset(self) -> None:
        """Restore the initial state of the *current* scenario (same seed: bit-identical replay)."""

    @abstractmethod
    def step(self) -> StepRecord:
        """Advance simulated time by one step and return the telemetry of that step."""

    @property
    @abstractmethod
    def done(self) -> bool:
        """True once the episode ended (collision, stopped, or time limit)."""

    @abstractmethod
    def ground_truth(self) -> GroundTruthState:
        """Current ground-truth state of ego and lead vehicle."""

    @abstractmethod
    def perception_telemetry(self) -> PerceptionTelemetry:
        """Perception outputs of the most recent step."""

    @abstractmethod
    def planner_telemetry(self) -> PlannerTelemetry:
        """Planner decision of the most recent step."""

    @abstractmethod
    def events(self) -> List[SafetyEvent]:
        """All safety events reported so far."""

    @abstractmethod
    def limits(self) -> VehicleLimits:
        """Time step and actuation limits."""

    def describe(self) -> Dict:
        """Configuration identifying this SUT (stored in baselines so comparisons are explicit)."""
        return {'name': self.name}

    def check_supported(self, scenario: Scenario, interventions: Interventions) -> List[str]:
        """Names of requested features this SUT cannot model (empty = all fine)."""
        return []

    # -- convenience -----------------------------------------------------------------------------
    def run(self, scenario: Scenario, interventions: Optional[Interventions] = None) -> Episode:
        self.initialize(scenario, interventions)
        records = []
        while not self.done:
            records.append(self.step())
        return Episode(scenario=scenario, records=records, limits=self.limits(),
                       interventions=interventions or Interventions(), sut_info=self.describe())
