"""Runs scenarios against a SystemUnderTest and assesses them; counts and caches simulations."""
from __future__ import annotations

import time
from typing import Callable, Dict, Iterable, Optional, Tuple

from .adapters import default_sut_factory
from .interventions import apply
from .metrics import DEFAULT_CRITERIA, Assessment, FailureCriteria, assess
from .scenario import Scenario
from .sut import Episode, Interventions, SystemUnderTest


class BudgetExceeded(RuntimeError):
    """Raised when an evaluator's simulation budget is exhausted."""


class Evaluator:
    """One place to run simulations.

    ``n_sims`` counts distinct SUT episodes actually simulated (cache hits are free, and the kinematic
    oracles used for avoidability are not counted: they cost microseconds and run no perception stack).
    Budgets passed as ``max_sims`` are enforced here, so a search cannot silently overspend.
    """

    def __init__(self, sut_factory: Optional[Callable[[], SystemUnderTest]] = None,
                 criteria: FailureCriteria = DEFAULT_CRITERIA, max_sims: Optional[int] = None):
        self.sut_factory = sut_factory or default_sut_factory()
        self.sut = self.sut_factory()
        self.criteria = criteria
        self.max_sims = max_sims
        self.n_sims = 0
        self.n_cache_hits = 0
        self.sim_seconds = 0.0
        self._cache: Dict[Tuple[str, str], Assessment] = {}

    @property
    def sut_info(self) -> dict:
        return self.sut.describe()

    def remaining(self) -> Optional[int]:
        return None if self.max_sims is None else self.max_sims - self.n_sims

    def run(self, scenario: Scenario, interventions: Optional[Interventions] = None) -> Episode:
        if self.max_sims is not None and self.n_sims >= self.max_sims:
            raise BudgetExceeded(f'simulation budget of {self.max_sims} exhausted')
        t0 = time.perf_counter()
        ep = self.sut.run(scenario, interventions)
        self.sim_seconds += time.perf_counter() - t0
        self.n_sims += 1
        return ep

    def evaluate(self, scenario: Scenario, interventions: Optional[Interventions] = None) -> Assessment:
        iv = interventions or Interventions()
        key = (scenario.fingerprint(), iv.key())
        hit = self._cache.get(key)
        if hit is not None:
            self.n_cache_hits += 1
            return hit
        a = assess(self.run(scenario, iv), self.criteria)
        self._cache[key] = a
        return a

    def evaluate_with(self, scenario: Scenario, intervention_names: Iterable[str]) -> Assessment:
        sc, iv = apply(scenario, intervention_names)
        return self.evaluate(sc, iv)

    def episode_with(self, scenario: Scenario, intervention_names: Iterable[str] = ()) -> Episode:
        sc, iv = apply(scenario, intervention_names)
        return self.run(sc, iv)

    def stats(self) -> dict:
        return {'simulations': self.n_sims, 'cache_hits': self.n_cache_hits,
                'sim_seconds': round(self.sim_seconds, 3)}
