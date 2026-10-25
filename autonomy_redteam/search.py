"""Scenario search (falsification): random baseline and CMA-ES, under an explicit simulation budget.

Objective (minimised): ``Assessment.objective`` - the ground-truth minimum gap in metres, or the negative
impact speed for a collision. Collisions that are physically unavoidable (or that a competent reference
policy also cannot avoid) are pushed to a plateau so the search does not chase impossible cases.

Both methods draw every random number from generators seeded by ``seed``; a search with the same
seed, budget and SUT reproduces exactly. The simulator noise seed of each candidate is part of the
candidate (drawn from the search RNG), so every reported failure replays from its saved scenario JSON.

CMA-ES is a compact implementation of Hansen's (mu/mu_w, lambda)-CMA-ES (no external dependency) on the unit
box of all scenario parameters, with restarts from random points once a run converges or stagnates, so the
budget is not spent re-finding one failure.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .metrics import Assessment
from .runner import BudgetExceeded, Evaluator
from .scenario import Scenario, ScenarioGenerator

METHODS = ('random', 'cmaes')


@dataclass
class Evaluation:
    index: int                    # 1-based simulation index within the search
    scenario: Scenario
    assessment: Assessment

    def to_dict(self) -> dict:
        a = self.assessment
        return {'index': self.index, 'scenario': self.scenario.to_dict(), 'failed': a.failed,
                'kinds': a.failure_kinds, 'objective': a.objective, 'min_gap': a.min_gap,
                'impact_speed': a.impact_speed, 'collision_class': a.collision_class, 'severity': a.severity}


@dataclass
class SearchResult:
    method: str
    seed: int
    budget: int
    evaluations: List[Evaluation] = field(default_factory=list)
    wall_seconds: float = 0.0
    sut: Dict = field(default_factory=dict)

    @property
    def sims_used(self) -> int:
        return len(self.evaluations)

    @property
    def failures(self) -> List[Evaluation]:
        return [e for e in self.evaluations if e.assessment.failed]

    @property
    def first_failure_sim(self) -> Optional[int]:
        f = self.failures
        return f[0].index if f else None

    def to_dict(self) -> dict:
        return {'method': self.method, 'seed': self.seed, 'budget': self.budget, 'sims_used': self.sims_used,
                'n_failures': len(self.failures), 'first_failure_sim': self.first_failure_sim,
                'wall_seconds': round(self.wall_seconds, 3), 'sut': self.sut,
                'evaluations': [e.to_dict() for e in self.evaluations]}


# --- CMA-ES --------------------------------------------------------------------------------------
class CMAES:
    """Minimal CMA-ES on [0, 1]^n (candidates are clipped into the box before evaluation)."""

    def __init__(self, x0: np.ndarray, sigma0: float, rng: np.random.Generator, popsize: Optional[int] = None):
        n = len(x0)
        self.n, self.rng = n, rng
        self.lam = popsize or 4 + int(3 * math.log(n))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.mean = np.array(x0, float)
        self.sigma = sigma0
        self.pc, self.ps = np.zeros(n), np.zeros(n)
        self.C, self.B, self.D = np.eye(n), np.eye(n), np.ones(n)
        self.gen = 0
        self._z = None

    def ask(self) -> List[np.ndarray]:
        z = self.rng.standard_normal((self.lam, self.n))
        y = (z * self.D) @ self.B.T
        self._y = y
        return [self.mean + self.sigma * yi for yi in y]

    def tell(self, xs: List[np.ndarray], fs: List[float]) -> None:
        order = np.argsort(fs)[:self.mu]
        y_sel = self._y[order]
        yw = self.w @ y_sel
        old_mean = self.mean
        self.mean = old_mean + self.sigma * yw
        invsqrtC = self.B @ np.diag(1.0 / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * (invsqrtC @ yw)
        hsig = (np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) / self.chiN
                < 1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        rank_mu = sum(wi * np.outer(yi, yi) for wi, yi in zip(self.w, y_sel))
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * rank_mu)
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.sigma = float(min(self.sigma, 1.0))
        self.C = (self.C + self.C.T) / 2
        d2, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(d2, 1e-20))
        self.gen += 1


# --- search drivers ------------------------------------------------------------------------------
def _record(res: SearchResult, ev: Evaluator, sc: Scenario) -> Evaluation:
    a = ev.evaluate(sc)
    e = Evaluation(len(res.evaluations) + 1, sc, a)
    res.evaluations.append(e)
    return e


def random_search(ev: Evaluator, budget: int, seed: int, bounds=None, fault_prob: float = 0.5) -> SearchResult:
    res = SearchResult('random', seed, budget, sut=ev.sut_info)
    gen = ScenarioGenerator(seed, bounds, fault_prob)
    t0 = time.perf_counter()
    while res.sims_used < budget:
        sc = gen.sample('random')
        sc = sc.replace(name=f'random-{res.sims_used + 1:04d}')
        n0 = ev.n_sims
        _record(res, ev, sc)
        if ev.n_sims == n0:                 # cache hit (duplicate): not a simulation, keep drawing
            res.evaluations.pop()
    res.wall_seconds = time.perf_counter() - t0
    return res


def cmaes_search(ev: Evaluator, budget: int, seed: int, bounds=None, sigma0: float = 0.3,
                 popsize: int = 12, fault_prob: float = 0.5) -> SearchResult:
    res = SearchResult('cmaes', seed, budget, sut=ev.sut_info)
    gen = ScenarioGenerator(seed, bounds, fault_prob)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xC3A]))
    n = len(ScenarioGenerator.vector_keys())
    t0 = time.perf_counter()
    restarts = 0
    while res.sims_used < budget:
        es = CMAES(rng.uniform(0.0, 1.0, n), sigma0, rng, popsize)     # every (re)start from a random point
        best, stall = math.inf, 0
        while res.sims_used < budget:
            xs = es.ask()
            fs = []
            for x in xs:
                if res.sims_used >= budget:
                    break
                sc = gen.from_unit(x, seed=int(rng.integers(0, 2 ** 31 - 1)),
                                   name=f'cmaes-{res.sims_used + 1:04d}')
                fs.append(_record(res, ev, sc).assessment.objective)
            if len(fs) < len(xs):
                break
            es.tell(xs, fs)
            gbest = min(fs)
            stall = 0 if gbest < best - 1e-3 else stall + 1
            best = min(best, gbest)
            if es.sigma < 1e-3 or stall >= 8:
                break
        restarts += 1
    res.wall_seconds = time.perf_counter() - t0
    return res


def run_search(method: str, ev: Evaluator, budget: int, seed: int, bounds=None, fault_prob: float = 0.5) -> SearchResult:
    if method not in METHODS:
        raise ValueError(f'method must be one of {METHODS}')
    prev = ev.max_sims
    ev.max_sims = ev.n_sims + budget             # hard cap: a driver cannot overspend
    fn = random_search if method == 'random' else cmaes_search
    try:
        return fn(ev, budget, seed, bounds, fault_prob=fault_prob)
    finally:
        ev.max_sims = prev
