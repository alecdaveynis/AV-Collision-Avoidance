"""Deterministic tools exposed to the investigation agent, with hard limits.

The agent can *ask* for experiments; every number it receives (collision status, gaps, failure rates) is computed
by the deterministic framework. There is intentionally no tool that edits files, the planner, ``config.py`` or the
metric definitions: the agent can only run scenarios inside the validated parameter ranges.

Limits (``Limits``) are enforced centrally in ``ToolBox.call``:
  max_tool_calls   number of tool invocations
  max_sims         simulations (also enforced by the Evaluator budget, so a tool cannot overspend)
  max_seconds      wall-clock time for the whole investigation
  max_tokens       LLM tokens (counted by the orchestrator from provider usage, or estimated)
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..diagnose import diagnose, wilson
from ..interventions import INTERVENTIONS
from ..minimize import derived_seeds, minimize
from ..runner import BudgetExceeded, Evaluator
from ..scenario import PARAMS, Scenario, ScenarioValidationError
from .schema import SchemaError, validate

HYPOTHESES = ('object_detection', 'distance_estimation', 'ttc_estimation', 'reaction_latency', 'planner_logic', 'other')
MAX_RESULT_CHARS = 7000


@dataclass
class Limits:
    max_tool_calls: int = 12
    max_sims: int = 600
    max_seconds: float = 120.0
    max_tokens: int = 60000


class LimitExceeded(RuntimeError):
    pass


_NUM = {'type': 'number'}
SCENARIO_FIELDS = {k: {'type': 'number', 'minimum': sp.low, 'maximum': sp.high} for k, sp in PARAMS.items()}
SCENARIO_FIELDS['seed'] = {'type': 'integer', 'minimum': 0, 'maximum': 2 ** 31 - 1}

TOOL_SPECS: List[dict] = [
    {'name': 'get_scenario', 'description': 'Parameters and deterministic outcome metrics of the failing scenario under investigation.',
     'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}},
    {'name': 'get_telemetry_window',
     'description': 'Rows of time-series telemetry (ground truth, perception, planner) of the scenario, at most 40 rows.',
     'parameters': {'type': 'object', 'additionalProperties': False,
                    'properties': {'t_start': {'type': 'number', 'minimum': 0, 'maximum': 30},
                                   't_end': {'type': 'number', 'minimum': 0, 'maximum': 30}},
                    'required': ['t_start', 't_end']}},
    {'name': 'minimize_scenario',
     'description': 'Run the failure minimiser. Returns the minimised scenario, necessary faults and reproducibility.',
     'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}},
    {'name': 'run_full_diagnosis',
     'description': 'Run the full factorial counterfactual diagnosis (all component repairs, reference policy). '
                    'Returns Shapley shares and hypothesis statuses. Costs 2^k * n_seeds simulations.',
     'parameters': {'type': 'object', 'additionalProperties': False,
                    'properties': {'n_seeds': {'type': 'integer', 'minimum': 2, 'maximum': 16}}}},
    {'name': 'run_counterfactual',
     'description': 'Replay the scenario with a set of named interventions over n_seeds noise seeds; returns failure rate with '
                    'a 90% interval and mean metrics. Interventions: ' + ', '.join(sorted(INTERVENTIONS)),
     'parameters': {'type': 'object', 'additionalProperties': False,
                    'properties': {'interventions': {'type': 'array', 'minItems': 1, 'maxItems': 8,
                                                     'items': {'type': 'string', 'enum': sorted(INTERVENTIONS)}},
                                   'n_seeds': {'type': 'integer', 'minimum': 1, 'maximum': 16}},
                    'required': ['interventions']}},
    {'name': 'run_scenario_variant',
     'description': 'Replay a modified scenario (any subset of the schema parameters, within their bounds) over n_seeds noise '
                    'seeds; returns failure rate and mean metrics. Use to probe severity or neighbouring conditions.',
     'parameters': {'type': 'object', 'additionalProperties': False,
                    'properties': {'overrides': {'type': 'object', 'additionalProperties': False,
                                                 'properties': SCENARIO_FIELDS},
                                   'n_seeds': {'type': 'integer', 'minimum': 1, 'maximum': 16}},
                    'required': ['overrides']}},
    {'name': 'finish',
     'description': 'Submit the final evidence-backed report. Every claim must cite run ids returned by earlier tools. '
                    'Do not state metrics you have not received from a tool.',
     'parameters': {'type': 'object', 'additionalProperties': False,
                    'properties': {
                        'summary': {'type': 'string', 'maxLength': 1500},
                        'hypotheses': {'type': 'array', 'maxItems': 6, 'items': {
                            'type': 'object', 'additionalProperties': False,
                            'properties': {'component': {'type': 'string', 'enum': list(HYPOTHESES)},
                                           'claim': {'type': 'string', 'maxLength': 600},
                                           'confidence': {'type': 'string', 'enum': ['low', 'medium', 'high']},
                                           'evidence_runs': {'type': 'array', 'maxItems': 10, 'items': {'type': 'string'}}},
                            'required': ['component', 'claim', 'confidence', 'evidence_runs']}},
                        'suggested_scenarios': {'type': 'array', 'maxItems': 5, 'items': {
                            'type': 'object', 'additionalProperties': False,
                            'properties': {'overrides': {'type': 'object', 'additionalProperties': False,
                                                         'properties': SCENARIO_FIELDS},
                                           'rationale': {'type': 'string', 'maxLength': 400}},
                            'required': ['overrides', 'rationale']}}},
                    'required': ['summary', 'hypotheses', 'suggested_scenarios']}},
]
TOOL_SCHEMAS = {t['name']: t['parameters'] for t in TOOL_SPECS}


def _short(x, nd=3):
    if isinstance(x, float):
        return None if not math.isfinite(x) else round(x, nd)
    return x


class ToolBox:
    """Executes validated tool calls against the deterministic framework and tracks usage."""

    def __init__(self, ev: Evaluator, scenario: Scenario, limits: Limits = Limits(), clock: Callable[[], float] = time.monotonic):
        self.ev, self.scenario, self.limits, self.clock = ev, scenario, limits, clock
        self.t0 = clock()
        self.calls = 0
        self.tokens = 0
        self.tokens_estimated = False
        self.sim_start = ev.n_sims
        self.runs: Dict[str, dict] = {}
        self.log: List[dict] = []
        self.finished: Optional[dict] = None
        self.minimization = None
        self.diagnosis = None
        self._ep = None
        ev.max_sims = self.sim_start + limits.max_sims
        self._tools: Dict[str, Callable[[dict], dict]] = {
            'get_scenario': self._get_scenario, 'get_telemetry_window': self._telemetry,
            'minimize_scenario': self._minimize, 'run_full_diagnosis': self._diagnosis,
            'run_counterfactual': self._counterfactual, 'run_scenario_variant': self._variant,
            'finish': self._finish}

    # -- limits -----------------------------------------------------------------------------------------
    @property
    def sims_used(self) -> int:
        return self.ev.n_sims - self.sim_start

    def elapsed(self) -> float:
        return self.clock() - self.t0

    def add_tokens(self, n: int, estimated: bool = False) -> None:
        self.tokens += int(n)
        self.tokens_estimated |= estimated

    def check_limits(self) -> None:
        L = self.limits
        if self.calls >= L.max_tool_calls:
            raise LimitExceeded(f'tool-call limit {L.max_tool_calls} reached')
        if self.elapsed() > L.max_seconds:
            raise LimitExceeded(f'time limit {L.max_seconds:.0f}s reached')
        if self.tokens > L.max_tokens:
            raise LimitExceeded(f'token limit {L.max_tokens} reached')
        if self.sims_used >= L.max_sims:
            raise LimitExceeded(f'simulation limit {L.max_sims} reached')

    def usage(self) -> dict:
        return {'tool_calls': self.calls, 'simulations': self.sims_used, 'seconds': round(self.elapsed(), 2),
                'tokens': self.tokens, 'tokens_estimated': self.tokens_estimated,
                'limits': {'max_tool_calls': self.limits.max_tool_calls, 'max_sims': self.limits.max_sims,
                           'max_seconds': self.limits.max_seconds, 'max_tokens': self.limits.max_tokens}}

    # -- dispatch -----------------------------------------------------------------------------------------------
    def call(self, name: str, args: Any) -> dict:
        """Validate and run one tool; never raises for bad input (returns {'error': ...}); raises LimitExceeded."""
        self.check_limits()
        self.calls += 1
        entry = {'tool': name, 'args': args}
        try:
            if name not in self._tools:
                raise SchemaError(f'unknown tool {name!r}')
            validate(args, TOOL_SCHEMAS[name])
            out = self._tools[name](args)
        except (SchemaError, ScenarioValidationError, ValueError, KeyError) as e:
            out = {'error': f'{type(e).__name__}: {e}'}
        except BudgetExceeded as e:
            out = {'error': f'simulation budget exhausted: {e}'}
        entry['result'] = out
        self.log.append(entry)
        return out

    def render(self, result: dict) -> str:
        s = json.dumps(result, separators=(',', ':'), default=str)
        return s if len(s) <= MAX_RESULT_CHARS else s[:MAX_RESULT_CHARS] + '...[truncated]'

    def _new_run(self, kind: str, payload: dict) -> str:
        rid = f'R{len(self.runs) + 1}'
        self.runs[rid] = {'kind': kind, **payload}
        return rid

    # -- tools ---------------------------------------------------------------------------------------------------------------
    def _get_scenario(self, args) -> dict:
        a = self.ev.evaluate(self.scenario)
        d = a.to_dict()
        keys = ('failed', 'failure_kinds', 'primary_kind', 'collision', 'impact_speed', 'min_gap', 'min_ttc',
                'brake_reaction_time', 'missed_hazards', 'unnecessary_braking_s', 'states_visited', 'collision_class',
                'lead_lost_steps', 'max_range_error', 'perception_anomaly')
        rid = self._new_run('original', {'scenario': self.scenario.to_dict()})
        return {'run_id': rid, 'scenario': self.scenario.to_dict(), 'active_faults': self.scenario.active_faults(),
                'planner_mode': self.ev.sut_info.get('planner_mode'),
                'outcome': {k: _short(d[k]) for k in keys}}

    def _telemetry(self, args) -> dict:
        if self._ep is None:
            self._ep = self.ev.run(self.scenario)
        lo, hi = args['t_start'], args['t_end']
        rows = [r for r in self._ep.records if lo <= r.t <= hi][:40]
        return {'columns': ['t', 'gt_gap', 'est_gap', 'ego_v', 'lead_v', 'gt_ttc', 'est_ttc', 'state', 'cmd_decel', 'lead_detected'],
                'rows': [[round(r.t, 1), _short(r.gt.gap, 2), _short(r.perception.est_gap, 2), _short(r.gt.ego_v, 2),
                          _short(r.gt.lead_v, 2), _short(r.gt.ttc, 2), _short(r.perception.est_ttc, 2), r.planner.state,
                          _short(r.planner.cmd_decel, 2), r.perception.lead_detected] for r in rows]}

    def _minimize(self, args) -> dict:
        res = minimize(self.ev, self.scenario, seeds=3)
        self.minimization = res
        rid = self._new_run('minimize', {'minimized': res.minimized.to_dict(), 'necessary_faults': res.necessary_faults})
        return {'run_id': rid, 'minimized_scenario': res.minimized.to_dict(), 'necessary_faults': res.necessary_faults,
                'removed_faults': res.removed_faults, 'reproducibility': res.reproducible,
                'changes': [c.to_dict() for c in res.changes][:20]}

    def _diagnosis(self, args) -> dict:
        target = self.minimization.minimized if self.minimization is not None else self.scenario
        d = diagnose(self.ev, target, n_seeds=args.get('n_seeds', 8))
        self.diagnosis = d
        rid = self._new_run('diagnosis', {'hypotheses': [(h['component'], h['status']) for h in d.hypotheses]})
        return {'run_id': rid, 'scenario_used': 'minimized' if self.minimization is not None else 'original',
                'failure_rate_original': d.variants[0].rate,
                'hypotheses': [{'component': h['component'], 'status': h['status'], 'share': h['share'],
                                'evidence': h['evidence']} for h in d.ranked_hypotheses()],
                'shapley': d.shapley, 'planner_logic_check': d.planner, 'notes': d.notes, 'simulations': d.sims}

    def _replicate(self, sc: Scenario, names, n: int) -> dict:
        from ..interventions import apply
        outs, assess = [], []
        for sd in derived_seeds(sc.seed, n):
            s2, iv = apply(sc.replace(seed=sd), names)
            a = self.ev.evaluate(s2, iv)
            assess.append(a)
        kind = self.scenario_kind()
        k = sum(kind in a.failure_kinds for a in assess)
        lo, hi = wilson(k, len(assess))
        return {'failure_kind': kind, 'n_seeds': len(assess), 'failures': k, 'failure_rate': round(k / len(assess), 4),
                'ci90': [round(lo, 4), round(hi, 4)],
                'mean_min_gap': round(float(np.mean([a.min_gap for a in assess])), 3),
                'collision_rate': round(float(np.mean([a.collision for a in assess])), 4),
                'mean_impact_speed_when_colliding': round(float(np.mean([a.impact_speed for a in assess if a.collision] or [0.0])), 3),
                'any_failure_kinds': sorted({k2 for a in assess for k2 in a.failure_kinds})}

    def scenario_kind(self) -> str:
        return self.ev.evaluate(self.scenario).primary_kind or 'collision'

    def _counterfactual(self, args) -> dict:
        names = list(dict.fromkeys(args['interventions']))
        n = args.get('n_seeds', 6)
        sc = self.minimization.minimized if self.minimization is not None else self.scenario
        if self.ev.remaining() is not None and self.ev.remaining() < n:
            raise BudgetExceeded('not enough simulations left for this request')
        res = self._replicate(sc, names, n)
        res['interventions'] = names
        res['scenario_used'] = 'minimized' if self.minimization is not None else 'original'
        res['run_id'] = self._new_run('counterfactual', {'interventions': names, 'result': res})
        return res

    def _variant(self, args) -> dict:
        n = args.get('n_seeds', 6)
        sc = self.scenario.replace(**args['overrides']).validate()
        if self.ev.remaining() is not None and self.ev.remaining() < n:
            raise BudgetExceeded('not enough simulations left for this request')
        res = self._replicate(sc, [], n)
        res['overrides'] = args['overrides']
        res['run_id'] = self._new_run('variant', {'overrides': args['overrides'], 'result': res})
        return res

    def _finish(self, args) -> dict:
        problems = []
        for h in args['hypotheses']:
            for r in h['evidence_runs']:
                if r not in self.runs:
                    problems.append(f'unknown run id {r!r}')
        validated = []
        for s in args['suggested_scenarios']:
            try:
                validated.append({'scenario': self.scenario.replace(**s['overrides']).validate().to_dict(),
                                  'rationale': s['rationale']})
            except ScenarioValidationError as e:
                problems.append(f'suggested scenario invalid: {e}')
        if problems:
            return {'error': 'report rejected: ' + '; '.join(problems)}
        self.finished = {'summary': args['summary'], 'hypotheses': args['hypotheses'], 'suggested_scenarios': validated}
        return {'accepted': True}
