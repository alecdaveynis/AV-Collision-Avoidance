"""Lightweight investigation orchestrator: a bounded tool-calling loop plus a deterministic fallback.

LLM mode        the model chooses which tools to call (hypothesis -> experiment -> comparison) and writes the report
                through the validated ``finish`` tool. It never sees raw simulator internals, cannot edit code or the
                planner, and cannot change metric definitions.
Deterministic   the same tools, called in a fixed order by ``DeterministicInvestigator`` (no API key, no network).
                Used when no key is configured and in CI.

Both produce an ``AgentReport``. The deterministic investigator additionally *runs* each suggested scenario so the report
states, from simulation, whether a suggestion really fails; in LLM mode suggestions are validated and then run the same way.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..runner import Evaluator
from ..scenario import Scenario
from .llm import LLMClient, LLMError, ToolCall
from .tools import TOOL_SPECS, LimitExceeded, Limits, ToolBox

SYSTEM_PROMPT = """You are a safety-test investigator for an autonomous-driving collision-avoidance planner.
You are given ONE failing scenario. Use the provided tools to investigate WHY it failed, then call `finish`.

Rules:
- Collision status, distances, TTC and every failure statistic come ONLY from tool results. Never invent or estimate numbers.
- Form explicit hypotheses (object detection, distance estimation, TTC estimation, reaction latency, planner logic), choose
  controlled experiments (run_counterfactual / run_scenario_variant / run_full_diagnosis), and compare the outcomes.
- A single intervention that flips the outcome is not proof of a root cause; components interact. Report uncertainty
  (failure rates come with intervals over noise seeds) and use 'low'/'medium'/'high' confidence honestly.
- Cite the run ids (R1, R2, ...) returned by tools as evidence for each claim.
- You cannot change the planner or the metrics. Suggest follow-up scenarios only as parameter overrides within the schema bounds.
- You have a strict budget of tool calls, simulations, tokens and time; stop and call `finish` before it runs out.
"""


@dataclass
class AgentReport:
    mode: str                       # 'llm:<client>' or 'deterministic'
    summary: str
    hypotheses: List[dict]
    suggested_scenarios: List[dict]
    suggestion_results: List[dict]
    usage: dict
    tool_log: List[dict]
    stopped_reason: str = 'finished'
    minimized_scenario: Optional[dict] = None

    def to_dict(self) -> dict:
        return {'mode': self.mode, 'summary': self.summary, 'hypotheses': self.hypotheses,
                'suggested_scenarios': self.suggested_scenarios, 'suggestion_results': self.suggestion_results,
                'usage': self.usage, 'stopped_reason': self.stopped_reason,
                'tool_log': [{'tool': e['tool'], 'args': e['args'], 'result_keys': sorted(e['result'])
                              if isinstance(e['result'], dict) else None} for e in self.tool_log]}

    def markdown(self) -> str:
        L = [f'_Mode: {self.mode}; stopped because: {self.stopped_reason}; usage: {json.dumps(self.usage["tool_calls"])} tool calls, '
             f'{self.usage["simulations"]} simulations, {self.usage["tokens"]} tokens'
             f'{" (estimated)" if self.usage.get("tokens_estimated") else ""}._\n', self.summary + '\n']
        for h in self.hypotheses:
            L.append(f'- **{h["component"]}** ({h["confidence"]} confidence): {h["claim"]}  evidence: {", ".join(h["evidence_runs"]) or "none cited"}')
        if self.suggestion_results:
            L.append('\nSuggested follow-up scenarios, replayed deterministically:\n')
            L.append('| rationale | outcome | failure kinds | min gap (m) |\n|---|---|---|---|')
            for s in self.suggestion_results:
                L.append(f'| {s["rationale"]} | {"FAILS" if s["failed"] else "passes"} | {", ".join(s["failure_kinds"]) or "-"} | {s["min_gap"]:.2f} |')
        return '\n'.join(L)


def _run_suggestions(ev: Evaluator, suggestions: List[dict]) -> List[dict]:
    out = []
    for s in suggestions:
        try:
            a = ev.evaluate(Scenario.from_dict(s['scenario']))
        except Exception as e:                       # budget exhausted etc.: say so instead of guessing
            out.append({'rationale': s['rationale'], 'failed': False, 'failure_kinds': [], 'min_gap': float('nan'),
                        'note': f'not run: {e}'})
            continue
        out.append({'rationale': s['rationale'], 'failed': a.failed, 'failure_kinds': a.failure_kinds, 'min_gap': a.min_gap,
                    'scenario': s['scenario']})
    return out


class DeterministicInvestigator:
    """Fixed investigation script over the same ToolBox (no LLM)."""

    def run(self, tb: ToolBox) -> AgentReport:
        reason = 'finished'
        try:
            sc = tb.call('get_scenario', {})
            tb.call('get_telemetry_window', {'t_start': 0.0, 't_end': 4.0})
            m = tb.call('minimize_scenario', {})
            d = tb.call('run_full_diagnosis', {'n_seeds': 8})
            runs = [r for r in ('R1', 'R2') if r in tb.runs]
            hyps = []
            sup = [h for h in d.get('hypotheses', []) if h['status'] in ('supported', 'weak')]
            rid_d = d.get('run_id', '')
            for h in sup[:3]:
                hyps.append({'component': h['component'], 'claim': f'{h["status"]}: ' + '; '.join(h['evidence'][:2]),
                             'confidence': 'medium' if h['status'] == 'supported' else 'low',
                             'evidence_runs': [rid_d] if rid_d else []})
            follow = []
            ov_base = m.get('minimized_scenario', sc.get('scenario', {}))
            # follow-up 1: does the failure get worse if the supported perturbation is made more severe?
            faults = m.get('necessary_faults', [])
            sugg = []
            if 'detection_loss' in faults:
                dur = ov_base.get('detection_loss_duration_s', 0.0)
                sugg.append(({'detection_loss_duration_s': min(4.0, round(dur * 2 + 0.25, 2))},
                             'longer detection loss than the minimised case'))
            if 'perception_latency' in faults:
                sugg.append(({'perception_latency_s': min(0.6, round(ov_base.get('perception_latency_s', 0.0) + 0.1, 2))},
                             'more perception latency than the minimised case'))
            sugg.append(({'lead_decel': min(9.0, round(ov_base.get('lead_decel', 0.0) + 2.0, 2))},
                         'harder lead braking than the minimised case'))
            sugg.append(({'ego_speed': max(6.0, round(ov_base.get('ego_speed', 14.0) - 3.0, 2))},
                         'slower ego: how far below the failing speed does the failure persist?'))
            used = []
            for ov, why in sugg:
                full = dict(ov_base)
                full.update(ov)
                full = {k: v for k, v in full.items() if k in SCENARIO_KEYS}
                v = tb.call('run_scenario_variant', {'overrides': full, 'n_seeds': 6})
                if 'error' in v:
                    continue
                used.append((full, why, v))
                follow.append(v['run_id'])
            summary = self._summary(sc, m, d, used)
            suggestions = [{'overrides': full, 'rationale': f'{why} (failure rate {v["failure_rate"]:.2f} over {v["n_seeds"]} seeds)'}
                           for full, why, v in used]
            tb.call('finish', {'summary': summary[:1500], 'hypotheses': hyps,
                               'suggested_scenarios': suggestions[:5]})
        except LimitExceeded as e:
            reason = str(e)
        fin = tb.finished or {'summary': 'Investigation stopped before a report was produced.', 'hypotheses': [],
                              'suggested_scenarios': []}
        return AgentReport('deterministic', fin['summary'], fin['hypotheses'], fin['suggested_scenarios'],
                           _run_suggestions(tb.ev, fin['suggested_scenarios']), tb.usage(), tb.log, reason,
                           tb.minimization.minimized.to_dict() if tb.minimization else None)

    @staticmethod
    def _summary(sc, m, d, used) -> str:
        top = [h for h in d.get('hypotheses', []) if h['status'] == 'supported']
        s = (f'Failure kind {sc["outcome"]["primary_kind"]}; necessary perturbations after minimisation: '
             f'{", ".join(m.get("necessary_faults", [])) or "none"}. ')
        s += ('Supported hypotheses: ' + ', '.join(h['component'] for h in top) + '. ') if top else 'No hypothesis reached the support threshold. '
        s += f'Original failure rate over noise seeds: {d.get("failure_rate_original", float("nan")):.2f}. '
        s += f'{len(used)} neighbouring scenarios were replayed (see table).'
        return s


from ..scenario import PARAMS as _P   # noqa: E402
SCENARIO_KEYS = set(_P) | {'seed'}


class LLMInvestigator:
    def __init__(self, client: LLMClient):
        self.client = client

    def run(self, tb: ToolBox) -> AgentReport:
        sc = tb.scenario
        messages: List[dict] = [{'role': 'user', 'content':
                                 'Investigate this failing scenario and call `finish` with an evidence-backed report.\n'
                                 f'Scenario: {sc.describe()}\nLimits: {json.dumps(tb.usage()["limits"])}'}]
        reason = 'finished'
        try:
            while tb.finished is None:
                tb.check_limits()
                try:
                    resp = self.client.chat(SYSTEM_PROMPT, messages, TOOL_SPECS, timeout=max(5.0, min(60.0, tb.limits.max_seconds - tb.elapsed())))
                except LLMError as e:
                    reason = f'LLM error: {e}'
                    break
                if resp.input_tokens is not None or resp.output_tokens is not None:
                    tb.add_tokens((resp.input_tokens or 0) + (resp.output_tokens or 0))
                else:
                    tb.add_tokens(sum(len(str(m.get('content', ''))) for m in messages) // 4, estimated=True)
                messages.append({'role': 'assistant', 'content': resp.text, 'tool_calls': resp.tool_calls})
                if not resp.tool_calls:
                    reason = 'model produced no tool call before finishing'
                    break
                for c in resp.tool_calls:
                    args = c.arguments if isinstance(c.arguments, dict) else {}
                    try:
                        out = tb.call(c.name, args)
                    except LimitExceeded:
                        raise
                    messages.append({'role': 'tool', 'tool_call_id': c.id, 'name': c.name, 'content': tb.render(out)})
                    if tb.finished is not None:
                        break
        except LimitExceeded as e:
            reason = str(e)
        fin = tb.finished or {'summary': f'No accepted report ({reason}).', 'hypotheses': [], 'suggested_scenarios': []}
        return AgentReport(f'llm:{self.client.name}', fin['summary'], fin['hypotheses'], fin['suggested_scenarios'],
                           _run_suggestions(tb.ev, fin['suggested_scenarios']), tb.usage(), tb.log, reason,
                           tb.minimization.minimized.to_dict() if tb.minimization else None)


def investigate_with_agent(ev: Evaluator, scenario: Scenario, client: Optional[LLMClient] = None,
                           limits: Limits = Limits()) -> AgentReport:
    prev = ev.max_sims
    tb = ToolBox(ev, scenario, limits)
    try:
        return (LLMInvestigator(client) if client is not None else DeterministicInvestigator()).run(tb)
    finally:
        ev.max_sims = prev


def run_agent_cli(args, make_evaluator) -> int:
    from .llm import client_from_env
    ev = make_evaluator(args)
    sc = Scenario.load(args.scenario)
    client = None if args.no_llm else client_from_env()
    limits = Limits(args.max_tool_calls, args.max_sims, args.max_seconds, args.max_tokens)
    if client is None:
        print('no LLM API key configured (or --no-llm): using the deterministic investigator')
    else:
        print(f'using LLM client {client.name}; limits {limits}')
    rep = investigate_with_agent(ev, sc, client, limits)
    out = Path(args.out or Path(args.scenario).with_suffix('').as_posix() + '_agent')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'agent_report.json').write_text(json.dumps(rep.to_dict(), indent=2))
    (out / 'agent_report.md').write_text(rep.markdown() + '\n')
    print(rep.markdown())
    print(f'\nsaved: {out}')
    return 0
