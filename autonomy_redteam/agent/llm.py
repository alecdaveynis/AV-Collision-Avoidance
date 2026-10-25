"""Minimal LLM clients (standard library only) and a scripted fake for tests.

Configuration comes from environment variables only; nothing is ever hardcoded or written to disk:
  ANTHROPIC_API_KEY                    -> Anthropic Messages API
  OPENAI_API_KEY or REDTEAM_LLM_API_KEY-> OpenAI-compatible chat completions
  REDTEAM_LLM_PROVIDER                 'anthropic' | 'openai' (optional; inferred from which key is set)
  REDTEAM_LLM_MODEL                    model name (REQUIRED when a key is set: no default is guessed)
  REDTEAM_LLM_BASE_URL                 OpenAI-compatible base URL (default https://api.openai.com/v1)

NOTE: these two HTTP clients follow the providers' documented tool-calling formats but have NOT been exercised against
a live API in this repository's tests (no key was available); the orchestrator is tested with ``ScriptedLLM``.

Neutral message format used by the orchestrator:
  {'role': 'user', 'content': str}
  {'role': 'assistant', 'content': str, 'tool_calls': [ToolCall]}
  {'role': 'tool', 'tool_call_id': str, 'name': str, 'content': str}
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    text: str = ''
    tool_calls: List[ToolCall] = field(default_factory=list)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class LLMError(RuntimeError):
    pass


class LLMClient:
    name = 'llm'

    def chat(self, system: str, messages: List[dict], tools: List[dict], timeout: float = 60.0) -> LLMResponse:
        raise NotImplementedError


def _post(url: str, headers: Dict[str, str], body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'content-type': 'application/json', **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise LLMError(f'HTTP {e.code} from {url}: {e.read().decode(errors="replace")[:300]}') from e
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        raise LLMError(f'request to {url} failed: {e}') from e


class OpenAICompatClient(LLMClient):
    name = 'openai-compatible'

    def __init__(self, api_key: str, model: str, base_url: str = 'https://api.openai.com/v1'):
        self.api_key, self.model, self.base_url = api_key, model, base_url.rstrip('/')

    def chat(self, system, messages, tools, timeout=60.0) -> LLMResponse:
        msgs = [{'role': 'system', 'content': system}]
        for m in messages:
            if m['role'] == 'assistant':
                d = {'role': 'assistant', 'content': m.get('content') or None}
                if m.get('tool_calls'):
                    d['tool_calls'] = [{'id': c.id, 'type': 'function',
                                        'function': {'name': c.name, 'arguments': json.dumps(c.arguments)}}
                                       for c in m['tool_calls']]
                msgs.append(d)
            elif m['role'] == 'tool':
                msgs.append({'role': 'tool', 'tool_call_id': m['tool_call_id'], 'content': m['content']})
            else:
                msgs.append({'role': 'user', 'content': m['content']})
        body = {'model': self.model, 'messages': msgs, 'temperature': 0,
                'tools': [{'type': 'function', 'function': {'name': t['name'], 'description': t['description'],
                                                            'parameters': t['parameters']}} for t in tools]}
        out = _post(f'{self.base_url}/chat/completions', {'authorization': f'Bearer {self.api_key}'}, body, timeout)
        try:
            msg = out['choices'][0]['message']
            calls = []
            for c in msg.get('tool_calls') or []:
                try:
                    args = json.loads(c['function']['arguments'] or '{}')
                except json.JSONDecodeError:
                    args = {'__invalid_json__': c['function']['arguments']}
                calls.append(ToolCall(c['id'], c['function']['name'], args))
            u = out.get('usage', {})
            return LLMResponse(msg.get('content') or '', calls, u.get('prompt_tokens'), u.get('completion_tokens'))
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f'unexpected response shape: {e}') from e


class AnthropicClient(LLMClient):
    name = 'anthropic'

    def __init__(self, api_key: str, model: str, base_url: str = 'https://api.anthropic.com'):
        self.api_key, self.model, self.base_url = api_key, model, base_url.rstrip('/')

    def chat(self, system, messages, tools, timeout=60.0) -> LLMResponse:
        msgs: List[dict] = []
        for m in messages:
            if m['role'] == 'assistant':
                blocks = []
                if m.get('content'):
                    blocks.append({'type': 'text', 'text': m['content']})
                for c in m.get('tool_calls') or []:
                    blocks.append({'type': 'tool_use', 'id': c.id, 'name': c.name, 'input': c.arguments})
                msgs.append({'role': 'assistant', 'content': blocks})
            elif m['role'] == 'tool':
                block = {'type': 'tool_result', 'tool_use_id': m['tool_call_id'], 'content': m['content']}
                if msgs and msgs[-1]['role'] == 'user' and isinstance(msgs[-1]['content'], list):
                    msgs[-1]['content'].append(block)
                else:
                    msgs.append({'role': 'user', 'content': [block]})
            else:
                msgs.append({'role': 'user', 'content': m['content']})
        body = {'model': self.model, 'max_tokens': 2048, 'system': system, 'messages': msgs,
                'tools': [{'name': t['name'], 'description': t['description'], 'input_schema': t['parameters']} for t in tools]}
        out = _post(f'{self.base_url}/v1/messages', {'x-api-key': self.api_key, 'anthropic-version': '2023-06-01'}, body, timeout)
        try:
            text, calls = [], []
            for b in out['content']:
                if b['type'] == 'text':
                    text.append(b['text'])
                elif b['type'] == 'tool_use':
                    calls.append(ToolCall(b['id'], b['name'], b['input']))
            u = out.get('usage', {})
            return LLMResponse(''.join(text), calls, u.get('input_tokens'), u.get('output_tokens'))
        except (KeyError, TypeError) as e:
            raise LLMError(f'unexpected response shape: {e}') from e


class ScriptedLLM(LLMClient):
    """Deterministic stand-in used in tests: replays a list of responses (callables receive the message history)."""
    name = 'scripted'

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.seen: List[List[dict]] = []

    def chat(self, system, messages, tools, timeout=60.0) -> LLMResponse:
        self.seen.append([dict(m) for m in messages])
        if self.calls >= len(self.script):
            raise LLMError('script exhausted')
        r = self.script[self.calls]
        self.calls += 1
        return r(messages) if callable(r) else r


def client_from_env(env=None) -> Optional[LLMClient]:
    """Return a configured client, or None when no API key is present. Raises LLMError for a half-configured env."""
    env = os.environ if env is None else env
    provider = env.get('REDTEAM_LLM_PROVIDER', '').lower()
    akey, okey = env.get('ANTHROPIC_API_KEY'), env.get('OPENAI_API_KEY') or env.get('REDTEAM_LLM_API_KEY')
    if not provider:
        provider = 'anthropic' if akey else ('openai' if okey else '')
    if not provider:
        return None
    model = env.get('REDTEAM_LLM_MODEL')
    if provider == 'anthropic':
        if not akey:
            return None
        if not model:
            raise LLMError('REDTEAM_LLM_MODEL must be set when using an LLM provider (no default model is assumed)')
        return AnthropicClient(akey, model)
    if provider == 'openai':
        if not okey:
            return None
        if not model:
            raise LLMError('REDTEAM_LLM_MODEL must be set when using an LLM provider (no default model is assumed)')
        return OpenAICompatClient(okey, model, env.get('REDTEAM_LLM_BASE_URL', 'https://api.openai.com/v1'))
    raise LLMError(f'unknown REDTEAM_LLM_PROVIDER {provider!r}')
