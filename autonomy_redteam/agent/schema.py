"""A tiny JSON-schema subset validator for tool-call arguments (no external dependency).

Supports: type (object/array/string/integer/number/boolean), enum, minimum/maximum, minItems/maxItems, items,
properties/required/additionalProperties=false. Anything outside the schema is rejected, never coerced.
"""
from __future__ import annotations

import math
from typing import Any, List


class SchemaError(ValueError):
    pass


def validate(value: Any, schema: dict, path: str = '$') -> None:
    errs: List[str] = []
    _check(value, schema, path, errs)
    if errs:
        raise SchemaError('; '.join(errs))


def _check(v, s, path, errs):
    t = s.get('type')
    if t == 'object':
        if not isinstance(v, dict):
            errs.append(f'{path}: expected object')
            return
        props = s.get('properties', {})
        for k in s.get('required', []):
            if k not in v:
                errs.append(f'{path}.{k}: required')
        if s.get('additionalProperties') is False:
            for k in v:
                if k not in props:
                    errs.append(f'{path}.{k}: unknown field')
        for k, sub in props.items():
            if k in v:
                _check(v[k], sub, f'{path}.{k}', errs)
    elif t == 'array':
        if not isinstance(v, list):
            errs.append(f'{path}: expected array')
            return
        if 'minItems' in s and len(v) < s['minItems']:
            errs.append(f'{path}: at least {s["minItems"]} items')
        if 'maxItems' in s and len(v) > s['maxItems']:
            errs.append(f'{path}: at most {s["maxItems"]} items')
        for i, x in enumerate(v):
            _check(x, s.get('items', {}), f'{path}[{i}]', errs)
    elif t == 'string':
        if not isinstance(v, str):
            errs.append(f'{path}: expected string')
        elif 'enum' in s and v not in s['enum']:
            errs.append(f'{path}: must be one of {s["enum"]}')
        elif 'maxLength' in s and len(v) > s['maxLength']:
            errs.append(f'{path}: longer than {s["maxLength"]}')
    elif t in ('integer', 'number'):
        ok = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        if t == 'integer':
            ok = ok and float(v).is_integer()
        if not ok:
            errs.append(f'{path}: expected {t}')
            return
        if 'minimum' in s and v < s['minimum']:
            errs.append(f'{path}: below minimum {s["minimum"]}')
        if 'maximum' in s and v > s['maximum']:
            errs.append(f'{path}: above maximum {s["maximum"]}')
    elif t == 'boolean':
        if not isinstance(v, bool):
            errs.append(f'{path}: expected boolean')
