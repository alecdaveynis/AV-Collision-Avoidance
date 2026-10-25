"""Failure equivalence and ranking.

Two failing scenarios are the *same finding* when they have the same signature:

    (failure kind, necessary fault set, lead regime)

* failure kind      : primary kind from ``metrics.assess``
* necessary faults  : the 1-minimal fault set found by ddmin (``Minimizer.ddmin_faults``), so a scenario carrying five
                      irrelevant perturbations and one carrying none are the same finding if neither needs them
* lead regime       : 'lead_braking' (lead_decel > 0), 'parked' (lead speed ~ 0, no braking) or 'steady'

Numeric parameters (speeds, gaps, ...) are deliberately *not* part of the signature: the failure region is continuous,
and counting every point in it as a new discovery would inflate the number of "unique" failures.
Signatures are conservative in the other direction too: two distinct mechanisms inside one regime with the same fault set
are merged. Both limits are stated in the README.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .runner import Evaluator
from .minimize import Minimizer
from .scenario import Scenario


def lead_regime(s: Scenario) -> str:
    if s.lead_decel > 0:
        return 'lead_braking'
    return 'parked' if s.lead_speed < 0.5 else 'steady'


def signature(s: Scenario, kind: str, necessary_faults: Iterable[str]) -> Tuple:
    return (kind, tuple(sorted(necessary_faults)), lead_regime(s))


def sig_str(sig: Tuple) -> str:
    return f'{sig[0]} | faults: {", ".join(sig[1]) or "none"} | {sig[2]}'


@dataclass
class Finding:
    signature: Tuple
    kind: str
    necessary_faults: Tuple[str, ...]
    regime: str
    representative: Scenario
    severity: float
    members: int = 1
    first_index: int = 0
    member_indices: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {'signature': sig_str(self.signature), 'kind': self.kind,
                'necessary_faults': list(self.necessary_faults), 'regime': self.regime,
                'severity': round(self.severity, 3), 'members': self.members, 'first_found_at_sim': self.first_index,
                'representative': self.representative.to_dict()}


def necessary_faults(ev: Evaluator, s: Scenario, kind: str, memo: Optional[Dict] = None) -> Optional[Tuple[str, ...]]:
    """1-minimal necessary fault set of a failing scenario (None if it does not reproduce)."""
    key = (s.fingerprint(), kind)
    if memo is not None and key in memo:
        return memo[key]
    m = Minimizer(ev, kind, seeds=1)
    try:
        out = tuple(sorted(m.ddmin_faults(s)))
    except ValueError:
        out = None
    if memo is not None:
        memo[key] = out
    return out


def group_failures(ev: Evaluator, evaluations, memo: Optional[Dict] = None, max_failures: int = 400) -> List[Finding]:
    """Group the failing evaluations of a search into ranked, de-duplicated findings."""
    groups: Dict[Tuple, Finding] = {}
    n = 0
    for e in evaluations:
        a = e.assessment
        if not a.failed:
            continue
        n += 1
        if n > max_failures:
            break
        kind = a.primary_kind
        nf = necessary_faults(ev, e.scenario, kind, memo)
        if nf is None:
            continue
        sig = signature(e.scenario, kind, nf)
        g = groups.get(sig)
        if g is None:
            groups[sig] = Finding(sig, kind, nf, lead_regime(e.scenario), e.scenario, a.severity, 1, e.index, [e.index])
        else:
            g.members += 1
            g.member_indices.append(e.index)
            if a.severity > g.severity:
                g.representative, g.severity = e.scenario, a.severity
    return sorted(groups.values(), key=lambda f: (-f.severity, -f.members))
