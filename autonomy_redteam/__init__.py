"""Autonomy Red Team: automated safety falsification for the repository's AV pipeline.

search -> minimize -> diagnose -> report -> regress.   CLI: ``python -m autonomy_redteam --help``.
"""
__version__ = '0.1.0'

from .scenario import Scenario, ScenarioGenerator, ScenarioValidationError   # noqa: F401
from .sut import SystemUnderTest                                            # noqa: F401
