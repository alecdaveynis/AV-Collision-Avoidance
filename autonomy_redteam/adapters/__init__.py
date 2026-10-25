"""SystemUnderTest adapters.

Implemented:  ``SafetySimSUT``  - the repository's closed-loop simulator + production planner.

Extension points (NOT implemented): a CARLA adapter would drive a CARLA world and read the stack's
topics; a ROS 2 adapter would replay bags / subscribe to the planner's outputs. Either only has to
implement :class:`autonomy_redteam.sut.SystemUnderTest`.
"""
from .safety_sim_adapter import SafetySimSUT, default_sut_factory   # noqa: F401
