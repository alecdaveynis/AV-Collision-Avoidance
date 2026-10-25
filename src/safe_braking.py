# safe_braking.py
"""Improved collision-avoidance braking policy (deterministic, no learning, no LLM).

The baseline ``BehaviorManager`` decides from *zones*: STOP when the gap is inside ``2 + closing^2 / (2 * 8)`` metres,
SLOW (a fixed 2.4 m/s^2) inside a headway zone or below a TTC threshold. The zone assumes full braking starts the
instant the gap enters it. In reality the brake takes effect only after the actuator lag, the escalation confirmation
and the smoothing delay of the range filter, so at 15 m/s closing the car has already used 4-6 m of its 2 m margin
budget (measured in docs/redteam/planner_root_cause.md).

This policy instead computes the deceleration the situation *requires*, on the delay-compensated gap::

    tau     = actuator lag + one decision frame + range-filter lag          (s)
    g_eff   = gap - standstill margin - closing * tau                       (m, distance left once braking works)
    a_req   = closing^2 / (2 * g_eff)                                       (m/s^2, constant-speed lead assumed)
    demand  = min(SAFE_MARGIN_FACTOR * a_req, EMERGENCY_DECEL)              (physical limit)

and maps it onto the *existing* three-state machine:

    DRIVE    demand below SAFE_A_SLOW and no headway / TTC trigger
    SLOWING  demand >= SAFE_A_SLOW, or the baseline headway / TTC rule; brake = max(0.3, demand / limit)
    STOPPED  demand >= SAFE_A_STOP, or inside the standstill margin;    brake = max(SAFE_A_STOP / limit, demand / limit)

State-transition stability: a demand >= SAFE_A_IMMEDIATE escalates at once, anything weaker needs the baseline
``ESCALATE_FRAMES`` confirmation, and a state is only left after the demand has fallen below ``SAFE_RELEASE_RATIO`` of its
entry threshold for ``RELEASE_FRAMES`` frames. TTC is no longer able to trigger STOPPED on its own: it can only add the
gentle SLOWING brake, so a mis-estimated TTC cannot cause a hard stop (or, if optimistic, hide a hazard that range and
closing speed show).

A new obstacle whose closing speed has not been measured yet (first frames, or a one-frame ghost return) is treated as a
static obstacle but its demand is capped at ``SAFE_A_UNCONFIRMED``, so one range sample cannot cause a hard stop.

Known limits: the lead is assumed to keep its current speed (a lead that brakes harder *later* is only handled through
the margin factor), the closing speed comes from the Kalman track (kalman mode) or the smoothed range-rate, and any
perception latency beyond the one-frame/filter lag modelled in ``tau`` is not compensated.
"""
import math

import config
from behavior_manager import BehaviorManager, BehaviorState, _BRAKE

PLANNERS = ('baseline', 'improved')


class SafeBehaviorManager(BehaviorManager):
    name = 'improved'

    # -- physics ---------------------------------------------------------------
    def response_delay(self):
        """Time between the world state and braking force actually acting (s)."""
        a = config.DIST_EMA_ALPHA
        return config.SAFE_ACTUATOR_LAG + self.dt + (1.0 - a) / a * self.dt

    def required_decel(self, dist, closing):
        """Constant deceleration that stops short of the margin once braking takes effect (inf if already impossible)."""
        if closing <= config.MIN_CLOSING_SPEED:
            return 0.0
        g_eff = dist - config.STOP_BASE_DIST - closing * self.response_delay()
        if g_eff <= 0.1:
            return math.inf
        return closing * closing / (2.0 * g_eff)

    # -- hooks -----------------------------------------------------------------
    def _classify(self, entities, slow_dist, v):
        lim = config.EMERGENCY_DECEL
        stop_thr = config.SAFE_A_STOP * (config.SAFE_RELEASE_RATIO if self.state == BehaviorState.STOP else 1.0)
        slow_thr = config.SAFE_A_SLOW * (config.SAFE_RELEASE_RATIO if self.state != BehaviorState.GO else 1.0)
        raw, immediate, demand = BehaviorState.GO, False, 0.0
        for dist, _zone, e_ttc, closing in entities:
            if closing is None:
                # Closing speed not observed yet (first frames of a track or LiDAR return, or a one-frame ghost): the
                # worst case is a static obstacle, but a single range sample must not trigger a hard stop. Demand is
                # capped at SAFE_A_UNCONFIRMED until the closing speed has been measured.
                d = min(config.SAFE_MARGIN_FACTOR * self.required_decel(dist, v), config.SAFE_A_UNCONFIRMED)
            else:
                d = min(config.SAFE_MARGIN_FACTOR * self.required_decel(dist, closing), lim)
            demand = max(demand, d)
            if dist < config.STOP_BASE_DIST or d >= stop_thr:
                raw = BehaviorState.STOP
            elif (d >= slow_thr or dist < slow_dist or e_ttc < config.TTC_SLOW) and raw == BehaviorState.GO:
                raw = BehaviorState.SLOW
            if d >= config.SAFE_A_IMMEDIATE or dist < 0.5 * config.STOP_BASE_DIST:
                immediate = True
        self._demand = demand
        return raw, immediate

    def _brake_command(self, state):
        if state == BehaviorState.GO:
            return 0.0
        lim = config.EMERGENCY_DECEL
        floor = _BRAKE[BehaviorState.SLOW] if state == BehaviorState.SLOW else config.SAFE_A_STOP / lim
        return min(1.0, max(floor, self._demand / lim))


def make_behavior_manager(planner='baseline', ttc_mode='analytic'):
    """Planner factory: ``baseline`` is the original BehaviorManager, ``improved`` the policy above."""
    if planner == 'baseline':
        return BehaviorManager(ttc_mode=ttc_mode)
    if planner == 'improved':
        return SafeBehaviorManager(ttc_mode=ttc_mode)
    raise ValueError(f'planner must be one of {PLANNERS}, got {planner!r}')
