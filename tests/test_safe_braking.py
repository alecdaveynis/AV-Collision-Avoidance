"""Improved braking policy (src/safe_braking.py) and its integration with the simulator and the red-team CLI."""
import math
import sys
import unittest
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from behavior_manager import BehaviorManager, BehaviorState  # noqa: E402
from safe_braking import PLANNERS, SafeBehaviorManager, make_behavior_manager  # noqa: E402

DT = 0.1


def drive(pm, dists, v, closing=None, hits=10, ttc=math.inf):
    """Feed one in-path obstacle per frame; returns the list of Decisions."""
    out = []
    for d in dists:
        ob = {'id': 1, 'dist': d, 'in_path': True, 'ttc_3d': ttc}
        if closing is not None:
            ob.update(closing_3d=closing, track_hits=hits)
        out.append(pm.update_state([ob], 99.0, dt=DT, ego_speed=v))
    return out


class TestRequiredDecel(unittest.TestCase):
    def setUp(self):
        self.pm = SafeBehaviorManager(ttc_mode='kalman')
        self.pm.dt = DT

    def test_response_delay_is_lag_plus_frame_plus_filter_lag(self):
        a = config.DIST_EMA_ALPHA
        self.assertAlmostEqual(self.pm.response_delay(), config.SAFE_ACTUATOR_LAG + DT + (1 - a) / a * DT)

    def test_formula(self):
        tau = self.pm.response_delay()
        g = 30.0 - config.STOP_BASE_DIST - 10.0 * tau
        self.assertAlmostEqual(self.pm.required_decel(30.0, 10.0), 100.0 / (2 * g))

    def test_not_closing_needs_nothing_and_no_room_is_infinite(self):
        self.assertEqual(self.pm.required_decel(5.0, 0.2), 0.0)
        self.assertTrue(math.isinf(self.pm.required_decel(5.0, 10.0)))

    def test_monotone_in_closing_and_gap(self):
        self.assertLess(self.pm.required_decel(40, 8), self.pm.required_decel(40, 10))
        self.assertLess(self.pm.required_decel(60, 10), self.pm.required_decel(40, 10))


class TestStateMachine(unittest.TestCase):
    def test_far_and_not_closing_drives(self):
        d = drive(make_behavior_manager('improved', 'kalman'), [60.0] * 10, 12.0, closing=0.0)[-1]
        self.assertEqual(d.state, BehaviorState.GO)
        self.assertEqual(d.brake, 0.0)

    def test_imminent_collision_stops_immediately_with_full_brake(self):
        d = drive(make_behavior_manager('improved', 'kalman'), [6.0], 12.0, closing=12.0)[-1]
        self.assertEqual(d.state, BehaviorState.STOP)          # demand >= SAFE_A_IMMEDIATE skips the confirmation
        self.assertEqual(d.brake, 1.0)

    def test_brake_is_proportional_to_demand_and_bounded(self):
        pm = make_behavior_manager('improved', 'kalman')
        for dist in (40.0, 30.0, 25.0, 20.0):
            d = drive(pm, [dist, dist], 12.0, closing=8.0)[-1]
            self.assertTrue(0.0 <= d.brake <= 1.0)
            if d.state == BehaviorState.STOP:
                self.assertAlmostEqual(d.brake, max(config.SAFE_A_STOP, d.demand_decel) / config.EMERGENCY_DECEL, places=6)

    def test_unconfirmed_closing_cannot_cause_a_hard_stop(self):
        # first sighting of an object 14 m ahead with an unknown closing speed (e.g. a one-frame ghost)
        pm = make_behavior_manager('improved', 'kalman')
        d = drive(pm, [14.0], 12.0, closing=None)[-1]
        self.assertNotEqual(d.state, BehaviorState.STOP)
        self.assertLessEqual(d.demand_decel, config.SAFE_A_UNCONFIRMED + 1e-9)

    def test_young_track_closing_is_not_trusted(self):
        pm = make_behavior_manager('improved', 'kalman')
        d = drive(pm, [30.0, 30.0], 12.0, closing=26.0, hits=3)[-1]   # prior-dominated velocity of a 3-hit track
        self.assertNotEqual(d.state, BehaviorState.STOP)
        mature = drive(make_behavior_manager('improved', 'kalman'), [30.0, 30.0], 12.0, closing=26.0, hits=10)[-1]
        self.assertEqual(mature.state, BehaviorState.STOP)

    def test_ttc_alone_cannot_stop_the_improved_planner_but_stops_the_baseline(self):
        imp = drive(make_behavior_manager('improved', 'kalman'), [60.0] * 4, 12.0, closing=0.0, ttc=0.4)[-1]
        base = drive(make_behavior_manager('baseline', 'kalman'), [60.0] * 4, 12.0, closing=0.0, ttc=0.4)[-1]
        self.assertNotEqual(imp.state, BehaviorState.STOP)
        self.assertEqual(base.state, BehaviorState.STOP)

    def test_release_hysteresis_holds_stop_until_demand_is_clearly_lower(self):
        pm = make_behavior_manager('improved', 'kalman')
        drive(pm, [8.0], 12.0, closing=12.0)                              # enter STOP
        self.assertEqual(pm.state, BehaviorState.STOP)
        states = [d.state for d in drive(pm, [60.0] * (config.RELEASE_FRAMES + 1), 12.0, closing=0.0)]
        self.assertTrue(all(s == BehaviorState.STOP for s in states[:config.RELEASE_FRAMES - 1]))
        self.assertNotEqual(states[-1], BehaviorState.STOP)

    def test_standstill_margin_always_stops(self):
        d = drive(make_behavior_manager('improved', 'kalman'), [1.0] * 3, 0.0, closing=0.0)[-1]
        self.assertEqual(d.state, BehaviorState.STOP)

    def test_factory(self):
        self.assertIs(type(make_behavior_manager('baseline')), BehaviorManager)
        self.assertIsInstance(make_behavior_manager('improved'), SafeBehaviorManager)
        self.assertEqual(PLANNERS, ('baseline', 'improved'))
        with self.assertRaises(ValueError):
            make_behavior_manager('bogus')


class TestBaselineUnchanged(unittest.TestCase):
    """The refactor added hooks to BehaviorManager; its decisions must be exactly the original zone/TTC rules."""

    def test_baseline_brake_levels_and_zones(self):
        pm = BehaviorManager('analytic')
        decs = drive(pm, [40.0, 38.0, 36.0, 34.0, 32.0, 30.0, 28.0, 26.0, 24.0, 22.0, 20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0,
                          6.0, 4.0], 15.0)
        self.assertTrue(all(d.brake in (0.0, 0.3, 1.0) for d in decs))
        self.assertEqual(decs[0].state, BehaviorState.GO)
        self.assertEqual(decs[-1].state, BehaviorState.STOP)
        self.assertEqual(decs[-1].demand_decel, 0.0)

    def test_baseline_ignores_improved_only_inputs(self):
        a = BehaviorManager('kalman')
        b = BehaviorManager('kalman')
        da = drive(a, [30.0, 28.0, 26.0, 24.0, 22.0, 20.0], 14.0, closing=None)
        db = drive(b, [30.0, 28.0, 26.0, 24.0, 22.0, 20.0], 14.0, closing=99.0, hits=1)
        self.assertEqual([(d.state, d.brake) for d in da], [(d.state, d.brake) for d in db])


class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from autonomy_redteam.adapters import SafetySimSUT
        from autonomy_redteam.metrics import assess
        from autonomy_redteam.scenario import Scenario
        cls.SafetySimSUT, cls.assess = SafetySimSUT, staticmethod(assess)
        cls.known = Scenario.load(ROOT / 'docs' / 'redteam' / 'known_failure' / 'minimized_scenario.json')

    def test_known_failure_baseline_collides_improved_does_not(self):
        base = self.assess(self.SafetySimSUT(planner='baseline').run(self.known))
        imp = self.assess(self.SafetySimSUT(planner='improved').run(self.known))
        self.assertTrue(base.collision)
        self.assertEqual(base.collision_class, 'avoidable')
        self.assertFalse(imp.collision)
        self.assertGreater(imp.min_gap, 2.0)
        self.assertFalse(imp.failed)

    def test_both_planners_are_deterministic(self):
        for pl in ('baseline', 'improved'):
            a = self.assess(self.SafetySimSUT(planner=pl).run(self.known)).to_dict()
            b = self.assess(self.SafetySimSUT(planner=pl).run(self.known)).to_dict()
            self.assertEqual(a, b)

    def test_describe_and_validation(self):
        self.assertEqual(self.SafetySimSUT(planner='improved').describe()['planner'], 'improved')
        with self.assertRaises(ValueError):
            self.SafetySimSUT(planner='nope')

    def test_cli_legacy_planner_alias(self):
        from autonomy_redteam.cli import resolve_planner
        self.assertEqual(resolve_planner(Namespace(planner='analytic', ttc_mode=None)), ('baseline', 'analytic'))
        self.assertEqual(resolve_planner(Namespace(planner='improved', ttc_mode=None)), ('improved', 'kalman'))
        self.assertEqual(resolve_planner(Namespace()), ('baseline', 'kalman'))


if __name__ == '__main__':
    unittest.main()
