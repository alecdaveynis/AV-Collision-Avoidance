"""Tests for the 3D tracker, motion prediction, kalman TTC mode and the safety simulator."""
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import predict
import safety_sim as ss
import tracker3d as t3
from behavior_manager import BehaviorManager, BehaviorState, combine_ttc


def blob(x, y, n=60, size=0.8, z=0.5, rng=None):
    rng = rng or np.random.default_rng(0)
    pts = np.column_stack((x + rng.uniform(-size / 2, size / 2, n), y + rng.uniform(-size / 2, size / 2, n),
                           rng.uniform(z - 0.3, z + 0.7, n), np.zeros(n)))
    return pts


def cluster(x, y, ext=0.8):
    return t3.Cluster(x=x, y=y, z=0.8, x_near=x - ext / 2, ext_x=ext, ext_y=ext, n=40, z_top=1.5)


def pose_at(x, y=0.0):
    p = np.eye(4)
    p[0, 3], p[1, 3] = x, y
    return p


def run_track(obj_vx, ego_v, n=25, dt=0.1, obj_y=0.0, obj_x0=30.0):
    """Ego drives at ego_v along +x; object starts obj_x0 ahead moving obj_vx. Returns the tracker + last tracks."""
    tr = t3.Tracker3D()
    tracks = []
    for k in range(n):
        ego_x = ego_v * dt * k
        obj_x = obj_x0 + obj_vx * dt * k
        tracks = tr.update([cluster(obj_x - ego_x, obj_y)], pose_at(ego_x), dt, np.array([ego_v, 0.0]))
    return tr, tracks


class ClusterTests(unittest.TestCase):
    def test_two_objects_are_two_clusters(self):
        pts = np.vstack((blob(10, 0), blob(20, 4)))
        cl = t3.cluster_obstacles(pts)
        self.assertEqual(len(cl), 2)
        self.assertAlmostEqual(sorted(c.x for c in cl)[0], 10, delta=0.5)

    def test_long_wall_is_not_an_object(self):
        wall = np.column_stack((np.linspace(10, 40, 400), np.full(400, 6.0), np.full(400, 1.0), np.zeros(400)))
        self.assertEqual(t3.cluster_obstacles(wall), [])

    def test_sparse_noise_ignored(self):
        self.assertEqual(t3.cluster_obstacles(blob(15, 0, n=4)), [])


class TrackerTests(unittest.TestCase):
    def test_parked_object_has_zero_world_speed_while_ego_moves(self):
        _, tracks = run_track(obj_vx=0.0, ego_v=12.0)
        self.assertEqual(len(tracks), 1)
        t = tracks[0]
        self.assertLess(t.speed_world, 0.7)
        self.assertAlmostEqual(t.closing_speed, 12.0, delta=1.0)
        self.assertTrue(t.is_static)

    def test_moving_object_closing_speed(self):
        _, tracks = run_track(obj_vx=5.0, ego_v=12.0)
        t = tracks[0]
        self.assertAlmostEqual(t.speed_world, 5.0, delta=0.8)
        self.assertAlmostEqual(t.closing_speed, 7.0, delta=1.0)
        ttc, lo, hi = t.ttc()
        self.assertTrue(lo <= ttc <= hi or math.isinf(hi))
        self.assertAlmostEqual(ttc, t.bumper_range / t.closing_speed, places=6)

    def test_receding_object_has_infinite_ttc(self):
        _, tracks = run_track(obj_vx=15.0, ego_v=12.0)
        self.assertTrue(math.isinf(tracks[0].ttc()[0]))

    def test_id_is_stable_and_ghost_is_not_confirmed(self):
        tr = t3.Tracker3D()
        ids = set()
        for k in range(20):
            cl = [cluster(25 - 0.5 * k, 0.0)]
            if k == 7:
                cl.append(cluster(15, 1.0))          # one-frame ghost
            tracks = tr.update(cl, pose_at(0.0), 0.1, np.zeros(2))
            ids |= {t.id for t in tracks}
            if k == 7:
                self.assertEqual(len(tracks), 1)
        self.assertEqual(len(ids), 1)

    def test_velocity_uncertainty_shrinks_with_hits(self):
        tr, tracks = run_track(obj_vx=0.0, ego_v=10.0, n=4)
        early = tracks[0].closing_std()
        _, tracks = run_track(obj_vx=0.0, ego_v=10.0, n=30)
        self.assertLess(tracks[0].closing_std(), early)

    def test_in_ego_path(self):
        _, tracks = run_track(obj_vx=0.0, ego_v=5.0, obj_y=0.0)
        self.assertTrue(t3.in_ego_path(tracks[0]))
        _, tracks = run_track(obj_vx=0.0, ego_v=5.0, obj_y=4.0)
        self.assertFalse(t3.in_ego_path(tracks[0]))

    def test_match_to_boxes_prefers_nearest_and_uses_each_once(self):
        a = SimpleNamespace(id=1, rel_pos=np.array([10.0, 0.0]))
        b = SimpleNamespace(id=2, rel_pos=np.array([20.0, 0.0]))
        boxes = [{'id': 7, 'box': (0, 0, 100, 100)}]
        out = t3.match_to_boxes([b, a], boxes, lambda t: (50, 50))
        self.assertEqual(out[7].id, 1)
        self.assertEqual(len(out), 1)


class PredictTests(unittest.TestCase):
    def tr(self, state, hr=0.0, x_off=0.4, ext_y=1.8):
        return SimpleNamespace(state=np.array(state, float), heading_rate=hr, x_off=x_off, ext_y=ext_y)

    def test_cv_and_static(self):
        t = self.tr([0, 0, 3, 1])
        np.testing.assert_allclose(predict.predict_xy(t, [2.0], 'cv'), [[6.0, 2.0]])
        np.testing.assert_allclose(predict.predict_xy(t, [2.0], 'static'), [[0.0, 0.0]])

    def test_ctrv_matches_cv_at_zero_rate_and_turns_otherwise(self):
        t = self.tr([0, 0, 5, 0])
        np.testing.assert_allclose(predict.predict_xy(t, [1.0], 'ctrv'), predict.predict_xy(t, [1.0], 'cv'))
        t = self.tr([0, 0, 5, 0], hr=0.5)
        p = predict.predict_xy(t, [2.0], 'ctrv')[0]
        self.assertGreater(p[1], 0.5)                       # turning left
        self.assertLess(float(np.linalg.norm(p)), 10.0)     # arc is shorter than the straight line

    def test_crossing_object_conflict_time(self):
        # object 30 m ahead, 4 m to the left, walking right at 1.5 m/s; ego at 10 m/s
        t = self.tr([30, 4, 0, -1.5], ext_y=0.6, x_off=0.3)
        tc = predict.time_to_conflict(t, np.eye(4), 10.0, 0.0, 'cv', horizon=4.0)
        self.assertTrue(2.0 < tc < 3.5, tc)

    def test_parallel_lane_object_never_conflicts(self):
        t = self.tr([30, 3.5, 8, 0])
        self.assertTrue(math.isinf(predict.time_to_conflict(t, np.eye(4), 10.0)))

    def test_ego_curvature_is_followed(self):
        pos, yaw = predict.ego_future(np.eye(4), 10.0, 0.05, [0.0, 2.0])
        self.assertGreater(pos[1, 1], 1.0)
        self.assertAlmostEqual(yaw[1], 0.05 * 20.0, places=6)


class KalmanModeTests(unittest.TestCase):
    def test_combine_kalman(self):
        self.assertEqual(combine_ttc(5.0, 2.0, 'kalman', 1.5), 1.5)
        self.assertEqual(combine_ttc(5.0, 2.0, 'kalman', None), 5.0)
        self.assertEqual(combine_ttc(5.0, 2.0, 'analytic', 1.5), 5.0)

    def test_kalman_ttc_can_trigger_planner_when_range_rate_has_not(self):
        pm = BehaviorManager(ttc_mode='kalman')
        pm_a = BehaviorManager(ttc_mode='analytic')
        for _ in range(4):
            d = pm.update_state([{'id': 1, 'dist': 40.0, 'in_path': True, 'ttc_3d': 0.8}], 99.0, dt=0.1, ego_speed=10.0)
            da = pm_a.update_state([{'id': 1, 'dist': 40.0, 'in_path': True, 'ttc_3d': 0.8}], 99.0, dt=0.1, ego_speed=10.0)
        self.assertEqual(d.state, BehaviorState.STOP)
        self.assertEqual(da.state, BehaviorState.GO)


class SafetySimTests(unittest.TestCase):
    def test_no_braking_hits_a_parked_car(self):
        r = ss.run_episode(ss.NonePolicy(), 'stationary', np.random.default_rng(1), v0=12.0)
        self.assertTrue(r['collision'])

    def test_braking_policies_avoid_a_parked_car(self):
        for factory in (ss.RSSPolicy, ss.AEBPolicy, ss.PredictivePolicy, lambda: ss.PlannerPolicy('analytic')):
            r = ss.run_episode(factory(), 'stationary', np.random.default_rng(1), v0=12.0)
            self.assertFalse(r['collision'], factory)

    def test_episodes_are_reproducible(self):
        a = ss.run_episode(ss.RSSPolicy(), 'cut_in', np.random.default_rng(5))
        b = ss.run_episode(ss.RSSPolicy(), 'cut_in', np.random.default_rng(5))
        self.assertEqual(a, b)

    def test_oncoming_traffic_is_not_a_hazard(self):
        r = ss.run_episode(ss.PredictivePolicy(), 'oncoming', np.random.default_rng(2), v0=12.0)
        self.assertFalse(r['collision'])
        self.assertLess(r['peak_decel'], 1.0)

    def test_summary_shapes(self):
        res = ss.evaluate(2, 0, {'none': ss.NonePolicy})
        table = ss.summarize(res)
        self.assertEqual(set(table['none']['per_scenario_collision']), set(ss.HAZARD))


if __name__ == '__main__':
    unittest.main()
