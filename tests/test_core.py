"""Unit tests for the planner, fusion helpers and timing utilities.

Run:  python -m unittest discover -s tests -v
"""
import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import config
import fusion_utils as fs
import lane_utils as lu
from calibration import load_calibration
import oxts_utils
import ttc_data
import ttc_model
from behavior_manager import BehaviorManager, BehaviorState, combine_ttc


def run_approach(start, end, speed, frames, in_path=True, ego_speed=None, dt=0.1):
    """Feed a single obstacle closing linearly; return the list of decisions."""
    pm = BehaviorManager()
    out = []
    for i in range(frames):
        d = start - (start - end) * i / (frames - 1)
        out.append(pm.update_state([{'id': 1, 'dist': d, 'in_path': in_path}], 99.0,
                                   dt=dt, ego_speed=ego_speed))
    return out


class PlannerTests(unittest.TestCase):
    def test_no_obstacles_drives(self):
        pm = BehaviorManager()
        for _ in range(10):
            dec = pm.update_state([], 99.0, dt=0.1)
        self.assertEqual(dec.state, BehaviorState.GO)

    def test_close_in_path_object_stops(self):
        decs = run_approach(15.0, 1.5, None, 40)
        self.assertEqual(decs[-1].state, BehaviorState.STOP)
        self.assertTrue(any(d.state == BehaviorState.SLOW for d in decs))

    def test_out_of_path_object_ignored(self):
        decs = run_approach(15.0, 1.5, None, 40, in_path=False)
        self.assertTrue(all(d.state == BehaviorState.GO for d in decs))

    def test_ttc_uses_real_dt(self):
        # 10 m -> 5 m over 10 frames at dt=0.1 s -> ~5 m/s closing
        decs = run_approach(10.0, 5.0, None, 10, dt=0.1)
        self.assertAlmostEqual(decs[-1].lead_closing, 5.0, delta=1.0)
        # same distances at dt=0.2 s -> half the closing speed
        decs2 = run_approach(10.0, 5.0, None, 10, dt=0.2)
        self.assertAlmostEqual(decs2[-1].lead_closing, 2.5, delta=0.6)

    def test_range_jump_is_not_a_collision(self):
        # In-path LiDAR minimum jumps 15 m -> 7 m (a new object entered the path).
        pm = BehaviorManager()
        for _ in range(5):
            pm.update_state([], 15.0, dt=0.1, ego_speed=5.0)
        dec = None
        for _ in range(5):
            dec = pm.update_state([], 7.0, dt=0.1, ego_speed=5.0)
        self.assertNotEqual(dec.state, BehaviorState.STOP)
        self.assertTrue(math.isinf(dec.ttc) or dec.ttc > config.TTC_STOP)

    def test_thresholds_scale_with_speed(self):
        slow = BehaviorManager().update_state([], 99.0, ego_speed=0.0)
        fast = BehaviorManager().update_state([], 99.0, ego_speed=15.0)
        self.assertGreater(fast.stop_dist, slow.stop_dist)
        self.assertGreater(fast.slow_dist, slow.slow_dist)

    def test_hysteresis_filters_single_frame_blip(self):
        pm = BehaviorManager()
        for _ in range(5):
            pm.update_state([], 99.0, dt=0.1)
        dec = pm.update_state([], 5.0, dt=0.1)         # one-frame blip inside slow zone
        self.assertEqual(dec.state, BehaviorState.GO)
        self.assertEqual(dec.raw_state, BehaviorState.SLOW)
        dec = pm.update_state([], 5.0, dt=0.1)         # persists -> escalate
        self.assertEqual(dec.state, BehaviorState.SLOW)

    def test_release_needs_persistence(self):
        pm = BehaviorManager()
        for _ in range(3):
            pm.update_state([], 5.0, dt=0.1)
        self.assertEqual(pm.state, BehaviorState.SLOW)
        for _ in range(config.RELEASE_FRAMES - 1):
            pm.update_state([], 99.0, dt=0.1)
        self.assertEqual(pm.state, BehaviorState.SLOW)
        pm.update_state([], 99.0, dt=0.1)
        self.assertEqual(pm.state, BehaviorState.GO)

    def test_lead_car_at_our_speed_is_not_a_wall(self):
        # Ego 11 m/s, lead car 9 m ahead moving at the same speed (distance constant).
        pm = BehaviorManager()
        for _ in range(15):
            dec = pm.update_state([{'id': 1, 'dist': 9.0, 'in_path': True}], 99.0,
                                  dt=0.1, ego_speed=11.0)
        self.assertEqual(dec.state, BehaviorState.SLOW)       # following too close: gentle, not a stop
        self.assertNotEqual(dec.state, BehaviorState.STOP)

    def test_static_obstacle_at_speed_stops_in_time(self):
        # Ego 11 m/s toward a static object: stop zone = 2 + 11^2/16 ~ 9.6 m
        pm = BehaviorManager()
        states = []
        for i in range(60):
            d = 30.0 - 1.1 * i
            dec = pm.update_state([{'id': 1, 'dist': d, 'in_path': True}], 99.0, dt=0.1, ego_speed=11.0)
            states.append((d, dec.state))
        first_stop = next(d for d, s in states if s == BehaviorState.STOP)
        self.assertGreater(first_stop, 7.0)                   # braking starts early enough
        self.assertLess(first_stop, 14.0)

    def test_combine_ttc_modes(self):
        self.assertEqual(combine_ttc(4.0, 1.0, 'analytic'), 4.0)
        self.assertEqual(combine_ttc(4.0, 1.0, 'learned'), 1.0)
        self.assertAlmostEqual(combine_ttc(4.0, 1.0, 'fused'), 2.0)
        self.assertEqual(combine_ttc(4.0, None, 'fused'), 4.0)           # no network output
        self.assertAlmostEqual(combine_ttc(math.inf, 2.5, 'fused'),
                               math.sqrt(config.TTC_MAX * 2.5))          # inf capped

    def test_learned_ttc_can_trigger_slow(self):
        # Distance-based rules say GO (25 m, not closing) but the network says TTC 1.5 s.
        mode_results = {}
        for mode in ('analytic', 'learned'):
            pm = BehaviorManager(ttc_mode=mode)
            for _ in range(6):
                dec = pm.update_state([{'id': 1, 'dist': 25.0, 'in_path': True, 'ttc_learned': 1.5}],
                                      99.0, dt=0.1, ego_speed=0.0)
            mode_results[mode] = dec.state
        self.assertEqual(mode_results['analytic'], BehaviorState.GO)
        self.assertNotEqual(mode_results['learned'], BehaviorState.GO)

    def test_stale_tracks_forgotten(self):
        pm = BehaviorManager()
        pm.update_state([{'id': 7, 'dist': 10.0}], 99.0)
        for _ in range(config.TRACK_MAX_AGE + 2):
            pm.update_state([], 99.0)
        self.assertNotIn(7, pm._tracks)


class FusionTests(unittest.TestCase):
    def test_projection_straight_ahead_hits_principal_point(self):
        pts = np.array([[10.0, 0.0, -0.1]])
        uv, valid = fs.project_points(pts, (375, 1242))
        self.assertTrue(valid[0])
        self.assertAlmostEqual(uv[0, 0], 609.0, delta=30)   # cx ~ 609

    def test_behind_camera_invalid(self):
        uv, valid = fs.project_points(np.array([[-5.0, 0.0, 0.0]]), (375, 1242))
        self.assertFalse(valid[0])

    def test_box_distance_prefers_nearest_surface(self):
        # 20 points on the object at 10 m, 20 background points at 30 m, all inside the box
        uv = np.tile(np.array([[100.0, 100.0]]), (40, 1))
        depth = np.array([10.0] * 20 + [30.0] * 20)
        d = fs.box_lidar_distance(uv, depth, (50, 50, 150, 150))
        self.assertAlmostEqual(d, 10.0, delta=0.1)

    def test_box_distance_none_without_support(self):
        uv = np.array([[100.0, 100.0]])
        self.assertIsNone(fs.box_lidar_distance(uv, np.array([10.0]), (50, 50, 150, 150)))

    def test_corridor_mask_is_ahead_of_car(self):
        mask = fs.corridor_mask((375, 1242))
        self.assertTrue(mask[-5, 609] > 0)       # bottom centre is inside the corridor
        self.assertEqual(mask[-5, 50], 0)        # far left is not
        self.assertEqual(mask[10, 609], 0)       # sky is not

    def test_ground_removed_on_flat_and_sloped_terrain(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(5, 40, 2000); y = rng.uniform(-8, 8, 2000)
        flat = np.column_stack((x, y, np.full(2000, config.GROUND_Z) + rng.normal(0, 0.02, 2000), np.ones(2000)))
        slope = np.column_stack((x, y, config.GROUND_Z + 0.04 * x + 0.03 * y, np.ones(2000)))  # embankment
        self.assertLess(len(fs.obstacle_points(flat)), 20)
        self.assertLess(len(fs.obstacle_points(slope)), 40)

    def test_car_kept_above_ground(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(5, 40, 3000); y = rng.uniform(-8, 8, 3000)
        ground = np.column_stack((x, y, np.full(3000, config.GROUND_Z), np.ones(3000)))
        cx = rng.uniform(14.0, 18.0, 300); cy = rng.uniform(-0.9, 0.9, 300)
        car = np.column_stack((cx, cy, rng.uniform(config.GROUND_Z + 0.3, config.GROUND_Z + 1.5, 300), np.ones(300)))
        obs = fs.obstacle_points(np.vstack([ground, car]))
        in_car = (obs[:, 0] > 13.9) & (obs[:, 0] < 18.1) & (np.abs(obs[:, 1]) < 1.0)
        self.assertGreater(in_car.sum(), 150)          # most of the car survives
        self.assertLess((~in_car).sum(), 30)           # the ground does not

    def test_in_path_distance_needs_cluster_support(self):
        self.assertEqual(fs.in_path_distance(np.array([4.0, 4.2])), 99.0)               # too few points
        self.assertEqual(fs.in_path_distance(np.array([4.0, 9.0, 15.0, 22.0, 30.0])), 99.0)  # scattered
        stray_plus_car = np.concatenate([[3.0], np.linspace(12.0, 12.6, 30)])
        self.assertAlmostEqual(fs.in_path_distance(stray_plus_car), 12.0, delta=0.1)     # stray ignored
        self.assertEqual(fs.in_path_distance(np.array([])), 99.0)

    def test_box_in_mask(self):
        mask = np.zeros((100, 200), np.uint8)
        mask[:, 80:120] = 1
        self.assertTrue(fs.box_in_mask(mask, (70, 20, 130, 90)))
        self.assertFalse(fs.box_in_mask(mask, (150, 20, 190, 90)))


def synthetic_track(start=20.0, closing=5.0, n=30, dt=0.1, ego=8.0, cls=2, real_h=1.5, f=721.0):
    """Rows like tracks.csv for an object approaching at constant closing speed."""
    rows = []
    for i in range(n):
        d = start - closing * dt * i
        h = f * real_h / d
        rows.append({'frame': i, 'time': i * dt, 'dt': dt, 'cls': cls,
                     'box': (600 - h, 200 - h, 600 + h, 200), 'img_shape': (375, 1242),
                     'ego': ego, 'mono': d, 'lidar': d})
    return rows


class TTCModelTests(unittest.TestCase):
    def _sequence(self, **kw):
        rows = synthetic_track(**kw)[:ttc_model.SEQ_LEN]
        base = [ttc_model.frame_features(r['box'], r['img_shape'], r['cls'], r['mono'], r['ego'])
                for r in rows]
        return ttc_model.build_sequence(base, [r['dt'] for r in rows])

    def test_sequence_shape(self):
        seq = self._sequence()
        self.assertEqual(seq.shape, (ttc_model.SEQ_LEN, ttc_model.FEATURE_DIM))

    def test_tau_prior_matches_true_ttc(self):
        # d = 20 - 5 t; over the first 0.8 s the object closes from 20 m to 16.5 m.
        seq = self._sequence(start=20.0, closing=5.0)
        tau = float(ttc_model.tau_ttc(seq[None])[0])
        true_ttc_mid = (20.0 - 5.0 * 0.35) / 5.0          # range / closing at the window mid-point
        self.assertAlmostEqual(tau, true_ttc_mid, delta=0.6)

    def test_tau_prior_not_closing_is_max(self):
        seq = self._sequence(start=20.0, closing=0.0)
        self.assertAlmostEqual(float(ttc_model.tau_ttc(seq[None])[0]), config.TTC_MAX, places=3)

    def test_untrained_net_equals_prior(self):
        net = ttc_model.TTCNet().eval()
        seq = self._sequence(start=20.0, closing=5.0)[None]
        self.assertAlmostEqual(float(net.predict_ttc(seq)[0]), float(ttc_model.tau_ttc(seq)[0]), places=4)

    def test_residual_is_bounded(self):
        net = ttc_model.TTCNet().eval()
        for p in net.head[-1].parameters():
            p.data.fill_(100.0)
        seq = self._sequence(start=20.0, closing=5.0)[None]
        x = torch.tensor(seq)
        prior = float(ttc_model.tau_log_ttc(x))
        out = float(net(x).detach())
        self.assertLessEqual(abs(out - prior), ttc_model.MAX_RESIDUAL + 1e-4)

    def test_features_contain_no_lidar(self):
        # mono distance is a feature, LiDAR range is not: changing 'lidar' must not change inputs
        a = synthetic_track()
        b = [dict(r, lidar=r['lidar'] + 5.0) for r in a]
        fa = ttc_model.frame_features(a[5]['box'], a[5]['img_shape'], a[5]['cls'], a[5]['mono'], a[5]['ego'])
        fb = ttc_model.frame_features(b[5]['box'], b[5]['img_shape'], b[5]['cls'], b[5]['mono'], b[5]['ego'])
        np.testing.assert_array_equal(fa, fb)


class TTCDataTests(unittest.TestCase):
    def test_label_from_future_range(self):
        t = np.arange(6) * 0.1
        d = 20.0 - 5.0 * t
        self.assertAlmostEqual(ttc_data.ttc_from_range_history(t, d, at_end=False), 4.0, places=3)
        self.assertAlmostEqual(ttc_data.ttc_from_range_history(t, d, at_end=True), (20 - 2.5) / 5.0, places=3)

    def test_label_not_closing(self):
        t = np.arange(6) * 0.1
        self.assertEqual(ttc_data.ttc_from_range_history(t, np.full(6, 15.0), at_end=False), config.TTC_MAX)
        self.assertEqual(ttc_data.ttc_from_range_history(t, 15.0 + 3.0 * t, at_end=False), config.TTC_MAX)

    def test_label_clipped(self):
        t = np.arange(6) * 0.1
        # 3 m away closing at 20 m/s -> 0.15 s, clipped up to TTC_MIN (0.3 s)
        self.assertEqual(ttc_data.ttc_from_range_history(t, 3.0 - 20.0 * t, at_end=False), config.TTC_MIN)

    def test_contiguous_runs_split_on_gaps(self):
        rows = [{'frame': f} for f in (0, 1, 2, 5, 6, 9)]
        self.assertEqual([[r['frame'] for r in run] for run in ttc_data.contiguous_runs(rows)],
                         [[0, 1, 2], [5, 6], [9]])

    def test_learned_ttc_inference_is_causal_and_needs_history(self):
        net = ttc_model.TTCNet().eval()
        inf = ttc_model.LearnedTTC(net)
        rows = synthetic_track(n=12)
        outs = []
        for r in rows:
            trk = [{'id': 1, 'class': r['cls'], 'box': r['box'], 'mono_dist': r['mono']}]
            inf.update(trk, r['ego'], r['dt'], r['frame'], r['img_shape'])
            outs.append(trk[0]['ttc_learned'])
        self.assertTrue(all(o is None for o in outs[:ttc_model.SEQ_LEN - 1]))   # not enough history yet
        self.assertIsNotNone(outs[ttc_model.SEQ_LEN - 1])
        # prediction at frame k must not depend on later frames: recompute with truncated input
        inf2 = ttc_model.LearnedTTC(net)
        last = None
        for r in rows[:ttc_model.SEQ_LEN + 1]:
            trk = [{'id': 1, 'class': r['cls'], 'box': r['box'], 'mono_dist': r['mono']}]
            inf2.update(trk, r['ego'], r['dt'], r['frame'], r['img_shape'])
            last = trk[0]['ttc_learned']
        self.assertAlmostEqual(last, outs[ttc_model.SEQ_LEN], places=5)

    def test_learned_ttc_restarts_after_gap(self):
        inf = ttc_model.LearnedTTC(ttc_model.TTCNet().eval())
        rows = synthetic_track(n=20)
        out = None
        for r in rows[:10] + rows[14:20]:      # 4-frame gap
            trk = [{'id': 1, 'class': r['cls'], 'box': r['box'], 'mono_dist': r['mono']}]
            inf.update(trk, r['ego'], r['dt'], r['frame'], r['img_shape'])
            out = trk[0]['ttc_learned']
        self.assertIsNone(out)                  # only 6 frames since the gap < SEQ_LEN


class LaneGeometryTests(unittest.TestCase):
    SHAPE = (375, 1242, 3)

    def setUp(self):
        self.calib = load_calibration()
        self.bev = lu.make_bev(self.calib, self.SHAPE)

    def _warped(self, y_left, y_right, x_from=7.0, x_to=28.0):
        img = np.zeros((self.bev.size[1], self.bev.size[0]), np.uint8)
        for y in (y_left, y_right):
            xs = np.linspace(x_from, x_to, 30)
            pts = np.column_stack(self.bev.to_pixels(xs, np.full_like(xs, y))).astype(np.int32)
            cv2.polylines(img, [pts], False, 255, 5)
        return img

    def test_bev_maps_ground_points_back_to_metres(self):
        pts = np.array([[10.0, 1.5, config.GROUND_Z], [20.0, -2.0, config.GROUND_Z]])
        uv, _ = fs.project_points(pts, self.SHAPE, self.calib)
        x, y = self.bev.image_to_ground(uv)
        np.testing.assert_allclose(x, [10.0, 20.0], atol=0.05)
        np.testing.assert_allclose(y, [1.5, -2.0], atol=0.05)

    def test_fit_recovers_straight_lane(self):
        fit, status = lu.fit_lane(self._warped(1.75, -1.75), self.bev)
        self.assertIsNotNone(fit)
        xs = np.linspace(8, 20, 5)
        np.testing.assert_allclose(fit.center(xs), 0.0, atol=0.15)
        np.testing.assert_allclose(fit.width(xs), 3.5, atol=0.2)

    def test_single_line_is_not_extended_by_default(self):
        fit, _ = lu.fit_lane(self._warped(1.75, 100.0), self.bev)    # only the left line is visible
        self.assertIsNone(fit)

    def test_gate_accepts_lane_where_the_car_is_heading(self):
        fit, _ = lu.fit_lane(self._warped(1.75, -1.75), self.bev)
        self.assertTrue(lu.consistent_with_path(fit, 0.0))
        self.assertFalse(lu.consistent_with_path(fit, 0.04))        # car turning hard: path is ~2 m to the side
        shifted, _ = lu.fit_lane(self._warped(4.0, 0.5), self.bev)
        if shifted is not None:
            self.assertFalse(lu.consistent_with_path(shifted, 0.0))


class TimingTests(unittest.TestCase):
    def test_frame_dts_from_timestamps(self):
        ts = np.array([0.0, 0.1, 0.25, 0.35])
        dts = oxts_utils.frame_dts(ts, 4)
        np.testing.assert_allclose(dts, [config.DEFAULT_DT, 0.1, 0.15, 0.1], atol=1e-9)

    def test_frame_dts_default_without_timestamps(self):
        dts = oxts_utils.frame_dts(None, 3)
        np.testing.assert_allclose(dts, [config.DEFAULT_DT] * 3)


if __name__ == '__main__':
    unittest.main()
