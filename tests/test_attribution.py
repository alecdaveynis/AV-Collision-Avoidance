"""Evidence-based attribution (autonomy_redteam/attribution.py)."""
import ast
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from autonomy_redteam import attribution as at  # noqa: E402
from autonomy_redteam.adapters import default_sut_factory  # noqa: E402
from autonomy_redteam.diagnose import diagnose  # noqa: E402
from autonomy_redteam.metrics import DEFAULT_CRITERIA  # noqa: E402
from autonomy_redteam.runner import Evaluator  # noqa: E402
from autonomy_redteam.scenario import PARAMS, Scenario  # noqa: E402

# A scenario the planner survives without the fault and fails with it (taken from the held-out planted set; any such scenario works).
PLANTED_LOSS = {"ego_speed": 16.099, "lead_speed": 14.744, "initial_gap": 52.702, "lead_decel": 8.896, "brake_onset_s": 0.758,
                "perception_latency_s": 0.0, "distance_noise_std_m": 0.0, "dropout_prob": 0.0, "detection_loss_start_s": 0.91,
                "detection_loss_duration_s": 2.5, "ttc_error_factor": 1.0, "ghost_prob": 0.0, "duration_s": 14.0,
                "seed": 1576496403, "name": "plant-detection_loss", "schema_version": 1}


class TestNoLabelLeak(unittest.TestCase):
    def test_module_never_reads_fault_parameters(self):
        faults = {k for k, v in PARAMS.items() if v.group == 'fault'}
        tree = ast.parse((ROOT / 'autonomy_redteam' / 'attribution.py').read_text())
        used = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in faults:
                used.add(node.attr)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in faults:
                used.add(node.value)
        self.assertEqual(used, set(), f'attribution.py reads scenario fault parameters: {sorted(used)}')

    def test_attribute_only_uses_telemetry_and_variant_outcomes(self):
        import inspect
        src = inspect.getsource(at.attribute) + inspect.getsource(at.critical_window) + inspect.getsource(at.observe)
        self.assertNotIn('scenario.', src)
        self.assertNotIn('.params()', src)


class TestAttribution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ev = Evaluator(default_sut_factory('kalman'), DEFAULT_CRITERIA)

    def _attr(self, sc):
        d = diagnose(self.ev, sc, n_seeds=8)
        return at.attribute(self.ev, sc, d)

    def test_planted_detection_loss_is_primary(self):
        sc = Scenario.from_dict(PLANTED_LOSS)
        self.assertTrue(self.ev.evaluate(sc).failed)
        a = self._attr(sc)
        self.assertEqual(a['verdict'], 'primary')
        self.assertEqual(a['primary'], ['object_detection'])
        self.assertTrue(a['components']['object_detection']['error_before_unsafe_decision'])
        self.assertNotIn('planner_logic', a['primary'] + a['contributing'])
        # the three concepts are reported separately
        self.assertIn('object_detection', a['outcome_improved_by'])

    def test_scenario_without_the_fault_survives(self):
        # sanity of the planted case: the detection loss is what makes it fail
        sc = Scenario.from_dict(dict(PLANTED_LOSS, detection_loss_duration_s=0.0))
        self.assertFalse(self.ev.evaluate(sc).failed)

    def test_known_failure_is_a_planner_failure(self):
        sc = Scenario.load(ROOT / 'docs' / 'redteam' / 'known_failure' / 'minimized_scenario.json')
        a = self._attr(sc)
        self.assertEqual(a['primary'], ['planner_logic'])
        self.assertFalse(any(a['components'][c]['contributed'] for c in at.PERCEPTION))

    def test_safe_scenario_is_inconclusive_not_forced(self):
        sc = Scenario.from_dict(dict(PLANTED_LOSS, detection_loss_duration_s=0.0, lead_decel=0.0, initial_gap=80.0))
        self.assertFalse(self.ev.evaluate(sc).failed)
        d = diagnose(self.ev, sc, kind='collision', n_seeds=4)
        a = at.attribute(self.ev, sc, d)
        self.assertEqual(a['verdict'], 'inconclusive')
        self.assertTrue(a['inconclusive_reasons'])


class TestThresholdsFromNominal(unittest.TestCase):
    def test_nominal_calibration_file_supports_thresholds(self):
        cal = json.loads((ROOT / 'docs' / 'redteam' / 'diagnosis_eval' / 'nominal_calibration.json').read_text())
        st = cal['statistics']
        self.assertLess(st['lost_run']['max'], at.LOST_RUN_STEPS)
        self.assertLess(st['lag_s']['max'], at.LAG_MIN_S)
        self.assertLess(st['miss_frac']['max'], at.MISS_FRACTION)
        # continuous thresholds: about 1.5 x the nominal p99 (rounded to a convenient value), never below 1.4 x
        for key, thr in (('dist_rms', at.DIST_RMS_M), ('raw_jitter', at.RAW_JITTER_M), ('ttc_log', at.TTC_LOG_RATIO)):
            self.assertGreaterEqual(thr, 1.4 * st[key]['p99'], key)
            self.assertLessEqual(thr, 1.6 * st[key]['p99'] + 0.02, key)


if __name__ == '__main__':
    unittest.main()
