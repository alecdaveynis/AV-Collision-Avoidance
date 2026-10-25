"""Tiered regression detection (autonomy_redteam/contracts.py) and the repository's regression suite / baselines."""
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from autonomy_redteam.adapters import default_sut_factory  # noqa: E402
from autonomy_redteam.contracts import (Policy, aggregate, compare_scenario, decide, fisher_one_sided, mcnemar_one_sided,  # noqa: E402
                                        run_regression_v2)
from autonomy_redteam.metrics import DEFAULT_CRITERIA  # noqa: E402
from autonomy_redteam.regress import RegressionCriteria, load_suite, run_regression  # noqa: E402
from autonomy_redteam.runner import Evaluator  # noqa: E402

SUITE = ROOT / 'scenarios' / 'regression'
BASE = ROOT / 'scenarios' / 'baselines'


def rec(fail=False, collision=False, gap=10.0, impact=0.0, unnec=0.0):
    return {'safety_failure': fail, 'failure_kinds': ['collision'] if collision else (['near_miss'] if fail else []),
            'collision': collision, 'min_gap': gap, 'impact_speed': impact, 'min_ttc': None, 'brake_reaction_time': 0.5,
            'unnecessary_braking_s': unnec, 'peak_cmd_decel': 5.0, 'n_states': 3, 'collision_class': 'none'}


class TestStatistics(unittest.TestCase):
    def test_mcnemar_exact_values(self):
        self.assertAlmostEqual(mcnemar_one_sided(5, 0), 1 / 32)
        self.assertAlmostEqual(mcnemar_one_sided(0, 5), 1.0)
        self.assertAlmostEqual(mcnemar_one_sided(3, 3), 0.65625)
        self.assertEqual(mcnemar_one_sided(0, 0), 1.0)

    def test_fisher_matches_hypergeometric(self):
        # 0/8 -> 8/8: only one arrangement out of C(16,8)
        self.assertAlmostEqual(fisher_one_sided(8, 8, 0, 8), 1 / math.comb(16, 8))
        self.assertAlmostEqual(fisher_one_sided(0, 8, 0, 8), 1.0)
        self.assertGreater(fisher_one_sided(2, 8, 1, 8), 0.4)


class TestTiers(unittest.TestCase):
    pol = Policy(n_seeds=12)

    def test_identical_runs_are_within_tolerance(self):
        runs = [rec(gap=3.0 + 0.1 * i) for i in range(12)]
        r = compare_scenario('s', 'safe_contract', runs, runs, self.pol, 10)
        self.assertEqual(r['status'], 'within_tolerance')

    def test_one_collision_breaks_a_safe_contract(self):
        base = [rec(gap=5.0) for _ in range(12)]
        cur = [rec(gap=5.0) for _ in range(11)] + [rec(fail=True, collision=True, gap=0.0, impact=4.0)]
        r = compare_scenario('s', 'safe_contract', base, cur, self.pol, 10)
        self.assertEqual(r['status'], 'hard_violation')
        self.assertIn('contract broken', r['hard'][0])

    def test_single_flip_in_a_marginal_scenario_is_not_a_violation(self):
        base = [rec(fail=(i < 3), collision=(i < 3), gap=0.0 if i < 3 else 2.0, impact=3.0 if i < 3 else 0.0) for i in range(12)]
        cur = [rec(fail=(i < 4), collision=(i < 4), gap=0.0 if i < 4 else 2.0, impact=3.0 if i < 4 else 0.0) for i in range(12)]
        r = compare_scenario('s', 'marginal', base, cur, self.pol, 10)
        self.assertNotIn(r['status'], ('hard_violation', 'degraded'))

    def test_large_shift_in_a_marginal_scenario_is_degradation(self):
        base = [rec(gap=3.0) for _ in range(12)]
        cur = [rec(fail=True, collision=True, gap=0.0, impact=6.0) for _ in range(12)]
        r = compare_scenario('s', 'marginal', base, cur, self.pol, 4)
        self.assertEqual(r['status'], 'degraded')

    def test_known_failure_getting_materially_worse_is_hard_but_better_is_not(self):
        base = [rec(fail=True, collision=True, gap=0.0, impact=8.0) for _ in range(12)]
        worse = [rec(fail=True, collision=True, gap=0.0, impact=14.0) for _ in range(12)]
        better = [rec(gap=4.0) for _ in range(12)]
        self.assertEqual(compare_scenario('s', 'known_failure', base, worse, self.pol, 10)['status'], 'hard_violation')
        self.assertEqual(compare_scenario('s', 'known_failure', base, better, self.pol, 10)['status'], 'improved')

    def test_pooled_tests_catch_diffuse_degradation(self):
        # ten scenarios, each only one seed worse: no per-scenario test fires, the pooled McNemar does
        items = []
        for k in range(10):
            base = [rec(gap=5.0) for _ in range(12)]
            cur = [rec(gap=5.0) for _ in range(12)]
            cur[0] = rec(fail=True, collision=True, gap=0.0, impact=3.0)
            items.append((f's{k}', 'marginal', base, cur))
        rows, suite = aggregate(items, self.pol)
        self.assertFalse(any(r['status'] in ('degraded', 'hard_violation') for r in rows))
        self.assertTrue(suite['degraded'])
        self.assertFalse(decide(rows, suite, self.pol)[1])


class TestRepositorySuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.improved = json.loads((BASE / 'v2_improved.json').read_text())

    def _ev(self, planner):
        return Evaluator(default_sut_factory('kalman', None, planner), DEFAULT_CRITERIA)

    def test_baseline_file_is_format_2_with_contracts(self):
        self.assertEqual(self.improved['baseline_format'], 2)
        roles = {v['role'] for v in self.improved['scenarios'].values()}
        self.assertIn('safe_contract', roles)
        self.assertEqual(self.improved['scenarios']['known_failure_gentle_lead_braking']['role'], 'safe_contract')

    def test_improved_planner_passes_its_own_baseline(self):
        res = run_regression_v2(self._ev('improved'), SUITE, self.improved)
        self.assertTrue(res['passed'], res['violations'])
        self.assertEqual(res['counts'], {'within_tolerance': len(load_suite(SUITE))})

    def test_original_planner_is_caught_by_the_fix_contract(self):
        res = run_regression_v2(self._ev('baseline'), SUITE, self.improved)
        self.assertFalse(res['passed'])
        self.assertEqual(res['tier'], 'hard_violation')
        self.assertIn('known_failure_gentle_lead_braking', res['violations'])

    def test_legacy_format_1_baseline_still_works(self):
        base = json.loads((BASE / 'v1.json').read_text())
        res = run_regression(self._ev('baseline'), SUITE, base, RegressionCriteria())
        self.assertTrue(res['passed'])
        self.assertEqual(res['counts'].get('missing_baseline', 0), len(load_suite(SUITE)) - len(base['scenarios']))

    def test_v2_baselines_are_deterministic(self):
        orig = json.loads((BASE / 'v2_original.json').read_text())
        res = run_regression_v2(self._ev('baseline'), SUITE, orig)
        self.assertTrue(res['passed'], res['violations'])


if __name__ == '__main__':
    unittest.main()
