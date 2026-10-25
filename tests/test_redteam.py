"""Tests for the Autonomy Red Team framework (autonomy_redteam/). No KITTI, GPU, CARLA or API key needed."""
import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from autonomy_redteam import cli                                                       # noqa: E402
from autonomy_redteam.adapters import SafetySimSUT, default_sut_factory                # noqa: E402
from autonomy_redteam.adapters.safety_sim_adapter import patched_config                # noqa: E402
from autonomy_redteam.agent import (DeterministicInvestigator, LLMError, LLMResponse, Limits,  # noqa: E402
                                    LimitExceeded, ScriptedLLM, ToolBox, ToolCall, client_from_env,
                                    investigate_with_agent)
from autonomy_redteam.agent import llm as llm_mod                                      # noqa: E402
from autonomy_redteam.agent.schema import SchemaError, validate                        # noqa: E402
from autonomy_redteam.diagnose import diagnose, shapley, wilson                        # noqa: E402
from autonomy_redteam.findings import group_failures, signature                        # noqa: E402
from autonomy_redteam.interventions import INTERVENTIONS, apply                        # noqa: E402
from autonomy_redteam.metrics import DEFAULT_CRITERIA, assess                          # noqa: E402
from autonomy_redteam.minimize import Minimizer, derived_seeds, distance, minimize     # noqa: E402
from autonomy_redteam.oracle import required_decel, run_oracle                         # noqa: E402
from autonomy_redteam.regress import (RegressionCriteria, create_baseline, load_suite, promote,  # noqa: E402
                                      run_regression, to_markdown)
from autonomy_redteam.runner import BudgetExceeded, Evaluator                          # noqa: E402
from autonomy_redteam.scenario import (FAULT_GROUPS, PARAMS, PHYSICAL_PARAMS, Scenario,   # noqa: E402
                                       ScenarioGenerator, ScenarioValidationError)
from autonomy_redteam.search import CMAES, run_search                                  # noqa: E402
from autonomy_redteam.sut import Interventions                                         # noqa: E402

# A planner that never brakes: lets tests build failures that do not depend on the production thresholds.
NEVER_BRAKE = {'SLOW_BASE_DIST': 0.0, 'SLOW_HEADWAY': 0.0, 'STOP_BASE_DIST': 0.0, 'TTC_SLOW': 0.0, 'TTC_STOP': 0.0,
               'EMERGENCY_DECEL': 1e9}
CLOSING = Scenario(ego_speed=14, lead_speed=4, initial_gap=60, lead_decel=0, dropout_prob=0, ghost_prob=0, seed=1)


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*a, **k)
    return rc, buf.getvalue()


# --- scenario --------------------------------------------------------------------------------------------------------
class TestScenario(unittest.TestCase):
    def test_json_roundtrip_and_fingerprint(self):
        s = Scenario(ego_speed=12.5, lead_decel=3.0, perception_latency_s=0.2, seed=7, name='a')
        self.assertEqual(Scenario.from_json(s.to_json()), s)
        self.assertEqual(s.fingerprint(), s.replace(name='other').fingerprint())
        self.assertNotEqual(s.fingerprint(), s.replace(seed=8).fingerprint())

    def test_validation_errors(self):
        with self.assertRaises(ScenarioValidationError):
            Scenario(ego_speed=99).validate()
        with self.assertRaises(ScenarioValidationError):
            Scenario(initial_gap=float('nan')).validate()
        with self.assertRaises(ScenarioValidationError):
            Scenario(seed=-1).validate()
        with self.assertRaises(ScenarioValidationError):
            Scenario.from_dict({'ego_speed': 10, 'lane_change': 1})          # no lateral behaviour is modelled
        with self.assertRaises(ScenarioValidationError):
            Scenario.from_json('{not json')
        with self.assertRaises(ScenarioValidationError):
            Scenario(lead_decel=3, brake_onset_s=20, duration_s=14).validate()

    def test_load_accepts_wrapped_files_and_reports_missing(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'x.json'
            p.write_text(json.dumps({'scenario': Scenario(seed=3).to_dict(), 'meta': {}}))
            self.assertEqual(Scenario.load(p).seed, 3)
            with self.assertRaises(ScenarioValidationError):
                Scenario.load(Path(d) / 'missing.json')

    def test_faults(self):
        s = Scenario(perception_latency_s=0.2, detection_loss_duration_s=1.0, dropout_prob=0.0, ghost_prob=0.0)
        self.assertEqual(set(s.active_faults()), {'perception_latency', 'detection_loss'})
        self.assertEqual(s.without_faults(['perception_latency']).active_faults(), ['detection_loss'])

    def test_generator_deterministic_bounded_and_not_born_in_hazard(self):
        a = [ScenarioGenerator(5).sample() for _ in range(1)]
        g1, g2 = ScenarioGenerator(5), ScenarioGenerator(5)
        s1, s2 = [g1.sample() for _ in range(60)], [g2.sample() for _ in range(60)]
        self.assertEqual(s1, s2)
        self.assertNotEqual(s1[0], ScenarioGenerator(6).sample())
        for s in s1:
            s.validate()
            self.assertLessEqual(s.initial_required_decel(), 2.5 + 1e-2)
        g = ScenarioGenerator(1, bounds={'ego_speed': (8, 10)})
        self.assertTrue(all(8 <= g.sample().ego_speed <= 10 for _ in range(20)))
        with self.assertRaises(ScenarioValidationError):
            ScenarioGenerator(1, bounds={'ego_speed': (0, 10)})             # outside the hard bounds
        with self.assertRaises(ScenarioValidationError):
            ScenarioGenerator(1, bounds={'nonsense': (0, 1)})

    def test_unit_vector_roundtrip(self):
        g = ScenarioGenerator(2)
        s = g.sample()
        s2 = g.from_unit(g.to_unit(s), s.seed)
        for k in PHYSICAL_PARAMS:
            self.assertAlmostEqual(getattr(s, k), getattr(s2, k), places=2)
        tiny = s.replace(perception_latency_s=0.01, detection_loss_duration_s=0.0)      # "almost off" faults snap to off
        self.assertEqual(g.from_unit(g.to_unit(tiny), 1).perception_latency_s, 0.0)


# --- SUT / simulator adapter -------------------------------------------------------------------------------------------
class TestAdapter(unittest.TestCase):
    def test_deterministic_replay_and_reset(self):
        s = Scenario(ego_speed=15, lead_speed=8, initial_gap=40, lead_decel=4, seed=11)
        a = SafetySimSUT().run(s)
        b = SafetySimSUT().run(s)
        self.assertEqual([r.gt.gap for r in a.records], [r.gt.gap for r in b.records])
        sut = SafetySimSUT()
        sut.initialize(s)
        first = [sut.step().gt.gap for _ in range(20)]
        sut.reset()
        again = [sut.step().gt.gap for _ in range(20)]
        self.assertEqual(first, again)

    def test_interface_surface(self):
        sut = SafetySimSUT()
        sut.initialize(CLOSING)
        rec = sut.step()
        self.assertEqual(sut.ground_truth().t, rec.gt.t)
        self.assertIn(sut.planner_telemetry().state, ('DRIVE', 'SLOWING', 'STOPPED'))
        self.assertIsNotNone(sut.perception_telemetry())
        self.assertIsInstance(sut.events(), list)
        with self.assertRaises(RuntimeError):
            SafetySimSUT().step()

    def test_matches_production_planner_on_noise_free_input(self):
        """The adapter's decision step must equal safety_sim.PlannerPolicy('kalman') on identical clusters."""
        import safety_sim as ss
        import tracker3d as t3
        s = Scenario(ego_speed=14, lead_speed=6, initial_gap=45, lead_decel=3, brake_onset_s=1.0, dropout_prob=0,
                     ghost_prob=0, seed=3)
        ep = SafetySimSUT().run(s, Interventions(oracle_distance=True))
        # manual closed loop with the repository's own classes
        pol = ss.PlannerPolicy('kalman')
        tracker = t3.Tracker3D()
        lead = ss.car(s.initial_gap + 2.25, 0.0, s.lead_speed)

        def script(a, t, ego_x, dt):
            if t >= s.brake_onset_s:
                a.vx = max(a.vx - s.lead_decel * dt, 0.0)
        lead.script = script
        v, ego_x, a_act, cmds = s.ego_speed, 0.0, 0.0, []
        for k in range(len(ep.records)):
            t = k * ss.DT
            pose = np.eye(4)
            pose[0, 3] = ego_x
            rear = lead.x - lead.length / 2 - ego_x
            clusters = []
            if rear <= 55.0:
                clusters = [t3.Cluster(x=rear + 0.4, y=0.0, z=0.8, x_near=rear, ext_x=0.8, ext_y=lead.width, n=40, z_top=1.5)]
            tracks = tracker.update(clusters, pose, ss.DT, np.array([v, 0.0]))
            cmd = float(np.clip(pol.decide(tracks, clusters, pose, v, ss.DT), 0, ss.MAX_DECEL))
            cmds.append(cmd)
            a_act += (cmd - a_act) * ss.DT / ss.LAG
            v = max(v - a_act * ss.DT, 0.0)
            ego_x += v * ss.DT
            lead.step(t, ego_x, ss.DT)
        got = [r.planner.cmd_decel for r in ep.records]
        np.testing.assert_allclose(got, cmds, atol=1e-9)

    def test_max_brake_policy_equals_kinematic_oracle(self):
        s = Scenario(ego_speed=18, lead_speed=3, initial_gap=30, lead_decel=2, seed=2)
        ep = SafetySimSUT().run(s, Interventions(policy='max_brake'))
        o = run_oracle(s, 'max_brake', ep.limits)
        a = assess(ep)
        self.assertEqual(a.collision, o.collided)
        self.assertAlmostEqual(a.min_gap, o.min_gap, places=6)

    def test_faults_have_their_stated_effect(self):
        # latency: planner input is stale -> estimated gap too large while closing
        s = CLOSING.replace(perception_latency_s=0.5)
        ep = SafetySimSUT().run(s)
        errs = [r.perception.range_error for r in ep.records if r.perception.tracked and r.gt.closing_speed > 5]
        self.assertTrue(errs and np.mean(errs) > 3.0)
        # total dropout: the planner never sees the lead
        ep = SafetySimSUT().run(CLOSING.replace(dropout_prob=0.7, detection_loss_start_s=0, detection_loss_duration_s=4.0))
        self.assertTrue(any(not r.perception.tracked for r in ep.records))
        # detection loss window really removes the lead cluster
        ep = SafetySimSUT().run(CLOSING.replace(detection_loss_start_s=1.0, detection_loss_duration_s=1.0))
        inside = [r.perception.lead_detected for r in ep.records if 1.0 <= r.t < 2.0]
        self.assertFalse(any(inside))
        self.assertTrue(any(r.perception.lead_detected for r in ep.records if r.t < 1.0))

    def test_removing_one_fault_does_not_shift_other_noise(self):
        a = SafetySimSUT().run(CLOSING.replace(dropout_prob=0.3, distance_noise_std_m=1.0, seed=5))
        b = SafetySimSUT().run(CLOSING.replace(dropout_prob=0.3, distance_noise_std_m=0.0, seed=5))
        self.assertEqual([r.perception.lead_detected for r in a.records][:15], [r.perception.lead_detected for r in b.records][:15])

    def test_unsupported_features_are_rejected_not_ignored(self):
        sut = SafetySimSUT(planner_mode='analytic')
        with self.assertRaises(ValueError):
            sut.initialize(CLOSING.replace(ttc_error_factor=1.5))
        with self.assertRaises(ValueError):
            sut.initialize(CLOSING, Interventions(oracle_ttc=True))
        with self.assertRaises(ValueError):
            SafetySimSUT().initialize(CLOSING, Interventions(policy='bogus'))

    def test_config_override_is_scoped(self):
        import config
        orig = config.TTC_SLOW
        with patched_config({'TTC_SLOW': 0.5}):
            self.assertEqual(config.TTC_SLOW, 0.5)
        self.assertEqual(config.TTC_SLOW, orig)
        with self.assertRaises(KeyError), patched_config({'NOT_A_KEY': 1}):
            pass

    def test_oracle_ttc_is_orthogonal_to_latency(self):
        s = Scenario(ego_speed=14, lead_speed=14, initial_gap=45, lead_decel=6, perception_latency_s=0.5, dropout_prob=0, ghost_prob=0, seed=3)
        a = SafetySimSUT().run(s)
        b = SafetySimSUT().run(s, Interventions(oracle_ttc=True))
        # with a stale-but-exact TTC the TTC the planner sees still lags the true TTC
        late = [(r.perception.est_ttc, r.gt.ttc) for r in b.records if r.perception.tracked and r.gt.ttc < 3]
        self.assertTrue(any(e > g + 0.2 for e, g in late))


# --- oracle and metrics --------------------------------------------------------------------------------------------------
class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.ev = Evaluator()

    def test_required_decel(self):
        self.assertEqual(required_decel(30, 0.0), 0.0)
        self.assertTrue(math.isinf(required_decel(2.0, 5.0)))
        self.assertAlmostEqual(required_decel(22, 10.0), 100 / (2 * 20))

    def test_unavoidable_collision_is_not_a_planner_failure(self):
        s = Scenario(ego_speed=20, lead_speed=0, initial_gap=10, dropout_prob=0, ghost_prob=0, seed=1)
        a = self.ev.evaluate(s)
        self.assertTrue(a.collision)
        self.assertEqual(a.collision_class, 'unavoidable')
        self.assertNotIn('collision', a.failure_kinds)
        self.assertTrue(a.oracle_max_brake_collides)

    def test_no_braking_collision_is_avoidable_and_missed(self):
        ep = SafetySimSUT().run(CLOSING, Interventions(policy='none'))
        a = assess(ep)
        self.assertTrue(a.collision)
        self.assertEqual(a.collision_class, 'avoidable')
        self.assertIn('collision', a.failure_kinds)
        self.assertGreaterEqual(a.missed_hazards, 1)
        self.assertEqual(a.max_state, 'DRIVE')
        self.assertIsNone(a.brake_reaction_time)
        json.dumps(a.to_dict())

    def test_competent_braking_passes_with_margin(self):
        ep = SafetySimSUT().run(CLOSING, Interventions(policy='reference'))
        a = assess(ep)
        self.assertFalse(a.collision)
        self.assertGreater(a.min_gap, 0.5)

    def test_unnecessary_hard_braking_detected(self):
        s = Scenario(ego_speed=10, lead_speed=20, initial_gap=40, dropout_prob=0, ghost_prob=0, seed=1)   # lead pulls away
        a = assess(SafetySimSUT().run(s, Interventions(policy='max_brake')))
        self.assertGreater(a.unnecessary_braking_s, 0.3)
        self.assertIn('unnecessary_braking', a.failure_kinds)

    def test_cached_and_counted(self):
        ev = Evaluator()
        ev.evaluate(CLOSING)
        ev.evaluate(CLOSING)
        self.assertEqual((ev.n_sims, ev.n_cache_hits), (1, 1))
        ev2 = Evaluator(max_sims=1)
        ev2.evaluate(CLOSING)
        with self.assertRaises(BudgetExceeded):
            ev2.evaluate(CLOSING.replace(seed=2))


# --- search ------------------------------------------------------------------------------------------------------------------
class TestSearch(unittest.TestCase):
    def test_budget_and_determinism(self):
        for method in ('random', 'cmaes'):
            r1 = run_search(method, Evaluator(), 30, 3)
            r2 = run_search(method, Evaluator(), 30, 3)
            r3 = run_search(method, Evaluator(), 30, 4)
            self.assertEqual(r1.sims_used, 30)
            self.assertEqual([e.scenario for e in r1.evaluations], [e.scenario for e in r2.evaluations])
            self.assertNotEqual([e.scenario for e in r1.evaluations], [e.scenario for e in r3.evaluations])
            self.assertEqual([e.assessment.objective for e in r1.evaluations], [e.assessment.objective for e in r2.evaluations])

    def test_cmaes_minimises_a_smooth_function(self):
        rng = np.random.default_rng(0)
        es = CMAES(np.full(6, 0.8), 0.3, rng, popsize=12)
        best = 1e9
        for _ in range(60):
            xs = es.ask()
            fs = [float(np.sum((np.clip(x, 0, 1) - 0.3) ** 2)) for x in xs]
            es.tell(xs, fs)
            best = min(best, min(fs))
        self.assertLess(best, 1e-3)

    def test_search_does_not_leave_budget_cap_behind(self):
        ev = Evaluator()
        run_search('random', ev, 5, 1)
        self.assertIsNone(ev.max_sims)

    def test_findings_are_deduplicated_by_necessary_faults(self):
        ev = Evaluator(default_sut_factory('kalman', NEVER_BRAKE))
        res = run_search('random', ev, 40, 9)
        findings = group_failures(ev, res.evaluations)
        n_fail = len(res.failures)
        self.assertGreater(n_fail, 5)
        self.assertLess(len(findings), n_fail)                                   # duplicates merged
        self.assertTrue(all(f.necessary_faults == () for f in findings if f.kind == 'collision'))
        s = CLOSING
        self.assertEqual(signature(s, 'collision', ['b', 'a']), signature(s.replace(ego_speed=15), 'collision', ['a', 'b']))


# --- minimisation ---------------------------------------------------------------------------------------------------------------
class _StubEval:
    """Evaluator stand-in: failure iff predicate(scenario)."""
    class A:
        def __init__(self, fail):
            self.failure_kinds = ['collision'] if fail else []
            self.failed = fail

    def __init__(self, pred):
        self.pred, self.calls = pred, 0

    def evaluate(self, s, iv=None):
        self.calls += 1
        return self.A(self.pred(s))


class TestMinimize(unittest.TestCase):
    ALL = Scenario(perception_latency_s=0.3, distance_noise_std_m=1.0, dropout_prob=0.4, detection_loss_start_s=1.0,
                   detection_loss_duration_s=1.0, ttc_error_factor=2.0, ghost_prob=0.1)

    def test_ddmin_finds_the_one_minimal_set(self):
        pred = lambda s: s.perception_latency_s > 0 and s.dropout_prob > 0       # needs exactly these two
        m = Minimizer(_StubEval(pred), 'collision')
        self.assertEqual(sorted(m.ddmin_faults(self.ALL)), ['perception_latency', 'sensor_dropout'])

    def test_ddmin_single_and_none(self):
        m = Minimizer(_StubEval(lambda s: s.detection_loss_duration_s > 0), 'collision')
        self.assertEqual(m.ddmin_faults(self.ALL), ['detection_loss'])
        m = Minimizer(_StubEval(lambda s: True), 'collision')
        self.assertEqual(m.ddmin_faults(self.ALL), [])
        with self.assertRaises(ValueError):
            Minimizer(_StubEval(lambda s: False), 'collision').ddmin_faults(self.ALL)

    def test_bisect_reduces_severity_to_the_threshold(self):
        m = Minimizer(_StubEval(lambda s: s.perception_latency_s >= 0.2), 'collision')
        out = m.bisect(self.ALL, 'perception_latency_s', 0.0)
        self.assertAlmostEqual(out.perception_latency_s, 0.2, delta=0.11)
        self.assertGreaterEqual(out.perception_latency_s, 0.2)               # never accepts a non-failing value

    def test_end_to_end_on_the_real_simulator(self):
        ev = Evaluator(default_sut_factory('kalman', NEVER_BRAKE))
        s = CLOSING.replace(perception_latency_s=0.3, distance_noise_std_m=1.0, detection_loss_start_s=2.0,
                            detection_loss_duration_s=1.0, ttc_error_factor=2.0, ghost_prob=0.1, dropout_prob=0.2, ego_speed=18)
        self.assertTrue(ev.evaluate(s).failed)
        res = minimize(ev, s)
        self.assertEqual(res.necessary_faults, [])                               # a planner that never brakes needs no faults
        self.assertEqual(res.minimized.active_faults(), [])
        self.assertTrue(ev.evaluate(res.minimized).failed)
        self.assertLess(res.distance_after, res.distance_before)
        self.assertTrue(res.reproducible['deterministic_replay'])
        self.assertEqual(res.reproducible['status'], 'reproducible')
        self.assertTrue(res.changes)
        json.dumps(res.to_dict())

    def test_minimizer_refuses_non_failing_scenarios(self):
        with self.assertRaises(ValueError):
            minimize(Evaluator(), Scenario(ego_speed=6, lead_speed=20, dropout_prob=0, ghost_prob=0))

    def test_derived_seeds(self):
        self.assertEqual(derived_seeds(5, 1), [5])
        self.assertEqual(len(set(derived_seeds(5, 10))), 10)
        self.assertEqual(derived_seeds(5, 4), derived_seeds(5, 4))


# --- diagnosis --------------------------------------------------------------------------------------------------------------------
class TestDiagnose(unittest.TestCase):
    def test_wilson_and_shapley(self):
        lo, hi = wilson(0, 8)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.1)
        self.assertEqual(wilson(0, 0), (0.0, 1.0))
        # additive game: Shapley recovers the weights
        v = {frozenset(): 0.0, frozenset('a'): 0.3, frozenset('b'): 0.5, frozenset('ab'): 0.8}
        sh = shapley(['a', 'b'], v)
        self.assertAlmostEqual(sh['a'], 0.3)
        self.assertAlmostEqual(sh['b'], 0.5)
        # interaction (needs both): credit split equally, sums to the total
        v = {frozenset(): 0.0, frozenset('a'): 0.0, frozenset('b'): 0.0, frozenset('ab'): 1.0}
        sh = shapley(['a', 'b'], v)
        self.assertAlmostEqual(sh['a'], 0.5)
        self.assertAlmostEqual(sum(sh.values()), 1.0)

    def test_planner_logic_is_blamed_when_perception_is_irrelevant(self):
        ev = Evaluator(default_sut_factory('kalman', NEVER_BRAKE))
        d = diagnose(ev, CLOSING.replace(ego_speed=18, perception_latency_s=0.3), n_seeds=4)
        hyp = {h['component']: h for h in d.hypotheses}
        self.assertEqual(hyp['planner_logic']['status'], 'supported')
        for c in ('object_detection', 'distance_estimation', 'ttc_estimation', 'reaction_latency'):
            self.assertNotEqual(hyp[c]['status'], 'supported')
        self.assertEqual(d.planner['failure_rate_ideal_perception'], 1.0)
        self.assertGreater(d.sims, 16 * 4)
        json.dumps(d.to_dict())

    def test_planted_latency_is_isolated(self):
        ev = Evaluator()
        s = Scenario(ego_speed=14, lead_speed=14, initial_gap=45, lead_decel=6, brake_onset_s=2, dropout_prob=0,
                     ghost_prob=0, perception_latency_s=0.5, seed=3)
        if ev.evaluate(s.replace(perception_latency_s=0.0)).failed or not ev.evaluate(s).failed:
            self.skipTest('production planner thresholds changed: the planted-latency precondition no longer holds')
        d = diagnose(ev, s, n_seeds=6)
        hyp = {h['component']: h for h in d.hypotheses}
        self.assertEqual(hyp['reaction_latency']['status'], 'supported')
        self.assertNotEqual(hyp['planner_logic']['status'], 'supported')

    def test_analytic_mode_reports_ttc_as_untestable(self):
        ev = Evaluator(default_sut_factory('analytic', NEVER_BRAKE))
        d = diagnose(ev, CLOSING.replace(ego_speed=18), n_seeds=3)
        self.assertIn('ttc_estimation', d.untestable)
        self.assertEqual({h['component']: h['status'] for h in d.hypotheses}['ttc_estimation'], 'not testable')

    def test_diagnosis_requires_a_failure(self):
        with self.assertRaises(ValueError):
            diagnose(Evaluator(), Scenario(ego_speed=6, lead_speed=20, dropout_prob=0, ghost_prob=0))

    def test_interventions_registry(self):
        sc, iv = apply(CLOSING.replace(dropout_prob=0.3, perception_latency_s=0.3), ['ideal_detection', 'no_latency', 'oracle_ttc'])
        self.assertEqual((sc.dropout_prob, sc.perception_latency_s, iv.oracle_ttc), (0.0, 0.0, True))
        with self.assertRaises(KeyError):
            apply(CLOSING, ['nope'])
        self.assertIn('reference_policy', INTERVENTIONS)


# --- regression ---------------------------------------------------------------------------------------------------------------------
class TestRegress(unittest.TestCase):
    def make_suite(self, d: Path):
        promote(CLOSING.replace(name='closing_fail', ego_speed=18, initial_gap=70), d, 'closing_fail')
        promote(Scenario(ego_speed=10, lead_speed=10, initial_gap=60, dropout_prob=0, ghost_prob=0, seed=2, name='easy'), d, 'easy')

    def test_identity_passes_and_mutant_is_flagged(self):
        with tempfile.TemporaryDirectory() as t:
            suite = Path(t) / 'suite'
            self.make_suite(suite)
            ev = Evaluator()
            base = create_baseline(ev, suite, 'v1', 'test')
            self.assertEqual(base['baseline_version'], 'v1')
            ok = run_regression(Evaluator(), suite, base)
            self.assertTrue(ok['passed'])
            bad = run_regression(Evaluator(default_sut_factory('kalman', NEVER_BRAKE)), suite, base)
            self.assertFalse(bad['passed'])
            statuses = {r['scenario']: r['status'] for r in bad['rows']}
            self.assertEqual(statuses['closing_fail'], 'new_failure')            # passed before, collides with a planner that never brakes
            md = to_markdown(bad)
            self.assertIn('FAIL', md)
            json.dumps(bad)

    def test_stale_and_missing_baseline_entries(self):
        with tempfile.TemporaryDirectory() as t:
            suite = Path(t) / 'suite'
            self.make_suite(suite)
            base = create_baseline(Evaluator(), suite, 'v1')
            promote(CLOSING.replace(name='closing_fail', ego_speed=17, initial_gap=70), suite, 'closing_fail')    # edited
            promote(CLOSING.replace(name='new_one', seed=9), suite, 'new_one')
            res = run_regression(Evaluator(), suite, base)
            st = {r['scenario']: r['status'] for r in res['rows']}
            self.assertEqual(st['closing_fail'], 'stale_baseline')
            self.assertEqual(st['new_one'], 'missing_baseline')
            self.assertFalse(res['passed'])                                      # stale baseline is a violation
            (suite / 'closing_fail.json').unlink()
            res = run_regression(Evaluator(), suite, base)
            self.assertIn('removed_from_suite', {r['status'] for r in res['rows']})
            strict = run_regression(Evaluator(), suite, base, RegressionCriteria(strict=True))
            self.assertIn('new_one', strict['violations'])

    def test_fixed_failure_is_reported_not_flagged(self):
        with tempfile.TemporaryDirectory() as t:
            suite = Path(t) / 'suite'
            promote(CLOSING.replace(name='f', ego_speed=18, initial_gap=70), suite, 'f')
            base = create_baseline(Evaluator(default_sut_factory('kalman', NEVER_BRAKE)), suite, 'broken')
            self.assertTrue(base['scenarios']['f']['result']['failure_kinds'])
            res = run_regression(Evaluator(), suite, base)
            self.assertEqual(res['rows'][0]['status'], 'fixed')
            self.assertTrue(res['passed'])

    def test_bad_inputs(self):
        with self.assertRaises(ScenarioValidationError):
            load_suite(Path('/nonexistent/dir'))
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ScenarioValidationError):
                load_suite(Path(t))
            with self.assertRaises(ScenarioValidationError):
                run_regression(Evaluator(), Path(t), {'baseline_format': 99, 'scenarios': {}})


# --- CLI -------------------------------------------------------------------------------------------------------------------------------
class TestCLI(unittest.TestCase):
    def test_search_minimize_diagnose_regress_roundtrip(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            rc, out = quiet(cli.main, ['search', '--budget', '25', '--seed', '1', '--out', str(t / 's'), '--top', '2'])
            self.assertEqual(rc, 0)
            self.assertTrue((t / 's' / 'search_results.json').exists())
            data = json.loads((t / 's' / 'search_results.json').read_text())
            self.assertEqual(data['sims_used'], 25)
            files = sorted((t / 's' / 'failures').glob('*.json'))
            self.assertTrue(files)
            rc, out = quiet(cli.main, ['minimize', '--scenario', str(files[0]), '--out', str(t / 'm')])
            self.assertEqual(rc, 0)
            self.assertIn('reproducibility', out)
            self.assertTrue((t / 'm' / 'minimized_scenario.json').exists())
            rc, out = quiet(cli.main, ['diagnose', '--scenario', str(t / 'm' / 'minimized_scenario.json'), '--seeds', '3',
                                       '--out', str(t / 'd.json')])
            self.assertEqual(rc, 0)
            suite = t / 'suite'
            rc, _ = quiet(cli.main, ['promote', '--scenario', str(t / 'm' / 'minimized_scenario.json'), '--suite', str(suite)])
            self.assertEqual(rc, 0)
            rc, _ = quiet(cli.main, ['baseline', '--suite', str(suite), '--out', str(t / 'b.json'), '--version', 'v1'])
            self.assertEqual(rc, 0)
            rc, _ = quiet(cli.main, ['regress', '--suite', str(suite), '--baseline', str(t / 'b.json'), '--out', str(t / 'r')])
            self.assertEqual(rc, 0)
            self.assertTrue((t / 'r' / 'regression_results.json').exists())
            self.assertTrue((t / 'r' / 'regression_summary.md').exists())
            rc, _ = quiet(cli.main, ['regress', '--suite', str(suite), '--baseline', str(t / 'b.json'), '--out', str(t / 'r2'),
                                     '--override', 'SLOW_BASE_DIST=0', '--override', 'SLOW_HEADWAY=0', '--override', 'STOP_BASE_DIST=0',
                                     '--override', 'TTC_SLOW=0', '--override', 'TTC_STOP=0', '--override', 'EMERGENCY_DECEL=1e9'])
            self.assertEqual(rc, 1)                                              # nonzero exit on regression

    def test_error_exit_codes(self):
        with tempfile.TemporaryDirectory() as t:
            bad = Path(t) / 'bad.json'
            bad.write_text('{"ego_speed": 999}')
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(['minimize', '--scenario', str(bad)]), 2)
                self.assertEqual(cli.main(['minimize', '--scenario', str(Path(t) / 'nope.json')]), 2)
                self.assertEqual(cli.main(['regress', '--suite', t, '--baseline', str(Path(t) / 'nope.json')]), 2)
                self.assertEqual(cli.main(['search', '--budget', '3', '--override', 'NOT_A_KEY=1', '--out', t]), 2)
            ok = Path(t) / 'ok.json'
            Scenario(ego_speed=6, lead_speed=20, dropout_prob=0, ghost_prob=0).save(ok)
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(['minimize', '--scenario', str(ok)]), 2)          # does not fail: nothing to minimise

    def test_demo_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as t:
            rc, out = quiet(cli.main, ['demo', '--budget', '40', '--seed', '42', '--no-gif', '--out', t])
            self.assertEqual(rc, 0)
            self.assertIn('regression check', out)
            rep = (Path(t) / 'investigation' / 'report.md').read_text()
            for h in ('## 1. Failure summary', '## 4. Minimized failure conditions', '## 6. Evidence from controlled interventions',
                      '## 8. Recommended engineering changes', '## 9. Reproduce'):
                self.assertIn(h, rep)
            self.assertTrue((Path(t) / 'investigation' / 'telemetry.png').exists())
            self.assertTrue((Path(t) / 'investigation' / 'counterfactuals.png').exists())

    def test_demo_reports_honestly_when_nothing_fails(self):
        # a search box where the production planner cannot fail: slow ego, huge gaps, no lead braking
        from autonomy_redteam import demo as demo_mod
        orig = demo_mod.run_search
        def fake(method, ev, budget, seed, *a, **k):
            return orig(method, ev, budget, seed, {'ego_speed': (6, 7), 'initial_gap': (70, 80), 'lead_decel': (0, 0.0),
                                                   'lead_speed': (19, 20)}, fault_prob=0.0)
        demo_mod.run_search = fake
        try:
            with tempfile.TemporaryDirectory() as t:
                rc, out = quiet(demo_mod.run_demo, Path(t), 20, 1, False)
            self.assertEqual(rc, 0)
            self.assertIn('No failure found', out)
        finally:
            demo_mod.run_search = orig


# --- agent ---------------------------------------------------------------------------------------------------------------------------------
def _finish_call(summary='s', hyps=(), sugg=(), cid='c'):
    return LLMResponse('', [ToolCall(cid, 'finish', {'summary': summary, 'hypotheses': list(hyps),
                                                    'suggested_scenarios': list(sugg)})], 60, 40)


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.ev = Evaluator(default_sut_factory('kalman', NEVER_BRAKE))
        self.sc = CLOSING.replace(ego_speed=18)

    def test_schema_validator(self):
        sch = {'type': 'object', 'additionalProperties': False, 'required': ['a'],
               'properties': {'a': {'type': 'integer', 'minimum': 1, 'maximum': 3}, 'b': {'type': 'string', 'enum': ['x']}}}
        validate({'a': 2, 'b': 'x'}, sch)
        for bad in ({}, {'a': 0}, {'a': 4}, {'a': 1.5}, {'a': True}, {'a': 1, 'c': 1}, {'a': 1, 'b': 'y'}, [], {'a': float('nan')}):
            with self.assertRaises(SchemaError):
                validate(bad, sch)

    def test_tools_validate_inputs_and_never_decide_metrics_themselves(self):
        tb = ToolBox(self.ev, self.sc)
        r = tb.call('get_scenario', {})
        self.assertEqual(r['outcome']['collision'], self.ev.evaluate(self.sc).collision)
        self.assertIn('error', tb.call('run_scenario_variant', {'overrides': {'ego_speed': 999}}))
        self.assertIn('error', tb.call('run_scenario_variant', {'overrides': {'not_a_param': 1}}))
        self.assertIn('error', tb.call('run_counterfactual', {'interventions': ['rm -rf']}))
        self.assertIn('error', tb.call('no_such_tool', {}))
        self.assertIn('error', tb.call('get_telemetry_window', {'t_start': 5, 't_end': 'x'}))
        n = tb.call('get_telemetry_window', {'t_start': 0, 't_end': 30})
        self.assertLessEqual(len(n['rows']), 40)
        v = tb.call('run_scenario_variant', {'overrides': {'ego_speed': 18.0}, 'n_seeds': 3})
        self.assertEqual(v['n_seeds'], 3)
        self.assertEqual(v['failure_rate'], 1.0)

    def test_limits_are_enforced(self):
        tb = ToolBox(self.ev, self.sc, Limits(max_tool_calls=2))
        tb.call('get_scenario', {})
        tb.call('get_scenario', {})
        with self.assertRaises(LimitExceeded):
            tb.call('get_scenario', {})
        tb = ToolBox(Evaluator(default_sut_factory('kalman', NEVER_BRAKE)), self.sc, Limits(max_sims=4))
        out = tb.call('run_scenario_variant', {'overrides': {'ego_speed': 18.0}, 'n_seeds': 8})
        self.assertIn('error', out)                                              # would exceed the simulation budget
        self.assertLessEqual(tb.sims_used, 4)
        tb = ToolBox(self.ev, self.sc, Limits(max_seconds=1.0), clock=iter([0.0, 5.0, 5.0, 5.0]).__next__)
        with self.assertRaises(LimitExceeded):
            tb.call('get_scenario', {})
        tb = ToolBox(self.ev, self.sc, Limits(max_tokens=10))
        tb.add_tokens(11)
        with self.assertRaises(LimitExceeded):
            tb.call('get_scenario', {})

    def test_budget_cap_is_restored(self):
        investigate_with_agent(self.ev, self.sc, None, Limits(max_sims=300))
        self.assertIsNone(self.ev.max_sims)

    def test_finish_rejects_unknown_evidence_and_invalid_suggestions(self):
        tb = ToolBox(self.ev, self.sc)
        bad = tb.call('finish', {'summary': 's', 'hypotheses': [{'component': 'planner_logic', 'claim': 'c', 'confidence': 'low',
                                                                   'evidence_runs': ['R99']}], 'suggested_scenarios': []})
        self.assertIn('error', bad)
        self.assertIsNone(tb.finished)
        bad = tb.call('finish', {'summary': 's', 'hypotheses': [], 'suggested_scenarios': [
            {'overrides': {'ego_speed': 18.0, 'lead_decel': 5.0, 'brake_onset_s': 5.0}, 'rationale': 'r'}]})
        self.assertNotIn('error', bad)                                           # valid combination accepted

    def test_scripted_llm_full_flow(self):
        script = [
            LLMResponse('', [ToolCall('1', 'get_scenario', {})], 100, 20),
            LLMResponse('', [ToolCall('2', 'run_counterfactual', {'interventions': ['ideal_detection'], 'n_seeds': 3})], 200, 30),
            lambda msgs: _finish_call('The planner fails regardless of detection.',
                                      [{'component': 'planner_logic', 'claim': 'fails with ideal detection', 'confidence': 'medium',
                                        'evidence_runs': ['R2']}],
                                      [{'overrides': {'ego_speed': 16.0}, 'rationale': 'slower ego'}]),
        ]
        llm = ScriptedLLM(script)
        rep = investigate_with_agent(self.ev, self.sc, llm, Limits())
        self.assertEqual(rep.stopped_reason, 'finished')
        self.assertEqual(rep.hypotheses[0]['component'], 'planner_logic')
        self.assertEqual(len(rep.suggestion_results), 1)
        self.assertIn('failed', rep.suggestion_results[0])                      # replayed deterministically, not claimed by the LLM
        self.assertEqual(rep.usage['tokens'], 450)
        self.assertFalse(rep.usage['tokens_estimated'])
        self.assertEqual(rep.usage['tool_calls'], 3)
        self.assertIn('planner_logic', rep.markdown())
        # tool results given to the model carry the deterministic numbers
        tool_msgs = [m for m in llm.seen[-1] if m['role'] == 'tool']
        self.assertIn('failure_rate', tool_msgs[1]['content'])

    def test_scripted_llm_cannot_exceed_tool_budget(self):
        calls = [LLMResponse('', [ToolCall(str(i), 'get_scenario', {})], 10, 10) for i in range(10)]
        rep = investigate_with_agent(self.ev, self.sc, ScriptedLLM(calls), Limits(max_tool_calls=3))
        self.assertIn('tool-call limit', rep.stopped_reason)
        self.assertEqual(rep.usage['tool_calls'], 3)

    def test_llm_errors_and_non_tool_replies_end_gracefully(self):
        rep = investigate_with_agent(self.ev, self.sc, ScriptedLLM([]), Limits())
        self.assertIn('LLM error', rep.stopped_reason)
        rep = investigate_with_agent(self.ev, self.sc, ScriptedLLM([LLMResponse('just text')]), Limits())
        self.assertIn('no tool call', rep.stopped_reason)

    def test_deterministic_fallback_without_key(self):
        rep = investigate_with_agent(self.ev, self.sc, None, Limits())
        self.assertEqual(rep.mode, 'deterministic')
        self.assertEqual(rep.stopped_reason, 'finished')
        self.assertTrue(rep.hypotheses)
        self.assertTrue(any(h['component'] == 'planner_logic' for h in rep.hypotheses))
        self.assertLessEqual(rep.usage['simulations'], Limits().max_sims)
        self.assertEqual(rep.usage['tokens'], 0)
        json.dumps(rep.to_dict())

    def test_deterministic_investigator_respects_a_tiny_budget(self):
        rep = investigate_with_agent(self.ev, self.sc, None, Limits(max_tool_calls=2))
        self.assertIn('limit', rep.stopped_reason)

    def test_client_selection_from_env(self):
        self.assertIsNone(client_from_env({}))
        with self.assertRaises(LLMError):
            client_from_env({'ANTHROPIC_API_KEY': 'k'})                          # key but no model: refuse to guess
        c = client_from_env({'ANTHROPIC_API_KEY': 'k', 'REDTEAM_LLM_MODEL': 'm'})
        self.assertEqual(c.name, 'anthropic')
        c = client_from_env({'OPENAI_API_KEY': 'k', 'REDTEAM_LLM_MODEL': 'm', 'REDTEAM_LLM_BASE_URL': 'http://x/v1/'})
        self.assertEqual((c.name, c.base_url), ('openai-compatible', 'http://x/v1'))
        with self.assertRaises(LLMError):
            client_from_env({'REDTEAM_LLM_PROVIDER': 'nope', 'OPENAI_API_KEY': 'k'})

    def test_http_clients_build_provider_payloads_and_parse_replies(self):
        seen = {}

        def fake_post(url, headers, body, timeout):
            seen['url'], seen['headers'], seen['body'] = url, headers, body
            if 'anthropic' in url:
                return {'content': [{'type': 'text', 'text': 'hi'}, {'type': 'tool_use', 'id': 't1', 'name': 'get_scenario', 'input': {}}],
                        'usage': {'input_tokens': 5, 'output_tokens': 7}}
            return {'choices': [{'message': {'content': None, 'tool_calls': [
                {'id': 'c1', 'function': {'name': 'get_scenario', 'arguments': '{}'}}]}}],
                'usage': {'prompt_tokens': 3, 'completion_tokens': 4}}
        orig = llm_mod._post
        llm_mod._post = fake_post
        try:
            msgs = [{'role': 'user', 'content': 'go'},
                    {'role': 'assistant', 'content': '', 'tool_calls': [ToolCall('a', 'get_scenario', {})]},
                    {'role': 'tool', 'tool_call_id': 'a', 'name': 'get_scenario', 'content': '{}'}]
            from autonomy_redteam.agent import TOOL_SPECS
            r = llm_mod.AnthropicClient('SECRET', 'm', 'https://api.anthropic.com').chat('sys', msgs, TOOL_SPECS)
            self.assertEqual((r.input_tokens, r.output_tokens, r.tool_calls[0].name), (5, 7, 'get_scenario'))
            self.assertEqual(seen['headers']['x-api-key'], 'SECRET')
            self.assertEqual(seen['body']['messages'][-1]['content'][0]['type'], 'tool_result')
            r = llm_mod.OpenAICompatClient('SECRET', 'm').chat('sys', msgs, TOOL_SPECS)
            self.assertEqual(r.tool_calls[0].id, 'c1')
            self.assertEqual(seen['body']['messages'][-1]['role'], 'tool')
            self.assertEqual(seen['body']['messages'][0]['role'], 'system')
            self.assertNotIn('SECRET', json.dumps(seen['body']))                 # the key goes in a header only
        finally:
            llm_mod._post = orig

    def test_no_secret_literals_in_package(self):
        import re
        for p in (ROOT / 'autonomy_redteam').rglob('*.py'):
            self.assertIsNone(re.search(r'sk-[A-Za-z0-9]{20,}', p.read_text()), p)


if __name__ == '__main__':
    unittest.main()
