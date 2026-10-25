# Autonomy Red Team evaluation ({'budget': 150, 'trials': 10, 'min_limit': 40, 'plant': 12, 'null': 20, 'suite_budget': 300, 'holdout': 300})

## A. Search at equal budget (mean +- std over trials; same seeds for both methods)

| regime | method | failing / budget | sims to first failure | unique reproducible findings | worst impact (m/s) |
|---|---|---|---|---|---|
| full_box | random | 51.3 +- 4.832 / 150 | 3.3 +- 1.337 | 14.6 +- 2.875 | 17.049 +- 1.792 |
| full_box | cmaes | 131.3 +- 13.466 / 150 | 2.4 +- 2.797 | 12.8 +- 4.264 | 19.975 +- 0.079 |
| rare_box | random | 25.0 +- 2.309 / 150 | 7.8 +- 6.408 | 8.8 +- 0.919 | 13.593 +- 1.221 |
| rare_box | cmaes | 125.0 +- 13.679 / 150 | 3.3 +- 3.917 | 13.3 +- 3.561 | 16.0 +- 0.0 |

full_box: CMA-ES minus random, unique reproducible findings per trial: mean -1.8 (95% bootstrap CI [-4.0, 0.2]); cmaes better in 3, tie 1, random better 6. Distinct signatures over all trials: random 51, cmaes 43, both 68.

rare_box: CMA-ES minus random, unique reproducible findings per trial: mean 4.5 (95% bootstrap CI [2.0, 6.8]); cmaes better in 9, tie 0, random better 1. Distinct signatures over all trials: random 28, cmaes 36, both 46.

## B. Minimisation

- 40 distinct failures minimised; still fail on a fresh SUT: 100%; deterministic replay: 100%
- active perturbations 3.475 -> 1.85 (mean); distance to boring scenario 3.997 -> 2.391; replays 75.375 per failure
- failure rate over 20 fresh noise seeds: original 0.761, minimised 0.505; seed-fragile (<25%): before 5%, after 22%

## C. Diagnosis (planted-fault recovery)

- 49 planted cases: top-1 Shapley accuracy 0.755, planted component labelled *supported* in 0.959, precision of *supported* labels 0.81, planner-logic false flags 0.0

| planted fault | cases | top-1 | expected supported | planner-logic false flag | could not construct |
|---|---|---|---|---|---|
| perception_latency | 12 | 0.25 | 0.917 | 0.0 | 0 |
| detection_loss | 12 | 1.0 | 1.0 | 0.0 | 0 |
| sensor_dropout | 12 | 0.917 | 1.0 | 0.0 | 0 |
| distance_noise | 11 | 1.0 | 1.0 | 0.0 | 1 |
| ttc_error | 2 | 0.0 | 0.5 | 0.0 | 10 |

No injected fault (20 failing scenarios): planner logic labelled supported in 0.7, a perception component falsely labelled supported in 0.3.

## D. Regression detection

- suite: 8 known failures + 6 near-boundary passes; holdout 300 scenarios (label threshold +-6 failing scenarios)
- harmful mutants detected: 4/4; non-harmful flagged: 8/15; identity flagged: False

| mutant | intent | holdout delta | label | flagged | violations |
|---|---|---|---|---|---|
| identity | none | +0 | neutral | False | - |
| TTC_SLOW=1.5 | harmful | +19 | harmful | True | fail_01_collision_lead_braking, fail_02_collision_lead_braking, fail_03_collision_lead_braking, fail_04_collision_lead_braking |
| TTC_STOP=0.5 | harmful | +4 | neutral | True | fail_04_collision_lead_braking, fail_05_collision_steady, fail_06_collision_lead_braking, fail_07_collision_steady |
| STOP_BASE_DIST=0.5 | harmful | +5 | neutral | True | pass_01_lead_braking, pass_02_lead_braking |
| SLOW_HEADWAY=0.3 | harmful | +18 | harmful | True | pass_01_lead_braking, pass_02_lead_braking, pass_03_steady |
| ESCALATE_FRAMES=5 | harmful | +40 | harmful | True | fail_01_collision_lead_braking, fail_02_collision_lead_braking, fail_03_collision_lead_braking, fail_04_collision_lead_braking |
| EMERGENCY_DECEL=20 | harmful | +5 | neutral | True | pass_01_lead_braking, pass_02_lead_braking |
| SLOW_BASE_DIST=2 | harmful | +7 | harmful | True | pass_02_lead_braking, pass_03_steady |
| MIN_CLOSING_SPEED=3 | harmful | +0 | neutral | False | - |
| DIST_EMA_ALPHA=0.45 | benign | -1 | neutral | True | fail_08_collision_lead_braking |
| RELEASE_FRAMES=6 | benign | -5 | neutral | True | fail_05_collision_steady |
| TRACK_MAX_AGE=6 | benign | +0 | neutral | False | - |
| TTC_STOP=1.05 | benign | +0 | neutral | False | - |
| STOP_BASE_DIST=2.1 | benign | +0 | neutral | False | - |
| SLOW_BASE_DIST=6.2 | benign | +0 | neutral | False | - |
| TTC_SLOW=4.0 | improvement | -9 | improvement | True | fail_05_collision_steady |
| STOP_BASE_DIST=4 | improvement | +2 | neutral | True | fail_05_collision_steady |
| SLOW_HEADWAY=1.5 | improvement | -11 | improvement | True | fail_08_collision_lead_braking |
| TTC_STOP=1.5 | improvement | -11 | improvement | False | - |

## E. Cost

Total 364.5s on one CPU core (search 90.2s, minimisation 22.5s, diagnosis 217.2s, regression 34.5s).

## F. Agent-guided

not run: no API key configured; the deterministic investigator is exercised in tests, not compared as a search method
