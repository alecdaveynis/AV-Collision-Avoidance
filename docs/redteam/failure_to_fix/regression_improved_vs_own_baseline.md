# Safety regression report: PASS (ok)

- Baseline: `improved-v1` (commit aeba567), suite `docs/redteam/failure_to_fix/suite`, 12 scenarios x 12 paired seeds, 144 simulations
- Baseline SUT: `{'name': 'safety_sim', 'planner': 'improved', 'planner_mode': 'kalman', 'config_overrides': {}, 'dt': 0.1, 'max_decel': 8.0, 'actuator_lag': 0.2}`
- Current SUT:  `{'name': 'safety_sim', 'planner': 'improved', 'planner_mode': 'kalman', 'config_overrides': {}, 'dt': 0.1, 'max_decel': 8.0, 'actuator_lag': 0.2}`
- Counts: within_tolerance: 12
- Suite level (exact McNemar over paired scenario/seed outcomes): 0 newly failing, 0 newly passing, one-sided p=1.0; no meaningful suite-level change
  - pooled sign test, minimum gap: 0 pairs worse vs 0 better (beyond tolerance), p=1.0
  - pooled sign test, impact speed: 0 pairs worse vs 0 better (beyond tolerance), p=1.0
  - pooled sign test, unnecessary braking time: 0 pairs worse vs 0 better (beyond tolerance), p=1.0

| scenario | role | tier | safety failures (base -> now) | details |
|---|---|---|---|---|
| `fail_01_collision_lead_braking` | safe_contract | within_tolerance | 0/12 -> 0/12 |  |
| `fail_02_collision_lead_braking` | safe_contract | within_tolerance | 0/12 -> 0/12 |  |
| `fail_03_collision_lead_braking` | safe_contract | within_tolerance | 0/12 -> 0/12 |  |
| `fail_04_collision_lead_braking` | safe_contract | within_tolerance | 0/12 -> 0/12 |  |
| `known_failure_gentle_lead_braking` | safe_contract | within_tolerance | 0/12 -> 0/12 |  |
| `pass_01_lead_braking` | safe_contract | within_tolerance | 0/12 -> 0/12 |  |
| `pass_02_steady` | safe_contract | within_tolerance | 0/12 -> 0/12 |  |
| `pass_03_parked` | safe_contract | within_tolerance | 0/12 -> 0/12 |  |
| `safe_gentle_brake_large_gap` | safe_contract | within_tolerance | 0/12 -> 0/12 |  |
| `safe_hard_brake_large_gap` | safe_contract | within_tolerance | 0/12 -> 0/12 |  |
| `safe_parked_lead_far` | safe_contract | within_tolerance | 0/12 -> 0/12 |  |
| `safe_slow_lead_approach` | safe_contract | within_tolerance | 0/12 -> 0/12 |  |
