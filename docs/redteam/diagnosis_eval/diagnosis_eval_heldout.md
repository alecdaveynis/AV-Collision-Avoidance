# Diagnosis evaluation (heldout, seed 9011)

50 planted-fault cases, 30 no-fault controls; could not construct: {'ttc_error': 10}. All scoring uses the planted label only after the fact.

## Legacy engine (Shapley share, status *supported*)

- top-1 Shapley accuracy 0.94; planted component labelled *supported* in 0.92; precision of *supported* labels 0.852; planner-logic false flags (planted cases) 0.0
- no-fault controls: planner logic *supported* in 0.633; a perception component falsely *supported* in 0.367

| planted fault | n | top-1 | planted supported |
|---|---|---|---|
| perception_latency | 12 | 0.833 | 0.917 |
| detection_loss | 12 | 1.0 | 1.0 |
| sensor_dropout | 12 | 1.0 | 0.833 |
| distance_noise | 12 | 1.0 | 0.917 |
| ttc_error | 2 | 0.5 | 1.0 |

## New engine (temporal evidence + controlled replacement, explicit joint / inconclusive)

- planted component is the unique primary root cause: **0.92**; correct but reported jointly with another component: 0.02; wrong: 0.0; inconclusive: 0.06
- false positives: any component other than the planted one flagged primary/contributing in 0.02 of planted cases (a perception component in 0.02; planner logic in 0.0)
- no-fault controls: planner logic the unique primary cause in 0.633; a perception component flagged in **0.033**; inconclusive 0.333

| planted fault | n | correct (unique) | correct, joint | wrong | inconclusive |
|---|---|---|---|---|---|
| perception_latency | 12 | 1.0 | 0.0 | 0.0 | 0.0 |
| detection_loss | 12 | 1.0 | 0.0 | 0.0 | 0.0 |
| sensor_dropout | 12 | 1.0 | 0.0 | 0.0 | 0.0 |
| distance_noise | 12 | 0.75 | 0.083 | 0.0 | 0.167 |
| ttc_error | 2 | 0.5 | 0.0 | 0.0 | 0.5 |