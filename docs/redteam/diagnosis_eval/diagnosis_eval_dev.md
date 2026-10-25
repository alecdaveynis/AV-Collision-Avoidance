# Diagnosis evaluation (dev, seed 11)

49 planted-fault cases, 20 no-fault controls; could not construct: {'distance_noise': 1, 'ttc_error': 10}. All scoring uses the planted label only after the fact.

## Legacy engine (Shapley share, status *supported*)

- top-1 Shapley accuracy 0.755; planted component labelled *supported* in 0.959; precision of *supported* labels 0.81; planner-logic false flags (planted cases) 0.0
- no-fault controls: planner logic *supported* in 0.7; a perception component falsely *supported* in 0.3

| planted fault | n | top-1 | planted supported |
|---|---|---|---|
| perception_latency | 12 | 0.25 | 0.917 |
| detection_loss | 12 | 1.0 | 1.0 |
| sensor_dropout | 12 | 0.917 | 1.0 |
| distance_noise | 11 | 1.0 | 1.0 |
| ttc_error | 2 | 0.0 | 0.5 |

## New engine (temporal evidence + controlled replacement, explicit joint / inconclusive)

- planted component is the unique primary root cause: **0.857**; correct but reported jointly with another component: 0.021; wrong: 0.0; inconclusive: 0.122
- false positives: any component other than the planted one flagged primary/contributing in 0.02 of planted cases (a perception component in 0.02; planner logic in 0.0)
- no-fault controls: planner logic the unique primary cause in 0.7; a perception component flagged in **0.05**; inconclusive 0.25

| planted fault | n | correct (unique) | correct, joint | wrong | inconclusive |
|---|---|---|---|---|---|
| perception_latency | 12 | 0.917 | 0.083 | 0.0 | 0.0 |
| detection_loss | 12 | 1.0 | 0.0 | 0.0 | 0.0 |
| sensor_dropout | 12 | 0.917 | 0.0 | 0.0 | 0.083 |
| distance_noise | 11 | 0.727 | 0.0 | 0.0 | 0.273 |
| ttc_error | 2 | 0.0 | 0.0 | 0.0 | 1.0 |