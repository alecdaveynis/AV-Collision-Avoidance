# Planner comparison (dev sets)

Improved-planner fingerprint `60cec23e5f2d5bf9`; failure criteria and simulator identical for every row. Rates show `% (k/n) [Wilson 95% CI]`. Scenario-with-noise-seed rows are not independent samples: the n for `known_failure_*` and `regression_suite` counts (scenario, seed) pairs.

## known_failure_minimized (dev, n=20)
the discovered failure after minimization (no faults) x 20 noise seeds

| metric | baseline | improved | baseline+EMERGENCY_DECEL=4 | reference_rss |
|---|---|---|---|---|
| avoidable collisions | 100.0% (20/20) [83.9, 100.0] | 0.0% (0/20) [0.0, 16.1] | 0.0% (0/20) [0.0, 16.1] | 0.0% (0/20) [0.0, 16.1] |
| hard collisions (reference also fails) | 0.0% (0/20) [0.0, 16.1] | 0.0% (0/20) [0.0, 16.1] | 0.0% (0/20) [0.0, 16.1] | 0.0% (0/20) [0.0, 16.1] |
| unavoidable collisions | 0.0% (0/20) [0.0, 16.1] | 0.0% (0/20) [0.0, 16.1] | 0.0% (0/20) [0.0, 16.1] | 0.0% (0/20) [0.0, 16.1] |
| unnecessary braking | 0.0% (0/20) [0.0, 16.1] | 0.0% (0/20) [0.0, 16.1] | 0.0% (0/20) [0.0, 16.1] | 0.0% (0/20) [0.0, 16.1] |
| margin violation (<1 m) | 100.0% (20/20) [83.9, 100.0] | 0.0% (0/20) [0.0, 16.1] | 0.0% (0/20) [0.0, 16.1] | 0.0% (0/20) [0.0, 16.1] |
| any failure | 100.0% (20/20) [83.9, 100.0] | 0.0% (0/20) [0.0, 16.1] | 0.0% (0/20) [0.0, 16.1] | 0.0% (0/20) [0.0, 16.1] |
| min gap median / p5 (m) | 0.00 / 0.00 | 7.74 / 7.41 | 4.63 / 2.54 | 2.13 / 2.09 |
| impact speed given collision, mean / max (m/s) | 6.58 / 7.30 (n=20) | 0.00 / 0.00 (n=0) | 0.00 / 0.00 (n=0) | 0.00 / 0.00 (n=0) |
| brake reaction time mean / p95 (s) | 0.00 / 0.00 (n=20) | nan / nan (n=0) | 0.00 / 0.00 (n=20) | 0.04 / 0.10 (n=19) |
| peak commanded decel mean / p95 (m/s^2) | 8.00 / 8.00 | 3.00 / 3.00 | 8.00 / 8.00 | 4.00 / 4.00 |
| state transitions per minute | 10.5 | 9.0 | 15.1 | 149.8 |

Paired difference improved minus baseline (same scenarios):

| outcome | rate baseline | rate improved | difference [bootstrap 95% CI] | only baseline / only improved | McNemar p |
|---|---|---|---|---|---|
| avoidable collision | 100.0% | 0.0% | -100.0 pp [-100.0, -100.0] | 20 / 0 | 1.91e-06 |
| any collision | 100.0% | 0.0% | -100.0 pp [-100.0, -100.0] | 20 / 0 | 1.91e-06 |
| unnecessary braking | 0.0% | 0.0% | +0.0 pp [+0.0, +0.0] | 0 / 0 | 1 |
| margin violation (<1 m) | 100.0% | 0.0% | -100.0 pp [-100.0, -100.0] | 20 / 0 | 1.91e-06 |
| any failure | 100.0% | 0.0% | -100.0 pp [-100.0, -100.0] | 20 / 0 | 1.91e-06 |

## known_failure_original (dev, n=20)
the discovered failure as found by the search (latency 0.3 s, distance noise, 1.4 s detection loss) x 20 noise seeds

| metric | baseline | improved | baseline+EMERGENCY_DECEL=4 | reference_rss |
|---|---|---|---|---|
| avoidable collisions | 100.0% (20/20) [83.9, 100.0] | 100.0% (20/20) [83.9, 100.0] | 100.0% (20/20) [83.9, 100.0] | 100.0% (20/20) [83.9, 100.0] |
| hard collisions (reference also fails) | 0.0% (0/20) [0.0, 16.1] | 0.0% (0/20) [0.0, 16.1] | 0.0% (0/20) [0.0, 16.1] | 0.0% (0/20) [0.0, 16.1] |
| unavoidable collisions | 0.0% (0/20) [0.0, 16.1] | 0.0% (0/20) [0.0, 16.1] | 0.0% (0/20) [0.0, 16.1] | 0.0% (0/20) [0.0, 16.1] |
| unnecessary braking | 0.0% (0/20) [0.0, 16.1] | 0.0% (0/20) [0.0, 16.1] | 0.0% (0/20) [0.0, 16.1] | 0.0% (0/20) [0.0, 16.1] |
| margin violation (<1 m) | 100.0% (20/20) [83.9, 100.0] | 100.0% (20/20) [83.9, 100.0] | 100.0% (20/20) [83.9, 100.0] | 100.0% (20/20) [83.9, 100.0] |
| any failure | 100.0% (20/20) [83.9, 100.0] | 100.0% (20/20) [83.9, 100.0] | 100.0% (20/20) [83.9, 100.0] | 100.0% (20/20) [83.9, 100.0] |
| min gap median / p5 (m) | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| impact speed given collision, mean / max (m/s) | 13.86 / 14.07 (n=20) | 12.16 / 12.69 (n=20) | 10.91 / 12.89 (n=20) | 11.78 / 12.71 (n=20) |
| brake reaction time mean / p95 (s) | 0.21 / 0.21 (n=20) | 0.00 / 0.00 (n=20) | 0.19 / 0.21 (n=20) | 0.10 / 0.10 (n=20) |
| peak commanded decel mean / p95 (m/s^2) | 2.96 / 8.00 | 5.88 / 7.58 | 8.00 / 8.00 | 8.00 / 8.00 |
| state transitions per minute | 47.5 | 60.3 | 71.7 | 74.8 |

Paired difference improved minus baseline (same scenarios):

| outcome | rate baseline | rate improved | difference [bootstrap 95% CI] | only baseline / only improved | McNemar p |
|---|---|---|---|---|---|
| avoidable collision | 100.0% | 100.0% | +0.0 pp [+0.0, +0.0] | 0 / 0 | 1 |
| any collision | 100.0% | 100.0% | +0.0 pp [+0.0, +0.0] | 0 / 0 | 1 |
| unnecessary braking | 0.0% | 0.0% | +0.0 pp [+0.0, +0.0] | 0 / 0 | 1 |
| margin violation (<1 m) | 100.0% | 100.0% | +0.0 pp [+0.0, +0.0] | 0 / 0 | 1 |
| any failure | 100.0% | 100.0% | +0.0 pp [+0.0, +0.0] | 0 / 0 | 1 |

## regression_suite (dev, n=140)
committed scenarios/regression/*.json x 20 noise seeds

| metric | baseline | improved | baseline+EMERGENCY_DECEL=4 | reference_rss |
|---|---|---|---|---|
| avoidable collisions | 30.0% (42/140) [23.0, 38.0] | 0.0% (0/140) [0.0, 2.7] | 5.7% (8/140) [2.9, 10.9] | 20.0% (28/140) [14.2, 27.4] |
| hard collisions (reference also fails) | 0.0% (0/140) [0.0, 2.7] | 0.0% (0/140) [0.0, 2.7] | 0.0% (0/140) [0.0, 2.7] | 0.0% (0/140) [0.0, 2.7] |
| unavoidable collisions | 0.0% (0/140) [0.0, 2.7] | 0.0% (0/140) [0.0, 2.7] | 0.0% (0/140) [0.0, 2.7] | 0.0% (0/140) [0.0, 2.7] |
| unnecessary braking | 0.7% (1/140) [0.1, 3.9] | 1.4% (2/140) [0.4, 5.1] | 2.1% (3/140) [0.7, 6.1] | 39.3% (55/140) [31.6, 47.6] |
| margin violation (<1 m) | 70.0% (98/140) [62.0, 77.0] | 0.0% (0/140) [0.0, 2.7] | 11.4% (16/140) [7.2, 17.8] | 45.0% (63/140) [37.0, 53.3] |
| any failure | 46.4% (65/140) [38.4, 54.7] | 1.4% (2/140) [0.4, 5.1] | 7.9% (11/140) [4.4, 13.5] | 47.9% (67/140) [39.8, 56.1] |
| min gap median / p5 (m) | 0.55 / 0.00 | 9.91 / 4.57 | 4.45 / 0.00 | 1.17 / 0.00 |
| impact speed given collision, mean / max (m/s) | 2.54 / 6.99 (n=42) | 0.00 / 0.00 (n=0) | 3.84 / 6.99 (n=8) | 3.01 / 6.20 (n=28) |
| brake reaction time mean / p95 (s) | 0.14 / 0.71 (n=118) | 0.00 / 0.00 (n=20) | 0.14 / 0.70 (n=115) | 0.15 / 0.70 (n=120) |
| peak commanded decel mean / p95 (m/s^2) | 7.56 / 8.00 | 3.64 / 8.00 | 7.64 / 8.00 | 4.89 / 8.00 |
| state transitions per minute | 15.9 | 18.5 | 17.1 | 91.2 |

Paired difference improved minus baseline (same scenarios):

| outcome | rate baseline | rate improved | difference [bootstrap 95% CI] | only baseline / only improved | McNemar p |
|---|---|---|---|---|---|
| avoidable collision | 30.0% | 0.0% | -30.0 pp [-37.9, -22.9] | 42 / 0 | 4.55e-13 |
| any collision | 30.0% | 0.0% | -30.0 pp [-37.9, -22.9] | 42 / 0 | 4.55e-13 |
| unnecessary braking | 0.7% | 1.4% | +0.7 pp [-1.4, +3.6] | 1 / 2 | 1 |
| margin violation (<1 m) | 70.0% | 0.0% | -70.0 pp [-77.2, -62.1] | 98 / 0 | 6.31e-30 |
| any failure | 46.4% | 1.4% | -45.0 pp [-53.6, -36.4] | 65 / 2 | 3.09e-17 |

## dev_random (dev, n=300)
300 random scenarios, faults on with p=0.5, ego_speed 6-16 m/s, generator seed 11

| metric | baseline | improved | baseline+EMERGENCY_DECEL=4 | reference_rss |
|---|---|---|---|---|
| avoidable collisions | 20.7% (62/300) [16.5, 25.6] | 3.7% (11/300) [2.1, 6.5] | 10.0% (30/300) [7.1, 13.9] | 22.3% (67/300) [18.0, 27.4] |
| hard collisions (reference also fails) | 0.0% (0/300) [0.0, 1.3] | 0.0% (0/300) [0.0, 1.3] | 0.0% (0/300) [0.0, 1.3] | 0.0% (0/300) [0.0, 1.3] |
| unavoidable collisions | 0.0% (0/300) [0.0, 1.3] | 0.0% (0/300) [0.0, 1.3] | 0.0% (0/300) [0.0, 1.3] | 0.0% (0/300) [0.0, 1.3] |
| unnecessary braking | 2.0% (6/300) [0.9, 4.3] | 3.0% (9/300) [1.6, 5.6] | 10.3% (31/300) [7.4, 14.3] | 18.3% (55/300) [14.4, 23.1] |
| margin violation (<1 m) | 25.7% (77/300) [21.1, 30.9] | 4.3% (13/300) [2.5, 7.3] | 13.0% (39/300) [9.7, 17.3] | 60.0% (180/300) [54.4, 65.4] |
| any failure | 23.3% (70/300) [18.9, 28.4] | 6.3% (19/300) [4.1, 9.7] | 20.7% (62/300) [16.5, 25.6] | 41.3% (124/300) [35.9, 47.0] |
| min gap median / p5 (m) | 3.69 / 0.00 | 8.83 / 1.23 | 6.19 / 0.00 | 0.71 / 0.00 |
| impact speed given collision, mean / max (m/s) | 7.08 / 15.01 (n=62) | 6.99 / 15.04 (n=11) | 6.94 / 13.89 (n=30) | 5.77 / 15.04 (n=67) |
| brake reaction time mean / p95 (s) | 0.33 / 1.40 (n=136) | 0.31 / 0.95 (n=30) | 0.33 / 1.40 (n=117) | 0.22 / 1.04 (n=193) |
| peak commanded decel mean / p95 (m/s^2) | 4.66 / 8.00 | 3.24 / 8.00 | 5.19 / 8.00 | 4.36 / 8.00 |
| state transitions per minute | 12.9 | 17.6 | 14.1 | 60.3 |

Paired difference improved minus baseline (same scenarios):

| outcome | rate baseline | rate improved | difference [bootstrap 95% CI] | only baseline / only improved | McNemar p |
|---|---|---|---|---|---|
| avoidable collision | 20.7% | 3.7% | -17.0 pp [-21.7, -12.7] | 53 / 2 | 8.55e-14 |
| any collision | 20.7% | 3.7% | -17.0 pp [-21.7, -12.7] | 53 / 2 | 8.55e-14 |
| unnecessary braking | 2.0% | 3.0% | +1.0 pp [-1.3, +3.3] | 5 / 8 | 0.581 |
| margin violation (<1 m) | 25.7% | 4.3% | -21.3 pp [-26.3, -17.0] | 65 / 1 | 1.82e-18 |
| any failure | 23.3% | 6.3% | -17.0 pp [-22.0, -12.3] | 58 / 7 | 4.27e-11 |

## dev_near_boundary (dev, n=100)
100 fault-free scenarios the RSS reference only just handles (min gap 0.5-3 m), ego 6-16 m/s, seed 12

| metric | baseline | improved | baseline+EMERGENCY_DECEL=4 | reference_rss |
|---|---|---|---|---|
| avoidable collisions | 5.0% (5/100) [2.1, 11.2] | 0.0% (0/100) [0.0, 3.7] | 0.0% (0/100) [0.0, 3.7] | 0.0% (0/100) [0.0, 3.7] |
| hard collisions (reference also fails) | 0.0% (0/100) [0.0, 3.7] | 0.0% (0/100) [0.0, 3.7] | 0.0% (0/100) [0.0, 3.7] | 0.0% (0/100) [0.0, 3.7] |
| unavoidable collisions | 0.0% (0/100) [0.0, 3.7] | 0.0% (0/100) [0.0, 3.7] | 0.0% (0/100) [0.0, 3.7] | 0.0% (0/100) [0.0, 3.7] |
| unnecessary braking | 0.0% (0/100) [0.0, 3.7] | 0.0% (0/100) [0.0, 3.7] | 3.0% (3/100) [1.0, 8.5] | 29.0% (29/100) [21.0, 38.5] |
| margin violation (<1 m) | 11.0% (11/100) [6.2, 18.6] | 0.0% (0/100) [0.0, 3.7] | 0.0% (0/100) [0.0, 3.7] | 82.0% (82/100) [73.3, 88.3] |
| any failure | 9.0% (9/100) [4.8, 16.2] | 0.0% (0/100) [0.0, 3.7] | 3.0% (3/100) [1.0, 8.5] | 29.0% (29/100) [21.0, 38.5] |
| min gap median / p5 (m) | 3.21 / 0.12 | 6.87 / 6.11 | 5.81 / 2.15 | 0.65 / 0.54 |
| impact speed given collision, mean / max (m/s) | 2.83 / 5.65 (n=5) | 0.00 / 0.00 (n=0) | 0.00 / 0.00 (n=0) | 0.00 / 0.00 (n=0) |
| brake reaction time mean / p95 (s) | 0.01 / 0.00 (n=57) | 0.00 / 0.00 (n=1) | 0.01 / 0.00 (n=56) | 0.06 / 0.20 (n=95) |
| peak commanded decel mean / p95 (m/s^2) | 5.59 / 8.00 | 2.87 / 4.11 | 5.70 / 8.00 | 4.12 / 4.00 |
| state transitions per minute | 14.7 | 21.1 | 17.2 | 142.6 |

Paired difference improved minus baseline (same scenarios):

| outcome | rate baseline | rate improved | difference [bootstrap 95% CI] | only baseline / only improved | McNemar p |
|---|---|---|---|---|---|
| avoidable collision | 5.0% | 0.0% | -5.0 pp [-10.0, -1.0] | 5 / 0 | 0.0625 |
| any collision | 5.0% | 0.0% | -5.0 pp [-10.0, -1.0] | 5 / 0 | 0.0625 |
| unnecessary braking | 0.0% | 0.0% | +0.0 pp [+0.0, +0.0] | 0 / 0 | 1 |
| margin violation (<1 m) | 11.0% | 0.0% | -11.0 pp [-17.0, -6.0] | 11 / 0 | 0.000977 |
| any failure | 9.0% | 0.0% | -9.0 pp [-15.0, -4.0] | 9 / 0 | 0.00391 |

## dev_faults (dev, n=150)
150 scenarios with dropout 0.2-0.5 and latency 0-0.4 s, ego 6-16 m/s, seed 13

| metric | baseline | improved | baseline+EMERGENCY_DECEL=4 | reference_rss |
|---|---|---|---|---|
| avoidable collisions | 10.7% (16/150) [6.7, 16.6] | 2.0% (3/150) [0.7, 5.7] | 4.7% (7/150) [2.3, 9.3] | 8.0% (12/150) [4.6, 13.5] |
| hard collisions (reference also fails) | 0.0% (0/150) [0.0, 2.5] | 0.0% (0/150) [0.0, 2.5] | 0.0% (0/150) [0.0, 2.5] | 0.0% (0/150) [0.0, 2.5] |
| unavoidable collisions | 0.0% (0/150) [0.0, 2.5] | 0.0% (0/150) [0.0, 2.5] | 0.0% (0/150) [0.0, 2.5] | 0.0% (0/150) [0.0, 2.5] |
| unnecessary braking | 0.0% (0/150) [0.0, 2.5] | 0.0% (0/150) [0.0, 2.5] | 2.7% (4/150) [1.0, 6.7] | 22.7% (34/150) [16.7, 30.0] |
| margin violation (<1 m) | 15.3% (23/150) [10.4, 22.0] | 2.0% (3/150) [0.7, 5.7] | 8.7% (13/150) [5.1, 14.3] | 54.0% (81/150) [46.0, 61.8] |
| any failure | 13.3% (20/150) [8.8, 19.7] | 2.0% (3/150) [0.7, 5.7] | 8.7% (13/150) [5.1, 14.3] | 38.0% (57/150) [30.6, 46.0] |
| min gap median / p5 (m) | 6.30 / 0.00 | 9.03 / 4.52 | 6.38 / 0.05 | 0.64 / 0.00 |
| impact speed given collision, mean / max (m/s) | 6.24 / 13.13 (n=16) | 4.73 / 4.83 (n=3) | 6.11 / 12.13 (n=7) | 3.83 / 12.38 (n=12) |
| brake reaction time mean / p95 (s) | 0.12 / 0.53 (n=55) | 0.14 / 0.62 (n=7) | 0.11 / 0.46 (n=49) | 0.12 / 0.54 (n=93) |
| peak commanded decel mean / p95 (m/s^2) | 3.76 / 8.00 | 2.55 / 4.71 | 4.13 / 8.00 | 3.60 / 8.00 |
| state transitions per minute | 9.8 | 14.0 | 10.8 | 60.4 |

Paired difference improved minus baseline (same scenarios):

| outcome | rate baseline | rate improved | difference [bootstrap 95% CI] | only baseline / only improved | McNemar p |
|---|---|---|---|---|---|
| avoidable collision | 10.7% | 2.0% | -8.7 pp [-14.0, -4.0] | 14 / 1 | 0.000977 |
| any collision | 10.7% | 2.0% | -8.7 pp [-14.0, -4.0] | 14 / 1 | 0.000977 |
| unnecessary braking | 0.0% | 0.0% | +0.0 pp [+0.0, +0.0] | 0 / 0 | 1 |
| margin violation (<1 m) | 15.3% | 2.0% | -13.3 pp [-19.3, -8.0] | 21 / 1 | 1.1e-05 |
| any failure | 13.3% | 2.0% | -11.3 pp [-17.3, -6.0] | 18 / 1 | 7.63e-05 |
