# Planner comparison (heldout sets)

Improved-planner fingerprint `60cec23e5f2d5bf9`; failure criteria and simulator identical for every row. Rates show `% (k/n) [Wilson 95% CI]`. Scenario-with-noise-seed rows are not independent samples: the n for `known_failure_*` and `regression_suite` counts (scenario, seed) pairs.

## heldout_random (heldout, n=500)
500 random scenarios, full parameter box, generator seed 90001 (unseen seeds)

| metric | baseline | improved | baseline+EMERGENCY_DECEL=4 | reference_rss |
|---|---|---|---|---|
| avoidable collisions | 31.6% (158/500) [27.7, 35.8] | 7.8% (39/500) [5.8, 10.5] | 15.4% (77/500) [12.5, 18.8] | 28.2% (141/500) [24.4, 32.3] |
| hard collisions (reference also fails) | 0.0% (0/500) [0.0, 0.8] | 0.0% (0/500) [0.0, 0.8] | 0.0% (0/500) [0.0, 0.8] | 0.0% (0/500) [0.0, 0.8] |
| unavoidable collisions | 0.0% (0/500) [0.0, 0.8] | 0.0% (0/500) [0.0, 0.8] | 0.0% (0/500) [0.0, 0.8] | 0.0% (0/500) [0.0, 0.8] |
| unnecessary braking | 6.0% (30/500) [4.2, 8.4] | 4.2% (21/500) [2.8, 6.3] | 17.0% (85/500) [14.0, 20.5] | 31.8% (159/500) [27.9, 36.0] |
| margin violation (<1 m) | 37.0% (185/500) [32.9, 41.3] | 9.0% (45/500) [6.8, 11.8] | 21.0% (105/500) [17.7, 24.8] | 59.6% (298/500) [55.2, 63.8] |
| any failure | 38.6% (193/500) [34.4, 42.9] | 12.0% (60/500) [9.4, 15.1] | 32.2% (161/500) [28.2, 36.4] | 57.2% (286/500) [52.8, 61.5] |
| min gap median / p5 (m) | 2.85 / 0.00 | 8.08 / 0.00 | 5.35 / 0.00 | 0.67 / 0.00 |
| impact speed given collision, mean / max (m/s) | 7.25 / 19.87 (n=158) | 8.31 / 19.87 (n=39) | 6.53 / 17.32 (n=77) | 5.87 / 19.87 (n=141) |
| brake reaction time mean / p95 (s) | 0.31 / 1.49 (n=263) | 0.32 / 1.70 (n=71) | 0.32 / 1.40 (n=237) | 0.23 / 1.06 (n=328) |
| peak commanded decel mean / p95 (m/s^2) | 5.12 / 8.00 | 3.54 / 8.00 | 5.68 / 8.00 | 4.48 / 8.00 |
| state transitions per minute | 13.8 | 17.5 | 16.0 | 66.7 |

Paired difference improved minus baseline (same scenarios):

| outcome | rate baseline | rate improved | difference [bootstrap 95% CI] | only baseline / only improved | McNemar p |
|---|---|---|---|---|---|
| avoidable collision | 31.6% | 7.8% | -23.8 pp [-27.6, -20.2] | 123 / 4 | 1.25e-31 |
| any collision | 31.6% | 7.8% | -23.8 pp [-27.6, -20.2] | 123 / 4 | 1.25e-31 |
| unnecessary braking | 6.0% | 4.2% | -1.8 pp [-4.2, +0.6] | 24 / 15 | 0.2 |
| margin violation (<1 m) | 37.0% | 9.0% | -28.0 pp [-32.4, -24.0] | 148 / 8 | 1.68e-34 |
| any failure | 38.6% | 12.0% | -26.6 pp [-30.8, -22.6] | 142 / 9 | 6.59e-32 |

## heldout_speed_range (heldout, n=300)
300 random scenarios with ego_speed 16-20 m/s, a range no dev set contains, seed 90002

| metric | baseline | improved | baseline+EMERGENCY_DECEL=4 | reference_rss |
|---|---|---|---|---|
| avoidable collisions | 46.0% (138/300) [40.5, 51.6] | 12.7% (38/300) [9.4, 16.9] | 18.3% (55/300) [14.4, 23.1] | 33.7% (101/300) [28.6, 39.2] |
| hard collisions (reference also fails) | 0.0% (0/300) [0.0, 1.3] | 0.0% (0/300) [0.0, 1.3] | 0.0% (0/300) [0.0, 1.3] | 0.0% (0/300) [0.0, 1.3] |
| unavoidable collisions | 0.0% (0/300) [0.0, 1.3] | 0.0% (0/300) [0.0, 1.3] | 0.0% (0/300) [0.0, 1.3] | 0.0% (0/300) [0.0, 1.3] |
| unnecessary braking | 10.7% (32/300) [7.7, 14.7] | 1.7% (5/300) [0.7, 3.8] | 29.3% (88/300) [24.5, 34.7] | 48.7% (146/300) [43.1, 54.3] |
| margin violation (<1 m) | 50.7% (152/300) [45.0, 56.3] | 15.7% (47/300) [12.0, 20.2] | 21.7% (65/300) [17.4, 26.7] | 62.7% (188/300) [57.1, 68.0] |
| any failure | 57.0% (171/300) [51.3, 62.5] | 15.7% (47/300) [12.0, 20.2] | 47.3% (142/300) [41.8, 53.0] | 74.0% (222/300) [68.8, 78.6] |
| min gap median / p5 (m) | 0.94 / 0.00 | 6.82 / 0.00 | 5.33 / 0.00 | 0.60 / 0.00 |
| impact speed given collision, mean / max (m/s) | 8.93 / 19.53 (n=138) | 9.31 / 19.53 (n=38) | 7.75 / 19.53 (n=55) | 7.08 / 19.53 (n=101) |
| brake reaction time mean / p95 (s) | 0.36 / 1.40 (n=217) | 0.32 / 1.30 (n=105) | 0.29 / 1.10 (n=186) | 0.25 / 1.30 (n=208) |
| peak commanded decel mean / p95 (m/s^2) | 6.79 / 8.00 | 4.82 / 8.00 | 7.45 / 8.00 | 5.71 / 8.00 |
| state transitions per minute | 21.1 | 20.5 | 24.5 | 91.9 |

Paired difference improved minus baseline (same scenarios):

| outcome | rate baseline | rate improved | difference [bootstrap 95% CI] | only baseline / only improved | McNemar p |
|---|---|---|---|---|---|
| avoidable collision | 46.0% | 12.7% | -33.3 pp [-39.3, -27.3] | 109 / 9 | 5.86e-23 |
| any collision | 46.0% | 12.7% | -33.3 pp [-39.3, -27.3] | 109 / 9 | 5.86e-23 |
| unnecessary braking | 10.7% | 1.7% | -9.0 pp [-12.7, -5.7] | 29 / 2 | 4.63e-07 |
| margin violation (<1 m) | 50.7% | 15.7% | -35.0 pp [-41.0, -28.7] | 114 / 9 | 2.68e-24 |
| any failure | 57.0% | 15.7% | -41.3 pp [-47.3, -35.0] | 133 / 9 | 1.92e-29 |

## heldout_near_boundary (heldout, n=150)
150 fault-free scenarios the RSS reference only just handles, full box, seed 90003

| metric | baseline | improved | baseline+EMERGENCY_DECEL=4 | reference_rss |
|---|---|---|---|---|
| avoidable collisions | 17.3% (26/150) [12.1, 24.2] | 0.0% (0/150) [0.0, 2.5] | 1.3% (2/150) [0.4, 4.7] | 0.0% (0/150) [0.0, 2.5] |
| hard collisions (reference also fails) | 0.0% (0/150) [0.0, 2.5] | 0.0% (0/150) [0.0, 2.5] | 0.0% (0/150) [0.0, 2.5] | 0.0% (0/150) [0.0, 2.5] |
| unavoidable collisions | 0.0% (0/150) [0.0, 2.5] | 0.0% (0/150) [0.0, 2.5] | 0.0% (0/150) [0.0, 2.5] | 0.0% (0/150) [0.0, 2.5] |
| unnecessary braking | 0.0% (0/150) [0.0, 2.5] | 0.0% (0/150) [0.0, 2.5] | 3.3% (5/150) [1.4, 7.6] | 42.0% (63/150) [34.4, 50.0] |
| margin violation (<1 m) | 26.0% (39/150) [19.6, 33.6] | 1.3% (2/150) [0.4, 4.7] | 5.3% (8/150) [2.7, 10.2] | 82.0% (123/150) [75.1, 87.3] |
| any failure | 22.0% (33/150) [16.1, 29.3] | 1.3% (2/150) [0.4, 4.7] | 6.7% (10/150) [3.7, 11.8] | 44.0% (66/150) [36.3, 52.0] |
| min gap median / p5 (m) | 2.95 / 0.00 | 6.83 / 4.62 | 6.01 / 1.09 | 0.65 / 0.53 |
| impact speed given collision, mean / max (m/s) | 5.60 / 10.02 (n=26) | 0.00 / 0.00 (n=0) | 6.46 / 7.54 (n=2) | 0.00 / 0.00 (n=0) |
| brake reaction time mean / p95 (s) | 0.03 / 0.20 (n=91) | 0.01 / 0.01 (n=19) | 0.03 / 0.20 (n=87) | 0.05 / 0.20 (n=140) |
| peak commanded decel mean / p95 (m/s^2) | 5.80 / 8.00 | 3.34 / 5.88 | 5.91 / 8.00 | 4.32 / 8.00 |
| state transitions per minute | 16.5 | 22.6 | 19.4 | 150.6 |

Paired difference improved minus baseline (same scenarios):

| outcome | rate baseline | rate improved | difference [bootstrap 95% CI] | only baseline / only improved | McNemar p |
|---|---|---|---|---|---|
| avoidable collision | 17.3% | 0.0% | -17.3 pp [-23.3, -12.0] | 26 / 0 | 2.98e-08 |
| any collision | 17.3% | 0.0% | -17.3 pp [-23.3, -12.0] | 26 / 0 | 2.98e-08 |
| unnecessary braking | 0.0% | 0.0% | +0.0 pp [+0.0, +0.0] | 0 / 0 | 1 |
| margin violation (<1 m) | 26.0% | 1.3% | -24.7 pp [-32.0, -18.0] | 37 / 0 | 1.46e-11 |
| any failure | 22.0% | 1.3% | -20.7 pp [-27.4, -14.0] | 32 / 1 | 7.92e-09 |

## heldout_faults (heldout, n=200)
200 scenarios with dropout 0.2-0.5 and latency 0-0.4 s, full box, seed 90004

| metric | baseline | improved | baseline+EMERGENCY_DECEL=4 | reference_rss |
|---|---|---|---|---|
| avoidable collisions | 23.5% (47/200) [18.2, 29.8] | 2.5% (5/200) [1.1, 5.7] | 8.0% (16/200) [5.0, 12.6] | 12.0% (24/200) [8.2, 17.2] |
| hard collisions (reference also fails) | 0.0% (0/200) [0.0, 1.9] | 0.0% (0/200) [0.0, 1.9] | 0.0% (0/200) [0.0, 1.9] | 0.0% (0/200) [0.0, 1.9] |
| unavoidable collisions | 0.0% (0/200) [0.0, 1.9] | 0.0% (0/200) [0.0, 1.9] | 0.0% (0/200) [0.0, 1.9] | 0.0% (0/200) [0.0, 1.9] |
| unnecessary braking | 1.0% (2/200) [0.3, 3.6] | 0.0% (0/200) [0.0, 1.9] | 5.5% (11/200) [3.1, 9.6] | 30.0% (60/200) [24.1, 36.7] |
| margin violation (<1 m) | 31.0% (62/200) [25.0, 37.7] | 4.0% (8/200) [2.0, 7.7] | 12.0% (24/200) [8.2, 17.2] | 60.0% (120/200) [53.1, 66.5] |
| any failure | 27.0% (54/200) [21.3, 33.5] | 3.0% (6/200) [1.4, 6.4] | 14.0% (28/200) [9.9, 19.5] | 47.0% (94/200) [40.2, 53.9] |
| min gap median / p5 (m) | 4.06 / 0.00 | 6.76 / 2.30 | 6.26 / 0.00 | 0.59 / 0.00 |
| impact speed given collision, mean / max (m/s) | 6.99 / 12.75 (n=47) | 5.27 / 10.06 (n=5) | 5.70 / 9.88 (n=16) | 3.42 / 9.58 (n=24) |
| brake reaction time mean / p95 (s) | 0.15 / 0.50 (n=98) | 0.09 / 0.40 (n=39) | 0.17 / 0.50 (n=84) | 0.10 / 0.40 (n=138) |
| peak commanded decel mean / p95 (m/s^2) | 4.92 / 8.00 | 3.35 / 8.00 | 5.15 / 8.00 | 4.52 / 8.00 |
| state transitions per minute | 11.7 | 16.0 | 13.2 | 70.0 |

Paired difference improved minus baseline (same scenarios):

| outcome | rate baseline | rate improved | difference [bootstrap 95% CI] | only baseline / only improved | McNemar p |
|---|---|---|---|---|---|
| avoidable collision | 23.5% | 2.5% | -21.0 pp [-27.0, -15.5] | 42 / 0 | 4.55e-13 |
| any collision | 23.5% | 2.5% | -21.0 pp [-27.0, -15.5] | 42 / 0 | 4.55e-13 |
| unnecessary braking | 1.0% | 0.0% | -1.0 pp [-2.5, +0.0] | 2 / 0 | 0.5 |
| margin violation (<1 m) | 31.0% | 4.0% | -27.0 pp [-34.0, -20.5] | 57 / 3 | 6.25e-14 |
| any failure | 27.0% | 3.0% | -24.0 pp [-30.5, -18.0] | 49 / 1 | 9.06e-14 |
