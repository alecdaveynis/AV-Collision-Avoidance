# Regression detector evaluation (heldout)

Label set: 600 random scenarios (seed 4242), paired exact McNemar, p<0.01 and >=1 %. Suite: 8 failing + 4 near-boundary passing scenarios (+6 robust safe scenarios in the extended suite).

## Suite: suite

- **old** (measured label): 7/7 harmful detected; 6/17 non-harmful flagged (neutral 3/11, improvements 3/6); identity flagged: False
- **new** (measured label): 6/7 harmful detected; 2/17 non-harmful flagged (neutral 2/11, improvements 0/6); identity flagged: False
- old false negatives: none; false positives: ['STOP_BASE_DIST=0.8', 'EMERGENCY_DECEL=14', 'SLOW_BASE_DIST=3', 'RELEASE_FRAMES=7', 'STOP_BASE_DIST=3.5', 'EMERGENCY_DECEL=6']
- new false negatives: ['SLOW_HEADWAY=0.4']; false positives: ['STOP_BASE_DIST=0.8', 'EMERGENCY_DECEL=14']

## Suite: suite_plus_safe

- **old** (measured label): 7/7 harmful detected; 6/17 non-harmful flagged (neutral 3/11, improvements 3/6); identity flagged: False
- **new** (measured label): 7/7 harmful detected; 3/17 non-harmful flagged (neutral 3/11, improvements 0/6); identity flagged: False
- old false negatives: none; false positives: ['STOP_BASE_DIST=0.8', 'EMERGENCY_DECEL=14', 'SLOW_BASE_DIST=3', 'RELEASE_FRAMES=7', 'STOP_BASE_DIST=3.5', 'EMERGENCY_DECEL=6']
- new false negatives: none; false positives: ['STOP_BASE_DIST=0.8', 'EMERGENCY_DECEL=14', 'SLOW_BASE_DIST=3']

## Noise-only false alarms (unchanged planner, different seeds)

- suite: new detector flagged 0/40 (0.0); old checks (one seed) flagged 40/40 (1.0); tiers {'ok': 40}
- suite_plus_safe: new detector flagged 0/40 (0.0); old checks (one seed) flagged 40/40 (1.0); tiers {'ok': 40}

## Per mutation

| mutation | intent | measured label (worse/better/net, p_harm) | old | new (suite) | trigger | new (+safe) | trigger |
|---|---|---|---|---|---|---|---|
| identity | none | neutral (0/0/+0, 1.0) | - | - | - | - | - |
| TTC_SLOW=1.2 | harmful | harmful (57/7/+50, 0.0) | FLAG | FLAG | contract+pooled+statistical+suite-level | FLAG | contract+pooled+statistical+suite-level |
| TTC_STOP=0.6 | harmful | harmful (31/4/+27, 0.0) | FLAG | FLAG | contract+pooled+suite-level | FLAG | contract+pooled+suite-level |
| STOP_BASE_DIST=0.8 | harmful | neutral (5/5/+0, 0.62305) | FLAG | FLAG | pooled+suite-level | FLAG | pooled+suite-level |
| SLOW_HEADWAY=0.4 | harmful | harmful (23/6/+17, 0.00116) | FLAG | - | - | FLAG | pooled |
| ESCALATE_FRAMES=4 | harmful | harmful (62/13/+49, 0.0) | FLAG | FLAG | contract+pooled+statistical+suite-level | FLAG | contract+pooled+statistical+suite-level |
| EMERGENCY_DECEL=14 | harmful | neutral (7/14/-7, 0.96082) | FLAG | FLAG | pooled+suite-level | FLAG | pooled+suite-level |
| SLOW_BASE_DIST=3 | harmful | neutral (14/5/+9, 0.03178) | FLAG | - | - | FLAG | pooled |
| DIST_EMA_ALPHA=0.2 | harmful | harmful (18/3/+15, 0.00074) | FLAG | FLAG | pooled | FLAG | pooled |
| MIN_CLOSING_SPEED=2 | harmful | neutral (0/0/+0, 1.0) | - | - | - | - | - |
| DIST_EMA_ALPHA=0.55 | benign | neutral (3/10/-7, 0.98877) | - | - | - | - | - |
| RELEASE_FRAMES=7 | benign | improvement (5/21/-16, 0.99973) | FLAG | - | - | - | - |
| TRACK_MAX_AGE=7 | benign | neutral (0/0/+0, 1.0) | - | - | - | - | - |
| TTC_STOP=0.95 | benign | harmful (10/1/+9, 0.00586) | FLAG | FLAG | pooled | FLAG | pooled |
| STOP_BASE_DIST=2.2 | benign | neutral (0/2/-2, 1.0) | - | - | - | - | - |
| SLOW_BASE_DIST=5.8 | benign | neutral (3/0/+3, 0.125) | - | - | - | - | - |
| SLOW_HEADWAY=1.05 | benign | neutral (2/6/-4, 0.96484) | - | - | - | - | - |
| MIN_CLOSING_SPEED=0.6 | benign | neutral (0/0/+0, 1.0) | - | - | - | - | - |
| ESCALATE_FRAMES=3 | benign | harmful (41/12/+29, 4e-05) | FLAG | FLAG | contract+pooled+suite-level | FLAG | contract+pooled+suite-level |
| TTC_SLOW=3.8 | improvement | improvement (6/33/-27, 1.0) | - | - | - | - | - |
| STOP_BASE_DIST=3.5 | improvement | improvement (9/25/-16, 0.99853) | FLAG | - | - | - | - |
| SLOW_HEADWAY=1.4 | improvement | improvement (4/27/-23, 1.0) | - | - | - | - | - |
| TTC_STOP=1.4 | improvement | improvement (17/57/-40, 1.0) | - | - | - | - | - |
| EMERGENCY_DECEL=6 | improvement | improvement (15/37/-22, 0.9994) | FLAG | - | - | - | - |
