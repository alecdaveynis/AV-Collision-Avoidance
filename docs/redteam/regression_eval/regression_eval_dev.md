# Regression detector evaluation (dev)

Label set: 600 random scenarios (seed 888), paired exact McNemar, p<0.01 and >=1 %. Suite: 8 failing + 6 near-boundary passing scenarios (+6 robust safe scenarios in the extended suite).

## Suite: suite

- **old** (measured label): 4/4 harmful detected; 8/15 non-harmful flagged (neutral 5/11, improvements 3/4); identity flagged: False
  - legacy label (net >= 2 % of 300): 4/4 harmful detected; 8/15 non-harmful flagged (neutral 6/12, improvements 2/3); identity flagged: False
- **new** (measured label): 4/4 harmful detected; 4/15 non-harmful flagged (neutral 4/11, improvements 0/4); identity flagged: False
  - legacy label (net >= 2 % of 300): 4/4 harmful detected; 4/15 non-harmful flagged (neutral 4/12, improvements 0/3); identity flagged: False
- old false negatives: none; false positives: ['STOP_BASE_DIST=0.5', 'EMERGENCY_DECEL=20', 'SLOW_BASE_DIST=2', 'DIST_EMA_ALPHA=0.45', 'RELEASE_FRAMES=6', 'TTC_SLOW=4.0', 'STOP_BASE_DIST=4', 'SLOW_HEADWAY=1.5']
- new false negatives: none; false positives: ['STOP_BASE_DIST=0.5', 'EMERGENCY_DECEL=20', 'SLOW_BASE_DIST=2', 'DIST_EMA_ALPHA=0.45']

## Suite: suite_plus_safe

- **old** (measured label): 4/4 harmful detected; 8/15 non-harmful flagged (neutral 5/11, improvements 3/4); identity flagged: False
  - legacy label (net >= 2 % of 300): 4/4 harmful detected; 8/15 non-harmful flagged (neutral 6/12, improvements 2/3); identity flagged: False
- **new** (measured label): 4/4 harmful detected; 4/15 non-harmful flagged (neutral 4/11, improvements 0/4); identity flagged: False
  - legacy label (net >= 2 % of 300): 4/4 harmful detected; 4/15 non-harmful flagged (neutral 4/12, improvements 0/3); identity flagged: False
- old false negatives: none; false positives: ['STOP_BASE_DIST=0.5', 'EMERGENCY_DECEL=20', 'SLOW_BASE_DIST=2', 'DIST_EMA_ALPHA=0.45', 'RELEASE_FRAMES=6', 'TTC_SLOW=4.0', 'STOP_BASE_DIST=4', 'SLOW_HEADWAY=1.5']
- new false negatives: none; false positives: ['STOP_BASE_DIST=0.5', 'EMERGENCY_DECEL=20', 'SLOW_BASE_DIST=2', 'DIST_EMA_ALPHA=0.45']

## Noise-only false alarms (unchanged planner, different seeds)

- suite: new detector flagged 0/40 (0.0); old checks (one seed) flagged 40/40 (1.0); tiers {'ok': 40}
- suite_plus_safe: new detector flagged 1/40 (0.025); old checks (one seed) flagged 40/40 (1.0); tiers {'ok': 39, 'hard_violation': 1}

## Per mutation

| mutation | intent | measured label (worse/better/net, p_harm) | old | new (suite) | trigger | new (+safe) | trigger |
|---|---|---|---|---|---|---|---|
| identity | none | neutral (0/0/+0, 1.0) | - | - | - | - | - |
| TTC_SLOW=1.5 | harmful | harmful (58/8/+50, 0.0) | FLAG | FLAG | contract+pooled+suite-level | FLAG | contract+pooled+suite-level |
| TTC_STOP=0.5 | harmful | harmful (28/8/+20, 0.0006) | FLAG | FLAG | pooled+suite-level | FLAG | pooled+suite-level |
| STOP_BASE_DIST=0.5 | harmful | neutral (9/4/+5, 0.13342) | FLAG | FLAG | pooled+suite-level | FLAG | pooled+suite-level |
| SLOW_HEADWAY=0.3 | harmful | harmful (22/8/+14, 0.00806) | FLAG | FLAG | pooled | FLAG | pooled |
| ESCALATE_FRAMES=5 | harmful | harmful (85/15/+70, 0.0) | FLAG | FLAG | contract+pooled+statistical+suite-level | FLAG | contract+pooled+statistical+suite-level |
| EMERGENCY_DECEL=20 | harmful | neutral (17/15/+2, 0.43003) | FLAG | FLAG | pooled+suite-level | FLAG | pooled+suite-level |
| SLOW_BASE_DIST=2 | harmful | neutral (16/9/+7, 0.11476) | FLAG | FLAG | pooled | FLAG | pooled |
| MIN_CLOSING_SPEED=3 | harmful | neutral (0/0/+0, 1.0) | - | - | - | - | - |
| DIST_EMA_ALPHA=0.45 | benign | neutral (1/3/-2, 0.9375) | FLAG | FLAG | pooled | FLAG | pooled |
| RELEASE_FRAMES=6 | benign | improvement (1/10/-9, 0.99951) | FLAG | - | - | - | - |
| TRACK_MAX_AGE=6 | benign | neutral (0/0/+0, 1.0) | - | - | - | - | - |
| TTC_STOP=1.05 | benign | neutral (1/2/-1, 0.875) | - | - | - | - | - |
| STOP_BASE_DIST=2.1 | benign | neutral (1/1/+0, 0.75) | - | - | - | - | - |
| SLOW_BASE_DIST=6.2 | benign | neutral (0/2/-2, 1.0) | - | - | - | - | - |
| TTC_SLOW=4.0 | improvement | improvement (7/21/-14, 0.99814) | FLAG | - | - | - | - |
| STOP_BASE_DIST=4 | improvement | neutral (14/18/-4, 0.81146) | FLAG | - | - | - | - |
| SLOW_HEADWAY=1.5 | improvement | improvement (9/26/-17, 0.99906) | FLAG | - | - | - | - |
| TTC_STOP=1.5 | improvement | improvement (17/50/-33, 0.99999) | - | - | - | - | - |
