#!/usr/bin/env bash
# Run the same commands as .github/workflows/ci.yml (minus dependency installation) from a clean scratch dir.
# This is a local approximation of CI, not a replacement for it: it does not reproduce the GitHub runner's OS/Python.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
SCRATCH=$(mktemp -d)
export MPLBACKEND=Agg
echo "== unit tests";            $PY -m pytest -q tests
echo "== simulator smoke test";  $PY src/safety_sim.py --n 5 --plots-dir "$SCRATCH/sim"        # directory intentionally absent
echo "== red-team regression";   $PY -m autonomy_redteam regress --suite scenarios/regression/ \
                                      --baseline scenarios/baselines/v1.json --out "$SCRATCH/regress"
echo "== red-team regression v2";  $PY -m autonomy_redteam regress --suite scenarios/regression/ \
                                      --baseline scenarios/baselines/v2_improved.json --planner improved --out "$SCRATCH/regress-v2"
echo "== failure-to-fix demo";   $PY -m autonomy_redteam fix-demo --no-gif --validation 60 --out "$SCRATCH/fix"
echo "== red-team demo smoke";   $PY -m autonomy_redteam demo --budget 30 --no-gif --out "$SCRATCH/demo"
echo "all CI-equivalent steps passed (scratch: $SCRATCH)"
