"""Compare planner decisions between runs made with different --ttc modes.

    python src/main.py --no-video --output-dir /tmp/run_analytic
    python src/main.py --no-video --output-dir /tmp/run_kalman --ttc kalman
    python src/compare_modes.py /tmp/run_analytic /tmp/run_kalman [/tmp/run_fused ...]

The first directory is the reference. For every drive: frames per state, frames whose state
differs from the reference, and raw state changes (flicker proxy).
"""
import csv
import sys
from collections import Counter
from pathlib import Path

STATES = ('DRIVE', 'SLOWING', 'STOPPED')


def read_states(path):
    with open(path) as f:
        return [r['state'] for r in csv.DictReader(f)]


def main(dirs):
    dirs = [Path(d) for d in dirs]
    ref = dirs[0]
    drives = sorted(p.name for p in ref.iterdir() if (p / 'telemetry.csv').exists())
    print(f"{'drive':<32}{'run':<16}{'DRIVE':>7}{'SLOW':>6}{'STOP':>6}{'differ':>8}{'changes':>9}")
    for d in drives:
        base = read_states(ref / d / 'telemetry.csv')
        for run in dirs:
            f = run / d / 'telemetry.csv'
            if not f.exists():
                continue
            st = read_states(f)
            c = Counter(st)
            differ = sum(a != b for a, b in zip(base, st))
            changes = sum(a != b for a, b in zip(st, st[1:]))
            print(f"{d:<32}{run.name:<16}{c['DRIVE']:7d}{c['SLOWING']:6d}{c['STOPPED']:6d}{differ:8d}{changes:9d}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1:])
