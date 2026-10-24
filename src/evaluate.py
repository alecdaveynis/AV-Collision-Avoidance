"""Summarise pipeline runs: monocular vs LiDAR distance error, planner timeline.

Usage:  python src/evaluate.py [--output-dir output] [--plots-dir docs]

Reads telemetry.csv / tracks.csv written by main.py for every drive under the
output dir, prints a summary per drive (also saved as summary.txt in each drive
folder) and writes figures to --plots-dir:
  mono_vs_lidar.png        all drives pooled: scatter + relative-error histogram
  timeline_<drive>.png     per drive: ego speed, in-path distance vs. zones, TTC, planner state
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def col(rows, key, cast=float, empty=np.nan):
    return np.array([cast(r[key]) if r[key] != '' else empty for r in rows], dtype=float)


def distance_metrics(tracks):
    mono = col(tracks, 'mono_dist')
    lidar = col(tracks, 'lidar_dist')
    ok = np.isfinite(lidar) & (lidar > 0) & (mono > 0)
    mono, lidar = mono[ok], lidar[ok]
    err = mono - lidar
    rel = err / lidar
    lines = [f"Tracked boxes with LiDAR support: {ok.sum()} / {len(tracks)} "
             f"({100 * ok.mean():.0f}%)"]
    if ok.sum() == 0:
        return lines, (mono, lidar)
    lines += [
        f"Monocular vs LiDAR (LiDAR = reference):",
        f"  MAE            : {np.mean(np.abs(err)):.2f} m",
        f"  median |error| : {np.median(np.abs(err)):.2f} m",
        f"  bias (mono-lidar): {np.mean(err):+.2f} m",
        f"  median |rel err|: {100 * np.median(np.abs(rel)):.0f}%",
        f"  within 10%     : {100 * np.mean(np.abs(rel) < 0.10):.0f}%   "
        f"within 25%: {100 * np.mean(np.abs(rel) < 0.25):.0f}%",
    ]
    for lo, hi in ((0, 10), (10, 20), (20, 1e9)):
        sel = (lidar >= lo) & (lidar < hi)
        if sel.sum():
            name = f"{lo}-{hi} m" if hi < 1e8 else f">{lo} m"
            lines.append(f"  {name:>8}: n={sel.sum():4d}  MAE {np.mean(np.abs(err[sel])):.2f} m  "
                         f"median rel {100 * np.median(np.abs(rel[sel])):.0f}%")
    return lines, (mono, lidar)


def planner_metrics(tel):
    state = [r['state'] for r in tel]
    raw = [r['raw_state'] for r in tel]
    changes = lambda seq: sum(a != b for a, b in zip(seq, seq[1:]))
    lines = ["Planner:"]
    for s in ('DRIVE', 'SLOWING', 'STOPPED'):
        lines.append(f"  {s:8s}: {state.count(s):4d} frames ({100 * state.count(s) / len(state):.0f}%)")
    lines.append(f"  state changes: {changes(state)} (without hysteresis: {changes(raw)})")
    lane = [r['lane_status'] for r in tel]
    lines.append(f"Lane: detected {100 * lane.count('detected') / len(lane):.0f}%, "
                 f"coasting {100 * lane.count('coasting') / len(lane):.0f}%, "
                 f"rejected by path gate {100 * lane.count('rejected') / len(lane):.0f}%, "
                 f"none {100 * lane.count('none') / len(lane):.0f}% of frames (corridor used when no lane)")
    tracks_total = sum(int(r['n_tracks']) for r in tel)
    in_path = sum(int(r['n_in_path']) for r in tel)
    lines.append(f"Objects in path: {in_path} of {tracks_total} track-frames "
                 f"({100 * in_path / max(tracks_total, 1):.0f}%)")
    return lines


def plot_distance(mono, lidar, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    if len(mono):
        lim = max(mono.max(), lidar.max()) * 1.05
        ax[0].scatter(lidar, mono, s=8, alpha=0.4)
        ax[0].plot([0, lim], [0, lim], 'r--', lw=1, label='ideal')
        ax[0].set_xlim(0, lim); ax[0].set_ylim(0, lim)
        ax[0].legend()
        rel = 100 * (mono - lidar) / lidar
        ax[1].hist(np.clip(rel, -100, 200), bins=50, color='tab:orange')
        ax[1].axvline(0, color='k', lw=1)
    ax[0].set_xlabel('LiDAR distance (m)'); ax[0].set_ylabel('Monocular distance (m)')
    ax[0].set_title('Monocular vs LiDAR distance per tracked box')
    ax[1].set_xlabel('Relative error of monocular estimate (%)'); ax[1].set_ylabel('count')
    ax[1].set_title('Monocular relative error (clipped to [-100, 200]%)')
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_timeline(tel, path, title='Run timeline'):
    t = np.cumsum(col(tel, 'dt'))
    speed = col(tel, 'ego_speed') * 3.6
    dist = col(tel, 'min_dist')
    stop, slow = col(tel, 'stop_dist'), col(tel, 'slow_dist')
    ttc = col(tel, 'ttc')
    state = [r['state'] for r in tel]
    fig, ax = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)

    ax[0].plot(t, speed, color='tab:blue'); ax[0].set_ylabel('Ego speed (km/h)')
    ax[0].set_title(title)

    ax[1].plot(t, np.minimum(dist, 40), label='in-path distance', color='k')
    ax[1].plot(t, slow, '--', color='orange', label='slow threshold')
    ax[1].plot(t, stop, '--', color='red', label='stop threshold')
    ax[1].set_ylabel('Distance (m)'); ax[1].set_ylim(0, 40); ax[1].legend(loc='upper right', ncol=3)

    ttc_a = col(tel, 'ttc_analytic') if 'ttc_analytic' in tel[0] else ttc
    ttc_l = col(tel, 'ttc_learned') if 'ttc_learned' in tel[0] else np.full(len(tel), np.nan)
    ax[2].plot(t, np.where(np.isfinite(ttc_a), np.minimum(ttc_a, 10), np.nan), color='tab:purple', label='analytic')
    if np.isfinite(ttc_l).any():
        ax[2].plot(t, np.minimum(ttc_l, 10), color='tab:green', lw=1.2, label='learned')
    ax[2].legend(loc='upper right')
    ax[2].axhline(config.TTC_SLOW, ls='--', color='orange'); ax[2].axhline(config.TTC_STOP, ls='--', color='red')
    ax[2].set_ylabel('TTC (s, capped 10)'); ax[2].set_xlabel('Time (s)')

    colors = {'DRIVE': 'tab:green', 'SLOWING': 'orange', 'STOPPED': 'red'}
    for a in ax:
        for i, s in enumerate(state):
            a.axvspan(t[i] - 0.05, t[i] + 0.05, color=colors[s], alpha=0.12, lw=0)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=Path, default=config.OUTPUT_DIR)
    p.add_argument('--plots-dir', type=Path, default=config.ROOT / 'docs')
    args = p.parse_args()

    runs = sorted(d for d in args.output_dir.iterdir() if (d / 'telemetry.csv').exists())
    if not runs:
        raise SystemExit(f"No runs with telemetry.csv under {args.output_dir}. Run `python src/main.py` first.")
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    all_mono, all_lidar = [], []
    for run in runs:
        tel = read_csv(run / 'telemetry.csv')
        tracks = read_csv(run / 'tracks.csv')
        dist_lines, (mono, lidar) = distance_metrics(tracks)
        all_mono.append(mono); all_lidar.append(lidar)
        text = '\n'.join([f"=== {run.name} ===", f"Frames: {len(tel)}"] + dist_lines + planner_metrics(tel))
        print(text + '\n')
        (run / 'summary.txt').write_text(text + '\n')
        plot_timeline(tel, args.plots_dir / f'timeline_{run.name}.png', title=f'Run timeline - {run.name}')

    mono, lidar = np.concatenate(all_mono), np.concatenate(all_lidar)
    plot_distance(mono, lidar, args.plots_dir / 'mono_vs_lidar.png')
    print(f"Plots written to {args.plots_dir}")


if __name__ == '__main__':
    main()
