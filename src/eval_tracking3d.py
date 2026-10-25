"""Evaluate the LiDAR 3D tracker: Kalman TTC vs hindsight labels, and motion prediction.

Usage:  python src/eval_tracking3d.py [--drive PATH ...] [--max-frames N] [--plots-dir docs]

Needs only LiDAR + OXTS (no camera, no YOLO), so it runs in seconds per drive.

TTC
  label     : hindsight TTC of a track = bumper range / closing speed from a linear fit of the
              track's range over the NEXT ``TTC_LABEL_WINDOW`` frames (kinematic TTC; KITTI has
              no collisions), same definition as ttc_data.py.
  methods   : Kalman state (causal, uses object velocity in the world frame)
              range-rate over the PAST window (what the planner used before)
              static-world assumption: range / ego speed (a parked object)
              constant 10 s
  Only in-path, confirmed tracks are scored. Also reports how often the label falls inside the
  Kalman 90 % TTC band.

Prediction
  Moving tracks (> 1.5 m/s); error = |prediction - filtered position| at +1, +2, +3 s for the
  static / constant-velocity / constant-turn-rate models.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import fusion_utils as fs
import predict
import tracker3d as t3
from calibration import load_calibration
from datasets import Drive, discover_drives
from ttc_data import ttc_from_range_history

HORIZONS = (1.0, 2.0, 3.0)
MOVING = 1.5


def track_drive(drive, calib, max_frames=None):
    """Run cluster + Kalman tracking over a drive. Returns (records by track id, per-frame stats)."""
    tracker = t3.Tracker3D()
    recs = {}
    n = min(len(drive), max_frames) if max_frames else len(drive)
    dts, times = drive.dts(), drive.times
    stage_ms = []
    for i in range(n):
        scan, pose = drive.lidar(i), drive.ego_pose_world(i, calib)
        if pose is None:
            continue
        t0 = time.perf_counter()
        clusters = t3.cluster_obstacles(fs.obstacle_points(scan)) if scan is not None else []
        t1 = time.perf_counter()
        speed = drive.ego_speed(i) or 0.0
        yaw = drive.yaw_rate(i) or 0.0
        curv = float(np.clip(yaw / speed, -config.CURVE_MAX, config.CURVE_MAX)) if speed > config.CURVE_MIN_SPEED else 0.0
        tracks = tracker.update(clusters, pose, dts[i], drive.ego_velocity_world(i))
        stage_ms.append(((t1 - t0) * 1e3, (time.perf_counter() - t1) * 1e3))
        for tr in tracks:
            ttc, lo, hi = tr.ttc()
            recs.setdefault(tr.id, []).append({
                'frame': i, 't': float(times[i]), 'range': tr.bumper_range, 'lat': tr.lateral,
                'closing': tr.closing_speed, 'sigma': tr.closing_std(), 'ttc': ttc, 'lo': lo, 'hi': hi,
                'in_path': t3.in_ego_path(tr, curv), 'ego': speed, 'speed': tr.speed_world,
                'state': tr.state.copy(), 'hr': tr.heading_rate,
            })
    return recs, np.array(stage_ms)


def _clip(v):
    return float(np.clip(v, config.TTC_MIN, config.TTC_MAX)) if math.isfinite(v) else float(config.TTC_MAX)


def ttc_samples(recs, drive_idx=0):
    """Rows (label, kalman, lo, hi, rangerate, static, range, closing, sigma, drive) over in-path samples."""
    W = config.TTC_LABEL_WINDOW
    rows = []
    for tid, rs in recs.items():
        # contiguous runs only
        runs, cur = [], [rs[0]]
        for r in rs[1:]:
            if r['frame'] - cur[-1]['frame'] == 1:
                cur.append(r)
            else:
                runs.append(cur)
                cur = [r]
        runs.append(cur)
        for run in runs:
            tt = np.array([r['t'] for r in run])
            rg = np.array([r['range'] for r in run])
            for j in range(W, len(run) - W):
                r = run[j]
                if not r['in_path'] or rg[j] > 40:
                    continue
                label = ttc_from_range_history(tt[j:j + W + 1], rg[j:j + W + 1], at_end=False)
                rr = ttc_from_range_history(tt[j - W:j + 1], rg[j - W:j + 1], at_end=True)
                static = _clip(r['range'] / r['ego']) if r['ego'] > config.MIN_CLOSING_SPEED else config.TTC_MAX
                rows.append((label, _clip(r['ttc']), _clip(r['lo']), _clip(r['hi']), rr, static, r['range'],
                             r['closing'], r['sigma'], drive_idx))
    return np.array(rows).reshape(-1, 10).T


def _prf(pred, label, thr=3.0):
    p, l = pred < thr, label < thr
    tp = int((p & l).sum()); fp = int((p & ~l).sum()); fn = int((~p & l).sum())
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return pr, rc, (2 * pr * rc / max(pr + rc, 1e-9)), fp


def _band(s, k):
    label, kal, lo, hi, rr, static, rng, c, sig, drive = s
    with np.errstate(divide='ignore'):
        lo_ = np.where(c + 1.645 * k * sig > config.MIN_CLOSING_SPEED, rng / (c + 1.645 * k * sig), config.TTC_MAX)
        hi_ = np.where(c - 1.645 * k * sig > config.MIN_CLOSING_SPEED, rng / (c - 1.645 * k * sig), config.TTC_MAX)
    return np.clip(lo_, config.TTC_MIN, config.TTC_MAX), np.clip(hi_, config.TTC_MIN, config.TTC_MAX)


def _coverage(s, k, mask):
    lo, hi = _band(s, k)
    label = s[0]
    return float(np.mean((label[mask] >= lo[mask] - 1e-6) & (label[mask] <= hi[mask] + 1e-6)))


def _fit_scale(s, mask, target=0.9):
    for k in np.arange(1.0, 20.01, 0.25):
        if _coverage(s, k, mask) >= target:
            return float(k)
    return 20.0


def report_ttc(s):
    label, kal, lo, hi, rr, static, rng, closing_v, sigma, drive = s
    lines = [f"in-path TTC samples: {label.shape[0]}  (hindsight TTC < 3 s: {int((label < 3).sum())})"]
    lines.append(f"{'method':<28}{'log-MAE':>8}{'MAE<6s':>8}{'prec':>7}{'recall':>7}{'F1':>6}{'FP':>5}")
    out = {}
    for name, pred in (('Kalman 3D (causal)', kal), ('range-rate, past window', rr),
                       ('static-world: range/ego', static), ('constant 10 s', np.full_like(label, config.TTC_MAX))):
        lm = float(np.mean(np.abs(np.log(pred) - np.log(label))))
        sel = label < 6
        mae = float(np.mean(np.abs(pred[sel] - label[sel]))) if sel.any() else float('nan')
        pr, rc, f1, fp = _prf(pred, label)
        lines.append(f"{name:<28}{lm:8.2f}{mae:8.2f}{pr:7.2f}{rc:7.2f}{f1:6.2f}{fp:5d}")
        out[name] = dict(log_mae=lm, mae_lt6=mae, precision=pr, recall=rc, f1=f1, fp=fp)
    closing = label < config.TTC_MAX - 0.05
    cov = _coverage(s, 1.0, closing)
    lines.append(f"Kalman 90% TTC band (raw filter covariance) contains the hindsight TTC for "
                 f"{100 * cov:.0f}% of closing samples (n={int(closing.sum())})  <- under-covers")
    out['band_coverage_raw'] = cov
    # calibrate a scale k on the filter sigma, leave-one-drive-out
    drives = np.unique(drive[closing])
    held = []
    for dv in drives:
        train = closing & (drive != dv)
        test = closing & (drive == dv)
        if train.sum() < 30 or test.sum() < 10:
            continue
        k = _fit_scale(s, train)
        held.append((int(dv), k, _coverage(s, k, test), int(test.sum())))
    k_all = _fit_scale(s, closing)
    if held:
        w = np.array([h[3] for h in held], float)
        mean_cov = float(np.sum([h[2] * h[3] for h in held]) / w.sum())
        lines.append(f"calibrated band: sigma scale k = {k_all:.1f} (pooled); leave-one-drive-out coverage "
                     f"{100 * mean_cov:.0f}% (per drive: " + ', '.join(f"{100 * h[2]:.0f}%" for h in held) + ")")
        width = np.median(np.log(_band(s, k_all)[1] / _band(s, k_all)[0])[closing])
        lines.append(f"median band width factor hi/lo = {math.exp(width):.1f}x")
        out.update(band_scale=k_all, band_coverage_loo=mean_cov, band_width_factor=math.exp(float(width)))
    return lines, out


def report_prediction(all_recs):
    errs = {m: {h: [] for h in HORIZONS} for m in predict.MODELS}
    n_tracks = 0
    for recs in all_recs:
        for rs in recs.values():
            run = [r for r in rs]
            hist = [(r['t'], *r['state'], r['hr']) for r in run if r['speed'] > MOVING]
            if len(hist) < 25:
                continue
            n_tracks += 1
            for m in predict.MODELS:
                e = predict.displacement_errors(hist, HORIZONS, m)
                for h in HORIZONS:
                    errs[m][h] += e[h]
    lines = [f"moving tracks (> {MOVING} m/s, >= 25 frames): {n_tracks}",
             f"{'model':<8}" + ''.join(f"{'ADE@' + str(int(h)) + 's':>10}" for h in HORIZONS)]
    out = {}
    for m in predict.MODELS:
        vals = [float(np.mean(errs[m][h])) if errs[m][h] else float('nan') for h in HORIZONS]
        lines.append(f"{m:<8}" + ''.join(f"{v:10.2f}" for v in vals))
        out[m] = dict(zip([f"{int(h)}s" for h in HORIZONS], vals))
    out['n'] = {f"{int(h)}s": len(errs['cv'][h]) for h in HORIZONS}
    return lines, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--drive', type=Path, action='append')
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--plots-dir', type=Path, default=config.ROOT / 'docs')
    ap.add_argument('--json', type=Path, default=config.ROOT / 'docs' / 'tracking3d_eval.json')
    args = ap.parse_args()
    drives = [Drive(p) for p in args.drive] if args.drive else discover_drives()
    calib = load_calibration()

    all_recs, stage = [], []
    for d in drives:
        recs, ms = track_drive(d, calib, args.max_frames)
        all_recs.append(recs)
        stage.append(ms)
        n_conf = len(recs)
        life = np.mean([len(v) for v in recs.values()]) if recs else 0
        print(f"[{d.name}] {len(d)} frames, {n_conf} confirmed tracks, mean life {life:.1f} frames")
    ms = np.concatenate(stage)
    s = np.concatenate([ttc_samples(r, i) for i, r in enumerate(all_recs) if r], axis=1) if any(all_recs) else np.zeros((10, 0))

    print("\n== TTC ==")
    result = {'drives': [d.name for d in drives]}
    if s.shape[1]:
        lines, result['ttc'] = report_ttc(s)
        print('\n'.join(lines))
    print("\n== Motion prediction ==")
    lines, result['prediction'] = report_prediction(all_recs)
    print('\n'.join(lines))
    print(f"\nlatency per frame: cluster {np.median(ms[:, 0]):.1f} ms, track {np.median(ms[:, 1]):.2f} ms (median)")
    result['latency_ms_median'] = {'cluster': float(np.median(ms[:, 0])), 'track': float(np.median(ms[:, 1]))}

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=1, default=float))
    if s.shape[1]:
        plot(s, args.plots_dir / 'tracking3d_eval.png')


def plot(s, path):
    label, kal, lo, hi, rr, static, rng = s[:7]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    closing = label < config.TTC_MAX - 0.05
    ax[0].scatter(label[closing], kal[closing], s=8, alpha=0.5, label='Kalman 3D')
    ax[0].scatter(label[closing], rr[closing], s=8, alpha=0.3, label='past range-rate')
    ax[0].plot([0, 10], [0, 10], 'k--', lw=1)
    ax[0].set_xlabel('hindsight TTC (s)'); ax[0].set_ylabel('estimated TTC (s)')
    ax[0].set_title('closing objects only'); ax[0].legend()
    order = np.argsort(label[closing])
    ax[1].fill_between(np.arange(order.size), lo[closing][order], hi[closing][order], alpha=0.3, label='Kalman 90% band')
    ax[1].plot(label[closing][order], 'k.', ms=3, label='hindsight TTC')
    ax[1].set_xlabel('sample (sorted by TTC)'); ax[1].set_ylabel('TTC (s)'); ax[1].legend()
    ax[1].set_title('uncertainty band vs label')
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=130)
    print(f"figure: {path}")


if __name__ == '__main__':
    main()
