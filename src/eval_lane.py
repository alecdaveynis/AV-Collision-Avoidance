"""Score ego-lane / ego-path estimates against where the car actually drove.

    python src/eval_lane.py [--drive PATH ...] [--debug-frames 3]

Ground truth (evaluation only, it uses the FUTURE): the OXTS trajectory of the next
~30 m, projected into the vehicle frame at each frame; the "driven footprint" is
that path +/- GT_HALF_WIDTH (vehicle width + margin). Everything is compared on a
ground-plane grid (x forward 7..30 m, y lateral), so numbers are in metres, not
image pixels.

Compared estimators
  straight corridor      |y| < 1.2 m
  curved corridor        constant curvature from yaw rate / speed (causal)
  lane (legacy)          previous image-space detector, converted to the ground plane
  lane (metric)          calibrated metric-BEV detector (lane_utils), two lines required
  lane (metric+single)   same, but a lone line may be extended by a fixed lane width
  lane (metric, gated)   metric lane kept only if it agrees with the yaw-rate path
  path: legacy           legacy lane  U straight corridor   (original pipeline)
  path: metric ungated   metric lane  U curved corridor
  path: metric gated     gated lane   U curved corridor     (current pipeline)

Metrics (summed over frames): recall = share of the driven footprint covered
(missed = potential missed hazard), precision = share of the estimate that is
really driven footprint, IoU. For lane detectors also: availability and the
lateral distance between the lane centre and the driven path.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import fusion_utils as fs
import lane_legacy as ll
import lane_utils as lu
from calibration import load_calibration
from datasets import Drive, discover_drives

GT_HALF_WIDTH = 1.1          # m: vehicle half width + margin
X_MIN, X_MAX, X_STEP = 7.0, 30.0, 0.5
Y_MIN, Y_MAX, Y_STEP = -5.0, 5.0, 0.1
XS = np.arange(X_MIN, X_MAX + 1e-6, X_STEP)
YS = np.arange(Y_MIN, Y_MAX + 1e-6, Y_STEP)
GX, GY = np.meshgrid(XS, YS, indexing='ij')           # (nx, ny)

METHODS = ['straight corridor', 'curved corridor', 'lane (legacy)', 'lane (metric)',
           'lane (metric+single)', 'lane (metric, gated)', 'path: legacy',
           'path: metric ungated', 'path: metric gated']


# ----------------------------------------------------------------- footprints ---
def gt_footprint(path_xy):
    """Boolean grid of the driven footprint and the x range it covers."""
    px, py = path_xy[:, 0], path_xy[:, 1]
    x_hi = min(X_MAX, px.max())
    in_range = (GX >= X_MIN) & (GX <= x_hi)
    y_gt = np.interp(GX, px, py)
    return in_range & (np.abs(GY - y_gt) < GT_HALF_WIDTH), in_range, np.interp(XS, px, py)


def lane_region(yl_fn, yr_fn, x_lo, x_hi):
    """Boolean grid between two lateral functions of x, for x in [x_lo, x_hi]."""
    inx = (GX >= x_lo) & (GX <= x_hi)
    return inx & (GY < yl_fn(GX)) & (GY > yr_fn(GX))


def corridor_region(curvature):
    yc = fs.corridor_centerline(GX, curvature)
    return np.abs(GY - yc) < config.FALLBACK_CORRIDOR_HALF_WIDTH


def metric_lane_region(fit):
    if fit is None:
        return None
    return lane_region(fit.y_left, fit.y_right, max(X_MIN, fit.x_start), fit.x_end)


def legacy_lane_to_ground(fit, img, bev):
    """Legacy fit (polynomials in its warped-pixel space) -> (left, right) ground polylines."""
    h = img.shape[0]
    _, Minv_old = ll.get_perspective_matrices(img)
    ploty = np.linspace(0, h - 1, 60)
    lines = []
    for coeffs in fit:
        warped_pts = np.column_stack((np.polyval(coeffs, ploty), ploty)).astype(np.float32)
        img_pts = cv2.perspectiveTransform(warped_pts.reshape(-1, 1, 2), Minv_old).reshape(-1, 2)
        gx, gy = bev.image_to_ground(img_pts)
        order = np.argsort(gx)
        lines.append((gx[order], gy[order]))
    return lines


def legacy_lane_region(fit, img, bev):
    if fit is None:
        return None, None
    (lx, ly), (rx, ry) = legacy_lane_to_ground(fit, img, bev)
    x_lo, x_hi = max(lx.min(), rx.min(), X_MIN), min(lx.max(), rx.max())
    if x_hi - x_lo < 3.0:
        return None, None
    yl = lambda x: np.interp(x, lx, ly)
    yr = lambda x: np.interp(x, rx, ry)
    return lane_region(yl, yr, x_lo, x_hi), (yl, yr, x_lo, x_hi)


# ------------------------------------------------------------------- accumulate ---
class Acc:
    def __init__(self):
        self.tp = self.pred = self.gt = 0
        self.frames = self.available = 0
        self.center_err, self.widths = [], []
        # same sums restricted to frames where the estimator exists
        self.tp_av = self.pred_av = self.gt_av = 0

    def add(self, region, gt, available=True):
        self.frames += 1
        g = int(gt.sum())
        if region is None:
            self.gt += g
            return
        self.available += 1
        tp = int((region & gt).sum())
        self.tp += tp; self.pred += int(region.sum()); self.gt += g
        self.tp_av += tp; self.pred_av += int(region.sum()); self.gt_av += g

    def stats(self, available_only=False):
        tp, pred, gt = ((self.tp_av, self.pred_av, self.gt_av) if available_only
                        else (self.tp, self.pred, self.gt))
        return {'recall': tp / gt if gt else np.nan, 'precision': tp / pred if pred else np.nan,
                'iou': tp / (pred + gt - tp) if (pred + gt - tp) else np.nan,
                'avail': self.available / self.frames if self.frames else np.nan}


# ---------------------------------------------------------------------- driver ---
def run_drive(drive, calib, debug_frames, out_dir):
    first = cv2.imread(str(drive.frames[0]))
    bev = lu.make_bev(calib, first.shape)
    trackers = {
        'metric': lu.LaneTracker(bev),
        'single': lu.LaneTracker(bev),
        'legacy': ll.LaneTracker(),
    }
    acc = {m: Acc() for m in METHODS}
    status_counts = {'metric': {}, 'legacy': {}}
    debug_ids = set(np.linspace(10, len(drive) - 40, max(debug_frames, 1)).astype(int)) if debug_frames else set()

    def centre_stats(name, fit, path):
        if fit is None:
            return
        xs = XS[(XS >= max(X_MIN, fit.x_start)) & (XS <= min(fit.x_end, path[:, 0].max()))]
        if len(xs):
            acc[name].center_err.append(float(np.mean(np.abs(fit.center(xs) - np.interp(xs, path[:, 0], path[:, 1])))))
            acc[name].widths.append(float(np.mean(fit.width(xs))))

    n_used = 0
    for i, f in enumerate(drive.frames):
        path = drive.driven_path(i, calib, length=X_MAX + 2.0, min_length=X_MAX - 2.0)
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        mask = lu.lane_color_mask(img)

        # --- estimators (always run so tracker state stays continuous) -----------------
        warped = cv2.warpPerspective(mask, bev.M, bev.size)
        saved = config.LANE_M_SINGLE_LINE
        config.LANE_M_SINGLE_LINE = False
        fit_m, st_m = trackers['metric'].update(warped)
        config.LANE_M_SINGLE_LINE = True
        fit_s, st_s = trackers['single'].update(warped)
        config.LANE_M_SINGLE_LINE = saved

        M_old, _ = ll.get_perspective_matrices(img)
        warped_old = cv2.warpPerspective(mask, M_old, (w, h))
        fit_l, st_l = trackers['legacy'].update(warped_old)

        for key, st in (('metric', st_m), ('legacy', st_l)):
            status_counts[key][st] = status_counts[key].get(st, 0) + 1
        if path is None:
            continue
        n_used += 1

        speed, yaw = drive.ego_speed(i), drive.yaw_rate(i)
        kappa = 0.0
        if speed and speed > config.CURVE_MIN_SPEED and yaw is not None:
            kappa = float(np.clip(yaw / speed, -config.CURVE_MAX, config.CURVE_MAX))

        gt, in_range, _ = gt_footprint(path)
        straight = corridor_region(0.0) & in_range
        curved = corridor_region(kappa) & in_range
        fit_g = fit_m if (fit_m is not None and lu.consistent_with_path(fit_m, kappa)) else None
        lane_m = metric_lane_region(fit_m)
        lane_s = metric_lane_region(fit_s)
        lane_g = metric_lane_region(fit_g)
        lane_l, leg_fns = legacy_lane_region(fit_l, img, bev)
        lane_m, lane_s, lane_g, lane_l = [None if r is None else r & in_range
                                          for r in (lane_m, lane_s, lane_g, lane_l)]
        union = lambda base, lane: base if lane is None else base | lane

        acc['straight corridor'].add(straight, gt)
        acc['curved corridor'].add(curved, gt)
        acc['lane (legacy)'].add(lane_l, gt)
        acc['lane (metric)'].add(lane_m, gt)
        acc['lane (metric+single)'].add(lane_s, gt)
        acc['lane (metric, gated)'].add(lane_g, gt)
        acc['path: legacy'].add(union(straight, lane_l), gt)
        acc['path: metric ungated'].add(union(curved, lane_m), gt)
        acc['path: metric gated'].add(union(curved, lane_g), gt)

        centre_stats('lane (metric)', fit_m, path)
        centre_stats('lane (metric+single)', fit_s, path)
        centre_stats('lane (metric, gated)', fit_g, path)
        if leg_fns is not None:
            yl, yr, x_lo, x_hi = leg_fns
            xs = XS[(XS >= x_lo) & (XS <= min(x_hi, path[:, 0].max()))]
            if len(xs):
                c = 0.5 * (yl(xs) + yr(xs))
                acc['lane (legacy)'].center_err.append(float(np.mean(np.abs(c - np.interp(xs, path[:, 0], path[:, 1])))))
                acc['lane (legacy)'].widths.append(float(np.mean(yl(xs) - yr(xs))))

        if i in debug_ids:
            save_debug(out_dir / f'{drive.name}_{i:04d}.png', img, path, fit_g, 'gated lane' if fit_g else
                       f"rejected ({st_m})", bev, calib, kappa)

    return acc, status_counts, n_used


def save_debug(path, img, driven, fit, status, bev, calib, kappa):
    """GT driven path (red), metric lane edges (green), curved corridor (orange)."""
    out = img.copy()
    xs = np.linspace(X_MIN, driven[:, 0].max(), 60)
    ys = np.interp(xs, driven[:, 0], driven[:, 1])
    for sign, col in ((0.0, (0, 0, 255)),):
        pts = np.column_stack((xs, ys + sign, np.full_like(xs, config.GROUND_Z)))
        uv, ok = fs.project_points(pts, img.shape, calib)
        lu.draw_polylines(out, [uv[ok].astype(np.int32)], col, 3)
    for sgn in (1, -1):
        pts = np.column_stack((xs, ys + sgn * GT_HALF_WIDTH, np.full_like(xs, config.GROUND_Z)))
        uv, ok = fs.project_points(pts, img.shape, calib)
        lu.draw_polylines(out, [uv[ok].astype(np.int32)], (0, 0, 200), 1)
    lu.draw_polylines(out, fs.corridor_edge_polylines(img.shape, calib, curvature=kappa), (0, 165, 255), 2)
    if fit is not None:
        lu.draw_polylines(out, lu.lane_edge_polylines(bev, fit), (0, 255, 0), 3)
    cv2.putText(out, f"lane: {status}  (red = driven path, green = lane, orange = corridor)",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out)


# ----------------------------------------------------------------------- report ---
def fmt_table(acc, title):
    lines = [title,
             f"  {'estimator':<28}{'avail':>7}{'recall':>8}{'prec':>7}{'IoU':>7}   "
             f"{'(when available) rec':>22}{'prec':>7}{'IoU':>7}   {'centre err':>10}{'width':>7}"]
    for m in METHODS:
        a = acc[m]
        s, sa = a.stats(), a.stats(available_only=True)
        ce = f"{np.median(a.center_err):.2f} m" if a.center_err else "   -"
        wd = f"{np.mean(a.widths):.1f} m" if a.widths else "  -"
        lines.append(f"  {m:<28}{s['avail']:>7.0%}{s['recall']:>8.2f}{s['precision']:>7.2f}{s['iou']:>7.2f}   "
                     f"{sa['recall']:>22.2f}{sa['precision']:>7.2f}{sa['iou']:>7.2f}   {ce:>10}{wd:>7}")
    return lines


def merge(accs):
    out = {m: Acc() for m in METHODS}
    for a in accs:
        for m in METHODS:
            o, s = out[m], a[m]
            o.tp += s.tp; o.pred += s.pred; o.gt += s.gt; o.frames += s.frames
            o.available += s.available; o.tp_av += s.tp_av; o.pred_av += s.pred_av; o.gt_av += s.gt_av
            o.center_err += s.center_err; o.widths += s.widths
    return out


def plot(acc_all, path):
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    x = np.arange(len(METHODS)); wdt = 0.27
    for k, (name, key) in enumerate((('recall (coverage of driven path)', 'recall'),
                                     ('precision', 'precision'), ('IoU', 'iou'))):
        ax[0].bar(x + (k - 1) * wdt, [np.nan_to_num(acc_all[m].stats()[key]) for m in METHODS], wdt, label=name)
    ax[0].set_xticks(x); ax[0].set_xticklabels(METHODS, rotation=25, ha='right', fontsize=8)
    ax[0].set_ylim(0, 1); ax[0].legend(fontsize=8); ax[0].set_title('Ego-path estimators vs. driven footprint (ground plane)')
    for m, c in (('lane (legacy)', 'tab:red'), ('lane (metric)', 'tab:orange'), ('lane (metric, gated)', 'tab:green')):
        if acc_all[m].center_err:
            ax[1].hist(acc_all[m].center_err, bins=np.arange(0, 3.01, 0.1), alpha=0.55, color=c,
                       label=f"{m} (n={len(acc_all[m].center_err)}, median {np.median(acc_all[m].center_err):.2f} m)")
    ax[1].set_xlabel('mean |lane centre - driven path| per frame (m)'); ax[1].set_ylabel('frames')
    ax[1].legend(fontsize=8); ax[1].set_title('Lane centre error when a lane is reported')
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--drive', type=Path, action='append')
    p.add_argument('--debug-frames', type=int, default=3)
    p.add_argument('--output-dir', type=Path, default=config.OUTPUT_DIR)
    p.add_argument('--plot', type=Path, default=config.ROOT / 'docs' / 'lane_eval.png')
    args = p.parse_args()

    calib = load_calibration()
    drives = [Drive(d) for d in args.drive] if args.drive else discover_drives()
    out, per_drive = [], []
    for drive in drives:
        acc, status, n = run_drive(drive, calib, args.debug_frames, args.output_dir / 'lane_debug')
        per_drive.append(acc)
        out += fmt_table(acc, f"\n== {drive.name}: {n} frames with >= {X_MAX - 2:.0f} m of future path ==")
        out.append("  lane status (metric): " + ", ".join(f"{k} {v}" for k, v in sorted(status['metric'].items())))
        out.append("  lane status (legacy): " + ", ".join(f"{k} {v}" for k, v in sorted(status['legacy'].items())))
    acc_all = merge(per_drive)
    out += fmt_table(acc_all, "\n== all drives pooled ==")
    text = '\n'.join(out)
    print(text)
    (args.output_dir / 'lane_summary.txt').write_text(text + '\n')
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    plot(acc_all, args.plot)
    print(f"\nPlot: {args.plot}   Debug frames: {args.output_dir / 'lane_debug'}")


if __name__ == '__main__':
    main()
