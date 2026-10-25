"""Autonomous perception & safety pipeline on KITTI.

Run from anywhere:
    python src/main.py                       # every drive found under data/
    python src/main.py --drive "data/2011_09_26_drive_0005_sync" --max-frames 40
    python src/main.py --ttc fused           # use the learned TTC (needs models/ttc_gru.pt)

Outputs, per drive, in <output-dir>/<drive name>/ (default ./output):
  frames/*.png, adu_perception_final.mp4, [demo.gif]
  telemetry.csv   one row per frame (planner state, analytic + learned TTC, ...)
  tracks.csv      one row per tracked object per frame (training data for the TTC net)
Then:  python src/evaluate.py   and   python src/train_ttc.py
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import lane_utils as lu
import obstacle_utils as ou
import video_utils as vu
import fusion_utils as fs
import predict
import tracker3d as t3
import ui_utils as ui
from behavior_manager import BehaviorManager, TTC_MODES
from calibration import load_calibration
from datasets import Drive, discover_drives
from ttc_model import LearnedTTC


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--drive', type=Path, action='append',
                   help='drive folder (repeatable). Default: every drive under data/')
    p.add_argument('--output-dir', type=Path, default=config.OUTPUT_DIR)
    p.add_argument('--model', type=Path, default=config.MODEL_PATH, help='YOLO weights')
    p.add_argument('--ttc', choices=TTC_MODES, default='analytic',
                   help='TTC source for the planner (learned/fused need --ttc-model)')
    p.add_argument('--ttc-model', type=Path, default=config.TTC_MODEL_PATH)
    p.add_argument('--max-frames', type=int, default=None)
    p.add_argument('--no-video', action='store_true')
    p.add_argument('--predict', action='store_true',
                   help='also SLOW for objects predicted to enter the ego path within 3 s (advisory: never STOPs; implies --ttc kalman)')
    p.add_argument('--profile', action='store_true', help='print per-stage latency (also saved to latency.csv)')
    p.add_argument('--gif', action='store_true', help='also write <drive>/demo.gif')
    return p.parse_args()


def run_drive(drive: Drive, args, calib):
    out_dir = args.output_dir / drive.name
    frames_dir = out_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob('*.png'):   # don't mix with a previous run
        old.unlink()

    model = YOLO(str(args.model))          # fresh tracker state per drive
    planner = BehaviorManager(ttc_mode=args.ttc)
    lane_tracker = None            # needs the image size: created with the first frame
    bev = None
    ttc_net = LearnedTTC.from_file(args.ttc_model) if args.ttc in ('learned', 'fused') else None

    frames = drive.frames[:args.max_frames] if args.max_frames else drive.frames
    dts, times = drive.dts(), drive.times

    print(f"\n[{drive.name}] {len(frames)} frames -> {out_dir}  (ttc mode: {args.ttc})")
    frame_rows, track_rows, track3d_rows, latency_rows = [], [], [], []
    tracker3d = t3.Tracker3D()

    for idx, f_path in enumerate(frames):
        t_frame = time.perf_counter()
        img = cv2.imread(str(f_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # ---- ego motion --------------------------------------------------------
        ego_speed = drive.ego_speed(idx)
        yaw_rate = drive.yaw_rate(idx)
        curvature = 0.0
        if config.USE_CURVED_CORRIDOR and ego_speed and ego_speed > config.CURVE_MIN_SPEED and yaw_rate is not None:
            curvature = float(np.clip(yaw_rate / ego_speed, -config.CURVE_MAX, config.CURVE_MAX))

        t_lane = time.perf_counter()
        # ---- lane detection (metric bird's-eye view) -----------------------------
        if bev is None:
            bev = lu.make_bev(calib, img.shape)
            lane_tracker = lu.LaneTracker(bev)
        mask = lu.lane_color_mask(img)
        warped = cv2.warpPerspective(mask, bev.M, bev.size)
        fit, lane_status = lane_tracker.update(warped)
        if fit is not None and not lu.consistent_with_path(fit, curvature):
            fit, lane_status = None, 'rejected'      # markings disagree with where the car is heading
        lane_mask = lu.lane_mask_from_fit(img.shape, bev, fit) if fit else None

        # Ego path in image space: detected lane UNION the (curved) corridor.
        path_mask, path_source = fs.build_path_mask(img.shape, lane_mask, calib, curvature=curvature)
        res_img = lu.draw_mask_overlay(
            img, path_mask,
            color=(0, 255, 0) if path_source == 'lane' else (0, 165, 255),
            alpha=0.25 if path_source == 'lane' else 0.15)

        # Edges: ego-path corridor (orange, always) and detected lane (bright green)
        lu.draw_polylines(res_img, fs.corridor_edge_polylines(img.shape, calib, curvature=curvature), (0, 165, 255), 2)
        if fit:
            lane_color = {'detected': (0, 255, 0), 'single': (255, 255, 0)}.get(lane_status, (0, 200, 120))
            lu.draw_polylines(res_img, lu.lane_edge_polylines(bev, fit), lane_color, 4)

        t_yolo = time.perf_counter()
        # ---- object tracking (BoT-SORT keeps persistent IDs) --------------------
        yolo_res = model.track(img, persist=True, tracker="botsort.yaml", verbose=False)[0]
        tracks = ou.extract_tracks(yolo_res, calib.fx)

        t_lidar = time.perf_counter()
        # ---- LiDAR: obstacle points, projected into the image ------------------
        scan = drive.lidar(idx)
        in_path_dist = 99.0
        n_path_pts = 0
        uv_valid = np.zeros((0, 2))
        depth_valid = np.zeros(0)
        if scan is not None:
            obs_pts = fs.obstacle_points(scan)
            uv, valid = fs.project_points(obs_pts, img.shape, calib)
            uv_valid, depth_valid = uv[valid], obs_pts[valid, 0]

            on_path = fs.in_path_lidar(obs_pts, img.shape, path_mask, calib, curvature=curvature)
            n_path_pts = int(on_path.sum())
            in_path_dist = fs.in_path_distance(obs_pts[on_path, 0])

            for (u, v), is_path in zip(uv[valid], on_path[valid]):
                if is_path:
                    cv2.circle(res_img, (int(u), int(v)), 1, (255, 0, 0), -1)

        # ---- LiDAR clusters -> 3D Kalman tracks (object velocity in the world frame) ----
        tracks3d = []
        pose = drive.ego_pose_world(idx, calib)
        if pose is not None:
            clusters = t3.cluster_obstacles(obs_pts) if scan is not None else []
            tracks3d = tracker3d.update(clusters, pose, dts[idx], drive.ego_velocity_world(idx))

        def centre_uv(tr):
            uv, ok = fs.project_points(np.array([[tr.rel_pos[0], tr.rel_pos[1], tr.z]]), img.shape, calib)
            return uv[0] if ok[0] else None
        matched3d = t3.match_to_boxes(tracks3d, tracks, centre_uv)
        for tr in tracks3d:
            ttc3, lo3, hi3 = tr.ttc()
            track3d_rows.append({
                'frame': idx, 'id': tr.id, 'x': round(float(tr.rel_pos[0]), 3), 'y': round(float(tr.rel_pos[1]), 3),
                'vx_rel': round(float(tr.rel_vel[0]), 3), 'vy_rel': round(float(tr.rel_vel[1]), 3),
                'speed_world': round(tr.speed_world, 3), 'range': round(tr.bumper_range, 3),
                'closing': round(tr.closing_speed, 3),
                'ttc': '' if not math.isfinite(ttc3) else round(ttc3, 3),
                'ttc_lo': '' if not math.isfinite(lo3) else round(lo3, 3),
                'ttc_hi': '' if not math.isfinite(hi3) else round(hi3, 3),
                'in_path': int(t3.in_ego_path(tr, curvature)),
            })
        t_fuse = time.perf_counter()

        # ---- fusion: per-object distance + in-path flag -------------------------
        for t in tracks:
            lidar_d = fs.box_lidar_distance(uv_valid, depth_valid, t['box']) if len(uv_valid) else None
            t['lidar_dist'] = lidar_d
            t['dist'], t['dist_source'] = (lidar_d, 'lidar') if lidar_d is not None else (t['mono_dist'], 'mono')
            t['in_path'] = fs.box_in_mask(path_mask, t['box'])
            m3 = matched3d.get(t['id'])
            if m3 is not None:
                t['ttc_3d'] = m3.ttc()[0]
                t['obj_speed'] = m3.speed_world

        # ---- learned TTC (camera + ego speed only) ------------------------------
        if ttc_net is not None:
            ttc_net.update(tracks, ego_speed, float(dts[idx]), idx, img.shape)

        for t in tracks:
            x1, y1, x2, y2 = t['box']
            ld = t['lidar_dist']
            tl = t.get('ttc_learned')
            track_rows.append({
                'frame': idx, 'time': round(float(times[idx]), 4), 'dt': round(float(dts[idx]), 4),
                'id': t['id'], 'cls': t['class'], 'class': model.names[t['class']],
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'img_w': w, 'img_h': h,
                'ego_speed': '' if ego_speed is None else round(ego_speed, 3),
                'mono_dist': round(t['mono_dist'], 3),
                'lidar_dist': '' if ld is None else round(ld, 3),
                'in_path': int(t['in_path']),
                'ttc_learned': '' if tl is None else round(tl, 3),
                'ttc_3d': '' if t.get('ttc_3d') is None or not math.isfinite(t['ttc_3d']) else round(t['ttc_3d'], 3),
                'obj_speed': '' if t.get('obj_speed') is None else round(t['obj_speed'], 3),
            })

        # ---- behaviour decision -------------------------------------------------
        # Predicted ego-path conflicts (CTRV prediction): objects not in the corridor *yet* but on course
        # to be. They enter the planner as extra in-path obstacles carrying their conflict time.
        extras = []
        if args.predict and pose is not None:
            for tr in tracks3d:
                if t3.in_ego_path(tr, curvature) or tr.is_static:
                    continue
                tc = predict.time_to_conflict(tr, pose, ego_speed or 0.0, curvature, model='ctrv', horizon=3.0)
                if math.isfinite(tc):
                    extras.append({'id': 1_000_000 + tr.id, 'dist': tr.bumper_range, 'in_path': True, 'advisory': True, 'ttc_3d': max(tc, 0.05)})
        t_plan = time.perf_counter()
        decision = planner.update_state(tracks + extras, in_path_dist, dt=dts[idx], ego_speed=ego_speed)
        t_render = time.perf_counter()

        # ---- visualisation ------------------------------------------------------
        res_img = ou.draw_tracked_obstacles(res_img, tracks, model.names)
        res_img = ui.draw_dashboard(res_img, tracks, decision)
        res_img = ui.draw_status(res_img, decision, path_source, lane_status)
        cv2.imwrite(str(frames_dir / f_path.name), res_img)
        t_end = time.perf_counter()
        latency_rows.append({
            'frame': idx,
            'lane_ms': round((t_yolo - t_lane) * 1e3, 2), 'yolo_ms': round((t_lidar - t_yolo) * 1e3, 2),
            'lidar_3d_ms': round((t_fuse - t_lidar) * 1e3, 2), 'fusion_ms': round((t_plan - t_fuse) * 1e3, 2),
            'planner_ms': round((t_render - t_plan) * 1e3, 3), 'render_ms': round((t_end - t_render) * 1e3, 2),
            'perception_planning_ms': round((t_render - t_frame) * 1e3, 2)})

        finite = lambda v: '' if (v is None or not math.isfinite(v)) else round(v, 3)
        frame_rows.append({
            'frame': idx, 'dt': round(float(dts[idx]), 4),
            'ego_speed': '' if ego_speed is None else round(ego_speed, 3),
            'state': decision.state, 'raw_state': decision.raw_state,
            'min_dist': round(decision.min_dist, 3),
            'ttc': finite(decision.ttc), 'ttc_analytic': finite(decision.ttc_analytic),
            'ttc_learned': finite(decision.ttc_learned),
            'stop_dist': round(decision.stop_dist, 3), 'slow_dist': round(decision.slow_dist, 3),
            'n_tracks': len(tracks), 'n_in_path': sum(t['in_path'] for t in tracks),
            'lidar_path_pts': n_path_pts,
            'lane_status': lane_status, 'path_source': path_source,
        })
        if idx % 25 == 0:
            print(f"  frame {idx + 1}/{len(frames)}  {decision.state:8s}  path={path_source}/{lane_status}")

    _write_csv(out_dir / 'telemetry.csv', frame_rows)
    _write_csv(out_dir / 'tracks.csv', track_rows)
    _write_csv(out_dir / 'tracks3d.csv', track3d_rows)
    _write_csv(out_dir / 'latency.csv', latency_rows)
    if args.profile and latency_rows:
        _print_latency(latency_rows)

    if not args.no_video:
        video_path = out_dir / 'adu_perception_final.mp4'
        vu.save_to_video(frames_dir, str(video_path), config.VIDEO_FPS)
        print(f"  video: {video_path}")
    if args.gif:
        vu.save_gif(frames_dir, out_dir / 'demo.gif', config.VIDEO_FPS)
        print(f"  gif:   {out_dir / 'demo.gif'}")


def _print_latency(rows):
    keys = ['lane_ms', 'yolo_ms', 'lidar_3d_ms', 'fusion_ms', 'planner_ms', 'render_ms', 'perception_planning_ms']
    print(f"  latency (ms)      {'median':>8}{'p95':>8}{'max':>8}")
    for k in keys:
        v = np.array([r[k] for r in rows])
        print(f"  {k:<18}{np.median(v):8.1f}{np.percentile(v, 95):8.1f}{v.max():8.1f}")
    total = np.array([r['perception_planning_ms'] for r in rows])
    print(f"  -> {1000 / np.median(total):.1f} fps median, {1000 / np.percentile(total, 95):.1f} fps at p95 "
          f"(includes image decode + LiDAR parsing, excludes drawing/video; KITTI runs at 10 Hz)")


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.predict:
        args.ttc = 'kalman'
    if args.ttc != 'analytic' and args.ttc != 'kalman' and not args.ttc_model.exists():
        sys.exit(f"--ttc {args.ttc} needs a trained model at {args.ttc_model}. "
                 "Run `python src/train_ttc.py` first.")
    drives = [Drive(p) for p in args.drive] if args.drive else discover_drives()
    if not drives:
        sys.exit(f"No drives found under {config.DATA_DIR}")
    calib = load_calibration()
    for drive in drives:
        run_drive(drive, args, calib)
    print("\nDone. Next: `python src/evaluate.py` (metrics/plots), `python src/train_ttc.py` (learned TTC).")


if __name__ == '__main__':
    main()
