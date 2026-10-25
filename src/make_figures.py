"""Generate the README figures: the architecture diagram and the demo-frame PDF.

    python src/make_figures.py                       # both, default frame
    python src/make_figures.py --frame 85            # pick another frame of the demo drive
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / 'docs'

STAGES = {          # fill, edge
    'input': ('#eef2f7', '#5b6b7f'),
    'percep': ('#e3f1e6', '#2e7d4f'),
    'fusion': ('#fff3dc', '#b9770e'),
    'decide': ('#e6e9fb', '#4a56b8'),
    'output': ('#f6e4f0', '#9b3d83'),
}


def _box(ax, x, y, w, h, title, lines, stage):
    fill, edge = STAGES[stage]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
                                fc=fill, ec=edge, lw=1.6))
    ax.text(x + w / 2, y + h - 0.2, title, ha='center', va='top', fontsize=11.5, weight='bold', color='#222')
    ax.text(x + w / 2, y + h - 0.62, '\n'.join(lines), ha='center', va='top', fontsize=9, color='#333',
            linespacing=1.45)
    return dict(l=(x, y + h / 2), r=(x + w, y + h / 2), t=(x + w / 2, y + h), b=(x + w / 2, y))


def _arrow(ax, p, q, label=None, dashed=False, rad=0.0):
    ax.annotate('', xy=q, xytext=p, arrowprops=dict(
        arrowstyle='-|>', lw=1.5, color='#444', ls='--' if dashed else '-', shrinkA=2, shrinkB=2,
        connectionstyle=f'arc3,rad={rad}'))
    if label:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.12, label, fontsize=8, ha='center', color='#555',
                style='italic')


def architecture(out_path):
    fig, ax = plt.subplots(figsize=(20, 8.6))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 8.6)
    ax.axis('off')

    for x, w, name, stage in ((0.2, 3.2, 'SENSORS', 'input'), (4.1, 4.0, 'PERCEPTION', 'percep'),
                              (8.8, 4.0, 'FUSION', 'fusion'), (13.5, 3.6, 'ESTIMATION & DECISION', 'decide'),
                              (17.6, 2.2, 'OUTPUT', 'output')):
        ax.text(x + w / 2, 8.35, name, ha='center', fontsize=12, weight='bold', color=STAGES[stage][1])

    cam = _box(ax, 0.2, 6.5, 3.2, 1.5, 'Camera', ['KITTI image_02', '1242 x 375, 10 Hz'], 'input')
    lid = _box(ax, 0.2, 4.7, 3.2, 1.5, 'Velodyne LiDAR', ['64-beam point cloud', '.bin (sync) / .txt (extract)'], 'input')
    oxt = _box(ax, 0.2, 2.5, 3.2, 1.5, 'OXTS GPS/IMU', ['ego speed, yaw rate', 'nearest-timestamp match'], 'input')
    ax.add_patch(FancyBboxPatch((0.2, 0.3), 3.2, 1.5, boxstyle='round,pad=0.02,rounding_size=0.12',
                                fc='#f8f8f8', ec='#5b6b7f', lw=1.4, ls='--'))
    ax.text(1.8, 1.55, 'Calibration', ha='center', va='top', fontsize=11.5, weight='bold')
    ax.text(1.8, 1.1, 'used by the bird\'s-eye homography,\nLiDAR projection and in-path tests', ha='center',
            va='top', fontsize=9, color='#333', linespacing=1.45)

    lane = _box(ax, 4.1, 5.9, 4.0, 2.1, 'Lane detection (metric)',
                ['HLS white/yellow + top-hat mask', 'ground-plane homography -> bird\'s-eye view',
                 'quadratic fit in metres + width/ego checks', 'gate vs yaw-rate path (<= 1 m)'], 'percep')
    trk = _box(ax, 4.1, 3.3, 4.0, 1.7, 'Object tracking',
               ['YOLOv8 detection + BoT-SORT IDs', 'monocular range from box height'], 'percep')
    ego = _box(ax, 4.1, 1.3, 4.0, 1.5, 'Ego motion', ['curvature k = yaw rate / speed', 'causal, clipped to +/-0.05 1/m'],
               'percep')

    path = _box(ax, 8.8, 6.6, 4.0, 1.5, 'Ego-path mask',
                ['curved corridor(k)  U  gated lane', 'orange = corridor, green = lane edges'], 'fusion')
    obs = _box(ax, 8.8, 4.6, 4.0, 1.5, 'LiDAR obstacles + 3D tracks',
               ['local per-cell ground removal, projected', 'BEV clusters -> Kalman tracks (world frame)',
                'object velocity, TTC with 90% band'], 'fusion')
    fus = _box(ax, 8.8, 1.3, 4.0, 2.7, 'Per-object fusion',
               ['nearest-surface LiDAR range per box', 'cluster-supported in-path distance',
                'ground-footprint in-path test', 'monocular fallback'], 'fusion')

    rng = _box(ax, 13.5, 5.5, 3.6, 2.3, 'Range & analytic TTC',
               ['EMA range + range-jump gate', 'closing speed -> TTC', 'LiDAR range-rate safety layer',
                'Kalman TTC (--ttc kalman)'], 'decide')
    ttc = _box(ax, 13.5, 3.4, 3.6, 1.7, 'TTC fusion',
               ['physics-informed GRU', '(camera + ego speed only)', 'analytic / learned / fused / kalman'], 'decide')
    fsm = _box(ax, 13.5, 0.9, 3.6, 2.3, 'Behavior planner (FSM)',
               ['DRIVE / SLOWING / STOPPED', 'stop zone 2 m + c^2/2a', 'slow zone 6 m + 1 s headway',
                'TTC 1 s / 3 s, hysteresis'], 'decide')
    out = _box(ax, 17.7, 2.6, 2.1, 2.6, 'Outputs',
               ['annotated video', 'demo GIF', 'telemetry.csv', 'tracks.csv'], 'output')

    def line(*pts):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color='#444', lw=1.5, solid_capstyle='round')

    # sensors -> perception / fusion
    _arrow(ax, cam['r'], (4.1, 7.0))
    _arrow(ax, cam['r'], (4.1, 4.4))
    _arrow(ax, oxt['r'], ego['l'])
    _arrow(ax, lid['r'], (8.8, 5.45))
    # perception -> fusion
    _arrow(ax, lane['r'], (8.8, 7.5))
    line((8.1, 2.05), (8.45, 2.05), (8.45, 6.9))
    _arrow(ax, (8.45, 6.9), (8.8, 6.9))
    _arrow(ax, trk['r'], (8.8, 3.2))
    _arrow(ax, path['b'], obs['t'])
    _arrow(ax, obs['b'], (10.8, 4.0))
    # fusion -> decision (shared vertical channel)
    line((12.8, 2.7), (13.15, 2.7), (13.15, 6.6))
    _arrow(ax, (13.15, 6.6), (13.5, 6.6))
    _arrow(ax, (13.15, 4.25), (13.5, 4.25))
    _arrow(ax, rng['b'], ttc['t'])
    _arrow(ax, ttc['b'], fsm['t'])
    # ego speed -> planner along the bottom
    line((6.1, 1.3), (6.1, 0.4), (15.3, 0.4))
    _arrow(ax, (15.3, 0.4), (15.3, 0.9))
    ax.text(10.4, 0.5, 'ego speed -> stop / slow zones', fontsize=8, color='#555', style='italic', ha='center')
    # planner -> outputs
    _arrow(ax, fsm['r'], (17.7, 3.2))

    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', out_path)


def demo_pdf(frame_path, pdf_path, png_path, caption_rows):
    img = mpimg.imread(frame_path)
    fig = plt.figure(figsize=(11, 6.2))
    ax = fig.add_axes([0.03, 0.30, 0.94, 0.58])
    ax.imshow(img)
    ax.axis('off')
    fig.text(0.5, 0.945, 'Lane and object detection with LiDAR fusion - KITTI 2011_09_26 drive 0013',
             ha='center', fontsize=14, weight='bold')
    fig.text(0.5, 0.905, 'Single frame from the demo pipeline output', ha='center', fontsize=10, color='#555')
    ax2 = fig.add_axes([0.05, 0.03, 0.90, 0.24])
    ax2.axis('off')
    tbl = ax2.table(cellText=caption_rows, colWidths=[0.2, 0.8], cellLoc='left', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.55)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#cccccc')
        if c == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f1f3f6')
    fig.savefig(pdf_path, facecolor='white')
    fig.savefig(png_path, dpi=150, facecolor='white')
    plt.close(fig)
    print('wrote', pdf_path, 'and', png_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drive', default='2011_09_26_drive_0013_extract')
    ap.add_argument('--frame', type=int, default=83)
    args = ap.parse_args()

    DOCS.mkdir(exist_ok=True)
    architecture(DOCS / 'architecture.png')
    frame = ROOT / 'output' / args.drive / 'frames' / f'{args.frame:010d}.png'
    rows = [
        ['Orange lines', 'Ego-path corridor edges, bent by the measured yaw rate (always shown)'],
        ['Green lines / fill', 'Marking-based lane, drawn only because it agrees with the car\'s heading (path gate)'],
        ['Boxes', 'YOLOv8 + BoT-SORT tracks; label = ID, class, range, source (L = LiDAR, M = monocular)'],
        ['Box colour', 'Yellow/red = in the ego path (red if closer than 7 m), the planner reacts to it; green = off path'],
        ['HUD', 'Planner state, in-path distance, ego speed, analytic vs learned TTC'],
        ['Mini-map', 'Top-down view of the tracked objects relative to the ego vehicle'],
    ]
    demo_pdf(frame, DOCS / 'demo_frame.pdf', DOCS / 'demo_frame.png', rows)


if __name__ == '__main__':
    main()
