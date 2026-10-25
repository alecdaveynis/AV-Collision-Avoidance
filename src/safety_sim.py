"""Closed-loop safety evaluation of the planner on generated scenarios.

KITTI has no near-collisions, so logged data cannot say whether the planner would have
*prevented* one. This replays the real perception back end (LiDAR cluster measurements ->
``Tracker3D`` Kalman tracker -> planner) in a small kinematic simulator where the ego car's
braking changes what happens next.

Scenarios (randomised per episode)
  hazards : lead_brake, stationary (parked car in lane), cut_in, ped_crossing
  benign  : follow, adjacent_pass, ped_sidewalk, oncoming   (braking here is a false alarm)

Sensor model: cluster centroid noise (0.1 m + 0.2 % of range), 10 % dropout, range limits
(cars 55 m, pedestrians 35 m), and one-frame ghost clusters in the ego lane.
Actuation: commanded deceleration through a first-order lag (tau = 0.2 s), max 8 m/s^2.

Policies compared
  planner-analytic : BehaviorManager, range-rate TTC          (the production planner)
  planner-kalman   : BehaviorManager, TTC from the 3D Kalman track
  planner-improved : SafeBehaviorManager (src/safe_braking.py): delay-compensated required-deceleration braking, Kalman mode
  rss              : Responsibility-Sensitive Safety longitudinal rule (Shalev-Shwartz et al. 2017)
  aeb              : Kalman TTC AEB (partial brake < 2.5 s, full < 1.2 s)
  predictive       : constant-velocity prediction of ego/object conflict within 4 s
  none             : never brakes (shows which episodes are actually hazards)

Usage: python src/safety_sim.py [--n 100] [--seed 0] [--plots-dir docs]
"""
import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import predict
import tracker3d as t3
from behavior_manager import BehaviorManager
from safe_braking import make_behavior_manager

DT = 0.1
MAX_DECEL = 8.0
LAG = 0.2
EGO_LEN = 4.5
EGO_HALF_W = 1.0
EPISODE_S = 14.0
SPEED_RANGE = (8.0, 16.0)       # m/s ego start speed

HAZARD = ('lead_brake', 'stationary', 'cut_in', 'ped_crossing')
BENIGN = ('follow', 'adjacent_pass', 'ped_sidewalk', 'oncoming')


# --- actors / scenarios ---------------------------------------------------------------
@dataclass
class Actor:
    kind: str            # 'car' | 'ped'
    x: float
    y: float
    vx: float
    vy: float = 0.0
    length: float = 4.5
    width: float = 1.8
    script: object = None        # callable(actor, t, ego_x, dt)

    def step(self, t, ego_x, dt):
        if self.script:
            self.script(self, t, ego_x, dt)
        self.x += self.vx * dt
        self.y += self.vy * dt


def car(x, y, vx, **kw):
    return Actor('car', x, y, vx, **kw)


def ped(x, y, vx, vy, **kw):
    return Actor('ped', x, y, vx, vy, length=0.6, width=0.6, **kw)


def sc_lead_brake(rng, v0):
    lead = car(rng.uniform(25, 45), 0.0, v0)
    t_b, a = rng.uniform(2, 4), rng.uniform(3, 8)

    def script(a_, t, ego_x, dt):
        if t >= t_b:
            a_.vx = max(a_.vx - a * dt, 0.0)
    lead.script = script
    return [lead]


def sc_stationary(rng, v0):
    return [car(rng.uniform(50, 90), rng.uniform(-0.3, 0.3), 0.0)]


def sc_cut_in(rng, v0):
    side = rng.choice([-1.0, 1.0])
    c = car(rng.uniform(14, 28), side * 3.5, v0 * rng.uniform(0.55, 0.9))
    t0, lat = rng.uniform(1.0, 3.0), rng.uniform(0.9, 1.6)

    def script(a_, t, ego_x, dt):
        if t >= t0 and abs(a_.y) > 0.05:
            a_.vy = -math.copysign(lat, a_.y)
        elif abs(a_.y) <= 0.05:
            a_.vy, a_.y = 0.0, 0.0
    c.script = script
    return [c]


def sc_ped_crossing(rng, v0):
    side = rng.choice([-1.0, 1.0])
    y0 = side * rng.uniform(4.0, 5.5)
    x_p = rng.uniform(40, 60)
    t_arrive = x_p / v0
    start = max(t_arrive - abs(y0) / 1.4 + rng.uniform(-0.3, 0.8), 0.0)
    p = ped(x_p, y0, 0.0, 0.0)

    def script(a_, t, ego_x, dt):
        a_.vy = -side * 1.4 if (t >= start and abs(a_.y) < abs(y0) + 1.0 and a_.y * side > -4.0) else 0.0
    p.script = script
    return [p]


def sc_follow(rng, v0):
    # a lead at a normal 2-3 s time headway, matching speed: nothing here should need braking
    return [car(max(v0 * rng.uniform(2.0, 3.0), 25.0), 0.0, v0 + rng.uniform(-1.0, 1.0))]


def sc_adjacent_pass(rng, v0):
    side = rng.choice([-1.0, 1.0])
    return [car(rng.uniform(-20, 30), side * 3.5, v0 + rng.uniform(-3, 3))]


def sc_ped_sidewalk(rng, v0):
    side = rng.choice([-1.0, 1.0])
    return [ped(rng.uniform(25, 50), side * rng.uniform(4, 6), rng.uniform(-1.4, 1.4), 0.0)]


def sc_oncoming(rng, v0):
    return [car(rng.uniform(60, 90), 3.5, -rng.uniform(8, 14))]


SCENARIOS = {'lead_brake': sc_lead_brake, 'stationary': sc_stationary, 'cut_in': sc_cut_in,
             'ped_crossing': sc_ped_crossing, 'follow': sc_follow, 'adjacent_pass': sc_adjacent_pass,
             'ped_sidewalk': sc_ped_sidewalk, 'oncoming': sc_oncoming}


# --- sensor model -----------------------------------------------------------------------
def measure(actors, ego_x, rng):
    clusters = []
    for a in actors:
        rear = a.x - a.length / 2 - ego_x
        if a.x + a.length / 2 < ego_x - 1.0 or rear < 0.3 and a.x < ego_x:
            continue
        r = max(rear, 0.0)
        if r > (55.0 if a.kind == 'car' else 35.0) or rng.random() < 0.10:
            continue
        depth = 0.8 if a.kind == 'car' else 0.3
        sx = 0.1 + 0.002 * r
        near = max(rear + rng.normal(0, sx), 0.1)
        clusters.append(t3.Cluster(x=near + depth / 2, y=a.y + rng.normal(0, 0.1), z=0.8,
                                   x_near=near, ext_x=depth, ext_y=a.width, n=40, z_top=1.5))
    if rng.random() < 0.04:        # ghost: one-frame in-lane clutter
        gx = rng.uniform(10, 40)
        clusters.append(t3.Cluster(x=gx, y=rng.uniform(-1, 1), z=0.8, x_near=gx - 0.2,
                                   ext_x=0.4, ext_y=0.4, n=12, z_top=1.2))
    return clusters


# --- policies -----------------------------------------------------------------------------
def _inpath(tr):
    return t3.in_ego_path(tr)


class NonePolicy:
    name = 'none'

    def decide(self, tracks, clusters, pose, v, dt):
        return 0.0


class PlannerPolicy:
    def __init__(self, mode, planner='baseline'):
        self.pm = make_behavior_manager(planner, mode)
        self.name = f'planner-{mode}' if planner == 'baseline' else f'planner-{planner}-{mode}'

    def decide(self, tracks, clusters, pose, v, dt):
        obs = []
        for tr in tracks:
            raw = [c.x_near for c in clusters
                   if abs(c.x - tr.rel_pos[0]) < 1.5 and abs(c.y - tr.rel_pos[1]) < 1.5]
            obs.append({'id': tr.id, 'dist': min(raw) if raw else tr.bumper_range,
                        'in_path': _inpath(tr), 'ttc_3d': tr.ttc()[0], 'closing_3d': tr.closing_speed, 'track_hits': tr.hits})
        lid = min([c.x_near for c in clusters
                   if c.x_near > 0 and abs(c.y) < config.PATH_MAX_LATERAL + 0.5 * c.ext_y], default=99.0)
        return MAX_DECEL * self.pm.update_state(obs, lid, dt=dt, ego_speed=v).brake


class RSSPolicy:
    """Longitudinal RSS: brake at >= b_min whenever the gap is below the safe distance."""
    name = 'rss'
    RHO, A_ACC, B_MIN, B_MAX = 0.5, 2.0, 4.0, 8.0

    def decide(self, tracks, clusters, pose, v, dt):
        cmd = 0.0
        for tr in tracks:
            if not _inpath(tr):
                continue
            vo = max(v - tr.closing_speed, 0.0)          # object forward speed (>= 0 : worst case)
            vr = v + self.RHO * self.A_ACC
            d_min = max(v * self.RHO + 0.5 * self.A_ACC * self.RHO ** 2 + vr ** 2 / (2 * self.B_MIN)
                        - vo ** 2 / (2 * self.B_MAX), 0.0)
            gap = tr.bumper_range
            if gap < d_min:
                cmd = max(cmd, self.B_MIN if gap > 0.5 * d_min else MAX_DECEL)
        return cmd


class AEBPolicy:
    name = 'aeb'

    def decide(self, tracks, clusters, pose, v, dt):
        cmd = 0.0
        for tr in tracks:
            if not _inpath(tr):
                continue
            ttc = tr.ttc()[0]
            if ttc < 1.2:
                cmd = max(cmd, MAX_DECEL)
            elif ttc < 2.5:
                cmd = max(cmd, 0.4 * MAX_DECEL)
        return cmd


class PredictivePolicy:
    """Brake on the predicted ego/object footprint overlap, not on the current lane membership."""
    name = 'predictive'

    def __init__(self, model='cv'):
        self.model = model

    def decide(self, tracks, clusters, pose, v, dt):
        cmd = 0.0
        for tr in tracks:
            tc = predict.time_to_conflict(tr, pose, v, 0.0, model=self.model, horizon=4.0)
            if tc < 1.5:
                cmd = max(cmd, MAX_DECEL)
            elif tc < 2.8:
                cmd = max(cmd, 0.4 * MAX_DECEL)
        return cmd


POLICIES = {
    'none': NonePolicy,
    'planner-analytic': lambda: PlannerPolicy('analytic'),
    'planner-kalman': lambda: PlannerPolicy('kalman'),
    'planner-improved': lambda: PlannerPolicy('kalman', planner='improved'),
    'rss': RSSPolicy,
    'aeb': AEBPolicy,
    'predictive': PredictivePolicy,
}


# --- episode ------------------------------------------------------------------------------
def run_episode(policy, scenario, rng, v0=None):
    v = v0 if v0 is not None else rng.uniform(*SPEED_RANGE)
    v_start = v
    actors = SCENARIOS[scenario](rng, v)
    tracker = t3.Tracker3D()
    ego_x, a_act, peak_decel = 0.0, 0.0, 0.0
    min_gap, min_ttc, collided, impact = math.inf, math.inf, False, 0.0
    first_brake = None
    for k in range(int(EPISODE_S / DT)):
        t = k * DT
        pose = np.eye(4)
        pose[0, 3] = ego_x
        clusters = measure(actors, ego_x, rng)
        tracks = tracker.update(clusters, pose, DT, np.array([v, 0.0]))
        cmd = float(np.clip(policy.decide(tracks, clusters, pose, v, DT), 0.0, MAX_DECEL))
        a_act += (cmd - a_act) * DT / LAG
        if a_act > 0.5 and first_brake is None:
            first_brake = t
        peak_decel = max(peak_decel, a_act)
        v = max(v - a_act * DT, 0.0)
        ego_x += v * DT
        for a in actors:
            a.step(t, ego_x, DT)
            # ground truth
            overlap_y = abs(a.y) < EGO_HALF_W + a.width / 2
            gap = a.x - a.length / 2 - ego_x
            if overlap_y and a.x + a.length / 2 > ego_x - EGO_LEN:
                if gap > 0:
                    min_gap = min(min_gap, gap)
                    closing = v - a.vx
                    if closing > 0.1:
                        min_ttc = min(min_ttc, gap / closing)
                elif not collided and a.x - a.length / 2 <= ego_x:
                    collided, impact = True, v
        if collided or (v < 0.05 and t > 3.0):
            break
    return dict(collision=collided, impact_speed=impact, min_gap=min_gap, min_ttc=min_ttc,
                peak_decel=peak_decel, speed_loss=v_start - v, first_brake=first_brake)


def evaluate(n, seed, policies=None):
    results = {}
    for pname, factory in (policies or POLICIES).items():
        results[pname] = {}
        for sc in SCENARIOS:
            rows = []
            for e in range(n):
                rng = np.random.default_rng((seed, e, list(SCENARIOS).index(sc)))      # same episodes for every policy
                rows.append(run_episode(factory(), sc, rng))
            results[pname][sc] = rows
    return results


def summarize(results):
    table = {}
    for pname, per in results.items():
        h = [r for sc in HAZARD for r in per[sc]]
        b = [r for sc in BENIGN for r in per[sc]]
        table[pname] = {
            'collision_rate': float(np.mean([r['collision'] for r in h])),
            'impact_speed_mean': float(np.mean([r['impact_speed'] for r in h if r['collision']] or [0.0])),
            'false_brake_rate': float(np.mean([r['peak_decel'] > 2.0 for r in b])),
            'hard_false_brake_rate': float(np.mean([r['peak_decel'] > 5.0 for r in b])),
            'speed_loss_benign': float(np.mean([r['speed_loss'] for r in b])),
            'peak_decel_hazard': float(np.mean([r['peak_decel'] for r in h])),
            'per_scenario_collision': {sc: float(np.mean([r['collision'] for r in per[sc]])) for sc in HAZARD},
            'per_scenario_false_brake': {sc: float(np.mean([r['peak_decel'] > 2.0 for r in per[sc]])) for sc in BENIGN},
        }
    return table


def print_table(table):
    hdr = f"{'policy':<18}{'collide%':>9}{'impact m/s':>11}{'false-brk%':>11}{'hard FB%':>9}{'dv benign':>10}{'peak dec':>9}"
    print(hdr)
    for p, t in table.items():
        print(f"{p:<18}{100 * t['collision_rate']:9.1f}{t['impact_speed_mean']:11.1f}"
              f"{100 * t['false_brake_rate']:11.1f}{100 * t['hard_false_brake_rate']:9.1f}"
              f"{t['speed_loss_benign']:10.2f}{t['peak_decel_hazard']:9.2f}")
    print("\ncollision rate per hazard scenario (%)")
    print(f"{'policy':<18}" + ''.join(f"{s:>14}" for s in HAZARD))
    for p, t in table.items():
        print(f"{p:<18}" + ''.join(f"{100 * t['per_scenario_collision'][s]:14.0f}" for s in HAZARD))
    print("\nfalse-brake rate per benign scenario (%)")
    print(f"{'policy':<18}" + ''.join(f"{s:>14}" for s in BENIGN))
    for p, t in table.items():
        print(f"{p:<18}" + ''.join(f"{100 * t['per_scenario_false_brake'][s]:14.0f}" for s in BENIGN))


def plot(table, path):
    names = list(table)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(names))
    ax[0].bar(x, [100 * table[n]['collision_rate'] for n in names], color='tab:red')
    ax[0].set_title('collision rate, hazard scenarios (%)')
    ax[1].bar(x - 0.2, [100 * table[n]['false_brake_rate'] for n in names], 0.4, label='> 2 m/s^2')
    ax[1].bar(x + 0.2, [100 * table[n]['hard_false_brake_rate'] for n in names], 0.4, label='> 5 m/s^2')
    ax[1].set_title('false-brake rate, benign scenarios (%)'); ax[1].legend()
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(names, rotation=30, ha='right')
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--n', type=int, default=100, help='episodes per scenario')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--plots-dir', type=Path, default=config.ROOT / 'docs')
    a = ap.parse_args()
    table = summarize(evaluate(a.n, a.seed))
    print_table(table)
    a.plots_dir.mkdir(parents=True, exist_ok=True)
    (a.plots_dir / 'safety_sim.json').write_text(json.dumps(table, indent=1))
    plot(table, a.plots_dir / 'safety_sim.png')
    print(f"\nwrote {a.plots_dir / 'safety_sim.png'}")


if __name__ == '__main__':
    main()
