import math

import cv2

from behavior_manager import BehaviorState


def draw_dashboard(image, tracks, decision):
    """Bird's-eye mini-map (top right) with the ego path and tracked obstacles."""
    h, w = image.shape[:2]
    map_w, map_h = 200, 300
    overlay = image.copy()
    cv2.rectangle(overlay, (w - map_w - 20, 20), (w - 20, map_h + 20), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

    center_x = w - (map_w // 2) - 20
    base_y = map_h + 10
    scale = 5  # 1 meter = 5 pixels

    # Ego path + planner zones
    cv2.line(image, (center_x - 20, base_y), (center_x - 20, 20), (0, 255, 0), 1)
    cv2.line(image, (center_x + 20, base_y), (center_x + 20, 20), (0, 255, 0), 1)
    for dist, color in ((decision.stop_dist, (0, 0, 255)), (decision.slow_dist, (0, 165, 255))):
        y = int(base_y - dist * scale)
        if 20 < y < map_h + 20:
            cv2.line(image, (center_x - 30, y), (center_x + 30, y), color, 1)

    for t in tracks:
        pos_y = int(base_y - t['dist'] * scale)
        if not (20 < pos_y < map_h + 20):
            continue
        if t.get('in_path', True):
            x = center_x
            color = (0, 0, 255) if t['dist'] < 7 else (0, 255, 255)
        else:
            bx = (t['box'][0] + t['box'][2]) / 2
            x = center_x + (-45 if bx < w / 2 else 45)
            color = (0, 200, 0)
        cv2.circle(image, (x, pos_y), 5, color, -1)
        cv2.putText(image, f"ID:{t['id']}", (x + 10, pos_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.rectangle(image, (center_x - 10, base_y - 20), (center_x + 10, base_y), (255, 255, 255), -1)
    return image


def draw_status(image, decision, path_source, lane_status):
    """Telemetry text block (top left)."""
    txt_color = (0, 0, 255) if decision.state == BehaviorState.STOP else (0, 255, 0)
    if decision.state == BehaviorState.SLOW:
        txt_color = (0, 165, 255)

    put = lambda text, y, scale, col: cv2.putText(
        image, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2)

    put(f"STATUS: {decision.state}", 60, 0.9, txt_color)
    put(f"ACTION: {decision.text}", 100, 0.9, (255, 255, 255))
    lid = f"{decision.min_dist:.1f}m" if decision.min_dist < 99 else "--"
    put(f"IN-PATH DIST: {lid}", 140, 0.7, (255, 255, 0))
    spd = f"{decision.ego_speed * 3.6:.0f} km/h" if decision.ego_speed is not None else "n/a"
    put(f"EGO: {spd}   PATH: {path_source} ({lane_status})", 175, 0.6, (255, 255, 255))
    fmt = lambda v: f"{v:.1f}s" if (v is not None and math.isfinite(v)) else "--"
    put(f"TTC analytic: {fmt(decision.ttc_analytic)}   learned: {fmt(decision.ttc_learned)}",
        235, 0.6, (255, 255, 255))
    if decision.lead_id is not None and decision.lead_speed is not None:
        put(f"LEAD ID:{decision.lead_id} {decision.lead_dist:.1f}m  "
            f"v={decision.lead_speed * 3.6:.0f} km/h", 205, 0.6, (255, 255, 255))
    return image
