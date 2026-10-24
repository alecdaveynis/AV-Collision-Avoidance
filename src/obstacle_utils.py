import cv2

import config


def estimate_distance(cls, pixel_height, focal_length):
    """Monocular depth from box height and an assumed real-world height."""
    real_h = config.CLASS_HEIGHT_M.get(int(cls), config.DEFAULT_HEIGHT_M)
    return (focal_length * real_h) / pixel_height if pixel_height > 0 else 0.0


def extract_tracks(yolo_results, focal_length):
    """Convert YOLO/BoT-SORT output into a list of track dicts.

    Each track has: id, class, box (x1,y1,x2,y2), mono_dist. Fusion and planning
    stages add lidar_dist, dist, dist_source and in_path.
    """
    tracks = []
    boxes_obj = yolo_results.boxes
    if boxes_obj is None or boxes_obj.id is None:
        return tracks

    boxes = boxes_obj.xyxy.cpu().numpy().astype(int)
    ids = boxes_obj.id.cpu().numpy().astype(int)
    clss = boxes_obj.cls.cpu().numpy().astype(int)
    confs = boxes_obj.conf.cpu().numpy()

    for box, track_id, cls, conf in zip(boxes, ids, clss, confs):
        if cls not in config.TRACKED_CLASSES or conf <= config.MIN_CONF:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        tracks.append({
            'id': int(track_id),
            'class': int(cls),
            'box': (x1, y1, x2, y2),
            'mono_dist': estimate_distance(cls, y2 - y1, focal_length),
        })
    return tracks


def draw_tracked_obstacles(image, tracks, model_names):
    """Draw boxes. Red/yellow = on the ego path (close/far), green = off path."""
    for t in tracks:
        x1, y1, x2, y2 = t['box']
        dist = t['dist']
        if not t.get('in_path', True):
            color = (0, 200, 0)
        else:
            color = (0, 0, 255) if dist < 7 else (0, 255, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        tag = 'L' if t.get('dist_source') == 'lidar' else 'M'
        label = f"ID:{t['id']} {model_names[t['class']]} {dist:.1f}m {tag}"
        if t.get('ttc_learned') is not None and t['ttc_learned'] < config.TTC_MAX - 0.05:
            label += f" ttc{t['ttc_learned']:.1f}s"
        cv2.putText(image, label, (x1, max(y1 - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image
