import cv2

def draw_dashboard(image, tracked_objects, lidar_dist):
    h, w = image.shape[:2]
    # 1. Create a semi-transparent black overlay for the map
    map_w, map_h = 200, 300
    overlay = image.copy()
    cv2.rectangle(overlay, (w - map_w - 20, 20), (w - 20, map_h + 20), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

    # 2. Draw 'Safety Tunnel' on Map
    center_x = w - (map_w // 2) - 20
    base_y = map_h + 10
    scale = 5  # 1 meter = 5 pixels
    
    cv2.line(image, (center_x - 20, base_y), (center_x - 20, 20), (0, 255, 0), 1)
    cv2.line(image, (center_x + 20, base_y), (center_x + 20, 20), (0, 255, 0), 1)

    # 3. Draw Tracked Obstacles on Map
    for obs in tracked_objects:
        dist = obs['dist']
        pos_y = int(base_y - (dist * scale))
        if 20 < pos_y < map_h + 20:
            # Color by distance (Red if close)
            color = (0, 0, 255) if dist < 7 else (0, 255, 255)
            cv2.circle(image, (center_x, pos_y), 5, color, -1)
            cv2.putText(image, f"ID:{obs['id']}", (center_x + 10, pos_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 4. Draw Ego-Vehicle
    cv2.rectangle(image, (center_x - 10, base_y - 20), (center_x + 10, base_y), (255, 255, 255), -1)
    
    return image