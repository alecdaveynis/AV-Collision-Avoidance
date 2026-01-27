import cv2

def estimate_distance(cls, pixel_height, focal_length):
    # Monocular Depth Estimation
    # Person/Bike height approx 1.7m, Car approx 1.5m
    real_h = 1.7 if cls in [0, 1] else 1.5
    return (focal_length * real_h) / pixel_height if pixel_height > 0 else 0

def draw_tracked_obstacles(image, yolo_results, model_names, focal_length):
    tracked_objects = [] # List to store data for the Behavior Manager
    
    if yolo_results.boxes is not None and yolo_results.boxes.id is not None:
        boxes = yolo_results.boxes.xyxy.cpu().numpy().astype(int)
        ids = yolo_results.boxes.id.cpu().numpy().astype(int)
        clss = yolo_results.boxes.cls.cpu().numpy().astype(int)
        confs = yolo_results.boxes.conf.cpu().numpy()

        for box, track_id, cls, conf in zip(boxes, ids, clss, confs):
            x1, y1, x2, y2 = box
            
            if cls in [0, 1, 2, 3, 7] and conf > 0.4:
                # 1. Calculate distance
                pixel_height = y2 - y1
                dist_m = estimate_distance(cls, pixel_height, focal_length)
                
                # 2. Append to our data list for Phase 3
                tracked_objects.append({'id': track_id, 'class': cls, 'dist': dist_m})
                
                # 3. Drawing logic
                color = (0, 0, 255) if dist_m < 7 else (0, 255, 255)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                label = f"ID:{track_id} {model_names[cls]} {dist_m:.1f}m"
                cv2.putText(image, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
    return image, tracked_objects 