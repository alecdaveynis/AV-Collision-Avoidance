import cv2
import numpy as np
import os, glob
from ultralytics import YOLO

# Utility modules for perception, fusion, and UI
import lane_utils as lu
import obstacle_utils as ou
import video_utils as vu
import fusion_utils as fs
import ui_utils as ui
from behavior_manager import BehaviorManager

# config
model = YOLO('yolov8s.pt')
INPUT_DIR = 'data/2011_09_26_drive_0005_sync/image_02/data'
LIDAR_DIR = 'data/2011_09_26_drive_0005_sync/velodyne_points/data'
OUTPUT_DIR = 'results'
FOCAL_LENGTH = 700  # Calibration constant for KITTI camera

planner = BehaviorManager()
if not os.path.exists(OUTPUT_DIR): 
    os.makedirs(OUTPUT_DIR)

# Retrieve and sort input frames for temporal consistency
frames = sorted(glob.glob(os.path.join(INPUT_DIR, '*.png')))

print(f"Processing {len(frames)} frames. Exporting to {OUTPUT_DIR}...")

for f_path in frames:
    img = cv2.imread(f_path)
    if img is None: continue
    
    # lane detection
    # Convert to HLS for robust white/yellow line filtering
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(hls, np.array([0, 200, 0]), np.array([180, 255, 255]))
    
    # Bird's Eye View transformation
    M, Minv = lu.get_perspective_matrices(img)
    warped = cv2.warpPerspective(mask, M, (img.shape[1], img.shape[0]))
    
    res_img = img.copy()
    if np.any(warped):
        l_idx, r_idx, nx, ny = lu.find_lane_pixels(warped)
        res_img = lu.draw_lane_overlay(img, warped, l_idx, r_idx, nx, ny, Minv)
    
    # object tracking
    # BoT-SORT maintains persistent IDs across frames
    yolo_res = model.track(img, persist=True, tracker="botsort.yaml", verbose=False)[0]
    res_img, tracks = ou.draw_tracked_obstacles(res_img, yolo_res, model.names, FOCAL_LENGTH)
    
    # lidar fusion & spatial filtering
    lidar_f = os.path.join(LIDAR_DIR, os.path.basename(f_path).replace('.png', '.bin'))
    l_dist_min = 99.0 
    
    if os.path.exists(lidar_f):
        # Parse binary point cloud: [x, y, z, r]
        scan = np.fromfile(lidar_f, dtype=np.float32).reshape(-1, 4)
        
        # Define Safety Tunnel: Ignore ego-vehicle, ground, and lateral noise
        env_mask = (scan[:, 0] > 1.5) & (scan[:, 2] > -1.3) & (np.abs(scan[:, 1]) < 0.8)
        f_scan = scan[env_mask]
        
        # Overlay 3D points on 2D image
        pts_2d, _ = fs.project_lidar_to_camera(f_scan, res_img.shape)
        for i in range(len(pts_2d)):
            cv2.circle(res_img, (int(pts_2d[i,0]), int(pts_2d[i,1])), 1, (255, 0, 0), -1)
        
        if len(f_scan) > 0:
            l_dist_min = np.min(f_scan[:, 0])

    # behavior decision
    # Translates distance and TTC into simple commands
    action_text = planner.update_state(tracks, l_dist_min)
    
    # visualization
    res_img = ui.draw_dashboard(res_img, tracks, l_dist_min)
    
    # Visual cues: Red for STOP, Green for DRIVE
    txt_color = (0, 0, 255) if planner.state == "STOPPED" else (0, 255, 0)
    
    # dashboard telemetry
    cv2.putText(res_img, f"STATUS: {planner.state}", (50, 60), 0, 0.9, txt_color, 2)
    cv2.putText(res_img, f"ACTION: {action_text}", (50, 100), 0, 0.9, (255, 255, 255), 2)
    cv2.putText(res_img, f"LIDAR: {l_dist_min:.1f}m", (50, 140), 0, 0.7, (255, 255, 0), 2)
    
    # Save processed frame to results directory
    output_path = os.path.join(OUTPUT_DIR, os.path.basename(f_path))
    cv2.imwrite(output_path, res_img)

# export video
vu.save_to_video(OUTPUT_DIR, 'adu_perception_final.mp4', 10)
print("Pipeline Complete. Video saved.")