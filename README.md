Autonomous Perception & Safety Pipeline (KITTI)
- An end-to-end perception system for autonomous navigation, featuring sensor fusion, object tracking, and behavior planning.

🚀 Key Features
- Sensor Fusion: Projects 64-layer Velodyne LiDAR point clouds onto 2D camera frames for 3D ground-truth validation.

- Lane Detection: Implements HLS color filtering and a sliding-window polynomial fit to define the drivable path.

- Object Tracking: Uses YOLOv8 with BoT-SORT to maintain persistent IDs across frames, enabling temporal awareness.

- Safety Planning: A Behavior State Machine calculates Time-to-Collision (TTC) to trigger emergency braking and velocity control.

🛠 Tech Stack
- Languages: Python 3.x

- Libraries: OpenCV (CV2), NumPy, Ultralytics (YOLOv8), Matplotlib

- Data: KITTI Vision Benchmark Suite (City Dataset)

📈 Methodology
- Perception: Obstacles are detected via YOLO; lane boundaries are warped to Bird's Eye View for geometric analysis.

- Fusion: LiDAR points are filtered for "Ego-Vehicle" noise and ground surface before being projected into image space.

- Planning: Detections are fed into a Finite State Machine (FSM) that dictates vehicle commands based on proximity and TTC.