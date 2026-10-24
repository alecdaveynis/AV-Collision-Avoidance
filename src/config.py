"""Central configuration: paths, calibration, and tunable thresholds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Data / output -----------------------------------------------------------
DATA_DIR = ROOT / 'data'    # every drive found below here is processed
# KITTI calibration files (calib_cam_to_cam.txt / calib_velo_to_cam.txt) are
# picked up automatically from here if present; otherwise the defaults in
# calibration.py are used (calibration is per recording day, so one set
# covers all 2011_09_26 drives).
CALIB_DIR = DATA_DIR
MODEL_PATH = ROOT / 'yolov8s.pt'
OUTPUT_DIR = ROOT / 'output'
TTC_MODEL_PATH = ROOT / 'models' / 'ttc_gru.pt'

# --- Timing ------------------------------------------------------------------
DEFAULT_DT = 0.1            # s, used if timestamps are unavailable (KITTI = 10 Hz)
VIDEO_FPS = 10
LIDAR_SYNC_TOLERANCE = 0.06  # s, max camera<->LiDAR time offset still used for fusion

# --- Lidar geometry (Velodyne frame: x fwd, y left, z up) ----------------------
LIDAR_HEIGHT = 1.73         # m, sensor above ground on the KITTI car
GROUND_Z = -LIDAR_HEIGHT
OBSTACLE_MIN_HEIGHT = 0.5   # m above the LOCAL ground estimate (see fusion_utils.obstacle_points)
GRID_MAX_X = 80.0           # m, range / half-width of the ground-estimation grid
GRID_MAX_Y = 30.0
OBSTACLE_MIN_X = 1.5        # drop ego-vehicle returns
FALLBACK_CORRIDOR_HALF_WIDTH = 1.2   # m, used when no lane is detected
CORRIDOR_X_RANGE = (2.0, 60.0)       # m, ahead of the sensor
USE_CURVED_CORRIDOR = True  # corridor follows the yaw rate (constant-curvature path)
CURVE_MIN_SPEED = 2.0       # m/s, below this the yaw rate / speed ratio is meaningless
CURVE_MAX = 0.05            # 1/m, clamp curvature (radius >= 20 m)
PATH_MAX_LATERAL = 2.0      # m, in-path LiDAR points must be within this of the centre line ...
PATH_LATERAL_RANGE = 30.0   # m, ... up to this range (beyond it only the lane/corridor mask decides)
LIDAR_MIN_CLUSTER = 5       # in-path returns needed within LIDAR_CLUSTER_BAND of depth ...
LIDAR_CLUSTER_BAND = 1.0    # m ... before an in-path obstacle distance is reported

# --- Fusion -------------------------------------------------------------------
BOX_SHRINK = 0.2            # use central (1 - 2*shrink) of a box when sampling lidar
BOX_MIN_POINTS = 3          # min lidar points inside a box to trust the fused distance
BOX_NEAR_PERCENTILE = 20    # nearest-surface estimate inside a box ...
BOX_NEAR_BAND = 1.5         # m ... keep points within this band of it, take the median

# --- Obstacles ----------------------------------------------------------------
# COCO ids: 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck
TRACKED_CLASSES = (0, 1, 2, 3, 5, 7)
CLASS_HEIGHT_M = {0: 1.7, 1: 1.7, 2: 1.5, 3: 1.5, 5: 3.0, 7: 3.0}
DEFAULT_HEIGHT_M = 1.5
MIN_CONF = 0.4

# --- Lane detection (metric BEV; see lane_utils.py) -------------------------------------
BEV_X_NEAR = 6.5            # m, closest ground the camera sees at the bottom of the image
BEV_X_FAR = 30.0            # m
BEV_HALF_WIDTH = 6.0        # m, lateral half extent of the bird's-eye view
BEV_SCALE = 25.0            # px per metre
LANE_M_WINDOWS = 8          # sliding windows along x
LANE_M_MARGIN = 0.4         # m, half width of a search window
LANE_M_MINPIX = 12          # pixels needed in a window to recentre / count it
LANE_M_BASE_RANGE = 8.0     # m, near range used to find each line's starting position
LANE_M_MIN_OFFSET = 0.6     # m, a lane line is at least this far from the vehicle centre line ...
LANE_M_MAX_OFFSET = 3.6     # m, ... and at most this far
LANE_M_MIN_POINTS = 80      # BEV pixels of support per line
LANE_M_MIN_SPAN = 6.0       # m of range the support must cover
LANE_M_MAX_C2 = 0.03        # |x^2 coefficient| limit (radius of curvature > ~17 m)
LANE_M_EXTRAPOLATE = 2.0    # m the lane polygon may extend beyond the last supported window
LANE_M_MIN_WIDTH = 2.4      # m, plausible lane width ...
LANE_M_MAX_WIDTH = 4.6      # m
LANE_M_MAX_WIDTH_RATIO = 1.25  # max/min width over the fitted stretch (parallel edges)
LANE_M_EGO_TOL = 0.3        # m, ego centre may sit this far outside a fitted line
LANE_M_MAX_CENTER_OFFSET = 1.5  # m, lane centre vs vehicle centre line at the near edge
LANE_M_MAX_JUMP = 0.5       # m, frame-to-frame lateral jump treated as an outlier
LANE_M_SINGLE_LINE = False  # infer the missing line from one visible line + lane width (hurts: see eval_lane.py)
LANE_M_PATH_TOL = 1.0       # m, lane centre must agree with the yaw-rate path to be trusted (gate)
LANE_M_GATE_RANGE = 8.0     # m of range (from the near edge) the gate checks
LANE_M_DEFAULT_WIDTH = 3.5  # m
LANE_SMOOTH_ALPHA = 0.25    # EMA weight of the newest fit
LANE_MAX_MISSING = 4        # frames to coast on the previous fit

# Legacy image-space detector (lane_legacy.py), kept as the evaluation baseline.
LANE_MIN_PIXELS = 150
LANE_MIN_WIDTH_FRAC = 0.15
LANE_MAX_WIDTH_FRAC = 0.65
LANE_EGO_TOLERANCE_FRAC = 0.02
LANE_MAX_OFFSET_FRAC = 0.3
LANE_MAX_WIDTH_RATIO = 1.6
LANE_MAX_JUMP_FRAC = 0.10

# --- Planner ------------------------------------------------------------------
STOP_BASE_DIST = 2.0        # m at standstill (original behaviour)
SLOW_BASE_DIST = 6.0        # m at standstill (original behaviour)
EMERGENCY_DECEL = 8.0       # m/s^2, stop zone grows with v^2 / (2a)
SLOW_HEADWAY = 1.0          # s, slow zone grows with v * headway
TTC_STOP = 1.0              # s
TTC_SLOW = 3.0              # s
MIN_CLOSING_SPEED = 0.5     # m/s, below this TTC is considered infinite
MAX_RANGE_RATE = 20.0       # m/s, faster range change = different object, reset filter
DIST_EMA_ALPHA = 0.5        # smoothing of per-track distance
ESCALATE_FRAMES = 2         # consecutive frames needed to move to a worse state
RELEASE_FRAMES = 5          # consecutive frames needed to move to a better state
TRACK_MAX_AGE = 5           # frames before a vanished track is forgotten

# --- Learned TTC ----------------------------------------------------------------
TTC_SEQ_LEN = 8             # frames of track history fed to the network (~0.8 s)
TTC_MIN = 0.3               # s, labels / predictions are clipped to [TTC_MIN, TTC_MAX]
TTC_MAX = 10.0
TTC_LABEL_WINDOW = 5        # frames of future (label) / past (baseline) range history
