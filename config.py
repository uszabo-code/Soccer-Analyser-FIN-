"""All configurable parameters for the soccer analyzer pipeline."""

# Detection
# football_yolov8.pt: ball=0, goalkeeper=1, player=2, referee=3 (soccer-specific, 14x more ball dets)
# yolov8m.pt:         person=0, sports_ball=32 (COCO generic — fallback)
YOLO_MODEL = "football_yolov8.pt"
DEVICE = "auto"  # "auto", "mps", "cpu", or "cuda"
FRAME_SKIP = 2  # Every 2nd frame: 47% fewer tracks, 14% less noise vs FRAME_SKIP=3 at only 1.5x compute
CONFIDENCE_THRESHOLD = 0.15  # YOLO model conf — set to ball threshold (class filtering below)
PERSON_CONFIDENCE_THRESHOLD = 0.3  # Post-detection filter for persons
BALL_CONFIDENCE_THRESHOLD = 0.15  # Post-detection filter for ball (small object)

# Class IDs — must match YOLO_MODEL's class mapping
# football_yolov8.pt: BALL_CLASS_ID=0, PERSON_CLASS_IDS=[1,2,3]
# yolov8m.pt (COCO):  BALL_CLASS_ID=32, PERSON_CLASS_IDS=[0]
BALL_CLASS_ID = 0
PERSON_CLASS_IDS = [1, 2, 3]  # goalkeeper, player, referee — all tracked as "persons"
PERSON_CLASS_ID = 2            # Primary player class (for backwards compat)

# Overlay filtering — skip detections in these frame regions (stream overlays)
OVERLAY_TOP_FRACTION = 0.08  # Top 8% of frame
OVERLAY_BOTTOM_FRACTION = 0.05  # Bottom 5% of frame

# Jersey OCR
OCR_SAMPLES_PER_TRACK = 20
OCR_MIN_VOTES = 3
JERSEY_CROP_PADDING = 0.15  # Padding around bbox for OCR crop
MIN_PLAYER_HEIGHT_PX = 120  # Skip OCR on small/distant players (jersey unreadable below this)

# Tracking
# BoT-SORT adds GMC (camera motion compensation) + ReID appearance features over ByteTrack
# This reduces player #15 from 22 fragmented tracks → target ≤5 per game
TRACKER_CONFIG = "botsort_soccer.yaml"

# Field calibration (U13-U15 default)
# Tuned via autoresearch: camera shows ~60m of field width, not the full pitch
FIELD_WIDTH_METERS = 60.0
SPRINT_SPEED_THRESHOLD = 4.0  # m/s

# Post-processing (tuned via autoresearch)
SPEED_OUTLIER_CAP = 12.0  # m/s — cap unrealistic speeds
SMOOTHING_WINDOW = 5  # Moving average window for position smoothing (odd number)
MIN_MOVEMENT_PX = 10  # Minimum pixel displacement to count as real motion
DIRECTION_CHANGE_ANGLE = 120  # Minimum angle for "sharp" direction change (degrees)

# Claude API
CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_MAX_TOKENS = 4096
CLAUDE_TEMPERATURE = 0.3

# Output
OUTPUT_DIR = "output"
