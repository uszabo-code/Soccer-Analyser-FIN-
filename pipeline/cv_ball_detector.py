"""
CV-based ball detector for indoor futsal.

Uses HSV color masking (targeting hi-vis yellow-green ball) + Hough circle
transform to find ball candidates, then applies Kalman-gated tracking to select
the most likely true ball from hundreds of candidate circles per frame.

Design:
  - Hough with param2=10 gives ~75% theoretical recall but 500+ circles/frame
  - A KalmanGate tracks ball trajectory and picks the closest Hough candidate
    each frame, reducing 500+ circles → 1 selected detection per frame
  - Static detections (same position for 5+ consecutive processed frames)
    are dropped as field-artifact false positives
"""
from __future__ import annotations

import math
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np

# HSV range for hi-vis yellow-green futsal ball
# Turf is same hue but darker (V ~80-130); ball is brighter (V ~145+)
BALL_HSV_LOW  = np.array([55,  70, 140])
BALL_HSV_HIGH = np.array([90, 170, 255])

# Hough parameters — param2=10 kept low to maximise recall;
# false positives culled by KalmanGate rather than Hough thresholding
HOUGH_DP        = 1.0
HOUGH_MIN_DIST  = 20
HOUGH_PARAM1    = 50
HOUGH_PARAM2    = 10
HOUGH_MIN_R     = 7
HOUGH_MAX_R     = 28

# Gate radius: maximum px the ball can travel between consecutive processed
# frames.  Set generously to handle fast kicks (≈35 m/s → ~90 px/frame).
GATE_RADIUS_PX = 110

# Static filter: drop detection clusters where centre barely moves
STATIC_MIN_MOVE_PX    = 15   # px — below this counts as "not moved"
STATIC_MIN_COUNT      = 5    # processed-frames before cluster is flagged static
STATIC_MAX_GAP_FRAMES = 12   # raw-frame gap that still counts as same cluster


def detect_hough_candidates(
    frame: np.ndarray,
    y_min: int,
    y_max: int,
) -> List[Tuple[float, float, float]]:
    """Return (cx, cy, r) for every Hough circle in the yellow-green mask."""
    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask  = cv2.inRange(hsv, BALL_HSV_LOW, BALL_HSV_HIGH)
    mask[:y_min, :]  = 0
    mask[y_max:,  :] = 0

    blurred = cv2.GaussianBlur(mask, (5, 5), 1.5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP, minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_R, maxRadius=HOUGH_MAX_R,
    )
    if circles is None:
        return []
    return [(float(cx), float(cy), float(r)) for cx, cy, r in circles[0]]


class CVBallGate:
    """
    Kalman-gated ball selector.

    Maintains a position estimate (cx, cy).  On each processed frame:
      1. Caller seeds with a YOLO detection (high confidence).
      2. Or caller passes Hough candidates → gate picks the closest within
         GATE_RADIUS_PX of the current estimate.

    State is reset when no detection occurs for max_lost_frames.
    """

    def __init__(self, gate_radius: float = GATE_RADIUS_PX, max_lost: int = 15):
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None
        self._lost: int           = max_lost  # start lost so first seed initialises
        self._max_lost            = max_lost
        self.gate_radius          = gate_radius

    def seed(self, cx: float, cy: float) -> None:
        """Force-update the estimate from a high-confidence source (YOLO)."""
        self.cx     = cx
        self.cy     = cy
        self._lost  = 0

    def select(
        self,
        candidates: List[Tuple[float, float, float]],
    ) -> Optional[Tuple[float, float, float]]:
        """
        Pick the candidate closest to current estimate within gate_radius.
        Returns (cx, cy, r) or None.  Updates internal state.
        """
        if not candidates:
            self._lost += 1
            if self._lost >= self._max_lost:
                self.cx = self.cy = None
            return None

        if self.cx is None:
            # No estimate yet — cannot gate; caller must seed first
            self._lost += 1
            return None

        best: Optional[Tuple[float, float, float]] = None
        best_dist = self.gate_radius

        for cx, cy, r in candidates:
            dist = math.sqrt((cx - self.cx) ** 2 + (cy - self.cy) ** 2)
            if dist < best_dist:
                best_dist = dist
                best = (cx, cy, r)

        if best is not None:
            self.cx, self.cy = best[0], best[1]
            self._lost = 0
        else:
            self._lost += 1
            if self._lost >= self._max_lost:
                self.cx = self.cy = None

        return best

    @property
    def has_estimate(self) -> bool:
        return self.cx is not None


def find_nearest_to_feet(
    circles: List[Tuple[float, float, float]],
    person_bboxes: List[Tuple[float, float, float, float]],
    max_dist: float = 80.0,
) -> Optional[Tuple[float, float, float]]:
    """
    Find the Hough circle closest to any player's feet (bottom-center of bbox).
    Returns (cx, cy, r) or None.
    """
    if not circles or not person_bboxes:
        return None

    # Compute foot positions: bottom-center of each person bbox
    feet = []
    for x1, y1, x2, y2 in person_bboxes:
        feet.append(((x1 + x2) / 2, y2))  # bottom center

    best_circle = None
    best_dist = max_dist

    for cx, cy, r in circles:
        for fx, fy in feet:
            dist = math.sqrt((cx - fx) ** 2 + (cy - fy) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_circle = (cx, cy, r)

    return best_circle


def filter_static_ball_detections(
    detections: List[Dict],
    min_move_px: float = STATIC_MIN_MOVE_PX,
    min_static_count: int = STATIC_MIN_COUNT,
    max_gap: int = STATIC_MAX_GAP_FRAMES,
) -> List[Dict]:
    """
    Drop consecutive-run clusters where the ball centre barely moves.
    These are field markings / reflective spots on the pitch, not the ball.

    Only operates on non-interpolated detections; interpolated ones are
    re-injected unchanged.
    """
    real  = sorted([d for d in detections if not d.get("interpolated")],
                   key=lambda d: d["frame_num"])
    interp = [d for d in detections if d.get("interpolated")]

    n    = len(real)
    keep = [True] * n

    i = 0
    while i < n:
        x1, y1, x2, y2 = real[i]["bbox"]
        cx0, cy0 = (x1 + x2) / 2, (y1 + y2) / 2
        group_end = i

        for j in range(i + 1, n):
            # break on large raw-frame gap
            if real[j]["frame_num"] - real[j - 1]["frame_num"] > max_gap:
                break
            bx1, by1, bx2, by2 = real[j]["bbox"]
            cxj, cyj = (bx1 + bx2) / 2, (by1 + by2) / 2
            if math.sqrt((cxj - cx0) ** 2 + (cyj - cy0) ** 2) > min_move_px:
                break
            group_end = j

        if (group_end - i + 1) >= min_static_count:
            for k in range(i, group_end + 1):
                keep[k] = False
            i = group_end + 1
        else:
            i += 1

    filtered = [d for d, k in zip(real, keep) if k]
    combined = filtered + interp
    combined.sort(key=lambda d: d["frame_num"])
    return combined
