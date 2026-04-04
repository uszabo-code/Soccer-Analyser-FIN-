"""
Kalman filter ball tracker with gap interpolation.

Fills short gaps (≤ MAX_INTERP_FRAMES processed frames) in ball detections
using a constant-velocity Kalman filter, reducing the effect of momentary
detection failures on the ball detection score and downstream ball-proximity
features.

State vector : [cx, cy, vx, vy]
Observation  : [cx, cy]

Usage (called automatically by pipeline/detect.py):
    from pipeline.ball_tracker import smooth_ball_detections
    ball_detections = smooth_ball_detections(ball_detections, fps=25.0, frame_skip=2)
"""
from __future__ import annotations

import math
from typing import List, Dict

import numpy as np

# Maximum gap (in processed frames) that will be filled by interpolation.
# At FRAME_SKIP=2 and 25fps, 10 processed frames ≈ 0.8 s.
# Gaps wider than this likely mean the ball is genuinely out of play.
MAX_INTERP_FRAMES = 10

# Physics sanity check: reject predictions that imply implausible ball speed.
# A hard-kicked ball travels ≤ 35 m/s. At 60m/1920px, 25fps ≈ 45 px/frame.
# Generous cap to cover varied frame-skip settings.
MAX_BALL_SPEED_PX_PER_FRAME = 60  # px per raw frame

# Kalman filter noise parameters (tuned for 25fps, pixel coordinates)
PROCESS_NOISE = 5.0    # Position process noise (Q)
MEAS_NOISE = 3.0       # Measurement noise (R)


class _BallKalmanFilter:
    """
    4-state Kalman filter: position (cx, cy) + velocity (vx, vy).

    State: x = [cx, cy, vx, vy]
    Obs:   z = H @ x = [cx, cy]
    """

    def __init__(self, cx: float, cy: float,
                 process_noise: float = PROCESS_NOISE,
                 meas_noise: float = MEAS_NOISE,
                 dt: float = 1.0):
        # State transition
        self.F = np.array([[1, 0, dt, 0],
                            [0, 1, 0, dt],
                            [0, 0, 1,  0],
                            [0, 0, 0,  1]], dtype=float)
        # Observation matrix
        self.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]], dtype=float)
        # Process noise covariance
        self.Q = np.eye(4) * process_noise
        # Measurement noise covariance
        self.R = np.eye(2) * meas_noise
        # Initial state covariance
        self.P = np.diag([100.0, 100.0, 10.0, 10.0])
        # Initial state
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=float)

    def predict(self):
        """Predict next state. Returns predicted (cx, cy)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0]), float(self.x[1])

    def update(self, cx: float, cy: float):
        """Update with observation (cx, cy). Returns corrected (cx, cy)."""
        z = np.array([cx, cy], dtype=float)
        y = z - self.H @ self.x                     # innovation
        S = self.H @ self.P @ self.H.T + self.R     # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)    # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return float(self.x[0]), float(self.x[1])


def smooth_ball_detections(
    ball_detections: List[Dict],
    fps: float = 25.0,
    frame_skip: int = 2,
    max_interp_frames: int = MAX_INTERP_FRAMES,
    max_speed_px_per_frame: float = MAX_BALL_SPEED_PX_PER_FRAME,
) -> List[Dict]:
    """
    Apply Kalman smoothing and gap interpolation to ball detections.

    Fills gaps of ≤ max_interp_frames processed frames with Kalman-predicted
    ball positions. Interpolated entries are flagged with 'interpolated': True
    so downstream code can optionally exclude them from precision calculations.

    Args:
        ball_detections: list of detection dicts from pipeline/detect.py
        fps: video frame rate (used for speed sanity checks)
        frame_skip: frames skipped between detections
        max_interp_frames: maximum gap (processed frames) to interpolate
        max_speed_px_per_frame: physics cap on ball speed (raw frames)

    Returns:
        Augmented detection list (original + interpolated entries), sorted by frame_num.
    """
    if not ball_detections:
        return ball_detections

    dets = sorted(ball_detections, key=lambda d: d["frame_num"])

    def center(det):
        b = det["bbox"]
        return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

    interpolated = []
    kf: _BallKalmanFilter | None = None
    prev_det = None

    for det in dets:
        cx, cy = center(det)

        if kf is None:
            kf = _BallKalmanFilter(cx, cy)
            kf.update(cx, cy)
            prev_det = det
            continue

        gap_frames = det["frame_num"] - prev_det["frame_num"]
        gap_processed = max(1, gap_frames // frame_skip)

        if gap_processed > max_interp_frames:
            # Gap too large — restart filter
            kf = _BallKalmanFilter(cx, cy)
            kf.update(cx, cy)
            prev_det = det
            continue

        # Generate synthetic detections across the gap
        prev_cx, prev_cy = center(prev_det)
        prev_conf = prev_det["confidence"]
        curr_conf = det["confidence"]

        # Average bbox dimensions for interpolated boxes
        pb = prev_det["bbox"]
        cb = det["bbox"]
        w = ((pb[2] - pb[0]) + (cb[2] - cb[0])) / 2
        h = ((pb[3] - pb[1]) + (cb[3] - cb[1])) / 2

        gap_stopped = False
        for step in range(1, gap_processed):
            pred_cx, pred_cy = kf.predict()

            # Physics sanity check
            dist = math.sqrt((pred_cx - prev_cx) ** 2 + (pred_cy - prev_cy) ** 2)
            if dist > max_speed_px_per_frame * frame_skip * step:
                gap_stopped = True
                break

            t = step / gap_processed
            interp_conf = round(prev_conf + t * (curr_conf - prev_conf), 4)
            interp_frame = prev_det["frame_num"] + step * frame_skip

            interpolated.append({
                "frame_num": interp_frame,
                "track_id": -1,
                "bbox": [pred_cx - w / 2, pred_cy - h / 2,
                          pred_cx + w / 2, pred_cy + h / 2],
                "confidence": interp_conf,
                "class_id": det["class_id"],
                "interpolated": True,
                "source": "kalman",
            })

        if gap_stopped:
            # Restart filter if prediction diverged
            kf = _BallKalmanFilter(cx, cy)

        kf.update(cx, cy)
        prev_det = det

    if interpolated:
        combined = ball_detections + interpolated
        combined.sort(key=lambda d: d["frame_num"])
        return combined
    return ball_detections
