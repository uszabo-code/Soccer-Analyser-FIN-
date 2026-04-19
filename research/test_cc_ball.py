"""
Test connected-component analysis as an alternative to raw Hough circles
for ball detection candidate generation.

Goal: reduce ball candidates from ~556/frame (Hough) to ~5-20/frame,
making multi-hypothesis tracking viable.
"""
from __future__ import annotations

import json
import math
import sys
import os

import cv2
import numpy as np

# Add project root to path so we can import config and detector
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from pipeline.cv_ball_detector import (
    BALL_HSV_LOW, BALL_HSV_HIGH,
    detect_hough_candidates,
)

# ---------- paths ----------
CLIP_PATH = os.path.join(os.path.dirname(__file__), "clips", "eval_clip.mp4")
GT_PATH = os.path.join(os.path.dirname(__file__), "ground_truth.json")

# ---------- CC filter params ----------
CC_HSV_LOW  = np.array([55,  70, 140])
CC_HSV_HIGH = np.array([90, 170, 255])

MORPH_OPEN_KERNEL  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
MORPH_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

CC_AREA_MIN = 100
CC_AREA_MAX = 3000
CC_ASPECT_MIN = 0.4
CC_ASPECT_MAX = 2.5
CC_CIRCULARITY_MIN = 0.4

MATCH_RADIUS_PX = 40


def load_ground_truth():
    """Load GT and return dict: frame_num -> (cx, cy) for in-play frames with ball."""
    with open(GT_PATH) as f:
        gt = json.load(f)

    gt_frames = {}
    for frame_key, frame_data in gt["frames"].items():
        ball = frame_data.get("ball")
        if ball and ball.get("in_play", False):
            bbox = ball["bbox"]  # [x1, y1, x2, y2]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            gt_frames[int(frame_key)] = (cx, cy)

    return gt_frames


def detect_cc_candidates(frame, y_min, y_max):
    """
    Connected-component ball candidate detection.
    Returns list of (cx, cy, equiv_radius) for each surviving blob.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, CC_HSV_LOW, CC_HSV_HIGH)
    mask[:y_min, :] = 0
    mask[y_max:, :] = 0

    # Morphological cleanup
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_OPEN_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_CLOSE_KERNEL)

    # Connected components with stats
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    candidates = []
    for i in range(1, num_labels):  # skip label 0 (background)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < CC_AREA_MIN or area > CC_AREA_MAX:
            continue

        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if h == 0 or w == 0:
            continue

        aspect = w / h
        if aspect < CC_ASPECT_MIN or aspect > CC_ASPECT_MAX:
            continue

        # Circularity: compare area to bounding-box area as a proxy
        # True circularity needs contour perimeter, but we can approximate:
        # A perfect circle fills pi/4 ~ 0.785 of its bounding box
        # We also compute circularity from the component mask directly
        component_mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            perimeter = cv2.arcLength(contours[0], True)
            if perimeter > 0:
                circularity = 4 * math.pi * area / (perimeter * perimeter)
                if circularity < CC_CIRCULARITY_MIN:
                    continue

        cx, cy = centroids[i]
        equiv_r = math.sqrt(area / math.pi)
        candidates.append((float(cx), float(cy), float(equiv_r)))

    return candidates


def is_near_gt(candidates, gt_cx, gt_cy, radius=MATCH_RADIUS_PX):
    """Check if any candidate centroid is within radius of GT position."""
    for cx, cy, _ in candidates:
        if math.sqrt((cx - gt_cx) ** 2 + (cy - gt_cy) ** 2) <= radius:
            return True
    return False


def main():
    gt_frames = load_ground_truth()
    print(f"Ground truth: {len(gt_frames)} in-play frames with ball annotations")

    cap = cv2.VideoCapture(CLIP_PATH)
    if not cap.isOpened():
        print(f"ERROR: cannot open {CLIP_PATH}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    y_min = int(h_frame * config.OVERLAY_TOP_FRACTION)
    y_max = int(h_frame * (1.0 - config.OVERLAY_BOTTOM_FRACTION))
    frame_skip = config.FRAME_SKIP

    print(f"Video: {total_frames} frames, {w_frame}x{h_frame}")
    print(f"Overlay crop: y_min={y_min}, y_max={y_max}")
    print(f"FRAME_SKIP={frame_skip}")
    print(f"CC params: area=[{CC_AREA_MIN},{CC_AREA_MAX}], "
          f"aspect=[{CC_ASPECT_MIN},{CC_ASPECT_MAX}], "
          f"circularity>{CC_CIRCULARITY_MIN}")
    print(f"Match radius: {MATCH_RADIUS_PX}px")
    print()
    print(f"{'Frame':>6}  {'Hough':>6}  {'CC':>4}  {'Hough_GT':>9}  {'CC_GT':>6}")
    print("-" * 42)

    # Accumulators
    total_hough = 0
    total_cc = 0
    hough_hits = 0
    cc_hits = 0
    evaluated = 0

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only process frames matching FRAME_SKIP cadence
        if frame_num % frame_skip != 0:
            frame_num += 1
            continue

        # Only evaluate on GT-annotated frames
        if frame_num not in gt_frames:
            frame_num += 1
            continue

        gt_cx, gt_cy = gt_frames[frame_num]

        # Hough approach
        hough_cands = detect_hough_candidates(frame, y_min, y_max)
        n_hough = len(hough_cands)
        hough_match = is_near_gt(hough_cands, gt_cx, gt_cy)

        # CC approach
        cc_cands = detect_cc_candidates(frame, y_min, y_max)
        n_cc = len(cc_cands)
        cc_match = is_near_gt(cc_cands, gt_cx, gt_cy)

        total_hough += n_hough
        total_cc += n_cc
        if hough_match:
            hough_hits += 1
        if cc_match:
            cc_hits += 1
        evaluated += 1

        h_tag = "HIT" if hough_match else "miss"
        c_tag = "HIT" if cc_match else "miss"
        print(f"{frame_num:>6}  {n_hough:>6}  {n_cc:>4}  {h_tag:>9}  {c_tag:>6}")

        frame_num += 1

    cap.release()

    print()
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    if evaluated == 0:
        print("No frames evaluated!")
        return

    avg_hough = total_hough / evaluated
    avg_cc = total_cc / evaluated
    hough_recall = 100.0 * hough_hits / evaluated
    cc_recall = 100.0 * cc_hits / evaluated

    print(f"Frames evaluated:          {evaluated}")
    print(f"Avg Hough candidates/frame: {avg_hough:.1f}")
    print(f"Avg CC candidates/frame:    {avg_cc:.1f}")
    print(f"Hough recall (GT within {MATCH_RADIUS_PX}px): {hough_recall:.1f}% ({hough_hits}/{evaluated})")
    print(f"CC recall    (GT within {MATCH_RADIUS_PX}px): {cc_recall:.1f}% ({cc_hits}/{evaluated})")
    print(f"Candidate reduction:        {avg_hough:.0f} -> {avg_cc:.0f} "
          f"({100*(1 - avg_cc/avg_hough):.0f}% fewer)" if avg_hough > 0 else "")


if __name__ == "__main__":
    main()
