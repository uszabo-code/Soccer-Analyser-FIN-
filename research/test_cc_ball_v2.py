"""
Test connected-component ball detection with multiple parameter sets
to find the sweet spot between recall and candidate count.

Key insight from v1: CC_AREA_MIN=100 and morphological opening killed
the ball blob in most frames (11.5% recall). Need to relax filters.
"""
from __future__ import annotations

import json
import math
import sys
import os

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from pipeline.cv_ball_detector import (
    BALL_HSV_LOW, BALL_HSV_HIGH,
    detect_hough_candidates,
)

CLIP_PATH = os.path.join(os.path.dirname(__file__), "clips", "eval_clip.mp4")
GT_PATH = os.path.join(os.path.dirname(__file__), "ground_truth.json")
MATCH_RADIUS_PX = 40


def load_ground_truth():
    with open(GT_PATH) as f:
        gt = json.load(f)
    gt_frames = {}
    for frame_key, frame_data in gt["frames"].items():
        ball = frame_data.get("ball")
        if ball and ball.get("in_play", False):
            bbox = ball["bbox"]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            gt_frames[int(frame_key)] = (cx, cy)
    return gt_frames


def detect_cc(frame, y_min, y_max, area_min=30, area_max=3000,
              aspect_min=0.3, aspect_max=3.5, use_open=False, use_close=True,
              circ_min=0.0):
    """CC detection with configurable filters."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BALL_HSV_LOW, BALL_HSV_HIGH)
    mask[:y_min, :] = 0
    mask[y_max:, :] = 0

    if use_open:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if use_close:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    candidates = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < area_min or area > area_max:
            continue
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if h == 0 or w == 0:
            continue
        aspect = w / h
        if aspect < aspect_min or aspect > aspect_max:
            continue

        cx, cy = centroids[i]
        equiv_r = math.sqrt(area / math.pi)
        candidates.append((float(cx), float(cy), float(equiv_r)))

    return candidates


def is_near_gt(candidates, gt_cx, gt_cy, radius=MATCH_RADIUS_PX):
    for cx, cy, _ in candidates:
        if math.sqrt((cx - gt_cx) ** 2 + (cy - gt_cy) ** 2) <= radius:
            return True
    return False


def test_params(frames_data, y_min, y_max, label, **kwargs):
    """Test a parameter set and return (recall%, avg_candidates)."""
    hits = 0
    total_cands = 0
    n = len(frames_data)

    for frame, gt_cx, gt_cy in frames_data:
        cands = detect_cc(frame, y_min, y_max, **kwargs)
        total_cands += len(cands)
        if is_near_gt(cands, gt_cx, gt_cy):
            hits += 1

    recall = 100.0 * hits / n if n > 0 else 0
    avg_cands = total_cands / n if n > 0 else 0
    print(f"  {label:40s}  recall={recall:5.1f}%  avg_cands={avg_cands:6.1f}  hits={hits}/{n}")
    return recall, avg_cands


def main():
    gt_frames = load_ground_truth()
    print(f"Ground truth: {len(gt_frames)} in-play frames with ball annotations")

    cap = cv2.VideoCapture(CLIP_PATH)
    if not cap.isOpened():
        print(f"ERROR: cannot open {CLIP_PATH}")
        sys.exit(1)

    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    y_min = int(h_frame * config.OVERLAY_TOP_FRACTION)
    y_max = int(h_frame * (1.0 - config.OVERLAY_BOTTOM_FRACTION))
    frame_skip = config.FRAME_SKIP

    # Pre-load all GT frames
    frames_data = []
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % frame_skip == 0 and frame_num in gt_frames:
            gt_cx, gt_cy = gt_frames[frame_num]
            frames_data.append((frame.copy(), gt_cx, gt_cy))
        frame_num += 1
    cap.release()

    print(f"Loaded {len(frames_data)} GT-annotated frames")
    print()

    # Also get Hough baseline
    hough_hits = 0
    hough_total = 0
    for frame, gt_cx, gt_cy in frames_data:
        cands = detect_hough_candidates(frame, y_min, y_max)
        hough_total += len(cands)
        if is_near_gt(cands, gt_cx, gt_cy):
            hough_hits += 1
    n = len(frames_data)
    print(f"  {'Hough baseline':40s}  recall={100*hough_hits/n:5.1f}%  avg_cands={hough_total/n:6.1f}")
    print()

    # Test parameter combinations
    print("CC parameter sweep:")
    print("=" * 80)

    configs = [
        # (label, kwargs)
        ("v1: area>=100, open+close, circ>=0.4",
         dict(area_min=100, use_open=True, use_close=True, circ_min=0.4)),

        ("no-open, area>=100, close",
         dict(area_min=100, use_open=False, use_close=True)),

        ("no-open, area>=50, close",
         dict(area_min=50, use_open=False, use_close=True)),

        ("no-open, area>=30, close",
         dict(area_min=30, use_open=False, use_close=True)),

        ("no-open, area>=15, close",
         dict(area_min=15, use_open=False, use_close=True)),

        ("no-morph, area>=30",
         dict(area_min=30, use_open=False, use_close=False)),

        ("no-morph, area>=15",
         dict(area_min=15, use_open=False, use_close=False)),

        ("no-morph, area>=15, wide-aspect",
         dict(area_min=15, use_open=False, use_close=False,
              aspect_min=0.2, aspect_max=5.0)),

        ("close-only, area>=15, wide-aspect",
         dict(area_min=15, use_open=False, use_close=True,
              aspect_min=0.2, aspect_max=5.0)),

        ("close-only, area>=30, area<=1500",
         dict(area_min=30, area_max=1500, use_open=False, use_close=True)),

        ("close-only, area>=15, area<=2000",
         dict(area_min=15, area_max=2000, use_open=False, use_close=True)),
    ]

    results = []
    for label, kwargs in configs:
        r, c = test_params(frames_data, y_min, y_max, label, **kwargs)
        results.append((label, r, c))

    print()
    print("=" * 80)
    print("RANKED by recall (filtering configs with <20% recall):")
    print()
    for label, recall, cands in sorted(results, key=lambda x: -x[1]):
        ratio = f"{cands:.0f}" if cands > 0 else "N/A"
        quality = "★" if recall >= 50 and cands <= 50 else ""
        print(f"  {recall:5.1f}% recall, {cands:5.1f} cands/frame  {quality}  {label}")

    # Diagnostic: for the best high-recall config, show per-frame details for misses
    print()
    print("=" * 80)
    best_cfg = dict(area_min=15, use_open=False, use_close=True,
                    aspect_min=0.2, aspect_max=5.0)
    print("Diagnostic: per-frame analysis for 'close-only, area>=15, wide-aspect'")
    print("Showing first 10 misses with ball pixel info:")
    miss_count = 0
    for frame, gt_cx, gt_cy in frames_data:
        cands = detect_cc(frame, y_min, y_max, **best_cfg)
        if not is_near_gt(cands, gt_cx, gt_cy):
            # Look at what's happening at the GT position
            gx, gy = int(gt_cx), int(gt_cy)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Sample 5x5 patch around GT
            patch = hsv[max(0,gy-2):gy+3, max(0,gx-2):gx+3]
            h_vals = patch[:,:,0].flatten()
            s_vals = patch[:,:,1].flatten()
            v_vals = patch[:,:,2].flatten()

            # Check mask
            mask = cv2.inRange(hsv, BALL_HSV_LOW, BALL_HSV_HIGH)
            mask_val = mask[gy, gx] if 0 <= gy < mask.shape[0] and 0 <= gx < mask.shape[1] else -1

            # Check mask region around GT
            region = mask[max(0,gy-15):gy+16, max(0,gx-15):gx+16]
            mask_pixels = np.count_nonzero(region)

            print(f"  MISS gt=({gx},{gy})  mask@gt={mask_val}  "
                  f"mask_pixels_31x31={mask_pixels}  "
                  f"H={h_vals.min()}-{h_vals.max()} "
                  f"S={s_vals.min()}-{s_vals.max()} "
                  f"V={v_vals.min()}-{v_vals.max()}  "
                  f"n_cands={len(cands)}")

            miss_count += 1
            if miss_count >= 15:
                break


if __name__ == "__main__":
    main()
