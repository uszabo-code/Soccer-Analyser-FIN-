"""
Test Hough circle detection at different param2 values + radius filtering.
Goal: find settings that reduce candidates from 560/frame while keeping recall >80%.
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

CLIP_PATH = os.path.join(os.path.dirname(__file__), "clips", "eval_clip.mp4")
GT_PATH = os.path.join(os.path.dirname(__file__), "ground_truth.json")
MATCH_RADIUS_PX = 40

# HSV for yellow-green ball
BALL_HSV_LOW = np.array([55, 70, 140])
BALL_HSV_HIGH = np.array([90, 170, 255])


def load_ground_truth():
    with open(GT_PATH) as f:
        gt = json.load(f)
    gt_frames = {}
    for fk, fd in gt["frames"].items():
        b = fd.get("ball")
        if b and b.get("in_play"):
            bbox = b["bbox"]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            gt_frames[int(fk)] = (cx, cy)
    return gt_frames


def hough_detect(frame, y_min, y_max, param2=10, min_r=7, max_r=28,
                 min_dist=20, param1=50):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BALL_HSV_LOW, BALL_HSV_HIGH)
    mask[:y_min, :] = 0
    mask[y_max:, :] = 0
    blurred = cv2.GaussianBlur(mask, (5, 5), 1.5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.0, minDist=min_dist,
        param1=param1, param2=param2,
        minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        return []
    return [(float(cx), float(cy), float(r)) for cx, cy, r in circles[0]]


def is_near_gt(candidates, gt_cx, gt_cy, radius=MATCH_RADIUS_PX):
    for cx, cy, _ in candidates:
        if math.sqrt((cx - gt_cx)**2 + (cy - gt_cy)**2) <= radius:
            return True
    return False


def test_config(frames_data, y_min, y_max, label, **kwargs):
    hits = 0
    total_cands = 0
    n = len(frames_data)
    for frame, gt_cx, gt_cy in frames_data:
        cands = hough_detect(frame, y_min, y_max, **kwargs)
        total_cands += len(cands)
        if is_near_gt(cands, gt_cx, gt_cy):
            hits += 1
    recall = 100.0 * hits / n
    avg = total_cands / n
    print(f"  {label:50s}  recall={recall:5.1f}%  avg={avg:6.1f}  hits={hits}/{n}")
    return recall, avg


def main():
    gt_frames = load_ground_truth()
    cap = cv2.VideoCapture(CLIP_PATH)
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    y_min = int(h_frame * config.OVERLAY_TOP_FRACTION)
    y_max = int(h_frame * (1.0 - config.OVERLAY_BOTTOM_FRACTION))

    frames_data = []
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % config.FRAME_SKIP == 0 and frame_num in gt_frames:
            frames_data.append((frame.copy(), gt_frames[frame_num][0], gt_frames[frame_num][1]))
        frame_num += 1
    cap.release()
    print(f"Loaded {len(frames_data)} GT frames\n")

    configs = [
        # param2 sweep (main control for circle count)
        ("param2=10 (current)", dict(param2=10)),
        ("param2=12", dict(param2=12)),
        ("param2=15", dict(param2=15)),
        ("param2=18", dict(param2=18)),
        ("param2=20", dict(param2=20)),
        ("param2=25", dict(param2=25)),
        ("param2=30", dict(param2=30)),

        # Radius filtering (ball is r=10-14)
        ("param2=10, r=8-20", dict(param2=10, min_r=8, max_r=20)),
        ("param2=10, r=8-18", dict(param2=10, min_r=8, max_r=18)),
        ("param2=15, r=8-20", dict(param2=15, min_r=8, max_r=20)),
        ("param2=15, r=7-22", dict(param2=15, min_r=7, max_r=22)),
        ("param2=12, r=8-20", dict(param2=12, min_r=8, max_r=20)),

        # Higher param1 (edge sensitivity)
        ("param2=10, param1=70", dict(param2=10, param1=70)),
        ("param2=10, param1=100", dict(param2=10, param1=100)),
        ("param2=15, param1=70", dict(param2=15, param1=70)),

        # Larger minDist (prevents overlapping circles)
        ("param2=10, minDist=30", dict(param2=10, min_dist=30)),
        ("param2=10, minDist=40", dict(param2=10, min_dist=40)),
        ("param2=15, minDist=30", dict(param2=15, min_dist=30)),

        # Best combo candidates
        ("param2=15, r=7-22, minDist=30", dict(param2=15, min_r=7, max_r=22, min_dist=30)),
        ("param2=12, r=8-20, minDist=30", dict(param2=12, min_r=8, max_r=20, min_dist=30)),
        ("param2=12, r=8-20, param1=70", dict(param2=12, min_r=8, max_r=20, param1=70)),
    ]

    results = []
    for label, kwargs in configs:
        r, c = test_config(frames_data, y_min, y_max, label, **kwargs)
        results.append((label, r, c))

    print()
    print("=" * 80)
    print("SWEET SPOT: configs with recall ≥70% AND ≤100 cands/frame")
    print()
    sweet = [(l, r, c) for l, r, c in results if r >= 70 and c <= 100]
    sweet.sort(key=lambda x: (x[2], -x[1]))  # lowest cands first, then highest recall
    for l, r, c in sweet:
        print(f"  ★ {r:5.1f}% recall, {c:5.1f} cands/frame  {l}")

    if not sweet:
        print("  (none found — showing best tradeoffs)")
        # Show configs closest to the sweet spot
        results.sort(key=lambda x: (-x[1], x[2]))
        for l, r, c in results[:10]:
            print(f"    {r:5.1f}% recall, {c:5.1f} cands/frame  {l}")


if __name__ == "__main__":
    main()
