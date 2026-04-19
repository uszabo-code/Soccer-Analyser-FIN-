"""
Test contrast-based scoring of Hough circle candidates to reduce from ~560 to ~20.

Key insight: the ball is distinguishable from turf by LOCAL CONTRAST in the V channel.
Ball interior is V~160-186, surrounding turf is V~80-130 → contrast ~40-80.
False Hough circles on uniform turf have near-zero contrast.

For each circle, compute:
  contrast = mean_V_inside - mean_V_outside_annulus
Then rank/filter by contrast.
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


def score_circles_by_contrast(circles, frame_v, frame_s):
    """
    Score each Hough circle by local V-channel contrast.

    Returns list of (cx, cy, r, contrast, saturation, score) tuples.
    """
    h, w = frame_v.shape
    scored = []

    for cx, cy, r in circles:
        ix, iy = int(cx), int(cy)
        ir = max(int(r), 5)

        # Create masks for inside circle and outside annulus
        # Inside: circle of radius r
        # Outside: annulus from r to 2r
        y_lo = max(0, iy - 2*ir)
        y_hi = min(h, iy + 2*ir + 1)
        x_lo = max(0, ix - 2*ir)
        x_hi = min(w, ix + 2*ir + 1)

        if y_hi - y_lo < 3 or x_hi - x_lo < 3:
            continue

        # Local patch
        v_patch = frame_v[y_lo:y_hi, x_lo:x_hi].astype(float)
        s_patch = frame_s[y_lo:y_hi, x_lo:x_hi].astype(float)

        # Create coordinate grids relative to circle center
        yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        dist_sq = (xx - ix)**2 + (yy - iy)**2

        inside_mask = dist_sq <= ir**2
        annulus_mask = (dist_sq > ir**2) & (dist_sq <= (2*ir)**2)

        n_inside = inside_mask.sum()
        n_annulus = annulus_mask.sum()

        if n_inside < 5 or n_annulus < 5:
            continue

        v_inside = v_patch[inside_mask[y_lo-y_lo:y_hi-y_lo, x_lo-x_lo:x_hi-x_lo]].mean() if n_inside > 0 else 0
        v_outside = v_patch[annulus_mask[y_lo-y_lo:y_hi-y_lo, x_lo-x_lo:x_hi-x_lo]].mean() if n_annulus > 0 else 0

        # Recompute on the local patch correctly
        py, px = np.mgrid[0:v_patch.shape[0], 0:v_patch.shape[1]]
        center_py = iy - y_lo
        center_px = ix - x_lo
        d2 = (px - center_px)**2 + (py - center_py)**2

        in_mask = d2 <= ir**2
        ann_mask = (d2 > ir**2) & (d2 <= (2*ir)**2)

        if in_mask.sum() < 3 or ann_mask.sum() < 3:
            continue

        v_in = v_patch[in_mask].mean()
        v_out = v_patch[ann_mask].mean()
        s_in = s_patch[in_mask].mean()

        contrast = v_in - v_out  # positive = brighter inside

        # Combined score: high contrast + moderate saturation + brightness
        # Ball has V~160-186, S~111-143, contrast~40-80
        score = contrast  # Start simple: just contrast

        scored.append((cx, cy, r, float(contrast), float(s_in), float(v_in), score))

    return scored


def main():
    gt_frames = load_ground_truth()
    print(f"Ground truth: {len(gt_frames)} in-play frames with ball annotations")

    cap = cv2.VideoCapture(CLIP_PATH)
    if not cap.isOpened():
        print(f"ERROR: cannot open {CLIP_PATH}")
        sys.exit(1)

    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    y_min = int(h_frame * config.OVERLAY_TOP_FRACTION)
    y_max = int(h_frame * (1.0 - config.OVERLAY_BOTTOM_FRACTION))
    frame_skip = config.FRAME_SKIP

    # Pre-load GT frames
    frames_data = []
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % frame_skip == 0 and frame_num in gt_frames:
            gt_cx, gt_cy = gt_frames[frame_num]
            frames_data.append((frame.copy(), gt_cx, gt_cy, frame_num))
        frame_num += 1
    cap.release()
    print(f"Loaded {len(frames_data)} GT-annotated frames\n")

    # Analyze contrast distribution
    ball_contrasts = []
    ball_ranks = []

    # Test different top-K thresholds
    thresholds = [5, 10, 20, 30, 50]
    topk_hits = {k: 0 for k in thresholds}
    contrast_thresh_hits = {t: 0 for t in [10, 15, 20, 25, 30]}
    contrast_thresh_counts = {t: 0 for t in [10, 15, 20, 25, 30]}

    n = len(frames_data)

    print(f"{'Frame':>6} {'Circles':>7} {'BallContr':>9} {'BallV':>6} {'BallS':>6} "
          f"{'BallRank':>8} {'Top5':>5} {'Top10':>6} {'Top20':>6}")
    print("-" * 80)

    for frame, gt_cx, gt_cy, fnum in frames_data:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_v = hsv[:, :, 2]
        frame_s = hsv[:, :, 1]

        circles = detect_hough_candidates(frame, y_min, y_max)
        if not circles:
            print(f"{fnum:>6} {'0':>7} {'---':>9}")
            continue

        scored = score_circles_by_contrast(circles, frame_v, frame_s)
        if not scored:
            continue

        # Sort by contrast descending
        scored.sort(key=lambda x: -x[6])  # score = contrast

        # Find the circle nearest to GT
        best_dist = 999
        best_idx = -1
        best_entry = None
        for idx, (cx, cy, r, contrast, s_in, v_in, score) in enumerate(scored):
            dist = math.sqrt((cx - gt_cx)**2 + (cy - gt_cy)**2)
            if dist < best_dist and dist <= MATCH_RADIUS_PX:
                best_dist = dist
                best_idx = idx
                best_entry = (cx, cy, r, contrast, s_in, v_in, score)

        if best_entry is None:
            print(f"{fnum:>6} {len(scored):>7} {'NO_MATCH':>9}")
            continue

        cx, cy, r, contrast, s_in, v_in, score = best_entry
        rank = best_idx + 1  # 1-based rank
        ball_contrasts.append(contrast)
        ball_ranks.append(rank)

        in_top = {k: rank <= k for k in thresholds}
        for k in thresholds:
            if in_top[k]:
                topk_hits[k] += 1

        # Count circles above contrast thresholds
        for t in contrast_thresh_hits:
            above = [s for s in scored if s[3] >= t]
            if any(math.sqrt((s[0]-gt_cx)**2 + (s[1]-gt_cy)**2) <= MATCH_RADIUS_PX
                   for s in above):
                contrast_thresh_hits[t] += 1
            contrast_thresh_counts[t] += len(above)

        t5 = "✓" if in_top[5] else " "
        t10 = "✓" if in_top[10] else " "
        t20 = "✓" if in_top[20] else " "

        print(f"{fnum:>6} {len(scored):>7} {contrast:>9.1f} {v_in:>6.1f} {s_in:>6.1f} "
              f"{rank:>8} {t5:>5} {t10:>6} {t20:>6}")

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if ball_contrasts:
        arr = np.array(ball_contrasts)
        print(f"\nBall contrast stats:")
        print(f"  Min={arr.min():.1f}  Max={arr.max():.1f}  "
              f"Mean={arr.mean():.1f}  Median={np.median(arr):.1f}  "
              f"Std={arr.std():.1f}")

        arr_r = np.array(ball_ranks)
        print(f"\nBall rank (by contrast) stats:")
        print(f"  Min={arr_r.min()}  Max={arr_r.max()}  "
              f"Mean={arr_r.mean():.1f}  Median={np.median(arr_r):.1f}")

    print(f"\nTop-K recall (ball in top K by contrast):")
    for k in thresholds:
        recall = 100.0 * topk_hits[k] / n if n > 0 else 0
        print(f"  Top-{k:>2}: {recall:5.1f}% ({topk_hits[k]}/{n})")

    print(f"\nContrast threshold filtering:")
    for t in sorted(contrast_thresh_hits.keys()):
        recall = 100.0 * contrast_thresh_hits[t] / n if n > 0 else 0
        avg_cands = contrast_thresh_counts[t] / n if n > 0 else 0
        print(f"  contrast >= {t:>2}: recall={recall:5.1f}%  avg_cands={avg_cands:.1f}")


if __name__ == "__main__":
    main()
