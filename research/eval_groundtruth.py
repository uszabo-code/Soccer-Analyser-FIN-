"""
Ground-truth-based evaluation for ball and target player detection.

Simplified format — only annotate the ball and ONE target player (your son).
No need to label all 22 players.

Ground truth format (research/ground_truth.json):
{
  "clip_info": {
    "fps": 25.0,
    "annotated_every_n_frames": 5
  },
  "frames": {
    "0": {
      "ball": {
        "bbox": [x1, y1, x2, y2],  // omit or null if ball not visible
        "in_play": true             // false = dead ball (throw-in, goal kick, etc.)
      },
      "target_player": {
        "bbox": [x1, y1, x2, y2]   // omit if player off-screen
      }
    },
    "5": {
      "ball": { "bbox": null, "in_play": false },
      "target_player": { "bbox": [310, 240, 370, 360] }
    }
  }
}

Notes:
- Annotate every 5th frame (so frame 0, 5, 10, 15, ...)
- bbox format: [x1, y1, x2, y2] in pixels (top-left, bottom-right)
- For ball: set in_play=false during dead-ball stoppages — these frames are
  excluded from the ball recall denominator so they don't penalise the score
- For target_player: omit the entry entirely when they're off-screen

Usage:
    python research/eval_groundtruth.py output/eval_baseline/detections.json research/ground_truth.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def compute_iou(box_a: list, box_b: list) -> float:
    """Standard IoU between two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 1e-6 else 0.0


def load_detections(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_ground_truth(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Metric 1: Ball recall (in-play frames only)
# ---------------------------------------------------------------------------

def ball_recall_inplay(
    detections: dict,
    ground_truth: dict,
    iou_threshold: float = 0.25,
) -> Tuple[float, dict]:
    """
    What fraction of in-play frames where the ball is annotated does the
    pipeline detect the ball?

    Uses iou_threshold=0.25 (generous — ball is small in overhead footage).
    Counts both real and Kalman-interpolated detections.
    """
    gt_frames = ground_truth.get("frames", {})

    # Index pipeline ball detections by frame number
    ball_by_frame: Dict[int, List[dict]] = defaultdict(list)
    for det in detections.get("ball_detections", []):
        ball_by_frame[det["frame_num"]].append(det)

    tp, fn = 0, 0
    missing_frames = []  # frames where ball was annotated but not detected

    for frame_str, frame_gt in gt_frames.items():
        frame_num = int(frame_str)
        ball_gt = frame_gt.get("ball")

        if not ball_gt:
            continue  # frame not annotated for ball
        if not ball_gt.get("in_play", True):
            continue  # dead ball — skip from denominator
        if not ball_gt.get("bbox"):
            continue  # ball not visible this frame

        gt_bbox = ball_gt["bbox"]

        # Check nearby frames too (±2 raw frames) to handle annotation/detection offset
        matched = False
        best_iou = 0.0
        for offset in range(-4, 5):
            for det in ball_by_frame.get(frame_num + offset, []):
                iou = compute_iou(gt_bbox, det["bbox"])
                best_iou = max(best_iou, iou)
                if iou >= iou_threshold:
                    matched = True
                    break
            if matched:
                break

        if matched:
            tp += 1
        else:
            fn += 1
            missing_frames.append({
                "frame": frame_num,
                "best_iou": round(best_iou, 3),
                "pipeline_ball_dets_nearby": sum(
                    len(ball_by_frame.get(frame_num + o, []))
                    for o in range(-4, 5)
                ),
            })

    total = tp + fn
    recall = tp / total if total > 0 else 0.0

    return recall, {
        "recall": round(recall, 4),
        "true_positives": tp,
        "false_negatives": fn,
        "total_inplay_annotated_frames": total,
        "iou_threshold": iou_threshold,
        "missed_frames": missing_frames,
    }


# ---------------------------------------------------------------------------
# Metric 2: Target player detection recall
# ---------------------------------------------------------------------------

def target_player_recall(
    detections: dict,
    ground_truth: dict,
    iou_threshold: float = 0.30,
) -> Tuple[float, dict]:
    """
    What fraction of annotated frames is the target player detected by the pipeline?

    Checks a ±4 raw frame window around each annotated frame to handle
    annotation/detection timing offsets.
    """
    gt_frames = ground_truth.get("frames", {})

    person_by_frame: Dict[int, List[dict]] = defaultdict(list)
    for det in detections.get("person_detections", []):
        person_by_frame[det["frame_num"]].append(det)

    tp, fn = 0, 0
    missed = []

    for frame_str, frame_gt in gt_frames.items():
        frame_num = int(frame_str)
        player_gt = frame_gt.get("target_player")

        if not player_gt or not player_gt.get("bbox"):
            continue  # not annotated this frame

        gt_bbox = player_gt["bbox"]
        matched = False
        best_iou = 0.0

        for offset in range(-4, 5):
            for det in person_by_frame.get(frame_num + offset, []):
                iou = compute_iou(gt_bbox, det["bbox"])
                best_iou = max(best_iou, iou)
                if iou >= iou_threshold:
                    matched = True
                    break
            if matched:
                break

        if matched:
            tp += 1
        else:
            fn += 1
            missed.append({"frame": frame_num, "best_iou": round(best_iou, 3)})

    total = tp + fn
    recall = tp / total if total > 0 else 0.0

    return recall, {
        "recall": round(recall, 4),
        "true_positives": tp,
        "false_negatives": fn,
        "total_annotated_frames": total,
        "iou_threshold": iou_threshold,
        "missed_frames": missed,
    }


# ---------------------------------------------------------------------------
# Metric 3: Target player tracking continuity
# ---------------------------------------------------------------------------

def target_player_continuity(
    detections: dict,
    ground_truth: dict,
    iou_threshold: float = 0.30,
) -> Tuple[float, dict]:
    """
    Of the frames where the target player IS detected, what fraction are
    covered by a single dominant track ID?

    A score of 1.0 means the player has exactly one track across the whole clip.
    A score of 0.5 means two equally-sized tracks (fragmented).

    Target: ≥ 0.70
    """
    gt_frames = ground_truth.get("frames", {})

    person_by_frame: Dict[int, List[dict]] = defaultdict(list)
    for det in detections.get("person_detections", []):
        person_by_frame[det["frame_num"]].append(det)

    track_hits: Dict[int, int] = defaultdict(int)  # track_id → matched frame count
    total_matched = 0

    for frame_str, frame_gt in gt_frames.items():
        frame_num = int(frame_str)
        player_gt = frame_gt.get("target_player")
        if not player_gt or not player_gt.get("bbox"):
            continue

        gt_bbox = player_gt["bbox"]
        for offset in range(-4, 5):
            for det in person_by_frame.get(frame_num + offset, []):
                if compute_iou(gt_bbox, det["bbox"]) >= iou_threshold:
                    tid = det.get("track_id", -1)
                    if tid >= 0:
                        track_hits[tid] += 1
                    total_matched += 1
                    break

    if not track_hits or total_matched == 0:
        return 0.0, {"error": "no matched detections for target player"}

    dominant_tid = max(track_hits, key=track_hits.get)
    dominant_count = track_hits[dominant_tid]
    continuity = dominant_count / total_matched

    return continuity, {
        "continuity": round(continuity, 4),
        "dominant_track_id": dominant_tid,
        "dominant_track_frames": dominant_count,
        "total_matched_frames": total_matched,
        "all_track_fragments": dict(sorted(track_hits.items(), key=lambda x: -x[1])),
        "n_fragments": len(track_hits),
    }


# ---------------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------------

def full_report(detections: dict, ground_truth: dict) -> Tuple[dict, dict]:
    ball_recall, ball_detail = ball_recall_inplay(detections, ground_truth)
    player_recall, player_detail = target_player_recall(detections, ground_truth)
    continuity, cont_detail = target_player_continuity(detections, ground_truth)

    summary = {
        "ball_recall_inplay":    round(ball_recall, 4),
        "ball_target":           0.80,
        "ball_pass":             ball_recall >= 0.80,
        "player_recall":         round(player_recall, 4),
        "player_target":         0.85,
        "player_pass":           player_recall >= 0.85,
        "player_continuity":     round(continuity, 4),
        "continuity_target":     0.70,
        "continuity_pass":       continuity >= 0.70,
    }
    detail = {
        "ball_detection":       ball_detail,
        "player_recall":        player_detail,
        "player_continuity":    cont_detail,
    }
    return summary, detail


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ground-truth evaluation — ball + target player only"
    )
    parser.add_argument("detections_json", help="Path to detections.json")
    parser.add_argument("ground_truth_json", help="Path to ground_truth.json")
    parser.add_argument("--iou-ball",    type=float, default=0.25)
    parser.add_argument("--iou-player",  type=float, default=0.30)
    args = parser.parse_args()

    dets = load_detections(args.detections_json)
    gt   = load_ground_truth(args.ground_truth_json)

    ball_recall, ball_detail     = ball_recall_inplay(dets, gt, args.iou_ball)
    player_recall, player_detail = target_player_recall(dets, gt, args.iou_player)
    continuity, cont_detail      = target_player_continuity(dets, gt, args.iou_player)

    W = 52
    print(f"\n{'='*W}")
    print(f"  GROUND-TRUTH EVALUATION")
    print(f"{'='*W}")

    def row(label, value, target, passed):
        mark = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {label:<28} {value:>6.1%}  {mark} (target ≥{target:.0%})")

    print()
    row("Ball recall (in-play)",   ball_recall,   0.80, ball_recall   >= 0.80)
    print(f"    TP={ball_detail['true_positives']}  FN={ball_detail['false_negatives']}  "
          f"annotated frames={ball_detail['total_inplay_annotated_frames']}")

    print()
    row("Target player recall",    player_recall, 0.85, player_recall >= 0.85)
    print(f"    TP={player_detail['true_positives']}  FN={player_detail['false_negatives']}  "
          f"annotated frames={player_detail['total_annotated_frames']}")

    print()
    row("Target player continuity", continuity,   0.70, continuity    >= 0.70)
    n = cont_detail.get("n_fragments", "?")
    dtid = cont_detail.get("dominant_track_id", "?")
    print(f"    dominant track #{dtid}  total fragments={n}")

    if ball_detail.get("missed_frames"):
        print(f"\n  Missed ball frames (first 5):")
        for m in ball_detail["missed_frames"][:5]:
            print(f"    frame {m['frame']:5d}  best_iou={m['best_iou']:.3f}  "
                  f"nearby_dets={m['pipeline_ball_dets_nearby']}")

    if player_detail.get("missed_frames"):
        print(f"\n  Missed player frames (first 5):")
        for m in player_detail["missed_frames"][:5]:
            print(f"    frame {m['frame']:5d}  best_iou={m['best_iou']:.3f}")

    all_pass = ball_recall >= 0.80 and player_recall >= 0.85 and continuity >= 0.70
    print(f"\n{'='*W}")
    print(f"  {'✓ ALL TARGETS MET' if all_pass else '✗ NOT THERE YET — keep improving!'}")
    print(f"{'='*W}\n")

    sys.exit(0 if all_pass else 1)
