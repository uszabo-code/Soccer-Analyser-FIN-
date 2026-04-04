"""
Evaluation harness for soccer analyzer experiments.

Provides metric functions that score pipeline output quality.
Each metric returns a float (higher = better) and a dict of details.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Tuple


def load_detections(output_dir: str) -> dict:
    path = Path(output_dir) / "detections.json"
    with open(path) as f:
        return json.load(f)


def load_stats(output_dir: str) -> dict:
    path = Path(output_dir) / "player_stats.json"
    with open(path) as f:
        return json.load(f)


def load_inplay_windows(csv_path: str) -> list:
    """
    Load in-play frame windows from a CSV file.

    CSV format (no header required, or with header start_frame,end_frame):
        0,750
        900,1800
        ...

    Returns list of (start_frame, end_frame) tuples.
    """
    import csv
    windows = []
    with open(csv_path) as f:
        reader_csv = csv.reader(f)
        for row in reader_csv:
            if not row or row[0].strip().lower() in ("start_frame", "#"):
                continue  # skip header / comments
            try:
                windows.append((int(row[0].strip()), int(row[1].strip())))
            except (IndexError, ValueError):
                pass
    return windows


def _is_inplay(frame_num: int, inplay_windows) -> bool:
    """Return True if frame_num falls within any in-play window (or if no windows defined)."""
    if not inplay_windows:
        return True
    return any(s <= frame_num <= e for s, e in inplay_windows)


# ---------------------------------------------------------------------------
# Metric 1: Ball detection quality
# ---------------------------------------------------------------------------

def ball_detection_score(detections: dict,
                          inplay_windows: list = None) -> Tuple[float, dict]:
    """
    Score ball detection quality.

    Components:
    - detection_rate: ball detections per processed frame (0 to 1+)
    - temporal_coverage: fraction of 1-second windows with at least one ball detection
    - consistency: fraction of ball detections that have a neighbor within 5 frames

    Args:
        detections: loaded detections.json dict
        inplay_windows: optional list of (start_frame, end_frame) tuples.
            When provided, only frames within these windows count in the denominator.
            This prevents dead-ball periods from inflating the total frame count and
            suppressing the apparent ball detection rate.

    Final score = weighted combination, 0-100 scale.
    """
    fps = detections["fps"]
    frame_skip = detections["frame_skip"]
    ball_dets = detections["ball_detections"]
    person_dets = detections["person_detections"]

    if not person_dets:
        return 0.0, {"error": "no person detections"}

    # Approximate total processed frames from person detections
    # When inplay_windows provided, filter to only in-play frames
    all_frames_raw = set(d["frame_num"] for d in person_dets)
    all_frames_raw.update(d["frame_num"] for d in ball_dets)

    if inplay_windows:
        all_frames = {f for f in all_frames_raw if _is_inplay(f, inplay_windows)}
        ball_dets_filtered = [d for d in ball_dets if _is_inplay(d["frame_num"], inplay_windows)]
        inplay_note = f"filtered to {len(all_frames)}/{len(all_frames_raw)} in-play frames"
    else:
        all_frames = all_frames_raw
        ball_dets_filtered = ball_dets
        inplay_note = "no in-play filter (all frames counted)"

    total_processed = len(all_frames) if all_frames else 1

    # Detection rate
    detection_rate = len(ball_dets_filtered) / total_processed
    # Clamp contribution: rate of 0.5+ is perfect
    rate_score = min(detection_rate / 0.5, 1.0)

    # Temporal coverage: divide video into 1-second windows
    if all_frames:
        min_frame = min(all_frames)
        max_frame = max(all_frames)
    else:
        min_frame, max_frame = 0, 1
    window_frames = int(fps)  # 1 second
    if window_frames < 1:
        window_frames = 30

    ball_frames = set(d["frame_num"] for d in ball_dets_filtered)
    n_windows = max(1, (max_frame - min_frame) // window_frames)
    windows_with_ball = 0
    for w in range(n_windows):
        w_start = min_frame + w * window_frames
        w_end = w_start + window_frames
        if any(w_start <= f < w_end for f in ball_frames):
            windows_with_ball += 1
    coverage = windows_with_ball / n_windows

    # Consistency: for each ball detection, is there another within 5 frames?
    sorted_ball_frames = sorted(ball_frames)
    consistent = 0
    for i, f in enumerate(sorted_ball_frames):
        has_neighbor = False
        if i > 0 and f - sorted_ball_frames[i - 1] <= 5 * frame_skip:
            has_neighbor = True
        if i < len(sorted_ball_frames) - 1 and sorted_ball_frames[i + 1] - f <= 5 * frame_skip:
            has_neighbor = True
        if has_neighbor:
            consistent += 1
    consistency = consistent / max(len(sorted_ball_frames), 1)

    score = (rate_score * 40 + coverage * 35 + consistency * 25)

    return score, {
        "ball_detections": len(ball_dets_filtered),
        "ball_detections_total": len(ball_dets),
        "total_processed_frames": total_processed,
        "inplay_note": inplay_note,
        "detection_rate": round(detection_rate, 4),
        "temporal_coverage": round(coverage, 4),
        "consistency": round(consistency, 4),
        "rate_score_contrib": round(rate_score * 40, 2),
        "coverage_contrib": round(coverage * 35, 2),
        "consistency_contrib": round(consistency * 25, 2),
    }


# ---------------------------------------------------------------------------
# Metric 2: Speed realism
# ---------------------------------------------------------------------------

def speed_realism_score(detections: dict, max_realistic_speed: float = 10.0,
                        field_width_m: float = 90.0) -> Tuple[float, dict]:
    """
    Score how realistic the computed speeds are.

    Runs a lightweight speed computation on raw detections and checks
    what fraction falls within [0, max_realistic_speed] m/s.

    Score 0-100.
    """
    fps = detections["fps"]
    frame_width = detections["width"]
    px_per_meter = frame_width / field_width_m

    # Group person detections by track
    tracks: dict[int, list] = {}
    for d in detections["person_detections"]:
        tid = d["track_id"]
        if tid >= 0:
            tracks.setdefault(tid, []).append(d)

    all_speeds = []
    for tid, dets in tracks.items():
        if len(dets) < 5:
            continue
        dets.sort(key=lambda d: d["frame_num"])
        for i in range(1, len(dets)):
            bbox_prev = dets[i - 1]["bbox"]
            bbox_curr = dets[i]["bbox"]
            cx_prev = (bbox_prev[0] + bbox_prev[2]) / 2
            cy_prev = (bbox_prev[1] + bbox_prev[3]) / 2
            cx_curr = (bbox_curr[0] + bbox_curr[2]) / 2
            cy_curr = (bbox_curr[1] + bbox_curr[3]) / 2
            dx = cx_curr - cx_prev
            dy = cy_curr - cy_prev
            dist_px = math.sqrt(dx * dx + dy * dy)
            dt = (dets[i]["frame_num"] - dets[i - 1]["frame_num"]) / fps
            if dt > 0:
                speed = (dist_px / px_per_meter) / dt
                all_speeds.append(speed)

    if not all_speeds:
        return 0.0, {"error": "no speeds computed"}

    realistic = sum(1 for s in all_speeds if 0 <= s <= max_realistic_speed)
    pct_realistic = realistic / len(all_speeds)
    max_speed = max(all_speeds)
    p99_speed = sorted(all_speeds)[int(len(all_speeds) * 0.99)]

    # Score: % realistic is primary, penalize extreme outliers
    outlier_penalty = min(max_speed / 100, 1.0) * 20  # up to -20 for extreme max
    score = max(0, pct_realistic * 100 - outlier_penalty)

    return score, {
        "total_speed_samples": len(all_speeds),
        "pct_realistic": round(pct_realistic, 4),
        "max_speed_mps": round(max_speed, 2),
        "p99_speed_mps": round(p99_speed, 2),
        "mean_speed_mps": round(sum(all_speeds) / len(all_speeds), 2),
        "outlier_penalty": round(outlier_penalty, 2),
    }


# ---------------------------------------------------------------------------
# Metric 3: Tracking smoothness
# ---------------------------------------------------------------------------

def _smooth_centers(centers: list, window: int) -> list:
    """Apply moving average smoothing to center positions."""
    if window <= 1 or len(centers) < window:
        return centers
    # Ensure odd window
    if window % 2 == 0:
        window += 1
    half = window // 2
    smoothed = []
    for i in range(len(centers)):
        start = max(0, i - half)
        end = min(len(centers), i + half + 1)
        avg_x = sum(c[0] for c in centers[start:end]) / (end - start)
        avg_y = sum(c[1] for c in centers[start:end]) / (end - start)
        smoothed.append((avg_x, avg_y))
    return smoothed


def tracking_smoothness_score(detections: dict, smoothing_window: int = 1,
                              min_movement_px: int = 5,
                              direction_change_angle: int = 90) -> Tuple[float, dict]:
    """
    Score tracking quality based on jitter and direction change plausibility.

    Parameters can be tuned by the research loop:
    - smoothing_window: moving average window for position smoothing (1 = no smoothing)
    - min_movement_px: minimum pixel displacement to count as real motion
    - direction_change_angle: threshold for "sharp" direction change

    Score 0-100.
    """
    fps = detections["fps"]

    tracks: dict[int, list] = {}
    for d in detections["person_detections"]:
        tid = d["track_id"]
        if tid >= 0:
            tracks.setdefault(tid, []).append(d)

    if not tracks:
        return 0.0, {"error": "no tracks"}

    total_moves = 0
    smooth_moves = 0
    direction_changes = 0
    sharp_changes = 0
    track_lengths = []

    for tid, dets in tracks.items():
        if len(dets) < 10:
            continue
        dets.sort(key=lambda d: d["frame_num"])
        track_lengths.append(len(dets))

        raw_centers = []
        for d in dets:
            b = d["bbox"]
            raw_centers.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))

        centers = _smooth_centers(raw_centers, smoothing_window)

        for i in range(1, len(centers)):
            dx = centers[i][0] - centers[i - 1][0]
            dy = centers[i][1] - centers[i - 1][1]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < min_movement_px:
                continue  # Skip sub-threshold jitter

            total_moves += 1

            # "Smooth" = movement < 50 pixels between consecutive processed frames
            if dist < 50:
                smooth_moves += 1

        for i in range(2, len(centers)):
            dx1 = centers[i - 1][0] - centers[i - 2][0]
            dy1 = centers[i - 1][1] - centers[i - 2][1]
            dx2 = centers[i][0] - centers[i - 1][0]
            dy2 = centers[i][1] - centers[i - 1][1]
            len1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
            len2 = math.sqrt(dx2 * dx2 + dy2 * dy2)
            if len1 > min_movement_px and len2 > min_movement_px:
                dot = dx1 * dx2 + dy1 * dy2
                cos_a = max(-1, min(1, dot / (len1 * len2)))
                angle = math.degrees(math.acos(cos_a))
                direction_changes += 1
                if angle > direction_change_angle:
                    sharp_changes += 1

    smooth_frac = smooth_moves / max(total_moves, 1)
    plausible_frac = 1 - (sharp_changes / max(direction_changes, 1))
    avg_track_len = sum(track_lengths) / max(len(track_lengths), 1)
    # Normalize: track length of 200+ is great
    track_len_score = min(avg_track_len / 200, 1.0)

    score = smooth_frac * 40 + plausible_frac * 40 + track_len_score * 20

    return score, {
        "total_moves": total_moves,
        "smooth_fraction": round(smooth_frac, 4),
        "direction_changes": direction_changes,
        "sharp_changes": sharp_changes,
        "plausible_fraction": round(plausible_frac, 4),
        "avg_track_length": round(avg_track_len, 1),
        "num_significant_tracks": len(track_lengths),
    }


# ---------------------------------------------------------------------------
# Metric 4: Track fragmentation
# ---------------------------------------------------------------------------

def fragmentation_score(detections: dict, expected_players: int = 22,
                         min_det_threshold: int = 10) -> Tuple[float, dict]:
    """
    Score track fragmentation quality.

    A perfect score (100) means the number of significant tracks equals the number
    of expected players — i.e., each player has exactly one track.

    Score = min((expected_players / significant_tracks) * 100, 100)

    Lower fragmentation → higher score.

    Args:
        detections: loaded detections.json dict
        expected_players: total players expected on field (default 22 + referees ≈ 24)
        min_det_threshold: minimum detections for a track to be "significant"

    Returns:
        (score 0-100, detail dict)
    """
    from collections import defaultdict
    tracks: dict = defaultdict(int)
    for d in detections.get("person_detections", []):
        tid = d.get("track_id", -1)
        if tid >= 0:
            tracks[tid] += 1

    significant = [tid for tid, cnt in tracks.items() if cnt >= min_det_threshold]
    n_significant = len(significant)

    if n_significant == 0:
        return 0.0, {"error": "no significant tracks found"}

    fragments_per_player = n_significant / expected_players
    # Score: 1.0 fragments/player = 100, 10.0 = 10, etc.
    score = min((expected_players / n_significant) * 100, 100.0)

    return score, {
        "significant_tracks": n_significant,
        "expected_players": expected_players,
        "fragments_per_player": round(fragments_per_player, 2),
        "total_tracks": len(tracks),
        "min_det_threshold": min_det_threshold,
    }


# ---------------------------------------------------------------------------
# Combined score
# ---------------------------------------------------------------------------

def combined_score(detections: dict, field_width_m: float = 90.0,
                   inplay_windows: list = None) -> Tuple[float, dict]:
    """
    Run all metrics and return a weighted combined score.

    Args:
        detections: loaded detections.json dict
        field_width_m: assumed field width in metres for speed calibration
        inplay_windows: optional list of (start_frame, end_frame) tuples for
            filtering the ball detection denominator to in-play frames only
    """
    ball_score, ball_detail = ball_detection_score(detections, inplay_windows=inplay_windows)
    speed_score, speed_detail = speed_realism_score(detections, field_width_m=field_width_m)
    track_score, track_detail = tracking_smoothness_score(detections)
    frag_score, frag_detail = fragmentation_score(detections)

    combined = ball_score * 0.4 + speed_score * 0.3 + track_score * 0.3

    return combined, {
        "combined": round(combined, 2),
        "ball_detection": {"score": round(ball_score, 2), **ball_detail},
        "speed_realism": {"score": round(speed_score, 2), **speed_detail},
        "tracking_smoothness": {"score": round(track_score, 2), **track_detail},
        "fragmentation": {"score": round(frag_score, 2), **frag_detail},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Evaluate soccer analyzer output")
    parser.add_argument("output_dir", help="Path to output directory with detections.json")
    parser.add_argument("--field-width", type=float, default=90.0,
                        help="Field width in metres for speed calibration (default: 90)")
    parser.add_argument("--inplay", metavar="CSV",
                        help="Path to CSV with start_frame,end_frame in-play windows. "
                             "When provided, ball detection score only counts in-play frames.")
    args = parser.parse_args()

    try:
        dets = load_detections(args.output_dir)
    except FileNotFoundError:
        print(f"No detections.json found in {args.output_dir}", file=sys.stderr)
        sys.exit(1)

    inplay = None
    if args.inplay:
        inplay = load_inplay_windows(args.inplay)
        print(f"Loaded {len(inplay)} in-play windows from {args.inplay}")

    score, details = combined_score(dets, field_width_m=args.field_width,
                                    inplay_windows=inplay)

    print(f"\n{'='*50}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  Combined Score: {score:.1f}/100\n")

    for metric_name, metric_data in details.items():
        if isinstance(metric_data, dict):
            s = metric_data.pop("score", "N/A")
            print(f"  [{metric_name}] Score: {s}")
            for k, v in metric_data.items():
                print(f"    {k}: {v}")
            print()
        else:
            print(f"  {metric_name}: {metric_data}")
