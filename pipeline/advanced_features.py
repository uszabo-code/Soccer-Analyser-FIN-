"""Advanced feature extraction: off-ball movement, spacing, positional intelligence."""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict

import numpy as np

import config


def run(detections_path: str, identity_path: str, stats_path: str, output_dir: str) -> str:
    """
    Compute advanced metrics for the target player:
    - Off-ball movement quality (how active/purposeful when not near the ball)
    - Spacing relative to teammates (compactness, width coverage)
    - Positional discipline (how well they hold their zone)
    - Work rate distribution (active vs idle time)

    Returns path to advanced_stats.json
    """
    print("\n[Stage 3b] Advanced Feature Extraction")

    with open(detections_path) as f:
        det_data = json.load(f)
    with open(identity_path) as f:
        id_data = json.load(f)

    fps = det_data["fps"]
    W = det_data["width"]
    H = det_data["height"]
    px_per_m = W / config.FIELD_WIDTH_METERS

    target_tids = set(id_data["target_track_ids"])
    target_team = id_data.get("target_team", "unknown")

    # Get teammate track IDs
    teammate_tids = set()
    for p in id_data["players"]:
        if p.get("team") == target_team and not p.get("is_target"):
            teammate_tids.update(p["track_ids"])

    # Build per-frame position lookups
    target_by_frame = {}
    teammates_by_frame = defaultdict(list)
    all_by_frame = defaultdict(list)

    for d in det_data["person_detections"]:
        tid = d["track_id"]
        fn = d["frame_num"]
        bbox = d["bbox"]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        if tid in target_tids:
            target_by_frame[fn] = (cx, cy)
        if tid in teammate_tids:
            teammates_by_frame[fn].append((cx, cy))
        all_by_frame[fn].append((tid, cx, cy))

    if not target_by_frame:
        print("  No target player detections found.")
        return _save_empty(output_dir)

    target_frames = sorted(target_by_frame.keys())
    print(f"  Target visible in {len(target_frames)} frames")
    print(f"  Teammates tracked: {len(teammate_tids)} track IDs")

    # ===== 1. Movement Purposefulness =====
    # Measure: ratio of "net displacement" to "total distance" in sliding windows
    # High ratio = purposeful straight runs. Low ratio = wandering/oscillating.
    window_frames = int(3 * fps / det_data["frame_skip"])  # 3-second windows
    purposefulness_scores = []
    for i in range(0, len(target_frames) - window_frames, window_frames // 2):
        window = target_frames[i:i + window_frames]
        if len(window) < 5:
            continue
        positions = [target_by_frame[f] for f in window]
        total_dist = sum(
            math.sqrt((positions[j][0] - positions[j-1][0])**2 +
                       (positions[j][1] - positions[j-1][1])**2)
            for j in range(1, len(positions))
        )
        net_dist = math.sqrt(
            (positions[-1][0] - positions[0][0])**2 +
            (positions[-1][1] - positions[0][1])**2
        )
        if total_dist > 10:  # minimum movement to count
            purposefulness_scores.append(net_dist / total_dist)

    avg_purposefulness = np.mean(purposefulness_scores) if purposefulness_scores else 0

    # ===== 2. Spacing Analysis =====
    # For each frame where target + teammates are visible, compute:
    # - Avg distance to nearest teammate
    # - Avg distance to team centroid
    # - Width coverage (spread)
    nearest_teammate_dists = []
    centroid_dists = []
    team_spreads = []

    for fn in target_frames:
        if fn not in teammates_by_frame or len(teammates_by_frame[fn]) < 2:
            continue
        tx, ty = target_by_frame[fn]
        mates = teammates_by_frame[fn]

        # Nearest teammate distance
        dists = [math.sqrt((tx - mx)**2 + (ty - my)**2) for mx, my in mates]
        nearest_teammate_dists.append(min(dists) / px_per_m)

        # Team centroid
        all_x = [tx] + [m[0] for m in mates]
        all_y = [ty] + [m[1] for m in mates]
        cx_team = np.mean(all_x)
        cy_team = np.mean(all_y)
        centroid_dists.append(math.sqrt((tx - cx_team)**2 + (ty - cy_team)**2) / px_per_m)

        # Team spread (width coverage)
        spread_x = (max(all_x) - min(all_x)) / px_per_m
        team_spreads.append(spread_x)

    avg_nearest_mate = np.mean(nearest_teammate_dists) if nearest_teammate_dists else 0
    avg_centroid_dist = np.mean(centroid_dists) if centroid_dists else 0
    avg_team_spread = np.mean(team_spreads) if team_spreads else 0

    # ===== 3. Positional Discipline =====
    # How consistently does the player stay in their inferred zone?
    # Compute std deviation of x-position over time (lower = more disciplined)
    target_xs = [target_by_frame[f][0] for f in target_frames]
    target_ys = [target_by_frame[f][1] for f in target_frames]
    x_std_m = np.std(target_xs) / px_per_m
    y_std_m = np.std(target_ys) / px_per_m
    position_consistency = 1.0 / (1.0 + x_std_m * 0.1)  # 0-1 score, higher = more consistent

    # Zone breakdown: what % of time in each horizontal band
    zones = {"own_third": 0, "middle_third": 0, "final_third": 0}
    for x in target_xs:
        frac = x / W
        if frac < 0.33:
            zones["own_third"] += 1
        elif frac < 0.67:
            zones["middle_third"] += 1
        else:
            zones["final_third"] += 1
    total = sum(zones.values()) or 1
    zones = {k: round(v / total * 100, 1) for k, v in zones.items()}

    # ===== 4. Work Rate Phases =====
    # Classify each moment as: idle (<1 m/s), jogging (1-3), running (3-5), sprinting (>5)
    phase_counts = {"idle": 0, "jogging": 0, "running": 0, "sprinting": 0}
    for i in range(1, len(target_frames)):
        fn_prev = target_frames[i-1]
        fn_curr = target_frames[i]
        p1 = target_by_frame[fn_prev]
        p2 = target_by_frame[fn_curr]
        dist_px = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        dt = (fn_curr - fn_prev) / fps
        if dt <= 0:
            continue
        speed = min((dist_px / px_per_m) / dt, config.SPEED_OUTLIER_CAP)
        if speed < 1.0:
            phase_counts["idle"] += 1
        elif speed < 3.0:
            phase_counts["jogging"] += 1
        elif speed < 5.0:
            phase_counts["running"] += 1
        else:
            phase_counts["sprinting"] += 1

    total_phases = sum(phase_counts.values()) or 1
    phase_pcts = {k: round(v / total_phases * 100, 1) for k, v in phase_counts.items()}

    # ===== 5. Lateral vs Vertical Movement =====
    lateral_dist = 0
    vertical_dist = 0
    for i in range(1, len(target_frames)):
        p1 = target_by_frame[target_frames[i-1]]
        p2 = target_by_frame[target_frames[i]]
        dx = abs(p2[0] - p1[0]) / px_per_m
        dy = abs(p2[1] - p1[1]) / px_per_m
        lateral_dist += dx  # side to side
        vertical_dist += dy  # up/down field

    total_movement = lateral_dist + vertical_dist or 1
    lateral_pct = round(lateral_dist / total_movement * 100, 1)
    vertical_pct = round(vertical_dist / total_movement * 100, 1)

    # Build output
    advanced = {
        "movement_purposefulness": {
            "score": round(avg_purposefulness, 3),
            "interpretation": (
                "High (purposeful, direct runs)" if avg_purposefulness > 0.6
                else "Medium (mix of direct and reactive)" if avg_purposefulness > 0.35
                else "Low (reactive, lots of direction changes)"
            ),
            "window_count": len(purposefulness_scores),
        },
        "spacing": {
            "avg_nearest_teammate_m": round(avg_nearest_mate, 1),
            "avg_distance_from_centroid_m": round(avg_centroid_dist, 1),
            "avg_team_spread_m": round(avg_team_spread, 1),
            "interpretation": (
                "Too close to teammates (should spread wider)" if avg_nearest_mate < 5
                else "Good spacing" if avg_nearest_mate < 15
                else "Isolated from team (too far from teammates)"
            ),
        },
        "positional_discipline": {
            "x_variability_m": round(x_std_m, 1),
            "y_variability_m": round(y_std_m, 1),
            "consistency_score": round(position_consistency, 3),
            "zone_distribution": zones,
        },
        "work_rate_phases": phase_pcts,
        "movement_direction": {
            "lateral_pct": lateral_pct,
            "vertical_pct": vertical_pct,
            "interpretation": (
                "Very lateral — needs more vertical (toward goal) runs" if lateral_pct > 70
                else "Good balance of lateral and vertical movement" if lateral_pct > 40
                else "Strong vertical movement — good attacking intent"
            ),
        },
    }

    output_path = os.path.join(output_dir, "advanced_stats.json")
    with open(output_path, "w") as f:
        json.dump(advanced, f, indent=2)
    print(f"  Saved to {output_path}")

    # Print summary
    print(f"  Movement purposefulness: {avg_purposefulness:.2f} ({advanced['movement_purposefulness']['interpretation']})")
    print(f"  Nearest teammate: {avg_nearest_mate:.1f}m ({advanced['spacing']['interpretation']})")
    print(f"  Work rate: idle {phase_pcts['idle']}%, jog {phase_pcts['jogging']}%, run {phase_pcts['running']}%, sprint {phase_pcts['sprinting']}%")
    print(f"  Movement: {lateral_pct}% lateral / {vertical_pct}% vertical ({advanced['movement_direction']['interpretation']})")

    return output_path


def _save_empty(output_dir: str) -> str:
    path = os.path.join(output_dir, "advanced_stats.json")
    with open(path, "w") as f:
        json.dump({"error": "no target player data"}, f)
    return path
