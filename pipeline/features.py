"""Stage 3: Feature extraction — movement stats, key moments, positional analysis."""

import json
import math
import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import config
from models.data import Detection, PlayerStats, TimeRange, KeyMoment
from utils.video import frame_to_timestamp, frame_to_seconds


def run(detections_path: str, identity_path: str, output_dir: str) -> str:
    """
    Extract player statistics and key moments.

    Returns path to the output player stats JSON file.
    """
    print(f"\n[Stage 3] Feature Extraction")

    # Load data
    with open(detections_path) as f:
        det_data = json.load(f)
    with open(identity_path) as f:
        id_data = json.load(f)

    fps = det_data["fps"]
    frame_width = det_data["width"]
    frame_height = det_data["height"]
    frame_skip = det_data["frame_skip"]
    person_dets = [Detection.from_dict(d) for d in det_data["person_detections"]]
    ball_dets = [Detection.from_dict(d) for d in det_data["ball_detections"]]

    target_track_ids = set(id_data["target_track_ids"])
    players_by_id = {}
    for p in id_data["players"]:
        for tid in p["track_ids"]:
            players_by_id[tid] = p

    # Pixel-to-meter conversion (approximate)
    # Assume the full frame width corresponds to the configured field width
    px_per_meter = frame_width / config.FIELD_WIDTH_METERS
    time_per_frame = frame_skip / fps if fps > 0 else 0.033  # seconds between processed frames

    # Group detections by track
    tracks = defaultdict(list)
    for det in person_dets:
        if det.track_id >= 0:
            tracks[det.track_id].append(det)

    # Sort each track by frame number
    for tid in tracks:
        tracks[tid].sort(key=lambda d: d.frame_num)

    # Group ball detections by frame
    ball_by_frame = {}
    for bd in ball_dets:
        ball_by_frame[bd.frame_num] = bd.center

    # Determine which tracks to analyze in detail
    # Always analyze the target player, plus teammates and opponents for context
    analyze_track_ids = set()
    for p in id_data["players"]:
        # Include all tracks with enough detections
        for tid in p["track_ids"]:
            if len(tracks.get(tid, [])) > 50:
                analyze_track_ids.add(tid)
    # Always include target
    analyze_track_ids.update(target_track_ids)

    print(f"  Analyzing {len(analyze_track_ids)} significant tracks")

    all_stats = []

    for track_id in tqdm(analyze_track_ids, desc="Features", unit="player"):
        dets = tracks.get(track_id, [])
        if len(dets) < 10:
            continue

        player_info = players_by_id.get(track_id, {})
        is_target = track_id in target_track_ids

        stats = _compute_player_stats(
            track_id=track_id,
            dets=dets,
            player_info=player_info,
            ball_by_frame=ball_by_frame,
            px_per_meter=px_per_meter,
            time_per_frame=time_per_frame,
            fps=fps,
            frame_width=frame_width,
            frame_height=frame_height,
            is_target=is_target,
        )
        all_stats.append(stats.to_dict())

    # Sort: target player first, then by team
    all_stats.sort(key=lambda s: (not s["is_target"], s["team"], s["player_id"]))

    # Save
    output_path = os.path.join(output_dir, "player_stats.json")
    output_data = {
        "fps": fps,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "field_width_m": config.FIELD_WIDTH_METERS,
        "players": all_stats,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"  Saved to {output_path}")
    return output_path


def _smooth_positions(centers: list, window: int) -> list:
    """Apply moving average smoothing to center positions."""
    if window <= 1 or len(centers) < window:
        return centers
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


def _compute_player_stats(
    track_id, dets, player_info, ball_by_frame,
    px_per_meter, time_per_frame, fps,
    frame_width, frame_height, is_target
) -> PlayerStats:
    """Compute comprehensive stats for a single player track."""

    # Centers and frame numbers
    raw_centers = [d.center for d in dets]
    frames = [d.frame_num for d in dets]

    # Apply position smoothing to reduce tracking jitter
    centers = _smooth_positions(raw_centers, config.SMOOTHING_WINDOW)

    # --- Distance and Speed ---
    distances = []  # pixel distances between consecutive detections
    speeds = []  # m/s

    for i in range(1, len(centers)):
        dx = centers[i][0] - centers[i - 1][0]
        dy = centers[i][1] - centers[i - 1][1]
        dist_px = math.sqrt(dx * dx + dy * dy)

        # Skip sub-threshold jitter
        if dist_px < config.MIN_MOVEMENT_PX:
            distances.append(0)
            speeds.append(0)
            continue

        distances.append(dist_px)

        # Time between these detections
        frame_diff = frames[i] - frames[i - 1]
        dt = frame_diff / fps if fps > 0 else time_per_frame
        if dt > 0:
            speed_px = dist_px / dt
            speed_m = speed_px / px_per_meter
            # Cap unrealistic speeds
            speed_m = min(speed_m, config.SPEED_OUTLIER_CAP)
            speeds.append(speed_m)
        else:
            speeds.append(0)

    total_distance_px = sum(distances)
    total_distance_m = total_distance_px / px_per_meter

    avg_speed = np.mean(speeds) if speeds else 0
    max_speed = max(speeds) if speeds else 0

    # --- Position / Field Thirds ---
    xs = [c[0] for c in centers]
    avg_x = np.mean(xs)
    third_width = frame_width / 3

    time_in_thirds = {"defensive": 0, "middle": 0, "attacking": 0}
    for x in xs:
        if x < third_width:
            time_in_thirds["defensive"] += 1
        elif x < 2 * third_width:
            time_in_thirds["middle"] += 1
        else:
            time_in_thirds["attacking"] += 1

    total_pts = sum(time_in_thirds.values()) or 1
    time_in_thirds = {k: (v / total_pts) * 100 for k, v in time_in_thirds.items()}

    # --- Inferred Position ---
    avg_x_frac = avg_x / frame_width
    avg_y_frac = np.mean([c[1] for c in centers]) / frame_height

    if avg_x_frac < 0.25:
        inferred_pos = "goalkeeper" if avg_x_frac < 0.1 else "defender"
    elif avg_x_frac < 0.5:
        inferred_pos = "defensive midfielder" if avg_x_frac < 0.37 else "midfielder"
    elif avg_x_frac < 0.75:
        inferred_pos = "attacking midfielder"
    else:
        inferred_pos = "forward"

    # --- Sprint Episodes ---
    sprints = []
    in_sprint = False
    sprint_start = None

    for i, speed in enumerate(speeds):
        if speed >= config.SPRINT_SPEED_THRESHOLD:
            if not in_sprint:
                in_sprint = True
                sprint_start = frames[i]
        else:
            if in_sprint:
                sprint_end = frames[i]
                if sprint_end - sprint_start > fps * 0.5:  # At least 0.5s
                    sprints.append(TimeRange(
                        start_frame=sprint_start,
                        end_frame=sprint_end,
                        start_time=frame_to_timestamp(sprint_start, fps),
                        end_time=frame_to_timestamp(sprint_end, fps),
                    ))
                in_sprint = False

    # Close final sprint if still running
    if in_sprint and sprint_start is not None:
        sprints.append(TimeRange(
            start_frame=sprint_start,
            end_frame=frames[-1],
            start_time=frame_to_timestamp(sprint_start, fps),
            end_time=frame_to_timestamp(frames[-1], fps),
        ))

    # --- Ball Proximity ---
    ball_proximity_episodes = []
    near_ball = False
    prox_start = None
    BALL_PROXIMITY_PX = frame_width * 0.05  # Within 5% of frame width

    for det in dets:
        ball_pos = ball_by_frame.get(det.frame_num)
        if ball_pos is not None:
            dx = det.center[0] - ball_pos[0]
            dy = det.center[1] - ball_pos[1]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < BALL_PROXIMITY_PX:
                if not near_ball:
                    near_ball = True
                    prox_start = det.frame_num
            else:
                if near_ball:
                    ball_proximity_episodes.append(TimeRange(
                        start_frame=prox_start,
                        end_frame=det.frame_num,
                        start_time=frame_to_timestamp(prox_start, fps),
                        end_time=frame_to_timestamp(det.frame_num, fps),
                    ))
                    near_ball = False

    if near_ball and prox_start is not None:
        ball_proximity_episodes.append(TimeRange(
            start_frame=prox_start,
            end_frame=dets[-1].frame_num,
            start_time=frame_to_timestamp(prox_start, fps),
            end_time=frame_to_timestamp(dets[-1].frame_num, fps),
        ))

    # --- Key Moments ---
    key_moments = []

    # Sprints are key moments
    for sprint in sprints:
        key_moments.append(KeyMoment(
            moment_type="sprint",
            time_range=sprint,
            description=f"Sprint detected",
        ))

    # Direction changes (significant change in movement direction)
    for i in range(2, len(centers)):
        dx1 = centers[i - 1][0] - centers[i - 2][0]
        dy1 = centers[i - 1][1] - centers[i - 2][1]
        dx2 = centers[i][0] - centers[i - 1][0]
        dy2 = centers[i][1] - centers[i - 1][1]

        len1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
        len2 = math.sqrt(dx2 * dx2 + dy2 * dy2)

        if len1 > config.MIN_MOVEMENT_PX and len2 > config.MIN_MOVEMENT_PX:
            dot = dx1 * dx2 + dy1 * dy2
            cos_angle = dot / (len1 * len2)
            cos_angle = max(-1, min(1, cos_angle))
            angle = math.degrees(math.acos(cos_angle))

            if angle > config.DIRECTION_CHANGE_ANGLE and speeds[i - 1] > 2.0:
                key_moments.append(KeyMoment(
                    moment_type="direction_change",
                    time_range=TimeRange(
                        start_frame=frames[i - 1],
                        end_frame=frames[i],
                        start_time=frame_to_timestamp(frames[i - 1], fps),
                        end_time=frame_to_timestamp(frames[i], fps),
                    ),
                    description=f"Sharp direction change ({angle:.0f} degrees)",
                ))

    # Ball involvement moments
    for ep in ball_proximity_episodes:
        duration_s = (ep.end_frame - ep.start_frame) / fps if fps > 0 else 0
        if duration_s > 1.0:  # At least 1 second near ball
            key_moments.append(KeyMoment(
                moment_type="ball_involvement",
                time_range=ep,
                description=f"Near ball for {duration_s:.1f}s",
            ))

    # Limit to most interesting moments (avoid overwhelming the LLM)
    key_moments.sort(key=lambda m: m.time_range.start_frame)
    if len(key_moments) > 50:
        # Keep a spread across the game
        step = len(key_moments) // 50
        key_moments = key_moments[::step][:50]

    # --- Heatmap ---
    GRID_ROWS = 10
    GRID_COLS = 15
    heatmap = [[0] * GRID_COLS for _ in range(GRID_ROWS)]
    cell_w = frame_width / GRID_COLS
    cell_h = frame_height / GRID_ROWS

    for cx, cy in centers:
        col = min(int(cx / cell_w), GRID_COLS - 1)
        row = min(int(cy / cell_h), GRID_ROWS - 1)
        heatmap[row][col] += 1

    # --- Build PlayerStats ---
    display_name = player_info.get("display_name", f"Track-{track_id}")
    team = player_info.get("team", "unknown")
    jersey_num = player_info.get("jersey_number")

    total_time_s = (frames[-1] - frames[0]) / fps if fps > 0 and len(frames) > 1 else 0

    return PlayerStats(
        player_id=display_name,
        team=team,
        jersey_number=jersey_num,
        inferred_position=inferred_pos,
        total_distance_m=total_distance_m,
        avg_speed_mps=avg_speed,
        max_speed_mps=max_speed,
        sprint_count=len(sprints),
        time_in_thirds=time_in_thirds,
        sprint_episodes=sprints,
        ball_proximity_episodes=ball_proximity_episodes,
        key_moments=key_moments,
        heatmap=heatmap,
        frames_visible=len(dets),
        total_time_visible_s=total_time_s,
        is_target=is_target,
    )
