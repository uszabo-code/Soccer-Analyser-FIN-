"""Stage 3: Feature extraction — movement stats, key moments, positional analysis."""

import json
import math
import os
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

import config
from models.data import Detection, PlayerStats, TimeRange, KeyMoment
from utils.video import frame_to_timestamp, frame_to_seconds


# Minimum confidence to trust HSV field detection (fraction of frames that found green)
_HSV_CONFIDENCE_THRESHOLD = 0.4

# Gap between consecutive detections (in frames) that marks a new tracking fragment.
# Matches Stage 1b's MAX_GAP_FRAMES — any gap wider than this means the player
# was off-camera or undetected, so no distance should be accumulated across it.
FRAGMENT_GAP_FRAMES = 450


def auto_detect_field_width(video_path: str, fps: float, check_frames: int = 10) -> tuple:
    """
    Estimate field width in pixels by detecting the green pitch in early frames.

    Segments the pitch using HSV colour thresholding, finds the horizontal extent
    of the largest green region, and averages across multiple frames.

    Returns (field_width_px, confidence) where confidence is the fraction of
    sampled frames that contained a detectable green region (0.0–1.0).
    Returns (None, 0.0) if the video cannot be opened or no green is found.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_count = min(check_frames, total_frames)

    widths = []
    for i in range(sample_count):
        frame_pos = int(i * total_frames / sample_count)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if not ret:
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Natural grass green: H 35–85, S ≥40, V ≥40
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find horizontal extent of the green region
        cols_with_green = np.any(mask > 0, axis=0)
        if cols_with_green.sum() > 0:
            leftmost = int(np.argmax(cols_with_green))
            rightmost = int(len(cols_with_green) - np.argmax(cols_with_green[::-1]) - 1)
            width_px = rightmost - leftmost
            # Only count if green spans at least 30% of the frame width
            if width_px > frame.shape[1] * 0.3:
                widths.append(width_px)

    cap.release()

    if not widths:
        return None, 0.0

    confidence = len(widths) / sample_count
    return float(np.mean(widths)), confidence


def run(detections_path: str, identity_path: str, output_dir: str,
        video_path=None) -> str:
    """
    Extract player statistics and key moments.

    Args:
        detections_path: Path to detections.json from Stage 1/1b.
        identity_path:   Path to player_identity.json from Stage 2.
        output_dir:      Directory to write player_stats.json.
        video_path:      Optional path to the original video. When provided,
                         auto_detect_field_width() is used for a more accurate
                         pixel-to-metre conversion. Falls back to FIELD_WIDTH_METERS.

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

    # ── Pixel-to-metre conversion ──────────────────────────────────────────────
    # Try HSV auto-detection first; fall back to config value.
    calibration_confidence = 0.0
    if video_path:
        detected_width_px, calibration_confidence = auto_detect_field_width(video_path, fps)
        if detected_width_px and calibration_confidence >= _HSV_CONFIDENCE_THRESHOLD:
            px_per_meter = detected_width_px / config.FIELD_WIDTH_METERS
            print(f"  Field width auto-detected: {detected_width_px:.0f}px "
                  f"(confidence {calibration_confidence:.0%}) → "
                  f"{px_per_meter:.1f} px/m")
        else:
            px_per_meter = frame_width / config.FIELD_WIDTH_METERS
            print(f"  Field width: using config ({config.FIELD_WIDTH_METERS}m) — "
                  f"auto-detection confidence too low ({calibration_confidence:.0%}). "
                  f"Speed/distance estimates may be inaccurate. "
                  f"Override with --field-width.")
    else:
        px_per_meter = frame_width / config.FIELD_WIDTH_METERS

    time_per_frame = frame_skip / fps if fps > 0 else 0.033

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

    # Determine which tracks to analyse
    # Threshold: other players need ≥50 detections; target handled separately below
    MIN_DETS_OTHER = 50
    MIN_DETS_ABSOLUTE = 10   # Hard floor — below this stats are meaningless

    non_target_analyze_ids = set()
    for p in id_data["players"]:
        for tid in p["track_ids"]:
            if tid not in target_track_ids and len(tracks.get(tid, [])) >= MIN_DETS_OTHER:
                non_target_analyze_ids.add(tid)

    # --- Target player: combine ALL fragments into one stat entry ---
    # Player #15 may have many short track fragments across the game (tracker
    # loses and re-acquires them). Instead of one stat per fragment (which would
    # show only ~26s), we concatenate all fragments and compute stats across the
    # full game span, skipping distance at fragment boundaries to avoid
    # teleportation artifacts.
    combined_target_dets = []
    for tid in target_track_ids:
        combined_target_dets.extend(tracks.get(tid, []))
    combined_target_dets.sort(key=lambda d: d.frame_num)

    # Deduplicate: keep only the first detection per frame.
    # Safety net against overlapping tracks (e.g. sweep added tracks that are
    # simultaneously active with confirmed target tracks). Without this, multiple
    # detections at the same frame_num create zero-span micro-fragments, making
    # time_visible ≈ 0 and total_distance ≈ 0.
    seen_frames: dict = {}
    for d in combined_target_dets:
        if d.frame_num not in seen_frames:
            seen_frames[d.frame_num] = d
    combined_target_dets = sorted(seen_frames.values(), key=lambda d: d.frame_num)

    all_stats = []

    if combined_target_dets:
        target_jersey = id_data.get("target_jersey")
        target_player_info = {
            "team": id_data.get("target_team", "unknown"),
            "jersey_number": target_jersey,
            "display_name": f"#{target_jersey}" if target_jersey else "Selected Player",
        }
        n_frags = len(target_track_ids)
        print(f"  Target: combining {n_frags} fragment(s) "
              f"({len(combined_target_dets)} total detections)")
        target_stats = _compute_player_stats(
            track_id="target_player",
            dets=combined_target_dets,
            player_info=target_player_info,
            ball_by_frame=ball_by_frame,
            px_per_meter=px_per_meter,
            time_per_frame=time_per_frame,
            fps=fps,
            frame_width=frame_width,
            frame_height=frame_height,
            is_target=True,
            fragment_gap_frames=FRAGMENT_GAP_FRAMES,
        )
        all_stats.append(target_stats.to_dict())
    else:
        print(f"  ⚠️  WARNING: No detections found for target player tracks {target_track_ids}")

    print(f"  Analysing {len(non_target_analyze_ids)} other player tracks...")

    for track_id in tqdm(non_target_analyze_ids, desc="Features", unit="player"):
        dets = tracks.get(track_id, [])

        if len(dets) < MIN_DETS_ABSOLUTE:
            continue

        player_info = players_by_id.get(track_id, {})

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
            is_target=False,
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


def _smooth_positions_fragment_aware(dets: list, centers: list,
                                     fragment_gap_frames, window: int) -> list:
    """
    Apply position smoothing only WITHIN fragment boundaries, never across them.

    Without this guard, a standard moving-average window near a fragment boundary
    blends positions from two different fragments — e.g. a player at x=100 and
    x=900 get averaged to x=500, creating artificial distance in the smoothed path.
    """
    if window <= 1 or len(centers) < 2:
        return centers

    # Identify fragment start indices (boundary = track_id change or large gap)
    frag_starts = [0]
    for i in range(1, len(dets)):
        gap = dets[i].frame_num - dets[i - 1].frame_num
        tid_changed = (
            hasattr(dets[i], "track_id") and
            hasattr(dets[i - 1], "track_id") and
            dets[i].track_id != dets[i - 1].track_id
        )
        if tid_changed or (fragment_gap_frames is not None and gap > fragment_gap_frames):
            frag_starts.append(i)

    if len(frag_starts) == 1:
        # Single contiguous fragment — standard smoothing
        return _smooth_positions(centers, window)

    # Multi-fragment: smooth each fragment independently
    result = list(centers)  # copy
    frag_ends = frag_starts[1:] + [len(centers)]
    for start, end in zip(frag_starts, frag_ends):
        frag_centers = centers[start:end]
        smoothed = _smooth_positions(frag_centers, window)
        result[start:end] = smoothed
    return result


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
    frame_width, frame_height, is_target,
    fragment_gap_frames=None,
) -> PlayerStats:
    """
    Compute comprehensive stats for a player track (or combined multi-fragment track).

    fragment_gap_frames: when set, any consecutive pair of detections separated
    by more than this many frames is treated as a fragment boundary — no distance
    is accumulated across the gap (avoids teleportation distance for multi-fragment
    target players whose tracks span the full game with off-camera gaps).
    """

    # Centers and frame numbers
    raw_centers = [d.center for d in dets]
    frames = [d.frame_num for d in dets]

    # Apply position smoothing to reduce tracking jitter.
    # For multi-fragment (combined target) data, use fragment-aware smoothing so
    # the moving average never blends positions from different fragments.
    if fragment_gap_frames is not None:
        centers = _smooth_positions_fragment_aware(
            dets, raw_centers, fragment_gap_frames, config.SMOOTHING_WINDOW
        )
    else:
        centers = _smooth_positions(raw_centers, config.SMOOTHING_WINDOW)

    # --- Distance and Speed ---
    distances = []  # pixel distances between consecutive detections
    speeds = []  # m/s

    for i in range(1, len(centers)):
        frame_diff = frames[i] - frames[i - 1]

        # Skip distance/speed calculation across fragment boundaries.
        # A boundary occurs when:
        #   (a) The frame gap is larger than the threshold (player off-camera), OR
        #   (b) The track_id changes (different canonical tracks = defined fragment
        #       boundary, regardless of gap size)
        # Without this, the "teleportation" jump between fragment endpoints inflates
        # distance and speed with physically impossible values.
        is_fragment_boundary = False
        if fragment_gap_frames is not None:
            if frame_diff > fragment_gap_frames:
                is_fragment_boundary = True
            elif hasattr(dets[i], "track_id") and hasattr(dets[i - 1], "track_id"):
                if dets[i].track_id != dets[i - 1].track_id:
                    is_fragment_boundary = True
        if is_fragment_boundary:
            distances.append(0)
            speeds.append(0)
            continue

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

    # Fragment-aware time visible: sum of each fragment's span.
    # For a single continuous track this equals (last - first) / fps.
    # For multi-fragment (combined target) it sums only the visible windows,
    # correctly excluding off-camera gaps between fragments.
    if fragment_gap_frames is not None and len(frames) > 1:
        frag_time_s = 0.0
        frag_start = frames[0]
        prev_frame = frames[0]
        for idx in range(1, len(frames)):
            fn = frames[idx]
            # Fragment boundary: large frame gap OR track_id change
            gap = fn - prev_frame
            tid_changed = (
                hasattr(dets[idx], "track_id") and
                hasattr(dets[idx - 1], "track_id") and
                dets[idx].track_id != dets[idx - 1].track_id
            )
            if gap > fragment_gap_frames or tid_changed:
                frag_time_s += (prev_frame - frag_start) / fps
                frag_start = fn
            prev_frame = fn
        frag_time_s += (prev_frame - frag_start) / fps
        total_time_s = frag_time_s
    else:
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
