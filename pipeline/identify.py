"""Stage 2: Player identification via jersey number OCR and color clustering."""

import json
import os
from collections import Counter, defaultdict

import cv2
import numpy as np
from tqdm import tqdm

import config
from models.data import Detection, PlayerIdentity
from utils.ocr import read_jersey_number, read_jersey_number_multi, get_dominant_color, color_to_name
from utils.video import VideoReader


def run(video_path: str, detections_path: str, output_dir: str,
        target_jersey: int) -> str:
    """
    Identify players by jersey number and cluster into teams.

    Returns path to the output player identity JSON file.
    """
    print(f"\n[Stage 2] Player Identification")
    print(f"  Target jersey number: {target_jersey}")

    # Load detections
    with open(detections_path) as f:
        det_data = json.load(f)

    fps = det_data["fps"]
    frame_width = det_data["width"]
    frame_height = det_data["height"]
    person_dets = [Detection.from_dict(d) for d in det_data["person_detections"]]

    # Group detections by track_id
    tracks = defaultdict(list)
    for det in person_dets:
        if det.track_id >= 0:
            tracks[det.track_id].append(det)

    print(f"  Total tracks: {len(tracks)}")

    # Filter to significant tracks only — scale with frame_skip
    # FRAME_SKIP=3 → 30 dets (~2-3s), FRAME_SKIP=1 → 90 dets (~2-3s)
    frame_skip = det_data.get("frame_skip", config.FRAME_SKIP)
    MIN_TRACK_LENGTH = max(30, int(90 / frame_skip))
    significant_tracks = {tid: dets for tid, dets in tracks.items() if len(dets) >= MIN_TRACK_LENGTH}
    print(f"  Significant tracks (>={MIN_TRACK_LENGTH} detections): {len(significant_tracks)}")

    # For each track, select frames where player is largest (best OCR candidates)
    reader = VideoReader(video_path, frame_skip=1)  # Need full access for specific frames

    track_ocr_results = {}  # track_id -> list of OCR results
    track_colors = {}  # track_id -> list of (H,S,V)

    print("  Running jersey number OCR...")
    for track_id, dets in tqdm(significant_tracks.items(), desc="OCR", unit="track"):
        # Sort by bbox height (largest first) for best OCR candidates
        sorted_dets = sorted(dets, key=lambda d: d.height, reverse=True)
        # 5 best samples — multi-strategy gives 3 candidates per frame already
        ocr_samples = sorted_dets[:min(5, config.OCR_SAMPLES_PER_TRACK)]
        color_samples = sorted_dets[:3]

        ocr_numbers = []
        colors = []

        # Skip OCR entirely if largest bbox is too small to read
        skip_ocr = sorted_dets[0].height < config.MIN_PLAYER_HEIGHT_PX if sorted_dets else True

        # Color extraction first (fast)
        for det in color_samples:
            frame = reader.read_frame(det.frame_num)
            if frame is None:
                continue
            x1, y1, x2, y2 = det.bbox
            pad_x = (x2 - x1) * config.JERSEY_CROP_PADDING
            pad_y = (y2 - y1) * config.JERSEY_CROP_PADDING
            cx1 = max(0, int(x1 - pad_x))
            cy1 = max(0, int(y1 - pad_y))
            cx2 = min(frame_width, int(x2 + pad_x))
            cy2 = min(frame_height, int(y2 + pad_y))
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                colors.append(get_dominant_color(crop))

        for det in ocr_samples:
            if skip_ocr:
                break
            if det.height < config.MIN_PLAYER_HEIGHT_PX:
                continue

            frame = reader.read_frame(det.frame_num)
            if frame is None:
                continue

            # Crop with padding
            x1, y1, x2, y2 = det.bbox
            pad_x = (x2 - x1) * config.JERSEY_CROP_PADDING
            pad_y = (y2 - y1) * config.JERSEY_CROP_PADDING
            cx1 = max(0, int(x1 - pad_x))
            cy1 = max(0, int(y1 - pad_y))
            cx2 = min(frame_width, int(x2 + pad_x))
            cy2 = min(frame_height, int(y2 + pad_y))
            crop = frame[cy1:cy2, cx1:cx2]

            if crop.size == 0:
                continue

            # OCR on upper body (jersey area) using multi-strategy
            jersey_crop = crop[:int(crop.shape[0] * 0.6), :]
            if jersey_crop.size > 0:
                candidates = read_jersey_number_multi(jersey_crop)
                ocr_numbers.extend(candidates)

        track_ocr_results[track_id] = ocr_numbers
        if colors:
            track_colors[track_id] = np.mean(colors, axis=0).tolist()

    reader.close()

    # Assign jersey numbers by majority vote
    track_jerseys = {}
    for track_id, numbers in track_ocr_results.items():
        if numbers:
            counter = Counter(numbers)
            most_common_num, count = counter.most_common(1)[0]
            if count >= config.OCR_MIN_VOTES:
                track_jerseys[track_id] = most_common_num

    print(f"  Identified {len(track_jerseys)} tracks with jersey numbers: "
          f"{dict(sorted(track_jerseys.items(), key=lambda x: x[1]))}")

    # Find target player
    target_track_ids = [tid for tid, num in track_jerseys.items() if num == target_jersey]

    if not target_track_ids:
        print(f"  WARNING: Could not find jersey #{target_jersey} via OCR.")
        print(f"  Available numbers: {sorted(set(track_jerseys.values()))}")
        # Skip interactive fallback (blocks in headless/CLI context)
        # Instead, pick the longest unidentified track as best guess
        print(f"  Skipping interactive selection — will use largest unidentified track.")

    if target_track_ids:
        print(f"  Target player found: track IDs {target_track_ids}")
    else:
        print(f"  WARNING: Could not identify target player. Will analyze largest track.")
        # Fall back to the track with the most detections
        largest_track = max(tracks.keys(), key=lambda tid: len(tracks[tid]))
        target_track_ids = [largest_track]

    # Cluster tracks into teams by jersey color
    teams = _cluster_teams(track_colors, tracks)

    # Determine target player's team
    target_team = "unknown"
    for team_name, team_track_ids in teams.items():
        if any(tid in team_track_ids for tid in target_track_ids):
            target_team = team_name
            break

    # Build player identities (only for significant tracks)
    players = []
    for track_id in significant_tracks:
        color_hsv = track_colors.get(track_id, (0, 0, 0))
        jersey_num = track_jerseys.get(track_id)
        is_target = track_id in target_track_ids

        team = "unknown"
        for team_name, team_tids in teams.items():
            if track_id in team_tids:
                team = team_name
                break

        if jersey_num is not None:
            display = f"#{jersey_num}"
        else:
            display = f"Track-{track_id} ({color_to_name(tuple(int(c) for c in color_hsv))})"

        players.append(PlayerIdentity(
            track_ids=[track_id],
            jersey_number=jersey_num,
            jersey_color=color_to_name(tuple(int(c) for c in color_hsv)),
            team=team,
            display_name=display,
            is_target=is_target,
        ))

    # Save
    output_path = os.path.join(output_dir, "player_identity.json")
    output_data = {
        "target_jersey": target_jersey,
        "target_track_ids": target_track_ids,
        "target_team": target_team,
        "teams": {k: v for k, v in teams.items()},
        "players": [p.to_dict() for p in players],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"  Saved to {output_path}")
    return output_path


def _cluster_teams(track_colors: dict, tracks: dict) -> dict:
    """Cluster tracks into two teams based on jersey color."""
    if len(track_colors) < 4:
        return {"team_a": list(track_colors.keys()), "team_b": []}

    try:
        from sklearn.cluster import KMeans

        # Only cluster tracks with enough detections (likely actual players, not noise)
        significant_tracks = {
            tid: color for tid, color in track_colors.items()
            if len(tracks.get(tid, [])) > 30  # At least 30 detections
        }

        if len(significant_tracks) < 4:
            significant_tracks = track_colors

        track_ids = list(significant_tracks.keys())
        colors = np.array([significant_tracks[tid] for tid in track_ids])

        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(colors)

        teams = {"team_a": [], "team_b": []}
        for tid, label in zip(track_ids, labels):
            team_name = "team_a" if label == 0 else "team_b"
            teams[team_name].append(tid)

        # Assign remaining tracks to closest team
        for tid in track_colors:
            if tid not in significant_tracks:
                color = np.array(track_colors[tid]).reshape(1, -1)
                label = kmeans.predict(color)[0]
                team_name = "team_a" if label == 0 else "team_b"
                teams[team_name].append(tid)

        return teams
    except ImportError:
        # No sklearn, just split roughly
        all_tids = list(track_colors.keys())
        mid = len(all_tids) // 2
        return {"team_a": all_tids[:mid], "team_b": all_tids[mid:]}


def _interactive_select(video_path: str, tracks: dict, det_data: dict) -> list:
    """
    Fallback: show a frame and let the user click on their player.
    Returns list of track_ids or empty list if not possible.
    """
    try:
        # Find a frame in the middle of the video with many detections
        mid_frame_approx = det_data["total_frames"] // 2
        dets_by_frame = defaultdict(list)
        for d in det_data["person_detections"]:
            dets_by_frame[d["frame_num"]].append(d)

        # Find frame closest to midpoint with good detection count
        best_frame = min(dets_by_frame.keys(), key=lambda f: abs(f - mid_frame_approx))

        reader = VideoReader(video_path, frame_skip=1)
        frame = reader.read_frame(best_frame)
        reader.close()

        if frame is None:
            return []

        # Draw bounding boxes with track IDs
        for det in dets_by_frame[best_frame]:
            x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
            tid = det["track_id"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"T{tid}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show and wait for click
        selected = []

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                for det in dets_by_frame[best_frame]:
                    bx1, by1, bx2, by2 = [int(c) for c in det["bbox"]]
                    if bx1 <= x <= bx2 and by1 <= y <= by2:
                        selected.append(det["track_id"])
                        print(f"  Selected track ID: {det['track_id']}")
                        break

        cv2.namedWindow("Click on your player (press 'q' to confirm)")
        cv2.setMouseCallback("Click on your player (press 'q' to confirm)", on_click)

        # Resize for display if needed
        display_h = 720
        scale = display_h / frame.shape[0]
        display_frame = cv2.resize(frame, (int(frame.shape[1] * scale), display_h))

        cv2.imshow("Click on your player (press 'q' to confirm)", display_frame)
        print("  A window has opened. Click on your son in the image, then press 'q' to confirm.")

        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord("q") and selected:
                break
            if key == 27:  # ESC to cancel
                selected = []
                break

        cv2.destroyAllWindows()
        return list(set(selected))

    except Exception as e:
        print(f"  Interactive selection failed: {e}")
        print("  Continuing without target identification.")
        return []
