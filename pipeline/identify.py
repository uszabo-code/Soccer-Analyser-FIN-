"""Stage 2: Player identification via interactive picker (primary) or jersey OCR (fallback)."""

import json
import math
import os
from collections import Counter, defaultdict

import cv2
import numpy as np
from tqdm import tqdm

import config
from models.data import Detection, PlayerIdentity
from utils.ocr import read_jersey_number_multi, get_dominant_color, color_to_name
from utils.video import VideoReader


def run(video_path: str, detections_path: str, output_dir: str,
        target_jersey: int = None) -> str:
    """
    Identify the target player and cluster all players into teams.

    If target_jersey is given, attempts OCR to find that number; falls back to
    interactive picker if OCR fails. If target_jersey is None, opens the
    interactive picker immediately.

    Returns path to the output player identity JSON file.
    """
    print(f"\n[Stage 2] Player Identification")
    if target_jersey:
        print(f"  Target jersey number: {target_jersey}")
    else:
        print(f"  Mode: interactive player selection")

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
    frame_skip = det_data.get("frame_skip", config.FRAME_SKIP)
    MIN_TRACK_LENGTH = max(30, int(90 / frame_skip))
    significant_tracks = {tid: dets for tid, dets in tracks.items() if len(dets) >= MIN_TRACK_LENGTH}
    print(f"  Significant tracks (>={MIN_TRACK_LENGTH} detections): {len(significant_tracks)}")

    # --- Step 1: Extract jersey colors for all significant tracks (needed for team clustering) ---
    reader = VideoReader(video_path, frame_skip=1)
    track_colors = _extract_colors(significant_tracks, reader, frame_width, frame_height)
    reader.close()

    # --- Step 2: Identify target player ---
    target_track_ids = []
    track_jerseys = {}

    if target_jersey:
        # Try OCR first
        reader = VideoReader(video_path, frame_skip=1)
        track_jerseys = _run_ocr(significant_tracks, reader, frame_width, frame_height)
        reader.close()
        target_track_ids = [tid for tid, num in track_jerseys.items() if num == target_jersey]
        if target_track_ids:
            print(f"  OCR found jersey #{target_jersey}: track IDs {target_track_ids}")
        else:
            print(f"  OCR could not find jersey #{target_jersey}. Opening interactive picker...")

    if not target_track_ids:
        # Interactive picker — primary path when no jersey given, fallback when OCR fails
        target_track_ids = _interactive_select(video_path, tracks, det_data, fps)

    if not target_track_ids:
        print(f"  WARNING: Could not identify target player. Will analyze largest track.")
        largest_track = max(tracks.keys(), key=lambda tid: len(tracks[tid]))
        target_track_ids = [largest_track]
    else:
        print(f"  Target player: track IDs {target_track_ids}")

    # Cluster tracks into teams by jersey color
    teams = _cluster_teams(track_colors, tracks)

    # Determine target player's team
    target_team = "unknown"
    for team_name, team_track_ids in teams.items():
        if any(tid in team_track_ids for tid in target_track_ids):
            target_team = team_name
            break

    # --- Step 3: Sweep for additional target fragments ---
    # After OCR/picker gives us a seed set, find all other same-team tracks that
    # are temporally adjacent + spatially plausible + color-similar. These are
    # additional fragments of the same player that OCR missed (e.g. the player
    # was off-screen for >5s and got a new track ID when they re-entered frame).
    if target_track_ids and target_team != "unknown":
        target_track_ids = _sweep_for_target_fragments(
            target_track_ids=target_track_ids,
            significant_tracks=significant_tracks,
            track_colors=track_colors,
            teams=teams,
            target_team=target_team,
            fps=fps,
            frame_width=frame_width,
        )

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


def _extract_colors(significant_tracks: dict, reader, frame_width: int, frame_height: int) -> dict:
    """Extract dominant HSV jersey color for each significant track."""
    track_colors = {}
    for track_id, dets in significant_tracks.items():
        sorted_dets = sorted(dets, key=lambda d: d.height, reverse=True)
        colors = []
        for det in sorted_dets[:3]:
            frame = reader.read_frame(det.frame_num)
            if frame is None:
                continue
            x1, y1, x2, y2 = det.bbox
            pad_x = (x2 - x1) * config.JERSEY_CROP_PADDING
            pad_y = (y2 - y1) * config.JERSEY_CROP_PADDING
            crop = frame[
                max(0, int(y1 - pad_y)):min(frame_height, int(y2 + pad_y)),
                max(0, int(x1 - pad_x)):min(frame_width, int(x2 + pad_x)),
            ]
            if crop.size > 0:
                colors.append(get_dominant_color(crop))
        if colors:
            track_colors[track_id] = np.mean(colors, axis=0).tolist()
    return track_colors


def _run_ocr(significant_tracks: dict, reader, frame_width: int, frame_height: int) -> dict:
    """Run jersey number OCR on significant tracks. Returns {track_id: jersey_number}."""
    track_jerseys = {}
    print("  Running jersey number OCR...")
    for track_id, dets in tqdm(significant_tracks.items(), desc="OCR", unit="track"):
        sorted_dets = sorted(dets, key=lambda d: d.height, reverse=True)
        if not sorted_dets or sorted_dets[0].height < config.MIN_PLAYER_HEIGHT_PX:
            continue
        ocr_numbers = []
        for det in sorted_dets[:min(5, config.OCR_SAMPLES_PER_TRACK)]:
            if det.height < config.MIN_PLAYER_HEIGHT_PX:
                continue
            frame = reader.read_frame(det.frame_num)
            if frame is None:
                continue
            x1, y1, x2, y2 = det.bbox
            pad_x = (x2 - x1) * config.JERSEY_CROP_PADDING
            pad_y = (y2 - y1) * config.JERSEY_CROP_PADDING
            crop = frame[
                max(0, int(y1 - pad_y)):min(frame_height, int(y2 + pad_y)),
                max(0, int(x1 - pad_x)):min(frame_width, int(x2 + pad_x)),
            ]
            if crop.size == 0:
                continue
            jersey_crop = crop[:int(crop.shape[0] * 0.6), :]
            if jersey_crop.size > 0:
                ocr_numbers.extend(read_jersey_number_multi(jersey_crop))
        if ocr_numbers:
            counter = Counter(ocr_numbers)
            best_num, count = counter.most_common(1)[0]
            if count >= config.OCR_MIN_VOTES:
                track_jerseys[track_id] = best_num
    return track_jerseys


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


def _sweep_for_target_fragments(
    target_track_ids: list,
    significant_tracks: dict,
    track_colors: dict,
    teams: dict,
    target_team: str,
    fps: float,
    frame_width: int,
) -> list:
    """
    Expand target_track_ids by finding same-player fragments missed by OCR/picker.

    After OCR or the interactive picker gives us a seed set of target track IDs,
    many more fragments of the same player may exist in the game — the tracker lost
    the player and reacquired them under a different ID. This sweep finds those
    additional fragments by checking three conditions per candidate track:

      1. Same team colour (already filtered by 'teams' dict)
      2. Temporally adjacent to a known target fragment (gap ≤ SWEEP_MAX_GAP_FRAMES)
      3. Spatially plausible: last known target position → candidate start within
         the distance a player could run in the gap time
      4. HSV colour similarity: candidate hue matches target median hue

    Runs iteratively until no new fragments are added (convergence).

    Returns the (possibly expanded) list of target_track_ids.
    """
    SWEEP_MAX_GAP_FRAMES = 1800   # 60s — covers substitution delays, off-field moments
    SWEEP_HUE_TOLERANCE = 30      # Max hue difference (0-180 scale)
    MAX_SPEED_MPS = 8.0           # Physics cap (same as Stage 1b)
    JITTER_PX = 50                # Spatial jitter buffer

    try:
        from config import FIELD_WIDTH_METERS
        px_per_meter = frame_width / FIELD_WIDTH_METERS
    except Exception:
        px_per_meter = frame_width / 68.0

    # Build endpoints for all significant tracks
    def _endpoints(dets):
        sd = sorted(dets, key=lambda d: d.frame_num)
        return {
            "first_frame": sd[0].frame_num,
            "last_frame": sd[-1].frame_num,
            "first_pos": sd[0].center,
            "last_pos": sd[-1].center,
        }

    all_eps = {tid: _endpoints(dets) for tid, dets in significant_tracks.items()}

    # Target's median HSV color signature
    target_colors_available = [track_colors[tid] for tid in target_track_ids
                                if tid in track_colors]
    if not target_colors_available:
        return target_track_ids  # No color info — can't sweep

    target_hue = float(np.median([c[0] for c in target_colors_available]))

    # Candidate pool: same-team tracks not already in target
    same_team_tracks = set(teams.get(target_team, []))

    current_ids = list(target_track_ids)
    found_new = True

    while found_new:
        found_new = False
        candidate_ids = same_team_tracks - set(current_ids)

        for cid in candidate_ids:
            if cid not in all_eps or cid not in track_colors:
                continue

            cep = all_eps[cid]
            c_hue = track_colors[cid][0]

            # Check 4: HSV hue similarity
            h_diff = min(abs(target_hue - c_hue), 180.0 - abs(target_hue - c_hue))
            if h_diff > SWEEP_HUE_TOLERANCE:
                continue

            # Pre-check: reject candidate if it overlaps ANY confirmed target fragment.
            # The inner adjacency loop's `else: continue` only skips one pair — if the
            # candidate overlaps fragment A but is adjacent to fragment B, it would still
            # be added via fragment B without this guard.
            has_overlap = any(
                cep["first_frame"] <= all_eps[kid]["last_frame"]
                and cep["last_frame"] >= all_eps[kid]["first_frame"]
                for kid in current_ids
                if kid in all_eps
            )
            if has_overlap:
                continue

            # Check 2 & 3: temporal adjacency + spatial plausibility vs any known fragment
            for kid in current_ids:
                if kid not in all_eps:
                    continue
                kep = all_eps[kid]

                # Which ends first: known fragment or candidate?
                if kep["last_frame"] < cep["first_frame"]:
                    gap = cep["first_frame"] - kep["last_frame"]
                    pos_from, pos_to = kep["last_pos"], cep["first_pos"]
                elif cep["last_frame"] < kep["first_frame"]:
                    gap = kep["first_frame"] - cep["last_frame"]
                    pos_from, pos_to = cep["last_pos"], kep["first_pos"]
                else:
                    continue  # Temporal overlap — different player

                if gap > SWEEP_MAX_GAP_FRAMES:
                    continue

                # Physics check
                gap_s = gap / max(fps, 1.0)
                max_dist_px = MAX_SPEED_MPS * gap_s * px_per_meter + JITTER_PX
                dx = pos_to[0] - pos_from[0]
                dy = pos_to[1] - pos_from[1]
                actual_dist = math.sqrt(dx * dx + dy * dy)

                if actual_dist <= max_dist_px:
                    current_ids.append(cid)
                    found_new = True
                    break  # No need to check remaining known fragments

    added = len(current_ids) - len(target_track_ids)
    if added > 0:
        print(f"  Target sweep: found {added} additional fragment(s) → "
              f"{len(current_ids)} total track IDs for target player")

    return current_ids


def _interactive_select(video_path: str, tracks: dict, det_data: dict, fps: float) -> list:
    """
    Show a video frame and let the user click on their player.
    Arrow keys navigate between candidate frames (t=20s to t=90s).
    Returns list containing the selected track_id, or empty list on cancel.
    """
    try:
        dets_by_frame = defaultdict(list)
        for d in det_data["person_detections"]:
            dets_by_frame[d["frame_num"]].append(d)

        # Build candidate frames: ~10 evenly spaced frames from t=20s to t=90s
        t_start = int(fps * 20)
        t_end = int(fps * min(90, det_data["total_frames"] / fps - 5))
        available = sorted(dets_by_frame.keys())
        candidates = [f for f in available if t_start <= f <= t_end]
        if not candidates:
            candidates = available[:20]
        step = max(1, len(candidates) // 10)
        candidate_frames = candidates[::step][:10]
        if not candidate_frames:
            return []

        reader = VideoReader(video_path, frame_skip=1)

        WIN = "Click your player | Left/Right = change frame | Enter = confirm | Esc = cancel"
        DISPLAY_H = 720

        state = {"frame_idx": 0, "selected_tid": None, "confirmed": False, "cancelled": False}

        def _render(raw_frame, frame_dets, selected_tid, frame_idx, total):
            """Draw bboxes and instructions onto a display-sized copy."""
            h, w = raw_frame.shape[:2]
            scale = DISPLAY_H / h
            display_w = int(w * scale)
            img = cv2.resize(raw_frame, (display_w, DISPLAY_H))

            for det in frame_dets:
                x1, y1, x2, y2 = [int(c * scale) for c in det["bbox"]]
                tid = det["track_id"]
                color = (0, 220, 0) if tid == selected_tid else (200, 200, 200)
                thickness = 3 if tid == selected_tid else 1
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # Instruction overlay
            t = int(candidate_frames[frame_idx] / fps)
            info = f"Frame {frame_idx + 1}/{total}  t={t // 60}:{t % 60:02d}"
            if selected_tid is not None:
                info += f"  | Selected: track {selected_tid}"
            cv2.rectangle(img, (0, 0), (display_w, 28), (0, 0, 0), -1)
            cv2.putText(img, info, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            hint = "Left/Right: change frame  |  Click: select player  |  Enter: confirm  |  Esc: cancel"
            cv2.rectangle(img, (0, DISPLAY_H - 28), (display_w, DISPLAY_H), (0, 0, 0), -1)
            cv2.putText(img, hint, (6, DISPLAY_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            return img, scale

        def on_click(event, x, y, flags, param):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            fi = state["frame_idx"]
            frame_dets = dets_by_frame[candidate_frames[fi]]
            scale = param["scale"]
            for det in frame_dets:
                bx1, by1, bx2, by2 = [int(c * scale) for c in det["bbox"]]
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    state["selected_tid"] = det["track_id"]
                    break

        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, int(DISPLAY_H * det_data["width"] / det_data["height"]), DISPLAY_H + 30)

        print(f"  Interactive picker open — navigate frames with Left/Right, click your player, Enter to confirm.")

        scale_ref = {"scale": 1.0}
        cv2.setMouseCallback(WIN, on_click, scale_ref)

        raw_cache = {}

        while not state["confirmed"] and not state["cancelled"]:
            fi = state["frame_idx"]
            fn = candidate_frames[fi]

            if fn not in raw_cache:
                raw_cache[fn] = reader.read_frame(fn)
            raw = raw_cache[fn]
            if raw is None:
                state["frame_idx"] = (fi + 1) % len(candidate_frames)
                continue

            img, scale = _render(raw, dets_by_frame[fn], state["selected_tid"], fi, len(candidate_frames))
            scale_ref["scale"] = scale
            cv2.imshow(WIN, img)

            key = cv2.waitKey(50) & 0xFF
            if key in (81, 2):  # Left arrow
                state["frame_idx"] = (fi - 1) % len(candidate_frames)
            elif key in (83, 3):  # Right arrow
                state["frame_idx"] = (fi + 1) % len(candidate_frames)
            elif key == 13 and state["selected_tid"] is not None:  # Enter
                state["confirmed"] = True
            elif key == 27:  # Esc
                state["cancelled"] = True

        reader.close()
        cv2.destroyAllWindows()

        if state["confirmed"] and state["selected_tid"] is not None:
            print(f"  Player selected: track ID {state['selected_tid']}")
            return [state["selected_tid"]]
        return []

    except Exception as e:
        print(f"  Interactive selection failed: {e}")
        print("  Continuing without target identification.")
        return []
