"""
Stage 1b: Post-process track merging.

BoT-SORT significantly reduces fragmentation, but some ID switches still occur.
This stage merges residual fragments by matching:
  1. Same team color assignment
  2. Temporally non-overlapping (one track ends before the other starts)
  3. Spatially plausible last/first position (within walking distance for gap duration)

Input:  detections.json + player_identity.json (from Stage 1 + Stage 2)
Output: enriched detections.json with merged tracks (in-place) and a
        track_merge_map.json showing which track IDs were merged into which.
"""

import json
import math
import os
from collections import defaultdict


# Max gap in video-frame units to attempt merging.
# frame_num in detections.json is the raw video frame number (not processed-frame index).
# 900 = 30s at 30fps — covers set pieces, camera cuts, brief off-screen exits.
# Previously 90 (3s), which was tuned for low-fragmentation BoT-SORT output.
# With football_yolov8 + BoT-SORT we still see ~140 fragments/player, so extend to 30s.
MAX_GAP_FRAMES = 900

# Max pixel distance (at max gap). Scales linearly for shorter gaps via allowed_dist formula.
# At 30s gap: ~1500px allowed. At 3s gap: ~150px. Guards against cross-player false merges.
MAX_SPATIAL_DISTANCE_PX = 1500


def _get_track_endpoints(person_detections):
    """Return {track_id: {'first_frame', 'last_frame', 'first_pos', 'last_pos', 'detections'}}."""
    tracks = defaultdict(list)
    for det in person_detections:
        tid = det.get("track_id", -1)
        if tid < 0:
            continue
        tracks[tid].append(det)

    endpoints = {}
    for tid, dets in tracks.items():
        dets_sorted = sorted(dets, key=lambda d: d["frame_num"])
        first = dets_sorted[0]
        last = dets_sorted[-1]

        def center(bbox):
            x1, y1, x2, y2 = bbox
            return ((x1 + x2) / 2, (y1 + y2) / 2)

        endpoints[tid] = {
            "first_frame": first["frame_num"],
            "last_frame": last["frame_num"],
            "first_pos": center(first["bbox"]),
            "last_pos": center(last["bbox"]),
            "count": len(dets_sorted),
        }
    return endpoints, tracks


def _euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _build_color_map(player_identity_path):
    """Return {track_id: team_color} from player_identity.json."""
    if not os.path.exists(player_identity_path):
        return {}
    with open(player_identity_path) as f:
        identity = json.load(f)

    color_map = {}
    for player in identity.get("players", []):
        color = player.get("jersey_color", "unknown")
        for tid in player.get("track_ids", []):
            color_map[tid] = color
    return color_map


def find_merge_candidates(endpoints, color_map):
    """
    Find pairs of tracks to merge.

    Returns list of (track_a, track_b) where track_a ends before track_b starts,
    they share the same color, and are spatially plausible.
    """
    track_ids = sorted(endpoints.keys())
    candidates = []

    for i, tid_a in enumerate(track_ids):
        ep_a = endpoints[tid_a]
        color_a = color_map.get(tid_a, "unknown")

        for tid_b in track_ids[i + 1:]:
            ep_b = endpoints[tid_b]
            color_b = color_map.get(tid_b, "unknown")

            # Must share the same jersey color
            if color_a != color_b or color_a == "unknown":
                continue

            # Determine temporal order — a must end before b starts (or vice versa)
            if ep_a["last_frame"] < ep_b["first_frame"]:
                earlier, later = ep_a, ep_b
                tid_earlier, tid_later = tid_a, tid_b
            elif ep_b["last_frame"] < ep_a["first_frame"]:
                earlier, later = ep_b, ep_a
                tid_earlier, tid_later = tid_b, tid_a
            else:
                continue  # Tracks overlap in time — same player can't be in two places

            gap_frames = later["first_frame"] - earlier["last_frame"]
            if gap_frames > MAX_GAP_FRAMES:
                continue  # Too long a gap to reliably merge

            # Check spatial plausibility
            dist = _euclidean(earlier["last_pos"], later["first_pos"])
            # Allow more distance proportional to gap length
            allowed_dist = MAX_SPATIAL_DISTANCE_PX * (gap_frames / MAX_GAP_FRAMES)
            allowed_dist = max(MAX_SPATIAL_DISTANCE_PX * 0.3, allowed_dist)

            if dist <= allowed_dist:
                candidates.append((tid_earlier, tid_later, gap_frames, dist))

    return candidates


def _build_merge_map(candidates, endpoints):
    """
    Build a canonical merge map: {surviving_track_id: [merged_track_ids]}.

    Uses union-find to handle chains (A→B→C all merge into A).
    """
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return
        # Keep the track with earlier first_frame as the root
        if endpoints[px]["first_frame"] <= endpoints[py]["first_frame"]:
            parent[py] = px
        else:
            parent[px] = py

    for tid_a, tid_b, _, _ in candidates:
        union(tid_a, tid_b)

    # Group by canonical root
    groups = defaultdict(list)
    all_ids = set()
    for tid_a, tid_b, _, _ in candidates:
        all_ids.update([tid_a, tid_b])
    for tid in all_ids:
        groups[find(tid)].append(tid)

    return {root: sorted(members) for root, members in groups.items() if len(members) > 1}


def apply_merge_map(person_detections, merge_map):
    """
    Rewrite track_ids in detections so merged tracks all have the canonical (surviving) ID.
    Returns modified detections list.
    """
    # Build reverse map: old_id → canonical_id
    reverse = {}
    for canonical, members in merge_map.items():
        for tid in members:
            if tid != canonical:
                reverse[tid] = canonical

    if not reverse:
        return person_detections

    merged = []
    for det in person_detections:
        d = dict(det)
        if d.get("track_id") in reverse:
            d["track_id"] = reverse[d["track_id"]]
        merged.append(d)
    return merged


def run(detections_path: str, player_identity_path: str, output_dir: str):
    """
    Main entry point for Stage 1b.

    Reads detections.json, merges fragmented tracks, writes updated detections.json
    and track_merge_map.json.
    """
    print(f"\n[Stage 1b] Post-Process Track Merging")

    with open(detections_path) as f:
        det_data = json.load(f)

    person_detections = det_data["person_detections"]
    color_map = _build_color_map(player_identity_path)

    endpoints, _ = _get_track_endpoints(person_detections)
    print(f"  Input: {len(endpoints)} unique tracks, {len(person_detections)} person detections")

    if not color_map:
        print("  No player_identity.json found — skipping color-based merging")
        print("  (Run Stage 2 first to generate identity data, then re-run Stage 1b)")
        return detections_path

    # Log gap distribution to inform parameter tuning
    all_gaps = []
    for ep_a in endpoints.values():
        for ep_b in endpoints.values():
            if ep_a is ep_b:
                continue
            if ep_a["last_frame"] < ep_b["first_frame"]:
                gap = ep_b["first_frame"] - ep_a["last_frame"]
                if gap > 0:
                    all_gaps.append(gap)
    if all_gaps:
        all_gaps.sort()
        n = len(all_gaps)
        fps = det_data.get("fps", 30.0)
        print(f"  Gap distribution (frames @ {fps:.0f}fps):")
        print(f"    p50={all_gaps[n//2]} ({all_gaps[n//2]/fps:.1f}s)  "
              f"p90={all_gaps[int(n*0.9)]} ({all_gaps[int(n*0.9)]/fps:.1f}s)  "
              f"p99={all_gaps[int(n*0.99)]} ({all_gaps[int(n*0.99)]/fps:.1f}s)  "
              f"max={all_gaps[-1]} ({all_gaps[-1]/fps:.1f}s)")
        print(f"    MAX_GAP_FRAMES={MAX_GAP_FRAMES} covers "
              f"{sum(1 for g in all_gaps if g <= MAX_GAP_FRAMES)/n*100:.1f}% of pairs")

    candidates = find_merge_candidates(endpoints, color_map)
    print(f"  Merge candidates found: {len(candidates)}")

    merge_map = _build_merge_map(candidates, endpoints)
    total_merged = sum(len(v) - 1 for v in merge_map.values())
    print(f"  Merging {total_merged} tracks into {len(merge_map)} canonical tracks")

    if not merge_map:
        print("  No merges needed.")
        return detections_path

    updated_detections = apply_merge_map(person_detections, merge_map)
    det_data["person_detections"] = updated_detections

    # Recount unique tracks after merge
    new_track_ids = set(d["track_id"] for d in updated_detections if d.get("track_id", -1) >= 0)
    print(f"  Track count after merge: {len(new_track_ids)} (was {len(endpoints)})")

    # Save updated detections
    with open(detections_path, "w") as f:
        json.dump(det_data, f, indent=2)

    # Save merge map for audit
    merge_map_path = os.path.join(output_dir, "track_merge_map.json")
    with open(merge_map_path, "w") as f:
        json.dump({
            "merge_map": {str(k): v for k, v in merge_map.items()},
            "candidates": [
                {"track_a": a, "track_b": b, "gap_frames": g, "distance_px": round(d, 1)}
                for a, b, g, d in candidates
            ]
        }, f, indent=2)
    print(f"  Merge map saved to {merge_map_path}")

    return detections_path
