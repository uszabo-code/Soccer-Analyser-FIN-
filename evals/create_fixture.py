"""
create_fixture.py — Create an annotated eval fixture from a game video.

Usage:
    python evals/create_fixture.py \
        --video /path/to/game.mp4 \
        --start-frame 0 --end-frame 1800 \
        --target-jersey 15 \
        --output evals/fixtures/clip_001

What it does:
    1. Extracts frames start_frame..end_frame → clip.mp4 (NOT committed)
    2. Runs Stage 1 (detect.py) on clip.mp4 → detections.json (committed)
    3. Runs Stage 2 (identify.py) on detections.json → identity_raw.json (NOT committed)
    4. Prints a table of OCR-matched track IDs for --target-jersey
    5. Scaffolds ground_truth.json with all fields pre-filled;
       target_track_ids left as [] for human to fill in

Data flow:
    game.mp4  ──[clip extract]──▶  clip.mp4
                                       │
                                  detect.run()
                                       │
                                  detections.json  (cached, committed to git)
                                       │
                                 identify.run()
                                       │
                                  identity_raw.json  (not committed)
                                       │
                              print OCR-matched track table
                                       │
                              scaffold ground_truth.json  ◀── human fills target_track_ids
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2

# Add project root to path so pipeline imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import pipeline.detect as detect_stage
import pipeline.identify as identify_stage


def extract_clip(source_video: str, start_frame: int, end_frame: int,
                 output_path: str) -> dict:
    """Extract frames [start_frame, end_frame) from source_video to output_path.

    Returns dict with fps, width, height, frame_count.
    """
    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {source_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame >= end_frame:
        raise ValueError(f"start_frame ({start_frame}) must be < end_frame ({end_frame})")
    if start_frame >= total_frames:
        raise ValueError(f"start_frame ({start_frame}) >= total frames in video ({total_frames})")

    end_frame = min(end_frame, total_frames)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        written += 1

    cap.release()
    writer.release()

    if written == 0:
        raise RuntimeError("No frames written — check start/end frame values")

    print(f"  Extracted {written} frames ({written/fps:.1f}s) → {output_path}")
    return {"fps": fps, "width": width, "height": height, "frame_count": written}


def sha256_file(path: str) -> str:
    """Return hex SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Create an annotated eval fixture from a game video."
    )
    parser.add_argument("--video", required=True,
                        help="Path to source game video")
    parser.add_argument("--start-frame", type=int, required=True,
                        help="First frame to include (game-relative)")
    parser.add_argument("--end-frame", type=int, required=True,
                        help="Last frame (exclusive, game-relative)")
    parser.add_argument("--target-jersey", type=int, required=True,
                        help="Jersey number of the target player")
    parser.add_argument("--output", required=True,
                        help="Output fixture directory (e.g. evals/fixtures/clip_001)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_name = out_dir.name
    print(f"\n=== Creating fixture: {clip_name} ===")

    # ── Step 1: Extract clip ──────────────────────────────────────────────────
    print("\n[1/4] Extracting clip…")
    clip_path = str(out_dir / "clip.mp4")
    clip_info = extract_clip(
        args.video, args.start_frame, args.end_frame, clip_path
    )

    # ── Step 2: Run Stage 1 (detect) ─────────────────────────────────────────
    print("\n[2/4] Running Stage 1 (YOLO detection + tracking)…")
    detections_path = detect_stage.run(
        video_path=clip_path,
        output_dir=str(out_dir),
    )
    det_hash = sha256_file(detections_path)
    print(f"  detections.json SHA-256: {det_hash[:16]}…")

    # ── Step 3: Run Stage 2 (identify) ───────────────────────────────────────
    print(f"\n[3/4] Running Stage 2 (jersey OCR, target #{args.target_jersey})…")
    try:
        identity_path = identify_stage.run(
            video_path=clip_path,
            detections_path=detections_path,
            output_dir=str(out_dir),
            target_jersey=args.target_jersey,
        )
        # Rename to identity_raw.json so it's clearly not the canonical output
        raw_path = str(out_dir / "identity_raw.json")
        if identity_path != raw_path and os.path.exists(identity_path):
            os.rename(identity_path, raw_path)
            identity_path = raw_path

        with open(identity_path) as f:
            identity = json.load(f)
        result_track_ids = identity.get("target_track_ids", [])
    except Exception as e:
        print(f"  WARNING: Stage 2 failed: {e}")
        print("  You will need to fill in target_track_ids manually.")
        result_track_ids = []

    # ── Step 4: Print OCR-matched track table ─────────────────────────────────
    print(f"\n[4/4] OCR results for jersey #{args.target_jersey}:")

    with open(detections_path) as f:
        det_data = json.load(f)

    # Count frames per track ID from the flat person_detections list
    track_frames: dict = {}
    for det in det_data.get("person_detections", []):
        tid = det.get("track_id")
        if tid is not None:
            track_frames[tid] = track_frames.get(tid, 0) + 1

    if result_track_ids:
        print(f"\n  {'Track ID':>10}  {'Frames':>8}  {'In Stage 2 result':>18}")
        print(f"  {'─'*10}  {'─'*8}  {'─'*18}")
        for tid in sorted(result_track_ids):
            frames = track_frames.get(tid, "?")
            print(f"  {tid:>10}  {frames:>8}  {'← Stage 2 identified':>18}")
        print()
        print("  ↑ These are Stage 2's OCR-seeded sweep result.")
        print("  Inspect clip.mp4 to verify which IDs actually show the target player.")
    else:
        print("  No OCR-matched tracks found for this jersey number.")
        print("  This could mean:")
        print("    - The jersey number is not visible in this clip segment")
        print("    - OCR confidence was too low (blurry/small digits)")
        print("  You will need to identify the correct track IDs manually by")
        print("  inspecting clip.mp4 and detections.json.")

    # ── Step 5: Compute min_frames_visible from Stage 2 result ───────────────
    min_frames_visible = 0
    if result_track_ids:
        with open(detections_path) as f:
            det_data_full = json.load(f)
        id_set = set(result_track_ids)
        seen_frames: set = set()
        for det in det_data_full.get("person_detections", []):
            if det.get("track_id") in id_set:
                seen_frames.add(det["frame_num"])
        min_frames_visible = len(seen_frames)

    # ── Step 6: Scaffold ground_truth.json ───────────────────────────────────
    gt = {
        "clip_name": clip_name,
        "description": "TODO: describe this clip scenario",
        "source_video": str(Path(args.video).resolve()),
        "source_start_frame": args.start_frame,
        "source_end_frame": args.end_frame,
        "frame_numbering": "clip_relative",
        "target_jersey": args.target_jersey,
        "target_track_ids": [],
        "detections_sha256": det_hash,
        "min_frames_visible": min_frames_visible,
        "notes": "Fill in target_track_ids from the table printed by create_fixture.py",
    }
    gt_path = out_dir / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Fixture scaffolded: {out_dir}/")
    print(f"  detections.json   ← committed to git")
    print(f"  ground_truth.json ← fill in target_track_ids, then commit")
    print(f"  clip.mp4          ← NOT committed (listed in .gitignore)")
    print(f"  identity_raw.json ← NOT committed")
    print()
    print(f"Next: open {gt_path}")
    print(f"  Set target_track_ids to the correct IDs from the table above.")
    print(f"  Then run: python evals/eval_runner.py --fixture {out_dir} "
          f"--video {args.video} --output /tmp/eval_test/")


if __name__ == "__main__":
    main()
