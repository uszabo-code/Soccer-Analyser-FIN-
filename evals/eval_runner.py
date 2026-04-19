"""
eval_runner.py — Score Stage 2 output against annotated ground truth.

Usage:
    python evals/eval_runner.py \
        --fixture evals/fixtures/clip_001 \
        --video /path/to/game.mp4 \
        --output /tmp/eval_run/

Exit codes:
    0 = PASS  (track_f1 >= 0.80 AND frames_covered >= 0.70)
    1 = FAIL  (gate not met)
    2 = DRIFT (detections.json SHA-256 doesn't match ground_truth — re-annotate)

Data flow:
    fixture/ground_truth.json ──┐
    fixture/detections.json  ───┼──▶  SHA-256 check
                                │          │ mismatch → exit 2
                                │          │ match ↓
                                └──▶  identify.run(detections.json, target_jersey)
                                           │
                                     player_identity.json
                                           │
                                     compute metrics
                                           │
                                     print table → exit 0/1
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pipeline.identify as identify_stage

PASS_F1 = 0.80
PASS_COVERAGE = 0.70


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_ground_truth(fixture_dir: Path) -> dict:
    """Load and validate ground_truth.json. Raises ValueError on schema issues."""
    gt_path = fixture_dir / "ground_truth.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"ground_truth.json not found in {fixture_dir}")
    with open(gt_path) as f:
        gt = json.load(f)

    required = ["clip_name", "target_jersey", "target_track_ids",
                "detections_sha256", "min_frames_visible"]
    missing = [k for k in required if k not in gt]
    if missing:
        raise ValueError(f"ground_truth.json missing required fields: {missing}")

    if not isinstance(gt["target_track_ids"], list):
        raise ValueError("target_track_ids must be a list")

    if len(gt["target_track_ids"]) == 0:
        raise ValueError(
            "target_track_ids is empty — fixture not annotated yet.\n"
            "Run create_fixture.py and fill in ground_truth.json first."
        )
    return gt


def count_merged_frames(track_ids: list, detections_path: str) -> int:
    """Count unique frame numbers across all track_ids in detections.json.

    detections.json stores person_detections as a flat list:
    [{"frame_num": N, "track_id": T, "bbox": [...], ...}, ...]
    """
    with open(detections_path) as f:
        det_data = json.load(f)
    id_set = set(track_ids)
    seen: set = set()
    for det in det_data.get("person_detections", []):
        if det.get("track_id") in id_set:
            seen.add(det["frame_num"])
    return len(seen)


def run_eval(fixture_dir: str, video_path: str, output_dir: str) -> int:
    """
    Run the eval. Returns exit code: 0=PASS, 1=FAIL, 2=DRIFT.
    """
    fixture_path = Path(fixture_dir)
    detections_path = fixture_path / "detections.json"

    if not detections_path.exists():
        print(f"ERROR: detections.json not found in {fixture_dir}", file=sys.stderr)
        return 1

    # ── Step 1: Load and validate ground truth ────────────────────────────────
    try:
        gt = load_ground_truth(fixture_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    clip_name = gt["clip_name"]
    print(f"\n=== Eval: {clip_name} ===")

    # ── Step 2: SHA-256 drift check ───────────────────────────────────────────
    actual_hash = sha256_file(str(detections_path))
    expected_hash = gt["detections_sha256"]
    if actual_hash != expected_hash:
        print(
            f"\n[DRIFT] detections.json has changed since annotation.\n"
            f"  Expected: {expected_hash[:16]}…\n"
            f"  Actual:   {actual_hash[:16]}…\n"
            f"\n  Track IDs in ground_truth.json may be stale.\n"
            f"  Re-run create_fixture.py and re-annotate ground_truth.json.",
            file=sys.stderr,
        )
        return 2

    # ── Step 3: Run Stage 2 on cached detections ──────────────────────────────
    try:
        identity_path = identify_stage.run(
            video_path=video_path,
            detections_path=str(detections_path),
            output_dir=output_dir,
            target_jersey=gt["target_jersey"],
        )
    except Exception as e:
        print(f"ERROR: identify.run() failed: {e}", file=sys.stderr)
        return 1

    # identify.run() always writes to {output_dir}/player_identity.json
    expected_identity = os.path.join(output_dir, "player_identity.json")
    if identity_path != expected_identity and os.path.exists(identity_path):
        import shutil
        shutil.copy(identity_path, expected_identity)
    if not os.path.exists(expected_identity):
        print(f"ERROR: player_identity.json not found at {expected_identity}",
              file=sys.stderr)
        return 1

    with open(expected_identity) as f:
        identity = json.load(f)
    predicted_ids = set(identity.get("target_track_ids", []))

    # ── Step 4: Compute metrics ───────────────────────────────────────────────
    expected_ids = set(gt["target_track_ids"])
    intersection = predicted_ids & expected_ids

    # Precision: what fraction of predicted IDs are correct?
    if len(predicted_ids) == 0:
        precision = 1.0   # no predictions → technically no false positives
    else:
        precision = len(intersection) / len(predicted_ids)

    # Recall: what fraction of expected IDs were found?
    if len(expected_ids) == 0:
        recall = 1.0
    else:
        recall = len(intersection) / len(expected_ids)

    # F1
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    # Frames covered
    merged_frames = count_merged_frames(list(predicted_ids), str(detections_path))
    min_frames = max(gt["min_frames_visible"], 1)
    frames_covered = min(merged_frames / min_frames, 1.0)

    # Additional diagnostics
    # ocr_seeds_found: count of tracks that were initially identified by OCR.
    # player_identity.json doesn't expose this separately — we approximate by counting
    # tracks in the result that had a jersey match (target_track_ids before sweep).
    # The full sweep may add more tracks beyond the OCR seeds.
    # TODO: expose ocr_seed_track_ids in player_identity.json for a precise count.
    ocr_seeds_found = len(identity.get("target_track_ids", []))
    overlap_rejected = identity.get("overlap_rejected_count", 0)  # 0 if not exposed

    # ── Step 5: Print results ─────────────────────────────────────────────────
    missing_ids = expected_ids - predicted_ids
    extra_ids   = predicted_ids - expected_ids

    def sym(val, gate):
        return "✓" if val >= gate else "✗"

    print(f"  track_precision : {precision:.2f}  "
          f"{'✓' if precision >= PASS_F1 else '✗'}"
          + (f"  (extra: {sorted(extra_ids)})" if extra_ids else ""))
    print(f"  track_recall    : {recall:.2f}  "
          f"{'✓' if recall >= PASS_F1 else '✗'}"
          + (f"  (missing: {sorted(missing_ids)})" if missing_ids else ""))
    print(f"  track_f1        : {f1:.2f}  {sym(f1, PASS_F1)}")
    print(f"  frames_covered  : {frames_covered:.2f}  {sym(frames_covered, PASS_COVERAGE)}"
          f"  ({merged_frames}/{min_frames} frames)")
    print(f"  ocr_seeds_found : {ocr_seeds_found}")
    print(f"  overlap_rejected: {overlap_rejected}")
    print()

    # ── Step 6: Gate ──────────────────────────────────────────────────────────
    passed = f1 >= PASS_F1 and frames_covered >= PASS_COVERAGE
    if passed:
        print(f"PASS  (f1={f1:.2f} ≥ {PASS_F1}, coverage={frames_covered:.2f} ≥ {PASS_COVERAGE})")
        return 0
    else:
        reasons = []
        if f1 < PASS_F1:
            if precision < PASS_F1:
                reasons.append(f"precision={precision:.2f} — wrong tracks being merged")
            if recall < PASS_F1:
                reasons.append(f"recall={recall:.2f} — target tracks being missed")
        if frames_covered < PASS_COVERAGE:
            reasons.append(f"frames_covered={frames_covered:.2f} — player largely untracked")
        print(f"FAIL  {'; '.join(reasons)}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Score Stage 2 output against annotated ground truth."
    )
    parser.add_argument("--fixture", required=True,
                        help="Path to fixture directory (contains ground_truth.json)")
    parser.add_argument("--video", required=True,
                        help="Path to source video (needed by identify.run())")
    parser.add_argument("--output", required=True,
                        help="Output directory for temporary Stage 2 output")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    exit_code = run_eval(args.fixture, args.video, args.output)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
