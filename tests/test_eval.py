"""
Integration tests for evals/eval_runner.py — requires real fixtures.

These tests are marked @pytest.mark.slow and are skipped unless:
  - A real fixture exists at evals/fixtures/*/ground_truth.json
  - The ground_truth.json has a valid source_video path
  - clip.mp4 exists (extracted with create_fixture.py)

Run only the unit tests (fast, no video, no YOLO):
    pytest tests/test_eval_unit.py -v

Run everything including integration tests:
    pytest -v -m slow

Skip slow tests (default):
    pytest -v -m "not slow"

Architecture:
    test_eval.py
    │
    ├── test_eval_fixture_exists_and_valid()  [not slow]
    │   └── validates ground_truth.json schema for every fixture
    │       (no video, no identify.run — just schema check)
    │
    └── test_eval_fixture_pass()  [slow — real inference]
        └── dynamically parameterized over evals/fixtures/*/ground_truth.json
            └── calls eval_runner.run_eval() directly (module import, not subprocess)
"""

import glob
import json
import os
import sys
from pathlib import Path

import pytest

# Add project root so evals/ imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.eval_runner import run_eval

# ── Fixture discovery ─────────────────────────────────────────────────────────

FIXTURE_GLOB = str(Path(__file__).parent.parent / "evals" / "fixtures" / "*" / "ground_truth.json")
FIXTURES = sorted(glob.glob(FIXTURE_GLOB))


# ── Schema validation (fast — no inference) ───────────────────────────────────

REQUIRED_GT_FIELDS = [
    "clip_name",
    "target_jersey",
    "target_track_ids",
    "detections_sha256",
    "min_frames_visible",
    "source_video",
]


@pytest.mark.parametrize("gt_path", FIXTURES, ids=[Path(p).parent.name for p in FIXTURES] if FIXTURES else [])
def test_fixture_schema_valid(gt_path):
    """Every ground_truth.json must have required fields and valid types.

    This test runs without any video or model — it's a fast schema guard
    that catches annotation mistakes before a slow eval run.
    """
    with open(gt_path) as f:
        gt = json.load(f)

    missing = [k for k in REQUIRED_GT_FIELDS if k not in gt]
    assert not missing, f"{gt_path}: missing required fields: {missing}"

    assert isinstance(gt["target_track_ids"], list), \
        f"{gt_path}: target_track_ids must be a list"

    assert isinstance(gt["target_jersey"], int), \
        f"{gt_path}: target_jersey must be an int"

    assert isinstance(gt["min_frames_visible"], int), \
        f"{gt_path}: min_frames_visible must be an int"

    assert isinstance(gt["detections_sha256"], str) and len(gt["detections_sha256"]) == 64, \
        f"{gt_path}: detections_sha256 must be a 64-char hex string"

    # Warn if unannotated (not a hard failure — allows partial fixture directories)
    if len(gt["target_track_ids"]) == 0:
        pytest.skip(f"Fixture {Path(gt_path).parent.name} not yet annotated (target_track_ids=[])")

    # Verify detections.json exists alongside ground_truth.json
    det_path = Path(gt_path).parent / "detections.json"
    assert det_path.exists(), f"detections.json not found alongside {gt_path}"


@pytest.mark.parametrize("gt_path", FIXTURES, ids=[Path(p).parent.name for p in FIXTURES] if FIXTURES else [])
def test_fixture_detections_hash_stable(gt_path):
    """detections.json hash must match the recorded detections_sha256.

    Catches accidental Stage 1 re-runs that invalidate ground truth.
    Run this before any annotated eval to detect drift early.
    """
    import hashlib

    with open(gt_path) as f:
        gt = json.load(f)

    if not gt.get("target_track_ids"):
        pytest.skip(f"Fixture {Path(gt_path).parent.name} not yet annotated")

    det_path = Path(gt_path).parent / "detections.json"
    if not det_path.exists():
        pytest.skip(f"detections.json not found — run create_fixture.py first")

    h = hashlib.sha256(det_path.read_bytes()).hexdigest()
    assert h == gt["detections_sha256"], (
        f"[DRIFT] detections.json hash mismatch for {Path(gt_path).parent.name}.\n"
        f"  Expected: {gt['detections_sha256'][:16]}…\n"
        f"  Actual:   {h[:16]}…\n"
        f"  Re-run create_fixture.py and re-annotate ground_truth.json."
    )


# ── Integration tests (slow — real inference) ─────────────────────────────────

def _slow_fixture_ids():
    """IDs for slow parametrize — clip name or 'no-fixtures'."""
    if not FIXTURES:
        return ["no-fixtures"]
    return [Path(p).parent.name for p in FIXTURES]


def _slow_fixture_params():
    """Parameters for slow tests — empty list if no fixtures (test will skip)."""
    return FIXTURES if FIXTURES else [pytest.param(None, marks=pytest.mark.skip(reason="No fixtures found"))]


@pytest.mark.slow
@pytest.mark.parametrize("gt_path", _slow_fixture_params(), ids=_slow_fixture_ids())
def test_eval_fixture_pass(gt_path, tmp_path):
    """Full pipeline eval: Stage 2 must PASS (exit 0) for each annotated fixture.

    Requires:
      - ground_truth.json with target_track_ids filled in
      - detections.json hash matching detections_sha256
      - clip.mp4 extracted alongside the fixture
      - YOLO model weights available

    Skip with: pytest -m "not slow"
    Run with:  pytest -m slow -v
    """
    if gt_path is None:
        pytest.skip("No fixtures found at evals/fixtures/*/ground_truth.json")

    with open(gt_path) as f:
        gt = json.load(f)

    if not gt.get("target_track_ids"):
        pytest.skip(f"Fixture {Path(gt_path).parent.name} not yet annotated (target_track_ids=[])")

    fixture_dir = str(Path(gt_path).parent)
    source_video = gt.get("source_video", "")

    # clip.mp4 is extracted by create_fixture.py alongside the fixture
    clip_path = str(Path(gt_path).parent / "clip.mp4")
    if os.path.exists(clip_path):
        video_path = clip_path
    elif os.path.exists(source_video):
        video_path = source_video
    else:
        pytest.skip(
            f"No video available for {Path(gt_path).parent.name}.\n"
            f"Run: python evals/create_fixture.py --video <game.mp4> "
            f"--start-frame {gt.get('source_start_frame', 0)} "
            f"--end-frame {gt.get('source_end_frame', 1800)} "
            f"--target-jersey {gt.get('target_jersey', '?')} "
            f"--output {fixture_dir}"
        )

    output_dir = str(tmp_path / "eval_output")
    os.makedirs(output_dir, exist_ok=True)

    exit_code = run_eval(fixture_dir, video_path, output_dir)

    assert exit_code == 0, (
        f"Eval FAILED for fixture '{Path(gt_path).parent.name}' "
        f"(exit code {exit_code}). "
        f"Check the printed metrics above for which gate failed."
    )


# ── No-fixture guard ──────────────────────────────────────────────────────────

def test_no_fixtures_warning():
    """Informational: tells the developer how to create fixtures when none exist."""
    if FIXTURES:
        pytest.skip("Fixtures exist — this warning test is not needed")

    pytest.skip(
        "No eval fixtures found at evals/fixtures/*/ground_truth.json.\n\n"
        "To create your first fixture:\n"
        "  python evals/create_fixture.py \\\n"
        "    --video /path/to/game.mp4 \\\n"
        "    --start-frame 0 --end-frame 1800 \\\n"
        "    --target-jersey 15 \\\n"
        "    --output evals/fixtures/clip_001\n\n"
        "Then fill in target_track_ids in evals/fixtures/clip_001/ground_truth.json\n"
        "and run: pytest tests/test_eval.py -m slow -v"
    )
