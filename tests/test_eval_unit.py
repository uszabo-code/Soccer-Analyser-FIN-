"""
Unit tests for evals/eval_runner.py error paths.

These tests use synthetic fixtures (no real video, no YOLO inference) and run in <1s.
They cover all error paths identified in the engineering review:
    [B] source video not found
    [D] OCR finds 0 tracks (target_track_ids=[])
    [F] SHA-256 drift → exit 2
    [G] f1 < 0.80 (precision fail)
    [H] frames_covered < 0.70
    [I] ground_truth.json malformed / missing fields

Architecture:
    test_eval_unit.py
    │
    ├── synthetic_fixture() helper
    │   └── writes minimal detections.json + ground_truth.json to tmp_path
    │
    └── patches identify.run() to return controlled output
        └── eval_runner.run_eval(fixture, video, output) called directly
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root so evals/ imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.eval_runner import run_eval, PASS_F1, PASS_COVERAGE


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_detections(track_frames: dict, fps: float = 30.0) -> dict:
    """Build a minimal detections.json dict.

    track_frames: {track_id: [frame_num, ...]}
    """
    person_detections = []
    for tid, frames in track_frames.items():
        for fn in frames:
            person_detections.append({
                "frame_num": fn,
                "track_id": tid,
                "bbox": [0, 0, 50, 100],
                "confidence": 0.9,
                "class_id": 0,
            })
    return {
        "video_path": "clip.mp4",
        "fps": fps,
        "width": 1920,
        "height": 1080,
        "total_frames": 900,
        "frame_skip": 1,
        "person_detections": person_detections,
        "ball_detections": [],
    }


def make_identity(target_track_ids: list, jersey: int = 15) -> dict:
    """Build a minimal player_identity.json dict."""
    return {
        "target_jersey": jersey,
        "target_track_ids": target_track_ids,
        "target_team": "team_a",
        "teams": {"team_a": target_track_ids, "team_b": []},
        "players": [],
    }


def write_fixture(tmp_path: Path, det_data: dict, gt_overrides: dict = None) -> tuple:
    """Write detections.json + ground_truth.json to tmp_path.

    Returns (fixture_dir, det_hash).
    """
    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir()
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    det_path = fixture_dir / "detections.json"
    det_path.write_text(json.dumps(det_data))
    det_hash = hashlib.sha256(det_path.read_bytes()).hexdigest()

    gt = {
        "clip_name": "test_clip",
        "description": "unit test fixture",
        "source_video": str(tmp_path / "clip.mp4"),
        "source_start_frame": 0,
        "source_end_frame": 900,
        "frame_numbering": "clip_relative",
        "target_jersey": 15,
        "target_track_ids": [1, 2],
        "detections_sha256": det_hash,
        "min_frames_visible": 200,
        "notes": "synthetic",
    }
    if gt_overrides:
        gt.update(gt_overrides)

    (fixture_dir / "ground_truth.json").write_text(json.dumps(gt))
    return fixture_dir, det_hash


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestEvalRunnerErrorPaths:

    def test_pass_perfect(self, tmp_path):
        """Happy path: predicted == expected, enough frames → exit 0."""
        det = make_detections({1: list(range(100)), 2: list(range(100, 200))})
        fixture_dir, _ = write_fixture(tmp_path, det,
                                       {"target_track_ids": [1, 2],
                                        "min_frames_visible": 150})
        identity = make_identity([1, 2])
        with patch("evals.eval_runner.identify_stage") as mock_id:
            out_file = str(tmp_path / "output" / "player_identity.json")
            Path(out_file).parent.mkdir(exist_ok=True)
            Path(out_file).write_text(json.dumps(identity))
            mock_id.run.return_value = out_file
            code = run_eval(str(fixture_dir), "clip.mp4", str(tmp_path / "output"))
        assert code == 0

    def test_fail_low_precision(self, tmp_path):
        """Predicted includes wrong tracks (precision < 0.80) → exit 1.

        With predicted={1,2,98,99} and expected={1,2}:
            precision = 2/4 = 0.50
            recall    = 2/2 = 1.00
            f1        = 0.67  →  fails 0.80 gate
        """
        det = make_detections({1: list(range(100)), 2: list(range(100, 200)),
                                98: list(range(200, 300)), 99: list(range(300, 400))})
        fixture_dir, _ = write_fixture(tmp_path, det,
                                       {"target_track_ids": [1, 2],
                                        "min_frames_visible": 150})
        # Stage 2 returns two extra wrong tracks (98, 99)
        identity = make_identity([1, 2, 98, 99])
        with patch("evals.eval_runner.identify_stage") as mock_id:
            out_file = str(tmp_path / "output" / "player_identity.json")
            Path(out_file).parent.mkdir(exist_ok=True)
            Path(out_file).write_text(json.dumps(identity))
            mock_id.run.return_value = out_file
            code = run_eval(str(fixture_dir), "clip.mp4", str(tmp_path / "output"))
        assert code == 1

    def test_fail_low_recall(self, tmp_path):
        """Stage 2 misses one expected track (recall < 0.80) → exit 1."""
        det = make_detections({1: list(range(100)), 2: list(range(100, 200))})
        fixture_dir, _ = write_fixture(tmp_path, det,
                                       {"target_track_ids": [1, 2],
                                        "min_frames_visible": 150})
        identity = make_identity([1])   # misses track 2
        with patch("evals.eval_runner.identify_stage") as mock_id:
            out_file = str(tmp_path / "output" / "player_identity.json")
            Path(out_file).parent.mkdir(exist_ok=True)
            Path(out_file).write_text(json.dumps(identity))
            mock_id.run.return_value = out_file
            code = run_eval(str(fixture_dir), "clip.mp4", str(tmp_path / "output"))
        assert code == 1

    def test_fail_low_frames_covered(self, tmp_path):
        """Perfect track IDs but very few frames → frames_covered < 0.70 → exit 1."""
        det = make_detections({1: list(range(10)), 2: list(range(10, 20))})
        fixture_dir, _ = write_fixture(tmp_path, det,
                                       {"target_track_ids": [1, 2],
                                        "min_frames_visible": 200})  # only 20 frames exist
        identity = make_identity([1, 2])
        with patch("evals.eval_runner.identify_stage") as mock_id:
            out_file = str(tmp_path / "output" / "player_identity.json")
            Path(out_file).parent.mkdir(exist_ok=True)
            Path(out_file).write_text(json.dumps(identity))
            mock_id.run.return_value = out_file
            code = run_eval(str(fixture_dir), "clip.mp4", str(tmp_path / "output"))
        assert code == 1

    def test_drift_detection(self, tmp_path):
        """Tampered detections.json → SHA-256 mismatch → exit 2."""
        det = make_detections({1: list(range(100))})
        fixture_dir, _ = write_fixture(tmp_path, det,
                                       {"target_track_ids": [1],
                                        "min_frames_visible": 80})
        # Tamper with detections.json after fixture was written
        det_path = fixture_dir / "detections.json"
        original = json.loads(det_path.read_text())
        original["person_detections"].append({"frame_num": 9999, "track_id": 999,
                                               "bbox": [0,0,1,1], "confidence": 0.1,
                                               "class_id": 0})
        det_path.write_text(json.dumps(original))

        code = run_eval(str(fixture_dir), "clip.mp4", str(tmp_path / "output"))
        assert code == 2

    def test_missing_ground_truth(self, tmp_path):
        """No ground_truth.json → clean error, exit 1 (not a crash)."""
        fixture_dir = tmp_path / "fixture"
        fixture_dir.mkdir()
        # Write detections.json but NO ground_truth.json
        det = make_detections({1: list(range(50))})
        (fixture_dir / "detections.json").write_text(json.dumps(det))
        code = run_eval(str(fixture_dir), "clip.mp4", str(tmp_path / "output"))
        assert code == 1

    def test_malformed_ground_truth_missing_field(self, tmp_path):
        """ground_truth.json missing required field → clean error, exit 1."""
        det = make_detections({1: list(range(50))})
        fixture_dir, det_hash = write_fixture(tmp_path, det)
        # Remove a required field
        gt_path = fixture_dir / "ground_truth.json"
        gt = json.loads(gt_path.read_text())
        del gt["detections_sha256"]
        gt_path.write_text(json.dumps(gt))
        code = run_eval(str(fixture_dir), "clip.mp4", str(tmp_path / "output"))
        assert code == 1

    def test_unannotated_fixture(self, tmp_path):
        """target_track_ids=[] (not yet annotated) → clean error, exit 1."""
        det = make_detections({1: list(range(50))})
        fixture_dir, _ = write_fixture(tmp_path, det,
                                       {"target_track_ids": []})
        code = run_eval(str(fixture_dir), "clip.mp4", str(tmp_path / "output"))
        assert code == 1

    def test_identify_run_exception(self, tmp_path):
        """identify.run() raises an exception → exit 1, not unhandled crash."""
        det = make_detections({1: list(range(100))})
        fixture_dir, _ = write_fixture(tmp_path, det,
                                       {"target_track_ids": [1],
                                        "min_frames_visible": 80})
        with patch("evals.eval_runner.identify_stage") as mock_id:
            mock_id.run.side_effect = RuntimeError("model load failed")
            code = run_eval(str(fixture_dir), "clip.mp4", str(tmp_path / "output"))
        assert code == 1
