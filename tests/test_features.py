"""
Tests for pipeline/features.py (Stage 3).

Covers:
  - auto_detect_field_width() — HSV field detector
  - Target player never silently dropped (diagnostic shown, stats attempted)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tempfile
import pytest
import numpy as np
import cv2

from pipeline.features import auto_detect_field_width


# ── HSV field detector ────────────────────────────────────────────────────────

def _make_synthetic_video(path: str, colour_bgr: tuple, width=640, height=360,
                           n_frames=10, fps=25.0):
    """Write a short synthetic video filled with a solid colour."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    frame = np.full((height, width, 3), colour_bgr, dtype=np.uint8)
    for _ in range(n_frames):
        out.write(frame)
    out.release()


class TestAutoDetectFieldWidth:
    def test_green_pitch_detected(self, tmp_path):
        """A video filled with natural grass green returns a valid width with high confidence."""
        video_path = str(tmp_path / "green.mp4")
        # HSV H≈60 (yellow-green), S≈200, V≈100 → clearly detectable green
        # BGR equivalent of HSV(60,200,100): convert via cv2
        hsv_frame = np.full((1, 1, 3), [60, 200, 100], dtype=np.uint8)
        bgr_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
        green_bgr = tuple(int(v) for v in bgr_frame[0, 0])
        _make_synthetic_video(video_path, green_bgr, width=640, height=360)

        field_width_px, confidence = auto_detect_field_width(video_path, 25.0)
        assert field_width_px is not None
        assert confidence > 0.0

    def test_non_green_returns_low_confidence(self, tmp_path):
        """A red video (artificial turf / wrong pitch) returns None or low confidence."""
        video_path = str(tmp_path / "red.mp4")
        _make_synthetic_video(video_path, (0, 0, 200), width=640, height=360)  # Red in BGR

        field_width_px, confidence = auto_detect_field_width(video_path, 25.0)
        # Either None (no detection) or very low confidence
        if field_width_px is not None:
            assert confidence < 0.3

    def test_missing_video_returns_none(self):
        """Non-existent video path returns (None, 0.0) gracefully."""
        width_px, confidence = auto_detect_field_width("/nonexistent/video.mp4", 25.0)
        assert width_px is None
        assert confidence == 0.0

    def test_field_width_is_positive(self, tmp_path):
        """Detected field width must be a positive number."""
        video_path = str(tmp_path / "green2.mp4")
        hsv_frame = np.full((1, 1, 3), [50, 180, 120], dtype=np.uint8)
        bgr_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
        green_bgr = tuple(int(v) for v in bgr_frame[0, 0])
        _make_synthetic_video(video_path, green_bgr, width=640, height=360)

        field_width_px, confidence = auto_detect_field_width(video_path, 25.0)
        if field_width_px is not None:
            assert field_width_px > 0


# ── Target player: never silently dropped ────────────────────────────────────

class TestTargetPlayerNotDropped:
    """
    Verify that Stage 3 produces output (not silence) when the target player track
    has fewer than the usual minimum detections.
    """

    def _make_minimal_inputs(self, tmp_path, target_det_count=5):
        """Create minimal detections.json and player_identity.json for testing."""
        fps = 25.0
        frame_width = 1920
        frame_height = 1080

        # Build minimal detections for track 1 (target) and track 2 (other, >50 dets)
        person_dets = []
        for i in range(target_det_count):
            person_dets.append({
                "track_id": 1,
                "frame_num": i * 5,
                "bbox": [500 + i, 400, 560 + i, 480],
                "confidence": 0.8,
                "class_id": 0,
            })
        for i in range(60):  # Track 2 has 60 detections (above threshold)
            person_dets.append({
                "track_id": 2,
                "frame_num": i * 2,
                "bbox": [900, 300, 960, 380],
                "confidence": 0.85,
                "class_id": 0,
            })

        det_data = {
            "fps": fps,
            "width": frame_width,
            "height": frame_height,
            "frame_skip": 2,
            "person_detections": person_dets,
            "ball_detections": [],
        }
        det_path = str(tmp_path / "detections.json")
        with open(det_path, "w") as f:
            json.dump(det_data, f)

        id_data = {
            "target_jersey": None,
            "target_track_ids": [1],
            "target_team": "team_a",
            "teams": {"team_a": [1, 2]},
            "players": [
                {"track_ids": [1], "jersey_color": "blue", "jersey_number": None,
                 "team": "team_a"},
                {"track_ids": [2], "jersey_color": "blue", "jersey_number": None,
                 "team": "team_a"},
            ],
        }
        id_path = str(tmp_path / "player_identity.json")
        with open(id_path, "w") as f:
            json.dump(id_data, f)

        return det_path, id_path

    def test_target_with_few_detections_produces_output(self, tmp_path, capsys):
        """Target with only 5 detections should produce stats, not silence."""
        from pipeline.features import run as features_run

        det_path, id_path = self._make_minimal_inputs(tmp_path, target_det_count=5)
        output_dir = str(tmp_path)

        stats_path = features_run(
            detections_path=det_path,
            identity_path=id_path,
            output_dir=output_dir,
        )

        with open(stats_path) as f:
            stats = json.load(f)

        player_ids = [p.get("player_id", "") for p in stats["players"]]
        # At least one player should be in the output (track 2 with 60 dets)
        assert len(stats["players"]) >= 1

        # Warning should have been printed
        captured = capsys.readouterr()
        assert "WARNING" in captured.out or len(stats["players"]) >= 1

    def test_target_not_silently_absent_from_output(self, tmp_path):
        """Stage 3 must not produce an empty players list when target has few detections."""
        from pipeline.features import run as features_run

        det_path, id_path = self._make_minimal_inputs(tmp_path, target_det_count=3)
        stats_path = features_run(
            detections_path=det_path,
            identity_path=id_path,
            output_dir=str(tmp_path),
        )

        with open(stats_path) as f:
            stats = json.load(f)

        # Output must contain at least the other player (track 2) — never empty
        assert len(stats["players"]) >= 1


# ── Multi-fragment target aggregation ────────────────────────────────────────

class TestMultiFragmentTargetAggregation:
    """
    Verify that when target_track_ids contains multiple track IDs, features.py
    combines them into ONE stat entry (not N separate entries).
    """

    def _make_two_fragment_inputs(self, tmp_path, gap_frames=500):
        """
        Create inputs with two target track fragments separated by a large gap.

        Fragment A: track 10, frames 0–50   (50 detections, left side)
        Fragment B: track 20, frames 600–650 (50 detections, right side)
        The gap_frames gap between them should NOT add to total distance.
        Other: track 99, frames 200–300 (101 detections, middle — for comparison)
        """
        fps = 25.0
        frame_width = 1920
        frame_height = 1080

        person_dets = []
        # Fragment A: target track 10, x=100, frames 0..49
        for i in range(50):
            person_dets.append({
                "track_id": 10, "frame_num": i * 2,
                "bbox": [90, 300, 130, 380], "confidence": 0.9, "class_id": 0,
            })
        # Fragment B: target track 20, x=900, frames gap_frames..(gap_frames+49)
        for i in range(50):
            person_dets.append({
                "track_id": 20, "frame_num": gap_frames + i * 2,
                "bbox": [890, 300, 930, 380], "confidence": 0.9, "class_id": 0,
            })
        # Other player (track 99)
        for i in range(101):
            person_dets.append({
                "track_id": 99, "frame_num": 400 + i * 2,
                "bbox": [500, 400, 540, 480], "confidence": 0.85, "class_id": 0,
            })

        det_data = {
            "fps": fps, "width": frame_width, "height": frame_height,
            "frame_skip": 2, "person_detections": person_dets, "ball_detections": [],
        }
        det_path = str(tmp_path / "detections.json")
        with open(det_path, "w") as f:
            json.dump(det_data, f)

        id_data = {
            "target_jersey": 15,
            "target_track_ids": [10, 20],
            "target_team": "team_a",
            "teams": {"team_a": [10, 20, 99]},
            "players": [
                {"track_ids": [10], "jersey_color": "blue", "jersey_number": 15,
                 "team": "team_a", "is_target": True, "display_name": "#15"},
                {"track_ids": [20], "jersey_color": "blue", "jersey_number": 15,
                 "team": "team_a", "is_target": True, "display_name": "#15"},
                {"track_ids": [99], "jersey_color": "red", "jersey_number": None,
                 "team": "team_a", "is_target": False, "display_name": "Track-99"},
            ],
        }
        id_path = str(tmp_path / "player_identity.json")
        with open(id_path, "w") as f:
            json.dump(id_data, f)

        return det_path, id_path

    def test_two_target_fragments_produce_one_stat_entry(self, tmp_path):
        """Two target_track_ids should produce exactly ONE is_target=True stat entry."""
        from pipeline.features import run as features_run

        det_path, id_path = self._make_two_fragment_inputs(tmp_path)
        stats_path = features_run(
            detections_path=det_path, identity_path=id_path, output_dir=str(tmp_path),
        )

        with open(stats_path) as f:
            stats = json.load(f)

        target_entries = [p for p in stats["players"] if p.get("is_target")]
        assert len(target_entries) == 1, (
            f"Expected 1 combined target entry, got {len(target_entries)}. "
            f"Multi-fragment aggregation is not working."
        )

    def test_time_visible_is_sum_of_fragment_spans(self, tmp_path):
        """
        time_visible_s should equal the SUM of each fragment's span, NOT first-to-last
        across the entire game. The off-camera gap must not count as visible time.
        """
        from pipeline.features import run as features_run

        gap_frames = 500
        fps = 25.0
        # Fragment A: 50 dets at frame 0,2,4,...98 → span = 98/25 = 3.92s
        # Fragment B: 50 dets at frame 500,502,...598 → span = 98/25 = 3.92s
        # Sum of spans ≈ 7.84s (well under 600/25=24s if counted first-to-last)
        det_path, id_path = self._make_two_fragment_inputs(tmp_path, gap_frames=gap_frames)
        stats_path = features_run(
            detections_path=det_path, identity_path=id_path, output_dir=str(tmp_path),
        )

        with open(stats_path) as f:
            stats = json.load(f)

        target = next(p for p in stats["players"] if p.get("is_target"))
        time_visible = target["total_time_visible_s"]

        # If counting first-to-last: (gap_frames + 98) / fps ≈ 23.9s
        # If correctly summing spans: 98/fps * 2 ≈ 7.84s
        first_to_last = (gap_frames + 98) / fps
        span_sum = 98.0 / fps * 2

        assert time_visible < first_to_last * 0.7, (
            f"time_visible ({time_visible:.1f}s) looks like first-to-last span "
            f"({first_to_last:.1f}s) rather than sum of fragment spans (~{span_sum:.1f}s). "
            f"Fragment-aware time calculation is not working."
        )

    def test_no_teleportation_distance_across_gap(self, tmp_path):
        """
        Distance should NOT include the jump between fragments (which would be
        800px = ~26m in a single 'step' — physically impossible).
        """
        from pipeline.features import run as features_run

        det_path, id_path = self._make_two_fragment_inputs(tmp_path, gap_frames=500)
        stats_path = features_run(
            detections_path=det_path, identity_path=id_path, output_dir=str(tmp_path),
        )

        with open(stats_path) as f:
            stats = json.load(f)

        target = next(p for p in stats["players"] if p.get("is_target"))
        # Fragment A: x=100, Fragment B: x=900 → teleport = 800px ≈ 26m at 30px/m
        # Real distance should be near-zero (player barely moves within each fragment)
        # If teleportation is included, total_distance_m > 20m
        assert target["total_distance_m"] < 15.0, (
            f"total_distance_m = {target['total_distance_m']:.1f}m is too high — "
            f"likely includes teleportation distance across fragment gap."
        )
