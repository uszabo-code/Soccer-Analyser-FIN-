"""
Tests for pipeline/pdf_report.py — PDF generation (Stage 5b).

Covers:
  - Happy path: real output_v7 data → report.pdf created
  - LLM analysis absent (null values) → PDF still generates
  - Empty sprint_episodes → "No sprints detected" tile, no crash
  - Sparse detections (<50) → scatter fallback
  - Zero target detections → heatmap text fallback
  - Missing ball_proximity_episodes key → page 2 silently omits row
  - Empty improvements list → page 3 placeholder card
"""

import json
import os
import sys
import tempfile
import shutil

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.pdf_report import run, _target_positions_pitch, _compute_work_rate

# ── fixtures ──────────────────────────────────────────────────────────────────

OUTPUT_V7 = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output_v7",
)
HAS_OUTPUT_V7 = all(
    os.path.exists(os.path.join(OUTPUT_V7, f))
    for f in ("player_stats.json", "player_identity.json",
               "analysis.json", "detections.json")
)


def _minimal_stats(sprint_episodes=None, ball_proximity_episodes=None, frames=100):
    """Build a minimal player_stats.json dict for a single target player."""
    if sprint_episodes is None:
        sprint_episodes = [
            {"start_frame": 10, "end_frame": 20, "start_time": "00:00", "end_time": "00:01"},
        ]
    player = {
        "player_id": "track_1",
        "team": "team_a",
        "is_target": True,
        "inferred_position": "Midfielder",
        "total_distance_m": 500.0,
        "avg_speed_mps": 2.5,
        "max_speed_mps": 6.0,
        "sprint_count": len(sprint_episodes),
        "time_in_thirds": {"defensive": 30, "middle": 40, "attacking": 30},
        "sprint_episodes": sprint_episodes,
        "frames_visible": frames,
        "total_time_visible_s": frames / 30.0,
        "heatmap": [[0] * 15 for _ in range(10)],
        "key_moments": [],
    }
    if ball_proximity_episodes is not None:
        player["ball_proximity_episodes"] = ball_proximity_episodes
    return {"players": [player]}


def _minimal_identity(track_ids=(1,), jersey=15):
    return {
        "target_jersey": jersey,
        "target_track_ids": list(track_ids),
        "target_team": "team_a",
        "teams": {"team_a": list(track_ids), "team_b": []},
        "players": {},
    }


def _minimal_analysis(has_llm=True):
    if not has_llm:
        return {
            "player_summary": None,
            "improvements": None,
            "team_strategy": None,
            "target_player": "Player 1",
            "llm_model": None,
        }
    return {
        "player_summary": {
            "summary": "Player showed good movement and positioning throughout.",
            "strengths": ["Good positioning", "High work rate"],
            "areas_to_improve": ["Passing accuracy", "Shot selection"],
        },
        "improvements": {
            "suggestions": [
                {
                    "timestamp_start": "02:10",
                    "timestamp_end": "02:30",
                    "description": "Player lost the ball under pressure.",
                    "recommendation": "Improve first touch control.",
                    "reasoning": "Ball was lost 3 times in this period.",
                },
            ]
        },
        "team_strategy": {"observations": []},
        "target_player": "Player 1",
        "llm_model": "claude-test",
    }


def _minimal_detections(track_ids=(1,), n_dets=60, width=1920, height=1080):
    dets = []
    for i in range(n_dets):
        dets.append({
            "frame_num": i * 2,
            "track_id": list(track_ids)[i % len(track_ids)],
            "bbox": [400.0, 300.0, 450.0, 380.0],
            "confidence": 0.9,
            "class_id": 0,
        })
    return {
        "video_path": "test.mp4",
        "fps": 30.0,
        "width": width,
        "height": height,
        "total_frames": 1800,
        "frame_skip": 2,
        "person_detections": dets,
        "ball_detections": [],
    }


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


# ── helper: write all 4 input files into a tempdir ────────────────────────────

def _setup_dir(tmpdir, stats=None, identity=None, analysis=None, detections=None):
    track_ids = (1,)
    _write_json(os.path.join(tmpdir, "player_stats.json"),
                stats or _minimal_stats())
    _write_json(os.path.join(tmpdir, "player_identity.json"),
                identity or _minimal_identity(track_ids))
    _write_json(os.path.join(tmpdir, "analysis.json"),
                analysis or _minimal_analysis())
    _write_json(os.path.join(tmpdir, "detections.json"),
                detections or _minimal_detections(track_ids))


# ── tests ─────────────────────────────────────────────────────────────────────

class TestPdfReportHappyPath:

    @pytest.mark.skipif(not HAS_OUTPUT_V7, reason="output_v7 data not present")
    def test_creates_pdf_from_real_data(self, tmp_path):
        """Full integration test: real output_v7 data → report.pdf created."""
        out = run(
            stats_path=os.path.join(OUTPUT_V7, "player_stats.json"),
            identity_path=os.path.join(OUTPUT_V7, "player_identity.json"),
            analysis_path=os.path.join(OUTPUT_V7, "analysis.json"),
            detections_path=os.path.join(OUTPUT_V7, "detections.json"),
            output_dir=str(tmp_path),
        )
        assert out, "run() should return a non-empty path"
        assert os.path.exists(out), "report.pdf file must exist"
        assert os.path.getsize(out) > 1000, "PDF must be larger than 1KB"

    def test_creates_pdf_from_minimal_data(self, tmp_path):
        """Minimal synthetic inputs — smoke test that all 4 pages render."""
        _setup_dir(str(tmp_path))
        out = run(
            stats_path=str(tmp_path / "player_stats.json"),
            identity_path=str(tmp_path / "player_identity.json"),
            analysis_path=str(tmp_path / "analysis.json"),
            detections_path=str(tmp_path / "detections.json"),
            output_dir=str(tmp_path),
        )
        assert out, "run() should return a path"
        assert os.path.exists(out)
        assert os.path.getsize(out) > 1000


class TestPdfReportLlmUnavailable:

    def test_pdf_generates_without_llm(self, tmp_path):
        """analysis.json with all-null LLM sections → PDF still created."""
        _setup_dir(str(tmp_path), analysis=_minimal_analysis(has_llm=False))
        out = run(
            stats_path=str(tmp_path / "player_stats.json"),
            identity_path=str(tmp_path / "player_identity.json"),
            analysis_path=str(tmp_path / "analysis.json"),
            detections_path=str(tmp_path / "detections.json"),
            output_dir=str(tmp_path),
        )
        assert out, "PDF should still be created when LLM data is absent"
        assert os.path.exists(out)


class TestPdfReportEdgeCases:

    def test_empty_sprint_episodes(self, tmp_path):
        """sprint_episodes=[] → no crash; page 2 renders without timeline."""
        stats = _minimal_stats(sprint_episodes=[])
        _setup_dir(str(tmp_path), stats=stats)
        out = run(
            stats_path=str(tmp_path / "player_stats.json"),
            identity_path=str(tmp_path / "player_identity.json"),
            analysis_path=str(tmp_path / "analysis.json"),
            detections_path=str(tmp_path / "detections.json"),
            output_dir=str(tmp_path),
        )
        assert out and os.path.exists(out), "PDF must still be created"

    def test_sparse_detections_scatter_fallback(self, tmp_path):
        """<50 target detections → scatter plot used instead of KDE."""
        dets = _minimal_detections(track_ids=(1,), n_dets=20)  # only 20 dets
        _setup_dir(str(tmp_path), detections=dets)
        out = run(
            stats_path=str(tmp_path / "player_stats.json"),
            identity_path=str(tmp_path / "player_identity.json"),
            analysis_path=str(tmp_path / "analysis.json"),
            detections_path=str(tmp_path / "detections.json"),
            output_dir=str(tmp_path),
        )
        assert out and os.path.exists(out), "Sparse dets must still produce PDF"

    def test_zero_target_detections(self, tmp_path):
        """Target track ID not in detections → heatmap shows text fallback."""
        # Identity references track 99 but detections only have track 1
        identity = _minimal_identity(track_ids=(99,))
        _setup_dir(str(tmp_path), identity=identity)
        out = run(
            stats_path=str(tmp_path / "player_stats.json"),
            identity_path=str(tmp_path / "player_identity.json"),
            analysis_path=str(tmp_path / "analysis.json"),
            detections_path=str(tmp_path / "detections.json"),
            output_dir=str(tmp_path),
        )
        assert out and os.path.exists(out), "Zero dets must still produce PDF"

    def test_missing_ball_proximity_key(self, tmp_path):
        """ball_proximity_episodes key absent → page 2 silently omits the row."""
        stats = _minimal_stats(ball_proximity_episodes=None)  # key NOT set
        _setup_dir(str(tmp_path), stats=stats)
        out = run(
            stats_path=str(tmp_path / "player_stats.json"),
            identity_path=str(tmp_path / "player_identity.json"),
            analysis_path=str(tmp_path / "analysis.json"),
            detections_path=str(tmp_path / "detections.json"),
            output_dir=str(tmp_path),
        )
        assert out and os.path.exists(out), "Absent ball_proximity must not crash"

    def test_empty_improvements_page3_placeholder(self, tmp_path):
        """improvements.suggestions=[] → page 3 shows placeholder card, not crash."""
        analysis = _minimal_analysis()
        analysis["improvements"]["suggestions"] = []
        _setup_dir(str(tmp_path), analysis=analysis)
        out = run(
            stats_path=str(tmp_path / "player_stats.json"),
            identity_path=str(tmp_path / "player_identity.json"),
            analysis_path=str(tmp_path / "analysis.json"),
            detections_path=str(tmp_path / "detections.json"),
            output_dir=str(tmp_path),
        )
        assert out and os.path.exists(out), "Empty suggestions must still produce PDF"


class TestPdfReportHelpers:

    def test_target_positions_pitch_coordinate_transform(self):
        """Bbox centres are correctly transformed to pitch coordinates."""
        dets = [
            {"frame_num": 0, "track_id": 1,
             "bbox": [0.0, 0.0, 192.0, 108.0],   # top-left → pitch origin area
             "confidence": 0.9, "class_id": 0},
            {"frame_num": 1, "track_id": 1,
             "bbox": [1728.0, 972.0, 1920.0, 1080.0],  # bottom-right corner
             "confidence": 0.9, "class_id": 0},
        ]
        x, y = _target_positions_pitch(dets, {1}, 1920.0, 1080.0)
        assert len(x) == 2
        # first bbox centre px=(96, 54) → pitch≈(5.25, 3.4)
        assert abs(x[0] - 96 / 1920 * 105) < 0.1
        assert abs(y[0] - 54 / 1080 * 68) < 0.1
        # all values must be within pitch bounds
        assert all(0 <= xi <= 105 for xi in x)
        assert all(0 <= yi <= 68 for yi in y)

    def test_target_positions_filters_by_track_id(self):
        """Only detections with track_id in target_ids are returned."""
        dets = [
            {"frame_num": 0, "track_id": 1,
             "bbox": [0, 0, 100, 100], "confidence": 0.9, "class_id": 0},
            {"frame_num": 1, "track_id": 2,   # not a target
             "bbox": [500, 300, 600, 400], "confidence": 0.9, "class_id": 0},
        ]
        x, y = _target_positions_pitch(dets, {1}, 1920.0, 1080.0)
        assert len(x) == 1, "Only track_id=1 detection should be returned"

    def test_compute_work_rate_sums_to_100(self):
        """work rate percentages must sum to ~100."""
        stats = {
            "total_time_visible_s": 300,
            "avg_speed_mps": 2.0,
            "max_speed_mps": 6.0,
            "sprint_count": 10,
        }
        wr = _compute_work_rate(stats)
        total = sum(wr.values())
        assert abs(total - 100) < 1.0, f"Work rate total should be ~100, got {total:.1f}"
        for key in ("idle", "jogging", "running", "sprinting"):
            assert key in wr, f"Missing key: {key}"
            assert wr[key] >= 0, f"{key} must be non-negative"
