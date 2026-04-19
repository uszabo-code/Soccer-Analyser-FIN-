"""
Tests for pipeline/identify.py — specifically the target fragment sweep (Fix 3).

Covers:
  - _sweep_for_target_fragments: finds adjacent, color-similar, spatially plausible tracks
  - Sweep correctly rejects cross-team candidates
  - Sweep correctly rejects spatially implausible candidates (too far away)
  - Sweep is idempotent (calling twice doesn't double-add)
  - Sweep convergence (transitive chains of fragments are all found)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock
from pipeline.identify import _sweep_for_target_fragments


FPS = 30.0
FRAME_WIDTH = 1920


def _make_det(track_id, frame_num, x=500, y=400):
    """Create a minimal mock Detection with the needed attributes."""
    det = MagicMock()
    det.frame_num = frame_num
    det.center = (x, y)
    return det


def _make_tracks(specs):
    """
    specs: list of (track_id, first_frame, last_frame, x_pos, color_hue)
    Returns (significant_tracks, track_colors)
    """
    significant_tracks = {}
    track_colors = {}
    for tid, first_f, last_f, x_pos, hue in specs:
        # 2 detections per track: first and last frame
        dets = [
            _make_det(tid, first_f, x=x_pos),
            _make_det(tid, last_f, x=x_pos),
        ]
        significant_tracks[tid] = dets
        track_colors[tid] = [hue, 180.0, 150.0]  # [H, S, V]
    return significant_tracks, track_colors


class TestSweepForTargetFragments:
    """Tests for _sweep_for_target_fragments()."""

    def test_adjacent_same_color_track_added(self):
        """
        Fragment A ends at frame 100, Fragment B starts at frame 200 (gap=100 frames=3.3s).
        Same team, same color, close position → should be added to target_track_ids.
        """
        specs = [
            # (track_id, first_frame, last_frame, x_pos, hue)
            (10, 0,   100, 500, 110.0),   # Known target seed
            (20, 200, 300, 510, 112.0),   # Adjacent fragment — same color, close
        ]
        significant_tracks, track_colors = _make_tracks(specs)
        teams = {"team_a": [10, 20], "team_b": []}

        result = _sweep_for_target_fragments(
            target_track_ids=[10],
            significant_tracks=significant_tracks,
            track_colors=track_colors,
            teams=teams,
            target_team="team_a",
            fps=FPS,
            frame_width=FRAME_WIDTH,
        )

        assert 20 in result, "Adjacent same-color fragment should be swept into target"

    def test_cross_team_candidate_not_added(self):
        """
        Fragment B is on the opposing team — should never be added, regardless of
        temporal adjacency and color similarity.
        """
        specs = [
            (10, 0,   100, 500, 110.0),   # Known target (team_a)
            (20, 200, 300, 510, 112.0),   # Adjacent BUT on team_b
        ]
        significant_tracks, track_colors = _make_tracks(specs)
        teams = {"team_a": [10], "team_b": [20]}

        result = _sweep_for_target_fragments(
            target_track_ids=[10],
            significant_tracks=significant_tracks,
            track_colors=track_colors,
            teams=teams,
            target_team="team_a",
            fps=FPS,
            frame_width=FRAME_WIDTH,
        )

        assert 20 not in result, "Cross-team track must never be added to target"

    def test_spatially_implausible_candidate_rejected(self):
        """
        Fragment B starts at x=1800, far from target's last position at x=100.
        Gap is only 10 frames (0.33s). At 8 m/s and 30 px/m, max distance = 80px.
        Actual distance = 1700px >> 80px → rejected.
        """
        specs = [
            (10, 0,  100, 100, 110.0),    # Known target, ends at x=100
            (20, 110, 200, 1800, 112.0),  # Starts at x=1800 (1700px away), only 10-frame gap
        ]
        significant_tracks, track_colors = _make_tracks(specs)
        teams = {"team_a": [10, 20], "team_b": []}

        result = _sweep_for_target_fragments(
            target_track_ids=[10],
            significant_tracks=significant_tracks,
            track_colors=track_colors,
            teams=teams,
            target_team="team_a",
            fps=FPS,
            frame_width=FRAME_WIDTH,
        )

        assert 20 not in result, (
            "Spatially implausible track (1700px in 0.33s) must be rejected"
        )

    def test_different_color_candidate_rejected(self):
        """
        Fragment B has a very different jersey color (red hue vs blue hue).
        Should be rejected by the HSV gate even if temporally adjacent and close.
        """
        specs = [
            (10, 0,   100, 500, 110.0),   # Blue jersey (hue=110)
            (20, 200, 300, 505,   5.0),   # Red jersey (hue=5) — different team member
        ]
        significant_tracks, track_colors = _make_tracks(specs)
        teams = {"team_a": [10, 20], "team_b": []}

        result = _sweep_for_target_fragments(
            target_track_ids=[10],
            significant_tracks=significant_tracks,
            track_colors=track_colors,
            teams=teams,
            target_team="team_a",
            fps=FPS,
            frame_width=FRAME_WIDTH,
        )

        assert 20 not in result, "Different-color track must be rejected by HSV gate"

    def test_idempotent_repeated_sweep(self):
        """Calling sweep twice should not add duplicates."""
        specs = [
            (10, 0,   100, 500, 110.0),
            (20, 200, 300, 510, 112.0),
        ]
        significant_tracks, track_colors = _make_tracks(specs)
        teams = {"team_a": [10, 20], "team_b": []}

        result1 = _sweep_for_target_fragments(
            target_track_ids=[10],
            significant_tracks=significant_tracks,
            track_colors=track_colors,
            teams=teams,
            target_team="team_a",
            fps=FPS,
            frame_width=FRAME_WIDTH,
        )

        result2 = _sweep_for_target_fragments(
            target_track_ids=result1,
            significant_tracks=significant_tracks,
            track_colors=track_colors,
            teams=teams,
            target_team="team_a",
            fps=FPS,
            frame_width=FRAME_WIDTH,
        )

        assert len(result2) == len(set(result2)), "No duplicates after repeated sweep"
        assert set(result2) == set(result1)

    def test_transitive_chain_found(self):
        """
        A → B → C: each pair is adjacent, but A and C are far apart in time.
        The sweep should find B in the first pass, then C in the second pass
        (since B became a new anchor). Convergence should catch all.
        """
        # A ends at frame 100, B starts at frame 200 (gap=100), ends at frame 300
        # C starts at frame 400 (gap=100 from B), same color, close position
        specs = [
            (10, 0,   100, 500, 110.0),   # A (seed)
            (20, 200, 300, 510, 112.0),   # B (adjacent to A)
            (30, 400, 500, 520, 111.0),   # C (adjacent to B, not directly to A)
        ]
        significant_tracks, track_colors = _make_tracks(specs)
        teams = {"team_a": [10, 20, 30], "team_b": []}

        result = _sweep_for_target_fragments(
            target_track_ids=[10],
            significant_tracks=significant_tracks,
            track_colors=track_colors,
            teams=teams,
            target_team="team_a",
            fps=FPS,
            frame_width=FRAME_WIDTH,
        )

        assert 20 in result, "B (adjacent to seed A) should be found"
        assert 30 in result, "C (transitive via B) should be found via convergence"

    def test_gap_too_large_candidate_excluded(self):
        """
        Fragment B starts more than SWEEP_MAX_GAP_FRAMES (1800 frames = 60s) after
        the target ends. Player likely left the field — should not be swept.
        """
        specs = [
            (10, 0,    100, 500, 110.0),
            (20, 2000, 2100, 510, 112.0),  # 1900-frame gap > 1800 max
        ]
        significant_tracks, track_colors = _make_tracks(specs)
        teams = {"team_a": [10, 20], "team_b": []}

        result = _sweep_for_target_fragments(
            target_track_ids=[10],
            significant_tracks=significant_tracks,
            track_colors=track_colors,
            teams=teams,
            target_team="team_a",
            fps=FPS,
            frame_width=FRAME_WIDTH,
        )

        assert 20 not in result, "Track after 60s gap must not be swept (player left field)"

    def test_no_color_data_returns_original(self):
        """
        If the target tracks have no color information, the sweep cannot make
        decisions — it should return the original target_track_ids unchanged.
        """
        specs = [
            (10, 0,   100, 500, 110.0),
            (20, 200, 300, 510, 112.0),
        ]
        significant_tracks, _ = _make_tracks(specs)
        track_colors = {}  # No color data for ANY track

        teams = {"team_a": [10, 20], "team_b": []}

        result = _sweep_for_target_fragments(
            target_track_ids=[10],
            significant_tracks=significant_tracks,
            track_colors=track_colors,
            teams=teams,
            target_team="team_a",
            fps=FPS,
            frame_width=FRAME_WIDTH,
        )

        assert result == [10], "Without color data, original target_track_ids returned unchanged"

    def test_overlapping_track_rejected(self):
        """
        Candidate C overlaps with confirmed fragment A (frames 50–150 vs A's 0–100),
        but is adjacent to confirmed fragment B (frames 200–300).
        Without the pre-overlap guard, C would be added via B.
        With the guard, C must be rejected because it overlaps A.
        """
        specs = [
            (10, 0,   100, 500, 110.0),   # Seed fragment A: frames 0–100
            (20, 200, 300, 510, 112.0),   # Fragment B: frames 200–300, non-overlapping
            (30, 50,  150, 505, 111.0),   # Candidate C: overlaps A (50–150), adjacent to B
        ]
        significant_tracks, track_colors = _make_tracks(specs)
        teams = {"team_a": [10, 20, 30], "team_b": []}

        # First add fragment B (non-overlapping, should be added)
        result = _sweep_for_target_fragments(
            target_track_ids=[10, 20],
            significant_tracks=significant_tracks,
            track_colors=track_colors,
            teams=teams,
            target_team="team_a",
            fps=FPS,
            frame_width=FRAME_WIDTH,
        )

        assert 30 not in result, (
            "Candidate overlapping a confirmed fragment must be rejected "
            "even if it is adjacent to another confirmed fragment"
        )
