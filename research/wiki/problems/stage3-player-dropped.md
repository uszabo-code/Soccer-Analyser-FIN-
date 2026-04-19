---
name: Stage 3 Drops Target Player
type: problem
severity: critical
status: open
related_programs: [reid]
related_concepts: [track-merging]
---

## Summary
The target player's stats in Stage 3 cover only a tiny fraction of the game. Stage 3 produces stats, but only for the 0–5 OCR-identified fragments — not the full ~70 fragments. The result is misleading partial stats, not a clean error. The README description ("silently skipped") was inaccurate.

## How it actually works (code-verified)

**Stage 2 (`pipeline/identify.py`):**
- `significant_tracks` = tracks with ≥ `max(30, 90/frame_skip)` = **45 dets** at FRAME_SKIP=2
- OCR runs only on significant tracks; `target_track_ids` = all tracks where OCR read the target jersey
- If OCR finds nothing: falls back to the single largest track
- **Interactive picker is gone** — removed from the code (was mentioned in README but not implemented)
- Each `PlayerIdentity` object has exactly one `track_id` — no cross-fragment grouping

**Stage 3 (`pipeline/features.py`):**
- Builds `analyze_track_ids` from `players` where track has >50 dets
- Force-adds all `target_track_ids` (line 72) — bypasses the >50 filter
- Secondary filter at line 80: drops tracks with <10 dets (won't hit the largest-track fallback)
- Stats computed **per-fragment independently** — no aggregation across target fragments

## Root Cause
With ~70 target player fragments and unreliable OCR, `target_track_ids` contains only 0–5 OCR-identified fragments. Stage 3 analyzes those fragments independently and produces 0–5 partial `PlayerStats` entries tagged `is_target=True`. The other 65 fragments are analyzed as unnamed `Track-NNN` entries — their stats exist but are not attributed to the target player.

The fix requires Stage 1b to merge the target's fragments into one (or a few) long tracks **before** Stage 2 runs OCR. Then OCR has a long, identifiable track to work with, and Stage 3 computes stats on the full game track.

## What's Been Tried
- Nothing yet — blocked on Stage 1b validation.

## Open Questions
- After a successful Stage 1b merge, how many merged tracks should the target player produce? Ideally 1, realistically maybe 3–5.
- Should Stage 3 aggregate stats across multiple `is_target=True` fragments? Or is "one long merged track" the right precondition?
- Worth adding a warning when `is_target=True` tracks cover <20% of total game frames?

## References
- `pipeline/features.py` lines 64–80 (analyze_track_ids construction, secondary filter)
- `pipeline/identify.py` lines 44–150 (significant_tracks, target_track_ids, picker removal)
- [stage1b-merger-bug](stage1b-merger-bug.md)
- [track-fragmentation](track-fragmentation.md)
