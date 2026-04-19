---
name: Stage 1b Track Merger Bug
type: problem
severity: critical
status: in-progress
related_programs: [reid]
related_concepts: [union-find, track-merging]
---

## Summary
The Union-Find post-process track merger (Stage 1b, `pipeline/reidentify.py`) previously chain-merged entire teams into single mega-tracks. The root cause has been diagnosed and a fix implemented in commit `2de18a6`, but **the fix has not yet been validated on real game footage**.

## Root Cause
`MAX_GAP_FRAMES` was set to 900 (30s) to compensate for heavy fragmentation, but the large window meant player A's track end could match player B's track start just because they were nearby. Union-Find then propagated the chain: A+B, B+C → whole team in one track.

## Fix (implemented, unvalidated)

Four guards added in `2de18a6`:

| Guard | Value | Purpose |
|-------|-------|---------|
| `MAX_GAP_FRAMES` | 150 (was 900) | Limits gap to ~5s at 30fps |
| `WALK_SPEED_PX_PER_PROCESSED_FRAME` | 12.0 px/frame | Walking-speed distance model replaces fixed pixel cap |
| `MAX_MERGED_PER_GROUP` | 5 | Rejects oversized Union-Find groups (runaway chain guard) |
| `MAX_DIRECTION_CHANGE_DEG` | 90° | Rejects merges where exit→entry velocity angle is too large |

The gap history: 90 (initial) → 900 (b6dd698, too wide) → 150 (2de18a6, current).

## Current Understanding
- Fix is code-complete. Stage 1b is no longer disabled — `--no-merge` is still available as an escape hatch but is not the default.
- The O(n²) pairwise candidate search is acceptable given the guards.
- Not tested at full-game scale (1,540 tracks). Unknown whether `MAX_MERGED_PER_GROUP=5` is the right cap — heavy fragmentation (~70 fragments/player) means a single player could legitimately need more than 5 merges.

## Open Questions
- Does `MAX_MERGED_PER_GROUP=5` reject valid merges for heavily fragmented players? Should it be higher (e.g. 15–20)?
- What does the gap distribution look like on a full 20-minute game? Is `MAX_GAP_FRAMES=150` too tight for some legitimate gaps?
- End-to-end test needed: run Stage 1b on full game, check track count before/after, spot-check a few merged tracks visually.

## References
- `pipeline/reidentify.py`
- `research/programs/reid.md`
- Commit `2de18a6` (fix implementation)
