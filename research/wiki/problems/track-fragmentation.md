---
name: Track Fragmentation
type: problem
severity: significant
status: open
related_programs: [reid, tracking_smoothness]
related_concepts: [botsort, track-merging, union-find]
---

## Summary
A 20-minute game produces ~1,540 significant tracks for ~22 expected players — roughly 70 track fragments per player. Each time detection confidence drops (occlusion, clustering, low contrast), the tracker opens a new ID on re-detection instead of recovering the existing one. This cascades into the two critical bugs: Stage 1b merger can't fix it (see [stage1b-merger-bug](stage1b-merger-bug.md)), and Stage 3 drops the target player because no single track is long enough.

## Root Cause
YOLOv8 detection confidence falls below `match_thresh` during:
- Player clustering (multiple players overlapping)
- Partial occlusions (player behind a teammate)
- Low-contrast moments (shadow, white jersey on light grass)

When confidence drops below the tracker's association threshold, the tracker marks the track as lost. After `track_buffer` frames of absence it gets removed. When the player reappears, a new ID is assigned.

BoT-SORT's `track_buffer=90` (3 seconds at FRAME_SKIP=2, 25fps) helps but doesn't fully compensate for long occlusions.

## Current BoT-SORT Config (`botsort_soccer.yaml`, code-verified)

| Parameter | Value | Note |
|-----------|-------|------|
| `track_buffer` | **150** | Increased from 90→150 in 2de18a6 (~5s at 30fps) |
| `match_thresh` | **0.65** | Tightened from 0.7→0.65 in 2de18a6 |
| `with_reid` | False | Disabled — OSNet OOD for overhead soccer |
| `gmc_method` | none | Disabled — causes mega-track on fixed camera |
| `track_high_thresh` | 0.25 | |
| `new_track_thresh` | 0.25 | |

## Current Understanding
- **ReID disabled**: tested `with_reid=True`, `appearance_thresh=0.25` — track count tripled, median track length dropped to 3 frames. OSNet is trained on close-up pedestrians, not overhead 80–150px soccer players.
- **GMC disabled**: `sparseOptFlow` on a fixed camera finds near-zero global motion, corrupting Kalman predictions — caused 2 track IDs to absorb ~18,000 frames each.
- **FRAME_SKIP=2** reduced total tracks 262→140 on a 300-frame clip and produced the first perfect-continuity track (vs FRAME_SKIP=3).
- **track_buffer=150** gives ~5s gap tolerance. Higher values risk false merges between different players.
- **Tracking smoothness autoresearch** (separate from fragmentation count): best score 85.1% with `SMOOTHING_WINDOW=5`, `MIN_MOVEMENT_PX=10`, `DIRECTION_CHANGE_ANGLE=120` — matches current `config.py`. This optimizes movement plausibility, not track count.
- **Stage 1b** fix is implemented but unvalidated — see [stage1b-merger-bug](stage1b-merger-bug.md).
- Full-game fragmentation (~1,540 tracks / 22 players) has not been re-measured since `track_buffer` was increased to 150 and `match_thresh` tightened to 0.65.

## What's Been Tried
- ReID (`with_reid=True`): disabled — worse fragmentation on overhead footage
- GMC (`sparseOptFlow`): disabled — mega-track on fixed camera
- FRAME_SKIP=2: improved vs FRAME_SKIP=3
- `track_buffer` 30→90→150: incremental improvements, current ceiling unknown
- `match_thresh` 0.7→0.65: slightly more permissive re-association
- Stage 1b post-process merger: fix implemented, not yet validated

## Open Questions
- What is the actual fragmentation score with `track_buffer=150`, `match_thresh=0.65`? Last measured was ~1,540 tracks with older params.
- Is `track_buffer=150` the right ceiling, or would 200–300 reduce fragmentation without causing false merges?
- Could a domain-adapted appearance model (fine-tuned on overhead soccer crops from this footage) make ReID viable?
- Once Stage 1b is validated: what is the post-merge track count?

## References
- `research/programs/reid.md`
- `pipeline/reidentify.py`
- `botsort_soccer.yaml`
- README: "Track fragmentation" and "Why no ReID?" sections
