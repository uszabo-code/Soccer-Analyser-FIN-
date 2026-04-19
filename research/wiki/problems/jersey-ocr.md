---
name: Jersey OCR Unreliable
type: problem
severity: significant
status: open
related_programs: []
related_concepts: [ocr, player-identification]
---

## Summary
Jersey OCR is not merely "unreliable" — it is **effectively disabled** for the vast majority of this footage. The `MIN_PLAYER_HEIGHT_PX = 120` gate means OCR runs on only ~1% of the target player's detections. The root cause is player size, not model quality.

## Player size reality (code-verified from GT data)

Target player bbox heights measured across 180 annotated GT frames:

| Metric | Height (px) |
|--------|-------------|
| min | 53 |
| p25 | 69 |
| median | **77** |
| p75 | 90 |
| max | **120** |

- Frames ≥ 120px (`MIN_PLAYER_HEIGHT_PX`): **2 / 180 (1%)**
- Frames ≥ 150px: **0 / 180 (0%)**

OCR is skipped if the track's largest detection is below 120px (`pipeline/identify.py` line 69). With a median of 77px and max of 120px, OCR fires on ≈0–1 tracks per player across the whole game. This is a camera/field geometry constraint, not a tunable parameter.

## Root Cause
The Veo Cam 2 is mounted at an elevated fixed angle showing ~60m of pitch at 2868×1320. Players occupy ~5–7% of frame height. At 77px median, jersey numbers are ~15–20px — below the practical legibility floor for any OCR system. The overhead angle further flattens the jersey surface.

## What the OCR system actually does (code-verified)
- 3 strategies in `read_jersey_number_multi()`: (1) CLAHE+bilateral+sharpen, (2) Otsu binary threshold (white numbers on dark jersey), (3) inverted binary (dark numbers on light jersey)
- Stage 2 takes the 5 largest detections per significant track, runs all 3 strategies on each, accepts a number if it appears ≥ `OCR_MIN_VOTES=3` times
- But if even the largest detection is <120px, the entire track is skipped before any of this runs

## Current Understanding
- Team color clustering (black vs white jerseys confirmed from footage) works fine regardless of player height — it uses `get_dominant_color()` on the bbox, not OCR. Team splitting is solved.
- Jersey number identification is essentially unsolved for this camera/field setup.
- The interactive picker mentioned in the README was **removed** from the code — replaced by "use the largest unidentified track as fallback" (see [stage3-player-dropped](stage3-player-dropped.md)).
- EasyOCR was tested and performed worse than PaddleOCR. PaddleOCR v3.x segfaults on Apple Silicon MPS — forced CPU workaround in place.

## What's Been Tried
- PaddleOCR 3-strategy voting with multi-sample majority vote
- EasyOCR fallback (worse)
- Forced CPU mode for MPS segfault
- `MIN_PLAYER_HEIGHT_PX = 120` threshold to avoid wasting OCR on unreadable crops

## Open Questions
- Lowering `MIN_PLAYER_HEIGHT_PX` to ~50px would let OCR run more often, but jersey numbers at 50px are likely unreadable by any model. Worth testing to see if it helps or just adds noise.
- Could a fine-tuned model (e.g. CRNN trained on overhead soccer crops at 50–100px) reach useful accuracy?
- For the target player: the right UX is likely a one-time manual selection per game (click on the player in the first frame), stored in a session file. This is simpler and more reliable than OCR.
- For all 22 players: team color clustering already works (black vs white). Individual player ID within a team (by number) is a separate, harder problem — may not be needed for the core use case.

## References
- `utils/ocr.py` (3-strategy OCR, `_preprocess_crop`)
- `pipeline/identify.py` lines 44–50 (`MIN_TRACK_LENGTH`), 69 (`MIN_PLAYER_HEIGHT_PX` gate)
- `config.py` (`MIN_PLAYER_HEIGHT_PX=120`, `OCR_MIN_VOTES=3`, `OCR_SAMPLES_PER_TRACK=20`)
- [stage3-player-dropped](stage3-player-dropped.md) (interactive picker removal)
