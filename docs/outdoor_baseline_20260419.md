# Outdoor Baseline — 1H Game 19042026.mp4

**Date**: 2026-04-19
**Clip**: `/Users/uszabo/Downloads/1H Game 19042026.mp4` (1920×1078 @ 60fps, 66.7 min total)
**Segment analyzed**: minutes 10–20 (frames 36000–72000, 600s @ FRAME_SKIP=2 = 18000 processed frames)
**Purpose**: First outdoor measurement since season transition from indoor futsal. A/B comparing the base model vs. the indoor-fine-tuned ball model on outdoor (white/black ball, natural grass) footage.

## Setup

| Component | Run A (base) | Run B (finetuned) |
|---|---|---|
| Person model | `football_yolov8.pt` | `football_yolov8.pt` |
| Ball model | `football_yolov8.pt` | `futsal_ball_v1.pt` |
| Ball class id | 0 | 0 |
| Device | MPS | MPS |
| FRAME_SKIP | 2 | 2 |

Person tracking is identical between runs by design. Only the ball-prediction YOLO instance differs (see `pipeline/detect.py:113-135` dual-model architecture).

## Quantitative results

_Populated from `output/outdoor_ab/ab_summary.json` after the run completes._

| Metric | Run A (base) | Run B (finetuned) | Delta |
|---|---|---|---|
| Person detections (total) | | | |
| Unique person track IDs | | | |
| Ball detections (real) | | | |
| Ball detections (interpolated) | | | |
| Ball detections with conf ≥ 0.30 | | | |
| Ball mean confidence | | | |
| Ball temporal coverage | | | |
| Processed frames | | | |

## Qualitative notes (visual spot-check)

_To be filled after viewing output video or a frame grid._

- Ball-on-field frames: does model see the white/black outdoor ball? (Run A should; Run B trained on yellow-green may miss entirely.)
- False positives: does Run B fire on white field markings / goalposts / socks? (Likely, given HSV-adjacent features.)
- Player tracking: any degradation? Expect improvement vs indoor — `football_yolov8.pt`'s native domain.

## Decision

_Filled after comparison. Possible outcomes:_

- [ ] **Base ≫ finetuned outdoor** → revert `BALL_MODEL = None` as default. Next: annotate 30-50 outdoor frames → train `outdoor_ball_v1.pt` from base.
- [ ] **Finetuned ≈ base** → surprising generalization. Keep finetuned as default, still train outdoor version.
- [ ] **Both poor** → investigate clip (motion blur, camera angle, ball size). Something outside ball-model scope is broken.
- [ ] **Person tracking degrades vs indoor** → outdoor hypothesis was wrong. Player continuity (32.4%) blocker elevated.

## Known limitations of this measurement

- No outdoor ground truth — all quality assessment is heuristic and visual.
- 10-min sample from a single game clip; may not be representative of all outdoor lighting/angles.
- Temporal coverage ≤ 1.0 is a weak metric without IoU-based TP/FP split; `BALL_CONFIDENCE_THRESHOLD=0.05` is low, so high coverage can reflect mostly noise.
- Pre-existing issues NOT tested here: player continuity (32.4% fragmentation), TODO-003 silent field-width fallback, CV ball detector (yellow-green HSV) firing spuriously on outdoor white ball.

## Follow-ups (informed by result)

Ordered by likely ROI after seeing outcome:

1. Outdoor GT annotation (30-50 frames of this clip) via `research/pre_annotate.py` workflow → real recall numbers
2. Train `outdoor_ball_v1.pt` from `football_yolov8.pt` base (fresh, not incremental from futsal_ball_v1)
3. Player continuity: debug `pipeline/reidentify.py` target-player 4-fragment issue (venue-agnostic)
4. Venue config system: `configs/outdoor_7v7.json` with `FIELD_WIDTH_METERS=68`
5. TODO-003: non-silent field-width fallback warning
6. CV ball detector: conditional disable for outdoor (HSV range targets indoor yellow-green only)
