---
name: Ball Detection Rate Drop on Full Games
type: problem
severity: significant
status: open
related_programs: [ball_detection]
related_concepts: [yolo, ball-detection]
---

## Summary
Ball detection drops from ~30–40% on short eval clips to 5.6% on full 20-minute games. Ball proximity and involvement stats are unreliable as a result.

## Root Cause
Two components:

1. **Structural**: Many full-game frames have no visible ball — out of play, halftime, off-camera, wide shots. The eval clip was cherry-picked active play, biasing headline rates upward.
2. **Model limits**: `football_yolov8.pt` YOLO detection ceiling on the eval clip GT is ~15% recall (153 annotated frames). Autoresearch composite score peaked at **41.1%** across 16 experiments.

## Metric clarification
The autoresearch "ball detection score" (41.1%) is a **composite** — detection rate (40%) + temporal coverage (35%) + consistency (25%) — targeting 50% detection rate per processed frame. It is **not** the same as IoU-based GT recall. These numbers are not directly comparable to "5.6% full game."

## Current Architecture (code-verified, 2de18a6)
Two detectors run in parallel in `pipeline/detect.py`:
- **YOLO** (`football_yolov8.pt`, `BALL_CONFIDENCE_THRESHOLD=0.05`): model-based, catches any ball it was trained on
- **CV detector** (`pipeline/cv_ball_detector.py`): HSV+Hough circle transform + Kalman-gated trajectory

**Critical limitation**: The CV detector targets **hi-vis yellow-green futsal balls only** (HSV H=55–90, S=70–170, V=140–255). It will not fire on white, orange, or standard outdoor balls. The name "futsal ball" in the code is deliberate.

**Kalman ball tracker** (`pipeline/ball_tracker.py`): gap interpolation up to 10 processed frames (~0.8s at FRAME_SKIP=2). Added in 2de18a6.

## CV detector performance (from 2de18a6 commit)
- GT recall on 153 annotated eval frames: **75.8%** (CV) vs **15%** (YOLO alone)
- This is theoretical recall — false positive rate not measured
- Only valid for yellow-green futsal balls

## Autoresearch findings (16 experiments)
- Best composite score: **41.1%** — ceiling, multiple configs tied
- Best params require `FRAME_SKIP=1` (every frame), but current tracking config uses `FRAME_SKIP=2`
- **FRAME_SKIP trade-off**: lower skip = better ball detection but more player track fragments (FRAME_SKIP=2 is the tracking optimum)

## What's Been Tried
- `football_yolov8.pt` baseline: 5.6% full game, ~15% GT recall on eval clip
- Autoresearch: 16 experiments, composite ceiling at 41.1% with FRAME_SKIP=1
- CV detector (HSV+Hough+Kalman): 75.8% GT recall for yellow-green futsal ball, now integrated
- Kalman gap interpolation: up to 0.8s gaps filled, integrated in detect.py
- `BALL_CONFIDENCE_THRESHOLD=0.05`: very permissive, catches faint signals

## Ball color — confirmed

**The ball is white** (standard outdoor soccer ball). Confirmed from game footage screenshot (2026-04-19).

This means the CV detector contributes **zero detections** in production — its HSV mask targets H=55–90 (yellow-green) and will not fire on a white ball. The YOLO+CV ensemble degrades to YOLO-only for all Finnish youth club footage using standard outdoor balls.

The 75.8% GT recall figure for the CV detector is irrelevant for this use case.

## Open Questions
- `soccana_yolo11.pt` (40% composite score on eval clip) is the most promising next step — it's a YOLO-only improvement that doesn't depend on ball color.
- Should the CV detector be disabled for outdoor white ball games (or auto-detected from a calibration frame) to avoid wasted compute?
- Should isolated single-frame YOLO detections be post-filtered as false positives?
- Is there a CV approach that works on white balls — e.g. circular shape + motion-based filter rather than color?

## References
- `research/programs/ball_detection.md`
- `pipeline/ball_tracker.py`
- `pipeline/cv_ball_detector.py`
- README: "Ball detection drops to 5.6% on full games"
