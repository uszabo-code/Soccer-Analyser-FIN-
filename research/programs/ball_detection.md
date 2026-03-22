# Ball Detection Optimization

## Goal
Maximize the number of reliable ball detections per frame. The ball is a small object in a screen recording of a livestreamed U13-U15 soccer game at 2868x1320 resolution.

## Current Baseline
- 122 ball detections out of ~6,400 processed frames (1.9% detection rate)
- Ball detections cluster in short bursts — most of the video has zero ball detections
- YOLO confidence threshold is 0.3 which may be too high for small ball objects

## Metric
Score combines: detection rate (40%), temporal coverage across 1-second windows (35%), and temporal consistency — detections having neighbors within 5 frames (25%).

## Available Parameters
- CONFIDENCE_THRESHOLD: Lower values detect more balls but also more false positives. Try 0.1-0.25 range.
- OVERLAY_TOP_FRACTION / OVERLAY_BOTTOM_FRACTION: The ball can appear in overlay regions. Try reducing these to avoid filtering out ball detections near the top/bottom.
- FRAME_SKIP: Processing more frames (lower skip) captures more ball appearances but costs more time.

## Strategy
1. Start by lowering CONFIDENCE_THRESHOLD significantly (0.10-0.15) — the ball is small and low-confidence detections may still be valid
2. Try reducing overlay fractions — the ball may appear near broadcast overlays
3. Try FRAME_SKIP=2 or 1 if time budget allows — more frames = more chances to detect the ball
4. Balance: too low confidence will introduce false positives (random small objects), hurting consistency score

## Constraints
- Each experiment must complete within 2 minutes on this 30-second clip
- The primary goal is DETECTION RATE — we need the ball visible in at least 30% of frames
- False positives are acceptable if they're temporally consistent (real ball is in roughly the same area across frames)
