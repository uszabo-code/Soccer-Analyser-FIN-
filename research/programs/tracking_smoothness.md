# Tracking Smoothness Optimization

## Goal
Reduce tracking noise (jitter) and make direction change detection more plausible. Currently the pipeline reports implausible events like 5 sharp direction changes in a single second, caused by pixel-level tracking noise being interpreted as real movement.

## Current Baseline
- 80 sharp turns (>90 degrees) detected for the target player in 56 seconds of tracking
- 9 clusters of 3+ consecutive sharp turns — physically impossible at these frame rates
- Tracking jitter creates phantom micro-movements that inflate direction change counts
- No position smoothing is applied before computing angles or speeds

## Metric
Score combines: smooth movement fraction — movements under 50px (40%), direction change plausibility — fraction of angle changes under 90 degrees (40%), and average track length — longer tracks indicate better tracking continuity (20%).

## Available Parameters
- SMOOTHING_WINDOW: Apply moving average to positions before analysis. Odd values only (3, 5, 7, 9, 11). Higher = smoother but may lose real agility.
- MIN_MOVEMENT_PX: Ignore movements smaller than this threshold (filters sub-pixel jitter). Try 5-15px.
- DIRECTION_CHANGE_ANGLE: Threshold for what counts as a "sharp" direction change. Currently 90 degrees. Try 110-140 degrees to be more selective.
- FRAME_SKIP: Higher skip means larger movements between frames, naturally reducing jitter impact — but also loses temporal resolution.

## Strategy
1. Start with SMOOTHING_WINDOW=3 or 5 — even mild smoothing dramatically reduces jitter
2. Increase MIN_MOVEMENT_PX from 5 to 10-15 — sub-10px movements in a 2868px frame are likely noise
3. Try DIRECTION_CHANGE_ANGLE=120 — a 90-degree threshold catches too many normal turns
4. These three together should bring the sharp turn count from 80 to something reasonable (10-20)
5. Be careful not to over-smooth — real agility moves should still be detectable

## Constraints
- This metric runs instantly on cached detections (no re-detection needed)
- A youth soccer player in active play makes perhaps 5-15 truly sharp direction changes per minute
- The smoothing should preserve sprint detection (high-speed straight-line movements)
- Over-smoothing will make tracks look unrealistically straight, losing real movement information
