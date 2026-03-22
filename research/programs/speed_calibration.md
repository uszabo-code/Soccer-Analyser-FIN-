# Speed Calculation Calibration

## Goal
Make computed player speeds realistic for U13-U15 youth soccer. Currently the pipeline produces impossible speeds (max 56.89 m/s = 205 km/h) due to a naive pixel-to-meter conversion that assumes the full frame width equals the field width.

## Current Baseline
- Pixel-to-meter conversion: frame_width (2868px) / FIELD_WIDTH_METERS (90m) = 31.9 px/m
- This assumes the camera shows exactly the full field width in every frame — incorrect for a broadcast stream where the camera pans and zooms
- Maximum speed recorded: 56.89 m/s (should be < 10 m/s for youth players)
- Large tracking jitter spikes contribute to speed outliers

## Metric
Score = % of speed samples in [0, max_realistic_speed] range, with a penalty for extreme outliers. Higher is better.

## Available Parameters
- FIELD_WIDTH_METERS: The physical width assumption. A wider field means pixels represent more meters, so speeds go DOWN. Try 60-110m range. For a zoomed-in broadcast, the visible field might only be 40-60m across.
- SPEED_OUTLIER_CAP: Not yet implemented in pipeline — but this parameter signals what cap value would work best.
- SMOOTHING_WINDOW: Smoothing positions before computing speed reduces jitter-based outliers.
- MIN_MOVEMENT_PX: Minimum pixel displacement to count as real movement (filters sub-pixel jitter).

## Strategy
1. The real issue is FIELD_WIDTH_METERS — try much smaller values (60-80m) since the camera shows only part of the field
2. Increasing MIN_MOVEMENT_PX helps filter tracking jitter that creates phantom speed spikes
3. SMOOTHING_WINDOW > 1 averages positions over multiple frames, reducing noise
4. The goal is to get 95%+ of speeds into the realistic range while preserving real sprint detection

## Constraints
- This metric runs instantly on cached detections (no re-detection needed)
- Youth soccer players sprint at 6-8 m/s max, jog at 2-3 m/s, walk at 1-1.5 m/s
- The solution must preserve the ability to distinguish walking, jogging, and sprinting
