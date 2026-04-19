---
name: Speed & Distance Calibration
type: problem
severity: critical
status: open
related_programs: [speed_calibration]
related_concepts: [homography, pixel-to-metre-conversion]
---

## Summary
All physical stats (distance, speed, sprint detection) are meaningless because the pixel-to-metre conversion is wrong. Current output shows total distances of 97–591 km for a 20-minute game. Max speed recorded: 56.89 m/s (205 km/h). Reality: youth players sprint at 6–8 m/s.

## Root Cause
`config.py` uses a linear conversion: `pixels_per_metre = frame_width / FIELD_WIDTH_METERS`. This assumes the camera always shows exactly `FIELD_WIDTH_METERS` metres of pitch across the full frame width. Veo cameras crop and zoom per venue, making this assumption invalid. The visible pitch width varies shot by shot.

The correct fix is a **homography calibration** step: the user marks 4 known field landmarks (e.g. penalty spot, corner flag) in a reference frame, the system computes a perspective transform matrix mapping pixel coordinates to real-world metres. This transform is stable for a fixed-angle camera and can be computed once per venue.

## Current Understanding
- `FIELD_WIDTH_METERS = 60.0` was arrived at via autoresearch, not a manual guess. Comment in `config.py` confirms: "Tuned via autoresearch: camera shows ~60m of field width, not the full pitch."
- `SPEED_OUTLIER_CAP = 12.0 m/s` caps individual speed readings. This means max speed outputs ≤12 m/s (plausible), but **total distance is still wrong** — jitter accumulation in distance isn't capped.
- Autoresearch best score: **71.7%** of speed samples in the realistic range (6 experiments, `research/experiments/speed_calibration/`). Target is 95%+. The ceiling on the stopgap approach is confirmed.
- Current `config.py` has `SMOOTHING_WINDOW=5` — more smoothing than the autoresearch best config (`SMOOTHING_WINDOW=3`). This may slightly improve the 71.7% figure but won't close the gap to 95%.
- The "97–591 km distance" numbers from the README are pre-fix. Current distances are unknown but likely still inflated due to jitter accumulation.
- Homography is the standard approach in sports analytics. Standard field markings (centre circle diameter = 9.15m, penalty area = 40.32m × 16.5m) are reliable landmarks for Finnish youth clubs.

## What's Been Tried
- `FIELD_WIDTH_METERS` tuning via autoresearch: best value is 60.0, achieved 71.7% — ceiling reached.
- `SPEED_OUTLIER_CAP = 12.0`: caps individual speed readings but not distance accumulation.
- `SMOOTHING_WINDOW = 5`, `MIN_MOVEMENT_PX = 10`: reduce jitter contribution.
- All stopgaps together: can't reach 95% realistic-speed target without homography.

## Open Questions
- Should the calibration UI be a one-time setup per venue, or per game (in case camera is moved)?
- Which 4 landmarks are most reliably visible in Finnish youth club footage?
- Should calibration be a separate script or integrated as a Stage 0?

## References
- `research/programs/speed_calibration.md`
- `config.py` (`FIELD_WIDTH_METERS`, `SPEED_OUTLIER_CAP`)
- `pipeline/features.py` (pixel→metre conversion)
