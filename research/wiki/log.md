# Wiki Log

Append-only. Each entry: `## [YYYY-MM-DD] <operation> | <title>`

---

## [2026-04-19] explore | Jersey OCR Unreliable

Read utils/ocr.py, pipeline/identify.py, and measured GT bbox heights (180 frames). Critical finding: median player height is 77px vs MIN_PLAYER_HEIGHT_PX=120, so OCR fires on only 1% of target player detections. OCR is effectively disabled by player size, not model quality. Team color clustering (black vs white) works fine. Interactive picker already confirmed removed.

## [2026-04-19] source | Ball color confirmed from game screenshot

Ball is white (standard outdoor). CV detector (HSV yellow-green mask) produces zero detections on this footage. Ensemble degrades to YOLO-only. Updated ball-detection-drop.md.

## [2026-04-19] explore | Ball Detection Rate Drop

Read cv_ball_detector.py, ball_tracker.py, evaluate.py, and 16 ball_detection experiments. Key findings: CV detector is yellow-green futsal ball only (HSV mask) — won't fire on white balls; 75.8% GT recall vs 15% YOLO alone; autoresearch composite ceiling 41.1% at FRAME_SKIP=1; Kalman gap interpolation now integrated. Critical open question: what colour is the ball in these games?

## [2026-04-19] explore | Track Fragmentation

Read botsort_soccer.yaml and tracking smoothness experiments. Wiki was stale: track_buffer is now 150 (not 90), match_thresh is 0.65 (not 0.7) — both updated in 2de18a6. Tracking smoothness autoresearch peaked at 85.1% matching current config. Full-game fragmentation not re-measured since param changes. Key open question: what is the post-2de18a6 fragmentation count?

## [2026-04-19] explore | Speed & Distance Calibration

Read config.py, features.py, advanced_features.py, and speed_calibration experiments. Key findings: FIELD_WIDTH_METERS=60.0 was autoresearch-tuned (not a guess); best stopgap score is 71.7% realistic speeds (6 experiments); SPEED_OUTLIER_CAP caps speed readings but not distance accumulation. Homography still the required fix. Updated wiki page with confirmed numbers.

## [2026-04-19] explore | Stage 3 Drops Target Player

Read pipeline/identify.py + features.py. Wiki description was wrong: Stage 3 does NOT silently skip the target — it force-includes target_track_ids and produces partial stats. The real problem is that target stats cover only 0–5 OCR-identified fragments out of ~70. Interactive picker is gone from code (replaced with largest-track fallback). Fix still requires Stage 1b.

## [2026-04-19] explore | Stage 1b Track Merger Bug

Read pipeline/reidentify.py — fix was already implemented in 2de18a6 (MAX_GAP_FRAMES 900→150, walking-speed model, group cap, direction check). Wiki page was stale. Updated status to in-progress; key open question is whether MAX_MERGED_PER_GROUP=5 is too tight for ~70-fragment players.

## [2026-04-19] init | Wiki scaffolded

Initial wiki created. Six problems pages seeded from README Known Limitations. No sources ingested yet.
