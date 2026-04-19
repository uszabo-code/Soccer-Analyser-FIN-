# TODOS — Soccer Analyser FIN

Items captured during engineering review (2026-03-22). Each item has enough context
to be picked up cold in a future session.

---

## TODO-001: Interactive homography calibration tool

**What:** User clicks 4 known field landmarks (penalty spot, centre spot, 2 corner
flags) on first run at a new venue. System computes a perspective transform matrix and
saves it to `calibrations/{venue}.npz`. All future runs at that venue use the saved
matrix for pixel-perfect speed and distance calculations.

**Why:** The HSV auto-detect field width (shipped in this PR) achieves ~85% accuracy
for standard green pitches. For competitive accuracy (distance ±5% instead of ±15%),
homography is required. Also handles non-rectangular camera angles.

**Pros:** Pixel-accurate speed/distance. Works for any camera angle, zoom, or pitch
size. Per-venue calibration file is reusable across all games at the same ground.

**Cons:** Requires a one-time 30-second user interaction per venue. Needs a simple UI
(click on frame). More complex to implement than the HSV approach.

**Context:** The current implementation uses `px_per_meter = frame_width /
config.FIELD_WIDTH_METERS`. This was replaced in this PR by HSV auto-detection
(`auto_detect_field_width()` in `pipeline/features.py`). The homography tool is the
next level up — it uses OpenCV `getPerspectiveTransform` on 4 user-clicked points.
Reference: OpenCV docs on `findHomography` and `perspectiveTransform`. The calibration
file should store the matrix as a numpy `.npz` so it loads with `np.load()`.

**Depends on:** HSV auto-detect (this PR) must ship first as the fallback when no
calibration file exists for the venue.

**Effort:** human ~1 week / CC ~45 min

---

## TODO-002: Appearance-based player re-identification

**What:** Replace jersey OCR as the primary player identification method. Extract HSV
colour histogram from each track's median frame. Cluster histograms into 2 teams using
K-Means. Present the user with a grid of representative player crops (one per cluster
centroid). User clicks their child once. System re-identifies all fragments that match
the clicked appearance via cosine similarity on HSV histograms.

**Why:** Jersey OCR fails on overhead footage — jersey numbers are tiny (~80–150px
player height), frequently not visible (player turned, running, occluded). The
interactive picker currently selects a single track, which may be a small fragment.
Appearance-based re-ID would identify ALL fragments that look like the target player
and hand them to Stage 1b for merging.

**Pros:** Works regardless of jersey number visibility. Naturally integrates with
Stage 1b — same colour histograms used for both team clustering and track merging.
No training data required. Works out-of-the-box with standard kit colours.

**Cons:** Fails when teams wear similar colours (e.g. both teams in white with
different patterns — overhead camera can't see pattern). Requires the target player
to have a distinct enough kit from opponents.

**Context:** Current Stage 2 (`pipeline/identify.py`) runs PaddleOCR on 1,540 tracks,
then falls back to interactive picker. After Stage 1b is fixed, the number of tracks
should drop to ~50–100 meaningful tracks. At that scale, presenting a grid of crops
is practical. The HSV histogram approach is Layer 1 (standard computer vision) —
use `cv2.calcHist` on the HSV channels of each player's median bounding box crop.
Cluster with `sklearn.cluster.KMeans(n_clusters=2)`.

**Depends on:** Stage 1b merger fix (this PR). Should be implemented after verifying
that merged tracks have reasonable detection counts.

**Effort:** human ~1 week / CC ~30 min

---

## TODO-003: HSV field detector — non-green pitch fallback warning

**What:** When `auto_detect_field_width()` fails to find a green pitch region (e.g.,
artificial turf, indoor hall, poor lighting), it currently falls back silently to
`config.FIELD_WIDTH_METERS = 60.0` and produces wrong speed/distance data. Should:
1. Detect the failure (green region below a confidence threshold)
2. Print a clear warning: "WARNING: Could not auto-detect field width. Using
   FIELD_WIDTH_METERS=60.0 from config. Override with --field-width N."
3. Add a `calibration_confidence` field to `player_stats.json` so the report can
   show a calibration warning when stats are based on a fallback.

**Why:** Silent wrong data is worse than a visible error. This was flagged as a
critical gap in the engineering review — a user analysing an artificial turf game
would get 97km distance outputs with no indication that calibration failed.

**Pros:** Makes the system trustworthy. User knows when to override.

**Cons:** Trivial — this is a 5-minute CC task. No cons.

**Context:** `auto_detect_field_width()` is being added to `pipeline/features.py`
in this PR. The confidence metric is simple: if the largest green HSV region covers
less than 20% of the frame area, calibration is unreliable. The `--field-width`
CLI arg doesn't exist yet — would need to be added to `analyze.py`'s argparse.

**Depends on:** HSV auto-detect (this PR).

**Effort:** human ~2 hours / CC ~5 min

---

*Captured by /plan-eng-review — 2026-03-22*

---

## TODO-004: Per-player work rate breakdown in player_stats.json

**What:** Add `work_rate` dict to each player's entry in `player_stats.json`:
`{"idle": 45.2, "jogging": 33.1, "running": 14.7, "sprinting": 7.0}` (percent of
visible frames). Computed in `pipeline/features.py` from per-frame speed vs.
`config.SPRINT_SPEED_THRESHOLD` and configurable jog/run thresholds.

**Why:** `advanced_stats.json → work_rate_phases` is a whole-game aggregate (all
players combined). `pdf_report.py` currently derives per-player work rate inline
by approximating from sprint_count and total_time_visible_s. The inline derivation
is a workaround — the correct data should live in `player_stats.json` alongside
sprint_count, distance, and speed.

**Pros:** Single source of truth for per-player work rate. `pdf_report.py` becomes
simpler (reads the field directly instead of computing it). Enables future features
that need per-player effort zones (e.g., team comparison charts).

**Cons:** Minor change to features.py and player_stats.json schema. No downstream
breaks expected — it's a new key, not a modification to existing keys.

**Context:** `pipeline/features.py` already tracks per-frame speed as part of
distance calculation. The work rate breakdown is just binning those per-frame
speeds into 4 zones: idle (<0.5 m/s), jogging (0.5–2.5 m/s), running (2.5–4.0 m/s),
sprinting (>config.SPRINT_SPEED_THRESHOLD = 4.0 m/s). Thresholds should be
configurable via config.py. Once added, `pdf_report.py` can remove its inline
derivation and just read `target_stats["work_rate"]`.

**Depends on:** pdf_report.py (this PR) ships first with the workaround. Update
in a follow-up PR.

**Effort:** human ~2 hours / CC ~5 min

---

*Captured by /plan-eng-review — 2026-03-23*

---

## TODO-005: Closed-vocab OCR classifier for jersey numbers (Approach B)

**What:** Replace PaddleOCR's open-ended text recognition in `utils/ocr.py` with a
closed-vocabulary approach: digit-regex filtering (`r'^\d{1,2}$'`) + confidence-weighted
temporal voting across all frames in a tracklet, treating jersey recognition as a
classification problem over 0–99 rather than arbitrary text.

**Why:** The eval harness (TODO-005's prerequisite) will surface `ocr_seeds_found` per
fixture. If that metric is consistently low (< 2 seeds per clip), it's direct evidence
that PaddleOCR's general OCR is the bottleneck. Literature confirms: 87% accuracy on
clean soccer tracklets (Koshkina et al., CVPR 2024), lower on motion-blurred frames.
A closed-vocab approach should hit >95% by eliminating false positives (non-digit
OCR reads) and aggregating signal across frames rather than committing to a single read.

**Pros:** Fixes the root cause without new model weights. Uses existing PaddleOCR
infrastructure, just adds post-filtering. Directly improves OCR seed quality which
is the upstream input to the identity sweep.

**Cons:** Still relies on PaddleOCR for the initial read; if the digit is too small
(<30px height), no filtering approach reliably works. Super-resolution preprocessing
would be a follow-up, not a fix.

**Context:** Current code in `utils/ocr.py` (`read_jersey_number_multi`) already
aggregates reads across multiple frames via a `Counter` — the structure is right.
The fix is to: (1) filter each read to match `r'^\d{1,2}$'` before adding to the
Counter, (2) weight votes by PaddleOCR confidence score, (3) reject any result where
the top-voted number has < 3 supporting frames. This is ~20 lines of change.

**Depends on / blocked by:** Eval harness (the `evals/` PR) must exist first so the
improvement can be measured. Trigger condition: `ocr_seeds_found < 2` consistently
across 3+ fixtures.

**Effort:** human ~1 week / CC ~1 hour

---

## TODO-006: Autoresearch integration for Stage 1 parameter tuning

**What:** Once the eval harness exists and `eval_runner.py` produces a reliable scalar
(`track_f1`), wire it as the optimization metric for karpathy/autoresearch to
autonomously tune Stage 1 detection parameters: YOLO `conf_threshold`, `iou_threshold`,
and `frame_skip` in `config.py`.

**Why:** The Karpathy Loop (autoresearch) requires exactly three things: one editable
file, one scalar metric, and a cycle time < 10 minutes. After this PR: `config.py` is
the editable file, `track_f1` from `eval_runner.py` is the scalar, and a 30-second
clip runs Stage 1+2 in ~60s. All three conditions are met.

**Pros:** Autonomous parameter search at near-zero marginal cost. Can find non-obvious
interactions (e.g. high `conf_threshold` + low `frame_skip` may outperform the reverse)
without manual grid search. Runs overnight on a laptop.

**Cons:** autoresearch works best when the eval metric is fast (<5 min). If Stage 1
(YOLO inference) is slow on CPU-only, cycles will be long. Requires `pip install
autoresearch` and an OpenAI API key (autoresearch uses GPT-4 to propose code changes).
Config-only tuning avoids the multi-file context problem.

**Context:** See karpathy/autoresearch on GitHub. The implementation would be a
thin wrapper: `autoresearch_runner.py` that (1) calls `config.py` as the target,
(2) invokes `eval_runner.py --fixture evals/fixtures/clip_001` as the eval,
(3) passes `track_f1` from stdout as the metric. The autoresearch agent edits
`config.py` and iterates.

**Depends on / blocked by:** Eval harness (this PR). Requires autoresearch installed
and OpenAI key. Recommend running with clip_001 only initially (cheapest cycle time).

**Effort:** human ~2 days / CC ~45 min
