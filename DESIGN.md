# Soccer Analyser FIN — Design Document

**Status:** Post-v5 baseline run. Pipeline runs end-to-end but three critical issues block meaningful output.
**Last updated:** 2026-03-22

---

## 1. Product Goal

Given a video recording of a youth soccer game and a target player, produce a personalised coaching report with physical stats, positional analysis, and AI-generated improvement suggestions — without requiring a cloud subscription or manual annotation.

Primary user: a parent or coach at a Finnish youth club, using a Veo Cam 2 or similar fixed-angle elevated camera.

---

## 2. Pipeline Architecture

```
Video (.mp4)
    │
    ▼
[Stage 1]  Detection & Tracking
           football_yolov8.pt (YOLOv8) + BoT-SORT
           → detections.json  (all tracks + ball detections)
    │
    ▼
[Stage 1b] Track Merger  ← BROKEN (disabled)
           Union-Find post-process to stitch fragmented tracks
           → track_merge_map.json
    │
    ▼
[Stage 2]  Player Identification
           Jersey OCR (PaddleOCR) + K-Means team clustering
           Interactive picker fallback
           → player_identity.json  (target track ID + team assignments)
    │
    ▼
[Stage 3]  Feature Extraction
           Distance, speed, sprints, heatmap, key moments
           → player_stats.json
    │
    ▼
[Stage 3b] Advanced Features
           Purposefulness, spacing, positional discipline, work rate
           → advanced_stats.json
    │
    ▼
[Stage 4]  LLM Analysis
           Claude API — 3 calls: player summary, improvements, team strategy
           → analysis.json
    │
    ▼
[Stage 5]  Report
           Markdown + charts
           → report.md
```

Each stage writes a JSON file and can be resumed independently with `--stage N`.

---

## 3. Key Design Decisions

### 3.1 Detection model: football_yolov8.pt (not YOLOv8m COCO)

**Decision:** Use a football-specific YOLOv8 model instead of generic COCO weights.

**Evidence:** On a 300-frame eval clip:
- COCO weights: 0% ball detection, 262 tracks, 63% noise, 0 perfect-continuity tracks
- football_yolov8.pt: 30% ball detection, 39 tracks, 38% noise, 5/5 perfect-continuity tracks

Ball detection improvement is the most significant change — enables ball involvement, pass proximity, and shot detection features.

### 3.2 Tracker: BoT-SORT, no ReID, no GMC

**Decision:** BoT-SORT with `with_reid=False` and `gmc_method: none`.

**ReID disabled:** OSNet ReID was trained on close-up pedestrian datasets (Market-1501, DukeMTMC). Overhead soccer players at 80–150px are completely out-of-distribution. Testing with `appearance_thresh=0.25` tripled track count and reduced median track length to 3 frames. ReID was actively fragmenting tracks.

**GMC disabled:** `sparseOptFlow` GMC is designed for moving cameras. On a fixed overhead Veo camera it finds near-zero global motion, which corrupts Kalman filter predictions. In v4 testing, GMC caused 2 track IDs to absorb the entire game (~18,000 frames each) while 77% of all other tracks became noise (<10 frames). Always set `gmc_method: none` for fixed-angle cameras.

**Result:** BoT-SORT with spatial-only IOU matching on a fixed camera. Simple and correct for this use case.

### 3.3 FRAME_SKIP=2 (not 3)

**Decision:** Process every 2nd frame, not every 3rd.

**Evidence:** On a 300-frame eval clip:
- FRAME_SKIP=3: 262 tracks, longest track 199/300, 63% noise
- FRAME_SKIP=2: 140 tracks (-47%), longest track 300/300 (perfect), 49% noise

FRAME_SKIP=3 means 120ms between processed frames. If YOLOv8 misses a detection for even 2 consecutive source frames, that's a 240ms gap — enough for BoT-SORT to open a new track ID. FRAME_SKIP=2 halves that gap for only 1.5× compute cost.

### 3.4 Player identification: OCR + interactive picker

**Decision:** Primary = jersey number OCR (PaddleOCR, 3-strategy voting). Fallback = interactive click-to-identify picker.

**Known problem:** OCR is unreliable on overhead footage. Jersey numbers are small, angled, and frequently not visible (player turned away, ball tight). In testing, OCR identified the target player in Stage 2 but the selected track had too few detections to survive Stage 3's minimum threshold — because fragmentation means the "real" player is spread across many small track IDs.

**Open question:** Should identification shift to appearance clustering, positional heuristics, or a one-time manual selection at game start? See Section 5.3.

### 3.5 Speed/distance: linear pixel→metre conversion

**Decision (temporary):** `total_distance = pixel_displacement × (FIELD_WIDTH_METERS / frame_width_pixels)`

**Problem:** This assumes the full frame width maps to exactly `FIELD_WIDTH_METERS` of real-world distance. In practice, Veo cameras crop and zoom differently per venue. v5 testing produced 97–591 km distances for a 20-minute game, confirming the conversion is completely wrong.

**Required fix:** Homography calibration — user marks 4 known field landmarks (penalty spots, centre circle, corner flags), the system computes a perspective transform matrix, and all pixel→metre conversions use that matrix. This is a per-venue one-time setup step.

---

## 4. What the v5 Baseline Run Showed

End-to-end run on `KUP 1H.mp4` (20-minute youth match, Veo Cam 2):

| Stage | Status | Notes |
|---|---|---|
| Stage 1 | ✅ Completed | 7,798 tracks, 1,540 significant (≥45 dets), 5.6% ball detection |
| Stage 1b | ❌ Disabled | Chain-merge bug — was absorbing entire teams |
| Stage 2 | ✅ Completed | Interactive picker selected track 268 (team_b); OCR found no jersey |
| Stage 3 | ⚠️ Partial | Target player (track 268) silently skipped — too few detections; only 2 mega-tracks analysed |
| Stage 4 | ❌ Skipped | ANTHROPIC_API_KEY not loaded from .env |
| Stage 5 | ⚠️ Empty report | "No player statistics available" for target player |

**Root cause chain:**
1. Stage 1 produces too-fragmented tracks
2. Target player is spread across many small fragments
3. Stage 1b (which should merge them) is disabled/broken
4. The single selected fragment (track 268) has too few detections
5. Stage 3 skips it silently
6. Stage 4+5 have nothing to work with

---

## 5. Open Problems

### 5.1 🔴 Stage 1b Track Merger (Critical)

**Problem:** The Union-Find merger chain-links tracks too aggressively. In v4, it merged entire teams into 2 mega-tracks with 18,000+ detections each.

**Root cause hypothesis:** The spatial distance threshold is too loose, and the merger doesn't enforce team colour consistency during merging. Tracks from different players (same team, similar colours) get chain-linked.

**Design options:**
- **Option A:** Tighten spatial threshold + enforce colour channel distance between merge candidates
- **Option B:** Add velocity/direction consistency check — a player can't teleport across the field
- **Option C:** Limit merger to tracks within a temporal gap (e.g. ≤90 frames apart) AND spatial proximity AND same team colour
- **Recommended:** Option C with all three constraints. Currently only temporal gap is checked.

### 5.2 🔴 Speed/Distance Calibration (Critical)

**Problem:** v5 produces 97–591 km total distances for a 20-minute game. The linear `FIELD_WIDTH_METERS` conversion is wrong.

**Design options:**
- **Option A (current):** User sets `FIELD_WIDTH_METERS` in config. Simple, no calibration UI required. Still wrong unless camera framing is known.
- **Option B:** Homography calibration — user marks 4 field landmarks at first run per venue (e.g. click penalty spot, centre spot, two corner flags). System saves a `{venue}.npz` transform file. All future runs at that venue use the saved transform.
- **Option C:** Auto-detect field lines using green channel segmentation + Hough transforms to infer field boundaries.
- **Recommended:** Option B. One-time setup per venue, accurate, no ML required, straightforward to implement.

### 5.3 🟡 Player Identification Reliability (High Priority)

**Problem:** Jersey OCR fails frequently on overhead footage. The jersey number is often not visible (player turned, occluded, running).

**Current approach:** 3-strategy OCR voting (full crop, enhanced contrast, upper half only). Works occasionally, unreliable at scale.

**Design options:**
- **Option A (current):** OCR primary, interactive picker fallback. Problem: even when picker selects the right track, fragmentation means the selected track is too small to analyze.
- **Option B:** Appearance clustering. K-Means or DBSCAN on colour histograms of player crops. Group into 2 teams. Present the user with representative crops per team → user clicks their child once → system re-identifies across all fragments via colour similarity. Works regardless of jersey number visibility.
- **Option C:** Position + kit colour heuristic. If the user knows their child plays left midfield, filter to the left third and cluster by kit colour. Rough but fast.
- **Option D:** Fine-tuned number detection. Train a small classifier on jersey numbers from overhead soccer footage. Expensive to build but solves the problem permanently.
- **Recommended:** Option B as the new primary. One interactive selection per game, appearance-based re-identification across all fragments. Naturally handles the fragmentation problem if the appearance model is consistent enough.

### 5.4 🟡 Target Player Below Stage 3 Threshold (High Priority)

**Problem:** Even when Stage 2 correctly identifies the target track, Stage 3 silently skips it if it has too few detections. The user sees "No player statistics available" with no explanation.

**Fix options:**
- Lower Stage 3's minimum detection threshold (but produces noisy stats on very short tracks)
- Show a diagnostic when the target track is below threshold: "Target player track has only N detections. Run with --no-merge disabled and re-run Stage 1b."
- After Stage 1b merge, re-run Stage 2 identification on the merged tracks before Stage 3

### 5.5 🟢 ANTHROPIC_API_KEY Loading (Minor)

**Problem:** Stage 4 skipped because .env file not loaded. Requires `python-dotenv` and explicit load call.

**Fix:** Add `load_dotenv()` at the top of `analyze.py`. One line.

---

## 6. Player Identification Alternatives — Research Questions

Given the overhead footage constraints, the following alternatives to OCR warrant investigation before the next engineering cycle:

1. **Colour histogram re-identification:** Extract HSV histogram from each track's median frame. Cluster into 2 teams. Present representative crops to user → one click → reidentify across all tracks. Complexity: medium. Accuracy: high for clearly coloured jerseys.

2. **Temporal continuity merge:** If two tracks are <N frames apart and within X pixels, assume same player (current Stage 1b approach — broken implementation, sound concept). Key is bounding the constraints properly.

3. **Position-based filtering:** User specifies player's position (e.g. "left back"). Filter candidate tracks to those spending ≥60% of time in the defensive third, left half. Narrows candidates from 1,540 to ~50, then apply appearance clustering. Complexity: low. Useful as a pre-filter.

4. **Number detector fine-tuned on overhead data:** If 50–100 labelled overhead jersey crops can be collected, a small classifier (EfficientNet-B0) could reach 80%+ accuracy. Long-term investment, high payoff.

---

## 7. Technical Constraints

- **Platform:** macOS (Apple Silicon MPS) primary; Linux (CUDA) supported
- **Runtime target:** Full 20-min game analysis in <10 minutes
- **Memory:** Models load once, pipeline runs sequentially to manage memory
- **No GPU required:** CPU fallback functional; MPS preferred for detection speed
- **Input format:** Any FFmpeg-readable video; tested on Veo Cam 2 at 1920×1080 25fps
- **Output:** Markdown report + JSON files; HTML report on roadmap

---

## 8. What Good Looks Like (Success Criteria)

For the next working version to be considered a successful v1:

1. End-to-end pipeline produces a report for the target player (no silent skips)
2. Total distance ±20% of real value (requires homography calibration)
3. Average speed in range 1.5–3.0 m/s (reasonable for youth soccer)
4. Sprint count ±50% of manual count (reasonable for automated detection)
5. Target player correctly identified without jersey number OCR in ≥80% of test videos
6. Stage 1b reduces ~70 fragments/player to <5 without creating mega-tracks
7. Full pipeline runtime ≤10 minutes on a 20-minute game on M-series Mac

---

*Generated from v5 baseline testing results — 2026-03-22*
