# Soccer Analyser FIN

AI-powered soccer game analysis pipeline for Finnish youth club football. Tracks an individual player across a full match, extracts performance and tactical metrics, and generates personalised coaching reports — designed for fixed-angle elevated cameras (Veo Cam 2, Wisemen, Pixellot style).

---

## What It Does

Given a game video and a jersey number, the pipeline produces:

**Physical stats**
- Total distance covered, average and max speed, time in each speed zone
- Sprint detection: count, duration, distance per sprint episode
- Fatigue curve: performance split across 15-minute segments

**Positional & tactical stats**
- Field heatmap (10×15 grid showing where the player spent time)
- Time in each field third (defensive / middle / attacking)
- Movement purposefulness score — net displacement vs total distance (direct runner vs. reactive mover)
- Spacing from teammates (average, min, nearest player distance)
- Positional discipline score — how consistently the player holds their zone
- Lateral vs. vertical movement split

**Ball involvement**
- Time spent near the ball (proximity episodes with timestamps)
- Ball contact rate relative to teammates

**AI coaching report** (Claude API)
- Personalised narrative referencing specific timestamps from the game
- 3–5 specific, actionable improvement suggestions
- Team strategy observations

---

## Pipeline Stages

```
Stage 1   Detection & Tracking     Football-specific YOLOv8 + BoT-SORT (GMC, no ReID)
Stage 1b  Track Merging            Post-process fragmentation repair via Union-Find
Stage 2   Player Identification    Jersey OCR (PaddleOCR, 3-strategy voting) + K-Means team clustering
Stage 3   Feature Extraction       Distance, speed, sprints, heatmap, key moments
Stage 3b  Advanced Features        Purposefulness, spacing, positional discipline, work rate phases
Stage 4   LLM Analysis             Claude API — 3 focused calls: player, improvements, team strategy
Stage 5   Report                   Markdown report + visualisation charts
```

Each stage writes a JSON file to `output/` and can be resumed independently with `--stage N`.

---

## Setup

### Requirements
- Python 3.9+
- macOS (MPS) or Linux (CUDA/CPU)
- ~4 GB disk for models

```bash
git clone https://github.com/uszabo-code/Soccer-Analyser-FIN-.git
cd Soccer-Analyser-FIN-
./setup.sh
```

### Models

Download model weights separately (not included in repo due to file size):

| Model | Purpose | Detection rate | Source |
|---|---|---|---|
| `football_yolov8.pt` | Player + ball detection | 21% ball (eval) / 5.6% full game | [HuggingFace](https://huggingface.co/uisikdag/yolo-v8-football-players-detection) |
| `soccana_yolo11.pt` | Ball detection (higher rate) | 40% (eval clip) | [HuggingFace](https://huggingface.co/Adit-jain/soccana) |

Place `.pt` files in the project root. The ball detection rate drop from eval clip to full game is expected — many frames have no visible ball (out of play, halftime, off-camera).

### API Key

Create a `.env` file in the project root (never commit this — it is gitignored):

```bash
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
```

The pipeline loads it automatically via `python-dotenv`. Alternatively use a shell export, but the `.env` approach keeps the key out of your shell profile and away from any work/cloud credential setup.

---

## Usage

```bash
# Analyse a full game for player #15
python analyze.py videos/game.mp4 --jersey 15

# Skip Claude analysis (faster, no API key needed)
python analyze.py videos/game.mp4 --jersey 15 --no-llm

# Resume from a specific stage (e.g. after Stage 1 is done)
python analyze.py videos/game.mp4 --jersey 15 --stage 3

# Override device and frame skip
python analyze.py videos/game.mp4 --jersey 15 --device cpu --skip 3
```

Output is saved to `output/` (configurable via `--output-dir`).

### Output Files

```
output/
  detections.json       Stage 1: all player + ball detections with track IDs
  player_identity.json  Stage 2: track → jersey number + team assignment
  player_stats.json     Stage 3: physical stats + key moments per player
  advanced_stats.json   Stage 3b: purposefulness, spacing, positional discipline
  analysis.json         Stage 4: Claude coaching report (JSON)
  report.md             Stage 5: formatted match report
```

---

## Configuration

Key settings in `config.py`:

```python
YOLO_MODEL = "football_yolov8.pt"       # Detection model
FRAME_SKIP = 2                           # Process every 2nd frame (best continuity/speed tradeoff)
TRACKER_CONFIG = "botsort_soccer.yaml"  # BoT-SORT with GMC, no ReID
FIELD_WIDTH_METERS = 60.0               # Visible field width (U13–U15 Veo camera crop)
SPRINT_SPEED_THRESHOLD = 4.0            # m/s
SPEED_OUTLIER_CAP = 12.0                # m/s — caps tracking noise spikes
```

### Tracker Config (`botsort_soccer.yaml`)

Tuned for overhead fixed-angle cameras:

```yaml
tracker_type: botsort
track_buffer: 90    # 3s gap tolerance — players re-enter after occlusion
match_thresh: 0.7   # Association similarity threshold
gmc_method: none    # GMC disabled — see note below
with_reid: False    # ReID disabled — see note below
```

> **Why no ReID?** We tested BoT-SORT with `with_reid=True` and `appearance_thresh=0.25`. Track count tripled and median track length dropped to 3 frames — the OSNet appearance model made wrong decisions on overhead soccer players (out-of-distribution from its pedestrian training data) and fragmented tracks instead of merging them.

> **Why no GMC?** BoT-SORT's `sparseOptFlow` GMC is designed for moving cameras. On a fixed overhead Veo camera it finds near-zero global motion, which corrupts Kalman filter predictions and causes 2 track IDs to "absorb" the whole game (~18,000 frames each) while everything else fragments into noise. Always set `gmc_method: none` for fixed-angle cameras.

---

## Detection & Tracking Performance

### Model comparison (300-frame eval clip)

| Config | Ball detection | Unique tracks | Noise tracks | Perfect-continuity tracks |
|---|---|---|---|---|
| YOLOv8m COCO + ByteTrack + SKIP=3 | 0% | 262 | 63% | 0 |
| `football_yolov8.pt` + BoT-SORT + SKIP=2 | **30%** | **39** | **38%** | **5 / 5** |

### Full game results (20-minute match, FRAME_SKIP=2, v5 config)

| Metric | Value |
|---|---|
| Processed frames | 18,359 |
| Person detections | 294,509 |
| Ball detections | 1,029 (5.6%) |
| Total tracks | 7,798 |
| Significant tracks (≥45 detections) | 1,540 |
| Longest single track | ~1,200 frames |

Track fragmentation across a full game is the main open challenge (see Known Limitations). With ~22 players expected, 1,540 significant tracks means ~70 fragments per player on average. Stage 1b post-process merging is intended to address this but has a known critical bug (see Known Limitations).

### FRAME_SKIP impact (300-frame test)

| FRAME_SKIP | Total tracks | Longest track | Noise tracks |
|---|---|---|---|
| 3 | 262 | 199 / 300 | 63% |
| 2 | **140** | **300 / 300** | **49%** |

FRAME_SKIP=2 cut track count by 47% and produced the first perfect-continuity track (followed a player for all 300 frames without a single ID switch).

---

## Project Structure

```
analyze.py              Entry point — orchestrates all stages
config.py               All tunable parameters
pipeline/
  detect.py             Stage 1: YOLOv8 + BoT-SORT tracking
  identify.py           Stage 2: Jersey OCR + team clustering
  reidentify.py         Stage 1b: Post-process track merger (Union-Find)
  features.py           Stage 3: Distance, speed, heatmap, key moments
  advanced_features.py  Stage 3b: Purposefulness, spacing, positional discipline
  analyze.py            Stage 4: Claude LLM analysis (3 focused API calls)
  report.py             Stage 5: Report generation
models/
  data.py               Core dataclasses (Detection, PlayerStats, KeyMoment, etc.)
utils/
  video.py              VideoReader with frame_skip and timestamp helpers
  ocr.py                PaddleOCR wrapper — 3-strategy voting for jersey OCR
research/
  runner.py             Autoresearch loop — Claude-driven parameter optimisation
  evaluate.py           Domain-specific evaluation metrics
  programs/             Research programs (ball_detection.md, speed_calibration.md, tracking_smoothness.md)
botsort_soccer.yaml     BoT-SORT tracker config (tuned for overhead soccer)
```

---

## Camera Setup

Designed for **Finnish youth club cameras** — typically elevated fixed-angle installations showing approximately 60m of field width (not the full 105m pitch):

- Overhead angle (~15–30° elevation)
- Players at ~80–150px height in frame
- Camera pan/tilt compensation handled by BoT-SORT GMC
- 1920×1080 @ 25–30fps input
- `FIELD_WIDTH_METERS = 60.0` calibrated for this framing

If your camera shows a different field width, adjust `FIELD_WIDTH_METERS` in `config.py` — this directly affects all speed and distance calculations.

---

## Why This vs Veo Analytics

Veo Analytics is the standard tool in Finnish youth clubs and a direct reference for this project. Key differences:

| | Veo Analytics | Soccer Analyser FIN |
|---|---|---|
| Player identification | Jersey OCR (documented accuracy issues) | Jersey OCR + manual fallback |
| Coaching feedback | Stats display | Claude AI — personalised, timestamped narrative |
| Tactical depth | Heatmap + possession | Purposefulness score, spacing, positional discipline, work rate phases |
| Cost | Cloud subscription | Runs locally, open source |
| Ball detection | Commercial (unreported) | 5.6% full game, 30% short clips |
| Customisation | Fixed feature set | Autoresearch framework for parameter tuning |

---

## Known Limitations

### 🔴 Critical — blocks end-to-end analysis

**Speed and distance calibration is wrong.** Current output shows total distances of 97–591 km for a 20-minute game. Root cause: `FIELD_WIDTH_METERS` is used for a simple linear pixel→metre conversion assuming the full frame width maps to exactly 60m. In practice, Veo cameras crop and zoom differently per venue, making this assumption invalid. A homography calibration step (user marks 4 known field landmarks → computes a per-venue transform matrix) is required before any physical stats are meaningful.

**Target player track is dropped by Stage 3.** When the user selects a player via the interactive picker, the selected track ID typically has too few detections to meet Stage 3's minimum threshold. The player is identified correctly in Stage 2 but silently skipped in Stage 3. Root cause: the selected player is fragmented across many small tracks; Stage 1b must successfully merge those fragments before Stage 3 runs.

**Stage 1b (track merger) has a critical bug.** The Union-Find merger currently chain-merges far too aggressively, absorbing entire teams into single mega-tracks. This was caught in v4 testing and Stage 1b is currently disabled with `--no-merge`. Fixing the merger's spatial/temporal constraints is required before end-to-end analysis works.

### 🟡 Significant — degrades output quality

**Track fragmentation.** A 20-minute game produces ~1,540 significant tracks for ~22 players (~70 fragments per player). Root cause: YOLOv8 detection confidence drops during player clustering and occlusions, causing the tracker to open new IDs on re-detection. Fragmentation directly blocks target player analysis (see above).

**Jersey OCR unreliable on overhead footage.** PaddleOCR and EasyOCR were trained on document/close-up text. Overhead soccer footage gives tiny (~80–150px) players with jerseys at oblique angles — jersey numbers are often not visible at all. OCR is the current primary identification method but may need to be replaced or supplemented with appearance-based re-identification or manual selection only.

**Ball detection drops to 5.6% on full games.** Detection rate drops from ~30% on short clips to 5.6% on full games. Many frames have no visible ball (out of play, halftime, wide shots). Ball proximity and involvement stats should be interpreted with this in mind.

### 🟢 Minor — workarounds available

**PaddleOCR on macOS MPS.** PaddleOCR v3.x can segfault on Apple Silicon MPS. Force CPU (`device="cpu"` in `utils/ocr.py`) — OCR is CPU-bound with no real performance penalty.

**ANTHROPIC_API_KEY must be in environment.** The pipeline does not auto-load `.env` unless `python-dotenv` is installed and explicitly called. If Stage 4 is skipped with "Set ANTHROPIC_API_KEY", verify the key is exported in the shell running the pipeline.

---

## Roadmap

### Done
- [x] Football-specific YOLOv8 detection model (vs generic COCO weights)
- [x] BoT-SORT with ReID disabled (47% fewer tracks, perfect continuity on 300-frame eval)
- [x] GMC disabled for fixed-angle cameras (prevents mega-track absorption bug)
- [x] FRAME_SKIP=2 (47% track reduction, first perfect-continuity track on eval)
- [x] Post-process track merger — Stage 1b skeleton (Union-Find, spatial + temporal + semantic)
- [x] Interactive click-to-identify player selection (Stage 2 fallback when OCR fails)
- [x] Advanced features: purposefulness, spacing, positional discipline, work rate phases
- [x] Claude LLM coaching report (3 focused API calls)
- [x] Autoresearch framework (Claude-driven parameter optimisation)

### Critical path — required before end-to-end analysis works
- [ ] **Fix Stage 1b track merger** — chain-merge bug absorbs entire teams; needs constrained Union-Find with proper gap/distance thresholds
- [ ] **Homography calibration tool** — per-venue pixel→metre mapping; current linear approximation produces 97–591km distances on a 20-min game
- [ ] **Stage 3 minimum threshold** — target player track silently dropped if below detection count minimum; fix or lower threshold post-merge

### High priority
- [ ] **Player identification rethink** — OCR unreliable on overhead footage (jersey rarely visible); evaluate: appearance clustering, manual one-time selection per game, colour + position heuristics
- [ ] `soccana_yolo11.pt` integration (40% ball detection on eval vs 5.6% on full game)
- [ ] BoT-SORT parameter autoresearch (track_buffer, match_thresh, new_track_thresh tuning on real game footage)

### Nice to have
- [ ] HTML report (printable, self-contained, embeds charts and heatmap)
- [ ] Multi-player mode — full team report in one pass
- [ ] Season aggregation (SQLite — track player development across games)
- [ ] Shot map and pass string visualisations (Veo Analytics parity)

---

## License

MIT
