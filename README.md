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

```bash
export ANTHROPIC_API_KEY=your_key_here
```

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
track_buffer: 90           # 3s gap tolerance — players re-enter after occlusion
match_thresh: 0.7          # Association similarity threshold
gmc_method: sparseOptFlow  # Camera motion compensation (GMC)
with_reid: False           # ReID disabled — OSNet trained on close-up pedestrians,
                           # performs poorly on overhead 80–150px player crops
```

> **Why no ReID?** We tested BoT-SORT with `with_reid=True` and `appearance_thresh=0.25`. Track count tripled and median track length dropped to 3 frames — the OSNet appearance model made wrong decisions on overhead soccer players (out-of-distribution from its pedestrian training data) and fragmented tracks instead of merging them. GMC-only BoT-SORT gives 5 perfectly continuous tracks out of 5 on a 300-frame clip.

---

## Detection & Tracking Performance

### Model comparison (300-frame eval clip)

| Config | Ball detection | Unique tracks | Noise tracks | Perfect-continuity tracks |
|---|---|---|---|---|
| YOLOv8m COCO + ByteTrack + SKIP=3 | 0% | 262 | 63% | 0 |
| `football_yolov8.pt` + BoT-SORT + SKIP=2 | **30%** | **39** | **38%** | **5 / 5** |

### Full game results (90-minute match, FRAME_SKIP=2)

| Metric | Value |
|---|---|
| Processed frames | 18,359 |
| Person detections | 294,509 |
| Ball detections | 1,029 (5.6%) |
| Total tracks | 8,350 |
| Significant tracks (>10 detections) | 3,115 |
| Longest single track | 1,197 frames |

Track fragmentation across a full game is the main open challenge (see Known Limitations). Stage 1b post-process merging reduces this significantly.

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

**Track fragmentation:** A full 90-minute game produces ~3,100 significant tracks for ~22 players. Stage 1b merges many of these, but fragmentation remains the main accuracy bottleneck. Root cause: YOLOv8 detection drops below threshold when players overlap or cluster, causing the tracker to open new IDs on re-detection. Being addressed via Stage 1b tuning and autoresearch.

**Speed calibration:** `FIELD_WIDTH_METERS` assumes the full frame width shows exactly 60m of field. If the camera is zoomed differently, computed speeds will be wrong. A proper homography calibration tool (user marks 4 field landmarks → saves a per-venue transform matrix) is on the roadmap.

**PaddleOCR on macOS MPS:** PaddleOCR can segfault during Stage 2 on Apple Silicon when running on MPS. If this happens, force CPU for OCR by setting `use_gpu=False` in `utils/ocr.py`. OCR is CPU-bound anyway with no performance penalty.

**Ball detection on full games:** Ball detection rate drops from ~30% on short clips to ~5.6% on full games. Many frames have no visible ball (out of play, halftime, wide shots). Ball proximity and involvement stats should be interpreted with this in mind.

---

## Roadmap

- [x] Football-specific YOLOv8 detection model (vs generic COCO)
- [x] BoT-SORT with GMC, ReID disabled (47% fewer tracks, perfect continuity on eval)
- [x] FRAME_SKIP=2 tuning
- [x] Post-process track merger — Stage 1b (Union-Find, spatial + temporal + semantic constraints)
- [x] Advanced features: purposefulness, spacing, positional discipline, work rate phases
- [x] Claude LLM coaching report (3 focused API calls)
- [x] Autoresearch framework (Claude-driven parameter optimisation)
- [ ] `soccana_yolo11.pt` integration (40% ball detection on eval)
- [ ] Homography calibration tool (per-venue pixel → metres mapping)
- [ ] Click-to-identify player selection (more reliable than OCR for target player)
- [ ] HTML report (printable, self-contained, embeds charts)
- [ ] BoT-SORT parameter autoresearch (track_buffer, match_thresh tuning)
- [ ] Multi-player mode — full team report in one pass
- [ ] Season aggregation (SQLite — track player development across games)
- [ ] Shot map and pass string visualisations (Veo Analytics parity)

---

## License

MIT
