# Eval Harness — Stage 1-2 Quality Metrics

Measures the accuracy of Stage 2 (jersey OCR + identity sweep) against
manually annotated short clips. Produces actionable metrics in <90 seconds
per clip — the fastest feedback loop for detecting regressions or measuring
improvements to `pipeline/identify.py`.

---

## Quick Start

```bash
# 1. Create a fixture from a game video
python evals/create_fixture.py \
    --video /path/to/game.mp4 \
    --start-frame 0 --end-frame 1800 \
    --target-jersey 15 \
    --output evals/fixtures/clip_001

# 2. Inspect the printed track table, then fill in ground_truth.json
#    Open evals/fixtures/clip_001/ground_truth.json
#    Set "target_track_ids" to the correct IDs (see the printed table)

# 3. Run the eval
python evals/eval_runner.py \
    --fixture evals/fixtures/clip_001 \
    --video /path/to/game.mp4 \
    --output /tmp/eval_run/

# 4. Run all unit tests (fast, no video)
pytest tests/test_eval_unit.py -v

# 5. Run integration tests against real fixtures (requires clip.mp4 + model weights)
pytest tests/test_eval.py -m slow -v
```

---

## What Gets Measured

| Metric | Formula | What it tells you |
|--------|---------|------------------|
| `track_precision` | `\|predicted ∩ expected\| / \|predicted\|` | Are wrong tracks being included? |
| `track_recall` | `\|predicted ∩ expected\| / \|expected\|` | Are target tracks being missed? |
| `track_f1` | harmonic mean of P and R | Combined score |
| `frames_covered` | `merged_frames / min_frames_visible` | What fraction of the player's visible time is captured |
| `ocr_seeds_found` | count of tracks in Stage 2 result | How many tracks the OCR sweep identified |
| `overlap_rejected` | tracks rejected by overlap guard | Is the sweep guard working? (0 if not exposed by identify.py) |

**Pass gates (both must pass):**
- `track_f1 ≥ 0.80`
- `frames_covered ≥ 0.70`

**Exit codes:**
- `0` = PASS
- `1` = FAIL (gate not met, or bad fixture/schema)
- `2` = DRIFT (detections.json was modified since annotation — re-annotate)

---

## How a Fixture Works

```
evals/fixtures/clip_001/
  ground_truth.json     ← committed to git (annotated by human)
  detections.json       ← committed to git (cached Stage 1 output)
  clip.mp4              ← NOT committed (extract with create_fixture.py)
  identity_raw.json     ← NOT committed (Stage 2 output from create_fixture.py)
```

`detections.json` is committed so Stage 2 always runs against the same
stable track IDs. The SHA-256 of `detections.json` is recorded in
`ground_truth.json["detections_sha256"]` — if Stage 1 is re-run and
the hash changes, `eval_runner.py` exits 2 (DRIFT) and stops.

---

## ground_truth.json Schema

```json
{
  "clip_name": "clip_001",
  "description": "30s window, player clearly visible, low crowd density",
  "source_video": "/absolute/path/to/game.mp4",
  "source_start_frame": 0,
  "source_end_frame": 1800,
  "frame_numbering": "clip_relative",
  "target_jersey": 15,
  "target_track_ids": [3956, 4544],
  "detections_sha256": "a3f1c9...",
  "min_frames_visible": 400,
  "notes": "Derived from output_v7 manual inspection"
}
```

**`target_track_ids`** — the correct answer: which track IDs (from the
cached `detections.json`) belong to the target player. These are
*clip-relative* IDs from the Stage 1 run on the short clip, not the
full game.

**`min_frames_visible`** — number of unique frames the target player
appears in (from the cached detections). `create_fixture.py` computes
this automatically from Stage 2's initial OCR result. You can also set it
manually: count the unique `frame_num` values across all `target_track_ids`
in `detections.json`.

**`detections_sha256`** — computed automatically by `create_fixture.py`.
Locks the fixture to a specific Stage 1 run.

---

## Creating Fixtures

`create_fixture.py` automates the tedious parts:

```
[1/4] Extracts clip.mp4 from the source game video (cv2)
[2/4] Runs Stage 1 (YOLO detection + tracking) → detections.json
[3/4] Runs Stage 2 (jersey OCR) → identity_raw.json
[4/4] Prints OCR-matched track table for --target-jersey
      Scaffolds ground_truth.json with target_track_ids=[]
```

**After running `create_fixture.py`:**

1. Review the printed track table — it shows which track IDs Stage 2
   identified for the target jersey number.
2. Optionally open `clip.mp4` to visually verify which IDs are correct.
3. Edit `ground_truth.json` and fill in `target_track_ids`:
   ```json
   "target_track_ids": [3956, 4544]
   ```
4. Commit `ground_truth.json` and `detections.json` to git.
5. Do NOT commit `clip.mp4` or `identity_raw.json` (gitignored).

**Recommended fixture types:**
- `clip_001` — easy: player clearly visible, low crowd
- `clip_002` — crowded: multiple players nearby
- `clip_003` — occlusion: player goes in/out of frame

---

## Running Evals

### Single fixture

```bash
python evals/eval_runner.py \
    --fixture evals/fixtures/clip_001 \
    --video /path/to/game.mp4 \
    --output /tmp/eval_run/
```

Example output:
```
=== Eval: clip_001 ===
  track_precision : 1.00  ✓
  track_recall    : 0.85  ✗  (missing: [4544])
  track_f1        : 0.92  ✓
  frames_covered  : 0.91  ✓  (364/400 frames)
  ocr_seeds_found : 2
  overlap_rejected: 0

FAIL  recall=0.85 — target tracks being missed
```

### All fixtures (CI)

```bash
# Schema + hash checks (fast — no model)
pytest tests/test_eval.py -v

# Full integration tests (slow — real inference)
pytest tests/test_eval.py -m slow -v
```

### Unit tests (fast, no video or model)

```bash
pytest tests/test_eval_unit.py -v
```

---

## Interpreting Results

| Metric fails | Likely cause | Where to look |
|-------------|-------------|---------------|
| `precision < 0.80` | Sweep adding wrong-player tracks | `_sweep_for_target_fragments()` in `pipeline/identify.py` — overlap guard |
| `recall < 0.80` | OCR not seeding enough tracks, or sweep too conservative | `utils/ocr.py` — OCR confidence; sweep adjacency threshold |
| `frames_covered < 0.70` | Player mostly untracked by Stage 1, or many fragments missed | Stage 1 `conf_threshold` / `iou_threshold` in `config.py` |
| `ocr_seeds_found < 2` | OCR is failing to read the jersey (see TODO-005) | `utils/ocr.py` — consider closed-vocab classifier |

---

## CI Integration

`tests/test_eval.py` discovers all fixtures dynamically — add a new fixture
directory and it is automatically tested. No changes to the test file needed.

Add to your CI pipeline:
```yaml
- name: Schema validation (fast)
  run: pytest tests/test_eval.py -v -m "not slow"

- name: Integration evals (slow, optional)
  run: pytest tests/test_eval.py -m slow -v
  # requires: clip.mp4 extracted, model weights present
```

---

## What's NOT Measured Here

- Stage 1 detection quality (YOLO confidence, tracking accuracy) — the
  fixture caches Stage 1 output, so Stage 1 changes require re-running
  `create_fixture.py` and re-annotating.
- Per-frame bounding box accuracy — this harness only measures track ID
  assignment (which tracks belong to the target player).
- Multi-player scenarios — each fixture evaluates one target player.

For Stage 1 quality, see TODO-006 (autoresearch integration) which treats
`track_f1` as the optimization metric for tuning `config.py` parameters.
