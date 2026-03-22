# Research Program: Player Re-ID (Track Fragmentation Reduction)

## Goal
Reduce the number of distinct track IDs assigned to the same player across a game.
Player #15 currently fragments into 22 separate tracks in a single game.
Target: ≤5 tracks per player for the top 11 players per team.

## Metric
`fragmentation_score` = (total distinct track IDs) / (expected number of players)
- Lower is better (ideal = 1.0, meaning each player has exactly 1 track)
- Measure on the 30-second eval clip: research/eval_clip.mp4

## Current Baseline
- ByteTrack: fragmentation_score ≈ 3.8 (22 fragments for 6 expected players in clip)
- BoT-SORT baseline (your starting point): measure first before tuning

## Tracker Config Location
`/Users/uszabo/Downloads/soccer-analyzer/botsort_soccer.yaml`

## Tunable Parameters
```
track_buffer: [30, 60, 90, 120, 180]   # Frames to keep lost tracks alive
match_thresh: [0.5, 0.6, 0.7, 0.8]    # Association similarity threshold
appearance_thresh: [0.1, 0.25, 0.4, 0.6, 0.8]  # ReID appearance similarity
proximity_thresh: [0.3, 0.5, 0.7]     # Min IoU for ReID consideration
with_reid: [True, False]               # Enable/disable ReID model
gmc_method: [sparseOptFlow, orb, none] # Global motion compensation method
```

## Strategy
1. Start by measuring the baseline fragmentation with default BoT-SORT settings
2. The most impactful parameter is likely `track_buffer` — increase it to allow longer occlusions
3. `appearance_thresh` controls how strict ReID matching is — lower = more permissive merging
4. `gmc_method` matters for overhead camera with pan/tilt — test sparseOptFlow vs orb
5. If `with_reid=True` causes errors or is too slow, fall back to `with_reid=False`

## How to Measure Fragmentation
Run Stage 1 detection on the eval clip, then count unique track IDs vs expected players:

```python
import json
with open('research/output_clip/detections.json') as f:
    d = json.load(f)
from collections import defaultdict
tracks = defaultdict(int)
for det in d['person_detections']:
    tracks[det['track_id']] += 1
# Filter to significant tracks (>10 detections in 30s clip)
significant = {k: v for k, v in tracks.items() if v > 10}
print(f"Significant tracks: {len(significant)}")
# Expected: ~22 players on field → ideal score = 22/len(significant)
```

## Notes
- The eval clip is 30 seconds from a section where #15 is visible
- The goal is to reduce total significant tracks toward the expected number of players (~22)
- Track merging in Stage 1b handles residual fragmentation, so BoT-SORT doesn't need to be perfect
- Prioritize NOT increasing false merges (two different players sharing one track) — fragmentation is better than identity confusion
