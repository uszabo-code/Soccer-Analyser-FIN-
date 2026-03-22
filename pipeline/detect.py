"""Stage 1: Player detection and tracking using YOLOv8 + BoT-SORT."""

import json
import os
from pathlib import Path

from tqdm import tqdm

import config
from models.data import Detection
from utils.video import VideoReader


def get_device():
    """Auto-detect the best available compute device."""
    if config.DEVICE != "auto":
        return config.DEVICE
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def run(video_path: str, output_dir: str, model_name: str = None,
        device: str = None, frame_skip: int = None) -> str:
    """
    Run detection and tracking on a video.

    Returns path to the output detections JSON file.
    """
    from ultralytics import YOLO

    model_name = model_name or config.YOLO_MODEL
    device = get_device() if (device is None or device == "auto") else device
    frame_skip = frame_skip or config.FRAME_SKIP

    # Resolve tracker config path — if it's a local file, use absolute path
    tracker_cfg = config.TRACKER_CONFIG
    local_tracker = Path(tracker_cfg)
    if not local_tracker.is_absolute():
        # Try relative to the project root (where config.py lives)
        project_root = Path(__file__).parent.parent
        candidate = project_root / tracker_cfg
        if candidate.exists():
            tracker_cfg = str(candidate)

    print(f"[Stage 1] Detection & Tracking")
    print(f"  Model: {model_name}")
    print(f"  Tracker: {tracker_cfg}")
    print(f"  Device: {device}")
    print(f"  Frame skip: {frame_skip}")

    # Load model
    model = YOLO(model_name)

    # Open video
    reader = VideoReader(video_path, frame_skip=frame_skip)
    print(f"  Video: {reader.width}x{reader.height} @ {reader.fps:.1f}fps")
    print(f"  Duration: {reader.duration_s:.0f}s ({reader.total_frames} frames)")
    print(f"  Processing: ~{reader.frames_to_process} frames")

    # Overlay filter boundaries
    y_min = int(reader.height * config.OVERLAY_TOP_FRACTION)
    y_max = int(reader.height * (1 - config.OVERLAY_BOTTOM_FRACTION))

    all_detections = []
    ball_detections = []

    # Support both single PERSON_CLASS_ID (legacy) and PERSON_CLASS_IDS list (football model)
    person_class_ids = getattr(config, 'PERSON_CLASS_IDS', [config.PERSON_CLASS_ID])
    ball_conf = getattr(config, 'BALL_CONFIDENCE_THRESHOLD',
                        config.CONFIDENCE_THRESHOLD * 0.5)
    person_conf_thresh = getattr(config, 'PERSON_CONFIDENCE_THRESHOLD',
                                 config.CONFIDENCE_THRESHOLD)
    all_class_ids = person_class_ids + [config.BALL_CLASS_ID]

    print(f"  Person class IDs: {person_class_ids}, Ball class ID: {config.BALL_CLASS_ID}")

    with reader:
        pbar = tqdm(total=reader.frames_to_process, desc="Detecting", unit="frame")
        for frame_num, frame in reader.iter_frames():
            # Run YOLO with tracking — use lower confidence to catch balls
            results = model.track(
                frame,
                persist=True,
                tracker=tracker_cfg,
                conf=min(config.CONFIDENCE_THRESHOLD, ball_conf),
                device=device,
                verbose=False,
                classes=all_class_ids,
            )

            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy().tolist()
                    x1, y1, x2, y2 = bbox
                    conf = float(boxes.conf[i].cpu())
                    cls_id = int(boxes.cls[i].cpu())

                    # Apply class-specific confidence thresholds
                    if cls_id in person_class_ids and conf < person_conf_thresh:
                        continue
                    if cls_id == config.BALL_CLASS_ID and conf < ball_conf:
                        continue

                    # Get track ID (may be None if tracking fails for this box)
                    track_id = -1
                    if boxes.id is not None:
                        track_id = int(boxes.id[i].cpu())

                    # Filter overlay regions (only for persons)
                    if cls_id in person_class_ids:
                        center_y = (y1 + y2) / 2
                        if center_y < y_min or center_y > y_max:
                            continue

                    det = Detection(
                        frame_num=frame_num,
                        track_id=track_id,
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=cls_id,
                    )

                    if cls_id == config.BALL_CLASS_ID:
                        ball_detections.append(det.to_dict())
                    else:
                        all_detections.append(det.to_dict())

            pbar.update(1)
        pbar.close()

    # Count unique tracks
    track_ids = set(d["track_id"] for d in all_detections if d["track_id"] >= 0)
    print(f"  Detected {len(all_detections)} person detections across {len(track_ids)} tracks")
    print(f"  Detected {len(ball_detections)} ball detections")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "detections.json")
    output_data = {
        "video_path": video_path,
        "fps": reader.fps,
        "width": reader.width,
        "height": reader.height,
        "total_frames": reader.total_frames,
        "frame_skip": frame_skip,
        "person_detections": all_detections,
        "ball_detections": ball_detections,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"  Saved to {output_path}")
    return output_path
