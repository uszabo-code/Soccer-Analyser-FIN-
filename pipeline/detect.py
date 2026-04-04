"""Stage 1: Player detection and tracking using YOLOv8 + BoT-SORT."""

import json
import os
from pathlib import Path

from tqdm import tqdm

import config
from models.data import Detection
from utils.video import VideoReader
from pipeline.cv_ball_detector import CVBallGate, detect_hough_candidates, filter_static_ball_detections


def _interpolate_track_gaps(person_detections: list, max_gap: int = 3) -> list:
    """
    Fill very short gaps (≤ max_gap processed frames) within each track by linear
    interpolation. Helps smoothness score and downstream distance calculations.

    Adds synthetic detections flagged with 'interpolated': True.
    """
    import math
    from collections import defaultdict

    tracks = defaultdict(list)
    for d in person_detections:
        tid = d.get("track_id", -1)
        if tid >= 0:
            tracks[tid].append(d)

    synthetic = []
    for tid, dets in tracks.items():
        dets.sort(key=lambda d: d["frame_num"])
        for i in range(1, len(dets)):
            prev = dets[i - 1]
            curr = dets[i]
            gap = curr["frame_num"] - prev["frame_num"]
            if gap <= 1 or gap > max_gap * 2 + 1:
                # gap=1 means consecutive frames (no gap); gap too large = don't interpolate
                continue
            # Interpolate intermediate frames
            p_bbox = prev["bbox"]
            c_bbox = curr["bbox"]
            for step in range(1, gap):
                t = step / gap
                interp_bbox = [
                    p_bbox[j] + t * (c_bbox[j] - p_bbox[j])
                    for j in range(4)
                ]
                interp_frame = prev["frame_num"] + step
                synthetic.append({
                    "frame_num": interp_frame,
                    "track_id": tid,
                    "bbox": interp_bbox,
                    "confidence": min(prev["confidence"], curr["confidence"]),
                    "class_id": prev["class_id"],
                    "interpolated": True,
                })

    if synthetic:
        combined = person_detections + synthetic
        combined.sort(key=lambda d: (d["frame_num"], d["track_id"]))
        return combined
    return person_detections


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

    # Validate class ID mapping — catch mismatch between config and loaded model early
    model_names = model.names  # {class_id: class_name}
    print(f"  Model class names: {model_names}")
    ball_class_name = model_names.get(config.BALL_CLASS_ID, "NOT FOUND")
    if "ball" not in ball_class_name.lower():
        print(f"  ⚠️  WARNING: BALL_CLASS_ID={config.BALL_CLASS_ID} maps to "
              f"'{ball_class_name}' — check config.py BALL_CLASS_ID matches your model!")
    else:
        print(f"  ✓ Ball class ID {config.BALL_CLASS_ID} → '{ball_class_name}'")

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

    # Load optional ensemble ball model
    ensemble_model = None
    ensemble_ball_cls = getattr(config, 'ENSEMBLE_BALL_CLASS_ID', 1)
    ensemble_iou = getattr(config, 'ENSEMBLE_BALL_IOU_THRESHOLD', 0.30)
    ensemble_model_name = getattr(config, 'ENSEMBLE_BALL_MODEL', None)
    if ensemble_model_name:
        import os as _os
        if _os.path.exists(ensemble_model_name):
            ensemble_model = YOLO(ensemble_model_name)
            print(f"  Ensemble ball model: {ensemble_model_name} (ball class={ensemble_ball_cls})")
        else:
            print(f"  ⚠️  Ensemble model not found: {ensemble_model_name} — skipping ensemble")

    # CV-based ball gate — tracks ball trajectory to pick the best Hough circle
    cv_gate = CVBallGate()

    with reader:
        pbar = tqdm(total=reader.frames_to_process, desc="Detecting", unit="frame")
        for frame_num, frame in reader.iter_frames():
            # Run YOLO tracking for PERSONS only at person threshold.
            # Keeping tracker confidence high prevents low-confidence noise detections
            # from polluting BoT-SORT with spurious tracks that fragment immediately.
            results = model.track(
                frame,
                persist=True,
                tracker=tracker_cfg,
                conf=person_conf_thresh,
                device=device,
                verbose=False,
                classes=person_class_ids,
            )

            # Separate low-confidence ball predict (no tracking — ball has no persistent ID)
            ball_results = model.predict(
                frame,
                conf=ball_conf,
                device=device,
                verbose=False,
                classes=[config.BALL_CLASS_ID],
            )

            # --- Process person tracking results ---
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy().tolist()
                    x1, y1, x2, y2 = bbox
                    conf = float(boxes.conf[i].cpu())
                    cls_id = int(boxes.cls[i].cpu())

                    if cls_id not in person_class_ids:
                        continue  # tracker only runs person classes now
                    if conf < person_conf_thresh:
                        continue

                    track_id = -1
                    if boxes.id is not None:
                        track_id = int(boxes.id[i].cpu())

                    # Filter overlay regions
                    center_y = (y1 + y2) / 2
                    if center_y < y_min or center_y > y_max:
                        continue

                    all_detections.append(Detection(
                        frame_num=frame_num,
                        track_id=track_id,
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=cls_id,
                    ).to_dict())

            # --- Process ball predict results (separate low-conf pass) ---
            yolo_ball_this_frame = []
            if ball_results and ball_results[0].boxes is not None:
                for i in range(len(ball_results[0].boxes)):
                    bbox = ball_results[0].boxes.xyxy[i].cpu().numpy().tolist()
                    conf = float(ball_results[0].boxes.conf[i].cpu())
                    det = Detection(
                        frame_num=frame_num,
                        track_id=-1,
                        bbox=tuple(bbox),
                        confidence=conf,
                        class_id=config.BALL_CLASS_ID,
                    ).to_dict()
                    ball_detections.append(det)
                    yolo_ball_this_frame.append(bbox)

            # --- CV ball detection (Hough + gate) ---
            # Seed gate from YOLO when available; otherwise select best Hough circle
            if yolo_ball_this_frame:
                # Use YOLO detection to update gate position
                bx = yolo_ball_this_frame[0]
                cv_gate.seed((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2)
            else:
                hough_candidates = detect_hough_candidates(frame, y_min, y_max)
                selected = cv_gate.select(hough_candidates)
                if selected is not None:
                    cx, cy, r = selected
                    ball_detections.append({
                        "frame_num": frame_num,
                        "track_id": -1,
                        "bbox": [cx - r, cy - r, cx + r, cy + r],
                        "confidence": 0.40,
                        "class_id": config.BALL_CLASS_ID,
                        "source": "cv_hough",
                    })

            # Ensemble: run second model for ball detection only, merge with IoU dedup
            if ensemble_model is not None:
                ens_results = ensemble_model.predict(
                    frame, conf=ball_conf, classes=[ensemble_ball_cls],
                    device=device, verbose=False,
                )
                if ens_results and ens_results[0].boxes is not None:
                    ens_boxes = ens_results[0].boxes
                    frame_ball_bboxes = [d["bbox"] for d in ball_detections
                                         if d["frame_num"] == frame_num]
                    for i in range(len(ens_boxes)):
                        ebox = ens_boxes.xyxy[i].cpu().numpy().tolist()
                        econf = float(ens_boxes.conf[i].cpu())
                        if econf < ball_conf:
                            continue
                        # IoU dedup: skip if this detection overlaps an existing one
                        duplicate = False
                        ex1, ey1, ex2, ey2 = ebox
                        for existing in frame_ball_bboxes:
                            fx1, fy1, fx2, fy2 = existing
                            ix1, iy1 = max(ex1, fx1), max(ey1, fy1)
                            ix2, iy2 = min(ex2, fx2), min(ey2, fy2)
                            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                            area_e = (ex2 - ex1) * (ey2 - ey1)
                            area_f = (fx2 - fx1) * (fy2 - fy1)
                            union = area_e + area_f - inter
                            if union > 0 and inter / union >= ensemble_iou:
                                duplicate = True
                                break
                        if not duplicate:
                            ball_detections.append({
                                "frame_num": frame_num,
                                "track_id": -1,
                                "bbox": ebox,
                                "confidence": econf,
                                "class_id": ensemble_ball_cls,
                                "source": "ensemble",
                            })

            pbar.update(1)
        pbar.close()

    # Post-process: drop static false positives (field markings, cursor artifact)
    ball_before_filter = len(ball_detections)
    ball_detections = filter_static_ball_detections(ball_detections)
    n_removed = ball_before_filter - len(ball_detections)
    if n_removed:
        print(f"  Static filter: removed {n_removed} static ball false positives")

    # Post-process: Kalman ball interpolation to fill short gaps
    try:
        from pipeline.ball_tracker import smooth_ball_detections
        ball_before = len(ball_detections)
        ball_detections = smooth_ball_detections(ball_detections, fps=reader.fps,
                                                  frame_skip=frame_skip)
        ball_after = len(ball_detections)
        if ball_after > ball_before:
            print(f"  Ball interpolation: {ball_before} → {ball_after} detections "
                  f"(+{ball_after - ball_before} interpolated)")
    except ImportError:
        pass  # ball_tracker not yet available

    # Post-process: fill tiny intra-track gaps (1-3 frames) with linear interpolation
    all_detections = _interpolate_track_gaps(all_detections, max_gap=3)

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
