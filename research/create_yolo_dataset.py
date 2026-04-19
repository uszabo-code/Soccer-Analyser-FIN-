"""
Create a YOLO training dataset from ground truth annotations.

Extracts frames from eval_clip.mp4 and converts ground_truth.json
annotations to YOLO format for fine-tuning ball detection.

YOLO annotation format: class_id center_x center_y width height
(all normalized 0-1 relative to image dimensions)
"""
from __future__ import annotations

import json
import os
import random
import sys

import cv2

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, PROJECT_ROOT)

CLIP_PATH = os.path.join(SCRIPT_DIR, "clips", "eval_clip.mp4")
GT_PATH = os.path.join(SCRIPT_DIR, "ground_truth.json")
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "futsal_ball")

# Ball class ID in football_yolov8.pt
BALL_CLASS_ID = 0

# Train/val split ratio
VAL_RATIO = 0.2
RANDOM_SEED = 42


def main():
    # Load ground truth
    with open(GT_PATH) as f:
        gt = json.load(f)

    # Open video
    cap = cv2.VideoCapture(CLIP_PATH)
    if not cap.isOpened():
        print(f"ERROR: cannot open {CLIP_PATH}")
        sys.exit(1)

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {img_w}x{img_h}, {total_frames} frames")

    # Collect frames with ball annotations
    ball_frames = {}
    for frame_key, frame_data in gt["frames"].items():
        ball = frame_data.get("ball")
        if ball and ball.get("in_play", False):
            bbox = ball["bbox"]  # [x1, y1, x2, y2]
            ball_frames[int(frame_key)] = bbox
    print(f"Ground truth: {len(ball_frames)} frames with in-play ball annotations")

    # Create output directories
    for split in ["train", "val"]:
        os.makedirs(os.path.join(DATASET_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, "labels", split), exist_ok=True)

    # Split into train/val
    frame_nums = sorted(ball_frames.keys())
    random.seed(RANDOM_SEED)
    random.shuffle(frame_nums)
    n_val = max(1, int(len(frame_nums) * VAL_RATIO))
    val_frames = set(frame_nums[:n_val])
    train_frames = set(frame_nums[n_val:])
    print(f"Split: {len(train_frames)} train, {len(val_frames)} val")

    # Extract frames and write annotations
    frame_num = 0
    extracted = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num in ball_frames:
            bbox = ball_frames[frame_num]
            x1, y1, x2, y2 = bbox

            # Convert to YOLO format: normalized center + size
            cx = ((x1 + x2) / 2.0) / img_w
            cy = ((y1 + y2) / 2.0) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h

            # Clamp to [0, 1]
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            w = max(0.001, min(1, w))
            h = max(0.001, min(1, h))

            split = "val" if frame_num in val_frames else "train"
            fname = f"frame_{frame_num:06d}"

            # Save image
            img_path = os.path.join(DATASET_DIR, "images", split, f"{fname}.jpg")
            cv2.imwrite(img_path, frame)

            # Save YOLO annotation
            label_path = os.path.join(DATASET_DIR, "labels", split, f"{fname}.txt")
            with open(label_path, "w") as f:
                f.write(f"{BALL_CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            extracted += 1

        frame_num += 1

    cap.release()
    print(f"Extracted {extracted} frames")

    # Create dataset YAML
    # Use absolute paths for reliability
    dataset_yaml = {
        "path": os.path.abspath(DATASET_DIR),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "ball",
        },
        "nc": 1,
    }

    yaml_path = os.path.join(DATASET_DIR, "dataset.yaml")
    with open(yaml_path, "w") as f:
        # Write YAML manually to avoid pyyaml dependency
        f.write(f"path: {os.path.abspath(DATASET_DIR)}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"nc: 1\n")
        f.write(f"names:\n  0: ball\n")
    print(f"Dataset YAML: {yaml_path}")

    # Summary
    train_count = len(os.listdir(os.path.join(DATASET_DIR, "images", "train")))
    val_count = len(os.listdir(os.path.join(DATASET_DIR, "images", "val")))
    print(f"\nDataset created at {DATASET_DIR}")
    print(f"  Train images: {train_count}")
    print(f"  Val images: {val_count}")
    print(f"\nTo fine-tune:")
    print(f"  yolo train model=football_yolov8.pt data={yaml_path} epochs=50 imgsz=1280 batch=4")


if __name__ == "__main__":
    main()
