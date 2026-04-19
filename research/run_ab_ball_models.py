"""
A/B compare two ball models on a video segment.

Runs Stage 1 detection only (no identify/features/LLM) twice:
  Run A: uses YOLO_MODEL for ball (baseline)
  Run B: uses BALL_MODEL for ball (override)

Writes detections.json to each output dir. Prints a quick comparison.
Use when validating a new fine-tuned ball model before full-pipeline analysis.

Example:
    python research/run_ab_ball_models.py \\
        --video '/path/to/game.mp4' \\
        --ball-model futsal_ball_v1.pt \\
        --start-sec 600 --end-sec 1200 \\
        --output-root output/outdoor_ab
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from pipeline import detect


def _summarize(det_path: str, label: str) -> dict:
    with open(det_path) as f:
        data = json.load(f)
    # detections.json keys: person_detections, ball_detections
    persons = data.get("person_detections", data.get("detections", []))
    balls = data.get("ball_detections", [])
    person_tids = {d.get("track_id") for d in persons if d.get("track_id", -1) >= 0}
    interp_balls = sum(1 for b in balls if b.get("interpolated"))
    real_balls = [b for b in balls if not b.get("interpolated")]
    ball_confs = [b.get("confidence", 0.0) for b in real_balls]
    mean_conf = sum(ball_confs) / len(ball_confs) if ball_confs else 0.0
    high_conf_balls = sum(1 for b in real_balls if b.get("confidence", 0) >= 0.30)

    # Temporal coverage: unique frame_nums with at least one ball det vs all processed frames
    ball_frames = {b["frame_num"] for b in balls}
    person_frames = {d["frame_num"] for d in persons}
    processed_frames = len(person_frames | ball_frames)
    coverage = len(ball_frames) / processed_frames if processed_frames else 0.0

    summary = {
        "label": label,
        "person_detections": len(persons),
        "unique_person_tracks": len(person_tids),
        "ball_detections_total": len(balls),
        "ball_detections_interpolated": interp_balls,
        "ball_detections_real": len(real_balls),
        "ball_detections_high_conf": high_conf_balls,
        "ball_mean_confidence": round(mean_conf, 4),
        "ball_temporal_coverage": round(coverage, 4),
        "processed_frames": processed_frames,
    }
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="Path to source video")
    p.add_argument("--ball-model", required=True,
                   help="Path to fine-tuned ball model for Run B")
    p.add_argument("--ball-model-class-id", type=int, default=0,
                   help="Ball class id within ball-model (default: 0)")
    p.add_argument("--start-sec", type=float, default=0.0,
                   help="Start time in seconds (default: 0)")
    p.add_argument("--end-sec", type=float, default=None,
                   help="End time in seconds (default: full video)")
    p.add_argument("--output-root", default="output/ab_compare",
                   help="Parent directory; Run A and Run B written to subdirs")
    p.add_argument("--model", default=None,
                   help="Override person/baseline YOLO model (default: config.YOLO_MODEL)")
    p.add_argument("--device", default=None,
                   help="Override device (default: config.DEVICE auto)")
    p.add_argument("--skip", type=int, default=None,
                   help="Frame skip (default: config.FRAME_SKIP)")
    p.add_argument("--only", choices=["A", "B", "both"], default="both",
                   help="Run only one side (useful after a crash)")
    args = p.parse_args()

    if not os.path.isfile(args.video):
        print(f"Error: video not found: {args.video}")
        sys.exit(1)
    if not os.path.isfile(args.ball_model):
        print(f"Error: ball-model not found: {args.ball_model}")
        sys.exit(1)

    # Derive frame range
    import cv2
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    start_frame = int(args.start_sec * fps)
    end_frame = int(args.end_sec * fps) if args.end_sec is not None else total
    print(f"Video fps={fps:.2f}, total={total}, "
          f"range=[{start_frame}, {end_frame}) "
          f"({(end_frame - start_frame) / fps:.1f}s)")

    os.makedirs(args.output_root, exist_ok=True)
    out_a = os.path.join(args.output_root, "run_a_base")
    out_b = os.path.join(args.output_root, "run_b_finetuned")
    os.makedirs(out_a, exist_ok=True)
    os.makedirs(out_b, exist_ok=True)

    summaries = {}

    if args.only in ("A", "both"):
        print("\n" + "=" * 60)
        print("RUN A — baseline (YOLO_MODEL for ball)")
        print("=" * 60)
        config.BALL_MODEL = None  # explicit — use YOLO_MODEL for both
        t0 = time.time()
        det_a = detect.run(
            video_path=os.path.abspath(args.video),
            output_dir=out_a,
            model_name=args.model,
            device=args.device,
            frame_skip=args.skip,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        print(f"Run A elapsed: {time.time() - t0:.1f}s")
        summaries["A"] = _summarize(det_a, "Run A (base)")

    if args.only in ("B", "both"):
        print("\n" + "=" * 60)
        print(f"RUN B — fine-tuned ball model ({args.ball_model})")
        print("=" * 60)
        config.BALL_MODEL = os.path.abspath(args.ball_model)
        config.BALL_MODEL_CLASS_ID = args.ball_model_class_id
        t0 = time.time()
        det_b = detect.run(
            video_path=os.path.abspath(args.video),
            output_dir=out_b,
            model_name=args.model,
            device=args.device,
            frame_skip=args.skip,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        print(f"Run B elapsed: {time.time() - t0:.1f}s")
        summaries["B"] = _summarize(det_b, "Run B (finetuned)")
        # Reset for cleanliness
        config.BALL_MODEL = None

    # Write comparison json
    comp_path = os.path.join(args.output_root, "ab_summary.json")
    with open(comp_path, "w") as f:
        json.dump(summaries, f, indent=2)

    # Print table
    print("\n" + "=" * 60)
    print("A/B SUMMARY")
    print("=" * 60)
    keys = ["person_detections", "unique_person_tracks",
            "ball_detections_real", "ball_detections_interpolated",
            "ball_mean_confidence", "ball_temporal_coverage",
            "processed_frames"]
    a = summaries.get("A", {})
    b = summaries.get("B", {})
    print(f"  {'metric':32s}  {'Run A (base)':>15s}  {'Run B (finetuned)':>18s}")
    for k in keys:
        va = a.get(k, "-")
        vb = b.get(k, "-")
        print(f"  {k:32s}  {str(va):>15s}  {str(vb):>18s}")
    print(f"\nWritten: {comp_path}")


if __name__ == "__main__":
    main()
