#!/usr/bin/env python3
"""
Soccer Game Video Analyzer

Analyzes soccer game footage to track and evaluate player performance.

Usage:
    python analyze.py /path/to/game.mp4 --jersey 10

For full options:
    python analyze.py --help
"""

import argparse
import os
import sys
import time

# Load .env file if present (keeps API keys out of shell profiles)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def main():
    parser = argparse.ArgumentParser(
        description="Analyze soccer game video for player performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze.py game.mp4 --jersey 10
  python analyze.py game.mp4 --jersey 10 --device cpu --skip 5
  python analyze.py game.mp4 --jersey 10 --stage 4   # Resume from LLM analysis
  python analyze.py game.mp4 --jersey 10 --no-llm    # Skip Claude API
        """,
    )

    parser.add_argument("video", help="Path to the game video file")
    parser.add_argument("--jersey", type=int, default=None,
                        help="Target player's jersey number (1-99). If omitted, an interactive picker opens.")
    parser.add_argument("--model", default=config.YOLO_MODEL,
                        help=f"YOLO model name (default: {config.YOLO_MODEL})")
    parser.add_argument("--ball-model", default=None,
                        help="Path to ball-specific model (e.g. futsal_ball_v1.pt). "
                             "Only used for ball prediction; person tracking still "
                             "uses --model. Overrides config.BALL_MODEL.")
    parser.add_argument("--ball-model-class-id", type=int, default=None,
                        help="Ball class ID within --ball-model (default: 0 for nc=1 fine-tunes)")
    parser.add_argument("--device", default=config.DEVICE,
                        help=f"Compute device: auto, mps, cpu (default: {config.DEVICE})")
    parser.add_argument("--skip", type=int, default=config.FRAME_SKIP,
                        help=f"Process every Nth frame (default: {config.FRAME_SKIP})")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help="Start from this pipeline stage (default: 1)")
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR,
                        help=f"Output directory (default: {config.OUTPUT_DIR})")
    parser.add_argument("--field-width", type=float, default=config.FIELD_WIDTH_METERS,
                        help=f"Field width in meters (default: {config.FIELD_WIDTH_METERS})")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip Claude API analysis (Stage 4)")
    parser.add_argument("--no-merge", action="store_true",
                        help="Skip Stage 1b track merging (use when merger produces bad results)")
    parser.add_argument("--no-pdf", action="store_true",
                        help="Skip PDF report generation (Stage 5b)")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    if args.jersey is not None and not 1 <= args.jersey <= 99:
        print(f"Error: Jersey number must be 1-99, got {args.jersey}")
        sys.exit(1)

    # Apply config overrides
    config.YOLO_MODEL = args.model
    config.DEVICE = args.device
    config.FRAME_SKIP = args.skip
    config.FIELD_WIDTH_METERS = args.field_width
    if args.ball_model is not None:
        if not os.path.isfile(args.ball_model):
            print(f"Error: --ball-model file not found: {args.ball_model}")
            sys.exit(1)
        config.BALL_MODEL = args.ball_model
    if args.ball_model_class_id is not None:
        config.BALL_MODEL_CLASS_ID = args.ball_model_class_id

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    video_path = os.path.abspath(args.video)

    # File paths for intermediate outputs
    detections_path = os.path.join(output_dir, "detections.json")
    identity_path = os.path.join(output_dir, "player_identity.json")
    stats_path = os.path.join(output_dir, "player_stats.json")
    analysis_path = os.path.join(output_dir, "analysis.json")

    print("=" * 60)
    print("  SOCCER GAME ANALYZER")
    print("=" * 60)
    print(f"  Video:    {args.video}")
    print(f"  Jersey:   {'#' + str(args.jersey) if args.jersey else 'interactive picker'}")
    print(f"  Device:   {args.device}")
    print(f"  Output:   {output_dir}/")
    print(f"  Stage:    {args.stage}")
    print("=" * 60)

    start_time = time.time()

    # --- Stage 1: Detection & Tracking ---
    if args.stage <= 1:
        from pipeline import detect
        detections_path = detect.run(
            video_path=video_path,
            output_dir=output_dir,
            model_name=args.model,
            device=args.device,
            frame_skip=args.skip,
        )
    else:
        if not os.path.isfile(detections_path):
            print(f"Error: {detections_path} not found. Run from stage 1 first.")
            sys.exit(1)
        print(f"\n[Stage 1] Skipped (using {detections_path})")

    # --- Stage 1b: Post-Process Track Merging ---
    # Runs BEFORE Stage 2: extracts jersey colours directly from video pixels so
    # Stage 2 (OCR + interactive picker) sees ~50 merged tracks, not ~1,500.
    # Condition: stage <= 2 so it runs when resuming from --stage 2 as well.
    if args.stage <= 2 and not args.no_merge:
        from pipeline import reidentify
        reidentify.run(
            detections_path=detections_path,
            player_identity_path=os.path.join(output_dir, "player_identity.json"),
            output_dir=output_dir,
        )
    elif args.no_merge:
        print("\n[Stage 1b] Skipped (--no-merge)")

    # --- Stage 2: Player Identification ---
    if args.stage <= 2:
        from pipeline import identify
        identity_path = identify.run(
            video_path=video_path,
            detections_path=detections_path,
            output_dir=output_dir,
            target_jersey=args.jersey,
        )
    else:
        if not os.path.isfile(identity_path):
            print(f"Error: {identity_path} not found. Run from stage 2 first.")
            sys.exit(1)
        print(f"\n[Stage 2] Skipped (using {identity_path})")

    # --- Stage 3: Feature Extraction ---
    if args.stage <= 3:
        from pipeline import features
        stats_path = features.run(
            detections_path=detections_path,
            identity_path=identity_path,
            output_dir=output_dir,
            video_path=video_path,
        )
        # Stage 3b: Advanced features (spacing, off-ball, positional intelligence)
        from pipeline import advanced_features
        advanced_features.run(
            detections_path=detections_path,
            identity_path=identity_path,
            stats_path=stats_path,
            output_dir=output_dir,
        )
    else:
        if not os.path.isfile(stats_path):
            print(f"Error: {stats_path} not found. Run from stage 3 first.")
            sys.exit(1)
        print(f"\n[Stage 3] Skipped (using {stats_path})")

    # --- Stage 4: LLM Analysis ---
    if args.stage <= 4 and not args.no_llm:
        from pipeline import analyze as llm_analyze
        analysis_path = llm_analyze.run(
            stats_path=stats_path,
            identity_path=identity_path,
            output_dir=output_dir,
        )
    else:
        if not os.path.isfile(analysis_path):
            # Create empty analysis so report stage can still run
            from pipeline import analyze as llm_analyze
            analysis_path = llm_analyze._save_empty_analysis(output_dir)
        if args.no_llm:
            print(f"\n[Stage 4] Skipped (--no-llm)")
        else:
            print(f"\n[Stage 4] Skipped (using {analysis_path})")

    # --- Stage 5: Report ---
    from pipeline import report
    report_path = report.run(
        analysis_path=analysis_path,
        stats_path=stats_path,
        identity_path=identity_path,
        output_dir=output_dir,
    )

    # --- Stage 5b: PDF Report ---
    if not args.no_pdf:
        try:
            from pipeline import pdf_report
            pdf_report.run(
                stats_path=stats_path,
                identity_path=identity_path,
                analysis_path=analysis_path,
                detections_path=detections_path,
                output_dir=output_dir,
            )
        except ImportError:
            print("\n[Stage 5b] Skipped (mplsoccer not installed — pip install mplsoccer)")
    else:
        print("\n[Stage 5b] Skipped (--no-pdf)")

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print()
    print("=" * 60)
    print(f"  DONE in {minutes}m {seconds}s")
    print(f"  Report: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
