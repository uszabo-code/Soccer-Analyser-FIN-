"""
Autoresearch-style experiment runner for soccer analyzer optimization.

Inspired by karpathy/autoresearch: an AI agent reads a research program (markdown),
proposes config changes, runs a short pipeline experiment, evaluates the result,
and iterates — keeping the best configuration.

Usage:
    python research/runner.py --program ball_detection --experiments 10
    python research/runner.py --program speed_calibration --experiments 20
    python research/runner.py --program tracking_smoothness --experiments 15
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

import config as base_config
from research.evaluate import (
    load_detections, ball_detection_score, speed_realism_score,
    tracking_smoothness_score, combined_score,
)

# Tunable parameters and their valid ranges
TUNABLE_PARAMS = {
    "CONFIDENCE_THRESHOLD": {"min": 0.05, "max": 0.8, "type": "float", "description": "YOLO detection confidence threshold"},
    "BALL_CONFIDENCE_THRESHOLD": {"min": 0.05, "max": 0.5, "type": "float", "description": "Separate confidence for ball class (if supported)"},
    "OVERLAY_TOP_FRACTION": {"min": 0.0, "max": 0.15, "type": "float", "description": "Top overlay exclusion zone fraction"},
    "OVERLAY_BOTTOM_FRACTION": {"min": 0.0, "max": 0.10, "type": "float", "description": "Bottom overlay exclusion zone fraction"},
    "FRAME_SKIP": {"min": 1, "max": 6, "type": "int", "description": "Process every Nth frame"},
    "FIELD_WIDTH_METERS": {"min": 60.0, "max": 110.0, "type": "float", "description": "Assumed field width for pixel-to-meter conversion"},
    "SPRINT_SPEED_THRESHOLD": {"min": 2.0, "max": 6.0, "type": "float", "description": "Speed threshold for sprint detection (m/s)"},
    "SPEED_OUTLIER_CAP": {"min": 8.0, "max": 15.0, "type": "float", "description": "Maximum realistic speed before capping (m/s)"},
    "SMOOTHING_WINDOW": {"min": 1, "max": 11, "type": "int", "description": "Position smoothing window size (odd number)"},
    "MIN_MOVEMENT_PX": {"min": 3, "max": 20, "type": "int", "description": "Minimum pixel movement to count as real motion"},
    "DIRECTION_CHANGE_ANGLE": {"min": 60, "max": 150, "type": "int", "description": "Minimum angle for direction change detection"},
}

# Map program names to their primary metric function
METRIC_MAP = {
    "ball_detection": ball_detection_score,
    "speed_calibration": speed_realism_score,
    "tracking_smoothness": tracking_smoothness_score,
}


def load_program(program_name: str) -> str:
    """Load a research program markdown file."""
    program_path = Path(__file__).parent / "programs" / f"{program_name}.md"
    if not program_path.exists():
        print(f"Program not found: {program_path}", file=sys.stderr)
        sys.exit(1)
    return program_path.read_text()


def get_baseline_config() -> dict:
    """Get current config values as a dict."""
    return {
        "CONFIDENCE_THRESHOLD": base_config.CONFIDENCE_THRESHOLD,
        "OVERLAY_TOP_FRACTION": base_config.OVERLAY_TOP_FRACTION,
        "OVERLAY_BOTTOM_FRACTION": base_config.OVERLAY_BOTTOM_FRACTION,
        "FRAME_SKIP": base_config.FRAME_SKIP,
        "FIELD_WIDTH_METERS": base_config.FIELD_WIDTH_METERS,
        "SPRINT_SPEED_THRESHOLD": base_config.SPRINT_SPEED_THRESHOLD,
        "SPEED_OUTLIER_CAP": 12.0,
        "SMOOTHING_WINDOW": 1,
        "MIN_MOVEMENT_PX": 5,
        "DIRECTION_CHANGE_ANGLE": 90,
        "BALL_CONFIDENCE_THRESHOLD": base_config.CONFIDENCE_THRESHOLD,
    }


def write_experiment_config(config_overrides: dict, output_path: str):
    """Write a JSON config file for a single experiment."""
    with open(output_path, "w") as f:
        json.dump(config_overrides, f, indent=2)


def call_claude_for_proposal(
    program_text: str,
    current_config: dict,
    experiment_history: list,
    experiment_num: int,
    api_key: str,
) -> dict:
    """
    Ask Claude to propose the next config change.

    Returns a dict of param -> new_value to try.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    # Build history summary
    history_text = "No experiments run yet." if not experiment_history else ""
    for exp in experiment_history[-10:]:  # Last 10 experiments
        delta = exp.get("score_delta", 0)
        kept = "KEPT" if exp.get("kept") else "REVERTED"
        history_text += f"\n  Exp {exp['num']}: {exp['changes']} -> score {exp['score']:.1f} (delta {delta:+.1f}) [{kept}]"

    # Build param info
    param_info = ""
    for name, info in TUNABLE_PARAMS.items():
        current_val = current_config.get(name, "N/A")
        param_info += f"\n  {name}: current={current_val}, range=[{info['min']}, {info['max']}], type={info['type']} — {info['description']}"

    prompt = f"""You are an ML research agent optimizing a soccer video analysis pipeline.

## Research Program
{program_text}

## Current Configuration
{json.dumps(current_config, indent=2)}

## Available Parameters
{param_info}

## Experiment History
{history_text}

## Task
This is experiment #{experiment_num}. Based on the program instructions and experiment history,
propose ONE or TWO parameter changes to try next. Focus on changes most likely to improve the metric.

Respond with ONLY a JSON object mapping parameter names to new values. Example:
{{"CONFIDENCE_THRESHOLD": 0.15, "FRAME_SKIP": 2}}

No explanations — just the JSON object."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    # Extract JSON from response (handle markdown fences)
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    proposed = json.loads(text)

    # Validate and clamp values
    validated = {}
    for key, val in proposed.items():
        if key in TUNABLE_PARAMS:
            info = TUNABLE_PARAMS[key]
            if info["type"] == "int":
                val = int(val)
            else:
                val = float(val)
            val = max(info["min"], min(info["max"], val))
            validated[key] = val

    return validated


def run_detection_experiment(
    video_path: str, config_overrides: dict, output_dir: str, timeout: int = 120
) -> bool:
    """Run Stage 1 detection with overridden config. Returns True on success."""
    # Write a temporary config override file
    override_path = os.path.join(output_dir, "config_override.json")
    write_experiment_config(config_overrides, override_path)

    # Build the detection command — we run a minimal detection script
    script = f"""
import sys, os, json
sys.path.insert(0, '{PROJECT_ROOT}')
os.chdir('{PROJECT_ROOT}')

# Apply config overrides
import config
overrides = json.load(open('{override_path}'))
for k, v in overrides.items():
    if hasattr(config, k):
        setattr(config, k, v)

from pipeline import detect
detect.run('{video_path}', '{output_dir}')
"""
    script_path = os.path.join(output_dir, "_run_experiment.py")
    with open(script_path, "w") as f:
        f.write(script)

    env = os.environ.copy()
    env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=timeout,
            cwd=PROJECT_ROOT, env=env,
        )
        if result.returncode != 0:
            print(f"  Experiment failed: {result.stderr[-500:]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  Experiment timed out after {timeout}s")
        return False


def run_experiment_loop(
    program_name: str,
    video_path: str,
    num_experiments: int,
    api_key: str,
    timeout_per_experiment: int = 120,
):
    """Main autoresearch-style experiment loop."""
    program_text = load_program(program_name)
    metric_fn = METRIC_MAP.get(program_name)
    if metric_fn is None:
        print(f"Unknown program: {program_name}. Available: {list(METRIC_MAP.keys())}")
        sys.exit(1)

    # Setup
    experiment_dir = Path(PROJECT_ROOT) / "research" / "experiments" / program_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    log_path = experiment_dir / "experiment_log.json"

    current_config = get_baseline_config()
    experiment_history = []
    best_score = None
    best_config = copy.deepcopy(current_config)

    # Run baseline first
    print(f"\n{'='*60}")
    print(f"  AUTORESEARCH: {program_name}")
    print(f"  Experiments: {num_experiments}")
    print(f"  Video: {video_path}")
    print(f"{'='*60}")

    # Check if we need to run detection (expensive) or can reuse cached
    needs_detection = program_name == "ball_detection"

    if needs_detection:
        print("\n[Baseline] Running detection...")
        baseline_dir = str(experiment_dir / "baseline")
        os.makedirs(baseline_dir, exist_ok=True)
        success = run_detection_experiment(video_path, current_config, baseline_dir, timeout_per_experiment)
        if not success:
            print("Baseline detection failed!")
            sys.exit(1)
        baseline_dets = load_detections(baseline_dir)
    else:
        # For speed/tracking metrics, use the existing full detections
        existing_det_path = Path(PROJECT_ROOT) / "output" / "detections.json"
        if not existing_det_path.exists():
            print(f"No existing detections found at {existing_det_path}. Run Stage 1 first.")
            sys.exit(1)
        with open(existing_det_path) as f:
            baseline_dets = json.load(f)

    if program_name == "tracking_smoothness":
        baseline_score, baseline_detail = metric_fn(
            baseline_dets,
            smoothing_window=int(current_config.get("SMOOTHING_WINDOW", 1)),
            min_movement_px=int(current_config.get("MIN_MOVEMENT_PX", 5)),
            direction_change_angle=int(current_config.get("DIRECTION_CHANGE_ANGLE", 90)),
        )
    elif program_name == "speed_calibration":
        baseline_score, baseline_detail = metric_fn(
            baseline_dets, field_width_m=current_config.get("FIELD_WIDTH_METERS", 90.0)
        )
    else:
        baseline_score, baseline_detail = metric_fn(baseline_dets)
    best_score = baseline_score
    print(f"[Baseline] Score: {baseline_score:.1f}")
    print(f"  Details: {json.dumps(baseline_detail, indent=2)}")

    experiment_history.append({
        "num": 0, "type": "baseline", "config": current_config,
        "score": baseline_score, "details": baseline_detail,
        "changes": {}, "kept": True, "score_delta": 0,
    })

    # Experiment loop
    for exp_num in range(1, num_experiments + 1):
        print(f"\n--- Experiment {exp_num}/{num_experiments} ---")

        # Ask Claude for proposed changes
        try:
            proposed_changes = call_claude_for_proposal(
                program_text, current_config, experiment_history, exp_num, api_key
            )
        except Exception as e:
            print(f"  Claude proposal failed: {e}")
            continue

        if not proposed_changes:
            print("  No changes proposed, skipping")
            continue

        print(f"  Proposed: {proposed_changes}")

        # Apply changes to config
        test_config = copy.deepcopy(current_config)
        test_config.update(proposed_changes)

        if needs_detection:
            # Run detection with new config
            exp_output_dir = str(experiment_dir / f"exp_{exp_num:03d}")
            os.makedirs(exp_output_dir, exist_ok=True)

            start = time.time()
            success = run_detection_experiment(video_path, test_config, exp_output_dir, timeout_per_experiment)
            elapsed = time.time() - start
            print(f"  Detection took {elapsed:.0f}s")

            if not success:
                experiment_history.append({
                    "num": exp_num, "type": "failed", "config": test_config,
                    "score": 0, "details": {}, "changes": proposed_changes,
                    "kept": False, "score_delta": -best_score,
                })
                continue

            exp_dets = load_detections(exp_output_dir)
        else:
            # For non-detection metrics, score is computed on existing detections
            # but with different config assumptions (e.g., field_width_m)
            exp_dets = baseline_dets

        # Evaluate
        if program_name == "speed_calibration":
            field_w = test_config.get("FIELD_WIDTH_METERS", 90.0)
            exp_score, exp_detail = metric_fn(exp_dets, field_width_m=field_w)
        elif program_name == "tracking_smoothness":
            exp_score, exp_detail = metric_fn(
                exp_dets,
                smoothing_window=int(test_config.get("SMOOTHING_WINDOW", 1)),
                min_movement_px=int(test_config.get("MIN_MOVEMENT_PX", 5)),
                direction_change_angle=int(test_config.get("DIRECTION_CHANGE_ANGLE", 90)),
            )
        else:
            exp_score, exp_detail = metric_fn(exp_dets)

        score_delta = exp_score - best_score
        kept = score_delta > 0

        print(f"  Score: {exp_score:.1f} (delta: {score_delta:+.1f}) {'KEPT' if kept else 'REVERTED'}")

        if kept:
            best_score = exp_score
            best_config = copy.deepcopy(test_config)
            current_config = copy.deepcopy(test_config)

        experiment_history.append({
            "num": exp_num, "type": "experiment", "config": test_config,
            "score": exp_score, "details": exp_detail, "changes": proposed_changes,
            "kept": kept, "score_delta": score_delta, "timestamp": datetime.now().isoformat(),
        })

        # Save log after each experiment
        with open(log_path, "w") as f:
            json.dump({
                "program": program_name,
                "best_score": best_score,
                "best_config": best_config,
                "baseline_score": baseline_score,
                "experiments": experiment_history,
            }, f, indent=2)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  RESULTS: {program_name}")
    print(f"{'='*60}")
    print(f"  Baseline score: {baseline_score:.1f}")
    print(f"  Best score:     {best_score:.1f} ({best_score - baseline_score:+.1f})")
    print(f"  Best config:    {json.dumps(best_config, indent=2)}")
    print(f"  Log saved to:   {log_path}")

    # Save best config
    best_config_path = experiment_dir / "best_config.json"
    with open(best_config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"  Best config:    {best_config_path}")

    return best_score, best_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Autoresearch-style experiment runner for soccer analyzer"
    )
    parser.add_argument(
        "--program", required=True,
        choices=list(METRIC_MAP.keys()),
        help="Research program to run",
    )
    parser.add_argument(
        "--video", default="research/eval_clip.mp4",
        help="Video file for detection experiments (default: eval clip)",
    )
    parser.add_argument(
        "--experiments", type=int, default=10,
        help="Number of experiments to run",
    )
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="Timeout per experiment in seconds",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY environment variable", file=sys.stderr)
        sys.exit(1)

    run_experiment_loop(
        program_name=args.program,
        video_path=args.video,
        num_experiments=args.experiments,
        api_key=api_key,
        timeout_per_experiment=args.timeout,
    )
