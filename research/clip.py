"""Extract a short evaluation clip from the full video for fast experimentation."""
import argparse
import subprocess
import sys


def extract_clip(input_path: str, output_path: str, start_sec: float, duration_sec: float):
    """Extract a clip using ffmpeg (fast, no re-encoding)."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", input_path,
        "-t", str(duration_sec),
        "-c", "copy",
        output_path,
    ]
    print(f"Extracting {duration_sec}s clip starting at {start_sec}s")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract evaluation clip")
    parser.add_argument("input", help="Path to full video")
    parser.add_argument("-o", "--output", default="research/eval_clip.mp4", help="Output clip path")
    parser.add_argument("-s", "--start", type=float, default=84.0,
                        help="Start time in seconds (default: 84, where target player appears)")
    parser.add_argument("-d", "--duration", type=float, default=30.0,
                        help="Clip duration in seconds (default: 30)")
    args = parser.parse_args()
    extract_clip(args.input, args.output, args.start, args.duration)
