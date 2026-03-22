#!/bin/bash
set -e

echo "=== Soccer Analyzer Setup ==="

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Found Python $PYTHON_VERSION"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists."
fi

# Activate
source .venv/bin/activate
echo "Activated virtual environment."

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt

# Check for ffmpeg
if command -v ffmpeg &> /dev/null; then
    echo "ffmpeg found: $(ffmpeg -version 2>&1 | head -1)"
elif command -v brew &> /dev/null; then
    echo "Installing ffmpeg via Homebrew..."
    brew install ffmpeg
else
    echo "WARNING: ffmpeg not found and Homebrew not available."
    echo "  Most videos should still work, but if you hit codec issues, install ffmpeg:"
    echo "  brew install ffmpeg"
fi

# Check MPS availability
echo ""
echo "Checking Apple Silicon MPS support..."
python3 -c "
import torch
if torch.backends.mps.is_available():
    print('MPS (Apple Silicon GPU) is AVAILABLE - will use GPU acceleration')
else:
    print('MPS not available - will use CPU (slower but works)')
"

# Download YOLO model
echo ""
echo "Downloading YOLOv8m model..."
python3 -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Usage:"
echo "  source .venv/bin/activate"
echo "  python analyze.py /path/to/game.mp4 --jersey 10"
