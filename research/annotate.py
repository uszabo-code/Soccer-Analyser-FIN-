"""
Simple annotation tool for ground truth collection.

Shows every 5th frame from the eval clip. For each frame:
  - Drag a box around the BALL (press B first)
  - Drag a box around YOUR SON / TARGET PLAYER (press P first)
  - Press I to mark the ball as OUT OF PLAY (dead ball, no box needed)
  - Press SPACE to skip to the next frame (nothing visible)
  - Press Z to undo the last box drawn
  - Press S to save and quit

Saves results to research/ground_truth.json automatically.

Usage:
    python research/annotate.py research/clips/eval_clip.mp4
    python research/annotate.py research/clips/eval_clip.mp4 --every 5
"""
import argparse
import json
import os
import sys

import cv2

WINDOW = "Annotator  |  B=ball  P=player  I=out-of-play  SPACE=skip  Z=undo  S=save&quit"


def draw_instructions(frame, mode, frame_num, total, annotations):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Dark bar at top
    cv2.rectangle(overlay, (0, 0), (w, 56), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    mode_color = (0, 200, 255) if mode == "ball" else (100, 255, 100)
    mode_label = f"Mode: {'BALL' if mode == 'ball' else 'PLAYER'}"
    cv2.putText(frame, mode_label, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, mode_color, 2)

    progress = f"Frame {frame_num}  ({list(annotations.keys()).index(str(frame_num)) + 1 if str(frame_num) in annotations else '?'} done)"
    cv2.putText(frame, progress, (10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Show existing boxes for this frame
    ann = annotations.get(str(frame_num), {})
    ball = ann.get("ball", {})
    player = ann.get("target_player", {})
    if ball.get("bbox"):
        b = ball["bbox"]
        cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 255), 2)
        cv2.putText(frame, "BALL", (int(b[0]), int(b[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    if not ball.get("in_play", True) and not ball.get("bbox"):
        cv2.putText(frame, "OUT OF PLAY", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
    if player.get("bbox"):
        p = player["bbox"]
        cv2.rectangle(frame, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (100, 255, 100), 2)
        cv2.putText(frame, "YOUR SON", (int(p[0]), int(p[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
    return frame


class BoxDrawer:
    def __init__(self):
        self.drawing = False
        self.x0 = self.y0 = self.x1 = self.y1 = 0
        self.done_bbox = None

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.x0, self.y0 = x, y
            self.done_bbox = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.x1, self.y1 = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.x1, self.y1 = x, y
            x1, x2 = sorted([self.x0, self.x1])
            y1, y2 = sorted([self.y0, self.y1])
            if (x2 - x1) > 5 and (y2 - y1) > 5:
                self.done_bbox = [x1, y1, x2, y2]

    def live_rect(self):
        if self.drawing:
            return (self.x0, self.y0, self.x1, self.y1)
        return None


def run(video_path: str, output_path: str, every: int = 5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_nums = list(range(0, total_frames, every))

    # Load existing annotations if any
    annotations = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            data = json.load(f)
        annotations = data.get("frames", {})
        print(f"Loaded {len(annotations)} existing annotations from {output_path}")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)
    drawer = BoxDrawer()
    cv2.setMouseCallback(WINDOW, drawer.mouse)

    mode = "ball"  # current labelling mode
    idx = 0
    history = []  # for undo

    while idx < len(frame_nums):
        frame_num = frame_nums[idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, raw_frame = cap.read()
        if not ret:
            idx += 1
            continue

        drawer.done_bbox = None
        drawer.drawing = False

        while True:
            display = raw_frame.copy()
            display = draw_instructions(display, mode, frame_num, len(frame_nums), annotations)

            # Draw live rubber-band rect
            rect = drawer.live_rect()
            if rect:
                cv2.rectangle(display,
                               (rect[0], rect[1]), (rect[2], rect[3]),
                               (0, 200, 255) if mode == "ball" else (100, 255, 100), 1)

            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(16) & 0xFF

            # Handle completed box draw
            if drawer.done_bbox:
                bbox = drawer.done_bbox
                drawer.done_bbox = None
                ann = annotations.setdefault(str(frame_num), {})
                if mode == "ball":
                    ann["ball"] = {"bbox": bbox, "in_play": True}
                else:
                    ann["target_player"] = {"bbox": bbox}
                history.append((frame_num, mode))
                print(f"  Frame {frame_num}: {mode} bbox = {bbox}")

            if key == ord('b') or key == ord('B'):
                mode = "ball"
                print("Mode → BALL")
            elif key == ord('p') or key == ord('P'):
                mode = "player"
                print("Mode → PLAYER (your son)")
            elif key == ord('i') or key == ord('I'):
                # Mark ball as out of play (no bbox)
                ann = annotations.setdefault(str(frame_num), {})
                ann["ball"] = {"in_play": False}
                history.append((frame_num, "ball_oop"))
                print(f"  Frame {frame_num}: ball OUT OF PLAY")
                break  # move to next frame
            elif key == ord(' '):
                # Skip frame — nothing to annotate
                print(f"  Frame {frame_num}: skipped")
                idx += 1
                break
            elif key == ord('z') or key == ord('Z'):
                # Undo last annotation
                if history:
                    fn, m = history.pop()
                    ann = annotations.get(str(fn), {})
                    if m == "ball" or m == "ball_oop":
                        ann.pop("ball", None)
                    elif m == "player":
                        ann.pop("target_player", None)
                    if not ann:
                        annotations.pop(str(fn), None)
                    print(f"  Undid last annotation on frame {fn}")
            elif key == ord('s') or key == ord('S') or key == 27:  # S or ESC
                _save(annotations, output_path, fps, video_path)
                print(f"\nSaved {len(annotations)} frames to {output_path}")
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == 13 or key == ord('n') or key == ord('N'):
                # Enter or N = next frame
                idx += 1
                break

        # Auto-advance if both ball and player annotated
        ann = annotations.get(str(frame_num), {})
        has_ball = "ball" in ann
        has_player = "target_player" in ann
        if has_ball and has_player and key not in (ord('z'), ord('Z')):
            idx += 1

    _save(annotations, output_path, fps, video_path)
    print(f"\nAll frames done! Saved {len(annotations)} annotations to {output_path}")
    cap.release()
    cv2.destroyAllWindows()


def _save(annotations, output_path, fps, video_path):
    data = {
        "clip_info": {
            "source_video": video_path,
            "fps": fps,
            "annotated_every_n_frames": 5,
            "annotated_by": "",
            "notes": ""
        },
        "frames": annotations,
    }
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Click-to-annotate ground truth tool")
    parser.add_argument("video", help="Path to eval clip (e.g. research/clips/eval_clip.mp4)")
    parser.add_argument("--output", default="research/ground_truth.json",
                        help="Output JSON path (default: research/ground_truth.json)")
    parser.add_argument("--every", type=int, default=5,
                        help="Annotate every Nth frame (default: 5)")
    args = parser.parse_args()
    run(args.video, args.output, args.every)
