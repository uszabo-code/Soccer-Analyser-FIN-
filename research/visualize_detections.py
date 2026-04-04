"""
Quick visualiser: overlay ball detections and ground truth on video frames.

Usage:
    python research/visualize_detections.py output/eval_v4/detections.json \
        research/clips/eval_clip.mp4 research/ground_truth.json

Controls:
    Right / Left arrow  — next / previous annotated frame
    B                   — toggle ball detections
    G                   — toggle ground truth
    Q / Esc             — quit
"""
import argparse
import json
import sys
import cv2
from collections import defaultdict


def load_json(path):
    with open(path) as f:
        return json.load(f)


def draw(frame, ball_by_frame, gt_frames, frame_num, show_dets, show_gt):
    display = frame.copy()

    # Ground truth
    if show_gt and str(frame_num) in gt_frames:
        fgt = gt_frames[str(frame_num)]
        ball_gt = fgt.get("ball", {})
        if ball_gt.get("bbox"):
            b = ball_gt["bbox"]
            cv2.rectangle(display, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                          (0, 255, 0), 2)
            label = "GT ball" + ("" if ball_gt.get("in_play", True) else " (dead)")
            cv2.putText(display, label, (int(b[0]), int(b[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        tp = fgt.get("target_player", {})
        if tp.get("bbox"):
            p = tp["bbox"]
            cv2.rectangle(display, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])),
                          (100, 255, 100), 2)
            cv2.putText(display, "GT player", (int(p[0]), int(p[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

    # Pipeline ball detections (check ±4 raw frames)
    if show_dets:
        drawn = set()
        for offset in range(-4, 5):
            for det in ball_by_frame.get(frame_num + offset, []):
                key = tuple(int(x) for x in det["bbox"])
                if key in drawn:
                    continue
                drawn.add(key)
                b = det["bbox"]
                color = (0, 165, 255) if not det.get("interpolated") else (128, 128, 255)
                cv2.rectangle(display, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                              color, 2)
                label = f"ball {det['confidence']:.2f}"
                if det.get("interpolated"):
                    label += " (interp)"
                cv2.putText(display, label, (int(b[0]), int(b[3]) + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # HUD
    hud = f"Frame {frame_num}  |  N=next  P=prev  B=ball({'ON' if show_dets else 'OFF'})  G=GT({'ON' if show_gt else 'OFF'})  Q=quit"
    cv2.putText(display, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    return display


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("detections_json")
    parser.add_argument("video")
    parser.add_argument("ground_truth_json", nargs="?", default=None)
    args = parser.parse_args()

    dets = load_json(args.detections_json)
    gt = load_json(args.ground_truth_json) if args.ground_truth_json else {"frames": {}}
    gt_frames = gt.get("frames", {})

    ball_by_frame = defaultdict(list)
    for det in dets.get("ball_detections", []):
        ball_by_frame[det["frame_num"]].append(det)

    # Use annotated frames if GT available, else sample every 5th
    if gt_frames:
        frame_list = sorted(int(k) for k in gt_frames)
    else:
        total = dets.get("total_frames", 1000)
        frame_list = list(range(0, total, 5))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Cannot open video: {args.video}")
        sys.exit(1)

    WINDOW = "Detection Visualiser  |  ←/→ frames  B=dets  G=GT  Q=quit"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)

    idx = 0
    show_dets = True
    show_gt = True

    while True:
        frame_num = frame_list[idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            idx = (idx + 1) % len(frame_list)
            continue

        n_dets = sum(len(ball_by_frame.get(frame_num + o, [])) for o in range(-4, 5))
        gt_has_ball = str(frame_num) in gt_frames and bool(
            gt_frames[str(frame_num)].get("ball", {}).get("bbox"))

        display = draw(frame, ball_by_frame, gt_frames, frame_num, show_dets, show_gt)
        print(f"Frame {frame_num:5d}  nearby_ball_dets={n_dets:3d}  gt_ball={'YES' if gt_has_ball else 'no '}")

        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('n'), ord('N'), ord('d'), ord('D')):  # N or D = next
            idx = min(idx + 1, len(frame_list) - 1)
        elif key in (ord('p'), ord('P'), ord('a'), ord('A')):  # P or A = previous
            idx = max(idx - 1, 0)
        elif key in (ord('b'), ord('B')):
            show_dets = not show_dets
        elif key in (ord('g'), ord('G')):
            show_gt = not show_gt

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
