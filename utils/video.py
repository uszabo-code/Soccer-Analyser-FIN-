"""Video reading utilities and timestamp helpers."""

import cv2


class VideoReader:
    """Efficient video reader with frame skipping."""

    def __init__(self, video_path: str, frame_skip: int = 3):
        self.video_path = video_path
        self.frame_skip = frame_skip
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open video: {video_path}\n"
                "If this is a codec issue, try: brew install ffmpeg"
            )

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_s = self.total_frames / self.fps if self.fps > 0 else 0

    @property
    def frames_to_process(self) -> int:
        return self.total_frames // self.frame_skip

    def iter_frames(self):
        """Yield (frame_num, frame) tuples, respecting frame_skip."""
        frame_num = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_num % self.frame_skip == 0:
                yield frame_num, frame
            frame_num += 1

    def read_frame(self, frame_num: int):
        """Read a specific frame by number."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def frame_to_timestamp(frame_num: int, fps: float) -> str:
    """Convert frame number to MM:SS format."""
    if fps <= 0:
        return "00:00"
    seconds = frame_num / fps
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def frame_to_seconds(frame_num: int, fps: float) -> float:
    """Convert frame number to seconds."""
    if fps <= 0:
        return 0.0
    return frame_num / fps
