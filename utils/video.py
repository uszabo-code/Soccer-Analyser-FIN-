"""Video reading utilities and timestamp helpers."""

import cv2


class VideoReader:
    """Efficient video reader with frame skipping."""

    def __init__(self, video_path: str, frame_skip: int = 3,
                 start_frame: int = 0, end_frame: int = None):
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

        # Optional frame range (for running on a segment without re-encoding)
        self.start_frame = max(0, int(start_frame))
        self.end_frame = int(end_frame) if end_frame is not None else self.total_frames
        self.end_frame = min(self.end_frame, self.total_frames)
        if self.start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

    @property
    def frames_to_process(self) -> int:
        span = max(0, self.end_frame - self.start_frame)
        return span // self.frame_skip

    def iter_frames(self):
        """Yield (frame_num, frame) tuples, respecting frame_skip and range."""
        frame_num = self.start_frame
        while frame_num < self.end_frame:
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
