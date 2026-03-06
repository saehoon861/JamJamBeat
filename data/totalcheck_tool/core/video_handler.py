"""core/video_handler.py - OpenCV VideoCapture 래퍼 모듈 (labeling_tool 동일)"""

import cv2
import numpy as np


class VideoHandler:
    """OpenCV VideoCapture를 감싸는 래퍼 클래스."""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise IOError(f"동영상을 열 수 없습니다: {video_path}")

        self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def get_frame(self, frame_index: int) -> np.ndarray | None:
        """지정된 인덱스의 프레임을 읽어 반환합니다."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def release(self) -> None:
        """VideoCapture 리소스를 해제합니다."""
        if self.cap.isOpened():
            self.cap.release()
