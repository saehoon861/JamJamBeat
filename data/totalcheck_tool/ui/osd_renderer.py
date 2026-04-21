"""ui/osd_renderer.py - [View] OSD 및 타임라인 바 렌더링 모듈

labeling_tool의 osd_renderer.py를 기반으로, 단축키 가이드를 검수 도구에 맞게 수정.
(Enter:Save, n:Skip, Esc:Quit 추가, 숫자키 0~6 라벨 직접 설정 안내)
"""

import cv2
import numpy as np

from core.class_pipeline import LabelConfigManager


class OSDRenderer:
    """프레임 이미지 위에 OSD 텍스트 및 하단 타임라인 바를 합성하는 View 클래스."""

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SMALL = cv2.FONT_HERSHEY_PLAIN

    def __init__(self):
        self.config_mgr = LabelConfigManager()

    @staticmethod
    def _format_time(frame_idx: int, fps: float) -> str:
        """프레임 인덱스를 'mm:ss:ms' 형태 문자열로 변환합니다."""
        total_ms = int((frame_idx / fps) * 1000)
        minutes = total_ms // 60000
        seconds = (total_ms % 60000) // 1000
        ms = total_ms % 1000
        return f"{minutes:02d}:{seconds:02d}:{ms:03d}"

    def draw_ui(
        self,
        frame: np.ndarray,
        current_idx: int,
        total_frames: int,
        fps: float,
        current_label: int,
        target_class: int,
        start_marker: int | None,
        warning_msg: str,
        labels: list[int],
    ) -> np.ndarray:
        """OSD 텍스트를 프레임에 오버레이하고 타임라인을 합성하여 반환합니다."""
        display = frame.copy()
        h, w = display.shape[:2]

        # --- 좌측 상단: 프레임/시간 정보 ---
        current_time = self._format_time(current_idx, fps)
        total_time = self._format_time(total_frames - 1, fps)

        cv2.putText(display, f"Frame: {current_idx} / {total_frames - 1}",
                    (10, 30), self.FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, f"Time: {current_time} / {total_time}",
                    (10, 60), self.FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # --- 우측 상단: 현재 프레임 라벨 + 타겟 ---
        label_text = f"Label: {self.config_mgr.get_display_label(current_label)}"
        text_size = cv2.getTextSize(label_text, self.FONT, 0.7, 2)[0]
        cv2.putText(display, label_text, (w - text_size[0] - 10, 30),
                    self.FONT, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        target_text = f"Target: {self.config_mgr.get_display_label(target_class)}"
        text_size2 = cv2.getTextSize(target_text, self.FONT, 0.6, 2)[0]
        cv2.putText(display, target_text, (w - text_size2[0] - 10, 60),
                    self.FONT, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

        # --- S 마커 시각 피드백 ---
        if start_marker is not None:
            range_text = f"[RANGE MODE: S marked at frame {start_marker}]"
            rt_size = cv2.getTextSize(range_text, self.FONT, 0.7, 2)[0]
            rx = (w - rt_size[0]) // 2
            cv2.putText(display, range_text, (rx, 30),
                        self.FONT, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # --- 동적 경고 문구 ---
        if warning_msg:
            warn_size = cv2.getTextSize(warning_msg, self.FONT, 0.8, 2)[0]
            wx = (w - warn_size[0]) // 2
            wy = h // 2
            cv2.rectangle(display, (wx - 10, wy - 30),
                          (wx + warn_size[0] + 10, wy + 10), (0, 0, 0), -1)
            cv2.putText(display, warning_msg, (wx, wy),
                        self.FONT, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        # --- 하단 단축키 가이드 (검수 도구 전용) ---
        guide = "A/<-:Prev  D/->:Next  Q:-10  E:+10  Space:Toggle  0-6:Label  S:Start  F:Finish  Enter:Save  n:Skip  ESC:Quit"
        cv2.putText(display, guide, (10, h - 10),
                    self.FONT_SMALL, 1.0, (180, 180, 180), 1, cv2.LINE_AA)

        # --- 타임라인 바 합성 ---
        timeline = self._draw_timeline(
            width=w,
            current_idx=current_idx,
            total_frames=total_frames,
            labels=labels,
            start_marker=start_marker,
        )

        combined = np.vstack([display, timeline])
        return combined

    def _draw_timeline(
        self,
        width: int,
        current_idx: int,
        total_frames: int,
        labels: list[int],
        start_marker: int | None,
    ) -> np.ndarray:
        """하단 타임라인 바를 생성합니다."""
        bar_height = self.config_mgr.timeline_bar_height
        half_range = self.config_mgr.timeline_half_range
        total_display = half_range * 2 + 1

        timeline = np.zeros((bar_height, width, 3), dtype=np.uint8)
        pixel_per_frame = max(1, width // total_display)

        for i in range(total_display):
            frame_idx = current_idx - half_range + i
            x_start = i * pixel_per_frame
            x_end = min(x_start + pixel_per_frame, width)

            if frame_idx < 0 or frame_idx >= total_frames:
                continue

            label = labels[frame_idx]
            color = self.config_mgr.get_class_color(label)
            cv2.rectangle(timeline, (x_start, 0), (x_end, bar_height), color, -1)

        # 현재 프레임: 흰색 세로선
        center_x = half_range * pixel_per_frame + pixel_per_frame // 2
        if center_x < width:
            cv2.line(timeline, (center_x, 0), (center_x, bar_height), (255, 255, 255), 2)

        # S 마커: 노란색 세로선
        if start_marker is not None:
            marker_offset = start_marker - current_idx + half_range
            if 0 <= marker_offset < total_display:
                marker_x = marker_offset * pixel_per_frame + pixel_per_frame // 2
                if marker_x < width:
                    cv2.line(timeline, (marker_x, 0), (marker_x, bar_height), (0, 255, 255), 2)

        return timeline
