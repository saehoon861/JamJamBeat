"""main.py - [Controller] 동작테스트 라벨링 툴 진입점 및 이벤트 루프 오케스트레이션

실행 방법:
    $ cd /home/user/projects/JamJamBeat-model3/동작테스트/labeling_tool
    $ uv run python main.py
"""

import sys
import os
from pathlib import Path

# OpenCV 번들 Qt 플러그인이 시스템 xcb와 충돌하는 문제 우회
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""

import cv2

# 모듈 임포트를 위해 현재 파일 위치를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.class_pipeline import LabelConfigManager
from core.annotation_state import AnnotationState
from core.video_handler import VideoHandler
from ui.osd_renderer import OSDRenderer
from file_io.file_scanner import get_pending_videos
from file_io.data_exporter import export_labels


def get_project_paths() -> tuple[str, str]:
    """동작테스트 워크스페이스 경로를 계산하여 (raw_data_dir, labeled_data_dir)를 반환합니다."""
    tool_dir = Path(__file__).resolve().parent       # 동작테스트/labeling_tool/
    workspace_dir = tool_dir.parent                  # 동작테스트/
    raw_data_dir = str(workspace_dir / "raw_data")
    labeled_data_dir = str(workspace_dir / "labeled_data")
    return raw_data_dir, labeled_data_dir


def render_frame(
    video: VideoHandler,
    renderer: OSDRenderer,
    state: AnnotationState,
    config: LabelConfigManager,
    current_idx: int,
    warning_msg: str,
) -> None:
    """현재 프레임을 읽어 OSD + 타임라인을 합성하여 imshow로 표시합니다."""
    frame = video.get_frame(current_idx)
    if frame is None:
        return

    current_label = state.get_label(current_idx)
    combined = renderer.draw_ui(
        frame=frame,
        current_idx=current_idx,
        total_frames=video.total_frames,
        fps=config.fps,
        current_label=current_label,
        target_class=state.target_class,
        start_marker=state.start_marker,
        warning_msg=warning_msg,
        labels=state.labels,
    )
    cv2.imshow("Labeling Tool", combined)


def process_video(video_path: str, labeled_data_dir: str) -> bool:
    """단일 영상 파일에 대한 라벨링 세션을 수행합니다.

    Returns:
        True: 정상 저장 후 다음 영상으로 이동
        False: ESC로 종료 (sys.exit 호출됨)
    """
    config = LabelConfigManager()
    renderer = OSDRenderer()

    filename = os.path.basename(video_path)
    target_class = config.parse_target_from_filename(filename)

    print(f"\n{'='*60}")
    print(f"[INFO] 영상 로드: {filename}")
    print(f"[INFO] Target 클래스: {config.get_display_label(target_class)}")
    print(f"{'='*60}")

    video = VideoHandler(video_path)
    state = AnnotationState(
        total_frames=video.total_frames,
        target_class=target_class,
    )

    current_idx = 0
    warning_msg = ""

    # --- 최초 진입 시 첫 프레임 렌더링 (waitKey 블로킹 전 화면을 반드시 보여줌) ---
    render_frame(video, renderer, state, config, current_idx, warning_msg)

    # --- 이벤트 루프 ---
    while True:
        key = cv2.waitKey(0) & 0xFF

        # 매 루프마다 경고 메세지 초기화 (직전 턴 메세지 제거)
        warning_msg = ""

        # ===== ESC: 즉시 종료 (저장 없음) =====
        if key == 27:  # ESC
            print("[INFO] ESC 입력 - 저장 없이 프로그램을 즉시 종료합니다.")
            video.release()
            cv2.destroyAllWindows()
            sys.exit(0)

        # ===== Enter: 저장 시도 =====
        elif key == 13:  # Enter
            if state.is_range_active():
                warning_msg = "[WARNING: Cannot save! Close range with 'F' first, or press ESC to quit]"
                print(f"[WARN] S 마커가 활성 상태 (frame {state.start_marker}). 저장 불가.")
            else:
                csv_path = export_labels(
                    labels=state.labels,
                    video_filename=filename,
                    output_dir=labeled_data_dir,
                    fps=config.fps,
                )
                print(f"[INFO] 저장 완료: {csv_path}")
                video.release()
                cv2.destroyAllWindows()
                return True

        # ===== Navigation: D / 오른쪽 화살표 → 다음 프레임 =====
        elif key in (ord('d'), ord('D'), 83):  # 83 = 오른쪽 화살표
            current_idx = min(current_idx + 1, video.total_frames - 1)

        # ===== Navigation: A / 왼쪽 화살표 → 이전 프레임 =====
        elif key in (ord('a'), ord('A'), 81):  # 81 = 왼쪽 화살표
            current_idx = max(current_idx - 1, 0)

        # ===== Navigation: E → +10 프레임 =====
        elif key in (ord('e'), ord('E')):
            current_idx = min(current_idx + 10, video.total_frames - 1)

        # ===== Navigation: Q → -10 프레임 =====
        elif key in (ord('q'), ord('Q')):
            current_idx = max(current_idx - 10, 0)

        # ===== Labeling: Space → 단일 프레임 토글 =====
        elif key == 32:  # Space
            state.toggle_frame(current_idx)

        # ===== Labeling: S → 구간 시작 마킹 =====
        elif key in (ord('s'), ord('S')):
            state.set_range_start(current_idx)
            print(f"[INFO] S 마커 설정: frame {current_idx}")

        # ===== Labeling: F → 구간 끝 마킹 및 덮어쓰기 =====
        elif key in (ord('f'), ord('F')):
            if state.is_range_active():
                start = state.start_marker
                state.set_range_end(current_idx, target_class)
                print(f"[INFO] 구간 라벨링 완료: frame {start} ~ {current_idx}")
            # S가 없으면 조용히 무시 (pass)

        # --- 렌더링 갱신 ---
        render_frame(video, renderer, state, config, current_idx, warning_msg)


def main():
    """메인 진입점: 미작업 영상을 순회하며 라벨링 세션을 수행합니다."""
    config = LabelConfigManager()
    config.load_config()

    raw_data_dir, labeled_data_dir = get_project_paths()

    print("=" * 60)
    print("  🎮 동작테스트 Hand Gesture Frame Labeling Tool")
    print("=" * 60)

    pending = get_pending_videos(raw_data_dir, labeled_data_dir)

    if not pending:
        print("[INFO] 모든 영상이 이미 라벨링 완료되었습니다.")
        print(f"[INFO] raw_data: {raw_data_dir}")
        print(f"[INFO] labeled_data: {labeled_data_dir}")
        return

    print(f"[INFO] 미작업 영상 {len(pending)}개 발견:")
    for i, vp in enumerate(pending, 1):
        print(f"  {i}. {os.path.basename(vp)}")
    print()

    for video_path in pending:
        process_video(video_path, labeled_data_dir)

    print("\n[INFO] 모든 영상 라벨링이 완료되었습니다! 🎉")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
