"""main.py - [Controller] 통합 검수 도구 진입점 및 이벤트 루프

실행 방법:
    cd data/totalcheck_tool
    uv run python main.py

기능:
  - raw_data의 영상, labeled_data의 라벨, landmark_data의 손 랜드마크를 통합 시각화
  - 잘못된 라벨을 수정하고 Enter로 저장 (labeled_data CSV 덮어쓰기)
  - n 키로 저장 없이 다음 영상 스킵
  - Esc로 저장 없이 즉시 종료
"""

import sys
import os
from pathlib import Path

# OpenCV Qt 플러그인 충돌 방지
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""

import cv2

# sys.path 등록
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.class_pipeline import LabelConfigManager
from core.annotation_state import AnnotationState
from core.video_handler import VideoHandler
from ui.osd_renderer import OSDRenderer
from ui.skeleton_renderer import draw_landmarks
from file_io.data_loader import get_checkable_stems, load_session_data
from file_io.data_exporter import overwrite_labels


# --- 경로 설정 (__file__ 기준 상대 경로) ---
TOOL_DIR   = Path(__file__).resolve().parent         # totalcheck_tool/
DATA_DIR   = TOOL_DIR.parent                         # data/
RAW_DIR    = str(DATA_DIR / "raw_data")
LABELED_DIR = str(DATA_DIR / "labeled_data")
LANDMARK_DIR = str(DATA_DIR / "landmark_data")

# --- 숫자 키 매핑 (상단 숫자키 + 텐키패드 모두 포괄) ---
# 상단 숫자키 48(0)~54(6), 텐키패드 176(0)~182(6)
NUMBER_KEY_MAP = {}
for _digit in range(7):  # 0~6
    NUMBER_KEY_MAP[48 + _digit] = _digit   # 상단 키: '0'~'6'
    NUMBER_KEY_MAP[176 + _digit] = _digit  # 텐키패드: Numpad 0~6


def render_frame(
    video: VideoHandler,
    renderer: OSDRenderer,
    state: AnnotationState,
    config: LabelConfigManager,
    landmark_df,
    current_idx: int,
    limit_frames: int,
    warning_msg: str,
) -> None:
    """현재 프레임을 읽고, 랜드마크 + OSD + 타임라인을 합성하여 imshow로 표시합니다."""
    frame = video.get_frame(current_idx)
    if frame is None:
        return

    # 1) 랜드마크 스켈레톤 덧그리기 (OSD 이전에 적용)
    if current_idx < len(landmark_df):
        landmark_row = landmark_df.iloc[current_idx]
        draw_landmarks(frame, landmark_row)

    # 2) OSD + 타임라인 합성
    current_label = state.get_label(current_idx)
    combined = renderer.draw_ui(
        frame=frame,
        current_idx=current_idx,
        total_frames=limit_frames,
        fps=video.fps,
        current_label=current_label,
        target_class=state.target_class,
        start_marker=state.start_marker,
        warning_msg=warning_msg,
        labels=state.labels,
    )
    cv2.imshow("Total Check Tool", combined)


def process_video(stem: str, config: LabelConfigManager) -> str:
    """단일 영상에 대한 검수 세션을 실행합니다.

    Returns:
        'saved'   - Enter로 저장 완료
        'skipped' - n 키로 스킵
        'quit'    - Esc로 종료
    """
    print(f"\n{'='*60}")
    print(f"[검수] {stem}")
    print(f"{'='*60}")

    session = load_session_data(stem, RAW_DIR, LABELED_DIR, LANDMARK_DIR)
    if session is None:
        return "skipped"

    # 3개 데이터 길이 중 최솟값으로 유효 프레임 수 결정
    video_frames = int(session.cap.get(cv2.CAP_PROP_FPS and cv2.CAP_PROP_FRAME_COUNT))
    video_frames = int(session.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit_frames = min(
        video_frames,
        len(session.labeled_df),
        len(session.landmark_df),
    )
    session.cap.release()  # VideoHandler로 재오픈
    video = VideoHandler(session.video_path)

    print(f"[INFO] 유효 프레임: {limit_frames} (video={video_frames}, "
          f"labeled={len(session.labeled_df)}, landmark={len(session.landmark_df)})")

    # 기존 labeled_data의 gesture 컬럼으로 State 초기화
    filename = os.path.basename(session.video_path)
    target_class = config.parse_target_from_filename(filename)
    existing_labels = session.labeled_df["gesture"].tolist()[:limit_frames]
    state = AnnotationState(existing_labels=existing_labels, target_class=target_class)

    renderer = OSDRenderer()
    landmark_df = session.landmark_df

    current_idx = 0
    warning_msg = ""

    # 최초 진입 시 첫 프레임 렌더링
    render_frame(video, renderer, state, config, landmark_df,
                 current_idx, limit_frames, warning_msg)

    # 이벤트 루프
    while True:
        key = cv2.waitKey(0) & 0xFF
        warning_msg = ""  # 매 루프: 경고 메세지 초기화

        # ===== Esc: 저장 없이 즉시 종료 =====
        if key == 27:
            print("[INFO] ESC 입력 - 저장 없이 즉시 종료합니다.")
            video.release()
            cv2.destroyAllWindows()
            return "quit"

        # ===== Enter: 저장 시도 =====
        elif key == 13:
            if state.is_range_active():
                warning_msg = "[WARNING: Cannot save! Close range with 'F' first, or press ESC to quit]"
                print(f"[WARN] S 마커 활성 중 (frame {state.start_marker}). 저장 불가.")
            else:
                csv_path = overwrite_labels(
                    stem=stem,
                    labels=state.labels,
                    labeled_dir=LABELED_DIR,
                )
                print(f"[INFO] 저장 완료: {csv_path}")
                video.release()
                cv2.destroyAllWindows()
                return "saved"

        # ===== n / N: 저장 없이 다음 영상으로 스킵 =====
        elif key in (ord('n'), ord('N')):
            print("[INFO] 'n' 입력 - 저장하지 않고 다음 영상으로 넘어갑니다.")
            video.release()
            cv2.destroyAllWindows()
            return "skipped"

        # ===== Navigation: D / → 다음 프레임 =====
        elif key in (ord('d'), ord('D'), 83):
            current_idx = min(current_idx + 1, limit_frames - 1)

        # ===== Navigation: A / ← 이전 프레임 =====
        elif key in (ord('a'), ord('A'), 81):
            current_idx = max(current_idx - 1, 0)

        # ===== Navigation: E → +10 프레임 =====
        elif key in (ord('e'), ord('E')):
            current_idx = min(current_idx + 10, limit_frames - 1)

        # ===== Navigation: Q → -10 프레임 =====
        elif key in (ord('q'), ord('Q')):
            current_idx = max(current_idx - 10, 0)

        # ===== Labeling: Space → 단일 프레임 토글 =====
        elif key == 32:
            state.toggle_frame(current_idx)

        # ===== Labeling: 숫자 0~6 → 현재 프레임 직접 라벨 지정 =====
        elif key in NUMBER_KEY_MAP:
            class_id = NUMBER_KEY_MAP[key]
            state.set_label(current_idx, class_id)
            print(f"[INFO] frame {current_idx} → label {class_id}")

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
            # S 없으면 조용히 무시

        # --- 렌더링 갱신 ---
        render_frame(video, renderer, state, config, landmark_df,
                     current_idx, limit_frames, warning_msg)


def main():
    """메인 진입점: 검수 가능한 영상을 순회하며 검수 세션을 수행합니다."""
    print("=" * 60)
    print("  🔍 Total Check Tool - 통합 검수 도구")
    print("=" * 60)

    config = LabelConfigManager()
    config.load_config()

    stems = get_checkable_stems(LABELED_DIR, LANDMARK_DIR, RAW_DIR)

    if not stems:
        print("[INFO] 검수 가능한 영상이 없습니다.")
        print(f"  - labeled_data: {LABELED_DIR}")
        print(f"  - landmark_data: {LANDMARK_DIR}")
        print(f"  - raw_data: {RAW_DIR}")
        print("  위 세 폴더 모두에 동일한 stem의 파일이 있어야 합니다.")
        return

    print(f"[INFO] 검수 대상 영상 {len(stems)}개:")
    for i, s in enumerate(stems, 1):
        print(f"  {i}. {s}")
    print()

    saved_count = 0
    skipped_count = 0

    for idx, stem in enumerate(stems, 1):
        print(f"\n[{idx}/{len(stems)}] 검수 시작: {stem}")
        result = process_video(stem, config)

        if result == "saved":
            saved_count += 1
        elif result == "skipped":
            skipped_count += 1
        elif result == "quit":
            print(f"\n[INFO] 도구를 종료합니다. (저장: {saved_count}개, 스킵: {skipped_count}개)")
            sys.exit(0)

    print(f"\n{'='*60}")
    print(f"  🎉 검수 완료! (저장: {saved_count}개, 스킵: {skipped_count}개)")
    print(f"{'='*60}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
