"""main.py - [Controller] 통합 검수 도구 진입점 및 이벤트 루프 (Phase 2)

실행 방법:
    cd data/totalcheck_tool
    uv run python main.py

Phase 2 변경사항:
  - labeled_data 없이도 백지 라벨링 가능 (raw + landmark만 필요)
  - Enter 저장 시 labeled_data + total_data 동시 생성
  - S/F 구간 라벨링 3단계 State Machine (S→F→0~6 확정)
  - 1단계/3단계 키 입력 방어 및 OSD 경고 메세지
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
from file_io.data_exporter import save_all_data


# --- 경로 설정 (__file__ 기준 상대 경로) ---
TOOL_DIR   = Path(__file__).resolve().parent         # totalcheck_tool/
DATA_DIR   = TOOL_DIR.parent                         # data/
RAW_DIR    = str(DATA_DIR / "raw_data")
LABELED_DIR = str(DATA_DIR / "labeled_data")
LANDMARK_DIR = str(DATA_DIR / "landmark_data")
TOTAL_DIR  = str(DATA_DIR / "total_data")

# --- 숫자 키 매핑 (상단 숫자키 + 텐키패드 모두 포괄) ---
# 상단 숫자키 48(0)~54(6), 텐키패드 176(0)~182(6)
# --- 화살표 키 상수 (waitKeyEx 전체 키코드, 문자키와 충돌 방지) ---
# 좌우 화살표 키코드 83=ord('S'), 81=ord('Q')와 충돌하므로 waitKeyEx 사용 필수
KEY_LEFT  = 65361   # ← 화살표 (Linux/Qt)
KEY_RIGHT = 65363   # → 화살표 (Linux/Qt)

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

    # --- limit_frames: 3개(or 2개) 데이터 길이 중 최솟값 ---
    video_frames = int(session.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if session.labeled_df is not None:
        limit_frames = min(
            video_frames,
            len(session.landmark_df),
            len(session.labeled_df),
        )
    else:
        limit_frames = min(video_frames, len(session.landmark_df))

    session.cap.release()  # VideoHandler로 재오픈
    video = VideoHandler(session.video_path)

    labeled_status = f"labeled={len(session.labeled_df)}" if session.labeled_df is not None else "labeled=없음(백지)"
    print(f"[INFO] 유효 프레임: {limit_frames} (video={video_frames}, "
          f"landmark={len(session.landmark_df)}, {labeled_status})")

    # --- State 초기화 ---
    filename = os.path.basename(session.video_path)
    target_class = config.parse_target_from_filename(filename)

    if session.labeled_df is not None:
        existing_labels = session.labeled_df["gesture"].tolist()[:limit_frames]
        state = AnnotationState(existing_labels=existing_labels, target_class=target_class)
    else:
        state = AnnotationState(existing_labels=None, target_class=target_class, total_frames=limit_frames)

    renderer = OSDRenderer()
    landmark_df = session.landmark_df

    current_idx = 0
    warning_msg = ""

    # 최초 진입 시 첫 프레임 렌더링
    render_frame(video, renderer, state, config, landmark_df,
                 current_idx, limit_frames, warning_msg)

    # ===== 이벤트 루프 =====
    while True:
        # waitKeyEx: 화살표 키코드(65361/65363)와 문자 키코드(ord('S')=83)를 구분하기 위해 사용
        raw_key = cv2.waitKeyEx(0)
        key = raw_key & 0xFF  # 문자 키 비교용
        is_arrow_left = (raw_key == KEY_LEFT)
        is_arrow_right = (raw_key == KEY_RIGHT)

        # --- OSD Warning 상태 기반 자동 세팅 ---
        if state.is_waiting_for_class():
            warning_msg = f"[WAITING CLASS] Press 0~6 to fill range (frame {state.start_marker}~{state.end_marker}), or S to cancel"
        elif state.is_range_active():
            warning_msg = f"[Range Start: frame {state.start_marker}] Press F to set end, or S to cancel"
        else:
            warning_msg = ""

        # ===== 1. Esc: 상태 불문 즉시 강제 종료 (절대키) =====
        if key == 27:
            print("[INFO] ESC 입력 - 저장 없이 즉시 종료합니다.")
            video.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Qt 이벤트 루프 플러시 (창 깨짐 방지)
            return "quit"

        # ===== 2. 대기 모드(3단계) 방어: 0~6, S, Esc만 허용 =====
        if state.is_waiting_for_class():
            if key in NUMBER_KEY_MAP and not is_arrow_left and not is_arrow_right:
                class_id = NUMBER_KEY_MAP[key]
                start = state.start_marker
                end = state.end_marker
                state.fill_range_with_class(class_id)
                print(f"[INFO] 구간 라벨링 완료: frame {start} ~ {end} → class {class_id}")
            elif key in (ord('s'), ord('S')) and not is_arrow_right:
                state.cancel_range()
                print("[INFO] 구간 라벨링 취소됨 (S)")
            elif key in (ord('n'), ord('N')):
                warning_msg = "[WARNING: Cannot skip! Finish range or press S to cancel]"
                print("[WARN] 구간 작업 중에는 스킵 불가. F로 확정하거나 S로 취소하세요.")
            # 그 외 키는 전부 무시 (탐색, 스페이스, Enter, 화살표 등)
            render_frame(video, renderer, state, config, landmark_df,
                         current_idx, limit_frames, warning_msg)
            continue

        # ===== 3. 1단계(S 찍힌 상태) 방어: 탐색+F+S+Esc만 허용 =====
        if state.is_range_active() and not state.is_waiting_for_class():
            # 허용 키: F, S(취소), 탐색(A,D,Q,E,←/→), Esc (이미 위에서 처리)
            allowed_in_stage1 = False

            if key in (ord('f'), ord('F')) and not is_arrow_left and not is_arrow_right:
                state.set_range_end(current_idx)
                # F 입력 즉시 WAITING CLASS OSD 표시
                warning_msg = f"[WAITING CLASS] Press 0~6 to fill range (frame {state.start_marker}~{state.end_marker}), or S to cancel"
                print(f"[INFO] 구간 끝 마킹: frame {current_idx} (0~6 입력 대기)")
                allowed_in_stage1 = True
            elif key in (ord('s'), ord('S')) and not is_arrow_right:
                state.cancel_range()
                print("[INFO] 구간 라벨링 취소됨 (S)")
                allowed_in_stage1 = True
            elif is_arrow_right or key in (ord('d'), ord('D')):
                current_idx = min(current_idx + 1, limit_frames - 1)
                allowed_in_stage1 = True
            elif is_arrow_left or key in (ord('a'), ord('A')):
                current_idx = max(current_idx - 1, 0)
                allowed_in_stage1 = True
            elif key in (ord('e'), ord('E')):
                current_idx = min(current_idx + 10, limit_frames - 1)
                allowed_in_stage1 = True
            elif key in (ord('q'), ord('Q')) and not is_arrow_left:
                current_idx = max(current_idx - 10, 0)
                allowed_in_stage1 = True
            elif key in (ord('n'), ord('N')):
                warning_msg = "[WARNING: Cannot skip! Finish range or press S to cancel]"
                print("[WARN] 구간 작업 중에는 스킵 불가.")
                allowed_in_stage1 = True

            if not allowed_in_stage1:
                # 스페이스바, Enter, 0~6 등 그 외 키는 무시
                pass

            render_frame(video, renderer, state, config, landmark_df,
                         current_idx, limit_frames, warning_msg)
            continue

        # ===== 이하: 평시(0단계) 키 바인딩 =====

        # ===== Enter: 저장 시도 =====
        if key == 13:
            labeled_path, total_path = save_all_data(
                stem=stem,
                labels=state.labels,
                labeled_dir=LABELED_DIR,
                total_dir=TOTAL_DIR,
                landmark_df=landmark_df.copy(),  # .copy()로 in-memory 오염 방지
                raw_fps=video.fps,
            )
            print(f"[INFO] 저장 완료:")
            print(f"  labeled: {labeled_path}")
            print(f"  total:   {total_path}")
            video.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Qt 이벤트 루프 플러시 (창 깨짐 방지)
            return "saved"

        # ===== n / N: 저장 없이 다음 영상으로 스킵 =====
        elif key in (ord('n'), ord('N')):
            print("[INFO] 'n' 입력 - 저장하지 않고 다음 영상으로 넘어갑니다.")
            video.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Qt 이벤트 루프 플러시 (창 깨짐 방지)
            return "skipped"

        # ===== Navigation: D / → 다음 프레임 =====
        elif is_arrow_right or key in (ord('d'), ord('D')):
            current_idx = min(current_idx + 1, limit_frames - 1)

        # ===== Navigation: A / ← 이전 프레임 =====
        elif is_arrow_left or key in (ord('a'), ord('A')):
            current_idx = max(current_idx - 1, 0)

        # ===== Navigation: E → +10 프레임 =====
        elif key in (ord('e'), ord('E')):
            current_idx = min(current_idx + 10, limit_frames - 1)

        # ===== Navigation: Q → -10 프레임 =====
        elif key in (ord('q'), ord('Q')):
            current_idx = max(current_idx - 10, 0)

        # ===== Labeling: Space → 단일 프레임 토글 (target_class ↔ 0) =====
        elif key == 32:
            state.toggle_frame(current_idx)

        # ===== Labeling: 숫자 0~6 → 현재 프레임 직접 라벨 지정 =====
        elif key in NUMBER_KEY_MAP:
            class_id = NUMBER_KEY_MAP[key]
            state.set_label(current_idx, class_id)
            print(f"[INFO] frame {current_idx} → label {class_id}")

        # ===== Labeling: S → 구간 시작 마킹 (1단계 진입) =====
        elif key in (ord('s'), ord('S')):
            state.set_range_start(current_idx)
            print(f"[INFO] S 마커 설정: frame {current_idx}")

        # ===== Labeling: F → 평시에는 무시 =====
        elif key in (ord('f'), ord('F')):
            pass  # S가 안 찍혀있으면 아무 작업도 안 함

        # --- 렌더링 갱신 ---
        render_frame(video, renderer, state, config, landmark_df,
                     current_idx, limit_frames, warning_msg)


def main():
    """메인 진입점: 검수 가능한 영상을 순회하며 검수 세션을 수행합니다."""
    print("=" * 60)
    print("  🔍 Total Check Tool - 통합 검수 도구 (Phase 2)")
    print("=" * 60)

    config = LabelConfigManager()
    config.load_config()

    stems = get_checkable_stems(LABELED_DIR, LANDMARK_DIR, RAW_DIR)

    if not stems:
        print("[INFO] 검수 가능한 영상이 없습니다.")
        print(f"  - landmark_data: {LANDMARK_DIR}")
        print(f"  - raw_data: {RAW_DIR}")
        print("  위 두 폴더 모두에 동일한 stem의 파일이 있어야 합니다.")
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
