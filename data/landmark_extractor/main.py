"""main.py - MediaPipe Hand Landmarker를 이용한 핸드 랜드마크 자동 추출 도구

각 동영상 파일의 모든 프레임에서 21개 손 랜드마크(Landmarks, 정규화 좌표)의
x, y, z 값을 추출하여 CSV 파일로 저장합니다.

실행 방법:
    cd data/landmark_extractor
    uv run python main.py
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# --- MediaPipe Tasks API 임포트 ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# --- 경로 설정 (__file__ 기준 상대 경로) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # landmark_extractor → data → JamJamBeat
MODEL_PATH = str(PROJECT_ROOT / "hand_landmarker.task")
RAW_DATA_DIR = str(PROJECT_ROOT / "data" / "raw_data")
OUTPUT_DIR = str(PROJECT_ROOT / "data" / "landmark_data")

# --- 랜드마크 컬럼 헤더 생성 (x0, y0, z0, x1, y1, z1, ..., x20, y20, z20) ---
LANDMARK_COLUMNS = []
for i in range(21):
    LANDMARK_COLUMNS.extend([f"x{i}", f"y{i}", f"z{i}"])


def get_pending_videos(raw_dir: str, output_dir: str) -> list[str]:
    """raw_data에서 .mp4 파일을 스캔하고, 이미 처리된 파일을 제외한 목록을 반환합니다.

    OS별 정렬 불일치를 방지하기 위해 sorted()를 사용합니다.
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)

    # 이미 처리된 CSV의 stem 목록
    done_stems = set()
    if output_path.exists():
        done_stems = {f.stem for f in sorted(output_path.glob("*.csv"))}

    # 미처리 영상 필터링
    pending = []
    for mp4 in sorted(raw_path.glob("*.mp4")):
        if mp4.stem not in done_stems:
            pending.append(str(mp4))

    return pending


def format_timestamp(frame_idx: int, fps: float) -> str:
    """프레임 인덱스를 'mm:ss:ms' 형태 문자열로 변환합니다.
    
    예: frame_idx=26, fps=30.0 -> '00:00:866'
    """
    total_ms = int((frame_idx / fps) * 1000)
    minutes = total_ms // 60000
    seconds = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{minutes:02d}:{seconds:02d}:{ms:03d}"


def format_timestamp_from_ms(total_ms: int) -> str:
    """밀리초 값을 직접 받아 'mm:ss:ms' 형태 문자열로 변환합니다.
    
    OpenCV CAP_PROP_POS_MSEC 기반 실제 타임스탬프 저장용.
    예: total_ms=866 -> '00:00:866'
    """
    minutes = total_ms // 60000
    seconds = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{minutes:02d}:{seconds:02d}:{ms:03d}"


def extract_landmarks(result) -> list:
    """HandLandmarkerResult에서 21개 랜드마크의 x, y, z 좌표를 추출합니다.

    손이 감지되지 않으면 [None] * 63을 반환합니다 (IndexError 방지).
    result.hand_landmarks (Landmarks, 정규화 좌표)만 사용합니다.
    result.hand_world_landmarks (WorldLandmarks)는 사용하지 않습니다.
    """
    if len(result.hand_landmarks) > 0:
        landmarks = result.hand_landmarks[0]  # 첫 번째 손의 21개 랜드마크
        values = []
        for lm in landmarks:
            values.extend([lm.x, lm.y, lm.z])
        return values
    else:
        return [None] * 63


def process_video(video_path: str, output_dir: str, model_path: str) -> str:
    """단일 영상에서 프레임별 핸드 랜드마크를 추출하여 CSV로 저장합니다.

    HandLandmarker 인스턴스는 영상 1개당 with 블록 1개로 생성/소멸합니다.
    (여러 영상에서 재사용하면 timestamp monotonically increasing 에러 발생)

    Returns:
        저장된 CSV 파일의 절대 경로
    """
    filename = os.path.basename(video_path)
    stem = Path(filename).stem

    # 영상 열기 및 메타데이터 확보
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"동영상을 열 수 없습니다: {video_path}")

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  FPS: {actual_fps}, Total Frames: {total_frames}")

    # HandLandmarker 옵션 설정 (IMAGE 모드: 프레임 단위 독립 추론, 추적/예측 없음)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        # 주의: IMAGE 모드에서는 min_tracking_confidence를 설정하면 에러 발생하므로 삭제
    )

    rows = []

    # ✅ 영상마다 새 인스턴스 생성 (with 블록)
    with HandLandmarker.create_from_options(options) as landmarker:
        for frame_idx in range(total_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # BGR → RGB 변환 (MediaPipe 요구사항)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # OSD UI 표시용 수학적 타임스탬프 (라벨링 파일과 동일한 계산식)
            timestamp_ms = int(frame_idx / actual_fps * 1000)

            # MediaPipe Image 생성 및 추론 (IMAGE 모드: 타임스탬프 인자 없음)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect(mp_image)

            # 랜드마크 추출 (미감지 시 NaN)
            landmark_values = extract_landmarks(result)

            # 행 데이터 조립: [frame_idx, timestamp, gesture(None), x0, y0, z0, ..., x20, y20, z20]
            row = [frame_idx, format_timestamp(frame_idx, actual_fps), None] + landmark_values
            rows.append(row)

            # 진행률 출력 (30프레임마다 또는 마지막 프레임)
            if (frame_idx + 1) % 30 == 0 or frame_idx == total_frames - 1:
                pct = (frame_idx + 1) / total_frames * 100
                print(f"  Frame {frame_idx + 1}/{total_frames} ({pct:.1f}%)")

    cap.release()

    # DataFrame 조립 및 CSV 저장
    columns = ["frame_idx", "timestamp", "gesture"] + LANDMARK_COLUMNS
    df = pd.DataFrame(rows, columns=columns)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / f"{stem}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    return str(csv_path.resolve())


def main():
    """진입점: 미처리 영상을 순회하며 랜드마크 추출을 수행합니다."""
    print("=" * 60)
    print("  🖐️ MediaPipe Hand Landmark Extractor")
    print("=" * 60)

    # 모델 파일 존재 확인
    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        print(f"[ERROR] hand_landmarker.task 파일을 프로젝트 루트에 배치해 주세요.")
        return

    pending = get_pending_videos(RAW_DATA_DIR, OUTPUT_DIR)

    if not pending:
        print("[INFO] 모든 영상이 이미 처리 완료되었습니다.")
        print(f"[INFO] raw_data: {RAW_DATA_DIR}")
        print(f"[INFO] landmark_data: {OUTPUT_DIR}")
        return

    print(f"[INFO] 미처리 영상 {len(pending)}개 발견:")
    for i, vp in enumerate(pending, 1):
        print(f"  {i}. {os.path.basename(vp)}")
    print()

    for idx, video_path in enumerate(pending, 1):
        filename = os.path.basename(video_path)
        print(f"\n[{idx}/{len(pending)}] 처리 중: {filename}")
        print("-" * 40)

        try:
            csv_path = process_video(video_path, OUTPUT_DIR, MODEL_PATH)
            print(f"  ✅ 저장 완료: {csv_path}")
        except Exception as e:
            print(f"  ❌ 오류 발생: {e}")
            continue

    print(f"\n{'=' * 60}")
    print("  🎉 모든 영상 랜드마크 추출이 완료되었습니다!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
