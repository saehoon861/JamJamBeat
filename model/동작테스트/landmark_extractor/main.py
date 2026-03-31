"""main.py - 동작테스트 전용 MediaPipe Hand Landmarker 기반 랜드마크 추출 도구

실행 방법:
    $ cd /home/user/projects/JamJamBeat/model/동작테스트/landmark_extractor
    $ uv run python main.py
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import pandas as pd
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = str(WORKSPACE_ROOT / "hand_landmarker.task")
RAW_DATA_DIR = str(WORKSPACE_ROOT / "raw_data")
OUTPUT_DIR = str(WORKSPACE_ROOT / "landmark_data")
RUNNING_MODE = VisionRunningMode.VIDEO

LANDMARK_COLUMNS: list[str] = []
for index in range(21):
    LANDMARK_COLUMNS.extend([f"x{index}", f"y{index}", f"z{index}"])


def get_pending_videos(raw_dir: str, output_dir: str) -> list[str]:
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    done_stems = {file.stem for file in sorted(output_path.glob("*.csv"))}
    pending: list[str] = []
    for mp4 in sorted(raw_path.glob("*.mp4")):
        if mp4.stem not in done_stems:
            pending.append(str(mp4.resolve()))
    return pending


def format_timestamp(frame_idx: int, fps: float) -> str:
    total_ms = int((frame_idx / fps) * 1000)
    minutes = total_ms // 60000
    seconds = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{minutes:02d}:{seconds:02d}:{ms:03d}"


def extract_landmarks(result) -> list[float | None]:
    if len(result.hand_landmarks) > 0:
        landmarks = result.hand_landmarks[0]
        values: list[float | None] = []
        for landmark in landmarks:
            values.extend([landmark.x, landmark.y, landmark.z])
        return values
    return [None] * 63


def process_video(video_path: str, output_dir: str, model_path: str) -> str:
    filename = os.path.basename(video_path)
    stem = Path(filename).stem

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"동영상을 열 수 없습니다: {video_path}")

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RUNNING_MODE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    rows: list[list[object]] = []

    with HandLandmarker.create_from_options(options) as landmarker:
        for frame_idx in range(total_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(frame_idx / actual_fps * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            landmark_values = extract_landmarks(result)

            rows.append([frame_idx, format_timestamp(frame_idx, actual_fps), None, *landmark_values])

            if (frame_idx + 1) % 30 == 0 or frame_idx == total_frames - 1:
                pct = (frame_idx + 1) / total_frames * 100
                print(f"  Frame {frame_idx + 1}/{total_frames} ({pct:.1f}%)")

    cap.release()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / f"{stem}.csv"
    columns = ["frame_idx", "timestamp", "gesture", *LANDMARK_COLUMNS]
    pd.DataFrame(rows, columns=columns).to_csv(csv_path, index=False, encoding="utf-8")
    return str(csv_path.resolve())


def main() -> None:
    print("=" * 60)
    print("  🖐️ 동작테스트 MediaPipe Hand Landmark Extractor")
    print("=" * 60)

    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        return

    pending = get_pending_videos(RAW_DATA_DIR, OUTPUT_DIR)
    if not pending:
        print("[INFO] 모든 영상이 이미 처리 완료되었습니다.")
        print(f"[INFO] raw_data: {RAW_DATA_DIR}")
        print(f"[INFO] landmark_data: {OUTPUT_DIR}")
        return

    print(f"[INFO] 미처리 영상 {len(pending)}개 발견:")
    for idx, video_path in enumerate(pending, 1):
        print(f"  {idx}. {os.path.basename(video_path)}")
    print()

    for idx, video_path in enumerate(pending, 1):
        filename = os.path.basename(video_path)
        print(f"\n[{idx}/{len(pending)}] 처리 중: {filename}")
        print("-" * 40)
        try:
            csv_path = process_video(video_path, OUTPUT_DIR, MODEL_PATH)
            print(f"  ✅ 저장 완료: {csv_path}")
        except Exception as exc:
            print(f"  ❌ 오류 발생: {exc}")
            if "libGLESv2.so.2" in str(exc):
                print("  [HINT] WSL/Ubuntu에서는 MediaPipe 네이티브 의존성이 필요합니다.")
                print("  [HINT] sudo apt-get update && sudo apt-get install -y libgles2 libegl1 libglib2.0-0")

    print("\n[INFO] 모든 영상 랜드마크 추출이 완료되었습니다.")


if __name__ == "__main__":
    main()
