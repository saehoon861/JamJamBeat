"""file_io/data_loader.py - 3종 파일(mp4, labeled_data, landmark_data) 로드 및 유효성 검증 모듈

패키지명: file_io/  (파이썬 내장 io 모듈과의 이름 충돌을 방지하기 위해 file_io 사용)
"""

from pathlib import Path
from dataclasses import dataclass

import cv2
import pandas as pd


@dataclass
class SessionData:
    """3종 데이터를 묶는 컨테이너."""
    cap: cv2.VideoCapture
    labeled_df: pd.DataFrame | None  # None = 백지 라벨링 (labeled_data 미존재)
    landmark_df: pd.DataFrame
    stem: str
    video_path: str


def get_checkable_stems(labeled_dir: str, landmark_dir: str, raw_dir: str) -> list[str]:
    """검수 가능한 영상 stem 목록을 반환합니다.

    Phase 2 변경: raw_data ∩ landmark_data 교집합만으로 stem을 추출합니다.
    labeled_data가 없어도 백지 라벨링이 가능하므로 교집합 대상에서 제외합니다.
    OS별 정렬 불일치 방지를 위해 sorted()를 사용합니다.
    """
    landmark_path = Path(landmark_dir)
    raw_path = Path(raw_dir)

    landmark_stems = {f.stem for f in sorted(landmark_path.glob("*.csv"))}
    raw_stems = {f.stem for f in sorted(raw_path.glob("*.mp4"))}

    # .mp4.mp4 이중 확장자 대응: raw_data stem 정규화
    raw_stems_normalized = set()
    for s in raw_stems:
        norm = s
        while norm.endswith(".mp4"):
            norm = norm[:-4]
        raw_stems_normalized.add(norm)

    # raw ∩ landmark 교집합만 반환 (labeled 유무 상관 없음)
    checkable = sorted(landmark_stems & raw_stems_normalized)
    return checkable


def load_session_data(
    stem: str,
    raw_dir: str,
    labeled_dir: str,
    landmark_dir: str,
) -> SessionData | None:
    """영상 1개의 세션 데이터를 로드합니다.

    3개 파일 중 하나라도 누락/로드 실패 시 None을 반환합니다.

    Args:
        stem: 파일명 stem (확장자 없는 이름)
        raw_dir: raw_data 폴더 경로
        labeled_dir: labeled_data 폴더 경로
        landmark_dir: landmark_data 폴더 경로
    """
    raw_path = Path(raw_dir)
    labeled_path = Path(labeled_dir)
    landmark_path = Path(landmark_dir)

    # 경로 확인
    video_file = raw_path / f"{stem}.mp4"
    labeled_file = labeled_path / f"{stem}.csv"
    landmark_file = landmark_path / f"{stem}.csv"

    if not video_file.exists():
        print(f"[WARN] {stem} 파일 로드 실패 (raw_data mp4 누락) -> 스킵합니다.")
        return None
    if not landmark_file.exists():
        print(f"[WARN] {stem} 파일 로드 실패 (landmark_data 누락) -> 스킵합니다.")
        return None

    # 로드
    try:
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f"[WARN] {stem} 동영상을 열 수 없습니다 -> 스킵합니다.")
            return None

        # labeled_data는 없을 수 있음 (백지 라벨링 지원)
        labeled_df = None
        if labeled_file.exists():
            labeled_df = pd.read_csv(str(labeled_file))
        else:
            print(f"[INFO] {stem} labeled_data 없음 -> 백지 라벨링 모드로 진입합니다.")

        landmark_df = pd.read_csv(str(landmark_file))

    except Exception as e:
        print(f"[WARN] {stem} 로드 중 예외 발생: {e} -> 스킵합니다.")
        return None

    return SessionData(
        cap=cap,
        labeled_df=labeled_df,
        landmark_df=landmark_df,
        stem=stem,
        video_path=str(video_file),
    )
