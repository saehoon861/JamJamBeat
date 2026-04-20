"""io/file_scanner.py - .mp4 파일 스캔 및 기 처리 csv 대조 필터링 모듈"""

import os
from pathlib import Path


def get_pending_videos(raw_data_dir: str, labeled_data_dir: str) -> list[str]:
    """미작업 동영상 목록을 반환합니다.
    
    raw_data_dir 내의 .mp4 파일들과 labeled_data_dir 내의 .csv 파일들을
    파일명 stem 단위로 1:1 대조하여, 아직 csv가 없는 mp4 파일만 리턴합니다.
    
    Args:
        raw_data_dir: .mp4 원본 동영상 디렉토리 경로
        labeled_data_dir: .csv 결과물 디렉토리 경로
        
    Returns:
        아직 라벨링되지 않은 동영상의 절대 경로 리스트 (정렬됨)
    """
    raw_path = Path(raw_data_dir)
    labeled_path = Path(labeled_data_dir)

    # labeled_data 디렉토리가 없으면 생성
    labeled_path.mkdir(parents=True, exist_ok=True)

    # .mp4 파일 목록 수집
    mp4_files = sorted(raw_path.glob("*.mp4"))

    # 기 처리된 csv 파일명(stem) 집합
    done_stems = set()
    for csv_file in labeled_path.glob("*.csv"):
        done_stems.add(csv_file.stem)

    # 미작업 목록 필터링: stem 기준 대조
    pending = []
    for mp4 in mp4_files:
        # .mp4.mp4 같은 이중 확장자도 stem 처리
        stem = mp4.stem
        while stem.endswith(".mp4"):
            stem = stem[:-4]

        if stem not in done_stems:
            pending.append(str(mp4.resolve()))

    return pending
