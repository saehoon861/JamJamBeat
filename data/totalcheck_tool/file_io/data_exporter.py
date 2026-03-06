"""file_io/data_exporter.py - Phase 2: labeled_data + total_data 듀얼 저장 모듈

Enter 키를 누르면:
1. labeled_data/{stem}.csv — gesture 컬럼을 최신화 (기존 파일이 없으면 새로 생성)
2. total_data/{stem}.csv  — landmark_df + gesture 병합본을 통째로 저장
"""

from pathlib import Path

import pandas as pd


def format_timestamp(frame_idx: int, fps: float) -> str:
    """프레임 인덱스를 'mm:ss:ms' 형태 문자열로 변환합니다.

    landmark_extractor/main.py와 동일한 로직입니다.
    예: frame_idx=26, fps=30.0 -> '00:00:866'
    """
    total_ms = int((frame_idx / fps) * 1000)
    minutes = total_ms // 60000
    seconds = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{minutes:02d}:{seconds:02d}:{ms:03d}"


def save_all_data(
    stem: str,
    labels: list[int],
    labeled_dir: str,
    total_dir: str,
    landmark_df: pd.DataFrame,
    raw_fps: float,
) -> tuple[str, str]:
    """labeled_data와 total_data 두 곳에 동시 저장합니다.

    Args:
        stem: 파일명 stem (확장자 없음)
        labels: 수정된 gesture 라벨 리스트
        labeled_dir: labeled_data 폴더 경로
        total_dir: total_data 폴더 경로
        landmark_df: 랜드마크 DataFrame (원본의 .copy())
        raw_fps: 동영상의 실제 FPS (백지 라벨링 시 타임스탬프 계산용)

    Returns:
        (labeled_csv_path, total_csv_path) 튜플
    """
    limit = len(labels)

    # --- 디렉토리 사전 생성 ---
    Path(labeled_dir).mkdir(parents=True, exist_ok=True)
    Path(total_dir).mkdir(parents=True, exist_ok=True)

    # ===== 저장 1: labeled_data =====
    labeled_csv_path = Path(labeled_dir) / f"{stem}.csv"

    if labeled_csv_path.exists():
        # 기존 파일이 있으면: gesture 컬럼만 교체
        df_labeled = pd.read_csv(str(labeled_csv_path))
        safe_limit = min(len(df_labeled), limit)
        df_labeled.loc[:safe_limit - 1, "gesture"] = labels[:safe_limit]
    else:
        # 기존 파일이 없으면: 새로 생성 (frame_idx, timestamp, gesture)
        df_labeled = pd.DataFrame({
            "frame_idx": list(range(limit)),
            "timestamp": [format_timestamp(i, raw_fps) for i in range(limit)],
            "gesture": labels[:limit],
        })

    df_labeled.to_csv(str(labeled_csv_path), index=False, encoding="utf-8")

    # ===== 저장 2: total_data (landmark + gesture 병합) =====
    total_csv_path = Path(total_dir) / f"{stem}.csv"

    # landmark_df는 이미 .copy()된 사본이어야 함 (호출부에서 보장)
    df_total = landmark_df.iloc[:limit].copy()
    df_total["gesture"] = labels[:limit]
    df_total.to_csv(str(total_csv_path), index=False, encoding="utf-8")

    return str(labeled_csv_path.resolve()), str(total_csv_path.resolve())
