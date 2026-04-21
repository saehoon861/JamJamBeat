"""io/data_exporter.py - CSV 파일 내보내기 모듈"""

from pathlib import Path

import pandas as pd

from core.class_pipeline import LabelConfigManager


def export_labels(
    labels: list[int],
    video_filename: str,
    output_dir: str,
    fps: float = 30.0,
) -> str:
    """프레임 라벨 데이터를 CSV 파일로 내보냅니다.
    
    Args:
        labels: 프레임별 라벨 리스트 (0부터 시작하는 인덱스)
        video_filename: 원본 영상 파일명 (stem 추출용)
        output_dir: CSV 저장 디렉토리 경로
        fps: 동영상 FPS (타임스탬프 계산용)
        
    Returns:
        저장된 CSV 파일의 절대 경로
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 파일명 stem 도출 (.mp4.mp4 이중 확장자 대응)
    stem = Path(video_filename).stem
    while stem.endswith(".mp4"):
        stem = stem[:-4]

    csv_path = output_path / f"{stem}.csv"

    # 타임스탬프 계산 (mm:ss:ms 포맷)
    frame_indices = list(range(len(labels)))
    timestamps = []
    for idx in frame_indices:
        total_ms = int((idx / fps) * 1000)
        minutes = total_ms // 60000
        seconds = (total_ms % 60000) // 1000
        ms = total_ms % 1000
        timestamps.append(f"{minutes:02d}:{seconds:02d}:{ms:03d}")

    df = pd.DataFrame({
        "frame_idx": frame_indices,
        "timestamp": timestamps,
        "gesture": labels,
    })

    df.to_csv(csv_path, index=False, encoding="utf-8")
    return str(csv_path.resolve())
