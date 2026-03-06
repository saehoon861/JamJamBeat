"""file_io/data_exporter.py - labeled_data CSV 덮어쓰기 저장 모듈

수정된 제스처 라벨을 기존 CSV의 gesture 컬럼에 덮어씌워 저장합니다.
timestamp, frame_idx 등 나머지 컬럼은 원본 그대로 유지합니다.
"""

from pathlib import Path

import pandas as pd


def overwrite_labels(
    stem: str,
    labels: list[int],
    labeled_dir: str,
) -> str:
    """기존 labeled_data CSV의 gesture 컬럼을 수정된 labels 리스트로 교체하여 저장합니다.

    - frame_idx, timestamp 등 나머지 컬럼은 원본 그대로 유지합니다.
    - labels 길이와 CSV 행 수가 다를 경우, 짧은 쪽에 맞춰 안전하게 처리합니다.

    Args:
        stem: 파일명 stem (확장자 없음)
        labels: 수정된 gesture 라벨 리스트
        labeled_dir: labeled_data 폴더 경로

    Returns:
        저장된 CSV 파일의 절대 경로 문자열
    """
    csv_path = Path(labeled_dir) / f"{stem}.csv"

    # 기존 CSV 읽기
    df = pd.read_csv(str(csv_path))

    # gesture 컬럼 교체 (길이 불일치 대비: min으로 안전 처리)
    limit = min(len(df), len(labels))
    df.loc[:limit - 1, "gesture"] = labels[:limit]

    # 덮어쓰기 저장
    df.to_csv(str(csv_path), index=False, encoding="utf-8")
    return str(csv_path.resolve())
