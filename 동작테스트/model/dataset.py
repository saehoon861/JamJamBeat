# dataset.py - grab 제스처 학습용 sequence dataset 빌더
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# MediaPipe 21 관절 raw 좌표 컬럼 (x0,y0,z0 ... x20,y20,z20)
RAW_JOINT_COLS: list[str] = [f"{ax}{i}" for i in range(21) for ax in ("x", "y", "z")]  # 63d


def add_delta_features(x_seq: np.ndarray) -> np.ndarray:
    """(N, T, 63) 배열에 1차 차분을 이어붙여 (N, T, 126)을 반환한다.

    첫 번째 프레임의 delta는 0으로 채운다.
    """
    delta = np.zeros_like(x_seq, dtype=np.float32)
    delta[:, 1:, :] = x_seq[:, 1:, :] - x_seq[:, :-1, :]
    return np.concatenate([x_seq.astype(np.float32), delta], axis=2)


class SequenceDataset(Dataset):
    """(N, T, D) 시퀀스 배열과 레이블을 묶는 Dataset.

    DataLoader에서 (x, y, idx) 튜플을 반환한다.
    train.py의 weighted sampler가 .y를 직접 참조한다.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], idx


def _value_to_float(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _load_rows(csv_path: str | Path) -> list[dict[str, str]]:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _build_single_dataset(
    csv_path: str | Path,
    seq_len: int,
    seq_stride: int,
) -> SequenceDataset:
    rows = _load_rows(csv_path)
    if not rows:
        raise ValueError(f"{csv_path}: no rows found")

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        source_file = row.get("source_file") or "__single_source__"
        grouped.setdefault(source_file, []).append(row)

    windows: list[np.ndarray] = []
    labels: list[int] = []

    for source_file, source_rows in grouped.items():
        if len(source_rows) < seq_len:
            continue

        joint_rows = np.asarray(
            [
                [_value_to_float(row.get(col)) for col in RAW_JOINT_COLS]
                for row in source_rows
            ],
            dtype=np.float32,
        )
        gesture_rows = np.asarray(
            [int(float(row["gesture"])) if row.get("gesture", "") != "" else 0 for row in source_rows],
            dtype=np.int64,
        )

        for start in range(0, len(source_rows) - seq_len + 1, seq_stride):
            end = start + seq_len
            windows.append(joint_rows[start:end])
            labels.append(int(gesture_rows[end - 1]))

    if not windows:
        raise ValueError(
            f"{csv_path}: no sequence windows generated "
            f"(seq_len={seq_len}, seq_stride={seq_stride})"
        )

    x63 = np.stack(windows, axis=0).astype(np.float32)
    x126 = add_delta_features(x63)
    y = np.asarray(labels, dtype=np.int64)
    return SequenceDataset(x126, y)


def build(
    train_csv: str,
    val_csv: str | None,
    test_csv: str | None,
    seq_len: int = 8,
    seq_stride: int = 2,
) -> tuple[SequenceDataset, SequenceDataset | None, SequenceDataset | None]:
    """CSV를 읽어 sliding-window sequence dataset을 생성한다."""
    train_ds = _build_single_dataset(train_csv, seq_len=seq_len, seq_stride=seq_stride)
    val_ds = (
        _build_single_dataset(val_csv, seq_len=seq_len, seq_stride=seq_stride)
        if val_csv is not None
        else None
    )
    test_ds = (
        _build_single_dataset(test_csv, seq_len=seq_len, seq_stride=seq_stride)
        if test_csv is not None
        else None
    )
    return train_ds, val_ds, test_ds
