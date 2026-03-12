# mlp_sequence_delta/dataset.py - joint(63d) + delta(63d) sliding-window sequence dataset builder
from __future__ import annotations

import numpy as np

from _shared import RAW_JOINT_COLS, SequenceDataset, SplitData, sequence_arrays
from .model import SequenceDeltaMLP


def add_delta_features(x_seq: np.ndarray) -> np.ndarray:
    """
    x_seq: (N, T, 63)
    returns: (N, T, 126) = [joint, delta]
    """
    # delta는 프레임 간 1차 차분으로, 정적인 모양 외에 motion signal을 추가한다.
    delta = np.zeros_like(x_seq, dtype=np.float32)
    delta[:, 1:, :] = x_seq[:, 1:, :] - x_seq[:, :-1, :]
    return np.concatenate([x_seq.astype(np.float32), delta], axis=2)


def build(
    split: SplitData,
    angle_cols: list[str],
    seq_len: int,
    seq_stride: int,
    image_size: int,
    num_classes: int,
):
    """
    Returns: (model, mode, train_ds, val_ds, test_ds)
    mode = "sequence"
    입력: (B, T=seq_len, D=126) where D = joint(63) + delta(63)
    """
    # 라벨은 마지막 프레임 기준이고, 입력 feature만 joint+delta로 확장된다.
    trX, try_, trm = sequence_arrays(split.train_df, RAW_JOINT_COLS, seq_len=seq_len, stride=seq_stride)
    vaX, vay, vam = sequence_arrays(split.val_df, RAW_JOINT_COLS, seq_len=seq_len, stride=seq_stride)
    teX, tey, tem = sequence_arrays(split.test_df, RAW_JOINT_COLS, seq_len=seq_len, stride=seq_stride)

    trX = add_delta_features(trX)
    vaX = add_delta_features(vaX)
    teX = add_delta_features(teX)

    model = SequenceDeltaMLP(seq_len=seq_len, input_dim=trX.shape[2], num_classes=num_classes)

    return (
        model,
        "sequence",
        SequenceDataset(trX, try_, trm),
        SequenceDataset(vaX, vay, vam),
        SequenceDataset(teX, tey, tem),
    )
