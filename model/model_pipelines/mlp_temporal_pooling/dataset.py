# mlp_temporal_pooling/dataset.py - joint(63d) sliding-window dataset builder for temporal pooling MLP
from __future__ import annotations

from _shared import JOINT_COLS, SequenceDataset, SplitData, sequence_arrays
from .model import TemporalPoolingMLP


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
    입력: (B, T=seq_len, D=63) sliding window on joint-only features
    """
    # joint-only 시퀀스를 유지하되, 모델 쪽에서 temporal pooling으로 요약한다.
    trX, try_, trm = sequence_arrays(split.train_df, JOINT_COLS, seq_len=seq_len, stride=seq_stride)
    vaX, vay, vam = sequence_arrays(split.val_df, JOINT_COLS, seq_len=seq_len, stride=seq_stride)
    teX, tey, tem = sequence_arrays(split.test_df, JOINT_COLS, seq_len=seq_len, stride=seq_stride)

    model = TemporalPoolingMLP(input_dim=len(JOINT_COLS), num_classes=num_classes)

    return (
        model,
        "sequence",
        SequenceDataset(trX, try_, trm),
        SequenceDataset(vaX, vay, vam),
        SequenceDataset(teX, tey, tem),
    )
