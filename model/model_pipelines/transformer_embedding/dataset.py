# transformer_embedding/dataset.py - 슬라이딩 윈도우 시퀀스(16 x 156d) Dataset 빌더
from __future__ import annotations

from _shared import SequenceDataset, JOINT_COLS, BONE_COLS, SplitData, sequence_arrays
from .model import TemporalTransformer


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
    입력: (B, T=seq_len, D=156) sliding window
    """
    # Transformer는 full feature 시퀀스 전체를 self-attention으로 본다.
    full_cols = JOINT_COLS + BONE_COLS + angle_cols

    trX, try_, trm = sequence_arrays(split.train_df, full_cols, seq_len=seq_len, stride=seq_stride)
    vaX, vay,  vam = sequence_arrays(split.val_df,   full_cols, seq_len=seq_len, stride=seq_stride)
    teX, tey,  tem = sequence_arrays(split.test_df,  full_cols, seq_len=seq_len, stride=seq_stride)

    model = TemporalTransformer(
        seq_len=seq_len,
        input_dim=len(full_cols),
        num_classes=num_classes,
    )

    return (
        model,
        "sequence",
        SequenceDataset(trX, try_, trm),
        SequenceDataset(vaX, vay, vam),
        SequenceDataset(teX, tey, tem),
    )
