# mlp_sequence_joint/dataset.py - joint(63d) sliding-window sequence dataset builder for a simple sequence MLP
from __future__ import annotations

from _shared import RAW_JOINT_COLS, SequenceDataset, SplitData, repeated_sequence_arrays, sequence_arrays
from .model import SequenceJointMLP


def build(
    split: SplitData,
    angle_cols: list[str],
    seq_len: int,
    seq_stride: int,
    image_size: int,
    num_classes: int,
    test_sequence_policy: str = "sliding",
):
    """
    Returns: (model, mode, train_ds, val_ds, test_ds)
    mode = "sequence"
    입력: (B, T=seq_len, D=63) sliding window on joint-only features
    """
    # sequence_arrays는 각 윈도우의 마지막 프레임 gesture를 라벨로 사용한다.
    trX, try_, trm = sequence_arrays(split.train_df, RAW_JOINT_COLS, seq_len=seq_len, stride=seq_stride)
    vaX, vay, vam = sequence_arrays(split.val_df, RAW_JOINT_COLS, seq_len=seq_len, stride=seq_stride)
    if test_sequence_policy == "independent_repeat":
        teX, tey, tem = repeated_sequence_arrays(split.test_df, RAW_JOINT_COLS, seq_len=seq_len)
    elif test_sequence_policy == "sliding":
        teX, tey, tem = sequence_arrays(split.test_df, RAW_JOINT_COLS, seq_len=seq_len, stride=seq_stride)
    else:
        raise ValueError(f"Unsupported test_sequence_policy: {test_sequence_policy}")

    model = SequenceJointMLP(seq_len=seq_len, input_dim=len(RAW_JOINT_COLS), num_classes=num_classes)

    return (
        model,
        "sequence",
        SequenceDataset(trX, try_, trm),
        SequenceDataset(vaX, vay, vam),
        SequenceDataset(teX, tey, tem),
    )
