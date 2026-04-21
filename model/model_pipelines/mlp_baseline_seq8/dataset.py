# mlp_baseline_seq8/dataset.py - joint(63d) sliding-window sequence Dataset 빌더 (window_size=8 고정)
from __future__ import annotations

from _shared import RAW_JOINT_COLS, SequenceDataset, SplitData, repeated_sequence_arrays, sequence_arrays
from .model import MLPBaselineSeq

SEQ_LEN = 8  # window size 고정


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
    입력: (B, T=8, D=63) → flatten → (B, 504)
    mlp_baseline 동일 아키텍처(128→64)로 temporal context 효과만 비교한다.
    seq_len 인자는 무시하고 SEQ_LEN=8 고정.
    """
    # 이 실험은 비교 공정성을 위해 window size를 항상 8로 고정한다.
    trX, try_, trm = sequence_arrays(split.train_df, RAW_JOINT_COLS, seq_len=SEQ_LEN, stride=seq_stride)
    vaX, vay, vam  = sequence_arrays(split.val_df,   RAW_JOINT_COLS, seq_len=SEQ_LEN, stride=seq_stride)
    if test_sequence_policy == "independent_repeat":
        teX, tey, tem = repeated_sequence_arrays(split.test_df, RAW_JOINT_COLS, seq_len=SEQ_LEN)
    elif test_sequence_policy == "sliding":
        teX, tey, tem = sequence_arrays(split.test_df, RAW_JOINT_COLS, seq_len=SEQ_LEN, stride=seq_stride)
    else:
        raise ValueError(f"Unsupported test_sequence_policy: {test_sequence_policy}")

    model = MLPBaselineSeq(seq_len=SEQ_LEN, input_dim=len(RAW_JOINT_COLS), num_classes=num_classes)

    return (
        model,
        "sequence",
        SequenceDataset(trX, try_, trm),
        SequenceDataset(vaX, vay, vam),
        SequenceDataset(teX, tey, tem),
    )
