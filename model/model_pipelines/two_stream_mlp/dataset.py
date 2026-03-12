# two_stream_mlp/dataset.py - joint(63d) / bone+angle(93d) 두 스트림 Dataset 빌더
from __future__ import annotations

from _shared import TwoStreamDataset, JOINT_COLS, BONE_COLS, SplitData, frame_arrays
from .model import TwoStreamMLP


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
    mode = "two_stream"
    Stream-J: joint(63d), Stream-B: bone+angle(93d)
    """
    # joint와 bone+angle을 분리해 late fusion으로 결합하기 위해 입력도 두 갈래로 나눈다.
    bone_angle_cols = BONE_COLS + angle_cols

    trj, try_, trm = frame_arrays(split.train_df, JOINT_COLS)
    trb, _,    _   = frame_arrays(split.train_df, bone_angle_cols)
    vaj, vay,  vam = frame_arrays(split.val_df,   JOINT_COLS)
    vab, _,    _   = frame_arrays(split.val_df,   bone_angle_cols)
    tej, tey,  tem = frame_arrays(split.test_df,  JOINT_COLS)
    teb, _,    _   = frame_arrays(split.test_df,  bone_angle_cols)

    model = TwoStreamMLP(
        joint_dim=len(JOINT_COLS),
        bone_dim=len(bone_angle_cols),
        num_classes=num_classes,
    )

    return (
        model,
        "two_stream",
        TwoStreamDataset(trj, trb, try_, trm),
        TwoStreamDataset(vaj, vab, vay, vam),
        TwoStreamDataset(tej, teb, tey, tem),
    )
