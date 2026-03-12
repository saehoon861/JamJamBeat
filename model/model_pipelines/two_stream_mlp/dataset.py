# two_stream_mlp/dataset.py - xy(42d) / z(21d) 두 스트림 Dataset 빌더 (raw CSV 기준)
from __future__ import annotations

from _shared import TwoStreamDataset, RAW_JOINT_XY_COLS, RAW_JOINT_Z_COLS, SplitData, frame_arrays
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
    Stream-J: x,y 평면 좌표(42d), Stream-B: z 깊이(21d)
    """
    # raw CSV 기준: 공간(xy) 스트림과 깊이(z) 스트림으로 분리한다.
    trj, try_, trm = frame_arrays(split.train_df, RAW_JOINT_XY_COLS)
    trb, _,    _   = frame_arrays(split.train_df, RAW_JOINT_Z_COLS)
    vaj, vay,  vam = frame_arrays(split.val_df,   RAW_JOINT_XY_COLS)
    vab, _,    _   = frame_arrays(split.val_df,   RAW_JOINT_Z_COLS)
    tej, tey,  tem = frame_arrays(split.test_df,  RAW_JOINT_XY_COLS)
    teb, _,    _   = frame_arrays(split.test_df,  RAW_JOINT_Z_COLS)

    model = TwoStreamMLP(
        joint_dim=len(RAW_JOINT_XY_COLS),
        bone_dim=len(RAW_JOINT_Z_COLS),
        num_classes=num_classes,
    )

    return (
        model,
        "two_stream",
        TwoStreamDataset(trj, trb, try_, trm),
        TwoStreamDataset(vaj, vab, vay, vam),
        TwoStreamDataset(tej, teb, tey, tem),
    )
