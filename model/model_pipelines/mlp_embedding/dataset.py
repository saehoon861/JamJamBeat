# mlp_embedding/dataset.py - joint+bone+angle(156d) 전체 피처 Dataset 빌더
from __future__ import annotations

from _shared import FrameDataset, RAW_JOINT_COLS, SplitData, frame_arrays
from .model import MLPEmbedding


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
    mode = "frame"
    입력: joint + bone + angle (full 156d)
    """
    # joint / bone / angle을 concat한 full feature를 embedding MLP에 넣는다.
    cols = RAW_JOINT_COLS

    trX, try_, trm = frame_arrays(split.train_df, cols)
    vaX, vay, vam  = frame_arrays(split.val_df,   cols)
    teX, tey, tem  = frame_arrays(split.test_df,  cols)

    model = MLPEmbedding(input_dim=len(cols), num_classes=num_classes)

    return (
        model,
        "frame",
        FrameDataset(trX, try_, trm),
        FrameDataset(vaX, vay, vam),
        FrameDataset(teX, tey, tem),
    )
