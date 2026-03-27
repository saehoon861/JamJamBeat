# frame_spatial_transformer/dataset.py - spatial transformer frame dataset builder
from __future__ import annotations

import numpy as np

from _shared import FrameDataset, RAW_JOINT_COLS, SplitData, sample_metadata
from .model import LandmarkSpatialTransformer


def frame_arrays(df, cols):
    X = df[cols].to_numpy(dtype=np.float32).reshape(len(df), 21, 3)
    y = df["gesture"].to_numpy(dtype=np.int64)
    meta = [sample_metadata(df, i) for i in range(len(df))]
    return X, y, meta


def build(
    split: SplitData,
    angle_cols: list[str],
    seq_len: int,
    seq_stride: int,
    image_size: int,
    num_classes: int,
    test_sequence_policy: str = "sliding",
):
    cols = RAW_JOINT_COLS

    trX, try_, trm = frame_arrays(split.train_df, cols)
    vaX, vay, vam = frame_arrays(split.val_df, cols)
    teX, tey, tem = frame_arrays(split.test_df, cols)

    model = LandmarkSpatialTransformer(
        num_landmarks=21,
        coord_dim=3,
        num_classes=num_classes,
        d_model=64,
        num_heads=4,
        num_layers=2,
        ff_dim=128,
        dropout=0.2,
        use_cls_token=False,
    )

    return (
        model,
        "frame",
        FrameDataset(trX, try_, trm),
        FrameDataset(vaX, vay, vam),
        FrameDataset(teX, tey, tem),
    )
