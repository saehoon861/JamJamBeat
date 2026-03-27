# lappe_dist_mixer/dataset.py - Raw 63d hand-landmark frame dataset builder for LapPE DistMixer
from __future__ import annotations

import numpy as np

from _shared import FrameDataset, RAW_JOINT_COLS, SplitData, frame_arrays
from .model import LapPEDistMixer


def reshape_raw_landmarks(x: np.ndarray) -> np.ndarray:
    expected_dim = len(RAW_JOINT_COLS)
    if x.ndim != 2 or x.shape[1] != expected_dim:
        raise ValueError(f"Expected raw landmark array shape (N, {expected_dim}), got {x.shape}")
    return x.reshape(len(x), 21, 3).astype(np.float32, copy=False)


def build(
    split: SplitData,
    angle_cols: list[str],
    seq_len: int,
    seq_stride: int,
    image_size: int,
    num_classes: int,
    test_sequence_policy: str = "sliding",
    model_overrides: dict | None = None,
):
    """Build the LapPE DistMixer frame pipeline from raw 63D landmarks only."""

    del angle_cols, seq_len, seq_stride, image_size, test_sequence_policy

    trX, try_, trm = frame_arrays(split.train_df, RAW_JOINT_COLS)
    vaX, vay, vam = frame_arrays(split.val_df, RAW_JOINT_COLS)
    teX, tey, tem = frame_arrays(split.test_df, RAW_JOINT_COLS)

    trX = reshape_raw_landmarks(trX)
    vaX = reshape_raw_landmarks(vaX)
    teX = reshape_raw_landmarks(teX)

    model_kwargs = {
        "num_landmarks": 21,
        "coord_dim": 3,
        "num_classes": num_classes,
        "lappe_dim": 8,
        "hidden_dim": 64,
        "channel_mlp_hidden": 128,
        "num_layers": 3,
        "lappe_sign_flip": True,
    }
    if model_overrides:
        model_kwargs.update(model_overrides)

    model = LapPEDistMixer(**model_kwargs)

    return (
        model,
        "frame",
        FrameDataset(trX, try_, trm),
        FrameDataset(vaX, vay, vam),
        FrameDataset(teX, tey, tem),
    )
