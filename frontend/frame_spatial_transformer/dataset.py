from __future__ import annotations

from _shared import JOINT_COLS, SplitData
from .model import LandmarkSpatialTransformer
from torch.utils.data import Dataset
import torch
import numpy as np


class LandmarkFrameDataset(Dataset):
    def __init__(self, X, y, meta):
        self.X = X
        self.y = y
        self.meta = meta

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)   # (21, 3)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y, self.meta[idx]


def frame_arrays(df, cols):
    X = df[cols].to_numpy(dtype=np.float32)         # (N, 63)
    X = X.reshape(len(X), 21, 3)                    # (N, 21, 3)
    y = df["gesture"].to_numpy(dtype=np.int64)        # label 컬럼명은 실제에 맞게
    meta = df[["source_file", "frame_idx"]].to_dict("records")
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
    cols = JOINT_COLS  # [수정] Raw(x,y,z) 대신 Normalized(nx,ny,nz) 사용

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
        LandmarkFrameDataset(trX, try_, trm),
        LandmarkFrameDataset(vaX, vay, vam),
        LandmarkFrameDataset(teX, tey, tem),
    )