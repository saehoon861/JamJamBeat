# mlp_original/dataset.py - src/models/baseline GestureMLP raw 63d Dataset 빌더
from __future__ import annotations

from _shared import FrameDataset, RAW_JOINT_COLS, SplitData, frame_arrays
from .model import GestureMLP


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
    입력: raw joint 63d (x0~x20, y0~y20, z0~z20)
    구조: src/models/baseline/mlp_classifier.py GestureMLP 완전 동일
    """
    trX, try_, trm = frame_arrays(split.train_df, RAW_JOINT_COLS)
    vaX, vay,  vam = frame_arrays(split.val_df,   RAW_JOINT_COLS)
    teX, tey,  tem = frame_arrays(split.test_df,  RAW_JOINT_COLS)

    model = GestureMLP(input_dim=len(RAW_JOINT_COLS), num_classes=num_classes)

    return (
        model,
        "frame",
        FrameDataset(trX, try_, trm),
        FrameDataset(vaX, vay, vam),
        FrameDataset(teX, tey, tem),
    )
