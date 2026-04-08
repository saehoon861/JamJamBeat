# efficientnet_b0/dataset.py - 랜드마크 → 스켈레톤 이미지 렌더링 Dataset 빌더
from __future__ import annotations

from _shared import LandmarkImageDataset, RAW_JOINT_COLS, SplitData, frame_arrays
from .model import EfficientNetLike


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
    mode = "image"
    joint(63d) 좌표를 96x96 스켈레톤 이미지로 렌더링 후 CNN 입력
    """
    # image_size는 joint 좌표를 어떤 해상도의 skeleton map으로 그릴지 결정한다.
    trj, try_, trm = frame_arrays(split.train_df, RAW_JOINT_COLS)
    vaj, vay,  vam = frame_arrays(split.val_df,   RAW_JOINT_COLS)
    tej, tey,  tem = frame_arrays(split.test_df,  RAW_JOINT_COLS)

    model = EfficientNetLike(num_classes=num_classes)

    return (
        model,
        "image",
        LandmarkImageDataset(trj, try_, trm, image_size=image_size, raw_coords=True),
        LandmarkImageDataset(vaj, vay, vam, image_size=image_size, raw_coords=True),
        LandmarkImageDataset(tej, tey, tem, image_size=image_size, raw_coords=True),
    )
