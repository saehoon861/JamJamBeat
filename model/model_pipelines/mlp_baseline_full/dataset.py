# mlp_baseline_full/dataset.py - joint+bone+angle(156d) 전체 피처 Dataset 빌더 (bone/angle 가치 검증용)
from __future__ import annotations

from _shared import FrameDataset, JOINT_COLS, BONE_COLS, SplitData, frame_arrays
from .model import MLPBaseline


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
    입력: joint(63d) + bone(84d) + angle(9d) = 156d
    mlp_baseline과 동일 아키텍처로 피처 추가의 순수 효과를 비교한다.
    """
    # 모델 구조는 baseline과 같게 두고, 입력 feature만 156차원으로 확장한다.
    full_cols = JOINT_COLS + BONE_COLS + angle_cols

    trX, try_, trm = frame_arrays(split.train_df, full_cols)
    vaX, vay, vam  = frame_arrays(split.val_df,   full_cols)
    teX, tey, tem  = frame_arrays(split.test_df,  full_cols)

    model = MLPBaseline(input_dim=len(full_cols), num_classes=num_classes)

    return (
        model,
        "frame",
        FrameDataset(trX, try_, trm),
        FrameDataset(vaX, vay, vam),
        FrameDataset(teX, tey, tem),
    )
