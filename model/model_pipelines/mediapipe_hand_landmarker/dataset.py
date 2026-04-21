# mediapipe_hand_landmarker/dataset.py - MediaPipe 프론트엔드 + joint(63d) MLP Dataset 빌더
from __future__ import annotations

from _shared import FrameDataset, JOINT_COLS, SplitData, frame_arrays
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
    MediaPipe Hand Landmarker는 랜드마크 추출 프론트엔드.
    분류기는 joint(63d) MLP baseline과 동일 구조.
    """
    # 이름은 MediaPipe 기반 파이프라인이지만, 학습 입력 자체는 joint-only frame MLP와 같다.
    trX, try_, trm = frame_arrays(split.train_df, JOINT_COLS)
    vaX, vay,  vam = frame_arrays(split.val_df,   JOINT_COLS)
    teX, tey,  tem = frame_arrays(split.test_df,  JOINT_COLS)

    model = MLPBaseline(input_dim=len(JOINT_COLS), num_classes=num_classes)

    return (
        model,
        "frame",
        FrameDataset(trX, try_, trm),
        FrameDataset(vaX, vay, vam),
        FrameDataset(teX, tey, tem),
    )
