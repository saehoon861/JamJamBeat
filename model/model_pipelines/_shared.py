# _shared.py - 모든 모델 파이프라인이 공통으로 사용하는 Dataset 클래스 및 유틸
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = ImageDraw = None  # type: ignore

# ──────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]

JOINT_COLS = [f"n{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
BONE_COLS  = [f"b{axis}{i}" for i in range(21) for axis in ("x", "y", "z", "l")]

# Raw landmark 컬럼 (직접 63d joint 기준)
RAW_JOINT_COLS = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]


# ──────────────────────────────────────────────────────
# SplitData
# ──────────────────────────────────────────────────────
@dataclass
class SplitData:
    """공통 split 결과 컨테이너. 각 builder는 이 세 조각만 받아 dataset을 조립한다."""
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


# ──────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────
def detect_angle_cols(df: pd.DataFrame) -> list[str]:
    """전처리 CSV에서 flexion / abduction 각도 컬럼을 자동 탐지한다."""
    cols = [c for c in df.columns if c.startswith("flex_") or c.startswith("abd_")]
    return sorted(cols)


def sample_metadata(df: pd.DataFrame, idx: int) -> dict[str, Any]:
    """예측 결과 CSV에 다시 붙일 최소 메타데이터만 추출한다."""
    row = df.iloc[idx]
    return {
        "source_file": row.get("source_file", ""),
        "frame_idx": int(row.get("frame_idx", idx)),
        "timestamp": row.get("timestamp", ""),
    }


def frame_arrays(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """프레임 기반 모델용 변환. DataFrame 한 행이 그대로 샘플 하나가 된다."""
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["gesture"].to_numpy(dtype=np.int64)
    meta = [sample_metadata(df, i) for i in range(len(df))]
    return X, y, meta


def sequence_arrays(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """시퀀스 기반 모델용 슬라이딩 윈도우 변환.

    각 윈도우 라벨은 마지막 프레임의 gesture를 사용해 sequence labeling 기준을 맞춘다.
    """
    samples: list[np.ndarray] = []
    labels: list[int] = []
    meta: list[dict[str, Any]] = []

    for _, g in df.groupby("__source_group"):
        # 서로 다른 원본 파일을 섞지 않고, 파일 내부 시간축만 따라 윈도우를 만든다.
        g = g.sort_values("frame_idx").reset_index(drop=True)
        feats = g[feature_cols].to_numpy(dtype=np.float32)
        ys = g["gesture"].to_numpy(dtype=np.int64)

        if len(g) < seq_len:
            continue

        for s in range(0, len(g) - seq_len + 1, stride):
            e = s + seq_len
            samples.append(feats[s:e])
            # 윈도우 끝 프레임을 대표 라벨 / 메타 기준점으로 사용한다.
            labels.append(int(ys[e - 1]))
            meta.append(sample_metadata(g, e - 1))

    if not samples:
        raise ValueError("No sequence samples built. Check seq_len/stride and split sizes.")

    return np.stack(samples, axis=0), np.array(labels, dtype=np.int64), meta


def repeated_sequence_arrays(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """독립 정지사진을 sequence 모델에 넣기 위해 한 프레임을 seq_len만큼 반복한다."""
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["gesture"].to_numpy(dtype=np.int64)
    meta = [sample_metadata(df, i) for i in range(len(df))]

    if len(X) == 0:
        raise ValueError("No repeated sequence samples built. Check test split size.")

    seq = np.repeat(X[:, None, :], max(int(seq_len), 1), axis=1).astype(np.float32)
    return seq, y, meta


# ──────────────────────────────────────────────────────
# Dataset classes
# ──────────────────────────────────────────────────────
class FrameDataset(Dataset):
    """단일 프레임 입력 Dataset.

    반환 shape:
    - X: (D,)
    - y: scalar
    - idx: 예측 후 meta 역참조용 인덱스
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, meta: list[dict[str, Any]]):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.meta = meta

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
            torch.tensor(idx, dtype=torch.long),
        )


class SequenceDataset(Dataset):
    """슬라이딩 윈도우 시퀀스 Dataset.

    반환 shape:
    - X_seq: (T, D)
    - y: 마지막 프레임 기준 라벨
    - idx: 예측 후 meta 역참조용 인덱스
    """

    def __init__(self, X_seq: np.ndarray, y: np.ndarray, meta: list[dict[str, Any]]):
        self.X_seq = X_seq.astype(np.float32)
        self.y     = y.astype(np.int64)
        self.meta  = meta

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.X_seq[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
            torch.tensor(idx, dtype=torch.long),
        )


class LandmarkImageDataset(Dataset):
    """정규화 joint 좌표를 1채널 스켈레톤 이미지로 렌더링하는 Dataset."""

    def __init__(
        self,
        joint: np.ndarray,
        y: np.ndarray,
        meta: list[dict[str, Any]],
        image_size: int = 96,
        raw_coords: bool = False,
    ):
        if Image is None:
            raise ImportError("Pillow(PIL)가 필요합니다: uv pip install pillow")
        self.joint      = joint.astype(np.float32)
        self.y          = y.astype(np.int64)
        self.meta       = meta
        self.image_size = image_size
        self.raw_coords = raw_coords

    @staticmethod
    def _norm_to_px(v: np.ndarray, size: int) -> np.ndarray:
        # 전처리된 정규화 좌표 [-1.2, 1.2] → 픽셀 좌표
        clipped = np.clip(v, -1.2, 1.2)
        return ((clipped + 1.2) / 2.4) * (size - 1)

    @staticmethod
    def _raw_to_px(v: np.ndarray, size: int) -> np.ndarray:
        # 원시 MediaPipe 좌표 [0, 1] → 픽셀 좌표
        clipped = np.clip(v, 0.0, 1.0)
        return clipped * (size - 1)

    def _render(self, joint_vec: np.ndarray) -> np.ndarray:
        # 21개 랜드마크를 연결선 + 점으로 그려 CNN이 읽을 입력 텐서를 만든다.
        pts = joint_vec.reshape(21, 3)
        to_px = self._raw_to_px if self.raw_coords else self._norm_to_px
        x = to_px(pts[:, 0], self.image_size)
        y = (self.image_size - 1) - to_px(pts[:, 1], self.image_size)

        img  = Image.new("L", (self.image_size, self.image_size), color=0)
        draw = ImageDraw.Draw(img)

        for u, v in HAND_CONNECTIONS:
            draw.line(
                [(float(x[u]), float(y[u])), (float(x[v]), float(y[v]))],
                fill=160, width=2,
            )
        for i in range(21):
            r = 2
            draw.ellipse(
                [(float(x[i] - r), float(y[i] - r)), (float(x[i] + r), float(y[i] + r))],
                fill=255,
            )

        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr[None, :, :]  # (1, H, W)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        image = self._render(self.joint[idx])
        return (
            torch.from_numpy(image),
            torch.tensor(self.y[idx], dtype=torch.long),
            torch.tensor(idx, dtype=torch.long),
        )
