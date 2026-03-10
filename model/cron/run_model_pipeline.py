#!/usr/bin/env python3
"""
Run one JamJamBeat model-comparison pipeline (v2) end-to-end.

- Input: preprocessed CSV(s) with columns:
    source_file, frame_idx, timestamp, gesture,
    nx*, ny*, nz*, bx*, by*, bz*, bl*, flex_*, abd_*
- Output: model artifact, prediction CSV, and evaluation metrics/plots.

This script is intended to be called by cron (one model per cron job).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Optional heavy dependency checks (PyTorch)
# -----------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except Exception as e:  # pragma: no cover - runtime guard
    raise SystemExit(
        "[ERROR] PyTorch is required for run_model_pipeline.py\n"
        "Please install torch in your virtual environment first.\n"
        f"Original error: {e}"
    )

try:
    from PIL import Image, ImageDraw
except Exception as e:  # pragma: no cover - runtime guard
    raise SystemExit(
        "[ERROR] Pillow (PIL) is required for image backbones.\n"
        "Please install pillow in your virtual environment first.\n"
        f"Original error: {e}"
    )

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_MODULE_DIR = PROJECT_ROOT / "model" / "model_evaluation"
if str(EVAL_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_MODULE_DIR))

from evaluation_runtime import DEFAULT_CLASS_NAMES, EvaluationConfig, evaluate_predictions


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
HAND_CONNECTIONS = [
    (0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]

JOINT_COLS = [f"n{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
BONE_COLS = [f"b{axis}{i}" for i in range(21) for axis in ("x", "y", "z", "l")]

DEFAULT_INPUTS = [
    "model/data_fusion/man1_right_for_poc_output.csv",
    "model/data_fusion/man2_right_for_poc_output.csv",
    "model/data_fusion/man3_right_for_poc_output.csv",
    "model/data_fusion/woman1_right_for_poc_output.csv",
]

MODEL_CHOICES = [
    "mlp_baseline",
    "mlp_embedding",
    "two_stream_mlp",
    "cnn1d_tcn",
    "transformer_embedding",
    "mediapipe_hand_landmarker",
    "mobilenetv3_small",
    "shufflenetv2_x0_5",
    "efficientnet_b0",
]


@dataclass
class SplitData:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_paths(csv_paths: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for path in csv_paths:
        p = Path(path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        resolved.append(p.resolve())
    return resolved


def detect_angle_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("flex_") or c.startswith("abd_")]
    return sorted(cols)


def load_preprocessed_data(csv_paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input CSV not found: {path}")

        df = pd.read_csv(path)
        df["__source_group"] = path.stem

        if "source_file" not in df.columns:
            df["source_file"] = path.stem

        # ensure key columns exist
        required = {"gesture", "frame_idx", "timestamp"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")

        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    # basic cleanup
    merged = merged.dropna(subset=["gesture"]).copy()
    merged["gesture"] = merged["gesture"].astype(int)

    # ensure required feature columns for v2 pipeline
    missing_joint = [c for c in JOINT_COLS if c not in merged.columns]
    missing_bone = [c for c in BONE_COLS if c not in merged.columns]
    if missing_joint or missing_bone:
        raise ValueError(
            "Input must be preprocessed *_output.csv with joint/bone features. "
            f"Missing joint: {missing_joint[:5]}... bone: {missing_bone[:5]}..."
        )

    angle_cols = detect_angle_cols(merged)
    if len(angle_cols) < 4:
        raise ValueError("Could not detect angle columns (flex_*/abd_*).")

    merged = merged.dropna(subset=JOINT_COLS + BONE_COLS + angle_cols).reset_index(drop=True)
    return merged


def split_by_group(
    df: pd.DataFrame,
    group_col: str = "__source_group",
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> SplitData:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    groups = list(df[group_col].dropna().unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    if len(groups) >= 3:
        n_groups = len(groups)
        n_test = max(1, int(round(n_groups * test_ratio)))
        n_val = max(1, int(round(n_groups * val_ratio)))

        if n_test + n_val >= n_groups:
            n_test = 1
            n_val = 1

        test_groups = set(groups[:n_test])
        val_groups = set(groups[n_test:n_test + n_val])
        train_groups = set(groups[n_test + n_val:])

        if not train_groups:
            # guarantee train split non-empty
            spill = list(val_groups)
            train_groups.add(spill[0])
            val_groups.remove(spill[0])

        train_df = df[df[group_col].isin(train_groups)].copy()
        val_df = df[df[group_col].isin(val_groups)].copy()
        test_df = df[df[group_col].isin(test_groups)].copy()
    else:
        # fallback: random row split
        idx = np.arange(len(df))
        rng.shuffle(idx)

        n_test = max(1, int(len(idx) * test_ratio))
        n_val = max(1, int(len(idx) * val_ratio))

        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]

        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        test_df = df.iloc[test_idx].copy()

    return SplitData(
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
    )


def sample_metadata(df: pd.DataFrame, idx: int) -> dict[str, Any]:
    row = df.iloc[idx]
    return {
        "source_file": row.get("source_file", ""),
        "frame_idx": int(row.get("frame_idx", idx)),
        "timestamp": row.get("timestamp", ""),
    }


def create_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    classes, counts = np.unique(labels, return_counts=True)
    class_w = {int(c): 1.0 / max(int(cnt), 1) for c, cnt in zip(classes, counts)}
    sample_weights = np.array([class_w[int(y)] for y in labels], dtype=np.float64)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


# -----------------------------------------------------------------------------
# Dataset definitions
# -----------------------------------------------------------------------------
class FrameDataset(Dataset):
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


class TwoStreamDataset(Dataset):
    def __init__(self, X_joint: np.ndarray, X_bone: np.ndarray, y: np.ndarray, meta: list[dict[str, Any]]):
        self.X_joint = X_joint.astype(np.float32)
        self.X_bone = X_bone.astype(np.float32)
        self.y = y.astype(np.int64)
        self.meta = meta

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.X_joint[idx]),
            torch.from_numpy(self.X_bone[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
            torch.tensor(idx, dtype=torch.long),
        )


class SequenceDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, y: np.ndarray, meta: list[dict[str, Any]]):
        self.X_seq = X_seq.astype(np.float32)
        self.y = y.astype(np.int64)
        self.meta = meta

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.X_seq[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
            torch.tensor(idx, dtype=torch.long),
        )


class LandmarkImageDataset(Dataset):
    def __init__(self, joint: np.ndarray, y: np.ndarray, meta: list[dict[str, Any]], image_size: int = 96):
        self.joint = joint.astype(np.float32)
        self.y = y.astype(np.int64)
        self.meta = meta
        self.image_size = image_size

    @staticmethod
    def _norm_to_px(v: np.ndarray, size: int) -> np.ndarray:
        # joint coords are already normalized around roughly [-1, 1]
        clipped = np.clip(v, -1.2, 1.2)
        px = ((clipped + 1.2) / 2.4) * (size - 1)
        return px

    def _render(self, joint_vec: np.ndarray) -> np.ndarray:
        pts = joint_vec.reshape(21, 3)
        x = self._norm_to_px(pts[:, 0], self.image_size)
        y = self._norm_to_px(pts[:, 1], self.image_size)
        y = (self.image_size - 1) - y

        img = Image.new("L", (self.image_size, self.image_size), color=0)
        draw = ImageDraw.Draw(img)

        for u, v in HAND_CONNECTIONS:
            draw.line([(float(x[u]), float(y[u])), (float(x[v]), float(y[v]))], fill=160, width=2)
        for i in range(21):
            r = 2
            draw.ellipse(
                [
                    (float(x[i] - r), float(y[i] - r)),
                    (float(x[i] + r), float(y[i] + r)),
                ],
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


# -----------------------------------------------------------------------------
# Model definitions
# -----------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        weight = (1.0 - pt) ** self.gamma
        if self.alpha is not None:
            weight = weight * self.alpha[targets]
        return (weight * ce).mean()


class MLPBaseline(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPEmbedding(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


class StreamMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: tuple[int, int] = (128, 128), dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoStreamMLP(nn.Module):
    def __init__(self, joint_dim: int, bone_dim: int, num_classes: int):
        super().__init__()
        self.joint_stream = StreamMLP(joint_dim)
        self.bone_stream = StreamMLP(bone_dim)

        fusion_in = self.joint_stream.out_dim + self.bone_stream.out_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_in, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_joint: torch.Tensor, x_bone: torch.Tensor) -> torch.Tensor:
        j = self.joint_stream(x_joint)
        b = self.bone_stream(x_bone)
        return self.head(torch.cat([j, b], dim=-1))


class TCN1DClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, T, D)
        x = x_seq.transpose(1, 2)  # (B, D, T)
        h = self.net(x)
        pooled = h.mean(dim=-1)
        return self.head(pooled)


class TemporalTransformer(nn.Module):
    def __init__(self, seq_len: int, input_dim: int, num_classes: int, d_model: int = 128):
        super().__init__()
        self.frame_embed = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        h = self.frame_embed(x_seq) + self.pos_embed[:, : x_seq.size(1)]
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.head(pooled)


class DWConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ShuffleBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        assert channels % 2 == 0
        mid = channels // 2
        self.conv = nn.Sequential(
            nn.Conv2d(mid, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.Conv2d(mid, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
        b, c, h, w = x.size()
        x = x.view(b, groups, c // groups, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.view(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._channel_shuffle(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        out2 = self.conv(x2)
        return torch.cat([x1, out2], dim=1)


class MBConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, expand: int = 4, stride: int = 1):
        super().__init__()
        hidden = in_ch * expand
        self.use_res = stride == 1 and in_ch == out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        if self.use_res:
            return x + y
        return y


class MobileNetLike(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            DWConvBlock(16, 24, stride=1),
            DWConvBlock(24, 40, stride=2),
            DWConvBlock(40, 64, stride=2),
            DWConvBlock(64, 96, stride=2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class ShuffleNetLike(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.stage = nn.Sequential(
            ShuffleBlock(24),
            ShuffleBlock(24),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            ShuffleBlock(48),
            ShuffleBlock(48),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stage(self.stem(x)))


class EfficientNetLike(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            MBConvBlock(32, 32, expand=1, stride=1),
            MBConvBlock(32, 48, expand=4, stride=2),
            MBConvBlock(48, 64, expand=4, stride=2),
            MBConvBlock(64, 96, expand=4, stride=2),
            MBConvBlock(96, 128, expand=4, stride=1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


# -----------------------------------------------------------------------------
# Data builders
# -----------------------------------------------------------------------------
def frame_arrays(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
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
    samples: list[np.ndarray] = []
    labels: list[int] = []
    meta: list[dict[str, Any]] = []

    for _, g in df.groupby("__source_group"):
        g = g.sort_values("frame_idx").reset_index(drop=True)
        feats = g[feature_cols].to_numpy(dtype=np.float32)
        ys = g["gesture"].to_numpy(dtype=np.int64)

        if len(g) < seq_len:
            continue

        for s in range(0, len(g) - seq_len + 1, stride):
            e = s + seq_len
            samples.append(feats[s:e])
            labels.append(int(ys[e - 1]))  # last-frame label
            meta.append(sample_metadata(g, e - 1))

    if not samples:
        raise ValueError("No sequence samples built. Check seq_len/stride and split sizes.")

    return np.stack(samples, axis=0), np.array(labels, dtype=np.int64), meta


# -----------------------------------------------------------------------------
# Training / evaluation loops
# -----------------------------------------------------------------------------
def _to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device)
    return x


def compute_alpha(labels: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    alpha = inv / inv.sum()
    return torch.tensor(alpha, dtype=torch.float32, device=device)


def forward_batch(model: nn.Module, batch: tuple, mode: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mode == "frame":
        x, y, idx = batch
        logits = model(_to_device(x, device))
        return logits, _to_device(y, device), idx

    if mode == "two_stream":
        xj, xb, y, idx = batch
        logits = model(_to_device(xj, device), _to_device(xb, device))
        return logits, _to_device(y, device), idx

    if mode == "sequence":
        xs, y, idx = batch
        logits = model(_to_device(xs, device))
        return logits, _to_device(y, device), idx

    if mode == "image":
        ximg, y, idx = batch
        logits = model(_to_device(ximg, device))
        return logits, _to_device(y, device), idx

    raise ValueError(f"Unsupported batch mode: {mode}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    mode: str,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in loader:
        optimizer.zero_grad()
        logits, y, _ = forward_batch(model, batch, mode, device)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_count += bs

    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    mode: str,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in loader:
        logits, y, _ = forward_batch(model, batch, mode, device)
        loss = criterion(logits, y)

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_count += bs

    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


@torch.no_grad()
def predict_dataset(
    model: nn.Module,
    loader: DataLoader,
    dataset: Dataset,
    mode: str,
    num_classes: int,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    records: list[dict[str, Any]] = []

    for batch in loader:
        t0 = time.perf_counter()
        logits, y, idx = forward_batch(model, batch, mode, device)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        y_np = y.detach().cpu().numpy().astype(int)
        idx_np = idx.detach().cpu().numpy().astype(int)

        per_sample_ms = infer_ms / max(len(idx_np), 1)

        for i, sample_idx in enumerate(idx_np):
            meta = dataset.meta[sample_idx]
            p = probs[i]
            pred_cls = int(np.argmax(p))

            row = {
                "source_file": meta.get("source_file", ""),
                "frame_idx": int(meta.get("frame_idx", sample_idx)),
                "timestamp": meta.get("timestamp", ""),
                "gesture": int(y_np[i]),
                "pred_class": pred_cls,
                "p_max": float(np.max(p)),
                "t_mp_ms": 0.0,
                "t_feat_ms": 0.0,
                "t_mlp_ms": float(per_sample_ms),
                "t_post_ms": 0.0,
                "latency_total_ms": float(per_sample_ms),
            }
            for c in range(num_classes):
                row[f"p{c}"] = float(p[c])
            records.append(row)

    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# Experiment assembly
# -----------------------------------------------------------------------------
def build_experiment(
    model_id: str,
    split: SplitData,
    angle_cols: list[str],
    seq_len: int,
    seq_stride: int,
    image_size: int,
    num_classes: int,
) -> tuple[nn.Module, str, Dataset, Dataset, Dataset]:
    """
    Returns: model, mode, train_ds, val_ds, test_ds
    mode in {frame, two_stream, sequence, image}
    """
    full_cols = JOINT_COLS + BONE_COLS + angle_cols
    bone_angle_cols = BONE_COLS + angle_cols

    if model_id == "mlp_baseline":
        trX, try_, trm = frame_arrays(split.train_df, JOINT_COLS)
        vaX, vay, vam = frame_arrays(split.val_df, JOINT_COLS)
        teX, tey, tem = frame_arrays(split.test_df, JOINT_COLS)
        model = MLPBaseline(input_dim=len(JOINT_COLS), num_classes=num_classes)
        return model, "frame", FrameDataset(trX, try_, trm), FrameDataset(vaX, vay, vam), FrameDataset(teX, tey, tem)

    if model_id == "mlp_embedding":
        trX, try_, trm = frame_arrays(split.train_df, full_cols)
        vaX, vay, vam = frame_arrays(split.val_df, full_cols)
        teX, tey, tem = frame_arrays(split.test_df, full_cols)
        model = MLPEmbedding(input_dim=len(full_cols), num_classes=num_classes)
        return model, "frame", FrameDataset(trX, try_, trm), FrameDataset(vaX, vay, vam), FrameDataset(teX, tey, tem)

    if model_id == "two_stream_mlp":
        trj, try_, trm = frame_arrays(split.train_df, JOINT_COLS)
        trb, _, _ = frame_arrays(split.train_df, bone_angle_cols)
        vaj, vay, vam = frame_arrays(split.val_df, JOINT_COLS)
        vab, _, _ = frame_arrays(split.val_df, bone_angle_cols)
        tej, tey, tem = frame_arrays(split.test_df, JOINT_COLS)
        teb, _, _ = frame_arrays(split.test_df, bone_angle_cols)

        model = TwoStreamMLP(joint_dim=len(JOINT_COLS), bone_dim=len(bone_angle_cols), num_classes=num_classes)
        return (
            model,
            "two_stream",
            TwoStreamDataset(trj, trb, try_, trm),
            TwoStreamDataset(vaj, vab, vay, vam),
            TwoStreamDataset(tej, teb, tey, tem),
        )

    if model_id == "cnn1d_tcn":
        trX, try_, trm = sequence_arrays(split.train_df, full_cols, seq_len=seq_len, stride=seq_stride)
        vaX, vay, vam = sequence_arrays(split.val_df, full_cols, seq_len=seq_len, stride=seq_stride)
        teX, tey, tem = sequence_arrays(split.test_df, full_cols, seq_len=seq_len, stride=seq_stride)
        model = TCN1DClassifier(input_dim=len(full_cols), num_classes=num_classes)
        return model, "sequence", SequenceDataset(trX, try_, trm), SequenceDataset(vaX, vay, vam), SequenceDataset(teX, tey, tem)

    if model_id == "transformer_embedding":
        trX, try_, trm = sequence_arrays(split.train_df, full_cols, seq_len=seq_len, stride=seq_stride)
        vaX, vay, vam = sequence_arrays(split.val_df, full_cols, seq_len=seq_len, stride=seq_stride)
        teX, tey, tem = sequence_arrays(split.test_df, full_cols, seq_len=seq_len, stride=seq_stride)
        model = TemporalTransformer(seq_len=seq_len, input_dim=len(full_cols), num_classes=num_classes)
        return model, "sequence", SequenceDataset(trX, try_, trm), SequenceDataset(vaX, vay, vam), SequenceDataset(teX, tey, tem)

    if model_id == "mediapipe_hand_landmarker":
        # front-end assumption: landmarks already extracted by MediaPipe Hand Landmarker.
        trX, try_, trm = frame_arrays(split.train_df, JOINT_COLS)
        vaX, vay, vam = frame_arrays(split.val_df, JOINT_COLS)
        teX, tey, tem = frame_arrays(split.test_df, JOINT_COLS)
        model = MLPBaseline(input_dim=len(JOINT_COLS), num_classes=num_classes)
        return model, "frame", FrameDataset(trX, try_, trm), FrameDataset(vaX, vay, vam), FrameDataset(teX, tey, tem)

    if model_id == "mobilenetv3_small":
        trj, try_, trm = frame_arrays(split.train_df, JOINT_COLS)
        vaj, vay, vam = frame_arrays(split.val_df, JOINT_COLS)
        tej, tey, tem = frame_arrays(split.test_df, JOINT_COLS)
        model = MobileNetLike(num_classes=num_classes)
        return (
            model,
            "image",
            LandmarkImageDataset(trj, try_, trm, image_size=image_size),
            LandmarkImageDataset(vaj, vay, vam, image_size=image_size),
            LandmarkImageDataset(tej, tey, tem, image_size=image_size),
        )

    if model_id == "shufflenetv2_x0_5":
        trj, try_, trm = frame_arrays(split.train_df, JOINT_COLS)
        vaj, vay, vam = frame_arrays(split.val_df, JOINT_COLS)
        tej, tey, tem = frame_arrays(split.test_df, JOINT_COLS)
        model = ShuffleNetLike(num_classes=num_classes)
        return (
            model,
            "image",
            LandmarkImageDataset(trj, try_, trm, image_size=image_size),
            LandmarkImageDataset(vaj, vay, vam, image_size=image_size),
            LandmarkImageDataset(tej, tey, tem, image_size=image_size),
        )

    if model_id == "efficientnet_b0":
        trj, try_, trm = frame_arrays(split.train_df, JOINT_COLS)
        vaj, vay, vam = frame_arrays(split.val_df, JOINT_COLS)
        tej, tey, tem = frame_arrays(split.test_df, JOINT_COLS)
        model = EfficientNetLike(num_classes=num_classes)
        return (
            model,
            "image",
            LandmarkImageDataset(trj, try_, trm, image_size=image_size),
            LandmarkImageDataset(vaj, vay, vam, image_size=image_size),
            LandmarkImageDataset(tej, tey, tem, image_size=image_size),
        )

    raise ValueError(f"Unknown model_id: {model_id}")


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    csv_paths = resolve_paths(args.csv_path)
    df = load_preprocessed_data(csv_paths)
    angle_cols = detect_angle_cols(df)

    # split
    split = split_by_group(df, seed=args.seed)

    # class names
    class_names = args.class_names if args.class_names else DEFAULT_CLASS_NAMES
    num_classes = len(class_names)

    # output path
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_root)
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root
    run_dir = out_root / args.model_id / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # build experiment (dataset + model)
    model, mode, train_ds, val_ds, test_ds = build_experiment(
        model_id=args.model_id,
        split=split,
        angle_cols=angle_cols,
        seq_len=args.seq_len,
        seq_stride=args.seq_stride,
        image_size=args.image_size,
        num_classes=num_classes,
    )
    model = model.to(device)

    # dataloaders
    train_labels = np.array(getattr(train_ds, "y"), dtype=np.int64)
    sampler = create_weighted_sampler(train_labels)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # optimization
    alpha = compute_alpha(train_labels, num_classes=num_classes, device=device)
    criterion = FocalLoss(alpha=alpha, gamma=args.focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    # train
    history: list[dict[str, Any]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    stale = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, mode, device)
        va_loss, va_acc = validate_one_epoch(model, val_loader, criterion, mode, device)
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": va_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"[{args.model_id}] epoch {epoch:03d}/{args.epochs} "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
        )

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[{args.model_id}] early stopping at epoch={epoch}")
                break

    model.load_state_dict(best_state)

    # test prediction dataframe
    preds_df = predict_dataset(
        model,
        test_loader,
        test_ds,
        mode=mode,
        num_classes=num_classes,
        device=device,
    )

    preds_path = run_dir / "preds_test.csv"
    preds_df.to_csv(preds_path, index=False)

    # evaluation outputs
    eval_dir = run_dir / "evaluation"
    eval_cfg = EvaluationConfig(
        class_names=class_names,
        neutral_class_id=args.neutral_class_id,
        tau=args.tau,
        vote_n=args.vote_n,
        debounce_k=args.debounce_k,
        fallback_fps=args.fallback_fps,
    )
    metrics_summary = evaluate_predictions(preds_df, eval_dir, eval_cfg)

    # save model + run metadata
    ckpt_path = run_dir / "model.pt"
    torch.save(
        {
            "model_id": args.model_id,
            "model_state_dict": best_state,
            "class_names": class_names,
            "mode": mode,
            "seed": args.seed,
        },
        ckpt_path,
    )

    pd.DataFrame(history).to_csv(run_dir / "train_history.csv", index=False)

    run_summary = {
        "model_id": args.model_id,
        "mode": mode,
        "device": str(device),
        "inputs": [str(p) for p in csv_paths],
        "split_sizes": {
            "train": int(len(split.train_df)),
            "val": int(len(split.val_df)),
            "test": int(len(split.test_df)),
        },
        "dataset_sizes": {
            "train": int(len(train_ds)),
            "val": int(len(val_ds)),
            "test": int(len(test_ds)),
        },
        "best_val_loss": float(best_val_loss),
        "epochs_ran": int(len(history)),
        "output_dir": str(run_dir),
        "metrics": metrics_summary,
    }

    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    # keep a latest pointer (small JSON file)
    latest_info = out_root / args.model_id / "latest.json"
    latest_info.parent.mkdir(parents=True, exist_ok=True)
    with latest_info.open("w", encoding="utf-8") as f:
        json.dump({"latest_run": str(run_dir)}, f, ensure_ascii=False, indent=2)

    print(f"[{args.model_id}] done. Output: {run_dir}")
    return run_summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one JamJamBeat model-comparison pipeline and evaluation.",
    )
    parser.add_argument("--model-id", type=str, required=True, choices=MODEL_CHOICES)
    parser.add_argument(
        "--csv-path",
        action="append",
        default=[],
        help="Input preprocessed CSV path. Repeat this option to pass multiple files.",
    )
    parser.add_argument("--output-root", type=str, default="model/model_evaluation/runs")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--focal-gamma", type=float, default=2.0)

    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--seq-stride", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=96)

    parser.add_argument("--neutral-class-id", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.85)
    parser.add_argument("--vote-n", type=int, default=7)
    parser.add_argument("--debounce-k", type=int, default=3)
    parser.add_argument("--fallback-fps", type=float, default=30.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--class-names", nargs="*", default=[])

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.csv_path:
        args.csv_path = DEFAULT_INPUTS

    run(args)


if __name__ == "__main__":
    main()
