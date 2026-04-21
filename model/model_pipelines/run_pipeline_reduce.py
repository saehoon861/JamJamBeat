#!/usr/bin/env python3
"""
Run one JamJamBeat model-comparison pipeline end-to-end.

- Input: explicit role CSVs
    train / val / test / inference
- Output: model artifact, prediction CSV, and evaluation metrics/plots.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import inspect
import json
import random
import sys
import time
from datetime import datetime, timezone, timedelta
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
        "[ERROR] PyTorch is required for run_pipeline.py\n"
        "Please install torch in your virtual environment first.\n"
        f"Original error: {e}"
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVAL_MODULE_DIR = PROJECT_ROOT / "model" / "model_evaluation" / "모델검증관련파일"
if str(EVAL_MODULE_DIR) not in sys.path:
    # 평가 모듈은 별도 폴더에 있어 직접 경로를 열어준다.
    sys.path.insert(0, str(EVAL_MODULE_DIR))

MODEL_PIPELINES_DIR = PROJECT_ROOT / "model" / "model_pipelines"
if str(MODEL_PIPELINES_DIR) not in sys.path:
    # 각 모델 package를 model_id 문자열로 동적 import 하기 위한 경로다.
    sys.path.insert(0, str(MODEL_PIPELINES_DIR))

from evaluation_runtime import DEFAULT_CLASS_NAMES, EvaluationConfig, evaluate_predictions
from _shared import (
    BONE_COLS,
    DEFAULT_FOCAL_GAMMA,
    DEFAULT_LABEL_SMOOTHING,
    DEFAULT_LOSS_TYPE,
    DEFAULT_USE_ALPHA,
    DEFAULT_USE_LABEL_SMOOTHING,
    DEFAULT_USE_WEIGHTED_SAMPLER,
    JOINT_COLS,
    LOSS_TYPE_CHOICES,
    RAW_JOINT_COLS,
    SplitData,
    detect_angle_cols,
)
from checkpoint_verification import (
    fingerprint_state_dict,
    instantiate_model_from_state_dict,
    safe_torch_load,
    strict_load_state_dict,
)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
MODEL_CHOICES = [
    "mlp_original",
    "mlp_sequence_delta",
    "mlp_embedding",
    "cnn1d_tcn",
    "transformer_embedding",
    "mobilenetv3_small",
    "shufflenetv2_x0_5",
    "efficientnet_b0",
    "Landmark_Spatial_Transformer",
]


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """numpy / torch / python RNG를 함께 고정해 실험 재현성을 맞춘다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_paths(csv_paths: list[str]) -> list[Path]:
    """CLI에서 받은 상대 경로를 절대 경로로 정규화한다.
    우선순위: CWD 기준 → PROJECT_ROOT 기준
    """
    resolved: list[Path] = []
    for path in csv_paths:
        p = Path(path)
        if not p.is_absolute():
            cwd_candidate = Path.cwd() / p
            p = cwd_candidate if cwd_candidate.exists() else PROJECT_ROOT / p
        resolved.append(p.resolve())
    return resolved


def resolve_role_path(path_str: str) -> Path:
    return resolve_paths([path_str])[0]


def normalize_source_groups(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "source_file" in normalized.columns and normalized["source_file"].nunique() >= 1:
        normalized["__source_group"] = normalized["source_file"].astype(str)
    return normalized


def infer_dataset_key(train_csv_path: Path) -> str:
    stem = train_csv_path.stem
    suffix = "_train"
    if not stem.endswith(suffix):
        raise ValueError(f"Expected *_train.csv naming, got: {train_csv_path.name}")
    return stem[: -len(suffix)]


def infer_normalization_family(dataset_key: str) -> str:
    for family in ("baseline", "pos_only", "scale_only", "pos_scale"):
        if dataset_key == family or dataset_key.startswith(f"{family}_"):
            return family
    return "unknown"


def _source_groups(df: pd.DataFrame, group_col: str = "__source_group") -> list[str]:
    """summary/evaluation 기록용 source group 목록을 정렬해서 반환한다."""
    if group_col not in df.columns:
        return []
    return sorted(str(v) for v in df[group_col].dropna().unique().tolist())


def _rows_by_source_group(df: pd.DataFrame, group_col: str = "__source_group") -> dict[str, int]:
    """DataFrame 내 source group별 row 수를 JSON-friendly dict로 만든다."""
    if group_col not in df.columns:
        return {}
    counts = df[group_col].value_counts().sort_index()
    return {str(group): int(count) for group, count in counts.items()}


def _source_counts_by_role(
    split: SplitData,
    inference_df: pd.DataFrame | None = None,
    group_col: str = "__source_group",
) -> dict[str, int]:
    counts = {
        "train": len(_source_groups(split.train_df, group_col=group_col)),
        "val": len(_source_groups(split.val_df, group_col=group_col)),
        "test": len(_source_groups(split.test_df, group_col=group_col)),
    }
    if inference_df is not None:
        counts["inference"] = len(_source_groups(inference_df, group_col=group_col))
    return counts


def build_dataset_info(
    input_roles: dict[str, Path],
    merged_df: pd.DataFrame,
    split: SplitData,
    dataset_key: str,
    normalization_family: str,
    inference_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """run/evaluation 폴더에 같이 저장할 데이터셋 메타를 구성한다."""
    source_counts = _source_counts_by_role(split, inference_df=inference_df)
    info = {
        "dataset_key": dataset_key,
        "normalization_family": normalization_family,
        "fixed_video_level_split": True,
        "source_counts": source_counts,
        "test_kind": "static_images_63d",
        "test_sequence_policy": "independent_repeat",
        "inference_sequence_policy": "sliding",
        "official_ranking_basis": "test_csv_static_images",
        "temporal_metrics_policy": "disabled_for_static_images",
        "input_roles": {role: str(path) for role, path in input_roles.items()},
        "input_csv_paths": [str(p) for p in input_roles.values()],
        "input_csv_names": [p.name for p in input_roles.values()],
        "source_groups": _source_groups(merged_df),
        "total_rows": int(len(merged_df)),
        "rows_by_source_group": _rows_by_source_group(merged_df),
        "split": {
            "train": {
                "rows": int(len(split.train_df)),
                "source_groups": _source_groups(split.train_df),
                "rows_by_source_group": _rows_by_source_group(split.train_df),
            },
            "val": {
                "rows": int(len(split.val_df)),
                "source_groups": _source_groups(split.val_df),
                "rows_by_source_group": _rows_by_source_group(split.val_df),
            },
            "test": {
                "rows": int(len(split.test_df)),
                "source_groups": _source_groups(split.test_df),
                "rows_by_source_group": _rows_by_source_group(split.test_df),
            },
        },
    }
    if inference_df is not None:
        info["split"]["inference"] = {
            "rows": int(len(inference_df)),
            "source_groups": _source_groups(inference_df),
            "rows_by_source_group": _rows_by_source_group(inference_df),
        }
    return info


def load_preprocessed_data(csv_paths: list[Path]) -> pd.DataFrame:
    """여러 전처리 CSV를 하나의 학습용 DataFrame으로 병합한다."""
    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input CSV not found: {path}")

        df = pd.read_csv(path)
        df = df.copy()  # fragmented DataFrame defrag (PerformanceWarning 방지)
        # 원본 파일 단위 split과 sequence window 구성을 위해 source group을 심는다.
        df["__source_group"] = path.stem

        if "source_file" not in df.columns:
            df["source_file"] = path.stem

        # 후속 평가 / 시퀀스 정렬에 필요한 최소 컬럼을 강제한다.
        required = {"gesture", "frame_idx", "timestamp"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")

        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    # gesture가 비어 있는 행은 학습 샘플로 의미가 없으므로 제거한다.
    merged = merged.dropna(subset=["gesture"]).copy()
    merged["gesture"] = merged["gesture"].astype(int)

    # 전처리 CSV (nx*/bx* 컬럼) 또는 raw CSV (x*/y*/z* 컬럼) 둘 다 허용한다.
    has_preprocessed = all(c in merged.columns for c in JOINT_COLS[:3])
    has_raw = all(c in merged.columns for c in RAW_JOINT_COLS[:3])

    if not has_preprocessed and not has_raw:
        raise ValueError(
            "Input CSV에 유효한 landmark 컬럼이 없습니다. "
            "전처리 CSV (nx0,ny0...) 또는 raw CSV (x0,y0,z0...) 중 하나여야 합니다."
        )

    angle_cols = detect_angle_cols(merged)

    # 실제 존재하는 컬럼만 dropna 대상으로 사용한다.
    feature_cols = [c for c in (JOINT_COLS + BONE_COLS + angle_cols) if c in merged.columns]
    if not feature_cols:
        feature_cols = [c for c in RAW_JOINT_COLS if c in merged.columns]

    merged = merged.dropna(subset=feature_cols).reset_index(drop=True)
    return merged


def _block_split_single(
    df: pd.DataFrame,
    rng: np.random.Generator,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seq_len: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """단일 source 내에서 seq_len 블록 단위로 train/val/test를 나눈다.

    frame_idx 순으로 정렬 후 seq_len 크기 블록을 셔플해 분배하므로
    슬라이딩 윈도우가 split 경계를 넘지 않는다.
    마지막 불완전 블록은 train에 편입한다.
    """
    sort_col = "frame_idx" if "frame_idx" in df.columns else None
    sdf = df.sort_values(sort_col).reset_index(drop=True) if sort_col else df.reset_index(drop=True)

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    block_size = max(seq_len, 1)
    n_rows = len(sdf)
    n_blocks = n_rows // block_size

    block_idx = np.arange(n_blocks)
    rng.shuffle(block_idx)

    n_test = max(1, int(round(n_blocks * test_ratio)))
    n_val  = max(1, int(round(n_blocks * val_ratio)))
    if n_test + n_val >= n_blocks:
        n_test, n_val = 1, 1

    test_blocks  = set(block_idx[:n_test])
    val_blocks   = set(block_idx[n_test:n_test + n_val])
    train_blocks = set(block_idx[n_test + n_val:])

    row_block = np.arange(n_rows) // block_size

    train_mask = np.array([b in train_blocks or b >= n_blocks for b in row_block])
    val_mask   = np.array([b in val_blocks   for b in row_block])
    test_mask  = np.array([b in test_blocks  for b in row_block])

    return sdf[train_mask].copy(), sdf[val_mask].copy(), sdf[test_mask].copy()


def split_by_group(
    df: pd.DataFrame,
    group_col: str = "__source_group",
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seq_len: int = 1,
) -> SplitData:
    """가능하면 source group 단위로 train / val / test를 나눈다.

    우선순위:
    1. __source_group 4개 이상 (기본, 4파일 입력) → group-level split
    2. source_file 컬럼에 여러 값 존재 (통합 CSV) → source_file 내부 블록 split
       각 source_file 안에서 seq_len 블록 단위로 나눠 모든 클래스가
       train/val/test에 고루 포함되도록 보장한다.
    3. 진짜 단일 source → 전체에 seq_len 블록 단위 split
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    groups = list(df[group_col].dropna().unique())
    rng = np.random.default_rng(seed)

    if len(groups) >= 4:
        # 우선순위 1: 4개 이상 파일 → group-level split (기존 동작 유지)
        rng.shuffle(groups)
        n_groups = len(groups)
        n_test = max(1, int(round(n_groups * test_ratio)))
        n_val  = max(1, int(round(n_groups * val_ratio)))

        if n_test + n_val >= n_groups:
            n_test = 1
            n_val  = 1

        test_groups  = set(groups[:n_test])
        val_groups   = set(groups[n_test:n_test + n_val])
        train_groups = set(groups[n_test + n_val:])

        if not train_groups:
            spill = list(val_groups)
            train_groups.add(spill[0])
            val_groups.remove(spill[0])

        train_df = df[df[group_col].isin(train_groups)].copy()
        val_df   = df[df[group_col].isin(val_groups)].copy()
        test_df  = df[df[group_col].isin(test_groups)].copy()

    elif "source_file" in df.columns and df["source_file"].nunique() >= 2:
        # 우선순위 2: 통합 CSV — source_file별 내부 블록 split
        # 각 source_file에서 독립적으로 split해 모든 클래스 대표성을 보장한다.
        train_parts, val_parts, test_parts = [], [], []
        for _, grp in df.groupby("source_file"):
            tr, va, te = _block_split_single(
                grp, rng, train_ratio, val_ratio, test_ratio, seq_len
            )
            train_parts.append(tr)
            val_parts.append(va)
            test_parts.append(te)

        train_df = pd.concat(train_parts, ignore_index=True)
        val_df   = pd.concat(val_parts,   ignore_index=True)
        test_df  = pd.concat(test_parts,  ignore_index=True)

    else:
        # 우선순위 3: 진짜 단일 source → seq_len 블록 단위 split
        train_df, val_df, test_df = _block_split_single(
            df, rng, train_ratio, val_ratio, test_ratio, seq_len
        )

    return SplitData(
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
    )


def create_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """클래스 불균형 완화를 위해 inverse-frequency sampler를 만든다."""
    classes, counts = np.unique(labels, return_counts=True)
    class_w = {int(c): 1.0 / max(int(cnt), 1) for c, cnt in zip(classes, counts)}
    sample_weights = np.array([class_w[int(y)] for y in labels], dtype=np.float64)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


# -----------------------------------------------------------------------------
# Loss function
# -----------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """쉬운 샘플의 기여를 줄이고 희소 클래스에 더 집중시키는 loss."""
    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = DEFAULT_FOCAL_GAMMA,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        weight = (1.0 - pt) ** self.gamma
        if self.alpha is not None:
            weight = weight * self.alpha[targets]
        return (weight * ce).mean()






# -----------------------------------------------------------------------------
# Training / evaluation loops
# -----------------------------------------------------------------------------
def _to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device)
    return x


def compute_alpha(labels: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    """Focal loss의 class weight(alpha)를 현재 train split 기준으로 계산한다."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    alpha = inv / inv.sum()
    return torch.tensor(alpha, dtype=torch.float32, device=device)


def build_training_recipe(
    *,
    labels: np.ndarray,
    num_classes: int,
    device: torch.device,
    loss_type: str,
    use_weighted_sampler: bool,
    use_alpha: bool,
    use_label_smoothing: bool,
    focal_gamma: float,
) -> tuple[WeightedRandomSampler | None, bool, nn.Module, dict[str, Any]]:
    """Train sampler / loss 조합을 현재 실험 설정에 맞게 조립한다."""
    sampler = create_weighted_sampler(labels) if use_weighted_sampler else None
    shuffle_train = sampler is None
    alpha = compute_alpha(labels, num_classes=num_classes, device=device) if use_alpha else None
    label_smoothing = DEFAULT_LABEL_SMOOTHING if use_label_smoothing else 0.0

    if loss_type == "cross_entropy":
        criterion = nn.CrossEntropyLoss(weight=alpha, label_smoothing=label_smoothing)
    elif loss_type == "focal":
        criterion = FocalLoss(
            alpha=alpha,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    recipe_meta = {
        "loss_type": loss_type,
        "sampler_policy": "weighted_sampler" if use_weighted_sampler else "shuffle",
        "alpha_policy": "enabled" if use_alpha else "disabled",
        "label_smoothing_enabled": bool(use_label_smoothing),
        "effective_label_smoothing": float(label_smoothing),
        "effective_focal_gamma": float(focal_gamma),
    }
    return sampler, shuffle_train, criterion, recipe_meta


def get_dataset_labels(dataset: Dataset, *, model_id: str, split_name: str = "train") -> np.ndarray:
    """Dataset이 sampler/loss 계산에 필요한 1차원 label 배열 계약을 만족하는지 검증한다."""
    dataset_class = dataset.__class__.__name__
    raw_labels = getattr(dataset, "y", None)
    if raw_labels is None:
        raise ValueError(
            f"{model_id} {split_name} dataset ({dataset_class}) must expose `dataset.y` "
            "as a 1D integer label array."
        )

    labels = np.asarray(raw_labels)
    if labels.ndim != 1:
        raise ValueError(
            f"{model_id} {split_name} dataset ({dataset_class}) has invalid `dataset.y` "
            f"shape {labels.shape}; expected a 1D label array."
        )
    if len(labels) != len(dataset):
        raise ValueError(
            f"{model_id} {split_name} dataset ({dataset_class}) has len(dataset.y)={len(labels)} "
            f"but len(dataset)={len(dataset)}."
        )

    try:
        return labels.astype(np.int64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{model_id} {split_name} dataset ({dataset_class}) must provide integer-like labels in `dataset.y`."
        ) from exc


def forward_batch(model: nn.Module, batch: tuple, mode: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dataset마다 다른 batch shape를 공통 (logits, y, idx) 계약으로 맞춘다."""
    if mode == "frame":
        x, y, idx = batch
        logits = model(_to_device(x, device))
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
    """한 epoch 학습 후 샘플 수 가중 평균 loss / accuracy를 반환한다."""
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
    """validation split에 대해 동일한 집계 지표를 계산한다."""
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


# @torch.no_grad()
# def predict_dataset(
#     model: nn.Module,
#     loader: DataLoader,
#     dataset: Dataset,
#     mode: str,
#     num_classes: int,
#     device: torch.device,
# ) -> pd.DataFrame:
#     """test split 전체를 순회하며 평가용 예측 원본 CSV를 만든다."""
#     model.eval()
#     records: list[dict[str, Any]] = []

#     for batch in loader:
#         t0 = time.perf_counter()
#         logits, y, idx = forward_batch(model, batch, mode, device)
#         infer_ms = (time.perf_counter() - t0) * 1000.0

#         probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
#         y_np = y.detach().cpu().numpy().astype(int)
#         idx_np = idx.detach().cpu().numpy().astype(int)

#         # 배치 단위 wall time을 샘플 수로 나눠 간단한 per-sample latency를 저장한다.
#         per_sample_ms = infer_ms / max(len(idx_np), 1)

#         for i, sample_idx in enumerate(idx_np):
#             meta = dataset.meta[sample_idx]
#             p = probs[i]
#             pred_cls = int(np.argmax(p))

#             row = {
#                 "source_file": meta.get("source_file", ""),
#                 "frame_idx": int(meta.get("frame_idx", sample_idx)),
#                 "timestamp": meta.get("timestamp", ""),
#                 "gesture": int(y_np[i]),
#                 "pred_class": pred_cls,
#                 "p_max": float(np.max(p)),
#                 "t_mp_ms": 0.0,
#                 "t_feat_ms": 0.0,
#                 "t_mlp_ms": float(per_sample_ms),
#                 "t_post_ms": 0.0,
#                 "latency_total_ms": float(per_sample_ms),
#             }
#             for c in range(num_classes):
#                 row[f"p{c}"] = float(p[c])
#             records.append(row)

#     return pd.DataFrame(records)

@torch.no_grad()
def predict_dataset(
    model: nn.Module,
    loader: DataLoader,
    dataset: Dataset,
    mode: str,
    num_classes: int,
    device: torch.device,
) -> pd.DataFrame:
    """test split 전체를 순회하며 평가용 예측 원본 CSV를 만든다."""
    model.eval()
    records: list[dict[str, Any]] = []

    for batch in loader:
        t0 = time.perf_counter()
        logits, y, idx = forward_batch(model, batch, mode, device)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        y_np = y.detach().cpu().numpy().astype(int)

        # 배치 단위 wall time을 샘플 수로 나눠 간단한 per-sample latency를 저장한다.
        per_sample_ms = infer_ms / max(len(y_np), 1)

        # 1) idx가 tensor인 기존 경우
        if torch.is_tensor(idx):
            idx_np = idx.detach().cpu().numpy().astype(int)

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

        # 2) idx가 dict(meta batch)인 현재 경우
        elif isinstance(idx, dict):
            batch_size = len(y_np)

            for i in range(batch_size):
                meta = {}
                for k, v in idx.items():
                    if torch.is_tensor(v):
                        meta[k] = v[i].item() if v.ndim > 0 else v.item()
                    elif isinstance(v, (list, tuple)):
                        meta[k] = v[i]
                    else:
                        meta[k] = v

                p = probs[i]
                pred_cls = int(np.argmax(p))

                row = {
                    "source_file": meta.get("source_file", ""),
                    "frame_idx": int(meta.get("frame_idx", i)),
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

        else:
            raise TypeError(f"Unsupported idx type in predict_dataset: {type(idx)}")

    return pd.DataFrame(records)

# -----------------------------------------------------------------------------
# Experiment assembly (dynamic dispatch)
# -----------------------------------------------------------------------------
def build_experiment(
    model_id: str,
    split: SplitData,
    angle_cols: list[str],
    seq_len: int,
    seq_stride: int,
    image_size: int,
    num_classes: int,
    test_sequence_policy: str = "sliding",
) -> tuple[nn.Module, str, Dataset, Dataset, Dataset]:
    """
    model_id로 대응되는 builder를 동적 import 해서 dataset / model 조립 책임을 위임한다.
    Returns: model, mode, train_ds, val_ds, test_ds
    mode in {frame, sequence, image}
    """
    mod = importlib.import_module(f"{model_id}.dataset")
    kwargs = dict(
        split=split,
        angle_cols=angle_cols,
        seq_len=seq_len,
        seq_stride=seq_stride,
        image_size=image_size,
        num_classes=num_classes,
    )
    if "test_sequence_policy" in inspect.signature(mod.build).parameters:
        kwargs["test_sequence_policy"] = test_sequence_policy
    return mod.build(
        **kwargs,
    )


def run(args: argparse.Namespace) -> dict:
    """단일 모델 실험의 전체 생명주기(train -> test -> evaluate -> save)를 실행한다."""
    set_seed(args.seed)

    if args.device == "auto":
        try:
            use_cuda = torch.cuda.is_available()
        except Exception:
            use_cuda = False
        device = torch.device("cuda" if use_cuda else "cpu")
    else:
        device = torch.device(args.device)

    input_roles: dict[str, Path] = {
        "train": resolve_role_path(args.train_csv),
        "val": resolve_role_path(args.val_csv),
        "test": resolve_role_path(args.test_csv),
    }
    if args.inference_csv:
        input_roles["inference"] = resolve_role_path(args.inference_csv)

    train_df = normalize_source_groups(load_preprocessed_data([input_roles["train"]]))
    val_df = normalize_source_groups(load_preprocessed_data([input_roles["val"]]))
    test_df = normalize_source_groups(load_preprocessed_data([input_roles["test"]]))
    inference_df = None
    if "inference" in input_roles:
        inference_df = normalize_source_groups(load_preprocessed_data([input_roles["inference"]]))

    reference_columns = list(train_df.columns)
    for role_name, frame in (
        ("val", val_df),
        ("test", test_df),
        ("inference", inference_df),
    ):
        if frame is None:
            continue
        if list(frame.columns) != reference_columns:
            raise ValueError(
                f"Column mismatch between train and {role_name}: "
                f"{input_roles['train'].name} vs {input_roles[role_name].name}"
            )

    split = SplitData(
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
    )
    merged_frames = [train_df, val_df, test_df]
    if inference_df is not None:
        merged_frames.append(inference_df)
    merged_df = pd.concat(merged_frames, ignore_index=True)
    angle_cols = detect_angle_cols(merged_df)

    dataset_key = infer_dataset_key(input_roles["train"])
    normalization_family = infer_normalization_family(dataset_key)
    dataset_info = build_dataset_info(
        input_roles=input_roles,
        merged_df=merged_df,
        split=split,
        dataset_key=dataset_key,
        normalization_family=normalization_family,
        inference_df=inference_df,
    )
    source_counts = dict(dataset_info.get("source_counts") or {})

    # 평가 리포트와 checkpoint 저장에 같은 class ordering을 사용한다.
    class_names = args.class_names if args.class_names else DEFAULT_CLASS_NAMES
    num_classes = len(class_names)

    # 모델별 / 실행시각별 폴더를 분리해 실험 결과를 누적 저장한다.
    KST = timezone(timedelta(hours=9))
    timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_root)
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root
    run_dir = out_root / args.model_id / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # 각 model_id는 자기 builder 안에서 feature 구성과 model 클래스를 결정한다.
    model, mode, train_ds, val_ds, test_ds = build_experiment(
        model_id=args.model_id,
        split=split,
        angle_cols=angle_cols,
        seq_len=args.seq_len,
        seq_stride=args.seq_stride,
        image_size=args.image_size,
        num_classes=num_classes,
        test_sequence_policy="independent_repeat",
    )
    model = model.to(device)
    dataset_info["dataset_sample_counts"] = {
        "train": int(len(train_ds)),
        "val": int(len(val_ds)),
        "test": int(len(test_ds)),
    }
    if inference_df is not None:
        inference_split = SplitData(
            train_df=split.train_df,
            val_df=split.val_df,
            test_df=inference_df.reset_index(drop=True),
        )
        _, inference_mode, _, _, inference_ds = build_experiment(
            model_id=args.model_id,
            split=inference_split,
            angle_cols=angle_cols,
            seq_len=args.seq_len,
            seq_stride=args.seq_stride,
            image_size=args.image_size,
            num_classes=num_classes,
            test_sequence_policy="sliding",
        )
        if inference_mode != mode:
            raise ValueError(
                f"Inference mode mismatch for {args.model_id}: {inference_mode} != {mode}"
            )
    else:
        inference_ds = None
    dataset_info["mode"] = mode

    train_labels = get_dataset_labels(train_ds, model_id=args.model_id, split_name="train")
    sampler, shuffle_train, criterion, recipe_meta = build_training_recipe(
        labels=train_labels,
        num_classes=num_classes,
        device=device,
        loss_type=args.loss_type,
        use_weighted_sampler=args.use_weighted_sampler,
        use_alpha=args.use_alpha,
        use_label_smoothing=args.use_label_smoothing,
        focal_gamma=args.focal_gamma,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle_train,
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
    if inference_ds is not None:
        inference_loader = DataLoader(
            inference_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        dataset_info["dataset_sample_counts"]["inference"] = int(len(inference_ds))
    else:
        inference_loader = None

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",                      # val_loss가 낮아질수록 좋음
        factor=args.lr_decay_factor,     # LR 감소 비율
        patience=args.lr_patience,       # 개선 없을 때 몇 epoch 기다릴지
        threshold=args.lr_threshold,     # 이 정도 이상 개선돼야 개선으로 간주
        threshold_mode="rel",
        min_lr=args.min_lr,              # LR 하한선
    )

    # best validation loss 기준으로 early stopping과 checkpoint selection을 수행한다.
    history: list[dict[str, Any]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    stale = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, mode, device
        )
        va_loss, va_acc = validate_one_epoch(
            model, val_loader, criterion, mode, device
        )

        # val_loss 기준으로 plateau 감지 후 learning rate 감소
        scheduler.step(va_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": va_acc,
                "lr": current_lr,
            }
        )

        print(
            f"[{args.model_id}] epoch {epoch:03d}/{args.epochs} "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} "
            f"lr={current_lr:.8f}"
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

    # evaluation_runtime은 이 예측 원본 CSV를 기준으로 모든 리포트를 만든다.
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

    if inference_loader is not None and inference_ds is not None:
        preds_inference_df = predict_dataset(
            model,
            inference_loader,
            inference_ds,
            mode=mode,
            num_classes=num_classes,
            device=device,
        )
        preds_inference_path = run_dir / "preds_inference.csv"
        preds_inference_df.to_csv(preds_inference_path, index=False)
    else:
        preds_inference_path = None

    # 후처리 threshold / voting / debounce까지 포함한 지표를 여기서 계산한다.
    eval_dir = run_dir / "evaluation"
    eval_cfg = EvaluationConfig(
        class_names=class_names,
        neutral_class_id=args.neutral_class_id,
        tau=args.tau,
        vote_n=args.vote_n,
        debounce_k=args.debounce_k,
        fallback_fps=args.fallback_fps,
        dataset_info=dataset_info,
    )
    metrics_summary = evaluate_predictions(preds_df, eval_dir, eval_cfg)

    # checkpoint에는 재로딩에 필요한 최소 실행 문맥과 fingerprint를 함께 저장한다.
    ckpt_path = run_dir / "model.pt"
    best_state_verification = fingerprint_state_dict(best_state)
    checkpoint_payload = {
        "model_id": args.model_id,
        "model_state_dict": best_state,
        "class_names": class_names,
        "mode": mode,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "seq_stride": args.seq_stride,
        "image_size": args.image_size,
        "checkpoint_verification": {
            **best_state_verification,
            "model_id": args.model_id,
            "mode": mode,
            "seq_len": int(args.seq_len),
            "image_size": int(args.image_size),
        },
    }
    torch.save(checkpoint_payload, ckpt_path)

    saved_checkpoint = safe_torch_load(ckpt_path, torch.device("cpu"))
    saved_state_dict = saved_checkpoint["model_state_dict"]
    saved_state_verification = fingerprint_state_dict(saved_state_dict)
    reloaded_model, _, _, _ = instantiate_model_from_state_dict(
        str(saved_checkpoint.get("model_id") or args.model_id),
        saved_state_dict,
        num_classes,
        seq_len_hint=int(saved_checkpoint.get("seq_len") or args.seq_len),
        default_seq_len=int(args.seq_len),
    )
    strict_reload_info = strict_load_state_dict(reloaded_model, saved_state_dict)
    stored_checkpoint_verification = dict(saved_checkpoint.get("checkpoint_verification") or {})
    checkpoint_verification = {
        "checkpoint_path": str(ckpt_path),
        "model_id": str(saved_checkpoint.get("model_id") or args.model_id),
        "mode": str(saved_checkpoint.get("mode") or mode),
        "seq_len": int(saved_checkpoint.get("seq_len") or args.seq_len),
        "image_size": int(saved_checkpoint.get("image_size") or args.image_size),
        **saved_state_verification,
        "saved_matches_best_state": (
            saved_state_verification["checkpoint_fingerprint"]
            == best_state_verification["checkpoint_fingerprint"]
        ),
        "stored_matches_saved_state": (
            stored_checkpoint_verification.get("checkpoint_fingerprint")
            == saved_state_verification["checkpoint_fingerprint"]
        ),
        "stored_checkpoint_fingerprint": stored_checkpoint_verification.get("checkpoint_fingerprint"),
        **strict_reload_info,
    }

    pd.DataFrame(history).to_csv(run_dir / "train_history.csv", index=False)

    run_summary = {
        "model_id": args.model_id,
        "dataset_key": dataset_key,
        "normalization_family": normalization_family,
        "mode": mode,
        "device": str(device),
        "inputs": [str(p) for p in input_roles.values()],
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "optimizer": "AdamW",
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "loss_type": args.loss_type,
            "use_weighted_sampler": args.use_weighted_sampler,
            "use_alpha": args.use_alpha,
            "use_label_smoothing": args.use_label_smoothing,
            "focal_gamma": args.focal_gamma,
            "label_smoothing": recipe_meta["effective_label_smoothing"],
            "sampler_policy": recipe_meta["sampler_policy"],
            "alpha_policy": recipe_meta["alpha_policy"],
            "tau": args.tau,
            "vote_n": args.vote_n,
            "debounce_k": args.debounce_k,
            "fallback_fps": args.fallback_fps,
            "lr_scheduler": "ReduceLROnPlateau",
            "lr_patience": args.lr_patience,
            "lr_decay_factor": args.lr_decay_factor,
            "lr_threshold": args.lr_threshold,
            "min_lr": args.min_lr,
        },
        "split_sizes": {
            "train": int(len(split.train_df)),
            "val": int(len(split.val_df)),
            "test": int(len(split.test_df)),
        },
        "dataset_info": dataset_info,
        "dataset_sizes": {
            "train": int(len(train_ds)),
            "val": int(len(val_ds)),
            "test": int(len(test_ds)),
        },
        "fixed_video_level_split": True,
        "source_counts": source_counts,
        "test_kind": "static_images_63d",
        "test_sequence_policy": "independent_repeat",
        "inference_sequence_policy": "sliding",
        "official_ranking_basis": "test_csv_static_images",
        "inference_used": inference_df is not None,
        "preds_inference_csv": str(preds_inference_path) if preds_inference_path else None,
        "best_val_loss": float(best_val_loss),
        "epochs_ran": int(len(history)),
        "output_dir": str(run_dir),
        "checkpoint_verification": checkpoint_verification,
        "metrics": metrics_summary,
    }
    if inference_df is not None and inference_ds is not None:
        run_summary["split_sizes"]["inference"] = int(len(inference_df))
        run_summary["dataset_sizes"]["inference"] = int(len(inference_ds))

    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    # latest.json은 run_all / 뷰어가 최신 실험 결과를 찾는 인덱스 역할을 한다.
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
    parser.add_argument("--train-csv", type=str, default=None, help="Train role CSV path.")
    parser.add_argument("--val-csv", type=str, default=None, help="Validation role CSV path.")
    parser.add_argument("--test-csv", type=str, default=None, help="Test role CSV path.")
    parser.add_argument("--inference-csv", type=str, default=None, help="Optional inference hold-out CSV path.")
    parser.add_argument(
        "--csv-path",
        action="append",
        default=[],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output-root", type=str, default="model/model_evaluation/pipelines")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument(
        "--loss-type",
        type=str,
        default=DEFAULT_LOSS_TYPE,
        choices=LOSS_TYPE_CHOICES,
        help=f"학습 loss 종류. 기본값: {DEFAULT_LOSS_TYPE}",
    )
    parser.add_argument(
        "--use-weighted-sampler",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_WEIGHTED_SAMPLER,
        help=f"train loader에 weighted sampler 사용 여부. 기본값: {DEFAULT_USE_WEIGHTED_SAMPLER}",
    )
    parser.add_argument(
        "--use-alpha",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_ALPHA,
        help=f"class alpha(weight) 사용 여부. 기본값: {DEFAULT_USE_ALPHA}",
    )
    parser.add_argument(
        "--use-label-smoothing",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_LABEL_SMOOTHING,
        help=(
            "label smoothing 사용 여부. "
            f"활성화 시 smoothing={DEFAULT_LABEL_SMOOTHING}"
        ),
    )
    parser.add_argument("--focal-gamma", type=float, default=DEFAULT_FOCAL_GAMMA, help=argparse.SUPPRESS)

    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--seq-stride", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=96)

    parser.add_argument("--neutral-class-id", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.90)
    parser.add_argument("--vote-n", type=int, default=7)
    parser.add_argument("--debounce-k", type=int, default=5)
    parser.add_argument("--fallback-fps", type=float, default=30.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--class-names", nargs="*", default=[])
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=8,
        help="val_loss 개선이 없을 때 learning rate를 줄이기 전 기다릴 epoch 수",
    )
    parser.add_argument(
        "--lr-decay-factor",
        type=float,
        default=0.5,
        help="learning rate 감소 비율 (new_lr = old_lr * factor)",
    )
    parser.add_argument(
        "--lr-threshold",
        type=float,
        default=1e-3,
        help="이 값 이상 개선되어야 val_loss 개선으로 간주",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        help="learning rate 최소값",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.csv_path:
        raise SystemExit(
            "--csv-path flow is deprecated. "
            "Use --train-csv --val-csv --test-csv [--inference-csv]."
        )
    missing = [
        name
        for name in ("train_csv", "val_csv", "test_csv")
        if getattr(args, name) in (None, "")
    ]
    if missing:
        raise SystemExit(
            "Missing required role CSVs: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )

    run(args)


if __name__ == "__main__":
    main()
