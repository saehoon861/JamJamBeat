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
import importlib
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
        "[ERROR] PyTorch is required for run_model_pipeline.py\n"
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
from _shared import JOINT_COLS, BONE_COLS, RAW_JOINT_COLS, SplitData, detect_angle_cols


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
DEFAULT_INPUTS = [
    "model/data_fusion/man1_right_for_poc_output.csv",
    "model/data_fusion/man2_right_for_poc_output.csv",
    "model/data_fusion/man3_right_for_poc_output.csv",
    "model/data_fusion/woman1_right_for_poc_output.csv",
]

MODEL_CHOICES = [
    "mlp_baseline",
    "mlp_baseline_full",
    "mlp_baseline_seq8",
    "mlp_sequence_joint",
    "mlp_temporal_pooling",
    "mlp_sequence_delta",
    "mlp_embedding",
    "two_stream_mlp",
    "cnn1d_tcn",
    "transformer_embedding",
    "mobilenetv3_small",
    "shufflenetv2_x0_5",
    "efficientnet_b0",
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
    """CLI에서 받은 상대 경로를 프로젝트 루트 기준 절대 경로로 정규화한다."""
    resolved: list[Path] = []
    for path in csv_paths:
        p = Path(path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        resolved.append(p.resolve())
    return resolved


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


def build_dataset_info(
    csv_paths: list[Path],
    merged_df: pd.DataFrame,
    split: SplitData,
) -> dict[str, Any]:
    """run/evaluation 폴더에 같이 저장할 데이터셋 메타를 구성한다."""
    return {
        "input_csv_paths": [str(p) for p in csv_paths],
        "input_csv_names": [p.name for p in csv_paths],
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


def split_by_group(
    df: pd.DataFrame,
    group_col: str = "__source_group",
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> SplitData:
    """가능하면 source group 단위로 train / val / test를 나눈다.

    원본 파일 수가 너무 적으면 학습 불능을 피하기 위해 row-level fallback을 사용한다.
    """
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


def forward_batch(model: nn.Module, batch: tuple, mode: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dataset마다 다른 batch shape를 공통 (logits, y, idx) 계약으로 맞춘다."""
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
        idx_np = idx.detach().cpu().numpy().astype(int)

        # 배치 단위 wall time을 샘플 수로 나눠 간단한 per-sample latency를 저장한다.
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
) -> tuple[nn.Module, str, Dataset, Dataset, Dataset]:
    """
    model_id로 대응되는 builder를 동적 import 해서 dataset / model 조립 책임을 위임한다.
    Returns: model, mode, train_ds, val_ds, test_ds
    mode in {frame, two_stream, sequence, image}
    """
    mod = importlib.import_module(f"{model_id}.dataset")
    return mod.build(
        split=split,
        angle_cols=angle_cols,
        seq_len=seq_len,
        seq_stride=seq_stride,
        image_size=image_size,
        num_classes=num_classes,
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

    csv_paths = resolve_paths(args.csv_path)
    df = load_preprocessed_data(csv_paths)
    angle_cols = detect_angle_cols(df)

    # 동일 source group이 train / test에 동시에 들어가지 않도록 먼저 split한다.
    split = split_by_group(df, seed=args.seed)
    dataset_info = build_dataset_info(csv_paths, df, split)

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
    )
    model = model.to(device)
    dataset_info["dataset_sample_counts"] = {
        "train": int(len(train_ds)),
        "val": int(len(val_ds)),
        "test": int(len(test_ds)),
    }
    dataset_info["mode"] = mode

    # train loader만 weighted sampler를 사용하고, val / test는 원본 분포를 유지한다.
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

    # sampler와 focal loss를 함께 써서 클래스 불균형 영향을 줄인다.
    alpha = compute_alpha(train_labels, num_classes=num_classes, device=device)
    criterion = FocalLoss(alpha=alpha, gamma=args.focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    # best validation loss 기준으로 early stopping과 checkpoint selection을 수행한다.
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

    # checkpoint에는 재로딩에 필요한 최소 실행 문맥도 함께 저장한다.
    ckpt_path = run_dir / "model.pt"
    torch.save(
        {
            "model_id": args.model_id,
            "model_state_dict": best_state,
            "class_names": class_names,
            "mode": mode,
            "seed": args.seed,
            "seq_len": args.seq_len,
            "seq_stride": args.seq_stride,
            "image_size": args.image_size,
        },
        ckpt_path,
    )

    pd.DataFrame(history).to_csv(run_dir / "train_history.csv", index=False)

    run_summary = {
        "model_id": args.model_id,
        "mode": mode,
        "device": str(device),
        "inputs": [str(p) for p in csv_paths],
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "optimizer": "AdamW",
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "focal_gamma": args.focal_gamma,
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
        "best_val_loss": float(best_val_loss),
        "epochs_ran": int(len(history)),
        "output_dir": str(run_dir),
        "metrics": metrics_summary,
    }

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
    parser.add_argument(
        "--csv-path",
        action="append",
        default=[],
        help="Input preprocessed CSV path. Repeat this option to pass multiple files.",
    )
    parser.add_argument("--output-root", type=str, default="model/model_evaluation/pipelines")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--focal-gamma", type=float, default=2.0)

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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.csv_path:
        args.csv_path = DEFAULT_INPUTS

    run(args)


if __name__ == "__main__":
    main()
