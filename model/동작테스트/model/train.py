# train.py - mlp_sequence_delta 독립 학습 스크립트 (동작테스트 전용)
from __future__ import annotations

import argparse
import copy
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import build as build_datasets
from model import SequenceDeltaMLP

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "runs"
DEFAULT_TRAIN_CSV = ROOT / "pos_scale_train.csv"


# ---------------------------------------------------------------------------
# 재현성
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """쉬운 샘플의 기여를 줄여 소수 클래스 학습을 강화하는 loss."""

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
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
            logits, targets, reduction="none", label_smoothing=self.label_smoothing
        )
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        weight = (1.0 - pt) ** self.gamma
        if self.alpha is not None:
            weight = weight * self.alpha[targets]
        return (weight * ce).mean()


def _class_alpha(labels: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    return torch.tensor(inv / inv.sum(), dtype=torch.float32, device=device)


def build_criterion(
    args: argparse.Namespace, labels: np.ndarray, device: torch.device
) -> nn.Module:
    alpha = _class_alpha(labels, args.num_classes, device) if args.use_alpha else None
    if args.loss_type == "focal":
        return FocalLoss(alpha=alpha, gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    return nn.CrossEntropyLoss(weight=alpha, label_smoothing=args.label_smoothing)


def create_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    classes, counts = np.unique(labels, return_counts=True)
    class_w = {int(c): 1.0 / max(int(cnt), 1) for c, cnt in zip(classes, counts)}
    weights = np.array([class_w[int(y)] for y in labels], dtype=np.float64)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ---------------------------------------------------------------------------
# 학습 / 검증 루프
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += int((logits.argmax(1) == y).sum().item())
        total_count += bs
    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += int((logits.argmax(1) == y).sum().item())
        total_count += bs
    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="mlp_sequence_delta 학습 (동작테스트 전용)")

    # 데이터
    parser.add_argument("--train-csv",   type=str, default=str(DEFAULT_TRAIN_CSV), help="학습 CSV 경로")
    parser.add_argument("--val-csv",     type=str, default=None,  help="검증 CSV 경로 (없으면 early stopping 비활성화)")
    parser.add_argument("--test-csv",    type=str, default=None,  help="테스트 CSV 경로 (없으면 test 평가 생략)")
    parser.add_argument("--num-classes", type=int, default=8,
                        help="분류 클래스 수 (기본 8: neutral+기존6+grab)")
    parser.add_argument("--seq-len",     type=int, default=8)
    parser.add_argument("--seq-stride",  type=int, default=2)

    # 학습
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--batch-size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience",     type=int,   default=6)

    # Loss / 샘플링
    parser.add_argument("--loss-type", type=str, default="cross_entropy",
                        choices=["cross_entropy", "focal"])
    parser.add_argument("--use-alpha", action=argparse.BooleanOptionalAction, default=True,
                        help="클래스 불균형 보정 alpha 사용 여부")
    parser.add_argument("--use-weighted-sampler", action=argparse.BooleanOptionalAction, default=True,
                        help="WeightedRandomSampler 사용 여부")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--focal-gamma",     type=float, default=2.0)

    # 실행 환경
    parser.add_argument("--device",      type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir",  type=str, default=str(DEFAULT_OUT))

    args = parser.parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"device: {device}")

    # 데이터셋 — dataset.py의 build() 구현 후 동작
    train_ds, val_ds, test_ds = build_datasets(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        seq_len=args.seq_len,
        seq_stride=args.seq_stride,
    )
    print(
        f"dataset | train={len(train_ds)}"
        + (f" val={len(val_ds)}" if val_ds is not None else " val=None")
        + (f" test={len(test_ds)}" if test_ds is not None else " test=None")
    )

    train_labels = np.asarray(train_ds.y)
    sampler   = create_weighted_sampler(train_labels) if args.use_weighted_sampler else None
    criterion = build_criterion(args, train_labels, device)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=sampler, shuffle=(sampler is None),
        num_workers=args.num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=pin,
    ) if val_ds is not None else None
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=pin,
    ) if test_ds is not None else None

    # 모델 — input_dim=126 고정 (joint 63 + delta 63)
    model = SequenceDeltaMLP(
        seq_len=args.seq_len, input_dim=126, num_classes=args.num_classes
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_state    = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    stale         = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        row: dict = {
            "epoch":      epoch,
            "train_loss": round(tr_loss, 6),
            "train_acc":  round(tr_acc,  6),
            "lr":         optimizer.param_groups[0]["lr"],
        }
        log_line = f"[{epoch:03d}/{args.epochs}] tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f}"

        if val_loader is not None:
            va_loss, va_acc = validate_one_epoch(model, val_loader, criterion, device)
            row["val_loss"] = round(va_loss, 6)
            row["val_acc"]  = round(va_acc,  6)
            log_line += f" | val_loss={va_loss:.4f} val_acc={va_acc:.4f}"

            if va_loss < best_val_loss:
                best_val_loss = va_loss
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    print(log_line)
                    print(f"early stopping at epoch {epoch}")
                    history.append(row)
                    break
        else:
            # val 없으면 마지막 epoch 가중치를 best로 사용
            best_state = copy.deepcopy(model.state_dict())

        history.append(row)
        print(log_line)

    # test 평가 (test_csv 지정 시에만)
    model.load_state_dict(best_state)
    if test_loader is not None:
        te_loss, te_acc = validate_one_epoch(model, test_loader, criterion, device)
        print(f"test | loss={te_loss:.4f} acc={te_acc:.4f}")
    else:
        te_loss, te_acc = None, None
        print("test 생략 (--test-csv 미지정)")

    # 저장
    KST = timezone(timedelta(hours=9))
    ts = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": best_state,
            "seq_len":     args.seq_len,
            "input_dim":   126,
            "num_classes": args.num_classes,
            "mode":        "sequence",
        },
        out_dir / "model.pt",
    )

    import pandas as pd
    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)

    summary = {
        "best_val_loss": best_val_loss if val_loader is not None else None,
        "test_loss":     te_loss,
        "test_acc":      te_acc,
        "epochs_ran":    len(history),
        "hyperparameters": vars(args),
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved → {out_dir}")


if __name__ == "__main__":
    main()
