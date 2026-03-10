#!/usr/bin/env python3
"""
Runtime evaluation utilities for JamJamBeat model experiments.

This module implements the metrics described in:
- model/model_evaluation/evaluation_guide.md
- model/model_evaluation/classifier_output_and_visualization.md

Outputs (per experiment run):
- confusion_matrix.csv
- per_class_report.csv
- confusion_matrix.png
- latency_cdf.png
- metrics_summary.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CLASS_NAMES = [
    "neutral",
    "fist",
    "open_palm",
    "V",
    "pinky",
    "animal",
    "k-heart",
]


@dataclass
class EvaluationConfig:
    class_names: list[str]
    neutral_class_id: int = 0
    tau: float = 0.85
    vote_n: int = 7
    debounce_k: int = 3
    fallback_fps: float = 30.0


def _safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d else 0.0


def compute_confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    num_classes: int,
) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def build_classification_report(
    cm: np.ndarray,
    class_names: list[str],
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []

    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)

        rows.append(
            {
                "class": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(support),
            }
        )

    report_df = pd.DataFrame(rows)

    macro = {
        "precision": float(report_df["precision"].mean()),
        "recall": float(report_df["recall"].mean()),
        "f1": float(report_df["f1"].mean()),
    }

    total = int(cm.sum())
    accuracy = _safe_div(np.trace(cm), total)

    summary = {
        "accuracy": accuracy,
        "macro_avg": macro,
        "total_samples": total,
    }
    return report_df, summary


def save_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    row_sum = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Normalized Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            val = normalized[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _ts_to_seconds(ts: object) -> float | None:
    if ts is None:
        return None
    if isinstance(ts, (int, float, np.integer, np.floating)):
        return float(ts)

    text = str(ts).strip()
    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        pass

    parts = text.split(":")
    try:
        if len(parts) == 3:
            mm, ss, ms = parts
            return int(mm) * 60 + int(ss) + int(ms) / 1000.0
        if len(parts) == 2:
            mm, ss = parts
            return int(mm) * 60 + float(ss)
    except ValueError:
        return None

    return None


def postprocess_events(
    df: pd.DataFrame,
    tau: float,
    vote_n: int,
    debounce_k: int,
    neutral_class_id: int,
) -> pd.DataFrame:
    """
    Threshold -> voting -> debounce pipeline.
    Returns detected trigger events.
    """
    from collections import deque

    events: list[dict] = []
    window: deque[int] = deque(maxlen=vote_n)
    debounce_count = 0
    last_trigger = neutral_class_id

    for _, row in df.iterrows():
        pred_raw = int(row["pred_class"])
        p_max = float(row["p_max"])
        pred = pred_raw if p_max >= tau else neutral_class_id

        window.append(pred)
        if len(window) < vote_n:
            continue

        # deterministic majority vote (smallest class id wins tie)
        vote_counts = {}
        for cls_id in window:
            vote_counts[cls_id] = vote_counts.get(cls_id, 0) + 1
        voted = sorted(vote_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]

        if voted != neutral_class_id:
            debounce_count += 1
            if debounce_count >= debounce_k and voted != last_trigger:
                events.append(
                    {
                        "frame_idx": row.get("frame_idx", None),
                        "timestamp": row.get("timestamp", None),
                        "triggered_class": int(voted),
                        "p_max": p_max,
                    }
                )
                last_trigger = int(voted)
        else:
            debounce_count = 0
            last_trigger = neutral_class_id

    return pd.DataFrame(events)


def calc_fp_per_min(
    df: pd.DataFrame,
    tau: float,
    vote_n: int,
    debounce_k: int,
    neutral_class_id: int,
    fallback_fps: float = 30.0,
) -> dict:
    if "gesture" not in df.columns:
        return {"fp_per_min": None, "fp_count": 0, "duration_min": 0.0, "reason": "gesture column missing"}

    neutral_df = df[df["gesture"] == neutral_class_id].copy()
    if neutral_df.empty:
        return {"fp_per_min": None, "fp_count": 0, "duration_min": 0.0, "reason": "neutral slice empty"}

    triggers = postprocess_events(
        neutral_df,
        tau=tau,
        vote_n=vote_n,
        debounce_k=debounce_k,
        neutral_class_id=neutral_class_id,
    )

    if "timestamp" in neutral_df.columns:
        sec = neutral_df["timestamp"].map(_ts_to_seconds)
        if sec.notna().sum() >= 2:
            duration_sec = float(sec.max() - sec.min())
        else:
            duration_sec = float(len(neutral_df) / max(fallback_fps, 1e-6))
    else:
        duration_sec = float(len(neutral_df) / max(fallback_fps, 1e-6))

    duration_min = max(duration_sec / 60.0, 1e-9)
    fp_count = int(len(triggers[triggers.get("triggered_class", 0) != neutral_class_id])) if not triggers.empty else 0
    fp_per_min = float(fp_count / duration_min)

    return {
        "fp_per_min": fp_per_min,
        "fp_count": fp_count,
        "duration_min": duration_min,
        "trigger_count_total": int(len(triggers)),
    }


def latency_summary(latency_ms: np.ndarray) -> dict:
    if latency_ms.size == 0:
        return {
            "count": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "std_ms": None,
        }
    return {
        "count": int(latency_ms.size),
        "mean_ms": float(np.mean(latency_ms)),
        "p50_ms": float(np.percentile(latency_ms, 50)),
        "p95_ms": float(np.percentile(latency_ms, 95)),
        "p99_ms": float(np.percentile(latency_ms, 99)),
        "std_ms": float(np.std(latency_ms)),
    }


def save_latency_cdf_plot(latency_ms: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    if latency_ms.size == 0:
        ax.text(0.5, 0.5, "No latency data", ha="center", va="center")
        ax.set_axis_off()
    else:
        x = np.sort(latency_ms)
        y = np.arange(1, len(x) + 1) / len(x)

        p50 = np.percentile(x, 50)
        p95 = np.percentile(x, 95)
        p99 = np.percentile(x, 99)

        ax.plot(x, y, linewidth=2)
        for p, label in [(p50, "p50"), (p95, "p95"), (p99, "p99")]:
            ax.axvline(p, linestyle="--", label=f"{label}={p:.2f}ms")

        ax.axvline(200.0, linestyle="-", color="red", alpha=0.5, label="target 200ms")
        ax.set_xlabel("Total Latency (ms)")
        ax.set_ylabel("CDF")
        ax.set_title("End-to-End Latency CDF")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_predictions(
    df_preds: pd.DataFrame,
    output_dir: str | Path,
    config: EvaluationConfig | None = None,
) -> dict:
    cfg = config or EvaluationConfig(class_names=DEFAULT_CLASS_NAMES)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    required = {"gesture", "pred_class", "p_max"}
    missing = required - set(df_preds.columns)
    if missing:
        raise ValueError(f"Missing required prediction columns: {sorted(missing)}")

    y_true = df_preds["gesture"].to_numpy(dtype=np.int64)
    y_pred = df_preds["pred_class"].to_numpy(dtype=np.int64)

    num_classes = len(cfg.class_names)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    report_df, base_summary = build_classification_report(cm, cfg.class_names)

    cm_df = pd.DataFrame(cm, index=cfg.class_names, columns=cfg.class_names)
    cm_df.to_csv(out_dir / "confusion_matrix.csv")
    report_df.to_csv(out_dir / "per_class_report.csv", index=False)
    save_confusion_matrix_plot(cm, cfg.class_names, out_dir / "confusion_matrix.png")

    class0_fp = int(cm[1:, cfg.neutral_class_id].sum()) if num_classes > 1 else 0
    class0_fn = int(cm[cfg.neutral_class_id, 1:].sum()) if num_classes > 1 else 0
    total_non0 = int(cm[1:, :].sum()) if num_classes > 1 else 0
    total_0 = int(cm[cfg.neutral_class_id, :].sum())

    class0_metrics = {
        "false_positive_rate": _safe_div(class0_fp, total_non0),
        "false_negative_rate": _safe_div(class0_fn, total_0),
        "false_positive_count": class0_fp,
        "false_negative_count": class0_fn,
    }

    fp_min_metrics = calc_fp_per_min(
        df_preds,
        tau=cfg.tau,
        vote_n=cfg.vote_n,
        debounce_k=cfg.debounce_k,
        neutral_class_id=cfg.neutral_class_id,
        fallback_fps=cfg.fallback_fps,
    )

    latency_arr = np.array([], dtype=np.float32)
    if "latency_total_ms" in df_preds.columns:
        latency_arr = df_preds["latency_total_ms"].dropna().to_numpy(dtype=np.float32)
    latency_metrics = latency_summary(latency_arr)
    save_latency_cdf_plot(latency_arr, out_dir / "latency_cdf.png")

    summary = {
        "accuracy": base_summary["accuracy"],
        "macro_avg": base_summary["macro_avg"],
        "total_samples": base_summary["total_samples"],
        "class0_metrics": class0_metrics,
        "fp_per_min_metrics": fp_min_metrics,
        "latency": latency_metrics,
    }

    with (out_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


__all__ = [
    "DEFAULT_CLASS_NAMES",
    "EvaluationConfig",
    "evaluate_predictions",
]
