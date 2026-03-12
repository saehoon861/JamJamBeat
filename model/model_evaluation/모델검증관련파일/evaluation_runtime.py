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
- dataset_info.json
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
    """평가와 후처리 파라미터를 한 번에 전달하는 설정 컨테이너."""

    class_names: list[str]
    neutral_class_id: int = 0
    tau: float = 0.85
    vote_n: int = 7
    debounce_k: int = 3
    fallback_fps: float = 30.0
    dataset_info: dict | None = None


def _safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d else 0.0


def compute_confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    num_classes: int,
) -> np.ndarray:
    """예측/정답 시퀀스를 정수 confusion matrix로 누적한다."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def build_classification_report(
    cm: np.ndarray,
    class_names: list[str],
) -> tuple[pd.DataFrame, dict]:
    """confusion matrix에서 per-class precision / recall / F1과 macro 평균을 만든다."""
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


def _infer_run_label(output_path: Path) -> str:
    """evaluation 출력 경로에서 model_id / run timestamp를 추론한다."""
    try:
        eval_dir = output_path.parent
        run_dir = eval_dir.parent
        model_id = run_dir.parent.name
        run_id = run_dir.name
        return f"{model_id} | {run_id}"
    except Exception:
        return output_path.stem


def save_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
    accuracy: float,
    macro_f1: float,
    total_samples: int,
) -> None:
    """행 정규화 confusion matrix를 시각화해서 PNG로 저장한다."""
    row_sum = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)
    run_label = _infer_run_label(output_path)

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
        title=(
            "Normalized Confusion Matrix\n"
            f"{run_label} | acc={accuracy:.4f} | macro_f1={macro_f1:.4f} | n={total_samples}"
        ),
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            val = normalized[i, j]
            color = "white" if val > 0.5 else "black"
            count = int(cm[i, j])
            ax.text(
                j,
                i,
                f"{count}\n{val:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=7,
            )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _ts_to_seconds(ts: object) -> float | None:
    """timestamp 컬럼을 초 단위 실수로 파싱한다.

    지원 형태:
    - float / int
    - MM:SS
    - MM:SS:ms
    """
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

    # evaluate_predictions는 frame-level 예측을 받아 실서비스와 비슷한 trigger 단위로 다시 묶는다.
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

        # tie가 나면 class id가 작은 쪽을 택해 postprocess를 deterministic하게 유지한다.
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
    """neutral 구간에서 발생한 오발동(trigger)을 분당 횟수로 환산한다."""
    if "gesture" not in df.columns:
        return {"fp_per_min": None, "fp_count": 0, "duration_min": 0.0, "reason": "gesture column missing"}

    # neutral slice만 따로 떼어 실제 '아무 동작이 없어야 하는 구간'에서의 오동작을 본다.
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
            # timestamp 품질이 낮으면 fps 기반 길이 추정으로 fallback 한다.
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
    """latency 벡터에서 비교용 백분위 통계를 계산한다."""
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


def _format_ms(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}ms"


def save_latency_cdf_plot(
    latency_ms: np.ndarray,
    output_path: Path,
    latency_metrics: dict,
) -> None:
    """latency 분포를 CDF 형태로 저장해 모델 간 tail latency를 비교한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_label = _infer_run_label(output_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    if latency_ms.size == 0:
        ax.text(
            0.5,
            0.5,
            f"{run_label}\nNo latency data",
            ha="center",
            va="center",
        )
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
        ax.set_title(f"End-to-End Latency CDF\n{run_label}")
        stats_text = (
            f"count={latency_metrics.get('count', 0)}\n"
            f"mean={_format_ms(latency_metrics.get('mean_ms'))}\n"
            f"p50={_format_ms(latency_metrics.get('p50_ms'))}\n"
            f"p95={_format_ms(latency_metrics.get('p95_ms'))}\n"
            f"p99={_format_ms(latency_metrics.get('p99_ms'))}"
        )
        ax.text(
            0.98,
            0.02,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
        )
        ax.legend(fontsize=8)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_predictions(
    df_preds: pd.DataFrame,
    output_dir: str | Path,
    config: EvaluationConfig | None = None,
) -> dict:
    """예측 원본 CSV에서 모든 평가 산출물을 생성하는 메인 진입점."""
    cfg = config or EvaluationConfig(class_names=DEFAULT_CLASS_NAMES)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_info = cfg.dataset_info or {}

    required = {"gesture", "pred_class", "p_max"}
    missing = required - set(df_preds.columns)
    if missing:
        raise ValueError(f"Missing required prediction columns: {sorted(missing)}")

    y_true = df_preds["gesture"].to_numpy(dtype=np.int64)
    y_pred = df_preds["pred_class"].to_numpy(dtype=np.int64)

    # 기본 분류 지표는 raw frame prediction 기준으로 계산한다.
    num_classes = len(cfg.class_names)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    report_df, base_summary = build_classification_report(cm, cfg.class_names)

    cm_df = pd.DataFrame(cm, index=cfg.class_names, columns=cfg.class_names)
    cm_df.to_csv(out_dir / "confusion_matrix.csv")
    report_df.to_csv(out_dir / "per_class_report.csv", index=False)
    save_confusion_matrix_plot(
        cm,
        cfg.class_names,
        out_dir / "confusion_matrix.png",
        accuracy=base_summary["accuracy"],
        macro_f1=base_summary["macro_avg"]["f1"],
        total_samples=base_summary["total_samples"],
    )

    # class0 관련 메트릭은 neutral 축에서의 누락 / 오발동을 따로 보기 위한 지표다.
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
    save_latency_cdf_plot(latency_arr, out_dir / "latency_cdf.png", latency_metrics=latency_metrics)

    # 최종 summary는 run_summary.json에 바로 들어갈 수 있는 compact 구조만 유지한다.
    summary = {
        "dataset_info": dataset_info,
        "accuracy": base_summary["accuracy"],
        "macro_avg": base_summary["macro_avg"],
        "total_samples": base_summary["total_samples"],
        "class0_metrics": class0_metrics,
        "fp_per_min_metrics": fp_min_metrics,
        "latency": latency_metrics,
    }

    with (out_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    with (out_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


__all__ = [
    "DEFAULT_CLASS_NAMES",
    "EvaluationConfig",
    "evaluate_predictions",
]
