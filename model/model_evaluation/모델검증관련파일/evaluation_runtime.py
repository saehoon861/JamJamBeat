#!/usr/bin/env python3
"""
Runtime evaluation utilities for JamJamBeat model experiments.

This module implements the metrics described in:
- model/model_evaluation/evaluation_guide.md
- model/model_evaluation/classifier_output_and_visualization.md

Outputs (per experiment run):
- confusion_matrix.csv
- per_class_report.csv
- per_class_summary.csv
- per_class_misclassifications.csv
- per_class_misclassifications.png
- per_test_case_report.csv
- per_test_case_misclassifications.csv
- per_image_pose_case_report.csv
- per_image_pose_case_misclassifications.csv
- confusion_matrix.png
- test_case_accuracy.png
- test_case_confusion.png
- image_pose_case_accuracy.png
- image_pose_case_confusion.png
- latency_cdf.png
- metrics_summary.json
- dataset_info.json
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IMAGE_DATASET_ROOT = PROJECT_ROOT / "model" / "data_fusion" / "추론용데이터셋"

DEFAULT_CLASS_NAMES = [
    "neutral",
    "fist",
    "open_palm",
    "V",
    "pinky",
    "animal",
    "k-heart",
]

POSE_CASE_LABELS = [
    "BaseP",
    "RollP",
    "PitchP",
    "YawP",
    "NoneNetural",
    "NoneOther",
]
POSE_CASE_ALIASES = {
    "BaseP": ("basep",),
    "RollP": ("rollp",),
    "PitchP": ("pitchp",),
    "YawP": ("yawp",),
    "NoneNetural": ("nonenetural", "noneneutral", "noneneutral", "nonenetural"),
    "NoneOther": ("noneother",),
}
POSE_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


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


def _class_name(class_names: list[str], class_id: int | None) -> str | None:
    if class_id is None:
        return None
    if 0 <= int(class_id) < len(class_names):
        return class_names[int(class_id)]
    return str(class_id)


def _cast_nullable_int_columns(df: pd.DataFrame) -> pd.DataFrame:
    converted = df.copy()
    target_columns = {
        "count",
        "support",
        "total_samples",
        "correct_samples",
        "incorrect_samples",
        "top_misclassified_count",
        "no_hand_count",
    }
    for column in converted.columns:
        if column.endswith("_id") or column in target_columns:
            try:
                converted[column] = converted[column].astype("Int64")
            except (TypeError, ValueError):
                continue
    return converted


def _top_misclassification_pair(
    counts: pd.Series,
    class_names: list[str],
    true_col: str,
    pred_col: str,
) -> dict[str, Any]:
    if counts.empty:
        return {
            "top_misclassified_true_class_id": None,
            "top_misclassified_true_class": None,
            "top_misclassified_pred_class_id": None,
            "top_misclassified_pred_class": None,
            "top_misclassified_count": 0,
        }

    top_key = counts.idxmax()
    if not isinstance(top_key, tuple):
        top_key = (top_key,)

    mapping = {
        "top_misclassified_count": int(counts.loc[top_key]),
    }
    if len(top_key) >= 1:
        mapping["top_misclassified_true_class_id"] = int(top_key[0])
        mapping["top_misclassified_true_class"] = _class_name(class_names, int(top_key[0]))
    else:
        mapping["top_misclassified_true_class_id"] = None
        mapping["top_misclassified_true_class"] = None
    if len(top_key) >= 2:
        mapping["top_misclassified_pred_class_id"] = int(top_key[1])
        mapping["top_misclassified_pred_class"] = _class_name(class_names, int(top_key[1]))
    else:
        mapping["top_misclassified_pred_class_id"] = None
        mapping["top_misclassified_pred_class"] = None
    return mapping


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


def build_per_class_summary(
    cm: np.ndarray,
    report_df: pd.DataFrame,
    class_names: list[str],
    no_hand_by_class: pd.Series | None = None,
) -> pd.DataFrame:
    """클래스별 총 샘플/정답 수와 대표 오분류 방향을 함께 정리한다."""
    report_lookup = report_df.set_index("class")
    rows: list[dict[str, Any]] = []

    for class_id, class_name in enumerate(class_names):
        total_samples = int(cm[class_id, :].sum())
        correct_samples = int(cm[class_id, class_id])
        incorrect_samples = total_samples - correct_samples
        mis_row = cm[class_id, :].copy()
        mis_row[class_id] = 0
        if mis_row.sum() > 0:
            top_pred_class_id = int(np.argmax(mis_row))
            top_pred_class = class_names[top_pred_class_id]
            top_pred_count = int(mis_row[top_pred_class_id])
        else:
            top_pred_class_id = None
            top_pred_class = None
            top_pred_count = 0

        report_row = report_lookup.loc[class_name]
        row = {
            "class_id": class_id,
            "class": class_name,
            "total_samples": total_samples,
            "correct_samples": correct_samples,
            "incorrect_samples": incorrect_samples,
            "accuracy": _safe_div(correct_samples, total_samples),
            "precision": float(report_row["precision"]),
            "recall": float(report_row["recall"]),
            "f1": float(report_row["f1"]),
            "support": int(report_row["support"]),
            "top_misclassified_pred_class_id": top_pred_class_id,
            "top_misclassified_pred_class": top_pred_class,
            "top_misclassified_count": top_pred_count,
        }
        if no_hand_by_class is not None:
            row["no_hand_count"] = int(no_hand_by_class.get(class_id, 0))
        rows.append(row)

    return _cast_nullable_int_columns(pd.DataFrame(rows))


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


def _normalize_group_values(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _extract_pose_case(image_name: object) -> str | None:
    if image_name is None:
        return None
    lowered = str(image_name).strip().lower()
    if not lowered:
        return None
    for canonical, aliases in POSE_CASE_ALIASES.items():
        for alias in aliases:
            if re.search(rf"(?:^|[_\-]){re.escape(alias)}(?:[_\-\.]|$)", lowered):
                return canonical
    return None


def _resolve_pose_case_from_test_images(
    df: pd.DataFrame,
    images_root: Path,
) -> tuple[pd.Series, dict[str, Any]]:
    """source_file + frame_idx로 추론용데이터셋 이미지를 역조회해 pose_case를 복원한다."""
    pose_case = pd.Series(index=df.index, dtype="object")
    metadata: dict[str, Any] = {
        "images_root": str(images_root),
        "generated": False,
        "matched_rows": 0,
        "unmatched_rows": int(len(df)),
    }

    if "source_file" not in df.columns:
        metadata["reason"] = "source_file column missing"
        return pose_case, metadata
    if "frame_idx" not in df.columns:
        metadata["reason"] = "frame_idx column missing"
        return pose_case, metadata
    if not images_root.exists():
        metadata["reason"] = f"images_root not found: {images_root}"
        return pose_case, metadata

    working = df[["source_file", "frame_idx"]].copy()
    working["source_file"] = _normalize_group_values(working["source_file"])
    if working["source_file"].eq("").all():
        metadata["reason"] = "source_file column empty"
        return pose_case, metadata

    frame_numeric = pd.to_numeric(working["frame_idx"], errors="coerce")
    valid_rows = working["source_file"].ne("") & frame_numeric.notna()
    if not valid_rows.any():
        metadata["reason"] = "frame_idx values could not be parsed"
        return pose_case, metadata

    working = working.loc[valid_rows].copy()
    working["frame_idx"] = frame_numeric.loc[valid_rows].astype(int)

    for source_file, source_df in working.groupby("source_file", sort=True):
        parts = str(source_file).split("_")
        if len(parts) < 4 or parts[1] != "test":
            continue

        gesture = parts[0]
        user = parts[-1]
        image_dir = images_root / user / gesture
        if not image_dir.exists():
            continue

        frame_to_pose: dict[int, str] = {}
        for frame_idx in sorted(source_df["frame_idx"].unique().tolist()):
            matches = sorted(image_dir.glob(f"{gesture}_{int(frame_idx)}_*_{user}.*"))
            for match in matches:
                if match.suffix.lower() not in POSE_IMAGE_SUFFIXES:
                    continue
                resolved = _extract_pose_case(match.name)
                if resolved is not None:
                    frame_to_pose[int(frame_idx)] = resolved
                    break

        if not frame_to_pose:
            continue

        for row_idx, frame_idx in zip(source_df.index, source_df["frame_idx"]):
            resolved = frame_to_pose.get(int(frame_idx))
            if resolved is not None:
                pose_case.at[row_idx] = resolved

    matched_rows = int(pose_case.notna().sum())
    metadata["matched_rows"] = matched_rows
    metadata["unmatched_rows"] = int(len(df) - matched_rows)
    metadata["generated"] = matched_rows > 0
    if matched_rows == 0:
        metadata["reason"] = "no pose_case matches found from source_file + frame_idx"
    return pose_case, metadata


def build_misclassification_table(
    df: pd.DataFrame,
    class_names: list[str],
    group_col: str | None = None,
) -> pd.DataFrame:
    """오분류만 long-form으로 집계해 사람이 읽기 쉬운 CSV를 만든다."""
    mis_df = df[df["gesture"] != df["pred_class"]].copy()
    if mis_df.empty:
        base_columns = []
        if group_col is not None:
            base_columns.append(group_col)
        base_columns.extend(
            [
                "true_class_id",
                "true_class",
                "pred_class_id",
                "pred_class",
                "count",
            ]
        )
        return pd.DataFrame(columns=base_columns)

    group_cols = ["gesture", "pred_class"] if group_col is None else [group_col, "gesture", "pred_class"]
    grouped = mis_df.groupby(group_cols, dropna=False).size().reset_index(name="count")
    rename_map = {"gesture": "true_class_id", "pred_class": "pred_class_id"}
    grouped = grouped.rename(columns=rename_map)
    grouped["true_class"] = grouped["true_class_id"].map(lambda v: _class_name(class_names, int(v)))
    grouped["pred_class"] = grouped["pred_class_id"].map(lambda v: _class_name(class_names, int(v)))
    ordered_cols = []
    if group_col is not None:
        ordered_cols.append(group_col)
    ordered_cols.extend(["true_class_id", "true_class", "pred_class_id", "pred_class", "count"])
    grouped = grouped[ordered_cols]
    sort_cols = ["count"]
    ascending = [False]
    if group_col is not None:
        sort_cols.insert(0, group_col)
        ascending.insert(0, True)
    grouped = grouped.sort_values(sort_cols, ascending=ascending, kind="stable").reset_index(drop=True)
    return _cast_nullable_int_columns(grouped)


def _series_mode_int(series: pd.Series) -> int | None:
    if series.empty:
        return None
    counts = series.value_counts(dropna=True)
    if counts.empty:
        return None
    return int(counts.index[0])


def build_group_report(
    df: pd.DataFrame,
    group_col: str,
    class_names: list[str],
) -> pd.DataFrame:
    """source_file / pose_case 같은 케이스 단위 성능 요약표를 만든다."""
    if group_col not in df.columns:
        return pd.DataFrame()

    working = df.copy()
    working[group_col] = _normalize_group_values(working[group_col])
    working = working[working[group_col] != ""].copy()
    if working.empty:
        return pd.DataFrame()

    has_status = "status" in working.columns
    rows: list[dict[str, Any]] = []

    for group_value, group_df in working.groupby(group_col, sort=True):
        total_samples = int(len(group_df))
        correct_samples = int((group_df["gesture"] == group_df["pred_class"]).sum())
        incorrect_samples = total_samples - correct_samples
        dominant_true_class_id = _series_mode_int(group_df["gesture"])
        mis_pair_counts = (
            group_df.loc[group_df["gesture"] != group_df["pred_class"], ["gesture", "pred_class"]]
            .value_counts(sort=True)
        )
        top_mis = _top_misclassification_pair(
            mis_pair_counts,
            class_names,
            true_col="gesture",
            pred_col="pred_class",
        )
        row = {
            group_col: str(group_value),
            "total_samples": total_samples,
            "correct_samples": correct_samples,
            "incorrect_samples": incorrect_samples,
            "accuracy": _safe_div(correct_samples, total_samples),
            "error_rate": _safe_div(incorrect_samples, total_samples),
            "dominant_true_class_id": dominant_true_class_id,
            "dominant_true_class": _class_name(class_names, dominant_true_class_id),
            **top_mis,
        }
        if has_status:
            row["no_hand_count"] = int((group_df["status"] == "no_hand").sum())
        rows.append(row)

    report_df = pd.DataFrame(rows)
    if report_df.empty:
        return report_df
    report_df = report_df.sort_values(["accuracy", group_col], ascending=[True, True], kind="stable").reset_index(drop=True)
    return _cast_nullable_int_columns(report_df)


def save_group_accuracy_plot(
    report_df: pd.DataFrame,
    group_col: str,
    output_path: Path,
    title: str,
) -> None:
    """케이스별 정확도를 수평 막대그래프로 저장한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * max(len(report_df), 1) + 1.5)))

    if report_df.empty:
        ax.text(0.5, 0.5, "No grouped report data", ha="center", va="center")
        ax.set_axis_off()
    else:
        plot_df = report_df.sort_values(["accuracy", group_col], ascending=[True, True], kind="stable")
        y_pos = np.arange(len(plot_df))
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(plot_df)))
        ax.barh(y_pos, plot_df["accuracy"], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df[group_col].astype(str))
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Accuracy")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        for idx, (_, row) in enumerate(plot_df.iterrows()):
            ax.text(
                min(float(row["accuracy"]) + 0.01, 0.98),
                idx,
                f"{int(row['correct_samples'])}/{int(row['total_samples'])}",
                va="center",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_group_misclassification_heatmap(
    mis_df: pd.DataFrame,
    row_col: str,
    class_names: list[str],
    output_path: Path,
    title: str,
) -> None:
    """케이스별 오분류를 예측 클래스 축으로 집계한 히트맵을 저장한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, max(4, 0.45 * max(len(mis_df[row_col].unique()) if not mis_df.empty else 1, 1) + 1.5)))

    if mis_df.empty:
        ax.text(0.5, 0.5, "No misclassifications", ha="center", va="center")
        ax.set_axis_off()
    else:
        pivot = (
            mis_df.pivot_table(index=row_col, columns="pred_class", values="count", aggfunc="sum", fill_value=0)
            .reindex(columns=class_names, fill_value=0)
            .sort_index()
        )
        matrix = pivot.to_numpy(dtype=float)
        im = ax.imshow(matrix, interpolation="nearest", cmap="Reds")
        ax.figure.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([str(v) for v in pivot.index])
        ax.set_xlabel("Predicted class (misclassified only)")
        ax.set_ylabel(row_col)
        ax.set_title(title)
        if matrix.size <= 400:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    value = int(matrix[i, j])
                    if value == 0:
                        continue
                    color = "white" if matrix[i, j] > matrix.max() * 0.5 else "black"
                    ax.text(j, i, str(value), ha="center", va="center", fontsize=8, color=color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_per_class_misclassification_heatmap(
    mis_df: pd.DataFrame,
    class_names: list[str],
    output_path: Path,
    title: str,
) -> None:
    """클래스 간 오분류 방향만 따로 보여주는 heatmap을 저장한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))

    if mis_df.empty:
        ax.text(0.5, 0.5, "No misclassifications", ha="center", va="center")
        ax.set_axis_off()
    else:
        pivot = (
            mis_df.pivot_table(
                index="true_class",
                columns="pred_class",
                values="count",
                aggfunc="sum",
                fill_value=0,
            )
            .reindex(index=class_names, columns=class_names, fill_value=0)
        )
        matrix = pivot.to_numpy(dtype=float)
        np.fill_diagonal(matrix, 0.0)
        im = ax.imshow(matrix, interpolation="nearest", cmap="Reds")
        ax.figure.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted class (misclassified only)")
        ax.set_ylabel("True class")
        ax.set_title(title)
        if matrix.size <= 400:
            vmax = matrix.max() if matrix.size else 0.0
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    value = int(matrix[i, j])
                    if value == 0:
                        continue
                    color = "white" if vmax > 0 and matrix[i, j] > vmax * 0.5 else "black"
                    ax.text(j, i, str(value), ha="center", va="center", fontsize=8, color=color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _has_status_no_hand(df: pd.DataFrame) -> bool:
    return "status" in df.columns and (df["status"] == "no_hand").any()


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
    dataset_info = dict(cfg.dataset_info or {})
    artifacts: dict[str, dict[str, Any]] = {}

    required = {"gesture", "pred_class", "p_max"}
    missing = required - set(df_preds.columns)
    if missing:
        raise ValueError(f"Missing required prediction columns: {sorted(missing)}")

    working_df = df_preds.copy()
    y_true = working_df["gesture"].to_numpy(dtype=np.int64)
    y_pred = working_df["pred_class"].to_numpy(dtype=np.int64)

    # 기본 분류 지표는 raw frame prediction 기준으로 계산한다.
    num_classes = len(cfg.class_names)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    report_df, base_summary = build_classification_report(cm, cfg.class_names)
    no_hand_by_class = None
    if _has_status_no_hand(working_df):
        no_hand_by_class = (
            working_df.loc[working_df["status"] == "no_hand"]
            .groupby("gesture")
            .size()
            .astype(int)
        )
    per_class_summary_df = build_per_class_summary(cm, report_df, cfg.class_names, no_hand_by_class=no_hand_by_class)
    per_class_mis_df = build_misclassification_table(working_df, cfg.class_names)

    cm_df = pd.DataFrame(cm, index=cfg.class_names, columns=cfg.class_names)
    cm_df.to_csv(out_dir / "confusion_matrix.csv")
    report_df.to_csv(out_dir / "per_class_report.csv", index=False)
    per_class_summary_df.to_csv(out_dir / "per_class_summary.csv", index=False)
    per_class_mis_df.to_csv(out_dir / "per_class_misclassifications.csv", index=False)
    save_confusion_matrix_plot(
        cm,
        cfg.class_names,
        out_dir / "confusion_matrix.png",
        accuracy=base_summary["accuracy"],
        macro_f1=base_summary["macro_avg"]["f1"],
        total_samples=base_summary["total_samples"],
    )
    save_per_class_misclassification_heatmap(
        per_class_mis_df,
        cfg.class_names,
        out_dir / "per_class_misclassifications.png",
        title="Per-Class Misclassifications",
    )
    artifacts["per_class_summary"] = {
        "generated": True,
        "path": "per_class_summary.csv",
    }
    artifacts["per_class_misclassifications"] = {
        "generated": True,
        "path": "per_class_misclassifications.csv",
        "plot_path": "per_class_misclassifications.png",
    }

    source_file_series = None
    if "source_file" in working_df.columns:
        source_file_series = _normalize_group_values(working_df["source_file"])
        if (source_file_series != "").any():
            working_df["source_file"] = source_file_series
            per_test_case_report_df = build_group_report(working_df, "source_file", cfg.class_names)
            per_test_case_mis_df = build_misclassification_table(working_df, cfg.class_names, group_col="source_file")
            per_test_case_report_df.to_csv(out_dir / "per_test_case_report.csv", index=False)
            per_test_case_mis_df.to_csv(out_dir / "per_test_case_misclassifications.csv", index=False)
            save_group_accuracy_plot(
                per_test_case_report_df,
                "source_file",
                out_dir / "test_case_accuracy.png",
                title="Per-Test-Case Accuracy",
            )
            save_group_misclassification_heatmap(
                per_test_case_mis_df,
                "source_file",
                cfg.class_names,
                out_dir / "test_case_confusion.png",
                title="Per-Test-Case Misclassifications",
            )
            artifacts["per_test_case"] = {
                "generated": True,
                "group_key": "source_file",
                "report_path": "per_test_case_report.csv",
                "misclassifications_path": "per_test_case_misclassifications.csv",
                "accuracy_plot_path": "test_case_accuracy.png",
                "confusion_plot_path": "test_case_confusion.png",
            }
        else:
            artifacts["per_test_case"] = {
                "generated": False,
                "group_key": "source_file",
                "reason": "source_file column empty",
            }
    else:
        artifacts["per_test_case"] = {
            "generated": False,
            "group_key": "source_file",
            "reason": "source_file column missing",
        }

    pose_case_series = None
    pose_case_metadata: dict[str, Any]
    if "image_name" in working_df.columns:
        pose_case_series = working_df["image_name"].map(_extract_pose_case)
        pose_case_metadata = {
            "images_root": None,
            "generated": bool(pose_case_series.notna().any()),
            "matched_rows": int(pose_case_series.notna().sum()),
            "unmatched_rows": int(len(working_df) - pose_case_series.notna().sum()),
        }
        if not pose_case_metadata["generated"]:
            pose_case_metadata["reason"] = "image_name column present but no supported pose tokens found"
    else:
        pose_case_series, pose_case_metadata = _resolve_pose_case_from_test_images(
            working_df,
            DEFAULT_IMAGE_DATASET_ROOT,
        )

    if pose_case_series is not None and pose_case_series.notna().any():
        working_df["pose_case"] = pose_case_series
        pose_df = working_df[working_df["pose_case"].notna()].copy()
        per_pose_report_df = build_group_report(pose_df, "pose_case", cfg.class_names)
        per_pose_mis_df = build_misclassification_table(pose_df, cfg.class_names, group_col="pose_case")
        per_pose_report_df.to_csv(out_dir / "per_image_pose_case_report.csv", index=False)
        per_pose_mis_df.to_csv(out_dir / "per_image_pose_case_misclassifications.csv", index=False)
        save_group_accuracy_plot(
            per_pose_report_df,
            "pose_case",
            out_dir / "image_pose_case_accuracy.png",
            title="Per-Image Pose-Case Accuracy",
        )
        save_group_misclassification_heatmap(
            per_pose_mis_df,
            "pose_case",
            cfg.class_names,
            out_dir / "image_pose_case_confusion.png",
            title="Per-Image Pose-Case Misclassifications",
        )
        artifacts["per_image_pose_case"] = {
            "generated": True,
            "group_key": "pose_case",
            "report_path": "per_image_pose_case_report.csv",
            "misclassifications_path": "per_image_pose_case_misclassifications.csv",
            "accuracy_plot_path": "image_pose_case_accuracy.png",
            "confusion_plot_path": "image_pose_case_confusion.png",
            "labels": POSE_CASE_LABELS,
            "images_root": pose_case_metadata.get("images_root"),
            "matched_rows": pose_case_metadata.get("matched_rows"),
            "unmatched_rows": pose_case_metadata.get("unmatched_rows"),
        }
    else:
        artifacts["per_image_pose_case"] = {
            "generated": False,
            "group_key": "pose_case",
            "images_root": pose_case_metadata.get("images_root"),
            "matched_rows": pose_case_metadata.get("matched_rows"),
            "unmatched_rows": pose_case_metadata.get("unmatched_rows"),
            "reason": pose_case_metadata.get("reason", "pose_case data unavailable"),
        }

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

    if (
        dataset_info.get("temporal_metrics_policy") == "disabled_for_static_images"
        or dataset_info.get("test_kind") == "static_images_63d"
    ):
        fp_min_metrics = {
            "fp_per_min": None,
            "fp_count": 0,
            "duration_min": 0.0,
            "reason": "disabled_for_static_images",
        }
    else:
        fp_min_metrics = calc_fp_per_min(
            df_preds,
            tau=cfg.tau,
            vote_n=cfg.vote_n,
            debounce_k=cfg.debounce_k,
            neutral_class_id=cfg.neutral_class_id,
            fallback_fps=cfg.fallback_fps,
        )

    latency_arr = np.array([], dtype=np.float32)
    if "latency_total_ms" in working_df.columns:
        latency_arr = working_df["latency_total_ms"].dropna().to_numpy(dtype=np.float32)
    latency_metrics = latency_summary(latency_arr)
    save_latency_cdf_plot(latency_arr, out_dir / "latency_cdf.png", latency_metrics=latency_metrics)

    dataset_info["evaluation_artifacts"] = artifacts
    if _has_status_no_hand(working_df):
        dataset_info["no_hand_stats"] = {
            "enabled": True,
            "total_no_hand_count": int((working_df["status"] == "no_hand").sum()),
        }
    else:
        dataset_info["no_hand_stats"] = {
            "enabled": False,
            "reason": "status column missing or no no_hand rows",
        }

    # 최종 summary는 run_summary.json에 바로 들어갈 수 있는 compact 구조만 유지한다.
    summary = {
        "dataset_info": dataset_info,
        "accuracy": base_summary["accuracy"],
        "macro_avg": base_summary["macro_avg"],
        "total_samples": base_summary["total_samples"],
        "class0_metrics": class0_metrics,
        "fp_per_min_metrics": fp_min_metrics,
        "latency": latency_metrics,
        "evaluation_artifacts": artifacts,
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
