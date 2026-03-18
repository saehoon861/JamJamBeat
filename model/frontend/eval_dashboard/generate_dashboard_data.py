#!/usr/bin/env python3
"""generate_dashboard_data.py - Build browser-ready evaluation data for the JamJamBeat dashboard."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PIPELINES_ROOT = PROJECT_ROOT / "model" / "model_evaluation" / "pipelines"
LATEST_SUITE_PATH = PIPELINES_ROOT / "latest_suite.json"
RAW_VIDEO_ROOT = PROJECT_ROOT / "data" / "raw_data"
LANDMARK_ROOT = PROJECT_ROOT / "data" / "landmark_data"
APP_ROOT = Path(__file__).resolve().parent
DATA_ROOT = APP_ROOT / "data"
MODELS_ROOT = DATA_ROOT / "models"
SUITES_ROOT = DATA_ROOT / "suites"
VIDEO_METADATA_CACHE: dict[str, dict[str, Any]] = {}
SUITE_LANDMARK_DIRNAME = "landmarks"

DEFAULT_CLASS_NAMES = ["neutral", "fist", "open_palm", "V", "pinky", "animal", "k-heart"]

GESTURE_LABELS = {
    "0": "neutral",
    "1": "fist",
    "2": "open_palm",
    "3": "V",
    "4": "pinky",
    "5": "animal",
    "6": "k-heart",
}

MODEL_EXPLAINERS = {
    "mlp_original": {
        "family": "MLP",
        "headline": "가장 단순한 프레임 단위 MLP 베이스라인",
        "summary": "원본 프레임 feature를 바로 분류하는 구조다. 비교적 해석이 쉽고 추론 흐름을 추적하기 좋다.",
        "strengths": [
            "구조가 단순해서 실패 원인을 추적하기 쉽다.",
            "프레임 단위 반응이 빨라 데모나 디버깅용 기준선으로 좋다.",
        ],
        "tradeoffs": [
            "시간축 문맥을 거의 쓰지 않아 전환 구간에서 불안정할 수 있다.",
            "사용자/비디오 조건 변화에 더 민감할 수 있다.",
        ],
    },
    "mlp_baseline": {
        "family": "MLP",
        "headline": "기본 feature 조합으로 학습한 프레임 베이스라인",
        "summary": "현재 실험군 중 가장 비교 기준으로 쓰기 쉬운 프레임 모델이다.",
        "strengths": [
            "입력 구성이 직관적이라 feature 영향도를 설명하기 쉽다.",
            "여러 변형 모델과의 상대 비교 기준선 역할을 한다.",
        ],
        "tradeoffs": [
            "오탐률과 class-0 누락을 함께 봐야 한다.",
            "sequence 계열보다 문맥 활용 폭이 좁다.",
        ],
    },
    "mlp_baseline_seq8": {
        "family": "Sequence MLP",
        "headline": "8-step 시퀀스를 입력으로 쓰는 짧은 문맥 모델",
        "summary": "짧은 길이의 sequence로 동작 전환을 포착하려는 가벼운 temporal 모델이다.",
        "strengths": [
            "프레임 모델보다 시간축 문맥을 본다.",
            "비교적 짧은 sequence라 구현이 단순하다.",
        ],
        "tradeoffs": [
            "문맥 길이가 짧아 긴 gesture 흐름을 놓칠 수 있다.",
            "frame 대비 샘플 수가 줄어 데이터 효율이 바뀐다.",
        ],
    },
    "mlp_sequence_joint": {
        "family": "Sequence MLP",
        "headline": "joint 중심 temporal 시퀀스를 요약하는 MLP",
        "summary": "joint 흐름을 sequence로 묶어 문맥을 읽도록 한 temporal 변형이다.",
        "strengths": [
            "시퀀스 문맥을 간단한 구조로 확인할 수 있다.",
            "joint 변화가 큰 gesture에서 패턴을 보기 좋다.",
        ],
        "tradeoffs": [
            "길이 설정에 따라 성능이 크게 흔들릴 수 있다.",
            "문맥은 보지만 sequence 모델 중 표현력이 가장 강한 편은 아니다.",
        ],
    },
    "mlp_temporal_pooling": {
        "family": "Temporal Pooling",
        "headline": "sequence를 pooling해서 안정성을 높인 temporal MLP",
        "summary": "연속 프레임을 풀링해 노이즈를 덜고, 사용자별 전환 구간을 완화하려는 구조다.",
        "strengths": [
            "프레임 노이즈를 줄여 보다 안정적인 예측을 만들기 쉽다.",
            "latency와 안정성의 균형을 보기 좋다.",
        ],
        "tradeoffs": [
            "짧은 순간 gesture의 sharp한 피크를 희석할 수 있다.",
            "풀링 창 길이에 따라 반응 속도가 느려질 수 있다.",
        ],
    },
    "mlp_sequence_delta": {
        "family": "Sequence Delta",
        "headline": "프레임 차분 변화량을 강조한 temporal MLP",
        "summary": "절대 좌표보다 변화량에 더 집중해 gesture 전환을 잡으려는 모델이다.",
        "strengths": [
            "동작 변화량이 중요한 gesture에서 시그널을 강조할 수 있다.",
            "사용자 포즈 편차보다 상대 변화에 집중하기 쉽다.",
        ],
        "tradeoffs": [
            "정적인 구간이나 작은 움직임에 약할 수 있다.",
            "변화량 노이즈가 커지면 false trigger가 늘 수 있다.",
        ],
    },
    "mlp_embedding": {
        "family": "Embedding MLP",
        "headline": "현재 최신 suite에서 상위권 성능을 보인 프레임 모델",
        "summary": "임베딩 표현을 통해 프레임 feature를 더 잘 분리하는 구조다. 최신 suite 기준 높은 macro F1을 보였다.",
        "strengths": [
            "프레임 계열 중 성능이 높아 빠른 추론과 정확도를 함께 보기 좋다.",
            "UI 데모에서 대표 모델로 쓰기 좋다.",
        ],
        "tradeoffs": [
            "frame 기반이라 temporal smoothing이 필요할 수 있다.",
            "false positive와 confidence 분포를 같이 봐야 한다.",
        ],
    },
    "cnn1d_tcn": {
        "family": "Temporal CNN/TCN",
        "headline": "1D convolution과 temporal receptive field를 쓰는 sequence 모델",
        "summary": "시퀀스 패턴을 convolution으로 읽어 손동작 리듬과 전환을 포착하려는 temporal 모델이다.",
        "strengths": [
            "시간축 패턴을 효과적으로 압축해 볼 수 있다.",
            "sequence 계열에서 latency와 성능 균형을 볼 만하다.",
        ],
        "tradeoffs": [
            "설계가 복잡해지면 에러 해석이 쉬운 편은 아니다.",
            "짧은 sequence 노이즈에 민감할 수 있다.",
        ],
    },
    "transformer_embedding": {
        "family": "Transformer",
        "headline": "최신 suite의 최고 성능권 sequence 모델",
        "summary": "임베딩과 attention 기반 문맥 파악으로 사용자별 gesture 전환을 더 넓게 읽는 모델이다.",
        "strengths": [
            "긴 문맥과 관계를 읽는 데 강하다.",
            "최신 suite에서 강한 macro F1과 낮은 fp/min 조합을 보였다.",
        ],
        "tradeoffs": [
            "구조 설명과 디버깅 난이도가 높다.",
            "프레임 모델보다 해석 오버헤드가 크다.",
        ],
    },
}

DEFAULT_VISIBLE_MODELS = {
    "mlp_original",
    "mlp_baseline",
    "mlp_embedding",
    "mlp_baseline_seq8",
    "mlp_sequence_joint",
    "mlp_temporal_pooling",
    "mlp_sequence_delta",
    "cnn1d_tcn",
    "transformer_embedding",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def coerce_number(value: str | None) -> int | float | str | None:
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return ""
    try:
        if re.fullmatch(r"-?\d+", text):
            return int(text)
        return float(text)
    except ValueError:
        return text


def path_from_dashboard(relative_to_project: Path) -> str:
    parts = ["..", "..", ".."] + list(relative_to_project.parts)
    return "/".join(parts)


def path_from_absolute(path: Path) -> str | None:
    try:
        relative = path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return None
    return path_from_dashboard(relative)


def parse_source_file(source_file: str) -> dict[str, str]:
    parts = source_file.split("_")
    gesture_key = parts[0] if parts else "unknown"
    motion = parts[1] if len(parts) > 1 else "unknown"
    side = parts[2] if len(parts) > 2 else "unknown"
    user = "_".join(parts[3:]) if len(parts) > 3 else "unknown"
    return {
        "source_file": source_file,
        "gesture_key": gesture_key,
        "gesture_name": GESTURE_LABELS.get(gesture_key, gesture_key),
        "motion": motion,
        "side": side,
        "user": user,
        "video_path": path_from_dashboard(Path("data") / "raw_data" / f"{source_file}.mp4"),
        "landmark_path": path_from_dashboard(Path("data") / "landmark_data" / f"{source_file}.csv"),
        "landmark_exists": (LANDMARK_ROOT / f"{source_file}.csv").exists(),
    }


def parse_dataset_user(name: str) -> str:
    match = re.match(r"^(man\d+|woman\d+)", str(name))
    if match:
        return match.group(1)
    return str(name).split("_")[0] if name else "unknown"


def build_dataset_user_splits(dataset_info: dict[str, Any] | None) -> dict[str, list[str]]:
    if not dataset_info:
        return {}

    user_splits: dict[str, set[str]] = defaultdict(set)
    split_info = dataset_info.get("split") or {}
    for split_name, split_payload in split_info.items():
        for source_group in split_payload.get("source_groups") or []:
            user_splits[parse_dataset_user(str(source_group))].add(str(split_name))

    if not user_splits:
        for source_group in dataset_info.get("source_groups") or []:
            user_splits[parse_dataset_user(str(source_group))].add("dataset")

    return {user: sorted(splits) for user, splits in sorted(user_splits.items())}


def load_video_metadata(source_file: str) -> dict[str, Any]:
    if source_file in VIDEO_METADATA_CACHE:
        return VIDEO_METADATA_CACHE[source_file]

    video_path = RAW_VIDEO_ROOT / f"{source_file}.mp4"
    metadata = {
        "video_exists": video_path.exists(),
        "fps": 30.0,
        "total_frames": 0,
        "duration_sec": 0.0,
        "width": 0,
        "height": 0,
    }

    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if fps <= 0:
                fps = 30.0
            metadata = {
                "video_exists": True,
                "fps": round(fps, 4),
                "total_frames": total_frames,
                "duration_sec": round(total_frames / max(fps, 1e-6), 4),
                "width": width,
                "height": height,
            }
        cap.release()

    VIDEO_METADATA_CACHE[source_file] = metadata
    return metadata


def resolve_suite_input_csv_paths(suite_meta: dict[str, Any]) -> list[Path]:
    resolved: list[Path] = []
    for raw_path in suite_meta.get("input_csv_paths") or []:
        candidate = Path(str(raw_path))
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        if candidate.exists():
            resolved.append(candidate)
    return resolved


def merge_fieldnames(existing: list[str], incoming: list[str]) -> list[str]:
    merged = list(existing)
    for name in incoming:
        if name not in merged:
            merged.append(name)
    return merged


def write_suite_landmark_exports(suite_name: str, suite_output_dir: Path, suite_meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    export_rows: dict[str, dict[str, Any]] = {}
    output_dir = suite_output_dir / SUITE_LANDMARK_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in resolve_suite_input_csv_paths(suite_meta):
        csv_href = path_from_absolute(csv_path)
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            if not fieldnames or "source_file" not in fieldnames:
                continue

            for row in reader:
                source_file = str(row.get("source_file") or "").strip()
                if not source_file or str(row.get("x0") or "").strip() == "":
                    continue
                export_entry = export_rows.setdefault(
                    source_file,
                    {
                        "fieldnames": fieldnames,
                        "rows": [],
                        "dataset_csv_path": csv_href,
                    },
                )
                export_entry["fieldnames"] = merge_fieldnames(export_entry["fieldnames"], fieldnames)
                if csv_href and not export_entry.get("dataset_csv_path"):
                    export_entry["dataset_csv_path"] = csv_href
                export_entry["rows"].append(dict(row))

    exports: dict[str, dict[str, Any]] = {}
    for source_file, export_entry in export_rows.items():
        rows = sorted(
            export_entry["rows"],
            key=lambda row: (
                int(str(row.get("frame_idx") or "0") or 0),
                str(row.get("timestamp") or ""),
            ),
        )
        output_path = output_dir / f"{source_file}.csv"
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=export_entry["fieldnames"])
            writer.writeheader()
            writer.writerows(rows)
        exports[source_file] = {
            "landmark_path": f"data/suites/{suite_name}/{SUITE_LANDMARK_DIRNAME}/{source_file}.csv",
            "landmark_exists": True,
            "landmark_source_kind": "suite_exported_csv",
            "landmark_frame_count": len(rows),
            "dataset_csv_path": export_entry.get("dataset_csv_path"),
        }

    return exports


def build_suite_source_catalog(
    suite_meta: dict[str, Any],
    suite_landmark_exports: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    suite_landmark_exports = suite_landmark_exports or {}
    catalog: dict[str, dict[str, Any]] = {}
    for csv_path in resolve_suite_input_csv_paths(suite_meta):
        csv_href = path_from_absolute(csv_path)
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "source_file" not in reader.fieldnames:
                continue

            for row in reader:
                source_file = str(row.get("source_file") or "").strip()
                if not source_file:
                    continue

                source_entry = catalog.setdefault(
                    source_file,
                    {
                        **parse_source_file(source_file),
                        **load_video_metadata(source_file),
                        "dataset_csv_path": suite_landmark_exports.get(source_file, {}).get("dataset_csv_path") or csv_href,
                        "dataset_source_row_count": 0,
                        "landmark_frame_count": 0,
                        "landmark_source_kind": "dedicated_csv"
                        if (LANDMARK_ROOT / f"{source_file}.csv").exists()
                        else "unavailable",
                    },
                )
                source_entry["dataset_source_row_count"] += 1
                if csv_href and not source_entry.get("dataset_csv_path"):
                    source_entry["dataset_csv_path"] = csv_href

                has_landmark = str(row.get("x0") or "").strip() != ""
                if has_landmark:
                    source_entry["landmark_frame_count"] += 1
                    if source_entry.get("landmark_source_kind") != "dedicated_csv":
                        export_meta = suite_landmark_exports.get(source_file)
                        if export_meta:
                            source_entry["landmark_path"] = export_meta["landmark_path"]
                            source_entry["landmark_exists"] = True
                            source_entry["landmark_source_kind"] = export_meta["landmark_source_kind"]
                        elif csv_href:
                            source_entry["landmark_path"] = csv_href
                            source_entry["landmark_exists"] = True
                            source_entry["landmark_source_kind"] = "suite_input_csv"

    for source_file, export_meta in suite_landmark_exports.items():
        source_entry = catalog.get(source_file)
        if not source_entry or source_entry.get("landmark_source_kind") == "dedicated_csv":
            continue
        source_entry["landmark_path"] = export_meta["landmark_path"]
        source_entry["landmark_exists"] = True
        source_entry["landmark_source_kind"] = export_meta["landmark_source_kind"]
        source_entry["landmark_frame_count"] = export_meta["landmark_frame_count"]
        if export_meta.get("dataset_csv_path") and not source_entry.get("dataset_csv_path"):
            source_entry["dataset_csv_path"] = export_meta["dataset_csv_path"]

    return dict(sorted(catalog.items()))


def normalize_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append({key: coerce_number(value) for key, value in row.items()})
    return normalized


def build_confusion_matrix(path: Path) -> dict[str, Any]:
    rows = read_csv_rows(path)
    labels = []
    matrix = []
    for row in rows:
        row_label = str(row.get("") or row.get("class") or "")
        if row_label:
            labels.append(row_label)
        matrix.append([int(float(value or 0)) for key, value in row.items() if key not in {"", "class"}])
    return {"labels": labels, "matrix": matrix}


def build_model_explainer(model_id: str, mode: str, run_summary: dict[str, Any], comparison_row: dict[str, Any]) -> dict[str, Any]:
    base = MODEL_EXPLAINERS.get(
        model_id,
        {
            "family": "Custom",
            "headline": f"{model_id} 실험 모델",
            "summary": "현재 프로젝트의 실험 모델이다. run summary와 metrics를 함께 해석해야 한다.",
            "strengths": ["현재 suite에서의 상대 성능을 직접 확인할 수 있다."],
            "tradeoffs": ["구조 설명이 별도로 정리되지 않아 raw artifact를 함께 봐야 한다."],
        },
    )
    metrics = run_summary.get("metrics", {})
    inputs = run_summary.get("inputs", [])
    return {
        **base,
        "mode": mode,
        "inputs": inputs,
        "key_metrics": {
            "accuracy": comparison_row.get("accuracy"),
            "macro_f1": comparison_row.get("macro_f1"),
            "fp_per_min": comparison_row.get("fp_per_min"),
            "latency_p95_ms": comparison_row.get("latency_p95_ms"),
            "epochs_ran": run_summary.get("epochs_ran"),
            "best_val_loss": comparison_row.get("best_val_loss"),
        },
        "interpretation": [
            "macro_f1는 클래스별 균형 성능을 본다.",
            "fp_per_min은 neutral/hard-negative 구간에서 잘못 울리는 빈도를 본다.",
            "latency p95는 실제 배포 시 체감 상한선을 가늠하는 지표다.",
            "class0 오류율은 neutral/background 구간의 방어력을 확인할 때 중요하다.",
        ],
        "dataset_context": metrics.get("dataset_info") or run_summary.get("dataset_info") or {},
    }


def build_source_views(
    pred_rows: list[dict[str, Any]],
    class_names: list[str],
    suite_source_catalog: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pred_rows:
        source_file = str(row.get("source_file") or "unknown")
        by_source[source_file].append(row)

    source_summaries: list[dict[str, Any]] = []
    video_entries: list[dict[str, Any]] = []

    source_files = sorted(set((suite_source_catalog or {}).keys()) | set(by_source.keys()))

    for source_file in source_files:
        rows = by_source.get(source_file, [])
        catalog_entry = dict((suite_source_catalog or {}).get(source_file) or {})
        meta = parse_source_file(source_file)
        video_meta = load_video_metadata(source_file)
        merged_entry = {
            **meta,
            **video_meta,
            **catalog_entry,
        }
        mismatch_count = sum(1 for row in rows if row.get("gesture") != row.get("pred_class"))
        pred_counter = Counter(
            class_names[int(row["pred_class"])]
            if isinstance(row.get("pred_class"), int) and int(row["pred_class"]) < len(class_names)
            else str(row.get("pred_class"))
            for row in rows
        )
        accuracy = round((len(rows) - mismatch_count) / len(rows), 4) if rows else None

        event_preview = []
        for row in rows[:400]:
            pred_index = row.get("pred_class")
            gesture_index = row.get("gesture")
            pred_label = class_names[int(pred_index)] if isinstance(pred_index, int) and int(pred_index) < len(class_names) else str(pred_index)
            gesture_label = class_names[int(gesture_index)] if isinstance(gesture_index, int) and int(gesture_index) < len(class_names) else str(gesture_index)
            probabilities = []
            for idx, class_name in enumerate(class_names):
                prob = row.get(f"p{idx}")
                if isinstance(prob, (int, float)):
                    probabilities.append({"label": class_name, "value": round(float(prob), 4)})
            event_preview.append(
                {
                    "frame_idx": row.get("frame_idx"),
                    "timestamp": row.get("timestamp"),
                    "ground_truth": gesture_label,
                    "predicted": pred_label,
                    "confidence": row.get("p_max"),
                    "latency_total_ms": row.get("latency_total_ms"),
                    "probabilities": probabilities,
                    "is_mismatch": gesture_label != pred_label,
                }
            )

        source_summaries.append(
            {
                **merged_entry,
                "frame_count": merged_entry.get("total_frames") or merged_entry.get("dataset_source_row_count") or len(rows),
                "prediction_frame_count": len(rows),
                "accuracy": accuracy,
                "mismatch_count": mismatch_count,
                "prediction_available": bool(rows),
                "dominant_predictions": pred_counter.most_common(3),
            }
        )
        video_entries.append(
            {
                **merged_entry,
                "frame_count": merged_entry.get("total_frames") or merged_entry.get("dataset_source_row_count") or len(rows),
                "prediction_frame_count": len(rows),
                "accuracy": accuracy,
                "prediction_available": bool(rows),
                "events": event_preview,
            }
        )

    return source_summaries, video_entries


def load_latest_suite_dir() -> Path:
    latest_suite = read_json(LATEST_SUITE_PATH).get("latest_suite")
    if not latest_suite:
        raise RuntimeError(f"`latest_suite` not found in {LATEST_SUITE_PATH}")
    return Path(str(latest_suite))


def discover_suite_dirs() -> list[Path]:
    suites = []
    for candidate in sorted(PIPELINES_ROOT.iterdir(), reverse=True):
        if not candidate.is_dir():
            continue
        if not (candidate / "comparison_results.csv").exists():
            continue
        if not (candidate / "comparison_suite.json").exists():
            continue
        suites.append(candidate)
    return suites


def build_model_payload(
    suite_dir: Path,
    comparison_row: dict[str, Any],
    suite_meta: dict[str, Any],
    suite_source_catalog: dict[str, dict[str, Any]],
    detail_path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_id = str(comparison_row["model_id"])
    latest_json = read_json(suite_dir / model_id / "latest.json")
    run_dir = Path(str(latest_json["latest_run"]))
    run_summary = read_json(run_dir / "run_summary.json")
    metrics_summary = read_json(run_dir / "evaluation" / "metrics_summary.json")
    per_class_report = normalize_rows(read_csv_rows(run_dir / "evaluation" / "per_class_report.csv"))
    confusion_matrix = build_confusion_matrix(run_dir / "evaluation" / "confusion_matrix.csv")
    train_history = normalize_rows(read_csv_rows(run_dir / "train_history.csv"))
    pred_rows = normalize_rows(read_csv_rows(run_dir / "preds_test.csv"))

    class_names = [str(row.get("class")) for row in per_class_report] or DEFAULT_CLASS_NAMES
    source_summaries, video_entries = build_source_views(pred_rows, class_names, suite_source_catalog=suite_source_catalog)
    users = sorted({entry["user"] for entry in source_summaries})
    motions = sorted({entry["motion"] for entry in source_summaries})
    sides = sorted({entry["side"] for entry in source_summaries})
    dataset_info = metrics_summary.get("dataset_info") or run_summary.get("dataset_info") or {}
    dataset_user_splits = build_dataset_user_splits(dataset_info)

    artifacts = {
        "run_summary": path_from_dashboard(Path(run_dir.relative_to(PROJECT_ROOT)) / "run_summary.json"),
        "metrics_summary": path_from_dashboard(Path(run_dir.relative_to(PROJECT_ROOT)) / "evaluation" / "metrics_summary.json"),
        "per_class_report": path_from_dashboard(Path(run_dir.relative_to(PROJECT_ROOT)) / "evaluation" / "per_class_report.csv"),
        "confusion_matrix_csv": path_from_dashboard(Path(run_dir.relative_to(PROJECT_ROOT)) / "evaluation" / "confusion_matrix.csv"),
        "confusion_matrix_png": path_from_dashboard(Path(run_dir.relative_to(PROJECT_ROOT)) / "evaluation" / "confusion_matrix.png"),
        "latency_cdf_png": path_from_dashboard(Path(run_dir.relative_to(PROJECT_ROOT)) / "evaluation" / "latency_cdf.png"),
        "preds_test": path_from_dashboard(Path(run_dir.relative_to(PROJECT_ROOT)) / "preds_test.csv"),
        "train_history": path_from_dashboard(Path(run_dir.relative_to(PROJECT_ROOT)) / "train_history.csv"),
    }

    explainer = build_model_explainer(model_id, str(comparison_row["mode"]), run_summary, comparison_row)

    detail_payload = {
        "suite_name": suite_dir.name,
        "suite_meta": suite_meta,
        "model_id": model_id,
        "mode": comparison_row["mode"],
        "comparison_row": comparison_row,
        "run_dir": str(run_dir),
        "class_names": class_names,
        "run_summary": run_summary,
        "metrics_summary": metrics_summary,
        "per_class_report": per_class_report,
        "confusion_matrix": confusion_matrix,
        "train_history": train_history,
        "source_summaries": source_summaries,
        "videos": video_entries,
        "available_filters": {
            "users": users,
            "dataset_users": sorted(dataset_user_splits),
            "dataset_user_splits": dataset_user_splits,
            "motions": motions,
            "sides": sides,
        },
        "artifacts": artifacts,
        "model_explainer": explainer,
        "dataset_user_splits": dataset_user_splits,
    }

    index_summary = {
        "model_id": model_id,
        "mode": comparison_row["mode"],
        "accuracy": comparison_row["accuracy"],
        "macro_f1": comparison_row["macro_f1"],
        "macro_recall": comparison_row.get("macro_recall"),
        "fp_per_min": comparison_row.get("fp_per_min"),
        "latency_p95_ms": comparison_row.get("latency_p95_ms"),
        "epochs_ran": comparison_row["epochs_ran"],
        "test_samples": comparison_row.get("test_samples"),
        "users": users,
        "dataset_users": sorted(dataset_user_splits),
        "motions": motions,
        "source_files": [summary["source_file"] for summary in source_summaries],
        "source_count": len(source_summaries),
        "detail_path": detail_path,
        "headline": explainer["headline"],
    }
    return index_summary, detail_payload


def build_index_payload(suite_dir: Path, suite_meta: dict[str, Any], model_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    users = sorted({user for model in model_summaries for user in model.get("dataset_users", []) or model.get("users", [])})
    motions = sorted({motion for model in model_summaries for motion in model.get("motions", [])})
    source_files = sorted({source for model in model_summaries for source in model.get("source_files", [])})
    return {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "suite_name": suite_dir.name,
        "latest_suite_name": suite_dir.name,
        "suite_meta": suite_meta,
        "models": sorted(model_summaries, key=lambda item: float(item.get("macro_f1") or 0), reverse=True),
        "available_users": users,
        "available_motions": motions,
        "available_source_files": source_files,
        "recommended_model_id": max(model_summaries, key=lambda item: float(item.get("macro_f1") or 0))["model_id"] if model_summaries else None,
        "dashboard_notes": [
            "모델 비교표는 comparison_results.csv를 기준으로 정렬된다.",
            "상세 패널은 최신 run의 summary/metrics/preds_test를 묶어서 보여준다.",
            "리뷰 메모는 브라우저 localStorage에 저장되어 사용자 로컬에 남는다.",
        ],
    }


def build_suite_catalog_entry(index_payload: dict[str, Any], index_path: str, is_latest: bool) -> dict[str, Any]:
    models = index_payload.get("models", [])
    best_model = models[0] if models else {}
    return {
        "suite_name": index_payload["suite_name"],
        "dataset_tag": index_payload.get("suite_meta", {}).get("dataset_tag"),
        "index_path": index_path,
        "is_latest": is_latest,
        "model_count": len(models),
        "source_count": len(index_payload.get("available_source_files", [])),
        "best_macro_f1": best_model.get("macro_f1"),
        "recommended_model_id": index_payload.get("recommended_model_id"),
    }


def write_suite_outputs(suite_dir: Path, is_latest: bool) -> dict[str, Any]:
    suite_meta = read_json(suite_dir / "comparison_suite.json")
    comparison_rows = [
        row
        for row in normalize_rows(read_csv_rows(suite_dir / "comparison_results.csv"))
        if str(row.get("model_id")) in DEFAULT_VISIBLE_MODELS
    ]
    suite_output_dir = SUITES_ROOT / suite_dir.name
    suite_models_dir = suite_output_dir / "models"
    suite_output_dir.mkdir(parents=True, exist_ok=True)
    suite_models_dir.mkdir(parents=True, exist_ok=True)
    suite_landmark_exports = write_suite_landmark_exports(suite_dir.name, suite_output_dir, suite_meta)
    suite_source_catalog = build_suite_source_catalog(suite_meta, suite_landmark_exports=suite_landmark_exports)

    model_summaries: list[dict[str, Any]] = []
    for comparison_row in comparison_rows:
        model_id = str(comparison_row["model_id"])
        detail_path = f"data/suites/{suite_dir.name}/models/{model_id}.json"
        summary, detail_payload = build_model_payload(
            suite_dir,
            comparison_row,
            suite_meta,
            suite_source_catalog=suite_source_catalog,
            detail_path=detail_path,
        )
        model_summaries.append(summary)
        output_text = json.dumps(detail_payload, ensure_ascii=False, indent=2)
        (suite_models_dir / f"{model_id}.json").write_text(output_text, encoding="utf-8")
        if is_latest:
            (MODELS_ROOT / f"{model_id}.json").write_text(output_text, encoding="utf-8")

    index_payload = build_index_payload(suite_dir, suite_meta, model_summaries)
    index_text = json.dumps(index_payload, ensure_ascii=False, indent=2)
    (suite_output_dir / "index.json").write_text(index_text, encoding="utf-8")
    if is_latest:
        (DATA_ROOT / "latest-suite-index.json").write_text(index_text, encoding="utf-8")

    return build_suite_catalog_entry(
        index_payload=index_payload,
        index_path=f"data/suites/{suite_dir.name}/index.json",
        is_latest=is_latest,
    )


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    SUITES_ROOT.mkdir(parents=True, exist_ok=True)

    latest_suite_dir = load_latest_suite_dir()
    suite_dirs = discover_suite_dirs()
    if not suite_dirs:
        raise RuntimeError(f"No evaluation suites found under {PIPELINES_ROOT}")

    catalog_entries = [
        write_suite_outputs(suite_dir, is_latest=(suite_dir.resolve() == latest_suite_dir.resolve()))
        for suite_dir in suite_dirs
    ]

    catalog_payload = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "default_suite_name": latest_suite_dir.name,
        "suites": catalog_entries,
    }
    (DATA_ROOT / "suite-catalog.json").write_text(
        json.dumps(catalog_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote dashboard data for suite: {latest_suite_dir.name}")
    print(f"Catalog: {DATA_ROOT / 'suite-catalog.json'}")
    print(f"Index: {DATA_ROOT / 'latest-suite-index.json'}")
    print(f"Models: {MODELS_ROOT}")


if __name__ == "__main__":
    main()
