#!/usr/bin/env python3
"""generate_full_landmark_inference.py - Generate viewer-style full landmark-frame inference artifacts for pipeline runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PIPELINES_ROOT = PROJECT_ROOT / "model" / "model_evaluation" / "pipelines"
RAW_VIDEO_ROOT = PROJECT_ROOT / "data" / "raw_data"
MODEL_PIPELINES_ROOT = PROJECT_ROOT / "model" / "model_pipelines"
VIDEO_CHECK_DIR = PROJECT_ROOT / "model" / "model_evaluation" / "모델별영상체크"
DEFAULT_OUTPUT_DIRNAME = "full_inference"
DEFAULT_OUTPUT_CSV = "full_preds_landmark_frames.csv"
DEFAULT_OUTPUT_SUMMARY = "summary.json"

for candidate in (str(MODEL_PIPELINES_ROOT), str(VIDEO_CHECK_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import video_check_app as vca


@dataclass(slots=True)
class RunTarget:
    suite_name: str
    model_id: str
    run_dir: Path


@dataclass(slots=True)
class ExtractedVideo:
    source_file: str
    video_path: Path
    fps: float
    total_frames: int
    has_hand: np.ndarray
    raw_landmarks: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate viewer-style full landmark-frame inference artifacts for pipeline runs. "
            "By default every run under model/model_evaluation/pipelines is scanned."
        )
    )
    parser.add_argument("--pipelines-root", default=str(PIPELINES_ROOT))
    parser.add_argument("--suite", action="append", default=[], help="Exact suite name filter. Repeatable.")
    parser.add_argument("--model", action="append", default=[], help="Exact model_id filter. Repeatable.")
    parser.add_argument("--run-id", action="append", default=[], help="Exact run directory name filter. Repeatable.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing full_inference outputs.")
    parser.add_argument("--limit-runs", type=int, default=0, help="Stop after N runs (0 = no limit).")
    parser.add_argument(
        "--output-dirname",
        default=DEFAULT_OUTPUT_DIRNAME,
        help=f"Subdirectory name to create inside each run folder. Default: {DEFAULT_OUTPUT_DIRNAME}",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def resolve_gt_label(class_names: list[str], gt_idx: int | None, gt_name: str | None) -> str:
    if gt_name:
        text = str(gt_name).strip()
        if text:
            return text
    if gt_idx is None:
        return ""
    if 0 <= gt_idx < len(class_names):
        return class_names[gt_idx]
    return str(gt_idx)


def csv_fieldnames(class_names: list[str]) -> list[str]:
    base = [
        "source_file",
        "frame_idx",
        "timestamp",
        "gesture",
        "gesture_name",
        "pred_class",
        "pred_label",
        "p_max",
        "status",
        "gt_available",
        "is_mismatch",
        "stored_test_available",
        "stored_test_pred_class",
        "stored_test_pred_label",
        "stored_test_p_max",
        "runtime_matches_stored_test",
    ]
    base.extend(f"p{idx}" for idx in range(len(class_names)))
    return base


def build_run_info(run_dir: Path) -> vca.RunInfo:
    summary_path = run_dir / "run_summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    metrics = summary.get("metrics") or {}
    macro = (metrics.get("macro_avg") or {}).get("f1")
    return vca.RunInfo(
        model_id=str(summary.get("model_id") or run_dir.parent.name),
        run_dir=run_dir,
        checkpoint_path=run_dir / "model.pt",
        summary_path=summary_path if summary_path.exists() else None,
        mode=str(summary.get("mode") or "unknown"),
        macro_f1=float(macro) if isinstance(macro, (int, float)) else None,
        display_name=run_dir.name,
    )


def discover_run_targets(
    pipelines_root: Path,
    suite_filters: set[str],
    model_filters: set[str],
    run_filters: set[str],
    limit: int,
) -> list[RunTarget]:
    targets: list[RunTarget] = []
    for checkpoint_path in sorted(pipelines_root.rglob("model.pt")):
        run_dir = checkpoint_path.parent
        rel = run_dir.relative_to(pipelines_root)
        if len(rel.parts) < 3:
            continue
        suite_name, model_id, run_id = rel.parts[0], rel.parts[1], rel.parts[2]
        if suite_filters and suite_name not in suite_filters:
            continue
        if model_filters and model_id not in model_filters:
            continue
        if run_filters and run_id not in run_filters:
            continue
        targets.append(RunTarget(suite_name=suite_name, model_id=model_id, run_dir=run_dir))
        if limit and len(targets) >= limit:
            break
    return targets


def load_gt_rows(input_csv_paths: list[str]) -> dict[str, dict[int, dict[str, str]]]:
    rows_by_source: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for csv_path_str in input_csv_paths:
        csv_path = Path(csv_path_str).resolve()
        if not csv_path.exists():
            print(f"[warn] missing input csv: {csv_path}", file=sys.stderr)
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                source_file = str(row.get("source_file") or csv_path.stem).strip()
                if not source_file:
                    continue
                frame_idx = safe_int(row.get("frame_idx"))
                if frame_idx is None:
                    continue
                rows_by_source[source_file][frame_idx] = row
    return rows_by_source


def load_stored_test_rows(run_dir: Path, class_names: list[str]) -> dict[str, dict[int, dict[str, Any]]]:
    preds_path = run_dir / "preds_test.csv"
    rows_by_source: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    if not preds_path.exists():
        return rows_by_source

    with preds_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_file = str(row.get("source_file") or "").strip()
            frame_idx = safe_int(row.get("frame_idx"))
            pred_class = safe_int(row.get("pred_class"))
            if not source_file or frame_idx is None:
                continue
            pred_label = class_names[pred_class] if pred_class is not None and 0 <= pred_class < len(class_names) else ""
            rows_by_source[source_file][frame_idx] = {
                "pred_class": pred_class,
                "pred_label": pred_label,
                "p_max": safe_float(row.get("p_max")),
                "timestamp": str(row.get("timestamp") or "").strip(),
            }
    return rows_by_source


def find_video_path(source_file: str) -> Path | None:
    for ext in vca.SUPPORTED_VIDEO_EXTS:
        candidate = RAW_VIDEO_ROOT / f"{source_file}{ext}"
        if candidate.exists():
            return candidate.resolve()
    for candidate in RAW_VIDEO_ROOT.iterdir():
        if candidate.stem == source_file and candidate.suffix.lower() in vca.SUPPORTED_VIDEO_EXTS:
            return candidate.resolve()
    return None


def extract_video_landmarks(
    source_file: str,
    video_path: Path,
    cache: dict[str, ExtractedVideo],
) -> ExtractedVideo:
    cached = cache.get(source_file)
    if cached is not None:
        return cached

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_landmarks: list[np.ndarray] = []
    hand_mask: list[bool] = []

    print(f"[extract] {source_file} ({video_path.name})")
    with vca.create_landmarker() as landmarker:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = int((frame_idx / max(fps, 1e-6)) * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = vca.mp.Image(image_format=vca.mp.ImageFormat.SRGB, data=rgb)
            mp_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            if not mp_result.hand_landmarks:
                hand_mask.append(False)
                frame_landmarks.append(np.full((21, 3), np.nan, dtype=np.float32))
            else:
                hand_mask.append(True)
                frame_landmarks.append(
                    np.array([[lm.x, lm.y, lm.z] for lm in mp_result.hand_landmarks[0]], dtype=np.float32)
                )
            frame_idx += 1
            if frame_idx % 300 == 0:
                print(f"  - progress {frame_idx}/{frame_count_hint or '?'}")

    cap.release()

    if frame_landmarks:
        raw_landmarks = np.stack(frame_landmarks, axis=0).astype(np.float32)
        has_hand = np.array(hand_mask, dtype=bool)
    else:
        raw_landmarks = np.zeros((0, 21, 3), dtype=np.float32)
        has_hand = np.zeros((0,), dtype=bool)

    extracted = ExtractedVideo(
        source_file=source_file,
        video_path=video_path,
        fps=fps,
        total_frames=int(raw_landmarks.shape[0]),
        has_hand=has_hand,
        raw_landmarks=raw_landmarks,
    )
    cache[source_file] = extracted
    return extracted


def iter_runtime_rows(
    runtime: vca.RuntimeModel,
    extracted: ExtractedVideo,
    gt_rows: dict[int, dict[str, str]],
    stored_rows: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    seq_buffer: deque[np.ndarray] = deque(maxlen=runtime.seq_len)
    rows: list[dict[str, Any]] = []
    counts = {
        "total_video_frames": extracted.total_frames,
        "landmark_frames": 0,
        "gt_frames": 0,
        "mismatch_frames": 0,
        "stored_overlap_frames": 0,
        "runtime_matches_stored_test": 0,
    }

    for frame_idx in range(extracted.total_frames):
        if not bool(extracted.has_hand[frame_idx]):
            seq_buffer.clear()
            continue

        raw_landmarks = extracted.raw_landmarks[frame_idx]
        feature_pack = vca.extract_feature_pack(raw_landmarks)
        status, pred_idx, confidence, probs = vca.predict_from_features(runtime, feature_pack, seq_buffer)

        gt_row = gt_rows.get(frame_idx, {})
        gt_idx = safe_int(gt_row.get("gesture"))
        gt_label = resolve_gt_label(runtime.class_names, gt_idx, gt_row.get("gesture_name"))
        timestamp = (
            str(gt_row.get("timestamp") or "").strip()
            or str((stored_rows.get(frame_idx) or {}).get("timestamp") or "").strip()
            or vca.format_timestamp(frame_idx, extracted.fps)
        )
        pred_label = runtime.class_names[pred_idx] if 0 <= pred_idx < len(runtime.class_names) else str(pred_idx)
        gt_available = gt_idx is not None
        is_mismatch = bool(gt_available and pred_idx != gt_idx)
        stored_row = stored_rows.get(frame_idx)
        stored_pred_idx = safe_int(stored_row.get("pred_class")) if stored_row else None
        stored_pred_label = str(stored_row.get("pred_label") or "") if stored_row else ""
        stored_p_max = safe_float(stored_row.get("p_max")) if stored_row else None
        runtime_matches_stored = stored_pred_idx is not None and pred_idx == stored_pred_idx

        row: dict[str, Any] = {
            "source_file": extracted.source_file,
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "gesture": "" if gt_idx is None else gt_idx,
            "gesture_name": gt_label,
            "pred_class": pred_idx,
            "pred_label": pred_label,
            "p_max": round(float(confidence), 6),
            "status": status,
            "gt_available": gt_available,
            "is_mismatch": "" if not gt_available else is_mismatch,
            "stored_test_available": stored_row is not None,
            "stored_test_pred_class": "" if stored_pred_idx is None else stored_pred_idx,
            "stored_test_pred_label": stored_pred_label,
            "stored_test_p_max": "" if stored_p_max is None else round(float(stored_p_max), 6),
            "runtime_matches_stored_test": "" if stored_pred_idx is None else runtime_matches_stored,
        }
        for idx, prob in enumerate(probs):
            row[f"p{idx}"] = round(float(prob), 8)
        rows.append(row)

        counts["landmark_frames"] += 1
        if gt_available:
            counts["gt_frames"] += 1
        if is_mismatch:
            counts["mismatch_frames"] += 1
        if stored_row is not None:
            counts["stored_overlap_frames"] += 1
            if runtime_matches_stored:
                counts["runtime_matches_stored_test"] += 1

    return rows, counts


def process_run(
    target: RunTarget,
    output_dirname: str,
    force: bool,
    extraction_cache: dict[str, ExtractedVideo],
) -> dict[str, Any]:
    run_dir = target.run_dir
    output_dir = run_dir / output_dirname
    output_csv_path = output_dir / DEFAULT_OUTPUT_CSV
    output_summary_path = output_dir / DEFAULT_OUTPUT_SUMMARY

    if output_csv_path.exists() and output_summary_path.exists() and not force:
        return {
            "suite_name": target.suite_name,
            "model_id": target.model_id,
            "run_dir": str(run_dir),
            "status": "skipped_existing",
            "output_csv": str(output_csv_path),
            "summary_json": str(output_summary_path),
        }

    run_info = build_run_info(run_dir)
    runtime = vca.load_runtime_model(run_info)
    summary = read_json(run_dir / "run_summary.json")
    dataset_info = summary.get("dataset_info") or {}
    input_csv_paths = list(dataset_info.get("input_csv_paths") or summary.get("inputs") or [])
    gt_rows_by_source = load_gt_rows(input_csv_paths)
    stored_rows_by_source = load_stored_test_rows(run_dir, runtime.class_names)

    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = csv_fieldnames(runtime.class_names)
    per_source_summary: dict[str, Any] = {}
    skipped_sources: list[dict[str, str]] = []

    total_landmark_frames = 0
    total_gt_frames = 0
    total_mismatch_frames = 0
    total_overlap_frames = 0
    total_runtime_matches = 0
    processed_sources = 0

    with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for source_file in sorted(gt_rows_by_source):
            video_path = find_video_path(source_file)
            if video_path is None:
                skipped_sources.append({"source_file": source_file, "reason": "video_not_found"})
                continue

            extracted = extract_video_landmarks(source_file, video_path, extraction_cache)
            rows, counts = iter_runtime_rows(
                runtime=runtime,
                extracted=extracted,
                gt_rows=gt_rows_by_source[source_file],
                stored_rows=stored_rows_by_source.get(source_file, {}),
            )

            for row in rows:
                writer.writerow(row)

            processed_sources += 1
            total_landmark_frames += counts["landmark_frames"]
            total_gt_frames += counts["gt_frames"]
            total_mismatch_frames += counts["mismatch_frames"]
            total_overlap_frames += counts["stored_overlap_frames"]
            total_runtime_matches += counts["runtime_matches_stored_test"]
            per_source_summary[source_file] = {
                "video_path": str(video_path),
                "total_video_frames": counts["total_video_frames"],
                "landmark_frames": counts["landmark_frames"],
                "gt_frames": counts["gt_frames"],
                "mismatch_frames": counts["mismatch_frames"],
                "stored_overlap_frames": counts["stored_overlap_frames"],
                "runtime_matches_stored_test": counts["runtime_matches_stored_test"],
            }

    summary_payload = {
        "generated_at_utc": utc_now_iso(),
        "suite_name": target.suite_name,
        "model_id": target.model_id,
        "run_dir": str(run_dir),
        "mode": runtime.mode,
        "class_names": runtime.class_names,
        "input_csv_paths": input_csv_paths,
        "source_count_processed": processed_sources,
        "source_count_skipped": len(skipped_sources),
        "landmark_frame_count": total_landmark_frames,
        "gt_frame_count": total_gt_frames,
        "mismatch_frame_count": total_mismatch_frames,
        "stored_test_overlap_frames": total_overlap_frames,
        "runtime_matches_stored_test": total_runtime_matches,
        "output_csv": str(output_csv_path),
        "per_source": per_source_summary,
        "skipped_sources": skipped_sources,
    }
    output_summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "suite_name": target.suite_name,
        "model_id": target.model_id,
        "run_dir": str(run_dir),
        "status": "ok",
        "output_csv": str(output_csv_path),
        "summary_json": str(output_summary_path),
        "landmark_frame_count": total_landmark_frames,
        "source_count_processed": processed_sources,
    }


def main() -> None:
    args = parse_args()
    pipelines_root = Path(args.pipelines_root).resolve()
    if not pipelines_root.exists():
        raise FileNotFoundError(f"Pipelines root not found: {pipelines_root}")

    targets = discover_run_targets(
        pipelines_root=pipelines_root,
        suite_filters=set(args.suite),
        model_filters=set(args.model),
        run_filters=set(args.run_id),
        limit=int(args.limit_runs or 0),
    )
    if not targets:
        print("No matching runs found.", file=sys.stderr)
        return

    print(f"Discovered {len(targets)} run(s).")
    extraction_cache: dict[str, ExtractedVideo] = {}
    results: list[dict[str, Any]] = []

    for index, target in enumerate(targets, start=1):
        rel = target.run_dir.relative_to(pipelines_root)
        print(f"[{index}/{len(targets)}] {rel}")
        try:
            result = process_run(
                target=target,
                output_dirname=args.output_dirname,
                force=bool(args.force),
                extraction_cache=extraction_cache,
            )
            results.append(result)
            print(f"  -> {result['status']} | {result.get('output_csv', '-')}")
        except Exception as exc:
            result = {
                "suite_name": target.suite_name,
                "model_id": target.model_id,
                "run_dir": str(target.run_dir),
                "status": "error",
                "error": str(exc),
            }
            results.append(result)
            print(f"  -> error | {exc}", file=sys.stderr)

    ok_count = sum(1 for item in results if item["status"] == "ok")
    skipped_count = sum(1 for item in results if item["status"] == "skipped_existing")
    error_count = sum(1 for item in results if item["status"] == "error")
    print(
        json.dumps(
            {
                "total_runs": len(results),
                "ok": ok_count,
                "skipped_existing": skipped_count,
                "error": error_count,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
