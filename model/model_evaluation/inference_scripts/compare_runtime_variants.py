#!/usr/bin/env python3
"""compare_runtime_variants.py - Compare checkpoint/runtime inputs across multiple runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VIEWER_ROOT = PROJECT_ROOT / "model" / "model_evaluation" / "모델별영상체크"
MODEL_PIPELINES_ROOT = PROJECT_ROOT / "model" / "model_pipelines"

if str(VIEWER_ROOT) not in sys.path:
    sys.path.insert(0, str(VIEWER_ROOT))
if str(MODEL_PIPELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_PIPELINES_ROOT))

import video_check_app as base
import video_check_app_train_aligned as aligned
from checkpoint_verification import summarize_array


def resolve_cli_path(path: Path, *, must_exist: bool = True) -> Path:
    if path.is_absolute():
        resolved = path.resolve()
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"Path not found: {resolved}")
        return resolved

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    repo_candidate = (PROJECT_ROOT / path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    if must_exist:
        raise FileNotFoundError(f"Path not found: {cwd_candidate} (also tried {repo_candidate})")
    return cwd_candidate


def load_run_info(run_dir_arg: Path) -> base.RunInfo:
    run_dir = base.resolve_run_dir_arg(resolve_cli_path(run_dir_arg))
    checkpoint_path = run_dir / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    summary_path = run_dir / "run_summary.json"
    mode = "unknown"
    macro_f1: float | None = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        mode = str(summary.get("mode", "unknown"))
        macro_f1 = summary.get("metrics", {}).get("macro_avg", {}).get("f1")

    return base.RunInfo(
        model_id=run_dir.parent.name,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        summary_path=summary_path if summary_path.exists() else None,
        mode=mode,
        macro_f1=float(macro_f1) if isinstance(macro_f1, (int, float)) else None,
        display_name=run_dir.as_posix(),
    )


def extract_raw_landmarks_from_video(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame {frame_idx} from {video_path}")

    timestamp_ms = int((frame_idx / max(fps, 1e-6)) * 1000.0)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with base.create_landmarker() as landmarker:
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not result.hand_landmarks:
        raise ValueError(f"No hand landmarks detected at frame {frame_idx} in {video_path}")

    return np.array([[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]], dtype=np.float32)


def load_raw_landmarks_json(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "raw_landmarks" in payload:
            payload = payload["raw_landmarks"]
        elif "landmarks" in payload:
            payload = payload["landmarks"]
    arr = np.asarray(payload, dtype=np.float32)
    if arr.shape != (21, 3):
        raise ValueError(f"Expected raw landmark shape (21, 3), got {arr.shape}")
    return arr


def load_raw_landmarks_npz(path: Path) -> np.ndarray:
    bundle = np.load(path)
    if "raw_landmarks" in bundle:
        arr = bundle["raw_landmarks"]
    else:
        first_key = next(iter(bundle.files), None)
        if first_key is None:
            raise ValueError(f"No arrays found in npz: {path}")
        arr = bundle[first_key]
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape != (21, 3):
        raise ValueError(f"Expected raw landmark shape (21, 3), got {arr.shape}")
    return arr


def topk_rows(class_names: list[str], probs: list[float], k: int) -> list[dict[str, Any]]:
    pairs = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)[:k]
    return [
        {
            "rank": rank + 1,
            "class_id": int(idx),
            "class_name": str(class_names[idx]),
            "score": float(score),
        }
        for rank, (idx, score) in enumerate(pairs)
    ]


def run_single_comparison(runtime: base.RuntimeModel, raw_landmarks: np.ndarray, top_k: int) -> dict[str, Any]:
    features = aligned.extract_feature_pack_training_aligned(
        raw_landmarks,
        dataset_variant=runtime.dataset_variant,
    )

    if runtime.mode == "frame":
        if runtime.input_dim is None:
            raise ValueError(f"Missing input_dim for frame model: {runtime.model_id}")
        final_input = aligned.select_feature_vector_training_aligned(features, runtime.input_dim)
        tensor = torch.from_numpy(final_input).unsqueeze(0).to(runtime.device)
        input_kind = "frame_vector"
    elif runtime.mode == "sequence":
        if runtime.input_dim is None:
            raise ValueError(f"Missing input_dim for sequence model: {runtime.model_id}")
        if runtime.model_id == "mlp_sequence_delta":
            if runtime.input_dim % 2 != 0:
                raise ValueError(f"Unexpected delta input_dim for {runtime.model_id}: {runtime.input_dim}")
            base_vec = aligned.select_feature_vector_training_aligned(features, runtime.input_dim // 2)
        else:
            base_vec = aligned.select_feature_vector_training_aligned(features, runtime.input_dim)
        final_input = np.repeat(base_vec[None, :], runtime.seq_len, axis=0).astype(np.float32)
        if runtime.model_id == "mlp_sequence_delta":
            final_input = base.add_runtime_delta_features(final_input)
        tensor = torch.from_numpy(final_input).unsqueeze(0).to(runtime.device)
        input_kind = "sequence_tensor_independent_repeat"
    elif runtime.mode == "image":
        final_input = aligned.render_train_skeleton_image(features.train_landmarks, runtime.image_size)
        tensor = torch.from_numpy(final_input).unsqueeze(0).to(runtime.device)
        input_kind = "image_tensor"
    else:
        raise ValueError(f"Unsupported mode: {runtime.mode}")

    with torch.inference_mode():
        logits = runtime.model(tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(float)
    logits_np = logits[0].detach().cpu().numpy().astype(np.float32)
    topk = topk_rows(runtime.class_names, probs.tolist(), top_k)

    return {
        "run_dir": str(runtime.run_info.run_dir),
        "model_id": runtime.model_id,
        "mode": runtime.mode,
        "dataset_variant": runtime.dataset_variant,
        "checkpoint_verification": runtime.checkpoint_verification,
        "input_kind": input_kind,
        "raw_landmarks": summarize_array("raw_landmarks", raw_landmarks),
        "variant_landmarks": summarize_array("variant_landmarks", features.train_landmarks),
        "final_model_input": summarize_array("final_model_input", final_input),
        "logits": summarize_array("logits", logits_np),
        "probs": summarize_array("probs", probs),
        "topk": topk,
    }


def build_comparison_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not results:
        return []

    reference = results[0]
    rows: list[dict[str, Any]] = []
    for result in results:
        top1 = result["topk"][0] if result["topk"] else {}
        rows.append(
            {
                "run_dir": result["run_dir"],
                "model_id": result["model_id"],
                "mode": result["mode"],
                "dataset_variant": result["dataset_variant"],
                "checkpoint_fingerprint": (
                    result.get("checkpoint_verification") or {}
                ).get("checkpoint_fingerprint"),
                "raw_sha256": result["raw_landmarks"]["sha256"],
                "variant_sha256": result["variant_landmarks"]["sha256"],
                "final_input_sha256": result["final_model_input"]["sha256"],
                "logits_sha256": result["logits"]["sha256"],
                "probs_sha256": result["probs"]["sha256"],
                "top1_class": top1.get("class_name"),
                "top1_score": top1.get("score"),
                "same_checkpoint_as_reference": (
                    (
                        (result.get("checkpoint_verification") or {}).get("checkpoint_fingerprint")
                    )
                    == (
                        (reference.get("checkpoint_verification") or {}).get("checkpoint_fingerprint")
                    )
                ),
                "same_variant_input_as_reference": (
                    result["variant_landmarks"]["sha256"] == reference["variant_landmarks"]["sha256"]
                ),
                "same_final_input_as_reference": (
                    result["final_model_input"]["sha256"] == reference["final_model_input"]["sha256"]
                ),
                "same_probs_as_reference": result["probs"]["sha256"] == reference["probs"]["sha256"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare checkpoint fingerprints, runtime inputs, and outputs across multiple runs."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        nargs="+",
        required=True,
        help="One or more run dirs, model dirs with latest.json, or suite/model dirs.",
    )
    parser.add_argument("--video", type=Path, default=None, help="Video path for same-frame comparison.")
    parser.add_argument("--frame-idx", type=int, default=None, help="Frame index used with --video.")
    parser.add_argument("--raw-landmarks-json", type=Path, default=None, help="JSON file containing a (21,3) landmark array.")
    parser.add_argument("--raw-landmarks-npz", type=Path, default=None, help="NPZ file containing raw_landmarks.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional report JSON path.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional comparison CSV path.")
    parser.add_argument("--top-k", type=int, default=3, help="How many classes to keep in the per-run summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_mode_count = sum(
        value is not None
        for value in (args.video, args.raw_landmarks_json, args.raw_landmarks_npz)
    )
    if source_mode_count != 1:
        raise SystemExit("Choose exactly one of --video, --raw-landmarks-json, or --raw-landmarks-npz.")
    if args.video is not None and args.frame_idx is None:
        raise SystemExit("--frame-idx is required with --video.")

    if args.video is not None:
        raw_landmarks = extract_raw_landmarks_from_video(resolve_cli_path(args.video), int(args.frame_idx))
        input_source = {
            "mode": "video_frame",
            "video": str(resolve_cli_path(args.video)),
            "frame_idx": int(args.frame_idx),
        }
    elif args.raw_landmarks_json is not None:
        raw_landmarks = load_raw_landmarks_json(resolve_cli_path(args.raw_landmarks_json))
        input_source = {"mode": "raw_landmarks_json", "path": str(resolve_cli_path(args.raw_landmarks_json))}
    else:
        raw_landmarks = load_raw_landmarks_npz(resolve_cli_path(args.raw_landmarks_npz))
        input_source = {"mode": "raw_landmarks_npz", "path": str(resolve_cli_path(args.raw_landmarks_npz))}

    results: list[dict[str, Any]] = []
    for run_dir_arg in args.run_dir:
        run_info = load_run_info(run_dir_arg)
        runtime = aligned.load_runtime_model_training_aligned(run_info)
        results.append(run_single_comparison(runtime, raw_landmarks, args.top_k))

    comparison_rows = build_comparison_rows(results)
    report = {
        "input_source": input_source,
        "raw_landmarks": summarize_array("raw_landmarks", raw_landmarks),
        "runs": results,
        "comparison_rows": comparison_rows,
    }

    for row in comparison_rows:
        print(
            "[compare-runtime] "
            f"model={row['model_id']} mode={row['mode']} variant={row['dataset_variant']} "
            f"top1={row['top1_class']} score={row['top1_score']:.4f} "
            f"same_ckpt={row['same_checkpoint_as_reference']} "
            f"same_input={row['same_final_input_as_reference']} "
            f"same_probs={row['same_probs_as_reference']}"
        )

    if args.output_json is not None:
        output_json = resolve_cli_path(args.output_json, must_exist=False)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[compare-runtime] wrote {output_json}")

    if args.output_csv is not None:
        output_csv = resolve_cli_path(args.output_csv, must_exist=False)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(output_csv, comparison_rows)
        print(f"[compare-runtime] wrote {output_csv}")


if __name__ == "__main__":
    main()
