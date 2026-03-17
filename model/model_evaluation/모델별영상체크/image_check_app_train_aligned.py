#!/usr/bin/env python3
"""image_check_app_train_aligned.py - Extract landmarks from labeled image folders and run train-aligned checkpoint inference."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

try:
    from . import video_check_app as base
    from . import video_check_app_train_aligned as aligned
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    import video_check_app as base
    import video_check_app_train_aligned as aligned


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IMAGES_ROOT = PROJECT_ROOT / "model" / "data_fusion" / "추론용데이터셋"
EVAL_RUNTIME_ROOT = PROJECT_ROOT / "model" / "model_evaluation" / "모델검증관련파일"

if str(EVAL_RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_RUNTIME_ROOT))

from evaluation_runtime import EvaluationConfig, evaluate_predictions


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
USER_RE = re.compile(r"(man\d+|woman\d+)", re.IGNORECASE)
NUM_RE = re.compile(r"\d+")
IMAGE_SEQUENCE_MODES = {"independent", "sequential"}


@dataclass(slots=True)
class ImageSample:
    """Single labeled image sample from a directory tree or a zip archive."""

    gt_label: int
    user_id: str
    group_key: str
    rel_path: str
    image_name: str
    file_path: Path | None = None
    zip_path: Path | None = None
    zip_member: str | None = None


class ZipPool:
    """Keep zip files open during inference to avoid reopening each member."""

    def __init__(self) -> None:
        self._pool: dict[Path, zipfile.ZipFile] = {}

    def read(self, zip_path: Path, member: str) -> bytes:
        archive = self._pool.get(zip_path)
        if archive is None:
            archive = zipfile.ZipFile(zip_path)
            self._pool[zip_path] = archive
        return archive.read(member)

    def close(self) -> None:
        for archive in self._pool.values():
            archive.close()
        self._pool.clear()


def resolve_cli_path(path: Path, *, must_exist: bool = True) -> Path:
    """Resolve CLI paths against cwd first, then against repo root for convenience."""
    if path.is_absolute():
        resolved = path.resolve()
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"Path not found: {resolved}")
        return resolved

    candidates = [
        (Path.cwd() / path).resolve(),
        (PROJECT_ROOT / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    fallback = candidates[0]
    if must_exist:
        raise FileNotFoundError(
            f"Path not found: {fallback} (also tried repo-root path: {candidates[1]})"
        )
    return fallback


def natural_key(text: str) -> list[Any]:
    """Sort paths in numeric order when filenames contain frame-like integers."""
    parts = re.split(r"(\d+)", text)
    key: list[Any] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def infer_user_id(parts: list[str], fallback: str) -> str:
    """Infer user id from path parts or filename tokens."""
    for part in parts:
        match = USER_RE.search(part)
        if match:
            return match.group(1).lower()
    match = USER_RE.search(fallback)
    if match:
        return match.group(1).lower()
    return "unknown"


def infer_gt_label(parts: list[str], filename: str) -> int | None:
    """Infer numeric ground-truth label from folder structure or filename prefix."""
    for part in reversed(parts[:-1]):
        if part.isdigit():
            return int(part)
    match = re.match(r"(\d+)(?:_|$)", filename)
    if match:
        return int(match.group(1))
    return None


def iter_regular_samples(images_root: Path) -> list[ImageSample]:
    """Collect image samples already extracted on disk."""
    samples: list[ImageSample] = []
    for path in sorted(images_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = path.relative_to(images_root)
        parts = list(rel.parts)
        gt_label = infer_gt_label(parts, path.name)
        if gt_label is None:
            continue
        user_id = infer_user_id(parts, path.name)
        group_parts = parts[:-1]
        if group_parts:
            group_key = "/".join(group_parts)
        else:
            group_key = f"{user_id}/{gt_label}"
        samples.append(
            ImageSample(
                gt_label=gt_label,
                user_id=user_id,
                group_key=group_key,
                rel_path=rel.as_posix(),
                image_name=path.name,
                file_path=path,
            )
        )
    return samples


def iter_zip_samples(images_root: Path) -> list[ImageSample]:
    """Collect image samples from zip archives without extracting them to disk."""
    samples: list[ImageSample] = []
    for zip_path in sorted(images_root.rglob("*.zip")):
        user_id = infer_user_id(list(zip_path.relative_to(images_root).parts), zip_path.name)
        with zipfile.ZipFile(zip_path) as archive:
            members = sorted(
                (info.filename for info in archive.infolist() if not info.is_dir()),
                key=natural_key,
            )
            for member in members:
                member_path = Path(member)
                if member_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                parts = list(member_path.parts)
                gt_label = infer_gt_label(parts, member_path.name)
                if gt_label is None:
                    continue
                group_parts = [user_id] + parts[:-1]
                group_key = "/".join(group_parts) if group_parts else f"{user_id}/{gt_label}"
                rel_path = f"{zip_path.relative_to(images_root).as_posix()}::{member}"
                samples.append(
                    ImageSample(
                        gt_label=gt_label,
                        user_id=user_id,
                        group_key=group_key,
                        rel_path=rel_path,
                        image_name=member_path.name,
                        zip_path=zip_path,
                        zip_member=member,
                    )
                )
    return samples


def collect_samples(images_root: Path, users: set[str] | None) -> list[ImageSample]:
    """Collect and sort labeled image samples from directories and archives."""
    samples = iter_regular_samples(images_root) + iter_zip_samples(images_root)
    if users:
        samples = [sample for sample in samples if sample.user_id in users]
    samples.sort(key=lambda sample: (sample.user_id, sample.group_key, natural_key(sample.rel_path)))
    return samples


def load_image_bgr(sample: ImageSample, zip_pool: ZipPool) -> np.ndarray:
    """Load BGR image data from disk or zip archive."""
    if sample.file_path is not None:
        image = cv2.imread(str(sample.file_path), cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"Could not read image: {sample.file_path}")
        return image

    if sample.zip_path is None or sample.zip_member is None:
        raise ValueError(f"Sample has no readable source: {sample.rel_path}")

    data = zip_pool.read(sample.zip_path, sample.zip_member)
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"Could not decode image from zip: {sample.rel_path}")
    return image


def create_image_landmarker() -> Any:
    """Create a MediaPipe hand landmarker configured for still images."""
    if not base.TASK_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing task model: {base.TASK_MODEL_PATH}")

    options = base.HandLandmarkerOptions(
        base_options=base.BaseOptions(model_asset_path=str(base.TASK_MODEL_PATH)),
        running_mode=base.VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return base.HandLandmarker.create_from_options(options)


def image_dataset_output_dir(run_dir: Path, images_root: Path) -> Path:
    """Default output directory under the selected run folder."""
    dataset_slug = re.sub(r"[^0-9A-Za-z._-]+", "_", images_root.name).strip("_") or "images"
    return run_dir / "image_inference" / dataset_slug


@aligned.torch.inference_mode()
def predict_sequence_independent(
    runtime: base.RuntimeModel,
    features: aligned.TrainingAlignedFeaturePack,
) -> tuple[str, int, float, list[float]]:
    """Predict sequence models from a single still image by repeating one feature vector seq_len times."""
    if runtime.mode != "sequence":
        raise ValueError(f"predict_sequence_independent expects a sequence model, got: {runtime.mode}")
    if runtime.input_dim is None:
        raise ValueError(f"Missing runtime input_dim for sequence model: {runtime.model_id}")

    if runtime.model_id == "mlp_sequence_delta":
        if runtime.input_dim % 2 != 0:
            raise ValueError(f"Unexpected delta input_dim for {runtime.model_id}: {runtime.input_dim}")
        base_vec = aligned.select_feature_vector_training_aligned(features, runtime.input_dim // 2)
    else:
        base_vec = aligned.select_feature_vector_training_aligned(features, runtime.input_dim)

    seq = np.repeat(base_vec[None, :], runtime.seq_len, axis=0).astype(np.float32)
    if runtime.model_id == "mlp_sequence_delta":
        seq = base.add_runtime_delta_features(seq)
        if seq.shape[1] != runtime.input_dim:
            raise ValueError(
                f"Delta feature mismatch for {runtime.model_id}: "
                f"got {seq.shape[1]}, expected {runtime.input_dim}"
            )

    tensor = aligned.torch.from_numpy(seq).unsqueeze(0).to(runtime.device)
    logits = runtime.model(tensor)
    probs = aligned.torch.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(float).tolist()
    pred_idx = int(np.argmax(probs))
    return "ready", pred_idx, float(probs[pred_idx]), probs


def predict_image_sample(
    runtime: base.RuntimeModel,
    features: aligned.TrainingAlignedFeaturePack,
    seq_buffer: deque[np.ndarray],
    image_sequence_mode: str,
) -> tuple[str, int, list[float]]:
    """Dispatch still-image inference for the requested sequence handling mode."""
    if runtime.mode != "sequence":
        status, pred_idx, _, probs = aligned.predict_from_features_training_aligned(runtime, features, seq_buffer)
        return status, pred_idx, probs

    if image_sequence_mode == "independent":
        status, pred_idx, _, probs = predict_sequence_independent(runtime, features)
        return status, pred_idx, probs

    if image_sequence_mode == "sequential":
        status, pred_idx, _, probs = aligned.predict_from_features_training_aligned(runtime, features, seq_buffer)
        return status, pred_idx, probs

    raise ValueError(f"Unsupported image_sequence_mode: {image_sequence_mode}")


def analyze_images(
    runtime: base.RuntimeModel,
    images_root: Path,
    samples: list[ImageSample],
    image_sequence_mode: str,
) -> pd.DataFrame:
    """Run landmark extraction and checkpoint inference across labeled images."""
    rows: list[dict[str, Any]] = []
    seq_buffer: deque[np.ndarray] = deque(maxlen=runtime.seq_len)
    current_group: str | None = None
    group_frame_idx = -1
    zip_pool = ZipPool()

    try:
        with create_image_landmarker() as landmarker:
            for idx, sample in enumerate(samples):
                if sample.group_key != current_group:
                    current_group = sample.group_key
                    group_frame_idx = 0
                    seq_buffer.clear()
                else:
                    group_frame_idx += 1

                image_bgr = load_image_bgr(sample, zip_pool)
                rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                started = time.perf_counter()
                result = landmarker.detect(mp_image)
                latency_ms = (time.perf_counter() - started) * 1000.0

                if not result.hand_landmarks:
                    seq_buffer.clear()
                    probs = base.neutral_probs(len(runtime.class_names), runtime.neutral_idx)
                    pred_idx = runtime.neutral_idx
                    status = "no_hand"
                else:
                    raw_landmarks = np.array(
                        [[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]],
                        dtype=np.float32,
                    )
                    features = aligned.extract_feature_pack_training_aligned(
                        raw_landmarks,
                        dataset_variant=runtime.dataset_variant,
                    )
                    status, pred_idx, probs = predict_image_sample(
                        runtime,
                        features,
                        seq_buffer,
                        image_sequence_mode,
                    )

                p_max = float(max(probs))
                pred_name = runtime.class_names[pred_idx]
                gt_name = runtime.class_names[sample.gt_label] if 0 <= sample.gt_label < len(runtime.class_names) else str(sample.gt_label)

                rows.append(
                    {
                        "image_rel_path": sample.rel_path,
                        "image_name": sample.image_name,
                        "source_file": sample.group_key,
                        "user_id": sample.user_id,
                        "dataset_variant": runtime.dataset_variant,
                        "image_sequence_mode": image_sequence_mode,
                        "frame_idx": group_frame_idx,
                        "gesture": int(sample.gt_label),
                        "gt_name": gt_name,
                        "pred_class": int(pred_idx),
                        "pred_name": pred_name,
                        "p_max": p_max,
                        "status": status,
                        "hand_detected": int(status != "no_hand"),
                        "top1_correct": int(int(pred_idx) == int(sample.gt_label)),
                        "latency_total_ms": float(latency_ms),
                    }
                )

                if (idx + 1) % 50 == 0 or idx + 1 == len(samples):
                    pct = ((idx + 1) / max(len(samples), 1)) * 100.0
                    print(f"[image-check] progress {idx + 1}/{len(samples)} ({pct:.1f}%)")
    finally:
        zip_pool.close()

    return pd.DataFrame(rows)


def write_summary(
    output_dir: Path,
    runtime: base.RuntimeModel,
    images_root: Path,
    samples: list[ImageSample],
    preds_df: pd.DataFrame,
    metrics: dict[str, Any],
    image_sequence_mode: str,
) -> None:
    """Write a compact inference summary next to CSV/evaluation outputs."""
    by_user = preds_df.groupby("user_id")["top1_correct"].agg(["count", "sum"]).reset_index()
    by_user_rows = []
    for _, row in by_user.iterrows():
        count = int(row["count"])
        correct = int(row["sum"])
        by_user_rows.append(
            {
                "user_id": str(row["user_id"]),
                "samples": count,
                "correct": correct,
                "accuracy": (correct / count) if count else 0.0,
            }
        )

    summary = {
        "run_dir": str(runtime.run_info.run_dir),
        "model_id": runtime.model_id,
        "mode": runtime.mode,
        "dataset_variant": runtime.dataset_variant,
        "image_sequence_mode": image_sequence_mode,
        "images_root": str(images_root),
        "total_samples": int(len(preds_df)),
        "hand_detected_samples": int(preds_df["hand_detected"].sum()) if not preds_df.empty else 0,
        "no_hand_samples": int((preds_df["status"] == "no_hand").sum()) if not preds_df.empty else 0,
        "warmup_samples": int((preds_df["status"] == "warmup").sum()) if not preds_df.empty else 0,
        "users": sorted({sample.user_id for sample in samples}),
        "metrics_summary": metrics,
        "accuracy_by_user": by_user_rows,
    }
    (output_dir / "inference_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI args for labeled-image inference."""
    parser = argparse.ArgumentParser(description="JamJamBeat image-folder inference using training-aligned checkpoints")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help=(
            "Run directory containing model.pt, or a model directory containing latest.json "
            "(e.g. model/model_evaluation/pipelines/{suite_name}/{model_id})"
        ),
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=DEFAULT_IMAGES_ROOT,
        help="Root directory containing labeled image folders and/or zip archives",
    )
    parser.add_argument(
        "--users",
        nargs="*",
        default=None,
        help="Optional subset of users to evaluate, e.g. --users man1 woman1",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap for quick smoke runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write preds/evaluation outputs (default: run_dir/image_inference/<dataset>)",
    )
    parser.add_argument(
        "--image-sequence-mode",
        type=str,
        default="independent",
        choices=sorted(IMAGE_SEQUENCE_MODES),
        help="How to evaluate sequence models on image folders: independent stills or sequential folder order",
    )
    return parser.parse_args()


def main() -> None:
    """Run image-folder inference and write predictions/evaluation artifacts."""
    args = parse_args()

    images_root = resolve_cli_path(args.images_root, must_exist=True)

    users = {user.lower() for user in args.users} if args.users else None
    run_dir = base.resolve_run_dir_arg(resolve_cli_path(args.run_dir, must_exist=True))
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

    run_info = base.RunInfo(
        model_id=run_dir.parent.name,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        summary_path=summary_path if summary_path.exists() else None,
        mode=mode,
        macro_f1=float(macro_f1) if isinstance(macro_f1, (int, float)) else None,
        display_name=run_dir.as_posix(),
    )
    runtime = base.load_runtime_model(run_info)
    image_sequence_mode = str(args.image_sequence_mode).strip().lower()

    print(f"[image-check] model={runtime.model_id} mode={runtime.mode} run={run_dir.name}")
    print(f"[image-check] dataset_variant={runtime.dataset_variant}")
    print(f"[image-check] images_root={images_root}")
    print(f"[image-check] image_sequence_mode={image_sequence_mode}")

    samples = collect_samples(images_root, users)
    if not samples:
        raise RuntimeError(f"No labeled images found under {images_root}")
    if args.max_images is not None:
        samples = samples[: max(args.max_images, 0)]
    if not samples:
        raise RuntimeError("No samples left after applying --max-images")

    print(f"[image-check] samples={len(samples)} users={sorted({sample.user_id for sample in samples})}")
    preds_df = analyze_images(runtime, images_root, samples, image_sequence_mode=image_sequence_mode)

    output_dir = (
        resolve_cli_path(args.output_dir, must_exist=False)
        if args.output_dir
        else image_dataset_output_dir(run_dir, images_root)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_path = output_dir / "preds_images.csv"
    preds_df.to_csv(preds_path, index=False, encoding="utf-8-sig")

    eval_dir = output_dir / "evaluation"
    dataset_info = {
        "dataset_type": "labeled_images",
        "images_root": str(images_root),
        "run_dir": str(run_dir),
        "model_id": runtime.model_id,
        "mode": runtime.mode,
        "dataset_variant": runtime.dataset_variant,
        "image_sequence_mode": image_sequence_mode,
        "users": sorted({sample.user_id for sample in samples}),
        "total_samples": int(len(preds_df)),
    }
    metrics = evaluate_predictions(
        preds_df,
        eval_dir,
        EvaluationConfig(class_names=runtime.class_names, dataset_info=dataset_info),
    )
    write_summary(output_dir, runtime, images_root, samples, preds_df, metrics, image_sequence_mode=image_sequence_mode)

    accuracy = float(preds_df["top1_correct"].mean()) if not preds_df.empty else 0.0
    print(f"[image-check] preds={preds_path}")
    print(f"[image-check] eval_dir={eval_dir}")
    print(f"[image-check] accuracy={accuracy:.4f} total={len(preds_df)}")


if __name__ == "__main__":
    main()
