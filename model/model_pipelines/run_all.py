#!/usr/bin/env python3
# run_all.py - 역할형 학습데이터셋 세트를 자동 인식해 전체 모델 비교를 순차 실행한다.
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KST = timezone(timedelta(hours=9))
PIPELINE_SCRIPT = PROJECT_ROOT / "model" / "model_pipelines" / "run_pipeline.py"
DATASET_ROOT = PROJECT_ROOT / "model" / "data_fusion" / "학습데이터셋"

CORE_MODELS = [
    "mlp_original",
    "mlp_baseline",
    "mlp_baseline_seq8",
    "mlp_sequence_joint",
    "mlp_temporal_pooling",
    "mlp_sequence_delta",
    "mlp_embedding",
    "cnn1d_tcn",
    "transformer_embedding",
]

IMAGE_MODELS = [
    "mobilenetv3_small",
    "shufflenetv2_x0_5",
    "efficientnet_b0",
]

ALL_MODELS = CORE_MODELS + IMAGE_MODELS

REQUIRED_SPLITS = ("train", "val", "inference", "test")


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        cwd_candidate = Path.cwd() / path
        path = cwd_candidate if cwd_candidate.exists() else PROJECT_ROOT / path
    return path.resolve()


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", text.strip()).strip("-")
    return slug or "dataset"


def normalization_family_from_key(dataset_key: str) -> str:
    for family in ("baseline", "pos_only", "scale_only", "pos_scale"):
        if dataset_key == family or dataset_key.startswith(f"{family}_"):
            return family
    return "unknown"


def scan_dataset_registry(dataset_root: Path) -> dict[str, dict[str, Path]]:
    registry: dict[str, dict[str, Path]] = {}
    for train_path in sorted(dataset_root.glob("*_train.csv")):
        dataset_key = train_path.stem[: -len("_train")]
        role_paths = {
            split: dataset_root / f"{dataset_key}_{split}.csv"
            for split in REQUIRED_SPLITS
        }
        if all(path.exists() for path in role_paths.values()):
            registry[dataset_key] = role_paths
    return registry


def build_suite_dir(base_output_root: Path, dataset_key: str) -> Path:
    timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    return base_output_root / f"{timestamp}__{_slugify(dataset_key)}"


def write_suite_manifest(
    suite_dir: Path,
    dataset_key: str,
    dataset_files: dict[str, Path],
    models_requested: list[str],
    status: str,
    failed_models: list[str] | None = None,
    comparison_path: Path | None = None,
) -> None:
    manifest = {
        "suite_name": suite_dir.name,
        "created_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
        "status": status,
        "suite_dir": str(suite_dir),
        "dataset_key": dataset_key,
        "normalization_family": normalization_family_from_key(dataset_key),
        "dataset_files": {role: str(path) for role, path in dataset_files.items()},
        "models_requested": models_requested,
        "failed_models": failed_models or [],
        "comparison_results_csv": str(comparison_path) if comparison_path else None,
        "fixed_video_level_split": True,
        "source_counts": {"train": 40, "val": 9, "inference": 7},
        "test_kind": "static_images_63d",
        "test_sequence_policy": "independent_repeat",
        "official_ranking_basis": "test_csv_static_images",
    }
    with (suite_dir / "comparison_suite.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def write_latest_suite(base_output_root: Path, suite_dir: Path) -> None:
    latest_path = base_output_root / "latest_suite.json"
    with latest_path.open("w", encoding="utf-8") as f:
        json.dump({"latest_suite": str(suite_dir)}, f, ensure_ascii=False, indent=2)


def build_cmd(
    model_id: str,
    args: argparse.Namespace,
    output_root: Path,
    dataset_files: dict[str, Path],
) -> list[str]:
    cmd = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--model-id",
        model_id,
        "--train-csv",
        str(dataset_files["train"]),
        "--val-csv",
        str(dataset_files["val"]),
        "--test-csv",
        str(dataset_files["test"]),
        "--inference-csv",
        str(dataset_files["inference"]),
        "--output-root",
        str(output_root),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--patience",
        str(args.patience),
        "--focal-gamma",
        str(args.focal_gamma),
        "--seq-len",
        str(args.seq_len),
        "--seq-stride",
        str(args.seq_stride),
        "--image-size",
        str(args.image_size),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
        "--tau",
        str(args.tau),
        "--vote-n",
        str(args.vote_n),
        "--debounce-k",
        str(args.debounce_k),
        "--fallback-fps",
        str(args.fallback_fps),
    ]
    if args.class_names:
        cmd += ["--class-names", *args.class_names]
    return cmd


def _load_summary(output_root: Path, model_id: str) -> dict | None:
    latest_path = output_root / model_id / "latest.json"
    if not latest_path.exists():
        return None
    try:
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        summary_path = Path(latest["latest_run"]) / "run_summary.json"
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _flatten_summary(model_id: str, summary: dict) -> dict:
    metrics = summary.get("metrics", {})
    macro = metrics.get("macro_avg", {})
    class0 = metrics.get("class0_metrics", {})
    latency = metrics.get("latency", {})
    fp_min = metrics.get("fp_per_min_metrics", {})
    fp_value = fp_min.get("fp_per_min")
    return {
        "model_id": model_id,
        "mode": summary.get("mode", ""),
        "accuracy": round(metrics.get("accuracy", 0.0), 4),
        "macro_f1": round(macro.get("f1", 0.0), 4),
        "class0_fpr": round(class0.get("false_positive_rate", 0.0), 4),
        "class0_fnr": round(class0.get("false_negative_rate", 0.0), 4),
        "fp_per_min": None if fp_value is None else round(float(fp_value), 3),
        "latency_p50_ms": round(latency.get("p50_ms", 0.0) or 0.0, 2),
        "epochs_ran": summary.get("epochs_ran", 0),
    }


def print_table(rows: list[dict]) -> None:
    if not rows:
        return

    cols = [
        "model_id",
        "mode",
        "accuracy",
        "macro_f1",
        "class0_fpr",
        "class0_fnr",
        "latency_p50_ms",
        "epochs_ran",
    ]
    widths = {col: max(len(col), max(len(str(row.get(col, ""))) for row in rows)) for col in cols}

    header = "  ".join(col.ljust(widths[col]) for col in cols)
    sep = "  ".join("-" * widths[col] for col in cols)
    print("\n" + header)
    print(sep)
    for row in rows:
        print("  ".join(str(row.get(col, "")).ljust(widths[col]) for col in cols))


def run_dataset_suite(
    args: argparse.Namespace,
    dataset_key: str,
    dataset_files: dict[str, Path],
    models_to_run: list[str],
) -> Path:
    base_output_root = Path(args.output_root)
    if not base_output_root.is_absolute():
        base_output_root = PROJECT_ROOT / base_output_root
    base_output_root.mkdir(parents=True, exist_ok=True)

    suite_dir = build_suite_dir(base_output_root, dataset_key)
    suite_dir.mkdir(parents=True, exist_ok=True)
    write_latest_suite(base_output_root, suite_dir)
    write_suite_manifest(
        suite_dir=suite_dir,
        dataset_key=dataset_key,
        dataset_files=dataset_files,
        models_requested=models_to_run,
        status="running",
    )

    print(f"\n{'=' * 72}")
    print(f"JamJamBeat explicit dataset suite ({dataset_key})")
    print(f"models: {len(models_to_run)}")
    print(f"output: {suite_dir}")
    print(f"{'=' * 72}\n")

    results: list[dict] = []
    failed: list[str] = []
    total = len(models_to_run)

    for index, model_id in enumerate(models_to_run, 1):
        print(f"\n[{index}/{total}] {dataset_key} :: {model_id} ...")
        cmd = build_cmd(model_id, args, output_root=suite_dir, dataset_files=dataset_files)
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
        elapsed = time.time() - t0
        print(f"[{model_id}] elapsed: {elapsed:.1f}s")

        if proc.returncode != 0:
            failed.append(model_id)
            print(f"[WARN] {model_id} failed with code {proc.returncode}")
            continue

        summary = _load_summary(suite_dir, model_id)
        if summary:
            results.append(_flatten_summary(model_id, summary))

    comparison_path = suite_dir / "comparison_results.csv"
    if results:
        pd.DataFrame(results).sort_values("accuracy", ascending=False).to_csv(
            comparison_path,
            index=False,
        )
        print_table(results)

    status = "completed" if not failed else "partial_failed"
    write_suite_manifest(
        suite_dir=suite_dir,
        dataset_key=dataset_key,
        dataset_files=dataset_files,
        models_requested=models_to_run,
        status=status,
        failed_models=failed,
        comparison_path=comparison_path if results else None,
    )

    if failed:
        print(f"\n[WARN] failed models: {', '.join(failed)}")
    print(f"\n[done] suite output: {suite_dir}")
    return suite_dir


def run_all(args: argparse.Namespace) -> None:
    dataset_root = resolve_path(args.dataset_root)
    registry = scan_dataset_registry(dataset_root)
    if not registry:
        raise SystemExit(f"No explicit dataset sets found under: {dataset_root}")

    if args.dataset_key:
        missing = [key for key in args.dataset_key if key not in registry]
        if missing:
            raise SystemExit(f"Unknown dataset key(s): {', '.join(missing)}")
        registry = {key: registry[key] for key in args.dataset_key}

    models_to_run = args.models or (ALL_MODELS if args.include_image_models else CORE_MODELS)

    print(f"\n{'=' * 72}")
    print("JamJamBeat explicit dataset batch runner")
    print(f"dataset root: {dataset_root}")
    print(f"datasets: {len(registry)}")
    print(f"models per dataset: {len(models_to_run)}")
    print(f"{'=' * 72}")

    for dataset_key, dataset_files in registry.items():
        run_dataset_suite(args, dataset_key, dataset_files, models_to_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run JamJamBeat models sequentially for explicit split datasets.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=f"실행할 모델 지정. 기본값: core 9개 ({CORE_MODELS})",
    )
    parser.add_argument(
        "--include-image-models",
        action="store_true",
        help="기본 core 9개에 image 모델 3종을 추가한다.",
    )
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    parser.add_argument("--dataset-key", nargs="*", default=None, help="특정 dataset key만 실행")
    parser.add_argument("--output-root", default="model/model_evaluation/pipelines")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--focal-gamma", type=float, default=2.0)

    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--seq-stride", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=96)

    parser.add_argument("--tau", type=float, default=0.90)
    parser.add_argument("--vote-n", type=int, default=7)
    parser.add_argument("--debounce-k", type=int, default=5)
    parser.add_argument("--fallback-fps", type=float, default=30.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--class-names", nargs="*", default=[])
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
