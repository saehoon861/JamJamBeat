# run_all.py - 9개 모델 파이프라인 순차 실행 및 비교 결과 집계
"""
Usage:
    python run_all.py
    python run_all.py --csv-path path/to/data.csv --csv-path path/to/data2.csv
    python run_all.py --epochs 30 --models mlp_baseline mlp_embedding two_stream_mlp
    python run_all.py --output-root model/model_evaluation/pipelines

결과:
    model/model_evaluation/pipelines/{model_id}/{timestamp}/run_summary.json  (모델별)
    model/model_evaluation/pipelines/comparison_results.csv                   (전체 비교)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ALL_MODELS = [
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

PIPELINE_SCRIPT = PROJECT_ROOT / "model" / "model_pipelines" / "run_pipeline.py"

DEFAULT_INPUTS = [
    "model/data_fusion/man1_right_for_poc_output.csv",
    "model/data_fusion/man2_right_for_poc_output.csv",
    "model/data_fusion/man3_right_for_poc_output.csv",
    "model/data_fusion/woman1_right_for_poc_output.csv",
]


def build_cmd(model_id: str, args: argparse.Namespace) -> list[str]:
    """run_pipeline.py에 넘길 공통 CLI 인자를 한 곳에서 조립한다."""
    cmd = [sys.executable, str(PIPELINE_SCRIPT), "--model-id", model_id]

    for p in (args.csv_path or DEFAULT_INPUTS):
        cmd += ["--csv-path", p]

    cmd += [
        "--output-root", args.output_root,
        "--epochs",      str(args.epochs),
        "--batch-size",  str(args.batch_size),
        "--lr",          str(args.lr),
        "--patience",    str(args.patience),
        "--focal-gamma", str(args.focal_gamma),
        "--seq-len",     str(args.seq_len),
        "--seq-stride",  str(args.seq_stride),
        "--image-size",  str(args.image_size),
        "--seed",        str(args.seed),
        "--device",      args.device,
        "--num-workers", str(args.num_workers),
        "--tau",         str(args.tau),
        "--vote-n",      str(args.vote_n),
        "--debounce-k",  str(args.debounce_k),
        "--fallback-fps",str(args.fallback_fps),
    ]
    return cmd


def _load_summary(output_root: str, model_id: str) -> dict | None:
    """latest pointer를 따라가 해당 모델의 가장 최근 run summary를 읽는다."""
    out_root = Path(output_root)
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root

    latest_path = out_root / model_id / "latest.json"
    if not latest_path.exists():
        return None
    try:
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        summary_path = Path(latest["latest_run"]) / "run_summary.json"
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _flatten_summary(model_id: str, s: dict) -> dict:
    """run summary에서 비교 CSV에 필요한 핵심 메트릭만 평탄화한다."""
    m = s.get("metrics", {})
    macro = m.get("macro_avg", {})
    class0 = m.get("class0_metrics", {})
    latency = m.get("latency", {})
    fp_min = m.get("fp_per_min_metrics", {})

    return {
        "model_id":        model_id,
        "mode":            s.get("mode", ""),
        "accuracy":        round(m.get("accuracy", 0.0), 4),
        "macro_f1":        round(macro.get("f1", 0.0), 4),
        "macro_precision": round(macro.get("precision", 0.0), 4),
        "macro_recall":    round(macro.get("recall", 0.0), 4),
        "class0_fpr":      round(class0.get("false_positive_rate", 0.0), 4),
        "class0_fnr":      round(class0.get("false_negative_rate", 0.0), 4),
        "fp_per_min":      round(fp_min.get("fp_per_min", 0.0) or 0.0, 3),
        "latency_p50_ms":  round(latency.get("p50_ms", 0.0) or 0.0, 2),
        "latency_p95_ms":  round(latency.get("p95_ms", 0.0) or 0.0, 2),
        "best_val_loss":   round(s.get("best_val_loss", 0.0), 4),
        "epochs_ran":      s.get("epochs_ran", 0),
        "train_samples":   s.get("dataset_sizes", {}).get("train", 0),
        "test_samples":    s.get("dataset_sizes", {}).get("test", 0),
        "output_dir":      s.get("output_dir", ""),
    }


def print_table(rows: list[dict]) -> None:
    """터미널에서 빠르게 비교할 수 있도록 요약 테이블을 그린다."""
    if not rows:
        return

    cols = ["model_id", "mode", "accuracy", "macro_f1", "class0_fpr",
            "class0_fnr", "fp_per_min", "latency_p50_ms", "epochs_ran"]
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}

    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    print("\n" + header)
    print(sep)
    for row in rows:
        print("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols))


def run_all(args: argparse.Namespace) -> None:
    """선택된 모델들을 독립 subprocess로 순차 실행하고 결과를 하나로 모은다."""
    models_to_run = args.models or ALL_MODELS
    results: list[dict] = []
    failed: list[str] = []

    out_root = Path(args.output_root)
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    total = len(models_to_run)
    print(f"\n{'='*60}")
    print(f"JamJamBeat 모델 비교 실험  ({total}개 모델 순차 실행)")
    print(f"{'='*60}\n")

    for i, model_id in enumerate(models_to_run, 1):
        print(f"\n[{i}/{total}] {model_id} ...")
        # 모델별 subprocess로 분리해 한 실험의 실패가 다음 모델 실행을 막지 않게 한다.
        cmd = build_cmd(model_id, args)
        t0 = time.time()

        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

        elapsed = time.time() - t0

        if proc.returncode != 0:
            print(f"  [FAIL] {model_id} (exit={proc.returncode}, {elapsed:.1f}s)")
            failed.append(model_id)
            continue

        print(f"  [OK] {model_id} ({elapsed:.1f}s)")

        summary = _load_summary(args.output_root, model_id)
        if summary:
            results.append(_flatten_summary(model_id, summary))

    # 개별 run_summary를 모아 최종 comparison_results.csv를 갱신한다.
    if results:
        print(f"\n{'='*60}")
        print("전체 비교 결과")
        print(f"{'='*60}")
        print_table(results)

        import csv
        comparison_path = out_root / "comparison_results.csv"
        fieldnames = list(results[0].keys())
        with comparison_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n비교 CSV 저장: {comparison_path}")

    if failed:
        print(f"\n[실패 모델] {', '.join(failed)}")

    print(f"\n완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="JamJamBeat 전체 모델 파이프라인 순차 실행 및 비교"
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help=f"실행할 모델 지정 (미지정 시 전체). 선택: {ALL_MODELS}",
    )
    parser.add_argument(
        "--csv-path", action="append", dest="csv_path", default=[],
        help="입력 CSV 경로 (반복 사용 가능). 미지정 시 DEFAULT_INPUTS 사용.",
    )
    parser.add_argument("--output-root", default="model/model_evaluation/pipelines")

    # 학습 하이퍼파라미터
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--patience",    type=int,   default=6)
    parser.add_argument("--focal-gamma", type=float, default=2.0)

    # 시퀀스 / 이미지
    parser.add_argument("--seq-len",     type=int, default=8)
    parser.add_argument("--seq-stride",  type=int, default=2)
    parser.add_argument("--image-size",  type=int, default=96)

    # 후처리 파라미터
    parser.add_argument("--tau",         type=float, default=0.90)
    parser.add_argument("--vote-n",      type=int,   default=7)
    parser.add_argument("--debounce-k",  type=int,   default=5)
    parser.add_argument("--fallback-fps",type=float, default=30.0)

    # 공통
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--device",      default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int,   default=0)

    args = parser.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
