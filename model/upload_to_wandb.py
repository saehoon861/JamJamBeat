# upload_to_wandb.py - 모델 평가 결과를 wandb에 업로드
import json
import os
import sys
from pathlib import Path

import pandas as pd
import wandb

ENTITY = "sgim49697-hancom"
PROJECT = "JamJamBeat"
PIPELINES_DIR = Path(__file__).parent / "model_evaluation" / "pipelines"


def detect_suite_name(path: Path) -> str:
    """legacy flat 구조와 suite 구조를 모두 지원하는 suite 식별자."""
    rel = path.relative_to(PIPELINES_DIR)
    if path.name == "run_summary.json" and len(rel.parts) >= 4:
        return rel.parts[0]
    if path.name == "comparison_results.csv" and len(rel.parts) >= 2:
        return rel.parts[0]
    return "legacy-flat"


def upload_model_run(summary_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model_id = summary["model_id"]
    metrics = summary["metrics"]
    latency = metrics.get("latency", {})
    class0 = metrics.get("class0_metrics", {})
    fp_per_min = metrics.get("fp_per_min_metrics", {}).get("fp_per_min", 0.0)
    hyperparams = summary.get("hyperparameters", {})
    suite_name = detect_suite_name(summary_path)
    run_name = f"{suite_name}__{model_id}" if suite_name != "legacy-flat" else model_id

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=run_name,
        group=suite_name,
        tags=[summary["mode"], "cpu", suite_name],
        config={
            "model_id": model_id,
            "suite_name": suite_name,
            "mode": summary["mode"],
            "device": summary.get("device", "cpu"),
            "epochs_ran": summary["epochs_ran"],
            "train_samples": summary["split_sizes"]["train"],
            "val_samples": summary["split_sizes"]["val"],
            "test_samples": summary["split_sizes"]["test"],
            **hyperparams,
        },
        reinit=True,
    )

    # 1. 학습 과정 지표 업로드 (train_history.csv) - 학습 곡선 및 learning rate 그래프
    history_path = summary_path.parent / "train_history.csv"
    if history_path.exists():
        history_df = pd.read_csv(history_path)
        for _, row in history_df.iterrows():
            step = int(row["epoch"])
            run.log({
                "train_loss": row["train_loss"],
                "train_acc": row["train_acc"],
                "val_loss": row["val_loss"],
                "val_acc": row["val_acc"],
                "lr": row["lr"],
                "epoch": step,
            }, step=step)

    # 2. 최종 평가 지표 업로드
    run.log({
        # 주요 지표
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_avg"]["f1"],
        "macro_precision": metrics["macro_avg"]["precision"],
        "macro_recall": metrics["macro_avg"]["recall"],
        # neutral 클래스 오류율
        "class0_fpr": class0.get("false_positive_rate", 0.0),
        "class0_fnr": class0.get("false_negative_rate", 0.0),
        # 실서비스 기준
        "fp_per_min": fp_per_min,
        # 레이턴시
        "latency_p50_ms": latency.get("p50_ms", 0.0),
        "latency_p95_ms": latency.get("p95_ms", 0.0),
        "latency_mean_ms": latency.get("mean_ms", 0.0),
        # 학습 정보
        "best_val_loss": summary["best_val_loss"],
        "epochs_ran": summary["epochs_ran"],
        # PoC 기준 달성 여부
        "poc_macro_f1_pass": int(metrics["macro_avg"]["f1"] >= 0.80),
        "poc_class0_fnr_pass": int(class0.get("false_negative_rate", 1.0) < 0.10),
        "poc_fp_per_min_pass": int(fp_per_min < 2.0),
        "poc_latency_pass": int(latency.get("p95_ms", 999.0) < 200.0),
    })

    run.finish()
    print(f"  ✅ {run_name}")


def upload_summary_table(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    suite_name = detect_suite_name(csv_path)
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=f"comparison-summary-{suite_name}",
        group=suite_name,
        tags=["summary", suite_name],
        reinit=True,
    )
    table = wandb.Table(dataframe=df)
    run.log({"model_comparison": table})
    run.finish()
    print(f"  ✅ comparison-summary table ({suite_name})")


def main():
    print(f"[wandb] entity={ENTITY}, project={PROJECT}")
    print("[wandb] 모델별 run 업로드 중...")

    summary_paths = sorted(PIPELINES_DIR.rglob("run_summary.json"))
    if not summary_paths:
        print("[ERROR] run_summary.json 파일을 찾을 수 없습니다.")
        sys.exit(1)

    for path in summary_paths:
        upload_model_run(path)

    print("[wandb] 전체 비교 테이블 업로드 중...")
    comparison_paths = sorted(PIPELINES_DIR.rglob("comparison_results.csv"))
    for csv_path in comparison_paths:
        upload_summary_table(csv_path)

    print(f"\n[완료] https://wandb.ai/{ENTITY}/{PROJECT}")


if __name__ == "__main__":
    main()
