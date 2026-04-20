#!/usr/bin/env python3
"""
Run the 4 topology-aware raw-landmark pipelines with explicit train/test CSVs.

- Input: one train CSV and one test CSV
- Validation split: disabled
- Checkpoint policy: last epoch
- Output: suite-level comparison metadata + per-model evaluation artifacts
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import run_all as suite_utils
import run_pipeline as rp


PROJECT_ROOT = rp.PROJECT_ROOT
KST = timezone(timedelta(hours=9))

MODEL_CHOICES = [
    "edge_stgu_mlp",
    "sparse_masked_mlp",
    "hierarchical_tree_mlp",
    "lappe_dist_mixer",
]

DEFAULT_LOSS_TYPE = "cross_entropy"
DEFAULT_USE_WEIGHTED_SAMPLER = False
DEFAULT_USE_ALPHA = False
DEFAULT_USE_LABEL_SMOOTHING = True
DEFAULT_LABEL_SMOOTHING_VALUE = rp.DEFAULT_LABEL_SMOOTHING
DEFAULT_PATIENCE = 6

MODEL_SET_RATIONALE = {
    "edge_stgu_mlp": (
        "SiT-MLP의 STGU 아이디어를 손 그래프용 edge-wise gated message passing으로 축소한 "
        "topology-aware MLP 실험군."
    ),
    "sparse_masked_mlp": (
        "MADE식 masked weight 제약을 손 adjacency에 맞게 적용해 연결된 관절만 직접 "
        "영향을 주도록 한 구조 prior 실험군."
    ),
    "hierarchical_tree_mlp": (
        "hand morphology와 directed skeleton routing 문헌을 참고해 parent-to-child "
        "전파를 고정한 tree prior 실험군."
    ),
    "lappe_dist_mixer": (
        "Laplacian PE와 SPD structural encoding을 DistMixer 형태로 결합해 global role과 "
        "relative distance를 함께 쓰는 구조 인코딩 실험군."
    ),
}

MODEL_SET_CAVEAT = (
    "이 suite는 4개 모두 topology-aware 변형이라, plain MLP anchor baseline 없이 "
    "구조 prior 자체의 순증가 효과를 분리 해석하는 데는 한계가 있다."
)

RESEARCH_SOURCES = [
    {
        "label": "SiT-MLP",
        "url": "https://arxiv.org/abs/2308.16018",
        "note": "Topology gating 기반 skeleton MLP 대안.",
    },
    {
        "label": "MADE",
        "url": "https://proceedings.mlr.press/v37/germain15.html",
        "note": "Masked weight 제약은 단순하고 GPU 친화적이라는 근거.",
    },
    {
        "label": "Graph Transformer with Laplacian PE",
        "url": "https://arxiv.org/abs/2012.09699",
        "note": "Laplacian eigenvectors를 graph positional encoding으로 사용.",
    },
    {
        "label": "Graphormer",
        "url": "https://arxiv.org/abs/2106.05234",
        "note": "SPD 기반 structural encoding의 대표 사례.",
    },
    {
        "label": "Benchmarking GNNs",
        "url": "https://jmlr.org/papers/v24/22-0567.html",
        "note": "구조 인코딩의 가치와 plain baseline 비교 필요성을 보여주는 기준 문헌.",
    },
    {
        "label": "Back to MLP",
        "url": "https://arxiv.org/abs/2207.01567",
        "note": "단순 MLP baseline도 강력할 수 있다는 참고 사례.",
    },
]


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", text.strip()).strip("-")
    return slug or "dataset"


def _resolve_path(path_str: str) -> Path:
    return rp.resolve_role_path(path_str)


def _canonical_feature_columns(df: pd.DataFrame) -> list[str]:
    base_cols = ["source_file", "frame_idx", "timestamp", "gesture", *rp.RAW_JOINT_COLS]
    if "__source_group" in df.columns:
        base_cols.append("__source_group")
    return [col for col in base_cols if col in df.columns]


def _load_explicit_csv(path: Path) -> pd.DataFrame:
    df = rp.normalize_source_groups(rp.load_preprocessed_data([path]))
    keep_cols = _canonical_feature_columns(df)
    return df.loc[:, keep_cols].copy()


def _infer_dataset_key(train_csv_path: Path) -> str:
    stem = train_csv_path.stem
    if stem.endswith("_train"):
        return stem[: -len("_train")]
    return stem


def _empty_like(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[0:0].copy()


def _random_train_val_split(
    df: pd.DataFrame,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, str, bool]:
    if len(df) < 2:
        raise ValueError("Train CSV must contain at least 2 rows to build a train/val split.")

    rng = rp.np.random.default_rng(seed)
    group_col = "__source_group" if "__source_group" in df.columns else None

    if group_col and df[group_col].nunique() >= 2:
        groups = list(df[group_col].dropna().astype(str).unique())
        rng.shuffle(groups)
        val_group_count = max(1, int(round(len(groups) * val_ratio)))
        val_group_count = min(val_group_count, len(groups) - 1)

        val_groups = set(groups[:val_group_count])
        train_df = df[~df[group_col].astype(str).isin(val_groups)].copy()
        val_df = df[df[group_col].astype(str).isin(val_groups)].copy()

        if len(train_df) > 0 and len(val_df) > 0:
            return (
                train_df.reset_index(drop=True),
                val_df.reset_index(drop=True),
                "random_group_split_8_2_from_train_csv",
                True,
            )

    indices = rng.permutation(len(df))
    val_size = max(1, int(round(len(df) * val_ratio)))
    val_size = min(val_size, len(df) - 1)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        "random_row_split_8_2_from_train_csv",
        False,
    )


def _build_suite_dir(base_output_root: Path, train_csv: Path, test_csv: Path) -> Path:
    timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    suite_name = f"{timestamp}__{_slugify(train_csv.stem)}__{_slugify(test_csv.stem)}"
    return base_output_root / suite_name


def _resolve_device(device_str: str) -> rp.torch.device:
    if device_str == "auto":
        try:
            use_cuda = rp.torch.cuda.is_available()
        except Exception:
            use_cuda = False
        return rp.torch.device("cuda" if use_cuda else "cpu")
    return rp.torch.device(device_str)


def _validate_label_smoothing_value(value: float) -> float:
    value = float(value)
    if not (0.0 <= value < 1.0):
        raise ValueError(f"--label-smoothing must satisfy 0.0 <= value < 1.0, got {value}")
    return value


def _validate_split_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    train_path: Path,
    test_path: Path,
) -> None:
    train_cols = list(train_df.columns)
    test_cols = list(test_df.columns)
    if train_cols != test_cols:
        raise ValueError(
            f"Column mismatch between train and test: {train_path.name} vs {test_path.name}"
        )


def _instantiate_new_model_from_state_dict(
    model_id: str,
    num_classes: int,
    model_overrides: dict[str, Any] | None = None,
) -> rp.nn.Module:
    mod = importlib.import_module(f"{model_id}.model")
    overrides = dict(model_overrides or {})

    if model_id == "edge_stgu_mlp":
        return mod.EdgeSTGUMLP(num_classes=num_classes, **overrides)
    if model_id == "sparse_masked_mlp":
        return mod.SparseMaskedMLP(num_classes=num_classes, **overrides)
    if model_id == "hierarchical_tree_mlp":
        return mod.HierarchicalTreeMLP(num_classes=num_classes, **overrides)
    if model_id == "lappe_dist_mixer":
        return mod.LapPEDistMixer(num_classes=num_classes, **overrides)

    raise ValueError(f"Unsupported model_id for reload verification: {model_id}")


def _build_training_recipe(
    *,
    labels: rp.np.ndarray,
    num_classes: int,
    device: rp.torch.device,
    loss_type: str,
    use_weighted_sampler: bool,
    use_alpha: bool,
    use_label_smoothing: bool,
    label_smoothing_value: float,
    focal_gamma: float,
) -> tuple[rp.WeightedRandomSampler | None, bool, rp.nn.Module, dict[str, Any]]:
    sampler = rp.create_weighted_sampler(labels) if use_weighted_sampler else None
    shuffle_train = sampler is None
    alpha = rp.compute_alpha(labels, num_classes=num_classes, device=device) if use_alpha else None
    label_smoothing = label_smoothing_value if use_label_smoothing else 0.0

    if loss_type == "cross_entropy":
        criterion = rp.nn.CrossEntropyLoss(weight=alpha, label_smoothing=label_smoothing)
    elif loss_type == "focal":
        criterion = rp.FocalLoss(
            alpha=alpha,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    recipe_meta = {
        "loss_type": loss_type,
        "sampler_policy": "weighted_sampler" if use_weighted_sampler else "shuffle",
        "alpha_policy": "enabled" if use_alpha else "disabled",
        "label_smoothing_enabled": bool(use_label_smoothing),
        "configured_label_smoothing": float(label_smoothing_value),
        "effective_label_smoothing": float(label_smoothing),
        "effective_focal_gamma": float(focal_gamma),
    }
    return sampler, shuffle_train, criterion, recipe_meta


def _write_suite_manifest(
    *,
    suite_dir: Path,
    dataset_key: str,
    train_csv: Path,
    test_csv: Path,
    models_requested: list[str],
    status: str,
    training_recipe: dict[str, object],
    split_policy: str,
    fixed_video_level_split: bool,
    failed_models: list[str] | None = None,
    comparison_path: Path | None = None,
) -> None:
    manifest = {
        "suite_name": suite_dir.name,
        "created_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
        "status": status,
        "suite_dir": str(suite_dir),
        "dataset_key": dataset_key,
        "normalization_family": rp.infer_normalization_family(dataset_key),
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "dataset_files": {
            "train": str(train_csv),
            "test": str(test_csv),
        },
        "models_requested": models_requested,
        "failed_models": failed_models or [],
        "comparison_results_csv": str(comparison_path) if comparison_path else None,
        "fixed_video_level_split": fixed_video_level_split,
        "source_counts": {
            "train": suite_utils.count_sources_in_csv(train_csv),
            "test": suite_utils.count_sources_in_csv(test_csv),
        },
        "test_kind": "static_images_63d",
        "official_ranking_basis": "test_csv_static_images",
        "split_policy": split_policy,
        "validation_from_train_csv": True,
        "train_val_ratio": {"train": 0.8, "val": 0.2},
        "training_recipe": training_recipe,
        "model_set_rationale": MODEL_SET_RATIONALE,
        "model_set_caveat": MODEL_SET_CAVEAT,
        "research_sources": RESEARCH_SOURCES,
    }
    with (suite_dir / "comparison_suite.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _run_one_model(
    *,
    args: argparse.Namespace,
    model_id: str,
    suite_dir: Path,
    train_csv: Path,
    test_csv: Path,
) -> dict[str, Any]:
    rp.set_seed(args.seed)
    device = _resolve_device(args.device)

    input_roles = {
        "train": train_csv,
        "test": test_csv,
    }
    train_df = _load_explicit_csv(train_csv)
    test_df = _load_explicit_csv(test_csv)
    _validate_split_columns(train_df, test_df, train_path=train_csv, test_path=test_csv)

    train_df, val_df, split_policy, fixed_video_level_split = _random_train_val_split(
        train_df,
        val_ratio=0.2,
        seed=args.seed,
    )
    split = rp.SplitData(
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
    )
    merged_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    angle_cols = rp.detect_angle_cols(merged_df)

    dataset_key = _infer_dataset_key(train_csv)
    normalization_family = rp.infer_normalization_family(dataset_key)
    model_overrides = rp.parse_model_overrides(getattr(args, "model_overrides", None))
    dataset_info = rp.build_dataset_info(
        input_roles=input_roles,
        merged_df=merged_df,
        split=split,
        dataset_key=dataset_key,
        normalization_family=normalization_family,
        inference_df=None,
    )
    dataset_info["explicit_no_val_split"] = False
    dataset_info["validation_from_train_csv"] = True
    dataset_info["train_val_ratio"] = {"train": 0.8, "val": 0.2}
    dataset_info["split_policy"] = split_policy
    dataset_info["test_csv"] = str(test_csv)
    dataset_info["fixed_video_level_split"] = fixed_video_level_split

    class_names = args.class_names if args.class_names else rp.DEFAULT_CLASS_NAMES
    num_classes = len(class_names)
    label_smoothing_value = _validate_label_smoothing_value(args.label_smoothing)

    timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    run_dir = suite_dir / model_id / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    model, mode, train_ds, val_ds, test_ds = rp.build_experiment(
        model_id=model_id,
        split=split,
        angle_cols=angle_cols,
        seq_len=args.seq_len,
        seq_stride=args.seq_stride,
        image_size=args.image_size,
        num_classes=num_classes,
        test_sequence_policy="independent_repeat",
        model_overrides=model_overrides,
    )
    model = model.to(device)

    dataset_info["dataset_sample_counts"] = {
        "train": int(len(train_ds)),
        "val": int(len(val_ds)),
        "test": int(len(test_ds)),
    }

    train_labels = rp.get_dataset_labels(train_ds, model_id=model_id, split_name="train")
    sampler, shuffle_train, criterion, recipe_meta = _build_training_recipe(
        labels=train_labels,
        num_classes=num_classes,
        device=device,
        loss_type=args.loss_type,
        use_weighted_sampler=args.use_weighted_sampler,
        use_alpha=args.use_alpha,
        use_label_smoothing=args.use_label_smoothing,
        label_smoothing_value=label_smoothing_value,
        focal_gamma=args.focal_gamma,
    )

    pin_memory = device.type == "cuda"
    train_loader = rp.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle_train,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = rp.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = rp.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    optimizer = rp.build_optimizer(
        model,
        optimizer_name=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )
    scheduler = rp.torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    history: list[dict[str, Any]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    stale = 0
    early_stopping_triggered = False

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = rp.train_one_epoch(model, train_loader, optimizer, criterion, mode, device)
        va_loss, va_acc = rp.validate_one_epoch(model, val_loader, criterion, mode, device)
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": va_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"[{model_id}] epoch {epoch:03d}/{args.epochs} "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
        )

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if args.patience > 0 and stale >= args.patience:
                early_stopping_triggered = True
                print(f"[{model_id}] early stopping at epoch={epoch}")
                break

    model.load_state_dict(best_state)

    preds_df = rp.predict_dataset(
        model,
        test_loader,
        test_ds,
        mode=mode,
        num_classes=num_classes,
        device=device,
    )
    preds_path = run_dir / "preds_test.csv"
    preds_df.to_csv(preds_path, index=False)

    eval_dir = run_dir / "evaluation"
    eval_cfg = rp.EvaluationConfig(
        class_names=class_names,
        neutral_class_id=args.neutral_class_id,
        tau=args.tau,
        vote_n=args.vote_n,
        debounce_k=args.debounce_k,
        fallback_fps=args.fallback_fps,
        dataset_info=dataset_info,
    )
    metrics_summary = rp.evaluate_predictions(preds_df, eval_dir, eval_cfg)

    ckpt_path = run_dir / "model.pt"
    best_state_verification = rp.fingerprint_state_dict(best_state)
    checkpoint_payload = {
        "model_id": model_id,
        "model_state_dict": best_state,
        "class_names": class_names,
        "mode": mode,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "seq_stride": args.seq_stride,
        "image_size": args.image_size,
        "model_overrides": model_overrides,
        "checkpoint_policy": "best_val_loss",
        "checkpoint_verification": {
            **best_state_verification,
            "model_id": model_id,
            "mode": mode,
            "seq_len": int(args.seq_len),
            "image_size": int(args.image_size),
        },
    }
    rp.torch.save(checkpoint_payload, ckpt_path)

    saved_checkpoint = rp.safe_torch_load(ckpt_path, rp.torch.device("cpu"))
    saved_state_dict = saved_checkpoint["model_state_dict"]
    saved_state_verification = rp.fingerprint_state_dict(saved_state_dict)
    reloaded_model = _instantiate_new_model_from_state_dict(
        model_id,
        num_classes=num_classes,
        model_overrides=dict(saved_checkpoint.get("model_overrides") or model_overrides or {}),
    )
    strict_reload_info = rp.strict_load_state_dict(reloaded_model, saved_state_dict)
    stored_checkpoint_verification = dict(saved_checkpoint.get("checkpoint_verification") or {})
    checkpoint_verification = {
        "checkpoint_path": str(ckpt_path),
        "model_id": str(saved_checkpoint.get("model_id") or model_id),
        "mode": str(saved_checkpoint.get("mode") or mode),
        "seq_len": int(saved_checkpoint.get("seq_len") or args.seq_len),
        "image_size": int(saved_checkpoint.get("image_size") or args.image_size),
        **saved_state_verification,
        "saved_matches_best_state": (
            saved_state_verification["checkpoint_fingerprint"]
            == best_state_verification["checkpoint_fingerprint"]
        ),
        "stored_matches_saved_state": (
            stored_checkpoint_verification.get("checkpoint_fingerprint")
            == saved_state_verification["checkpoint_fingerprint"]
        ),
        "stored_checkpoint_fingerprint": stored_checkpoint_verification.get("checkpoint_fingerprint"),
        **strict_reload_info,
    }

    pd.DataFrame(history).to_csv(run_dir / "train_history.csv", index=False)

    run_summary = {
        "model_id": model_id,
        "dataset_key": dataset_key,
        "normalization_family": normalization_family,
        "mode": mode,
        "device": str(device),
        "inputs": [str(train_csv), str(test_csv)],
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "optimizer": args.optimizer,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "loss_type": args.loss_type,
            "use_weighted_sampler": args.use_weighted_sampler,
            "use_alpha": args.use_alpha,
            "use_label_smoothing": args.use_label_smoothing,
            "focal_gamma": args.focal_gamma,
            "model_overrides": model_overrides,
            "configured_label_smoothing": recipe_meta["configured_label_smoothing"],
            "label_smoothing": recipe_meta["effective_label_smoothing"],
            "sampler_policy": recipe_meta["sampler_policy"],
            "alpha_policy": recipe_meta["alpha_policy"],
            "tau": args.tau,
            "vote_n": args.vote_n,
            "debounce_k": args.debounce_k,
            "fallback_fps": args.fallback_fps,
            "validation_from_train_csv": True,
            "train_val_ratio": {"train": 0.8, "val": 0.2},
            "split_policy": split_policy,
            "checkpoint_policy": "best_val_loss",
        },
        "split_sizes": {
            "train": int(len(split.train_df)),
            "val": int(len(split.val_df)),
            "test": int(len(split.test_df)),
        },
        "dataset_info": dataset_info,
        "dataset_sizes": {
            "train": int(len(train_ds)),
            "val": int(len(val_ds)),
            "test": int(len(test_ds)),
        },
        "fixed_video_level_split": fixed_video_level_split,
        "source_counts": {
            "train": int(dataset_info["source_counts"]["train"]),
            "val": int(dataset_info["source_counts"]["val"]),
            "test": int(dataset_info["source_counts"]["test"]),
        },
        "test_kind": "static_images_63d",
        "test_sequence_policy": "independent_repeat",
        "inference_sequence_policy": None,
        "official_ranking_basis": "test_csv_static_images",
        "inference_used": False,
        "preds_inference_csv": None,
        "best_val_loss": float(best_val_loss),
        "epochs_ran": int(len(history)),
        "early_stopping_triggered": early_stopping_triggered,
        "checkpoint_policy": "best_val_loss",
        "output_dir": str(run_dir),
        "checkpoint_verification": checkpoint_verification,
        "metrics": metrics_summary,
    }

    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    latest_info = suite_dir / model_id / "latest.json"
    latest_info.parent.mkdir(parents=True, exist_ok=True)
    with latest_info.open("w", encoding="utf-8") as f:
        json.dump({"latest_run": str(run_dir)}, f, ensure_ascii=False, indent=2)

    print(f"[{model_id}] done. Output: {run_dir}")
    return run_summary


def run_suite(args: argparse.Namespace) -> Path:
    train_csv = _resolve_path(args.train_csv)
    test_csv = _resolve_path(args.test_csv)
    preview_train_df = _load_explicit_csv(train_csv)
    _, _, split_policy, fixed_video_level_split = _random_train_val_split(
        preview_train_df,
        val_ratio=0.2,
        seed=args.seed,
    )

    base_output_root = Path(args.output_root)
    if not base_output_root.is_absolute():
        base_output_root = PROJECT_ROOT / base_output_root
    base_output_root.mkdir(parents=True, exist_ok=True)

    dataset_key = _infer_dataset_key(train_csv)
    suite_dir = _build_suite_dir(base_output_root, train_csv, test_csv)
    suite_dir.mkdir(parents=True, exist_ok=True)
    suite_utils.write_latest_suite(base_output_root, suite_dir)

    training_recipe = {
        "loss_type": args.loss_type,
        "use_weighted_sampler": args.use_weighted_sampler,
        "use_alpha": args.use_alpha,
        "use_label_smoothing": args.use_label_smoothing,
        "label_smoothing_value": _validate_label_smoothing_value(args.label_smoothing),
        "focal_gamma": args.focal_gamma,
        "optimizer": args.optimizer,
        "momentum": args.momentum,
        "model_overrides": rp.parse_model_overrides(getattr(args, "model_overrides", None)),
        "label_smoothing_default": DEFAULT_LABEL_SMOOTHING_VALUE,
        "patience": args.patience,
        "checkpoint_policy": "best_val_loss",
    }
    _write_suite_manifest(
        suite_dir=suite_dir,
        dataset_key=dataset_key,
        train_csv=train_csv,
        test_csv=test_csv,
        models_requested=args.models,
        status="running",
        training_recipe=training_recipe,
        split_policy=split_policy,
        fixed_video_level_split=fixed_video_level_split,
    )

    print(f"\n{'=' * 72}")
    print("JamJamBeat topology-aware explicit train/test suite")
    print(f"models: {len(args.models)}")
    print(f"train: {train_csv}")
    print(f"test : {test_csv}")
    print(f"output: {suite_dir}")
    print(f"{'=' * 72}\n")

    results: list[dict[str, Any]] = []
    failed: list[str] = []

    for index, model_id in enumerate(args.models, 1):
        print(f"\n[{index}/{len(args.models)}] {model_id} ...")
        t0 = time.time()
        try:
            summary = _run_one_model(
                args=args,
                model_id=model_id,
                suite_dir=suite_dir,
                train_csv=train_csv,
                test_csv=test_csv,
            )
        except Exception as exc:
            failed.append(model_id)
            print(f"[WARN] {model_id} failed: {exc}")
            continue

        elapsed = time.time() - t0
        print(f"[{model_id}] elapsed: {elapsed:.1f}s")
        results.append(suite_utils._flatten_summary(model_id, summary))

    comparison_path = suite_dir / "comparison_results.csv"
    if results:
        pd.DataFrame(results).sort_values("accuracy", ascending=False).to_csv(
            comparison_path,
            index=False,
        )
        suite_utils.print_table(results)

    status = "completed" if not failed else "partial_failed"
    _write_suite_manifest(
        suite_dir=suite_dir,
        dataset_key=dataset_key,
        train_csv=train_csv,
        test_csv=test_csv,
        models_requested=args.models,
        status=status,
        training_recipe=training_recipe,
        split_policy=split_policy,
        fixed_video_level_split=fixed_video_level_split,
        failed_models=failed,
        comparison_path=comparison_path if results else None,
    )

    if failed:
        print(f"\n[WARN] failed models: {', '.join(failed)}")
    print(f"\n[done] suite output: {suite_dir}")
    return suite_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="새 4개 raw-landmark 파이프라인 전용 runner (explicit train/test + random 8:2 val).",
    )
    parser.add_argument("--train-csv", type=str, required=True, help="Train CSV path.")
    parser.add_argument("--test-csv", type=str, required=True, help="Test CSV path.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_CHOICES),
        choices=MODEL_CHOICES,
        help="실행할 모델 목록. 기본값: 4개 전부",
    )
    parser.add_argument("--output-root", type=str, default="model/model_evaluation/pipelines")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer",
        type=str,
        default=rp.DEFAULT_OPTIMIZER,
        choices=rp.OPTIMIZER_CHOICES,
        help=f"optimizer 종류. 기본값: {rp.DEFAULT_OPTIMIZER}",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD/RMSprop용 momentum. 기본값: 0.9",
    )
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help=f"early stopping patience. 기본값: {DEFAULT_PATIENCE}")
    parser.add_argument(
        "--loss-type",
        type=str,
        default=DEFAULT_LOSS_TYPE,
        choices=rp.LOSS_TYPE_CHOICES,
        help=f"학습 loss 종류. 기본값: {DEFAULT_LOSS_TYPE}",
    )
    parser.add_argument(
        "--use-weighted-sampler",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_WEIGHTED_SAMPLER,
        help=f"train loader에 weighted sampler 사용 여부. 기본값: {DEFAULT_USE_WEIGHTED_SAMPLER}",
    )
    parser.add_argument(
        "--use-alpha",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_ALPHA,
        help=f"class alpha(weight) 사용 여부. 기본값: {DEFAULT_USE_ALPHA}",
    )
    parser.add_argument(
        "--use-label-smoothing",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_LABEL_SMOOTHING,
        help=(
            "label smoothing 사용 여부. "
            f"활성화 시 smoothing={DEFAULT_LABEL_SMOOTHING_VALUE}"
        ),
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=DEFAULT_LABEL_SMOOTHING_VALUE,
        help=f"label smoothing 값. 기본값: {DEFAULT_LABEL_SMOOTHING_VALUE}",
    )
    parser.add_argument("--focal-gamma", type=float, default=rp.DEFAULT_FOCAL_GAMMA, help=argparse.SUPPRESS)

    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--seq-stride", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=96)

    parser.add_argument("--neutral-class-id", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.90)
    parser.add_argument("--vote-n", type=int, default=7)
    parser.add_argument("--debounce-k", type=int, default=5)
    parser.add_argument("--fallback-fps", type=float, default=30.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--class-names", nargs="*", default=[])
    parser.add_argument(
        "--model-overrides",
        type=str,
        default="",
        help="모델 생성자 override JSON 문자열 또는 JSON 파일 경로.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_suite(args)


if __name__ == "__main__":
    main()
