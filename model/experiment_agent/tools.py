# tools.py - runtime helpers for the JamJamBeat autonomous experiment agent
from __future__ import annotations

import fcntl
import itertools
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, deque
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import mlflow
import optuna
import pandas as pd
import yaml
from optuna.trial import TrialState

from . import registry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "model"
MODEL_PIPELINES_DIR = MODEL_ROOT / "model_pipelines"
if str(MODEL_PIPELINES_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_PIPELINES_DIR))

import run_pipeline as rp  # noqa: E402


KST = timezone(timedelta(hours=9))
AGENT_RUNS_ROOT = MODEL_ROOT / "model_evaluation" / "agent_runs"
PIPELINE_MANAGED_ROOT = MODEL_ROOT / "model_evaluation" / "pipelines" / "agent_managed"
LATEST_AGENT_RUN_PATH = AGENT_RUNS_ROOT / "latest_agent_run.json"
ROLE_DATASET_DIRNAME = "role_datasets"
TRIAL_RUNTIME_DIRNAME = "trial_runtime"
ROLE_DATASET_CACHE_VERSION = 2


def now_kst() -> datetime:
    return datetime.now(KST)


def iso_now() -> str:
    return now_kst().isoformat(timespec="seconds")


def ensure_text_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def lock_path(path: Path) -> Path:
    return path.parent / f".{path.name}.lock"


@contextmanager
def advisory_lock(path: Path):
    ensure_text_dir(path)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def metadata_lock(agent_dir: Path):
    with advisory_lock(lock_path(agent_dir / "state.json")):
        yield


def atomic_write_text(path: Path, content: str) -> None:
    ensure_text_dir(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return deepcopy(default)
    return json.loads(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_text_dir(path)
    with advisory_lock(lock_path(path)):
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")


def worker_log(agent_dir: Path, message: str, *, slot: int | None = None) -> None:
    slot_label = f"[worker_{slot}]" if slot is not None else "[agent]"
    print(f"[{iso_now()}] [{agent_dir.name}] {slot_label} {message}", flush=True)


def slugify(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in text.strip())
    return safe.strip("-") or "unnamed"


def is_process_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def resolve_agent_dir(run_id: str) -> Path:
    return AGENT_RUNS_ROOT / run_id


def load_state(agent_dir: Path) -> dict[str, Any]:
    return read_json(agent_dir / "state.json", default={})


def save_state(agent_dir: Path, state: dict[str, Any]) -> None:
    write_json(agent_dir / "state.json", state)


def _normalize_worker_pids(state: dict[str, Any]) -> dict[str, int]:
    normalized: dict[str, int] = {}
    raw = state.get("worker_pids")
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                pid = int(value)
            except (TypeError, ValueError):
                continue
            if pid > 0:
                normalized[str(key)] = pid
    legacy = state.get("worker_pid")
    if legacy and not normalized:
        try:
            pid = int(legacy)
        except (TypeError, ValueError):
            pid = 0
        if pid > 0:
            normalized["0"] = pid
    return normalized


def _set_worker_pid(state: dict[str, Any], slot: int, pid: int | None) -> dict[str, Any]:
    state = deepcopy(state)
    worker_pids = _normalize_worker_pids(state)
    slot_key = str(slot)
    if pid and pid > 0:
        worker_pids[slot_key] = int(pid)
    else:
        worker_pids.pop(slot_key, None)
    state["worker_pids"] = worker_pids
    alive = [int(value) for value in worker_pids.values() if int(value) > 0]
    state["worker_pid"] = alive[0] if alive else None
    return state


def update_state_locked(agent_dir: Path, mutator) -> dict[str, Any]:
    with metadata_lock(agent_dir):
        state = load_state(agent_dir)
        updated = mutator(deepcopy(state))
        save_state(agent_dir, updated)
        return updated


def _inflight_path(agent_dir: Path) -> Path:
    return agent_dir / "inflight_candidates.json"


def load_inflight(agent_dir: Path) -> list[dict[str, Any]]:
    payload = read_json(_inflight_path(agent_dir), default={"items": []})
    items = payload.get("items", []) if isinstance(payload, dict) else []
    return [dict(item) for item in items if isinstance(item, dict)]


def save_inflight(agent_dir: Path, items: list[dict[str, Any]]) -> None:
    write_json(_inflight_path(agent_dir), {"items": items})


def clear_inflight_for_slot(agent_dir: Path, slot: int) -> None:
    with metadata_lock(agent_dir):
        items = [item for item in load_inflight(agent_dir) if int(item.get("worker_slot", -1)) != int(slot)]
        save_inflight(agent_dir, items)


def candidate_fingerprint(candidate: dict[str, Any]) -> str:
    payload = {
        "runner_kind": candidate["runner_kind"],
        "model_id": candidate["model_id"],
        "scenario_name": candidate["scenario_name"],
        "hyperparameters": candidate["hyperparameters"],
        "model_overrides": candidate.get("model_overrides"),
        "mutation": candidate.get("mutation"),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def load_resolved_config(agent_dir: Path) -> dict[str, Any]:
    return read_yaml(agent_dir / "resolved_config.yaml")


def update_latest_agent_run(run_id: str) -> None:
    AGENT_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(LATEST_AGENT_RUN_PATH, {"latest_agent_run": run_id})


def resolve_latest_run_id() -> str:
    latest = read_json(LATEST_AGENT_RUN_PATH, default={})
    run_id = latest.get("latest_agent_run")
    if not run_id:
        raise ValueError("No latest agent run recorded yet.")
    return str(run_id)


def _canonical_feature_columns(df: pd.DataFrame) -> list[str]:
    base_cols = ["source_file", "frame_idx", "timestamp", "gesture", *rp.RAW_JOINT_COLS]
    return [col for col in base_cols if col in df.columns]


def load_explicit_frame_csv_with_stats(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(path)
    raw_rows = int(len(df))
    raw_columns = list(df.columns)
    required = {"gesture", "frame_idx", *rp.RAW_JOINT_COLS}
    missing = sorted(required - set(raw_columns))
    if missing:
        raise ValueError(f"{path} missing required raw-frame columns: {sorted(missing)}")

    working = df.copy()
    if "source_file" not in working.columns:
        working["source_file"] = path.stem
    if "timestamp" not in working.columns:
        working["timestamp"] = ""

    keep_cols = _canonical_feature_columns(working)
    normalized = working.loc[:, keep_cols].copy()
    normalized["source_file"] = normalized["source_file"].fillna(path.stem).astype(str)
    normalized["timestamp"] = normalized["timestamp"].fillna("").astype(str)
    timestamp_blank_rows = int((normalized["timestamp"] == "").sum())
    normalized["gesture"] = pd.to_numeric(normalized["gesture"], errors="coerce")
    normalized["frame_idx"] = pd.to_numeric(normalized["frame_idx"], errors="coerce")
    feature_cols = [*rp.RAW_JOINT_COLS]
    before_drop_rows = int(len(normalized))
    normalized = normalized.dropna(subset=feature_cols + ["gesture", "frame_idx"]).reset_index(drop=True)
    normalized["gesture"] = normalized["gesture"].astype(int)
    normalized["frame_idx"] = normalized["frame_idx"].astype(int)
    stats = {
        "path": str(path),
        "raw_rows": raw_rows,
        "raw_columns": raw_columns,
        "canonical_columns": keep_cols,
        "required_missing": missing,
        "rows_after_canonical": before_drop_rows,
        "rows_after_dropna": int(len(normalized)),
        "dropped_rows": before_drop_rows - int(len(normalized)),
        "timestamp_blank_rows": timestamp_blank_rows,
    }
    return normalized, stats


def load_explicit_frame_csv(path: Path) -> pd.DataFrame:
    normalized, _ = load_explicit_frame_csv_with_stats(path)
    return normalized


def random_train_val_split(
    df: pd.DataFrame,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if len(df) < 2:
        raise ValueError("Train CSV must contain at least 2 rows to build a train/val split.")

    rng = rp.np.random.default_rng(seed)
    group_col = "source_file" if "source_file" in df.columns and df["source_file"].nunique() >= 2 else None

    if group_col:
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
                {
                    "split_policy": "random_group_split_8_2_from_explicit_train_csv",
                    "fixed_video_level_split": True,
                },
            )

    indices = rng.permutation(len(df))
    val_size = max(1, int(round(len(df) * val_ratio)))
    val_size = min(val_size, len(df) - 1)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        {
            "split_policy": "random_row_split_8_2_from_explicit_train_csv",
            "fixed_video_level_split": False,
        },
    )


def _role_manifest_needs_rebuild(manifest: dict[str, Any]) -> bool:
    return int(manifest.get("cache_version") or 0) < ROLE_DATASET_CACHE_VERSION


def _validate_role_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    rows = dict(manifest.get("rows") or {})
    reasons: list[str] = []
    for split_name in ("train", "val", "test", "inference"):
        if int(rows.get(split_name) or 0) <= 0:
            reasons.append(f"{split_name}_rows_zero")

    input_stats = dict(manifest.get("input_stats") or {})
    for role_name in ("train", "test"):
        stats = dict(input_stats.get(role_name) or {})
        missing = list(stats.get("required_missing") or [])
        if missing:
            reasons.append(f"{role_name}_missing_required_columns")
        if role_name == "test":
            test_columns = set(stats.get("canonical_columns") or [])
            if "gesture" not in test_columns:
                reasons.append("test_missing_gesture")
            if not all(col in test_columns for col in rp.RAW_JOINT_COLS):
                reasons.append("test_missing_canonical_features")

    reasons = sorted(dict.fromkeys(reasons))
    return {
        "valid": not reasons,
        "reasons": reasons,
    }


def prepare_role_dataset_cache(agent_dir: Path, scenario: dict[str, Any], seed: int) -> dict[str, Any]:
    cache_root = agent_dir / ROLE_DATASET_DIRNAME / f"{slugify(scenario['name'])}__seed{seed}"
    manifest_path = cache_root / "manifest.json"
    with advisory_lock(lock_path(manifest_path)):
        if manifest_path.exists():
            manifest = read_json(manifest_path, default={})
            if manifest and not _role_manifest_needs_rebuild(manifest):
                manifest["_cache_status"] = "hit"
                manifest["viability"] = _validate_role_manifest(manifest)
                return manifest

        cache_root.mkdir(parents=True, exist_ok=True)
        train_csv = Path(scenario["train_csv"])
        test_csv = Path(scenario["test_csv"])
        base_name = slugify(scenario["name"])
        train_path = cache_root / f"{base_name}_train.csv"
        val_path = cache_root / f"{base_name}_val.csv"
        test_path = cache_root / f"{base_name}_test.csv"
        inference_path = cache_root / f"{base_name}_inference.csv"
        manifest: dict[str, Any] = {
            "cache_version": ROLE_DATASET_CACHE_VERSION,
            "scenario_name": scenario["name"],
            "seed": seed,
            "source_train_csv": str(train_csv),
            "source_test_csv": str(test_csv),
            "paths": {
                "train": str(train_path),
                "val": str(val_path),
                "test": str(test_path),
                "inference": str(inference_path),
            },
            "rows": {
                "train": 0,
                "val": 0,
                "test": 0,
                "inference": 0,
            },
        }

        try:
            train_df, train_stats = load_explicit_frame_csv_with_stats(train_csv)
            test_df, test_stats = load_explicit_frame_csv_with_stats(test_csv)
            train_split_df, val_split_df, split_meta = random_train_val_split(train_df, seed=seed)

            train_split_df.to_csv(train_path, index=False)
            val_split_df.to_csv(val_path, index=False)
            test_df.to_csv(test_path, index=False)
            test_df.to_csv(inference_path, index=False)

            manifest.update(
                {
                    "split_policy": split_meta["split_policy"],
                    "fixed_video_level_split": split_meta["fixed_video_level_split"],
                    "input_stats": {
                        "train": train_stats,
                        "test": test_stats,
                    },
                    "rows": {
                        "train": int(len(train_split_df)),
                        "val": int(len(val_split_df)),
                        "test": int(len(test_df)),
                        "inference": int(len(test_df)),
                    },
                }
            )
        except Exception as exc:
            manifest.update(
                {
                    "split_policy": "cache_build_failed",
                    "fixed_video_level_split": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        manifest["viability"] = _validate_role_manifest(manifest)
        if manifest.get("error"):
            manifest["viability"]["valid"] = False
            reasons = list(manifest["viability"].get("reasons") or [])
            reasons.append(f"cache_build_error:{manifest['error']}")
            manifest["viability"]["reasons"] = sorted(dict.fromkeys(reasons))
        write_json(manifest_path, manifest)
        manifest["_cache_status"] = "miss"
        return manifest


def expand_whitelist_files(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in PROJECT_ROOT.glob(pattern):
            if path.is_file():
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    files.append(resolved)
    return sorted(files)


def capture_snapshot(snapshot_dir: Path, patterns: list[str]) -> dict[str, Any]:
    files_dir = snapshot_dir / "files"
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    files_dir.mkdir(parents=True, exist_ok=True)

    rel_paths: list[str] = []
    for src in expand_whitelist_files(patterns):
        rel = src.relative_to(PROJECT_ROOT)
        dst = files_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        rel_paths.append(str(rel))

    manifest = {
        "created_at_kst": iso_now(),
        "paths": rel_paths,
        "patterns": patterns,
    }
    write_json(snapshot_dir / "manifest.json", manifest)
    return manifest


def restore_snapshot(snapshot_dir: Path) -> dict[str, Any]:
    manifest = read_json(snapshot_dir / "manifest.json", default={})
    for rel_str in manifest.get("paths", []):
        rel = Path(rel_str)
        src = snapshot_dir / "files" / rel
        dst = PROJECT_ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return manifest


def refresh_snapshot_from_current(agent_dir: Path, snapshot_name: str) -> dict[str, Any]:
    config = load_resolved_config(agent_dir)
    patterns = list(config["mutation"]["allowed_globs"])
    snapshot_dir = agent_dir / "snapshots" / snapshot_name
    return capture_snapshot(snapshot_dir, patterns)


def restore_named_snapshot(agent_dir: Path, snapshot_name: str) -> dict[str, Any]:
    return restore_snapshot(agent_dir / "snapshots" / snapshot_name)


def setup_agent_run(agent_dir: Path, resolved_config: dict[str, Any], source_config_path: Path) -> dict[str, Any]:
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "logs").mkdir(exist_ok=True)
    (agent_dir / "periodic_reports").mkdir(exist_ok=True)
    (agent_dir / "snapshots").mkdir(exist_ok=True)
    (agent_dir / ROLE_DATASET_DIRNAME).mkdir(exist_ok=True)
    (agent_dir / TRIAL_RUNTIME_DIRNAME).mkdir(exist_ok=True)
    (PIPELINE_MANAGED_ROOT / agent_dir.name).mkdir(parents=True, exist_ok=True)

    write_yaml(agent_dir / "resolved_config.yaml", resolved_config)
    write_json(agent_dir / "leaderboard.csv.json", {"columns": []})
    write_json(agent_dir / "champions.json", {"history": []})
    write_json(agent_dir / "study_summary.json", {"created_at_kst": iso_now(), "trial_count": 0})
    write_json(agent_dir / "golden_snapshot_manifest.json", {})
    write_json(_inflight_path(agent_dir), {"items": []})

    golden_manifest = capture_snapshot(
        agent_dir / "snapshots" / "golden",
        list(resolved_config["mutation"]["allowed_globs"]),
    )
    capture_snapshot(
        agent_dir / "snapshots" / "champion",
        list(resolved_config["mutation"]["allowed_globs"]),
    )
    write_json(agent_dir / "golden_snapshot_manifest.json", golden_manifest)

    state = {
        "run_id": agent_dir.name,
        "created_at_kst": iso_now(),
        "updated_at_kst": iso_now(),
        "status": "initialized",
        "source_config_path": str(source_config_path.resolve()),
        "resolved_config_path": str((agent_dir / "resolved_config.yaml").resolve()),
        "worker_pid": None,
        "worker_pids": {},
        "phase_name": "broad_search",
        "phase_trial_index": 0,
        "cycle_index": 0,
        "attempted_trials": 0,
        "completed_trials": 0,
        "failed_attempts_consecutive": 0,
        "mutation_failure_streak": 0,
        "current_backoff_seconds": 0,
        "backoff_index": 0,
        "next_retry_at_kst": None,
        "last_success_at_kst": None,
        "last_failure_at_kst": None,
        "last_golden_reset_at_kst": None,
        "last_champion_update_at_kst": None,
        "stop_requested": False,
        "recent_model_id": None,
        "recent_model_streak": 0,
        "recent_scenario_name": None,
        "recent_scenario_streak": 0,
        "pair_cooldowns": {},
        "invalid_role_scenarios": {},
        "goal": dict(resolved_config["goal"]),
        "best": None,
        "last_candidate": None,
    }
    save_state(agent_dir, state)
    append_jsonl(
        agent_dir / "decision_log.jsonl",
        {
            "event": "agent_initialized",
            "timestamp_kst": iso_now(),
            "scenario_count": len(resolved_config["scenarios"]["items"]),
            "role_model_count": len(resolved_config["models"]["role_based"]),
            "explicit_model_count": len(resolved_config["models"]["explicit_train_test"]),
        },
    )
    update_latest_agent_run(agent_dir.name)
    return state


def _load_leaderboard_rows(agent_dir: Path) -> list[dict[str, Any]]:
    leaderboard_path = agent_dir / "leaderboard.csv"
    if not leaderboard_path.exists():
        return []
    return pd.read_csv(leaderboard_path).to_dict("records")


def _write_leaderboard_rows(agent_dir: Path, rows: list[dict[str, Any]]) -> None:
    leaderboard_path = agent_dir / "leaderboard.csv"
    if not rows:
        df = pd.DataFrame(
            columns=[
                "candidate_id",
                "phase",
                "model_id",
                "scenario_name",
                "runner_kind",
                "accuracy",
                "macro_f1",
                "class0_fpr",
                "best_val_loss",
                "score",
                "output_dir",
                "trial_number",
                "loss_type",
                "optimizer",
            ]
        )
        tmp = leaderboard_path.with_suffix(".csv.tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(leaderboard_path)
        return
    df = pd.DataFrame(rows).sort_values(
        by=["macro_f1", "accuracy", "class0_fpr"],
        ascending=[False, False, True],
    )
    tmp = leaderboard_path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(leaderboard_path)


def get_model_spec(config: dict[str, Any], model_id: str) -> dict[str, Any]:
    for bucket in ("role_based", "explicit_train_test"):
        if model_id in config["models"][bucket]:
            return dict(config["models"][bucket][model_id])
    raise KeyError(f"Unknown model in resolved config: {model_id}")


def get_scenario_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in config["scenarios"]["items"]}


def _effective_common_space(config: dict[str, Any], model_spec: dict[str, Any]) -> dict[str, list[Any]]:
    space = deepcopy(config["hyperparameters"]["common"])
    for key, values in dict(model_spec.get("common_space_overrides") or {}).items():
        space[key] = list(values)
    return space


def _resource_cap_for_tier(config: dict[str, Any], tier: str) -> int | None:
    search_cfg = dict(config.get("search") or {})
    if tier == "heavy":
        return int(search_cfg.get("max_parallel_heavy", 1))
    if tier == "medium":
        return int(search_cfg.get("max_parallel_medium", search_cfg.get("parallel_workers", 1)))
    return None


def _inflight_tier_counts(config: dict[str, Any], inflight: list[dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for item in inflight:
        model_id = item.get("model_id")
        if not model_id:
            continue
        spec = get_model_spec(config, str(model_id))
        counts[str(spec.get("resource_tier") or "medium")] += 1
    return counts


def _prune_expired_pair_cooldowns(state: dict[str, Any]) -> dict[str, Any]:
    state = deepcopy(state)
    attempted = int(state.get("attempted_trials") or 0)
    raw = dict(state.get("pair_cooldowns") or {})
    state["pair_cooldowns"] = {
        key: int(value)
        for key, value in raw.items()
        if int(value) > attempted
    }
    return state


def _pair_key(model_id: str, scenario_name: str) -> str:
    return f"{model_id}::{scenario_name}"


def _get_phase_sequence(config: dict[str, Any]) -> list[tuple[str, int]]:
    budgets = config["search"]["phase_trial_budgets"]
    phase_sequence = [
        ("broad_search", int(budgets.get("broad_search", 8))),
        ("refinement", int(budgets.get("refinement", 4))),
        ("grid_search", int(budgets.get("grid_search", 4))),
    ]
    if bool(config.get("mutation", {}).get("enabled")) and int(budgets.get("mutation_search", 0)) > 0:
        phase_sequence.append(("mutation_search", int(budgets.get("mutation_search", 2))))
    return phase_sequence


def maybe_advance_phase(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    state = deepcopy(state)
    phase_sequence = _get_phase_sequence(config)
    budgets = dict(phase_sequence)
    phase = str(state.get("phase_name") or "broad_search")
    progress = int(state.get("phase_trial_index") or 0)
    budget = int(budgets.get(phase, 1))
    if progress < budget:
        return state

    phase_names = [name for name, _ in phase_sequence]
    idx = phase_names.index(phase)
    next_idx = (idx + 1) % len(phase_names)
    next_phase = phase_names[next_idx]
    state["phase_name"] = next_phase
    state["phase_trial_index"] = 0
    if next_idx == 0:
        state["cycle_index"] = int(state.get("cycle_index") or 0) + 1
    append_jsonl(
        resolve_agent_dir(state["run_id"]) / "decision_log.jsonl",
        {
            "event": "phase_advanced",
            "timestamp_kst": iso_now(),
            "phase_from": phase,
            "phase_to": next_phase,
            "cycle_index": state["cycle_index"],
        },
    )
    return state


def _nearby_values(values: list[Any], current: Any, *, max_items: int = 3) -> list[Any]:
    if not values:
        return []
    unique_values = list(dict.fromkeys(values))
    if current not in unique_values:
        return unique_values[:max_items]
    idx = unique_values.index(current)
    window = [unique_values[idx]]
    left = idx - 1
    right = idx + 1
    while len(window) < min(max_items, len(unique_values)):
        if left >= 0:
            window.append(unique_values[left])
            if len(window) >= max_items:
                break
        if right < len(unique_values):
            window.append(unique_values[right])
        left -= 1
        right += 1
    ordered = []
    for value in window:
        if value not in ordered:
            ordered.append(value)
    return ordered


def _coerce_sample_to_allowed(
    sampled: Any,
    *,
    allowed_values: list[Any],
    full_values: list[Any],
    trial_number: int,
) -> Any:
    if not allowed_values:
        return sampled
    if sampled in allowed_values:
        return sampled

    if isinstance(sampled, (int, float)) and all(isinstance(v, (int, float)) for v in allowed_values):
        return min(allowed_values, key=lambda value: abs(float(value) - float(sampled)))

    if sampled in full_values:
        sampled_idx = full_values.index(sampled)
        closest = sorted(
            allowed_values,
            key=lambda value: abs(full_values.index(value) - sampled_idx) if value in full_values else 10**9,
        )
        if closest:
            return closest[0]

    return allowed_values[trial_number % len(allowed_values)]


def _exclude_values_if_possible(choices: list[str], blocked: set[str]) -> list[str]:
    if not blocked or len(choices) <= 1:
        return choices
    filtered = [choice for choice in choices if choice not in blocked]
    return filtered or choices


def _top_context_rows(agent_dir: Path, top_k: int) -> list[dict[str, Any]]:
    rows = _load_leaderboard_rows(agent_dir)
    return rows[:top_k]


def _phase_model_and_scenario_choices(
    *,
    agent_dir: Path,
    state: dict[str, Any],
    config: dict[str, Any],
    phase: str,
) -> tuple[list[str], list[str]]:
    scenario_map = get_scenario_map(config)
    all_scenarios = list(scenario_map.keys())
    all_models = list(config["models"]["role_based"].keys()) + list(config["models"]["explicit_train_test"].keys())
    champion = state.get("best") or {}

    if phase == "broad_search" or not champion:
        return all_models, all_scenarios

    top_rows = _top_context_rows(agent_dir, config["search"].get("broad_top_k_context", 5))
    top_models = [champion["model_id"]]
    top_scenarios = [champion["scenario_name"]]
    for row in top_rows:
        if row["model_id"] not in top_models:
            top_models.append(row["model_id"])
        if row["scenario_name"] not in top_scenarios:
            top_scenarios.append(row["scenario_name"])

    if phase == "mutation_search":
        mutable_models = [
            model_id
            for model_id in top_models
            if get_model_spec(config, model_id).get("supports_code_mutation")
        ]
        if not mutable_models:
            mutable_models = list(config["models"]["explicit_train_test"].keys())
        return mutable_models, top_scenarios[: max(1, min(len(top_scenarios), 3))]

    return top_models[: max(1, min(len(top_models), 4))], top_scenarios[: max(1, min(len(top_scenarios), 4))]


def _coverage_constrained_choices(
    *,
    agent_dir: Path,
    model_choices: list[str],
    scenario_choices: list[str],
) -> tuple[list[str], list[str]]:
    rows = _load_leaderboard_rows(agent_dir)
    completed_models = {str(row.get("model_id")) for row in rows if row.get("model_id")}
    completed_scenarios = {str(row.get("scenario_name")) for row in rows if row.get("scenario_name")}
    missing_models = [model_id for model_id in model_choices if model_id not in completed_models]
    missing_scenarios = [scenario_name for scenario_name in scenario_choices if scenario_name not in completed_scenarios]
    return (missing_models or model_choices, missing_scenarios or scenario_choices)


def _model_priority(model_id: str) -> int:
    if model_id in {"frame_spatial_transformer", "hierarchical_tree_mlp"}:
        return 0
    if model_id == "lappe_dist_mixer":
        return 1
    return 2


def _choose_candidate_pair(
    *,
    agent_dir: Path,
    state: dict[str, Any],
    config: dict[str, Any],
    phase: str,
    model_choices: list[str],
    scenario_choices: list[str],
    blocked_pairs: set[tuple[str, str]],
    worker_slot: int | None,
) -> tuple[str, str]:
    inflight = load_inflight(agent_dir)
    inflight_model_counts = Counter(str(item.get("model_id")) for item in inflight if item.get("model_id"))
    inflight_scenario_counts = Counter(str(item.get("scenario_name")) for item in inflight if item.get("scenario_name"))
    inflight_tier_counts = _inflight_tier_counts(config, inflight)

    max_same_model = int(config["search"].get("max_consecutive_same_model", 0))
    max_same_scenario = int(config["search"].get("max_consecutive_same_scenario", 0))
    recent_model = str(state.get("recent_model_id") or "")
    recent_scenario = str(state.get("recent_scenario_name") or "")
    recent_model_streak = int(state.get("recent_model_streak") or 0)
    recent_scenario_streak = int(state.get("recent_scenario_streak") or 0)
    cooldowns = dict((state.get("pair_cooldowns") or {}))
    invalid_role_scenarios = dict(state.get("invalid_role_scenarios") or {})

    if phase == "broad_search":
        model_choices, scenario_choices = _coverage_constrained_choices(
            agent_dir=agent_dir,
            model_choices=model_choices,
            scenario_choices=scenario_choices,
        )

    allowed_pairs: list[tuple[str, str]] = []
    for model_id, scenario_name in itertools.product(model_choices, scenario_choices):
        pair = (model_id, scenario_name)
        if pair in blocked_pairs:
            continue
        if _pair_key(model_id, scenario_name) in cooldowns:
            continue
        spec = get_model_spec(config, model_id)
        if spec["runner_kind"] == "role_based" and scenario_name in invalid_role_scenarios:
            continue
        tier = str(spec.get("resource_tier") or "medium")
        cap = _resource_cap_for_tier(config, tier)
        if cap is not None and int(inflight_tier_counts.get(tier, 0)) >= cap:
            continue
        if max_same_model > 0 and model_id == recent_model:
            if recent_model_streak + int(inflight_model_counts.get(model_id, 0)) >= max_same_model:
                continue
        if max_same_scenario > 0 and scenario_name == recent_scenario:
            if recent_scenario_streak + int(inflight_scenario_counts.get(scenario_name, 0)) >= max_same_scenario:
                continue
        allowed_pairs.append(pair)

    if not allowed_pairs:
        raise RuntimeError("Could not reserve a non-conflicting candidate after 24 attempts.")

    rng = random.Random(
        f"{agent_dir.name}:{phase}:{int(state.get('attempted_trials') or 0)}:{worker_slot or 0}:{len(allowed_pairs)}"
    )
    if phase in {"refinement", "grid_search", "mutation_search"}:
        top_rows = _top_context_rows(agent_dir, 3)
        top_pairs = [
            (str(row["model_id"]), str(row["scenario_name"]))
            for row in top_rows
            if (str(row["model_id"]), str(row["scenario_name"])) in allowed_pairs
        ]
        if top_pairs and rng.random() < 0.7:
            return top_pairs[rng.randrange(len(top_pairs))]

    allowed_pairs.sort(key=lambda pair: (_model_priority(pair[0]), pair[0], pair[1]))
    return allowed_pairs[rng.randrange(len(allowed_pairs))]


def _sample_common_hparams(
    trial: optuna.Trial,
    *,
    space: dict[str, list[Any]],
    allowed_space: dict[str, list[Any]] | None,
    phase: str,
    champion_hparams: dict[str, Any] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, values in space.items():
        full_values = list(values)
        sampled = trial.suggest_categorical(key, full_values)
        allowed_values = list((allowed_space or {}).get(key) or full_values)
        if phase in {"refinement", "grid_search", "mutation_search"} and champion_hparams and key in champion_hparams:
            narrowed = _nearby_values(full_values, champion_hparams[key], max_items=3)
            if narrowed:
                allowed_values = [value for value in narrowed if value in allowed_values] or allowed_values
        result[key] = _coerce_sample_to_allowed(
            sampled,
            allowed_values=allowed_values,
            full_values=full_values,
            trial_number=trial.number,
        )
    if not result.get("use_label_smoothing", False):
        result["label_smoothing"] = 0.0
    return result


def _sample_model_overrides(
    *,
    model_spec: dict[str, Any],
    phase: str,
    champion_overrides: dict[str, Any] | None,
    trial_number: int,
) -> dict[str, Any] | None:
    override_space = dict(model_spec.get("override_space") or {})
    if not override_space:
        return None
    overrides: dict[str, Any] = {}
    rng = random.Random(f"{model_spec.get('model_id')}:{phase}:{trial_number}")
    for key, values in override_space.items():
        full_values = list(values)
        allowed_values = list(full_values)
        if phase in {"refinement", "grid_search", "mutation_search"} and champion_overrides and key in champion_overrides:
            narrowed = _nearby_values(full_values, champion_overrides[key], max_items=3)
            if narrowed:
                allowed_values = narrowed
        overrides[key] = allowed_values[rng.randrange(len(allowed_values))]
    return overrides


def _build_mutation_spec(
    *,
    model_spec: dict[str, Any],
    model_overrides: dict[str, Any] | None,
) -> dict[str, Any] | None:
    target_file = model_spec.get("mutation_target_file")
    if not target_file or not model_overrides:
        return None
    return {
        "kind": "dataset_model_kwargs_defaults",
        "target_file": target_file,
        "values": dict(model_overrides),
    }


def _objective_score(metrics: dict[str, Any], best_val_loss: float | None) -> float:
    accuracy = float(metrics.get("accuracy") or 0.0)
    macro_f1 = float(((metrics.get("macro_avg") or {}).get("f1")) or 0.0)
    class0_fpr = float(((metrics.get("class0_metrics") or {}).get("false_positive_rate")) or 1.0)
    val_term = 0.0 if best_val_loss is None or math.isinf(best_val_loss) else max(0.0, 1.0 - float(best_val_loss))
    return (macro_f1 * 1000.0) + (accuracy * 100.0) + (val_term * 10.0) - (class0_fpr * 10.0)


def _rank_tuple(metrics: dict[str, Any], best_val_loss: float | None) -> tuple[float, float, float, float]:
    accuracy = float(metrics.get("accuracy") or 0.0)
    macro_f1 = float(((metrics.get("macro_avg") or {}).get("f1")) or 0.0)
    class0_fpr = float(((metrics.get("class0_metrics") or {}).get("false_positive_rate")) or 1.0)
    val_bonus = float("-inf") if best_val_loss is None or math.isinf(best_val_loss) else -float(best_val_loss)
    return (macro_f1, accuracy, -class0_fpr, val_bonus)


def goal_reached(goal: dict[str, Any], metrics: dict[str, Any]) -> bool:
    accuracy = float(metrics.get("accuracy") or 0.0)
    macro_f1 = float(((metrics.get("macro_avg") or {}).get("f1")) or 0.0)
    return accuracy >= float(goal["test_accuracy"]) and macro_f1 >= float(goal["test_macro_f1"])


def _trial_log_path(agent_dir: Path, candidate_id: str) -> Path:
    return agent_dir / "logs" / f"{candidate_id}.log"


def _trial_runtime_path(agent_dir: Path, candidate_id: str) -> Path:
    return agent_dir / TRIAL_RUNTIME_DIRNAME / f"{candidate_id}.json"


def _optuna_storage_uri(agent_dir: Path) -> str:
    return f"sqlite:///{(agent_dir / 'optuna_study.db').resolve()}"


def _load_study(agent_dir: Path, config: dict[str, Any]) -> optuna.study.Study:
    search_cfg = dict(config.get("search") or {})
    sampler = optuna.samplers.TPESampler(
        seed=int(search_cfg.get("seed", 42)),
        n_startup_trials=int(search_cfg.get("tpe_n_startup_trials", 12)),
        multivariate=bool(search_cfg.get("tpe_multivariate", True)),
    )
    with advisory_lock(lock_path(agent_dir / "optuna_study.db")):
        return optuna.create_study(
            study_name=f"jamjambeat_agent_{agent_dir.name}",
            storage=_optuna_storage_uri(agent_dir),
            load_if_exists=True,
            direction="maximize",
            sampler=sampler,
        )


def _apply_consecutive_repeat_guard(
    *,
    choices: list[str],
    last_value: str | None,
    streak: int,
    max_streak: int,
) -> list[str]:
    if max_streak <= 0 or streak < max_streak or not last_value or len(choices) <= 1:
        return choices
    guarded = [value for value in choices if value != last_value]
    return guarded or choices


def propose_candidate(agent_dir: Path, *, worker_slot: int | None = None) -> dict[str, Any]:
    config = load_resolved_config(agent_dir)
    scenario_map = get_scenario_map(config)
    study = _load_study(agent_dir, config)
    seed = int(config["search"]["seed"])

    with metadata_lock(agent_dir):
        state = maybe_advance_phase(_prune_expired_pair_cooldowns(load_state(agent_dir)), config)
        inflight = load_inflight(agent_dir)
        blocked_pairs = {
            (str(item.get("model_id")), str(item.get("scenario_name")))
            for item in inflight
            if item.get("model_id") and item.get("scenario_name")
        }
        blocked_fingerprints = {str(item.get("fingerprint")) for item in inflight if item.get("fingerprint")}

        phase = str(state.get("phase_name") or "broad_search")
        champion = state.get("best") or {}
        champion_hparams = dict((champion.get("hyperparameters") or {}))
        champion_overrides = dict((champion.get("model_overrides") or {}))

        model_choices, scenario_choices = _phase_model_and_scenario_choices(
            agent_dir=agent_dir,
            state=state,
            config=config,
            phase=phase,
        )
        model_choices = _apply_consecutive_repeat_guard(
            choices=model_choices,
            last_value=state.get("recent_model_id"),
            streak=int(state.get("recent_model_streak") or 0),
            max_streak=int(config["search"].get("max_consecutive_same_model", 0)),
        )
        scenario_choices = _apply_consecutive_repeat_guard(
            choices=scenario_choices,
            last_value=state.get("recent_scenario_name"),
            streak=int(state.get("recent_scenario_streak") or 0),
            max_streak=int(config["search"].get("max_consecutive_same_scenario", 0)),
        )

        attempts = 0
        while True:
            attempts += 1
            if attempts > 24:
                raise RuntimeError("Could not reserve a non-conflicting candidate after 24 attempts.")

            trial = study.ask()
            model_id, scenario_name = _choose_candidate_pair(
                agent_dir=agent_dir,
                state=state,
                config=config,
                phase=phase,
                model_choices=model_choices,
                scenario_choices=scenario_choices,
                blocked_pairs=blocked_pairs,
                worker_slot=worker_slot,
            )
            model_spec = get_model_spec(config, model_id)
            scenario = scenario_map[scenario_name]

            if model_spec["runner_kind"] == "role_based":
                role_manifest = prepare_role_dataset_cache(agent_dir, scenario=scenario, seed=seed)
                viability = dict(role_manifest.get("viability") or {})
                if not bool(viability.get("valid", False)):
                    reason = "; ".join(list(viability.get("reasons") or ["role_scenario_invalid"]))
                    invalid_role_scenarios = dict(state.get("invalid_role_scenarios") or {})
                    if invalid_role_scenarios.get(scenario_name) != reason:
                        invalid_role_scenarios[scenario_name] = reason
                        state["invalid_role_scenarios"] = invalid_role_scenarios
                        save_state(agent_dir, state)
                        append_jsonl(
                            agent_dir / "decision_log.jsonl",
                            {
                                "event": "scenario_invalid",
                                "timestamp_kst": iso_now(),
                                "scenario_name": scenario_name,
                                "model_id": model_id,
                                "reason": reason,
                                "rows": dict(role_manifest.get("rows") or {}),
                                "cache_status": role_manifest.get("_cache_status"),
                                "worker_slot": worker_slot,
                            },
                        )
                        worker_log(
                            agent_dir,
                            (
                                f"scenario invalid: scenario={scenario_name} model={model_id} "
                                f"reason={reason}"
                            ),
                            slot=worker_slot,
                        )
                    study.tell(int(trial.number), state=TrialState.PRUNED)
                    blocked_pairs.add((model_id, scenario_name))
                    continue

            hparams = _sample_common_hparams(
                trial,
                space=config["hyperparameters"]["common"],
                allowed_space=_effective_common_space(config, model_spec),
                phase=phase,
                champion_hparams=champion_hparams,
            )

            model_overrides = None
            mutation = None
            if bool(model_spec.get("supports_model_overrides")):
                model_overrides = _sample_model_overrides(
                    model_spec=model_spec,
                    phase=phase,
                    trial_number=int(trial.number),
                    champion_overrides=champion_overrides if champion.get("model_id") == model_id else None,
                )

            if phase == "mutation_search" and config["mutation"]["enabled"] and bool(model_spec.get("supports_code_mutation")):
                mutation = _build_mutation_spec(model_spec=model_spec, model_overrides=model_overrides)
                if mutation:
                    model_overrides = None

            candidate = {
                "candidate_id": f"trial_{trial.number:05d}",
                "trial_number": trial.number,
                "phase": phase,
                "runner_kind": model_spec["runner_kind"],
                "model_id": model_id,
                "scenario_name": scenario_name,
                "scenario": scenario,
                "hyperparameters": hparams,
                "model_overrides": model_overrides,
                "mutation": mutation,
                "created_at_kst": iso_now(),
                "worker_slot": worker_slot,
            }
            fingerprint = candidate_fingerprint(candidate)
            pair = (candidate["model_id"], candidate["scenario_name"])
            if pair in blocked_pairs or fingerprint in blocked_fingerprints:
                study.tell(int(trial.number), state=TrialState.PRUNED)
                continue

            inflight.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "trial_number": candidate["trial_number"],
                    "worker_slot": worker_slot,
                    "model_id": candidate["model_id"],
                    "scenario_name": candidate["scenario_name"],
                    "fingerprint": fingerprint,
                    "created_at_kst": iso_now(),
                }
            )
            save_inflight(agent_dir, inflight)
            save_state(agent_dir, state)
            break

    append_jsonl(
        agent_dir / "decision_log.jsonl",
        {
            "event": "candidate_proposed",
            "timestamp_kst": iso_now(),
            "candidate_id": candidate["candidate_id"],
            "phase": phase,
            "model_id": model_id,
            "scenario_name": scenario_name,
            "runner_kind": model_spec["runner_kind"],
            "mutation": bool(mutation),
            "worker_slot": worker_slot,
        },
    )
    worker_log(
        agent_dir,
        (
            f"candidate proposed: id={candidate['candidate_id']} phase={phase} "
            f"model={model_id} scenario={scenario_name} runner={model_spec['runner_kind']} "
            f"loss={hparams['loss_type']} optimizer={hparams['optimizer']} "
            f"weighted_sampler={hparams['use_weighted_sampler']} alpha={hparams['use_alpha']} "
            f"label_smoothing={'on' if hparams['use_label_smoothing'] else 'off'}:{hparams['label_smoothing']}"
        ),
        slot=worker_slot,
    )
    return candidate


def _python_literal(value: Any) -> str:
    return repr(value)


def apply_code_mutation(agent_dir: Path, mutation: dict[str, Any]) -> dict[str, Any]:
    target = PROJECT_ROOT / mutation["target_file"]
    text = target.read_text(encoding="utf-8")
    applied: dict[str, dict[str, str]] = {}
    for key, value in mutation["values"].items():
        needle_prefix = f'"{key}":'
        replaced = False
        new_lines: list[str] = []
        for line in text.splitlines():
            if needle_prefix in line and line.strip().startswith(f'"{key}"'):
                old_line = line
                prefix, _, suffix = line.partition(":")
                trailing = "," if line.rstrip().endswith(",") else ""
                indent = line[: len(line) - len(line.lstrip())]
                line = f"{indent}{prefix.strip()}: {_python_literal(value)}{trailing}"
                applied[key] = {"before": old_line.strip(), "after": line.strip()}
                replaced = True
            new_lines.append(line)
        if not replaced:
            raise ValueError(f"Could not locate mutation key {key!r} in {target}")
        text = "\n".join(new_lines) + "\n"
    target.write_text(text, encoding="utf-8")
    return {"target_file": str(target), "applied": applied}


class InvalidRoleScenarioError(ValueError):
    """Raised when a role-based scenario cannot build a viable cache."""


def _build_candidate_command(agent_dir: Path, candidate: dict[str, Any]) -> tuple[list[str], Path, dict[str, Any]]:
    scenario = candidate["scenario"]
    h = candidate["hyperparameters"]
    output_root = PIPELINE_MANAGED_ROOT / agent_dir.name / candidate["candidate_id"]
    output_root.mkdir(parents=True, exist_ok=True)
    runtime_meta: dict[str, Any] = {
        "train_csv": str(scenario["train_csv"]),
        "test_csv": str(scenario["test_csv"]),
        "role_cache_hit": False,
        "role_cache_miss": False,
        "role_cache_manifest_rows": {},
        "input_csv_rows": {},
        "canonical_csv_rows": {},
    }

    cmd = [sys.executable]
    if candidate["runner_kind"] == "explicit_train_test":
        train_df, train_stats = load_explicit_frame_csv_with_stats(Path(scenario["train_csv"]))
        test_df, test_stats = load_explicit_frame_csv_with_stats(Path(scenario["test_csv"]))
        runtime_meta.update(
            {
                "input_csv_rows": {"train": train_stats["raw_rows"], "test": test_stats["raw_rows"]},
                "canonical_csv_rows": {
                    "train": train_stats["rows_after_dropna"],
                    "test": test_stats["rows_after_dropna"],
                },
                "input_stats": {"train": train_stats, "test": test_stats},
            }
        )
        del train_df, test_df
        cmd += [
            str(MODEL_PIPELINES_DIR / "new_run_pipeline.py"),
            "--train-csv",
            str(Path(scenario["train_csv"]).resolve()),
            "--test-csv",
            str(Path(scenario["test_csv"]).resolve()),
            "--models",
            candidate["model_id"],
            "--output-root",
            str(output_root),
            "--epochs",
            str(h["epochs"]),
            "--batch-size",
            str(h["batch_size"]),
            "--lr",
            str(h["lr"]),
            "--weight-decay",
            str(h["weight_decay"]),
            "--patience",
            str(h["patience"]),
            "--loss-type",
            str(h["loss_type"]),
            "--focal-gamma",
            str(h["focal_gamma"]),
            "--device",
            "cuda",
            "--seed",
            str(int(load_resolved_config(agent_dir)["search"]["seed"])),
            "--num-workers",
            "0",
            "--tau",
            str(h["tau"]),
            "--vote-n",
            str(h["vote_n"]),
            "--debounce-k",
            str(h["debounce_k"]),
            "--fallback-fps",
            str(h["fallback_fps"]),
            "--optimizer",
            str(h["optimizer"]),
            "--momentum",
            str(h["momentum"]),
            "--label-smoothing",
            str(h["label_smoothing"]),
        ]
        cmd.append("--use-weighted-sampler" if h["use_weighted_sampler"] else "--no-use-weighted-sampler")
        cmd.append("--use-alpha" if h["use_alpha"] else "--no-use-alpha")
        cmd.append("--use-label-smoothing" if h["use_label_smoothing"] else "--no-use-label-smoothing")
        if candidate.get("model_overrides"):
            cmd += ["--model-overrides", json.dumps(candidate["model_overrides"], ensure_ascii=False)]
        return cmd, output_root, runtime_meta

    role_manifest = prepare_role_dataset_cache(
        agent_dir,
        scenario=scenario,
        seed=int(load_resolved_config(agent_dir)["search"]["seed"]),
    )
    viability = dict(role_manifest.get("viability") or {})
    if not bool(viability.get("valid", False)):
        reason = "; ".join(list(viability.get("reasons") or ["role_scenario_invalid"]))
        raise InvalidRoleScenarioError(reason)
    paths = role_manifest["paths"]
    runtime_meta.update(
        {
            "role_cache_hit": role_manifest.get("_cache_status") == "hit",
            "role_cache_miss": role_manifest.get("_cache_status") == "miss",
            "role_cache_manifest_rows": dict(role_manifest.get("rows") or {}),
            "input_csv_rows": {
                "train": int(((role_manifest.get("input_stats") or {}).get("train") or {}).get("raw_rows") or 0),
                "test": int(((role_manifest.get("input_stats") or {}).get("test") or {}).get("raw_rows") or 0),
            },
            "canonical_csv_rows": {
                "train": int(((role_manifest.get("input_stats") or {}).get("train") or {}).get("rows_after_dropna") or 0),
                "test": int(((role_manifest.get("input_stats") or {}).get("test") or {}).get("rows_after_dropna") or 0),
            },
            "input_stats": dict(role_manifest.get("input_stats") or {}),
            "role_manifest_path": str(Path(role_manifest["paths"]["train"]).parent / "manifest.json"),
        }
    )
    cmd += [
        str(MODEL_PIPELINES_DIR / "run_pipeline.py"),
        "--model-id",
        candidate["model_id"],
        "--train-csv",
        paths["train"],
        "--val-csv",
        paths["val"],
        "--test-csv",
        paths["test"],
        "--inference-csv",
        paths["inference"],
        "--output-root",
        str(output_root),
        "--epochs",
        str(h["epochs"]),
        "--batch-size",
        str(h["batch_size"]),
        "--lr",
        str(h["lr"]),
        "--weight-decay",
        str(h["weight_decay"]),
        "--patience",
        str(h["patience"]),
        "--loss-type",
        str(h["loss_type"]),
        "--focal-gamma",
        str(h["focal_gamma"]),
        "--device",
        "cuda",
        "--seed",
        str(int(load_resolved_config(agent_dir)["search"]["seed"])),
        "--num-workers",
        "0",
        "--tau",
        str(h["tau"]),
        "--vote-n",
        str(h["vote_n"]),
        "--debounce-k",
        str(h["debounce_k"]),
        "--fallback-fps",
        str(h["fallback_fps"]),
        "--optimizer",
        str(h["optimizer"]),
        "--momentum",
        str(h["momentum"]),
        "--label-smoothing",
        str(h["label_smoothing"]),
    ]
    cmd.append("--use-weighted-sampler" if h["use_weighted_sampler"] else "--no-use-weighted-sampler")
    cmd.append("--use-alpha" if h["use_alpha"] else "--no-use-alpha")
    cmd.append("--use-label-smoothing" if h["use_label_smoothing"] else "--no-use-label-smoothing")
    if candidate.get("model_overrides"):
        cmd += ["--model-overrides", json.dumps(candidate["model_overrides"], ensure_ascii=False)]
    return cmd, output_root, runtime_meta


def _read_run_summary_from_output(candidate: dict[str, Any], output_root: Path) -> tuple[dict[str, Any], Path]:
    if candidate["runner_kind"] == "explicit_train_test":
        latest_suite = read_json(output_root / "latest_suite.json", default={})
        suite_dir = Path(latest_suite["latest_suite"])
        latest_model = read_json(suite_dir / candidate["model_id"] / "latest.json", default={})
        run_dir = Path(latest_model["latest_run"])
    else:
        latest_model = read_json(output_root / candidate["model_id"] / "latest.json", default={})
        run_dir = Path(latest_model["latest_run"])
    summary_path = run_dir / "run_summary.json"
    summary = read_json(summary_path, default=None)
    if summary is None:
        raise ValueError(f"Missing run_summary.json for candidate {candidate['candidate_id']}")
    return summary, run_dir


def _configure_mlflow(agent_dir: Path) -> None:
    tracking_dir = (agent_dir / "mlruns").resolve()
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment(f"jamjambeat_agent_{agent_dir.name}")


def _tail_text_lines(text: str, limit: int = 100) -> list[str]:
    return text.splitlines()[-limit:]


def _count_artifact_inventory(root: Path | None) -> dict[str, int]:
    if root is None or not root.exists():
        return {"file_count": 0, "total_bytes": 0}
    file_count = 0
    total_bytes = 0
    for path in root.rglob("*"):
        if path.is_file():
            file_count += 1
            total_bytes += int(path.stat().st_size)
    return {"file_count": file_count, "total_bytes": total_bytes}


def _csv_row_count(path: Path | None) -> int | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        line_count = sum(1 for _ in handle)
    return max(line_count - 1, 0)


def _collect_disk_diagnostics(config: dict[str, Any]) -> dict[str, Any]:
    diagnostics_cfg = dict(config.get("diagnostics") or {})
    payload: dict[str, Any] = {}
    if bool(diagnostics_cfg.get("collect_proc_diskstats", False)):
        diskstats_path = Path("/proc/diskstats")
        if diskstats_path.exists():
            lines = diskstats_path.read_text(encoding="utf-8").splitlines()
            payload["proc_diskstats"] = lines[:128]
    if bool(diagnostics_cfg.get("collect_iostat", False)):
        try:
            completed = subprocess.run(
                ["iostat", "-dx"],
                capture_output=True,
                text=True,
                check=False,
            )
            payload["iostat"] = (completed.stdout or completed.stderr or "").splitlines()[-128:]
        except FileNotFoundError:
            payload["iostat"] = ["iostat not available"]
    return payload


def _write_trial_runtime(
    agent_dir: Path,
    candidate_id: str,
    payload: dict[str, Any],
) -> Path:
    path = _trial_runtime_path(agent_dir, candidate_id)
    write_json(path, payload)
    return path


def _nvidia_smi_snapshot() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return "nvidia-smi not available"
    payload = (completed.stdout or completed.stderr or "").strip()
    return payload or f"nvidia-smi exited with code {completed.returncode}"


def _run_command_streaming(cmd: list[str], *, cwd: Path, log_path: Path) -> tuple[int, list[str]]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    tail: deque[str] = deque(maxlen=100)
    with log_path.open("a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            tail.append(line.rstrip("\n"))
        returncode = proc.wait()
    return returncode, list(tail)


def _next_lower_choice(value: int, choices: list[int]) -> int:
    ordered = sorted({int(item) for item in choices})
    lowers = [item for item in ordered if item < int(value)]
    return lowers[-1] if lowers else ordered[0]


def _build_retry_hparams(
    *,
    config: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    model_spec = get_model_spec(config, candidate["model_id"])
    effective_space = _effective_common_space(config, model_spec)
    retry_hparams = deepcopy(candidate["hyperparameters"])
    batch_choices = [int(item) for item in effective_space.get("batch_size", [retry_hparams["batch_size"]])]
    current_batch = int(retry_hparams["batch_size"])
    retry_hparams["batch_size"] = _next_lower_choice(current_batch, batch_choices)

    if str(model_spec.get("resource_tier") or "medium") == "heavy":
        epoch_choices = [int(item) for item in config["hyperparameters"]["common"].get("epochs", [retry_hparams["epochs"]])]
        retry_hparams["epochs"] = _next_lower_choice(int(retry_hparams["epochs"]), epoch_choices)
    return retry_hparams


def execute_candidate(agent_dir: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "candidate_id": candidate["candidate_id"],
        "trial_number": candidate["trial_number"],
        "phase": candidate["phase"],
        "status": "failed",
        "timestamp_kst": iso_now(),
        "mutation_applied": False,
        "mutation_details": None,
        "retry_attempted": False,
        "retry_succeeded": False,
        "cooldown_requested": False,
        "cooldown_trials": 0,
        "effective_hyperparameters": deepcopy(candidate["hyperparameters"]),
        "returncode": None,
        "log_tail": [],
        "nvidia_smi_snapshot": None,
    }

    _configure_mlflow(agent_dir)
    log_path = _trial_log_path(agent_dir, candidate["candidate_id"])
    mutation = candidate.get("mutation")
    worker_slot = candidate.get("worker_slot")
    config = load_resolved_config(agent_dir)
    model_spec = get_model_spec(config, candidate["model_id"])
    started_at = time.perf_counter()
    runtime_meta: dict[str, Any] = {}
    output_root: Path | None = None
    run_dir: Path | None = None
    disk_pre = _collect_disk_diagnostics(config)

    try:
        if mutation:
            mutation_details = apply_code_mutation(agent_dir, mutation)
            result["mutation_applied"] = True
            result["mutation_details"] = mutation_details

        effective_candidate = deepcopy(candidate)
        cmd, output_root, runtime_meta = _build_candidate_command(agent_dir, effective_candidate)
        atomic_write_text(log_path, "")
        worker_log(
            agent_dir,
            (
                f"trial start: id={candidate['candidate_id']} model={candidate['model_id']} "
                f"scenario={candidate['scenario_name']} phase={candidate['phase']} "
                f"log={log_path.name}"
            ),
            slot=worker_slot,
        )
        returncode, log_tail = _run_command_streaming(cmd, cwd=MODEL_ROOT, log_path=log_path)
        result["returncode"] = int(returncode)
        result["log_tail"] = list(log_tail)

        if returncode == -9 and bool(config["search"].get("sigkill_retry_once", True)):
            retry_hparams = _build_retry_hparams(config=config, candidate=effective_candidate)
            if retry_hparams != effective_candidate["hyperparameters"]:
                result["retry_attempted"] = True
                effective_candidate["hyperparameters"] = retry_hparams
                result["effective_hyperparameters"] = deepcopy(retry_hparams)
                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(
                        "\n[experiment_agent] retrying after runner_exit_code_-9 "
                        f"with batch_size={retry_hparams['batch_size']} epochs={retry_hparams['epochs']}\n"
                    )
                worker_log(
                    agent_dir,
                    (
                        f"retry after SIGKILL: id={candidate['candidate_id']} model={candidate['model_id']} "
                        f"scenario={candidate['scenario_name']} batch_size={retry_hparams['batch_size']} "
                        f"epochs={retry_hparams['epochs']}"
                    ),
                    slot=worker_slot,
                )
                cmd, output_root, runtime_meta = _build_candidate_command(agent_dir, effective_candidate)
                returncode, log_tail = _run_command_streaming(cmd, cwd=MODEL_ROOT, log_path=log_path)
                result["returncode"] = int(returncode)
                result["log_tail"] = list(log_tail)
                result["retry_succeeded"] = returncode == 0

        if result["returncode"] != 0:
            if int(result["returncode"]) == -9:
                result["nvidia_smi_snapshot"] = _nvidia_smi_snapshot()
                retry_floor_reached = _build_retry_hparams(
                    config=config,
                    candidate={**effective_candidate, "hyperparameters": deepcopy(effective_candidate["hyperparameters"])},
                ) == effective_candidate["hyperparameters"]
                if result["retry_attempted"] or retry_floor_reached:
                    result["cooldown_requested"] = True
                    result["cooldown_trials"] = int(config["search"].get("sigkill_cooldown_trials", 12))
            result.update(
                {
                    "status": "failed",
                    "failure_reason": f"runner_exit_code_{result['returncode']}",
                    "log_path": str(log_path),
                    "output_root": str(output_root),
                }
            )
            worker_log(
                agent_dir,
                (
                    f"trial failed: id={candidate['candidate_id']} model={candidate['model_id']} "
                    f"scenario={candidate['scenario_name']} reason={result['failure_reason']} "
                    f"log={log_path.name}"
                ),
                slot=worker_slot,
            )
            return result

        result["effective_hyperparameters"] = deepcopy(effective_candidate["hyperparameters"])
        summary, run_dir = _read_run_summary_from_output(candidate, output_root)
        metrics = dict(summary.get("metrics") or {})
        best_val_loss = summary.get("best_val_loss")
        score = _objective_score(metrics, best_val_loss)

        with mlflow.start_run(run_name=candidate["candidate_id"]):
            mlflow.set_tags(
                {
                    "agent_run_id": agent_dir.name,
                    "candidate_id": candidate["candidate_id"],
                    "phase": candidate["phase"],
                    "model_id": candidate["model_id"],
                    "scenario_name": candidate["scenario_name"],
                    "runner_kind": candidate["runner_kind"],
                    "mutation": str(bool(mutation)).lower(),
                    "output_dir": str(run_dir),
                }
            )
            mlflow.log_params(
                {
                    "model_id": candidate["model_id"],
                    "scenario_name": candidate["scenario_name"],
                    "runner_kind": candidate["runner_kind"],
                    **{f"hp_{k}": v for k, v in effective_candidate["hyperparameters"].items()},
                    **{f"override_{k}": v for k, v in (candidate.get("model_overrides") or {}).items()},
                }
            )
            mlflow.log_metrics(
                {
                    "accuracy": float(metrics.get("accuracy") or 0.0),
                    "macro_f1": float(((metrics.get("macro_avg") or {}).get("f1")) or 0.0),
                    "macro_precision": float(((metrics.get("macro_avg") or {}).get("precision")) or 0.0),
                    "macro_recall": float(((metrics.get("macro_avg") or {}).get("recall")) or 0.0),
                    "class0_fpr": float(((metrics.get("class0_metrics") or {}).get("false_positive_rate")) or 1.0),
                    "best_val_loss": float(best_val_loss) if best_val_loss is not None else float("nan"),
                    "score": float(score),
                }
            )

        result.update(
            {
                "status": "completed",
                "summary": summary,
                "metrics": metrics,
                "best_val_loss": best_val_loss,
                "rank_tuple": list(_rank_tuple(metrics, best_val_loss)),
                "score": score,
                "output_dir": str(run_dir),
                "output_root": str(output_root),
                "log_path": str(log_path),
            }
        )
        worker_log(
            agent_dir,
            (
                f"trial completed: id={candidate['candidate_id']} model={candidate['model_id']} "
                f"scenario={candidate['scenario_name']} acc={float(metrics.get('accuracy') or 0.0):.4f} "
                f"macro_f1={float(((metrics.get('macro_avg') or {}).get('f1')) or 0.0):.4f} "
                f"class0_fpr={float(((metrics.get('class0_metrics') or {}).get('false_positive_rate')) or 1.0):.4f} "
                f"best_val_loss={best_val_loss} output_dir={run_dir} "
                f"retry={'yes' if result['retry_attempted'] else 'no'} tier={model_spec.get('resource_tier')}"
            ),
            slot=worker_slot,
        )
        return result
    except InvalidRoleScenarioError as exc:
        result.update(
            {
                "status": "invalid",
                "failure_reason": str(exc),
                "log_path": str(log_path),
                "output_root": str(output_root) if output_root else None,
            }
        )
        worker_log(
            agent_dir,
            (
                f"scenario invalid during execute: id={candidate['candidate_id']} "
                f"model={candidate['model_id']} scenario={candidate['scenario_name']} reason={exc}"
            ),
            slot=worker_slot,
        )
        return result
    except Exception as exc:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n[experiment_agent] exception: {exc}\n")
        result.update(
            {
                "status": "failed",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "log_path": str(log_path),
            }
        )
        worker_log(
            agent_dir,
            (
                f"trial exception: id={candidate['candidate_id']} model={candidate['model_id']} "
                f"scenario={candidate['scenario_name']} reason={result['failure_reason']} "
                f"log={log_path.name}"
            ),
            slot=worker_slot,
        )
        return result
    finally:
        finished_at = iso_now()
        elapsed = time.perf_counter() - started_at
        disk_post = _collect_disk_diagnostics(config)
        artifact_root = run_dir if run_dir is not None else output_root
        preds_test_path = (run_dir / "preds_test.csv") if run_dir is not None else None
        preds_inference_path = (run_dir / "preds_inference.csv") if run_dir is not None else None
        trial_runtime = {
            "candidate_id": candidate["candidate_id"],
            "trial_number": candidate["trial_number"],
            "phase": candidate["phase"],
            "runner_kind": candidate["runner_kind"],
            "model_id": candidate["model_id"],
            "scenario_name": candidate["scenario_name"],
            "status": result.get("status"),
            "returncode": result.get("returncode"),
            "failure_reason": result.get("failure_reason"),
            "started_at_kst": result["timestamp_kst"],
            "finished_at_kst": finished_at,
            "wall_time_seconds": round(elapsed, 4),
            "role_cache_hit": bool(runtime_meta.get("role_cache_hit", False)),
            "role_cache_miss": bool(runtime_meta.get("role_cache_miss", False)),
            "role_cache_manifest_rows": dict(runtime_meta.get("role_cache_manifest_rows") or {}),
            "train_csv_rows": (runtime_meta.get("input_csv_rows") or {}).get("train"),
            "test_csv_rows": (runtime_meta.get("input_csv_rows") or {}).get("test"),
            "canonical_train_rows": (runtime_meta.get("canonical_csv_rows") or {}).get("train"),
            "canonical_test_rows": (runtime_meta.get("canonical_csv_rows") or {}).get("test"),
            "input_stats": dict(runtime_meta.get("input_stats") or {}),
            "preds_test_rows": _csv_row_count(preds_test_path),
            "preds_inference_rows": _csv_row_count(preds_inference_path),
            "artifact_inventory": _count_artifact_inventory(artifact_root),
            "output_root": str(output_root) if output_root else None,
            "output_dir": str(run_dir) if run_dir else result.get("output_dir"),
            "log_path": str(log_path),
            "log_tail": list(result.get("log_tail") or []),
            "retry_attempted": bool(result.get("retry_attempted")),
            "retry_succeeded": bool(result.get("retry_succeeded")),
            "cooldown_requested": bool(result.get("cooldown_requested")),
            "cooldown_trials": int(result.get("cooldown_trials") or 0),
            "nvidia_smi_snapshot": result.get("nvidia_smi_snapshot"),
            "diagnostics": {
                "pre": disk_pre,
                "post": disk_post,
            },
        }
        runtime_path = _write_trial_runtime(agent_dir, candidate["candidate_id"], trial_runtime)
        result["trial_runtime_path"] = str(runtime_path)


def _is_better_result(result: dict[str, Any], current_best: dict[str, Any] | None) -> bool:
    if not current_best:
        return True
    current_metrics = current_best.get("metrics") or {}
    current_val = current_best.get("best_val_loss")
    return _rank_tuple(result["metrics"], result.get("best_val_loss")) > _rank_tuple(current_metrics, current_val)


def _append_leaderboard_row(agent_dir: Path, candidate: dict[str, Any], result: dict[str, Any]) -> None:
    rows = _load_leaderboard_rows(agent_dir)
    rows.append(
        {
            "candidate_id": candidate["candidate_id"],
            "trial_number": candidate["trial_number"],
            "phase": candidate["phase"],
            "model_id": candidate["model_id"],
            "scenario_name": candidate["scenario_name"],
            "runner_kind": candidate["runner_kind"],
            "accuracy": float(result["metrics"].get("accuracy") or 0.0),
            "macro_f1": float(((result["metrics"].get("macro_avg") or {}).get("f1")) or 0.0),
            "class0_fpr": float(((result["metrics"].get("class0_metrics") or {}).get("false_positive_rate")) or 1.0),
            "best_val_loss": result.get("best_val_loss"),
            "score": float(result["score"]),
            "output_dir": result["output_dir"],
            "loss_type": candidate["hyperparameters"]["loss_type"],
            "optimizer": candidate["hyperparameters"]["optimizer"],
        }
    )
    _write_leaderboard_rows(agent_dir, rows)


def _append_champion_history(agent_dir: Path, champion_payload: dict[str, Any]) -> None:
    data = read_json(agent_dir / "champions.json", default={"history": []})
    data.setdefault("history", []).append(champion_payload)
    write_json(agent_dir / "champions.json", data)


def _schedule_backoff(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    state = deepcopy(state)
    schedule = list(config["backoff"]["schedule_seconds"])
    idx = min(int(state.get("backoff_index") or 0), len(schedule) - 1)
    seconds = min(int(schedule[idx]), int(config["backoff"]["max_seconds"]))
    state["current_backoff_seconds"] = int(seconds)
    state["next_retry_at_kst"] = (now_kst() + timedelta(seconds=seconds)).isoformat(timespec="seconds")
    state["backoff_index"] = min(idx + 1, len(schedule) - 1)
    state["status"] = "backing_off"
    return state


def _reset_backoff(state: dict[str, Any]) -> dict[str, Any]:
    state = deepcopy(state)
    state["current_backoff_seconds"] = 0
    state["next_retry_at_kst"] = None
    state["backoff_index"] = 0
    return state


def _write_study_summary(agent_dir: Path) -> None:
    config = load_resolved_config(agent_dir)
    study = _load_study(agent_dir, config)
    trials = study.get_trials(deepcopy=False)
    counter = Counter(str(t.state.name) for t in trials)
    best_trial = study.best_trial if trials and any(t.state == TrialState.COMPLETE for t in trials) else None
    summary = {
        "updated_at_kst": iso_now(),
        "trial_count": len(trials),
        "state_counts": dict(counter),
        "best_trial_number": best_trial.number if best_trial else None,
        "best_value": best_trial.value if best_trial else None,
        "best_params": dict(best_trial.params) if best_trial else {},
    }
    write_json(agent_dir / "study_summary.json", summary)


def _render_report(agent_dir: Path, detail: str) -> str:
    state = load_state(agent_dir)
    rows = _load_leaderboard_rows(agent_dir)
    recent = rows[:10]
    best = state.get("best") or {}
    config = load_resolved_config(agent_dir)
    worker_pids = _normalize_worker_pids(state)
    inflight = load_inflight(agent_dir)
    decision_metrics = _decision_metrics(agent_dir)
    invalid_role_scenarios = dict(state.get("invalid_role_scenarios") or {})
    invalid_reason_counts = dict(Counter(invalid_role_scenarios.values()))
    attempted = int(state.get("attempted_trials") or 0)
    completed = int(state.get("completed_trials") or 0)
    lines = [
        f"# Agent Report: {agent_dir.name}",
        f"- status: {state.get('status')}",
        f"- phase: {state.get('phase_name')}",
        f"- attempted_trials: {attempted}",
        f"- completed_trials: {completed}",
        f"- completed/attempted: {(completed / attempted):.3f}" if attempted else "- completed/attempted: n/a",
        f"- parallel_workers: {int(config.get('search', {}).get('parallel_workers', 1))}",
        f"- worker_pids: {worker_pids}",
        f"- inflight_count: {len(inflight)}",
        f"- current_backoff_seconds: {state.get('current_backoff_seconds')}",
        f"- next_retry_at_kst: {state.get('next_retry_at_kst')}",
        f"- last_success_at_kst: {state.get('last_success_at_kst')}",
        f"- last_failure_at_kst: {state.get('last_failure_at_kst')}",
        f"- runner_exit_code_-9_count: {decision_metrics['runner_exit_code_-9_count']}",
        f"- recent_20_success_ratio: {decision_metrics['recent_20_success_ratio']}",
        f"- invalid_scenarios: {len(invalid_role_scenarios)}",
        (
            "- top_fail_models: "
            + ", ".join(
                f"{model_id}:{count}"
                for model_id, count in sorted(
                    decision_metrics["model_fail_counts"].items(),
                    key=lambda item: (-item[1], item[0]),
                )[:3]
            )
            if decision_metrics["model_fail_counts"]
            else "- top_fail_models: none"
        ),
        (
            "- invalid_reasons: "
            + ", ".join(
                f"{reason}:{count}"
                for reason, count in sorted(invalid_reason_counts.items(), key=lambda item: (-item[1], item[0]))[:3]
            )
            if invalid_reason_counts
            else "- invalid_reasons: none"
        ),
    ]
    if best:
        best_metrics = best.get("metrics") or {}
        lines += [
            "",
            "## Champion",
            f"- model_id: {best.get('model_id')}",
            f"- scenario_name: {best.get('scenario_name')}",
            f"- accuracy: {best_metrics.get('accuracy')}",
            f"- macro_f1: {(best_metrics.get('macro_avg') or {}).get('f1')}",
            f"- class0_fpr: {(best_metrics.get('class0_metrics') or {}).get('false_positive_rate')}",
            f"- output_dir: {best.get('output_dir')}",
        ]
    if recent:
        lines += ["", "## Leaderboard Top 10", ""]
        for row in recent:
            lines.append(
                f"- {row['candidate_id']}: model={row['model_id']} scenario={row['scenario_name']} "
                f"macro_f1={row['macro_f1']:.4f} acc={row['accuracy']:.4f} class0_fpr={row['class0_fpr']:.4f}"
            )
    if detail == "full":
        lines += [
            "",
            "## Notes",
            "- This agent repeatedly optimizes on test metrics, so strict hold-out validity is weakened.",
            "- Val remains an auxiliary signal via best_val_loss and early stopping, not the final stop criterion.",
            "",
            "## Failure Summary",
        ]
        model_fail_counts = decision_metrics["model_fail_counts"]
        if model_fail_counts:
            for model_id, count in sorted(model_fail_counts.items(), key=lambda item: (-item[1], item[0])):
                lines.append(f"- model_fail_count[{model_id}]: {count}")
        else:
            lines.append("- model_fail_count: none")
        failure_reason_counts = decision_metrics["failure_reason_counts"]
        if failure_reason_counts:
            for reason, count in sorted(failure_reason_counts.items(), key=lambda item: (-item[1], item[0])):
                lines.append(f"- failure_reason[{reason}]: {count}")
        else:
            lines.append("- failure_reason: none")
        if invalid_role_scenarios:
            lines += ["", "## Invalid Role Scenarios"]
            for scenario_name, reason in sorted(invalid_role_scenarios.items()):
                lines.append(f"- {scenario_name}: {reason}")
        decision_log = agent_dir / "decision_log.jsonl"
        if decision_log.exists():
            lines += ["", "## Recent Decisions", ""]
            entries = decision_log.read_text(encoding="utf-8").splitlines()[-10:]
            for entry in entries:
                lines.append(f"- {entry}")
    return "\n".join(lines) + "\n"


def write_periodic_report(agent_dir: Path, *, reason: str) -> Path:
    state = load_state(agent_dir)
    filename = f"{int(state.get('completed_trials') or 0):05d}_{slugify(reason)}.md"
    path = agent_dir / "periodic_reports" / filename
    atomic_write_text(path, _render_report(agent_dir, detail="summary"))
    return path


def update_after_trial(agent_dir: Path, candidate: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    config = load_resolved_config(agent_dir)
    study = _load_study(agent_dir, config)
    worker_slot = candidate.get("worker_slot")
    effective_hyperparameters = deepcopy(result.get("effective_hyperparameters") or candidate["hyperparameters"])

    if result["status"] == "completed":
        study.tell(int(candidate["trial_number"]), values=float(result["score"]), state=TrialState.COMPLETE)
        with metadata_lock(agent_dir):
            state = _prune_expired_pair_cooldowns(load_state(agent_dir))
            inflight = [item for item in load_inflight(agent_dir) if item.get("candidate_id") != candidate["candidate_id"]]
            save_inflight(agent_dir, inflight)

            state["attempted_trials"] = int(state.get("attempted_trials") or 0) + 1
            state["phase_trial_index"] = int(state.get("phase_trial_index") or 0) + 1
            state["updated_at_kst"] = iso_now()
            state["last_candidate"] = candidate
            previous_model_id = state.get("recent_model_id")
            previous_scenario_name = state.get("recent_scenario_name")
            state["recent_model_id"] = candidate["model_id"]
            state["recent_scenario_name"] = candidate["scenario_name"]
            state["recent_model_streak"] = (
                int(state.get("recent_model_streak") or 0) + 1
                if previous_model_id == candidate["model_id"]
                else 1
            )
            state["recent_scenario_streak"] = (
                int(state.get("recent_scenario_streak") or 0) + 1
                if previous_scenario_name == candidate["scenario_name"]
                else 1
            )

            state["completed_trials"] = int(state.get("completed_trials") or 0) + 1
            state["last_success_at_kst"] = iso_now()
            state["failed_attempts_consecutive"] = 0
            state["mutation_failure_streak"] = 0
            state = _reset_backoff(state)
            _append_leaderboard_row(agent_dir, candidate, result)
            append_jsonl(
                agent_dir / "decision_log.jsonl",
                {
                    "event": "trial_completed",
                    "timestamp_kst": iso_now(),
                    "candidate_id": candidate["candidate_id"],
                    "phase": candidate["phase"],
                    "model_id": candidate["model_id"],
                    "scenario_name": candidate["scenario_name"],
                    "accuracy": float(result["metrics"].get("accuracy") or 0.0),
                    "macro_f1": float(((result["metrics"].get("macro_avg") or {}).get("f1")) or 0.0),
                    "class0_fpr": float(
                        ((result["metrics"].get("class0_metrics") or {}).get("false_positive_rate")) or 1.0
                    ),
                    "retry_attempted": bool(result.get("retry_attempted")),
                    "trial_runtime_path": result.get("trial_runtime_path"),
                    "worker_slot": worker_slot,
                },
            )

            champion_updated = _is_better_result(result, state.get("best"))
            if champion_updated:
                state["best"] = {
                    "candidate_id": candidate["candidate_id"],
                    "trial_number": candidate["trial_number"],
                    "phase": candidate["phase"],
                    "model_id": candidate["model_id"],
                    "scenario_name": candidate["scenario_name"],
                    "runner_kind": candidate["runner_kind"],
                    "hyperparameters": effective_hyperparameters,
                    "model_overrides": candidate.get("model_overrides"),
                    "metrics": result["metrics"],
                    "best_val_loss": result.get("best_val_loss"),
                    "score": result["score"],
                    "output_dir": result["output_dir"],
                    "updated_at_kst": iso_now(),
                }
                state["last_champion_update_at_kst"] = iso_now()
                if result.get("mutation_applied"):
                    refresh_snapshot_from_current(agent_dir, "champion")
                _append_champion_history(
                    agent_dir,
                    {
                        "timestamp_kst": iso_now(),
                        "candidate_id": candidate["candidate_id"],
                        "model_id": candidate["model_id"],
                        "scenario_name": candidate["scenario_name"],
                        "score": result["score"],
                        "metrics": result["metrics"],
                        "output_dir": result["output_dir"],
                    },
                )
                append_jsonl(
                    agent_dir / "decision_log.jsonl",
                    {
                        "event": "champion_updated",
                        "timestamp_kst": iso_now(),
                        "candidate_id": candidate["candidate_id"],
                        "model_id": candidate["model_id"],
                        "scenario_name": candidate["scenario_name"],
                        "score": result["score"],
                        "worker_slot": worker_slot,
                    },
                )
                if bool(config["reporting"].get("champion_report_enabled", True)):
                    write_periodic_report(agent_dir, reason="champion_update")
                worker_log(
                    agent_dir,
                    (
                        f"champion updated: id={candidate['candidate_id']} model={candidate['model_id']} "
                        f"scenario={candidate['scenario_name']} score={float(result['score']):.6f} "
                        f"acc={float(result['metrics'].get('accuracy') or 0.0):.4f} "
                        f"macro_f1={float(((result['metrics'].get('macro_avg') or {}).get('f1')) or 0.0):.4f}"
                    ),
                    slot=worker_slot,
                )
            elif result.get("mutation_applied"):
                restore_named_snapshot(agent_dir, "champion")
                append_jsonl(
                    agent_dir / "decision_log.jsonl",
                    {
                        "event": "mutation_rollback_to_champion",
                        "timestamp_kst": iso_now(),
                        "candidate_id": candidate["candidate_id"],
                        "reason": "no_improvement",
                        "worker_slot": worker_slot,
                    },
                )
                worker_log(
                    agent_dir,
                    f"mutation rollback to champion: id={candidate['candidate_id']} reason=no_improvement",
                    slot=worker_slot,
                )

            if int(state["completed_trials"]) % int(config["reporting"]["periodic_every_completed_trials"]) == 0:
                report_path = write_periodic_report(agent_dir, reason=f"periodic_{state['completed_trials']}")
                worker_log(
                    agent_dir,
                    f"periodic report written: completed_trials={state['completed_trials']} path={report_path.name}",
                    slot=worker_slot,
                )

            state["status"] = "completed" if goal_reached(config["goal"], (state.get("best") or {}).get("metrics") or {}) else "running"
            _write_study_summary(agent_dir)
            save_state(agent_dir, state)
    elif result["status"] == "invalid":
        study.tell(int(candidate["trial_number"]), state=TrialState.PRUNED)
        with metadata_lock(agent_dir):
            state = _prune_expired_pair_cooldowns(load_state(agent_dir))
            inflight = [item for item in load_inflight(agent_dir) if item.get("candidate_id") != candidate["candidate_id"]]
            save_inflight(agent_dir, inflight)

            state["attempted_trials"] = int(state.get("attempted_trials") or 0) + 1
            state["phase_trial_index"] = int(state.get("phase_trial_index") or 0) + 1
            state["updated_at_kst"] = iso_now()
            invalid_role_scenarios = dict(state.get("invalid_role_scenarios") or {})
            invalid_role_scenarios[candidate["scenario_name"]] = str(result.get("failure_reason") or "role_scenario_invalid")
            state["invalid_role_scenarios"] = invalid_role_scenarios
            append_jsonl(
                agent_dir / "decision_log.jsonl",
                {
                    "event": "scenario_invalid",
                    "timestamp_kst": iso_now(),
                    "candidate_id": candidate["candidate_id"],
                    "phase": candidate["phase"],
                    "model_id": candidate["model_id"],
                    "scenario_name": candidate["scenario_name"],
                    "reason": result.get("failure_reason"),
                    "trial_runtime_path": result.get("trial_runtime_path"),
                    "worker_slot": worker_slot,
                },
            )
            state["status"] = "running"
            _write_study_summary(agent_dir)
            save_state(agent_dir, state)
    else:
        study.tell(int(candidate["trial_number"]), state=TrialState.FAIL)
        with metadata_lock(agent_dir):
            state = _prune_expired_pair_cooldowns(load_state(agent_dir))
            inflight = [item for item in load_inflight(agent_dir) if item.get("candidate_id") != candidate["candidate_id"]]
            save_inflight(agent_dir, inflight)

            state["attempted_trials"] = int(state.get("attempted_trials") or 0) + 1
            state["phase_trial_index"] = int(state.get("phase_trial_index") or 0) + 1
            state["updated_at_kst"] = iso_now()
            state["last_candidate"] = candidate
            previous_model_id = state.get("recent_model_id")
            previous_scenario_name = state.get("recent_scenario_name")
            state["recent_model_id"] = candidate["model_id"]
            state["recent_scenario_name"] = candidate["scenario_name"]
            state["recent_model_streak"] = (
                int(state.get("recent_model_streak") or 0) + 1
                if previous_model_id == candidate["model_id"]
                else 1
            )
            state["recent_scenario_streak"] = (
                int(state.get("recent_scenario_streak") or 0) + 1
                if previous_scenario_name == candidate["scenario_name"]
                else 1
            )

            state["last_failure_at_kst"] = iso_now()
            state["status"] = "running"
            append_jsonl(
                agent_dir / "decision_log.jsonl",
                {
                    "event": "trial_failed",
                    "timestamp_kst": iso_now(),
                    "candidate_id": candidate["candidate_id"],
                    "phase": candidate["phase"],
                    "model_id": candidate["model_id"],
                    "scenario_name": candidate["scenario_name"],
                    "returncode": result.get("returncode"),
                    "reason": result.get("failure_reason"),
                    "retry_attempted": bool(result.get("retry_attempted")),
                    "cooldown_requested": bool(result.get("cooldown_requested")),
                    "nvidia_smi_snapshot": result.get("nvidia_smi_snapshot"),
                    "log_tail": list(result.get("log_tail") or []),
                    "trial_runtime_path": result.get("trial_runtime_path"),
                    "worker_slot": worker_slot,
                },
            )
            worker_log(
                agent_dir,
                (
                    f"trial recorded as failed: id={candidate['candidate_id']} model={candidate['model_id']} "
                    f"scenario={candidate['scenario_name']} reason={result.get('failure_reason')}"
                ),
                slot=worker_slot,
            )
            if bool(result.get("cooldown_requested")):
                cooldown_trials = int(result.get("cooldown_trials") or 0)
                pair_key = _pair_key(candidate["model_id"], candidate["scenario_name"])
                state.setdefault("pair_cooldowns", {})
                state["pair_cooldowns"][pair_key] = int(state["attempted_trials"]) + cooldown_trials
            if result.get("mutation_applied"):
                restore_named_snapshot(agent_dir, "champion")
                state["mutation_failure_streak"] = int(state.get("mutation_failure_streak") or 0) + 1
                if state["mutation_failure_streak"] >= int(
                    config["mutation"]["max_consecutive_failures_before_golden_reset"]
                ):
                    restore_named_snapshot(agent_dir, "golden")
                    refresh_snapshot_from_current(agent_dir, "champion")
                    state["last_golden_reset_at_kst"] = iso_now()
                    state["mutation_failure_streak"] = 0
                    append_jsonl(
                        agent_dir / "decision_log.jsonl",
                        {
                            "event": "golden_reset",
                            "timestamp_kst": iso_now(),
                            "candidate_id": candidate["candidate_id"],
                            "reason": "mutation_failure_streak",
                            "worker_slot": worker_slot,
                        },
                    )
                    worker_log(
                        agent_dir,
                        "golden reset executed after consecutive mutation failures",
                        slot=worker_slot,
                    )
            _write_study_summary(agent_dir)
            save_state(agent_dir, state)

    return load_state(agent_dir)


def _prune_non_champion_artifacts(agent_dir: Path) -> None:
    config = load_resolved_config(agent_dir)
    keep_recent = int(config["retention"]["keep_recent_non_champions"])
    rows = _load_leaderboard_rows(agent_dir)
    if len(rows) <= keep_recent + 1:
        return

    champion_output = ((load_state(agent_dir).get("best") or {}).get("output_dir")) or ""
    protected = {Path(champion_output).resolve()} if champion_output else set()
    non_champion = [row for row in rows if row.get("output_dir")]
    non_champion.sort(key=lambda row: int(row.get("trial_number") or -1), reverse=True)
    old_rows = non_champion[keep_recent:]
    for row in old_rows:
        output_dir = Path(row["output_dir"]).resolve()
        if output_dir in protected or not output_dir.exists():
            continue
        for rel in ["model.pt", "preds_test.csv", "preds_inference.csv"]:
            target = output_dir / rel
            if target.exists():
                target.unlink()
        eval_dir = output_dir / "evaluation"
        if eval_dir.exists():
            for png in eval_dir.glob("*.png"):
                png.unlink()


def run_single_iteration(agent_dir: Path, *, worker_slot: int | None = None) -> dict[str, Any]:
    candidate = propose_candidate(agent_dir, worker_slot=worker_slot)
    result = execute_candidate(agent_dir, candidate)
    state = update_after_trial(agent_dir, candidate, result)
    _prune_non_champion_artifacts(agent_dir)
    return {
        "candidate": candidate,
        "result": result,
        "state": state,
    }


def should_wait_for_backoff(state: dict[str, Any]) -> bool:
    next_retry_at = state.get("next_retry_at_kst")
    if not next_retry_at:
        return False
    try:
        next_dt = datetime.fromisoformat(str(next_retry_at))
    except ValueError:
        return False
    return next_dt > now_kst()


def sleep_with_stop_check(agent_dir: Path, seconds: float) -> bool:
    stop_path = agent_dir / "stop_requested"
    remaining = float(seconds)
    while remaining > 0:
        if stop_path.exists():
            return True
        time.sleep(min(5.0, remaining))
        remaining -= 5.0
    return stop_path.exists()


def _read_decision_events(agent_dir: Path) -> list[dict[str, Any]]:
    path = agent_dir / "decision_log.jsonl"
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _decision_metrics(agent_dir: Path) -> dict[str, Any]:
    events = _read_decision_events(agent_dir)
    failure_events = [event for event in events if event.get("event") == "trial_failed"]
    outcome_events = [
        event
        for event in events
        if event.get("event") in {"trial_completed", "trial_failed"}
    ]
    recent = outcome_events[-20:]
    recent_successes = sum(1 for event in recent if event.get("event") == "trial_completed")
    recent_ratio = (recent_successes / len(recent)) if recent else None
    return {
        "runner_exit_code_-9_count": sum(
            1 for event in failure_events if str(event.get("reason")) == "runner_exit_code_-9"
        ),
        "failure_reason_counts": dict(Counter(str(event.get("reason")) for event in failure_events)),
        "model_fail_counts": dict(Counter(str(event.get("model_id")) for event in failure_events if event.get("model_id"))),
        "recent_20_success_ratio": recent_ratio,
    }


def build_status_payload(agent_dir: Path) -> dict[str, Any]:
    state = load_state(agent_dir)
    config = load_resolved_config(agent_dir)
    best = state.get("best") or {}
    best_metrics = best.get("metrics") or {}
    worker_pids = _normalize_worker_pids(state)
    workers_alive = {slot: is_process_alive(pid) for slot, pid in worker_pids.items()}
    inflight = load_inflight(agent_dir)
    decision_metrics = _decision_metrics(agent_dir)
    invalid_role_scenarios = dict(state.get("invalid_role_scenarios") or {})
    invalid_reason_counts = dict(Counter(invalid_role_scenarios.values()))
    attempted = int(state.get("attempted_trials") or 0)
    completed = int(state.get("completed_trials") or 0)
    return {
        "run_id": agent_dir.name,
        "status": state.get("status"),
        "worker_pid": state.get("worker_pid"),
        "worker_alive": is_process_alive(state.get("worker_pid")),
        "worker_pids": worker_pids,
        "workers_alive": workers_alive,
        "parallel_workers": int(config.get("search", {}).get("parallel_workers", 1)),
        "inflight_count": len(inflight),
        "phase": state.get("phase_name"),
        "attempted_trials": attempted,
        "completed_trials": completed,
        "completed_attempted_ratio": (completed / attempted) if attempted else None,
        "current_backoff_seconds": state.get("current_backoff_seconds"),
        "next_retry_at_kst": state.get("next_retry_at_kst"),
        "last_success_at_kst": state.get("last_success_at_kst"),
        "last_failure_at_kst": state.get("last_failure_at_kst"),
        "best_model_id": best.get("model_id"),
        "best_scenario_name": best.get("scenario_name"),
        "best_accuracy": best_metrics.get("accuracy"),
        "best_macro_f1": (best_metrics.get("macro_avg") or {}).get("f1"),
        "best_output_dir": best.get("output_dir"),
        "runner_exit_code_-9_count": decision_metrics["runner_exit_code_-9_count"],
        "model_fail_counts": decision_metrics["model_fail_counts"],
        "recent_20_success_ratio": decision_metrics["recent_20_success_ratio"],
        "invalid_scenarios": invalid_role_scenarios,
        "invalid_reason_counts": invalid_reason_counts,
    }


def render_report(agent_dir: Path, detail: str) -> str:
    return _render_report(agent_dir, detail=detail)
