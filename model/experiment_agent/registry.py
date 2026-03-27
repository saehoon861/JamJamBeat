# registry.py - dataset/model registry and config resolution for the experiment agent
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "model" / "data_fusion"
TEST_ROOT = DATA_ROOT / "테스트데이터셋"

TEST_CSV_BY_FAMILY = {
    "baseline": "total_data_test_baseline.csv",
    "pos_only": "total_data_test_pos_only.csv",
    "pos_scale": "total_data_test_pos_scale.csv",
    "scale_only": "total_data_test_scale_only.csv",
}

DEFAULT_ROLE_MODELS = [
    "mlp_original",
    "mlp_embedding",
    "frame_spatial_transformer",
]

DEFAULT_EXPLICIT_MODELS = [
    "edge_stgu_mlp",
    "sparse_masked_mlp",
    "hierarchical_tree_mlp",
    "lappe_dist_mixer",
]

DEFAULT_WHITELIST_GLOBS = [
    "model/model_pipelines/*/model.py",
    "model/model_pipelines/*/dataset.py",
    "model/model_pipelines/_shared.py",
    "model/model_pipelines/run_pipeline.py",
    "model/model_pipelines/new_run_pipeline.py",
]


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    train_csv: str
    test_csv: str
    family: str
    source: str
    notes: str = ""


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    runner_kind: str
    resource_tier: str = "medium"
    supports_model_overrides: bool = False
    supports_code_mutation: bool = False
    mutation_target_file: str | None = None
    common_space_overrides: dict[str, list[Any]] = field(default_factory=dict)
    override_space: dict[str, list[Any]] = field(default_factory=dict)


def infer_family_from_stem(stem: str) -> str | None:
    if stem.startswith("baseline") or stem.endswith("_none"):
        return "baseline"
    if "pos_scale" in stem:
        return "pos_scale"
    if "scale" in stem:
        return "scale_only"
    if "pos" in stem:
        return "pos_only"
    return None


def discover_default_scenarios(
    *,
    use_existing_root: bool = True,
    use_augmented_frame: bool = True,
    use_augmented_sequence: bool = False,
    family_locked_pairing: bool = True,
) -> list[ScenarioSpec]:
    scenarios: list[ScenarioSpec] = []
    seen: set[tuple[str, str]] = set()
    roots: list[tuple[str, Path]] = []
    if use_existing_root:
        roots.append(("existing", DATA_ROOT / "기존데이터셋"))
    if use_augmented_frame or use_augmented_sequence:
        roots.append(("augmented", DATA_ROOT / "증강데이터셋"))

    for source, root in roots:
        if not root.exists():
            continue
        for csv_path in sorted(root.glob("*.csv")):
            stem = csv_path.stem
            if source == "augmented":
                is_sequence_aug = "seq_aug" in stem
                is_frame_aug = "frame_aug" in stem
                if is_sequence_aug and not use_augmented_sequence:
                    continue
                if is_frame_aug and not use_augmented_frame:
                    continue
                if not is_sequence_aug and not is_frame_aug:
                    continue

            family = infer_family_from_stem(stem)
            if family is None:
                continue
            if family_locked_pairing:
                test_name = TEST_CSV_BY_FAMILY.get(family)
                if not test_name:
                    continue
                test_csv = TEST_ROOT / test_name
                if not test_csv.exists():
                    continue
                scenario = ScenarioSpec(
                    name=stem,
                    train_csv=str(csv_path.resolve()),
                    test_csv=str(test_csv.resolve()),
                    family=family,
                    source=source,
                    notes=f"family_locked:{family}",
                )
                dedupe_key = (scenario.name, scenario.train_csv)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                scenarios.append(scenario)
            else:
                for family_name, test_name in TEST_CSV_BY_FAMILY.items():
                    test_csv = TEST_ROOT / test_name
                    if not test_csv.exists():
                        continue
                    scenario = ScenarioSpec(
                        name=f"{stem}__to__{family_name}",
                        train_csv=str(csv_path.resolve()),
                        test_csv=str(test_csv.resolve()),
                        family=family,
                        source=source,
                        notes=f"cross_family:{family}->{family_name}",
                    )
                    dedupe_key = (scenario.name, scenario.train_csv)
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    scenarios.append(scenario)

    return scenarios


def get_model_registry() -> dict[str, ModelSpec]:
    registry = {
        "mlp_original": ModelSpec(
            model_id="mlp_original",
            runner_kind="role_based",
            resource_tier="light",
            common_space_overrides={"batch_size": [64, 128]},
        ),
        "mlp_sequence_delta": ModelSpec(model_id="mlp_sequence_delta", runner_kind="role_based"),
        "mlp_embedding": ModelSpec(
            model_id="mlp_embedding",
            runner_kind="role_based",
            resource_tier="light",
            common_space_overrides={"batch_size": [64, 128]},
        ),
        "cnn1d_tcn": ModelSpec(model_id="cnn1d_tcn", runner_kind="role_based"),
        "transformer_embedding": ModelSpec(model_id="transformer_embedding", runner_kind="role_based"),
        "frame_spatial_transformer": ModelSpec(
            model_id="frame_spatial_transformer",
            runner_kind="role_based",
            resource_tier="medium",
            common_space_overrides={"batch_size": [32, 64]},
        ),
        "edge_stgu_mlp": ModelSpec(
            model_id="edge_stgu_mlp",
            runner_kind="explicit_train_test",
            resource_tier="heavy",
            supports_model_overrides=True,
            supports_code_mutation=True,
            mutation_target_file="model/model_pipelines/edge_stgu_mlp/dataset.py",
            common_space_overrides={"batch_size": [16, 32]},
            override_space={
                "d_model": [96, 128, 160, 192, 224],
                "num_layers": [2, 3, 4],
                "gate_hidden": [32, 64, 96, 128],
            },
        ),
        "sparse_masked_mlp": ModelSpec(
            model_id="sparse_masked_mlp",
            runner_kind="explicit_train_test",
            resource_tier="heavy",
            supports_model_overrides=True,
            supports_code_mutation=True,
            mutation_target_file="model/model_pipelines/sparse_masked_mlp/dataset.py",
            common_space_overrides={"batch_size": [16, 32]},
            override_space={
                "hidden_dim": [32, 48, 64],
                "readout_dim": [64, 128, 192],
                "dropout": [0.0, 0.1, 0.2],
            },
        ),
        "hierarchical_tree_mlp": ModelSpec(
            model_id="hierarchical_tree_mlp",
            runner_kind="explicit_train_test",
            resource_tier="medium",
            supports_model_overrides=True,
            supports_code_mutation=True,
            mutation_target_file="model/model_pipelines/hierarchical_tree_mlp/dataset.py",
            common_space_overrides={"batch_size": [32, 64]},
            override_space={
                "hidden_dim": [48, 64, 96],
                "root_hidden_dim": [16, 32, 64],
                "edge_hidden_dim": [16, 32, 64],
                "readout_hidden_dim": [64, 128, 192],
            },
        ),
        "lappe_dist_mixer": ModelSpec(
            model_id="lappe_dist_mixer",
            runner_kind="explicit_train_test",
            resource_tier="medium",
            supports_model_overrides=True,
            supports_code_mutation=True,
            mutation_target_file="model/model_pipelines/lappe_dist_mixer/dataset.py",
            common_space_overrides={"batch_size": [32, 64]},
            override_space={
                "lappe_dim": [4, 8, 12],
                "hidden_dim": [48, 64, 96],
                "channel_mlp_hidden": [96, 128, 192],
                "num_layers": [2, 3, 4],
                "lappe_sign_flip": [True, False],
            },
        ),
    }
    return registry


def _resolved_models_from_ids(
    *,
    role_based_ids: list[str],
    explicit_ids: list[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    registry = get_model_registry()
    resolved: dict[str, dict[str, dict[str, Any]]] = {
        "role_based": {},
        "explicit_train_test": {},
    }

    for model_id in role_based_ids:
        spec = registry.get(model_id)
        if spec is None:
            raise ValueError(f"Unknown model id in role_based pool: {model_id}")
        if spec.runner_kind != "role_based":
            raise ValueError(f"{model_id} is not a role_based runner model")
        resolved["role_based"][model_id] = asdict(spec)

    for model_id in explicit_ids:
        spec = registry.get(model_id)
        if spec is None:
            raise ValueError(f"Unknown model id in explicit_train_test pool: {model_id}")
        if spec.runner_kind != "explicit_train_test":
            raise ValueError(f"{model_id} is not an explicit_train_test runner model")
        resolved["explicit_train_test"][model_id] = asdict(spec)

    return resolved


def resolve_agent_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    config = dict(raw_config)

    scenario_cfg = dict(config.get("scenarios") or {})
    use_existing_root = bool(scenario_cfg.get("use_existing_root", True))
    use_augmented_frame = bool(
        scenario_cfg.get("use_augmented_frame", scenario_cfg.get("include_augmented", True))
    )
    use_augmented_sequence = bool(
        scenario_cfg.get("use_augmented_sequence", scenario_cfg.get("include_sequence_augmented", False))
    )
    family_locked_pairing = bool(scenario_cfg.get("family_locked_pairing", True))
    auto_discover = bool(scenario_cfg.get("auto_discover", True))

    if auto_discover:
        resolved_scenarios = [
            asdict(s)
            for s in discover_default_scenarios(
                use_existing_root=use_existing_root,
                use_augmented_frame=use_augmented_frame,
                use_augmented_sequence=use_augmented_sequence,
                family_locked_pairing=family_locked_pairing,
            )
        ]
    else:
        resolved_scenarios = list(scenario_cfg.get("items") or [])

    if not resolved_scenarios:
        raise ValueError("No scenarios resolved for experiment agent.")

    model_cfg = dict(config.get("models") or {})
    if "role_based" in model_cfg:
        role_based_ids = list(model_cfg.get("role_based") or [])
    else:
        role_based_ids = list(DEFAULT_ROLE_MODELS)

    if "explicit_train_test" in model_cfg:
        explicit_ids = list(model_cfg.get("explicit_train_test") or [])
    else:
        explicit_ids = list(DEFAULT_EXPLICIT_MODELS)

    config["scenarios"] = {
        "auto_discover": auto_discover,
        "use_existing_root": use_existing_root,
        "use_augmented_frame": use_augmented_frame,
        "use_augmented_sequence": use_augmented_sequence,
        "family_locked_pairing": family_locked_pairing,
        "items": resolved_scenarios,
    }
    config["models"] = _resolved_models_from_ids(
        role_based_ids=role_based_ids,
        explicit_ids=explicit_ids,
    )

    mutation_cfg = dict(config.get("mutation") or {})
    config["mutation"] = {
        "enabled": bool(mutation_cfg.get("enabled", True)),
        "max_consecutive_failures_before_golden_reset": int(
            mutation_cfg.get("max_consecutive_failures_before_golden_reset", 3)
        ),
        "allowed_globs": list(mutation_cfg.get("allowed_globs") or DEFAULT_WHITELIST_GLOBS),
    }

    diagnostics_cfg = dict(config.get("diagnostics") or {})
    config["diagnostics"] = {
        "collect_proc_diskstats": bool(diagnostics_cfg.get("collect_proc_diskstats", False)),
        "collect_iostat": bool(diagnostics_cfg.get("collect_iostat", False)),
    }

    return config
