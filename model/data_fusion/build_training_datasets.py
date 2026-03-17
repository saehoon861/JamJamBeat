# build_training_datasets.py - 12개 학습원천을 source 단위로 40/6/10 분할해 역할형 CSV를 생성한다.
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TRAIN_SOURCE_DIR = ROOT / "기존데이터셋"
TEST_SOURCE_DIR = ROOT / "테스트데이터셋"
OUTPUT_DIR = ROOT / "학습데이터셋"

SEED = 42
TRAIN_SOURCE_COUNT = 40
VAL_SOURCE_COUNT = 6
INFERENCE_SOURCE_COUNT = 10
EXPECTED_SOURCE_COUNT = TRAIN_SOURCE_COUNT + VAL_SOURCE_COUNT + INFERENCE_SOURCE_COUNT


@dataclass(frozen=True)
class DatasetSpec:
    dataset_key: str
    normalization_family: str
    train_source: Path
    test_source: Path


SPECS = (
    DatasetSpec(
        dataset_key="baseline",
        normalization_family="baseline",
        train_source=TRAIN_SOURCE_DIR / "baseline.csv",
        test_source=TEST_SOURCE_DIR / "total_data_test_baseline.csv",
    ),
    DatasetSpec(
        dataset_key="baseline_ds_1_none",
        normalization_family="baseline",
        train_source=TRAIN_SOURCE_DIR / "baseline_ds_1_none.csv",
        test_source=TEST_SOURCE_DIR / "total_data_test_baseline.csv",
    ),
    DatasetSpec(
        dataset_key="baseline_ds_4_none",
        normalization_family="baseline",
        train_source=TRAIN_SOURCE_DIR / "baseline_ds_4_none.csv",
        test_source=TEST_SOURCE_DIR / "total_data_test_baseline.csv",
    ),
    DatasetSpec(
        dataset_key="pos_only",
        normalization_family="pos_only",
        train_source=TRAIN_SOURCE_DIR / "pos_only.csv",
        test_source=TEST_SOURCE_DIR / "total_data_test_pos_only.csv",
    ),
    DatasetSpec(
        dataset_key="pos_only_ds_1_pos",
        normalization_family="pos_only",
        train_source=TRAIN_SOURCE_DIR / "pos_only_ds_1_pos.csv",
        test_source=TEST_SOURCE_DIR / "total_data_test_pos_only.csv",
    ),
    DatasetSpec(
        dataset_key="pos_only_ds_4_pos",
        normalization_family="pos_only",
        train_source=TRAIN_SOURCE_DIR / "pos_only_ds_4_pos.csv",
        test_source=TEST_SOURCE_DIR / "total_data_test_pos_only.csv",
    ),
    DatasetSpec(
        dataset_key="scale_only",
        normalization_family="scale_only",
        train_source=TRAIN_SOURCE_DIR / "scale_only.csv",
        test_source=TEST_SOURCE_DIR / "total_data_test_scale_only.csv",
    ),
    DatasetSpec(
        dataset_key="scale_only_ds_1_scale",
        normalization_family="scale_only",
        train_source=TRAIN_SOURCE_DIR / "scale_only_ds_1_scale.csv",
        test_source=TEST_SOURCE_DIR / "total_data_test_scale_only.csv",
    ),
    DatasetSpec(
        dataset_key="scale_only_ds_4_scale",
        normalization_family="scale_only",
        train_source=TRAIN_SOURCE_DIR / "scale_only_ds_4_scale.csv",
        test_source=TEST_SOURCE_DIR / "total_data_test_scale_only.csv",
    ),
    DatasetSpec(
        dataset_key="pos_scale",
        normalization_family="pos_scale",
        train_source=TRAIN_SOURCE_DIR / "pos_scale.csv",
        test_source=TEST_SOURCE_DIR / "total_data_test_pos_scale.csv",
    ),
    DatasetSpec(
        dataset_key="pos_scale_ds_1_pos_scale",
        normalization_family="pos_scale",
        train_source=TRAIN_SOURCE_DIR / "pos_scale_ds_1_pos_scale.csv",
        test_source=TEST_SOURCE_DIR / "total_data_test_pos_scale.csv",
    ),
    DatasetSpec(
        dataset_key="pos_scale_ds_4_pos_scale",
        normalization_family="pos_scale",
        train_source=TRAIN_SOURCE_DIR / "pos_scale_ds_4_pos_scale.csv",
        test_source=TEST_SOURCE_DIR / "total_data_test_pos_scale.csv",
    ),
)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def _source_list(path: Path) -> list[str]:
    frame = pd.read_csv(path, usecols=["source_file"])
    return sorted(str(v) for v in frame["source_file"].dropna().unique().tolist())


def _canonical_source_split(specs: tuple[DatasetSpec, ...]) -> dict[str, set[str]]:
    if not specs:
        raise ValueError("No dataset specs configured")

    canonical_sources = _source_list(specs[0].train_source)
    if len(canonical_sources) != EXPECTED_SOURCE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SOURCE_COUNT} unique source_file values, "
            f"got {len(canonical_sources)} from {specs[0].train_source.name}"
        )

    canonical_set = set(canonical_sources)
    for spec in specs[1:]:
        current_sources = _source_list(spec.train_source)
        if current_sources != canonical_sources:
            raise ValueError(
                f"source_file mismatch for {spec.dataset_key}: {spec.train_source.name}"
            )

    rng = np.random.default_rng(SEED)
    shuffled = list(canonical_sources)
    rng.shuffle(shuffled)

    train_sources = set(shuffled[:TRAIN_SOURCE_COUNT])
    val_sources = set(
        shuffled[TRAIN_SOURCE_COUNT:TRAIN_SOURCE_COUNT + VAL_SOURCE_COUNT]
    )
    inference_sources = set(
        shuffled[
            TRAIN_SOURCE_COUNT + VAL_SOURCE_COUNT:
            TRAIN_SOURCE_COUNT + VAL_SOURCE_COUNT + INFERENCE_SOURCE_COUNT
        ]
    )

    if len(train_sources | val_sources | inference_sources) != len(canonical_set):
        raise ValueError("Source split does not cover the full canonical source set")

    return {
        "train": train_sources,
        "val": val_sources,
        "inference": inference_sources,
    }


def _filter_by_sources(frame: pd.DataFrame, sources: set[str]) -> pd.DataFrame:
    return frame[frame["source_file"].astype(str).isin(sources)].copy()


def build() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_split = _canonical_source_split(SPECS)
    summary_rows: list[dict[str, object]] = []

    for spec in SPECS:
        train_source_df = _load_csv(spec.train_source)
        test_source_df = _load_csv(spec.test_source)

        if list(train_source_df.columns) != list(test_source_df.columns):
            raise ValueError(
                f"Column mismatch for {spec.dataset_key}: "
                f"{spec.train_source.name} vs {spec.test_source.name}"
            )

        for split_name in ("train", "val", "inference"):
            split_df = _filter_by_sources(train_source_df, source_split[split_name])
            output_path = OUTPUT_DIR / f"{spec.dataset_key}_{split_name}.csv"
            _write_csv(split_df, output_path)
            summary_rows.append(
                {
                    "dataset_key": spec.dataset_key,
                    "normalization_family": spec.normalization_family,
                    "train_source_file": str(spec.train_source.relative_to(ROOT)),
                    "test_source_file": str(spec.test_source.relative_to(ROOT)),
                    "split": split_name,
                    "source_count": int(split_df["source_file"].nunique()),
                    "row_count": int(len(split_df)),
                    "seed": SEED,
                    "source_origin": str(spec.train_source.relative_to(ROOT)),
                    "output_file": str(output_path.relative_to(ROOT)),
                }
            )

        test_output_path = OUTPUT_DIR / f"{spec.dataset_key}_test.csv"
        _write_csv(test_source_df, test_output_path)
        summary_rows.append(
            {
                "dataset_key": spec.dataset_key,
                "normalization_family": spec.normalization_family,
                "train_source_file": str(spec.train_source.relative_to(ROOT)),
                "test_source_file": str(spec.test_source.relative_to(ROOT)),
                "split": "test",
                "source_count": int(test_source_df["source_file"].nunique()),
                "row_count": int(len(test_source_df)),
                "seed": SEED,
                "source_origin": str(spec.test_source.relative_to(ROOT)),
                "output_file": str(test_output_path.relative_to(ROOT)),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "dataset_manifest.csv", index=False)
    return summary


def main() -> None:
    summary = build()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
