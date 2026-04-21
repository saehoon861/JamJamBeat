# build_training_datasets.py - source 단위로 train/val만 분할해 역할형 CSV를 생성한다.
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TRAIN_SOURCE_DIR = ROOT / "기존데이터셋"
TEST_SOURCE_DIR = ROOT / "테스트데이터셋"
OUTPUT_DIR = ROOT / "학습데이터셋"

SEED = 42
TRAIN_SOURCE_COUNT = 49
VAL_SOURCE_COUNT = 7
EXPECTED_SOURCE_COUNT = TRAIN_SOURCE_COUNT + VAL_SOURCE_COUNT
EXPECTED_GESTURE_KEYS = tuple(str(i) for i in range(7))

# 추가: 증강 관련 컬럼 기본값
AUGMENTATION_DEFAULTS = {
    "aug_mirror": False,
    "aug_blp": False,
    "aug_noise_sigma": 0.0,
}


@dataclass(frozen=True)
class DatasetSpec:
    dataset_key: str
    normalization_family: str
    train_source: Path
    test_source: Path


SPECS = (
    DatasetSpec(
        dataset_key="pos_scale",
        normalization_family="pos_scale",
        train_source=TRAIN_SOURCE_DIR / "pos_scale_aug.csv",
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


def _gesture_key(source_name: str) -> str:
    return source_name.split("_", 1)[0]


def _canonical_source_split(specs: tuple[DatasetSpec, ...]) -> dict[str, object]:
    if not specs:
        raise ValueError("No dataset specs configured")

    canonical_sources = _source_list(specs[0].train_source)
    if len(canonical_sources) != EXPECTED_SOURCE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SOURCE_COUNT} unique source_file values, "
            f"got {len(canonical_sources)} from {specs[0].train_source.name}"
        )

    for spec in specs[1:]:
        current_sources = _source_list(spec.train_source)
        if current_sources != canonical_sources:
            raise ValueError(
                f"source_file mismatch for {spec.dataset_key}: {spec.train_source.name}"
            )

    gesture_groups: dict[str, list[str]] = {key: [] for key in EXPECTED_GESTURE_KEYS}
    for source_name in canonical_sources:
        gesture_key = _gesture_key(source_name)
        if gesture_key not in gesture_groups:
            raise ValueError(f"Unexpected gesture key in source_file: {source_name}")
        gesture_groups[gesture_key].append(source_name)

    for gesture_key in EXPECTED_GESTURE_KEYS:
        gesture_groups[gesture_key] = sorted(gesture_groups[gesture_key])
        if len(gesture_groups[gesture_key]) != 8:
            raise ValueError(
                f"Expected 8 sources for gesture {gesture_key}, "
                f"got {len(gesture_groups[gesture_key])}"
            )

    rng = np.random.default_rng(SEED)
    shuffled_sources = canonical_sources.copy()
    rng.shuffle(shuffled_sources)

    train_sources = sorted(shuffled_sources[:TRAIN_SOURCE_COUNT])
    val_sources = sorted(
        shuffled_sources[TRAIN_SOURCE_COUNT:TRAIN_SOURCE_COUNT + VAL_SOURCE_COUNT]
    )

    if len(train_sources) != TRAIN_SOURCE_COUNT:
        raise ValueError(f"Expected {TRAIN_SOURCE_COUNT} train sources, got {len(train_sources)}")
    if len(val_sources) != VAL_SOURCE_COUNT:
        raise ValueError(f"Expected {VAL_SOURCE_COUNT} val sources, got {len(val_sources)}")

    combined_sources = set(train_sources) | set(val_sources)
    if len(combined_sources) != len(canonical_sources):
        raise ValueError("Source split does not cover the full canonical source set")

    return {
        "train": train_sources,
        "val": val_sources,
        "gesture_groups": gesture_groups,
    }


def _filter_by_sources(frame: pd.DataFrame, sources: list[str]) -> pd.DataFrame:
    source_set = set(sources)
    return frame[frame["source_file"].astype(str).isin(source_set)].copy()


def _align_columns_with_augmentation_defaults(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    train/test 간 컬럼 불일치를 줄이기 위해,
    증강 메타 컬럼이 없는 쪽에는 기본값을 채워 넣고
    최종적으로 test 컬럼 순서를 train 기준으로 맞춘다.
    """

    # 1) 증강 컬럼 기본값 채우기
    for col, default_value in AUGMENTATION_DEFAULTS.items():
        if col not in train_df.columns:
            train_df[col] = default_value
        if col not in test_df.columns:
            test_df[col] = default_value

    # 2) 아직도 서로 다른 컬럼이 있으면 확인
    train_only_cols = [c for c in train_df.columns if c not in test_df.columns]
    test_only_cols = [c for c in test_df.columns if c not in train_df.columns]

    if train_only_cols or test_only_cols:
        raise ValueError(
            "Column mismatch remains after augmentation alignment.\n"
            f"train_only_cols={train_only_cols}\n"
            f"test_only_cols={test_only_cols}"
        )

    # 3) test 컬럼 순서를 train 기준으로 맞춤
    test_df = test_df[train_df.columns.tolist()]

    return train_df, test_df


def build() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_split = _canonical_source_split(SPECS)
    summary_rows: list[dict[str, object]] = []

    for spec in SPECS:
        train_source_df = _load_csv(spec.train_source)
        test_source_df = _load_csv(spec.test_source)

        # 추가: train/test 컬럼 정렬 및 증강 컬럼 보정
        train_source_df, test_source_df = _align_columns_with_augmentation_defaults(
            train_source_df,
            test_source_df,
        )

        if list(train_source_df.columns) != list(test_source_df.columns):
            raise ValueError(
                f"Column mismatch for {spec.dataset_key}: "
                f"{spec.train_source.name} vs {spec.test_source.name}"
            )

        for split_name in ("train", "val"):
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

    split_manifest = {
        "seed": SEED,
        "source_counts": {
            "train": TRAIN_SOURCE_COUNT,
            "val": VAL_SOURCE_COUNT,
        },
        "train_sources": source_split["train"],
        "val_sources": source_split["val"],
    }
    with (OUTPUT_DIR / "source_split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(split_manifest, f, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    summary = build()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()