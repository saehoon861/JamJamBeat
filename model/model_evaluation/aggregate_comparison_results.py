#!/usr/bin/env python3
# aggregate_comparison_results.py - pipelines의 comparison_results를 dataset_label 하나와 함께 간단히 모은다.
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINES_ROOT = PROJECT_ROOT / "model" / "model_evaluation" / "pipelines"
OUTPUT_CSV = PROJECT_ROOT / "model" / "model_evaluation" / "comparison_results_all_labeled.csv"
OUTPUT_MD = PROJECT_ROOT / "model" / "model_evaluation" / "comparison_results_all_labeled.md"
OUTPUT_XLSX = PROJECT_ROOT / "model" / "model_evaluation" / "comparison_results_all_labeled.xlsx"

DISPLAY_COLUMNS = [
    "dataset_label",
    "model_id",
    "mode",
    "accuracy",
    "macro_f1",
    "class0_fpr",
    "class0_fnr",
    "fp_per_min",
    "latency_p50_ms",
    "epochs_ran",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_dataset_label(input_names: list[str], suite_name: str) -> str:
    lower_names = [name.lower() for name in input_names]
    joined = " ".join(lower_names)

    dataset_keys = []
    for name in lower_names:
        for suffix in ("_train.csv", "_val.csv", "_inference.csv", "_test.csv"):
            if name.endswith(suffix):
                dataset_keys.append(name[: -len(suffix)])

    if dataset_keys:
        dataset_key = sorted(set(dataset_keys))[0]
        return dataset_key

    if any(name.endswith("_trainval.csv") for name in lower_names) and any(
        name.endswith("_test.csv") for name in lower_names
    ):
        if "pos_scale" in joined:
            return "고정 테스트셋 | pos_scale"
        if "pos_only" in joined:
            return "고정 테스트셋 | pos_only"
        if "scale_only" in joined:
            return "고정 테스트셋 | scale_only"
        return "고정 테스트셋 | baseline"

    for name in lower_names:
        match = re.match(r"(ds_[14])_(none|pos|scale|pos_scale)\.csv$", name)
        if match:
            ds_group = match.group(1).upper()
            variant = match.group(2)
            if variant == "none":
                variant = "baseline"
            elif variant == "pos":
                variant = "pos_only"
            elif variant == "scale":
                variant = "scale_only"
            return f"{ds_group} | {variant}"

    if lower_names == ["baseline.csv"]:
        return "전체 통합셋 | baseline"
    if lower_names == ["pos_only.csv"]:
        return "전체 통합셋 | pos_only"
    if lower_names == ["scale_only.csv"]:
        return "전체 통합셋 | scale_only"
    if lower_names == ["pos_scale.csv"]:
        return "전체 통합셋 | pos_scale"
    if lower_names == ["poc_base.csv"]:
        return "POC 통합셋 | baseline"

    if len(lower_names) >= 4 and all("right_for_poc_notnull.csv" in name for name in lower_names):
        return "POC 사용자별 4파일 | notnull"
    if len(lower_names) >= 4 and all("right_for_poc.csv" in name for name in lower_names):
        return "POC 사용자별 4파일 | raw"

    return f"기타 | {suite_name}"


def collect_rows() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for csv_path in sorted(PIPELINES_ROOT.glob("*/comparison_results.csv")):
        suite_dir = csv_path.parent
        suite_name = suite_dir.name
        suite_json_path = suite_dir / "comparison_suite.json"
        input_names: list[str] = []
        dataset_label: str | None = None
        if suite_json_path.exists():
            suite_json = load_json(suite_json_path)
            input_names = suite_json.get("input_csv_names") or []
            dataset_key = suite_json.get("dataset_key")
            if dataset_key:
                dataset_label = str(dataset_key)

        if not dataset_label:
            dataset_label = infer_dataset_label(input_names, suite_name)

        df = pd.read_csv(csv_path).copy()
        df.insert(0, "dataset_label", dataset_label)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No comparison_results.csv found under {PIPELINES_ROOT}")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.loc[:, DISPLAY_COLUMNS]
    merged = merged.sort_values(
        by=["dataset_label", "accuracy", "macro_f1", "model_id"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    return merged


def format_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = [[str(value) for value in row] for row in df.to_numpy().tolist()]
    widths = []
    for idx, header in enumerate(headers):
        values = [header] + [row[idx] for row in rows]
        widths.append(max(len(value) for value in values))

    header_line = "  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
    sep_line = "  ".join("-" * widths[idx] for idx in range(len(headers)))
    body_lines = [
        "  ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers)))
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body_lines])


def build_markdown(merged: pd.DataFrame) -> str:
    lines = ["# comparison_results_all_labeled", ""]

    for dataset_label, group_df in merged.groupby("dataset_label", sort=True):
        section_df = group_df.drop(columns=["dataset_label"]).reset_index(drop=True)
        lines.append(f"## {dataset_label}")
        lines.append("")
        lines.append("```text")
        lines.append(format_table(section_df))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    merged = collect_rows()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    # Excel on Windows recognizes UTF-8 CSV reliably when BOM is present.
    merged.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        merged.to_excel(writer, index=False, sheet_name="all_results")
    OUTPUT_MD.write_text(build_markdown(merged), encoding="utf-8")
    print(f"[aggregate] wrote {OUTPUT_CSV}")
    print(f"[aggregate] wrote {OUTPUT_XLSX}")
    print(f"[aggregate] wrote {OUTPUT_MD}")
    print(f"[aggregate] rows={len(merged)} datasets={merged['dataset_label'].nunique()}")


if __name__ == "__main__":
    main()
