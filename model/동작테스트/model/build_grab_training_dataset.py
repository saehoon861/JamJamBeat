# build_grab_training_dataset.py - grab 라벨/랜드마크 CSV를 학습용 CSV로 병합

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LABELED_DIR = ROOT / "labeled_data"
LANDMARK_DIR = ROOT / "landmark_data"
OUTPUT_DIR = Path(__file__).resolve().parent

DATASET_NAMES = ["7_grab", "7_grab_2"]
EPS = 1e-8


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_pos_scale_row(row: dict[str, str], value_fields: list[str]) -> dict[str, str]:
    if not any(row.get(field, "") for field in value_fields):
        return {field: "" for field in value_fields}

    points: list[list[float]] = []
    for landmark_index in range(21):
        coords = []
        for axis in ("x", "y", "z"):
            field = f"{axis}{landmark_index}"
            coords.append(float(row[field]))
        points.append(coords)

    origin_x, origin_y, origin_z = points[0]
    middle_x, middle_y, middle_z = points[9]
    denom = math.sqrt(
        (middle_x - origin_x) ** 2
        + (middle_y - origin_y) ** 2
        + (middle_z - origin_z) ** 2
    )
    scale = 1.0 if denom <= EPS else 1.0 / denom

    normalized: dict[str, str] = {}
    for landmark_index, (x, y, z) in enumerate(points):
        normalized[f"x{landmark_index}"] = repr((x - origin_x) * scale)
        normalized[f"y{landmark_index}"] = repr((y - origin_y) * scale)
        normalized[f"z{landmark_index}"] = repr((z - origin_z) * scale)
    return normalized


def merge_pair(dataset_name: str) -> tuple[list[dict[str, str]], list[str]]:
    label_path = LABELED_DIR / f"{dataset_name}.csv"
    landmark_path = LANDMARK_DIR / f"{dataset_name}.csv"

    label_rows = load_rows(label_path)
    landmark_rows = load_rows(landmark_path)

    if len(label_rows) != len(landmark_rows):
        raise ValueError(
            f"{dataset_name}: row count mismatch "
            f"(labels={len(label_rows)}, landmarks={len(landmark_rows)})"
        )

    if not landmark_rows:
        raise ValueError(f"{dataset_name}: no rows found")

    landmark_fields = list(landmark_rows[0].keys())
    landmark_value_fields = [field for field in landmark_fields if field not in {"frame_idx", "timestamp", "gesture"}]
    output_fields = ["source_file", "frame_idx", "timestamp", "gesture", *landmark_value_fields]

    merged_rows: list[dict[str, str]] = []
    for index, (label_row, landmark_row) in enumerate(zip(label_rows, landmark_rows), start=1):
        if label_row["frame_idx"] != landmark_row["frame_idx"] or label_row["timestamp"] != landmark_row["timestamp"]:
            raise ValueError(
                f"{dataset_name}: frame mismatch at row {index} "
                f"(label={label_row['frame_idx']} {label_row['timestamp']}, "
                f"landmark={landmark_row['frame_idx']} {landmark_row['timestamp']})"
            )

        merged_row = {
            "source_file": dataset_name,
            "frame_idx": label_row["frame_idx"],
            "timestamp": label_row["timestamp"],
            "gesture": label_row["gesture"],
        }
        merged_row.update(normalize_pos_scale_row(landmark_row, landmark_value_fields))
        merged_rows.append(merged_row)

    return merged_rows, output_fields


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    combined_rows: list[dict[str, str]] = []
    combined_fieldnames: list[str] | None = None

    for dataset_name in DATASET_NAMES:
        rows, fieldnames = merge_pair(dataset_name)
        if combined_fieldnames is None:
            combined_fieldnames = fieldnames
        elif combined_fieldnames != fieldnames:
            raise ValueError(f"{dataset_name}: fieldnames do not match previous dataset")

        write_rows(OUTPUT_DIR / f"{dataset_name}_train.csv", rows, fieldnames)
        combined_rows.extend(rows)

    if combined_fieldnames is None:
        raise ValueError("No dataset rows were generated")

    write_rows(OUTPUT_DIR / "pos_scale_train_grab.csv", combined_rows, combined_fieldnames)

    print(
        f"Generated {len(DATASET_NAMES)} per-source datasets and "
        f"{len(combined_rows)} combined rows at {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
