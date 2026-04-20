# build_labeling_queue.py - accepted + landmark_queue 완료 항목을 labeling queue manifest로 변환
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (
    LANDMARK_DATA_DIR,
    LANDMARK_QUEUE_DIR,
    LABELED_DATA_DIR,
    RAW_DATA_DIR,
    ROOT,
    SEED_PROFILE,
    ensure_seed_profile_current,
    write_jsonl,
)
from common import load_jsonl


DEFAULT_INPUT = ROOT / "manifests" / "quality_manifest.jsonl"
DEFAULT_OUTPUT = ROOT / "manifests" / "labeling_queue_manifest.jsonl"


def suggested_raw_filename(row: dict) -> str:
    stem = Path(row["path"]).stem
    source_name = row.get("source_name", "source")
    safe_source = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in source_name)
    return f"7_grab__{safe_source}__{stem}.mp4"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build labeling queue manifest from accepted videos.")
    parser.add_argument("--seed-profile", type=Path, default=SEED_PROFILE)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--landmark-dir", type=Path, default=LANDMARK_QUEUE_DIR)
    args = parser.parse_args()

    ensure_seed_profile_current(args.seed_profile)
    rows = load_jsonl(args.input)

    queue_rows: list[dict] = []
    for row in rows:
        if row.get("quality_status") != "accepted":
            continue
        landmark_csv = args.landmark_dir / f"{Path(row['path']).stem}.csv"
        if not landmark_csv.exists():
            continue
        queue_rows.append(
            {
                "original_inbox_path": row["path"],
                "landmark_queue_csv_path": str(landmark_csv.resolve()),
                "proposed_class_id": 7,
                "proposed_label": "grab",
                "suggested_final_raw_filename": suggested_raw_filename(row),
                "final_raw_path": str((RAW_DATA_DIR / suggested_raw_filename(row)).resolve()),
                "expected_label_csv_path": str((LABELED_DATA_DIR / f"{Path(suggested_raw_filename(row)).stem}.csv").resolve()),
                "final_landmark_csv_path": str((LANDMARK_DATA_DIR / f"{Path(suggested_raw_filename(row)).stem}.csv").resolve()),
                "quality_summary": {
                    "duration_sec": row.get("duration_sec"),
                    "fps": row.get("fps"),
                    "width": row.get("width"),
                    "height": row.get("height"),
                    "hand_detect_ratio": row.get("hand_detect_ratio"),
                },
                "source_metadata": {
                    "source_tier": row.get("source_tier"),
                    "source_name": row.get("source_name"),
                    "title": row.get("title"),
                    "original_url": row.get("original_url"),
                    "license_note": row.get("license_note"),
                },
            }
        )

    write_jsonl(args.output, queue_rows)
    print(json.dumps({"output": str(args.output), "count": len(queue_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
