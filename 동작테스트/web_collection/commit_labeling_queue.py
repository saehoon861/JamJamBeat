# commit_labeling_queue.py - labeling queue 항목을 raw_data 승격(stage) 및 landmark_data 등록(finalize)
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from common import (
    LANDMARK_DATA_DIR,
    LANDMARK_QUEUE_DIR,
    LABELED_DATA_DIR,
    RAW_DATA_DIR,
    ROOT,
    SEED_PROFILE,
    ensure_seed_profile_current,
    load_jsonl,
)


DEFAULT_QUEUE = ROOT / "manifests" / "labeling_queue_manifest.jsonl"


def load_queue(path: Path) -> list[dict]:
    return load_jsonl(path)


def queue_state(row: dict) -> str:
    raw_path = Path(row["final_raw_path"])
    label_path = Path(row["expected_label_csv_path"])
    landmark_queue_path = Path(row["landmark_queue_csv_path"])
    final_landmark_path = Path(row["final_landmark_csv_path"])

    if raw_path.exists() and label_path.exists() and final_landmark_path.exists():
        return "committed"
    if raw_path.exists() and label_path.exists() and landmark_queue_path.exists():
        return "ready_to_finalize"
    if raw_path.exists() and not label_path.exists():
        return "waiting_for_labeling"
    if not raw_path.exists() and Path(row["original_inbox_path"]).exists():
        return "ready_to_stage"
    if not raw_path.exists() and not Path(row["original_inbox_path"]).exists() and not final_landmark_path.exists():
        return "missing_source"
    return "unknown"


def stage_row(row: dict) -> dict:
    src = Path(row["original_inbox_path"])
    dst = Path(row["final_raw_path"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return {"action": "stage_skip", "reason": "raw_exists", "path": str(dst)}
    if not src.exists():
        return {"action": "stage_skip", "reason": "inbox_missing", "path": str(src)}
    shutil.move(str(src), str(dst))
    return {"action": "staged", "path": str(dst)}


def finalize_row(row: dict) -> dict:
    raw_path = Path(row["final_raw_path"])
    label_path = Path(row["expected_label_csv_path"])
    queue_landmark = Path(row["landmark_queue_csv_path"])
    final_landmark = Path(row["final_landmark_csv_path"])

    if not raw_path.exists():
        return {"action": "finalize_skip", "reason": "raw_missing", "path": str(raw_path)}
    if not label_path.exists():
        return {"action": "finalize_skip", "reason": "label_missing", "path": str(label_path)}
    if final_landmark.exists():
        return {"action": "finalize_skip", "reason": "landmark_already_committed", "path": str(final_landmark)}
    if not queue_landmark.exists():
        return {"action": "finalize_skip", "reason": "landmark_queue_missing", "path": str(queue_landmark)}

    final_landmark.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(queue_landmark), str(final_landmark))
    return {"action": "finalized", "path": str(final_landmark)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage/finalize items from labeling queue.")
    parser.add_argument("--seed-profile", type=Path, default=SEED_PROFILE)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--stage-all", action="store_true")
    parser.add_argument("--finalize-all-ready", action="store_true")
    parser.add_argument("--name", type=str, default=None, help="suggested final raw filename stem or full filename")
    args = parser.parse_args()

    ensure_seed_profile_current(args.seed_profile)
    rows = load_queue(args.queue)

    if args.name:
        normalized = args.name[:-4] if args.name.endswith(".mp4") else args.name
        rows = [row for row in rows if Path(row["final_raw_path"]).stem == normalized]

    actions: list[dict] = []
    states = Counter(queue_state(row) for row in rows)

    if args.stage_all:
        for row in rows:
            if queue_state(row) == "ready_to_stage":
                actions.append(stage_row(row))

    if args.finalize_all_ready:
        for row in rows:
            if queue_state(row) == "ready_to_finalize":
                actions.append(finalize_row(row))

    if not args.stage_all and not args.finalize_all_ready:
        actions = [
            {
                "item": Path(row["final_raw_path"]).name,
                "state": queue_state(row),
                "raw_path": row["final_raw_path"],
                "label_path": row["expected_label_csv_path"],
                "queue_landmark_path": row["landmark_queue_csv_path"],
                "final_landmark_path": row["final_landmark_csv_path"],
            }
            for row in rows
        ]

    print(
        json.dumps(
            {
                "queue_items": len(rows),
                "states": dict(states),
                "actions": actions,
                "raw_data_dir": str(RAW_DATA_DIR),
                "labeled_data_dir": str(LABELED_DATA_DIR),
                "landmark_data_dir": str(LANDMARK_DATA_DIR),
                "landmark_queue_dir": str(LANDMARK_QUEUE_DIR),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
