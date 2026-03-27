# profile_seed.py - 7_grab 시드 CSV를 읽어 검색/수집용 시드 프로파일을 생성
from __future__ import annotations

import argparse
import csv
from statistics import median
from pathlib import Path

from common import SEED_LABELS, SEED_PROFILE, SEED_VIDEO, build_file_stamp, write_json

DEFAULT_INCLUDE_QUERIES = [
    "hand grab gesture",
    "hand grasp gesture",
    "open hand to fist",
    "close hand gesture",
    "hand gripping motion",
    "grabbing object by hand",
    "hand clutch motion",
    "pick up object hand",
]

DEFAULT_EXCLUDE_TERMS = [
    "robot gripper",
    "traffic police",
    "sign language alphabet",
    "medical grasp test",
    "anime",
    "game",
    "cgi",
]


def parse_timestamp(value: str) -> float:
    value = value.strip()
    if not value:
        return 0.0
    parts = value.split(":")
    if len(parts) == 3:
        minutes, seconds, millis = parts
        return int(minutes) * 60.0 + int(seconds) + int(millis) / 1000.0
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60.0 + float(seconds)
    return float(value)

def load_rows(csv_path) -> list[dict]:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")
    return rows


def contiguous_spans(rows: list[dict], positive_class_id: int) -> tuple[list[dict], list[dict]]:
    spans: list[dict] = []
    current_label: int | None = None
    start_frame: int | None = None
    start_ts: float | None = None
    prev_frame: int | None = None
    prev_ts: float | None = None
    for row in rows:
        frame_idx = int(row["frame_idx"])
        timestamp_sec = parse_timestamp(row["timestamp"])
        gesture = int(row["gesture"])
        if (
            current_label is None
            or gesture != current_label
            or prev_frame is None
            or frame_idx != prev_frame + 1
        ):
            if current_label is not None:
                spans.append(
                    {
                        "gesture": current_label,
                        "start_frame": start_frame,
                        "end_frame": prev_frame,
                        "start_sec": start_ts,
                        "end_sec": prev_ts,
                        "duration_sec": max(0.0, float(prev_ts) - float(start_ts)),
                    }
                )
            current_label = gesture
            start_frame = frame_idx
            start_ts = timestamp_sec
        prev_frame = frame_idx
        prev_ts = timestamp_sec

    if current_label is not None:
        spans.append(
            {
                "gesture": current_label,
                "start_frame": start_frame,
                "end_frame": prev_frame,
                "start_sec": start_ts,
                "end_sec": prev_ts,
                "duration_sec": max(0.0, float(prev_ts) - float(start_ts)),
            }
        )

    positive_spans = [span for span in spans if span["gesture"] == positive_class_id]
    return spans, positive_spans


def estimate_fps(rows: list[dict]) -> int:
    seconds = [parse_timestamp(row["timestamp"]) for row in rows]
    deltas = [
        round(seconds[idx] - seconds[idx - 1], 6)
        for idx in range(1, len(seconds))
        if seconds[idx] > seconds[idx - 1]
    ]
    if not deltas:
        return 30
    sec_per_frame = median(deltas)
    if sec_per_frame <= 0:
        return 30
    return max(1, int(round(1.0 / sec_per_frame)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a seed profile from 7_grab labels.")
    parser.add_argument("--video", type=Path, default=SEED_VIDEO)
    parser.add_argument("--labels", type=Path, default=SEED_LABELS)
    parser.add_argument("--output", type=Path, default=SEED_PROFILE)
    args = parser.parse_args()

    rows = load_rows(args.labels)
    seed_stem = args.video.stem
    seed_prefix, seed_label = seed_stem.split("_", 1)
    seed_class_id = int(seed_prefix)
    spans, positive_spans = contiguous_spans(rows, seed_class_id)
    fps_estimate = estimate_fps(rows)
    positive_durations = [round(span["duration_sec"], 3) for span in positive_spans]

    pattern = {
        "pattern_name": "repeated_neutral_to_grab_bursts",
        "positive_class_id": seed_class_id,
        "positive_span_count": len(positive_spans),
        "interleaved_with_neutral": all(
            span["gesture"] in {0, seed_class_id} for span in spans
        ),
    }

    profile = {
        "seed_video": build_file_stamp(args.video),
        "seed_labels": build_file_stamp(args.labels),
        "seed_stem": seed_stem,
        "seed_label": seed_label,
        "seed_class_id": seed_class_id,
        "frame_count": len(rows),
        "fps_estimate": fps_estimate,
        "positive_frame_count": sum(1 for row in rows if int(row["gesture"]) == seed_class_id),
        "positive_span_count": len(positive_spans),
        "positive_span_duration_range_sec": {
            "min": min(positive_durations) if positive_durations else 0.0,
            "max": max(positive_durations) if positive_durations else 0.0,
        },
        "positive_spans": positive_spans,
        "neutral_to_grab_transition_pattern": pattern,
        "include_queries": DEFAULT_INCLUDE_QUERIES,
        "exclude_terms": DEFAULT_EXCLUDE_TERMS,
    }

    write_json(args.output, profile)
    print(f"Wrote seed profile to {args.output}")


if __name__ == "__main__":
    main()
