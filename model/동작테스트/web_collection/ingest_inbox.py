# ingest_inbox.py - inbox_videos를 스캔해 비디오 메타 intake manifest를 생성
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from common import ROOT, SEED_PROFILE, VIDEO_SUFFIXES, ensure_seed_profile_current, load_jsonl, write_jsonl


DEFAULT_DOWNLOADED = ROOT / "manifests" / "downloaded_manifest.jsonl"
DEFAULT_OUTPUT = ROOT / "manifests" / "intake_manifest.jsonl"
INBOX_ROOT = ROOT / "inbox_videos"


def collect_video_metadata(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    readable = cap.isOpened()
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) if readable else 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if readable else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) if readable else 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) if readable else 0
    duration_sec = float(frame_count / fps) if readable and fps > 0 else 0.0
    cap.release()
    return {
        "readable": readable,
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": round(duration_sec, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan inbox_videos and build intake manifest.")
    parser.add_argument("--seed-profile", type=Path, default=SEED_PROFILE)
    parser.add_argument("--downloaded-manifest", type=Path, default=DEFAULT_DOWNLOADED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    ensure_seed_profile_current(args.seed_profile)
    downloaded_rows = load_jsonl(args.downloaded_manifest)
    by_path = {Path(row["path"]).resolve(): row for row in downloaded_rows}

    rows: list[dict] = []
    for path in sorted(INBOX_ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        metadata = collect_video_metadata(path)
        source_row = by_path.get(path.resolve(), {})
        rows.append(
            {
                "path": str(path.resolve()),
                "stem": path.stem,
                "source_tier": source_row.get("source_tier", path.parent.name),
                "source_name": source_row.get("source_name", path.parent.name),
                "title": source_row.get("title", path.name),
                "original_url": source_row.get("original_url"),
                "license_note": source_row.get("license_note"),
                "filesize_bytes": path.stat().st_size,
                **metadata,
            }
        )

    write_jsonl(args.output, rows)
    print(json.dumps({"output": str(args.output), "count": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
