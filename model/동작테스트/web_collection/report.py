# report.py - Step 0 검증 후 수집/품질/queue 상태를 한 번에 요약
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import ROOT, SEED_PROFILE, VIDEO_SUFFIXES, ensure_seed_profile_current, load_jsonl
from commit_labeling_queue import queue_state


CANDIDATES = ROOT / "manifests" / "candidate_manifest.jsonl"
DOWNLOADED = ROOT / "manifests" / "downloaded_manifest.jsonl"
MANUAL = ROOT / "manifests" / "manual_backlog.csv"
INTAKE = ROOT / "manifests" / "intake_manifest.jsonl"
QUALITY = ROOT / "manifests" / "quality_manifest.jsonl"
LABELING_QUEUE = ROOT / "manifests" / "labeling_queue_manifest.jsonl"
SUMMARY = ROOT / "summary.md"
LANDMARK_QUEUE = ROOT / "landmark_queue"
DOWNLOAD_POINTER = ROOT / "logs" / "download_log.jsonl"
INBOX_ROOT = ROOT / "inbox_videos"


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt_bytes(value: int) -> str:
    gb = value / (1024 ** 3)
    return f"{gb:.2f} GB"


def load_download_pointer(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        first_line = text.splitlines()[0].strip()
        return json.loads(first_line) if first_line else {}


def main() -> None:
    ensure_seed_profile_current(SEED_PROFILE)

    candidate_rows = load_jsonl(CANDIDATES)
    downloaded_rows = load_jsonl(DOWNLOADED)
    intake_rows = load_jsonl(INTAKE)
    quality_rows = load_jsonl(QUALITY)
    queue_rows = load_jsonl(LABELING_QUEUE)
    manual_rows = load_csv(MANUAL)
    queue_state_counts = Counter(queue_state(row) for row in queue_rows)
    latest_download_run = load_download_pointer(DOWNLOAD_POINTER)

    total_bytes = sum(int(row.get("filesize_bytes", 0)) for row in downloaded_rows)
    candidate_counts = Counter(row.get("status", "unknown") for row in candidate_rows)
    eligible_candidate_count = sum(1 for row in candidate_rows if row.get("eligible_for_auto_download"))
    rejected_reasons = Counter(
        row.get("rejection_reason", "unknown")
        for row in candidate_rows
        if row.get("status") == "rejected_auto"
    )
    downloaded_by_tier = Counter(row.get("source_tier", "unknown") for row in downloaded_rows)
    quality_counts = Counter(row.get("quality_status", "unknown") for row in quality_rows)
    by_source_bytes: dict[str, int] = defaultdict(int)
    for row in downloaded_rows:
        by_source_bytes[row.get("source_name", "unknown")] += int(row.get("filesize_bytes", 0))

    landmark_queue_count = len(list(LANDMARK_QUEUE.glob("*.csv"))) if LANDMARK_QUEUE.exists() else 0
    inbox_file_count = (
        len([path for path in INBOX_ROOT.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES])
        if INBOX_ROOT.exists()
        else 0
    )

    lines = [
        "# web_collection summary",
        "",
        f"- candidate rows: {len(candidate_rows)}",
        f"- candidate status counts: {dict(candidate_counts)}",
        f"- auto-eligible candidate rows: {eligible_candidate_count}",
        f"- rejected over 3GB: {rejected_reasons.get('over_3gb', 0)}",
        f"- rejected unknown size: {rejected_reasons.get('unknown_size', 0)}",
        f"- downloaded files: {len(downloaded_rows)}",
        f"- total downloaded size: {fmt_bytes(total_bytes)}",
        f"- current inbox video files: {inbox_file_count}",
        f"- manual backlog items: {len(manual_rows)}",
        f"- intake rows: {len(intake_rows)}",
        f"- quality counts: {dict(quality_counts)}",
        f"- landmark queue csv count: {landmark_queue_count}",
        f"- labeling queue rows: {len(queue_rows)}",
        f"- labeling queue states: {dict(queue_state_counts)}",
        "",
        "## Latest Download Run",
        "",
        f"- run_id: {latest_download_run.get('run_id', 'none')}",
        f"- run_log: {latest_download_run.get('run_log', 'none')}",
        f"- dry_run: {latest_download_run.get('dry_run', 'n/a')}",
        "",
        "## Downloaded By Tier",
        "",
    ]

    for tier in ["file_direct", "repack_sources"]:
        lines.append(f"- `{tier}`: {downloaded_by_tier.get(tier, 0)} files")

    lines.extend(["", "## Downloaded By Source", ""])
    if by_source_bytes:
        for source_name, size in sorted(by_source_bytes.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"- `{source_name}`: {fmt_bytes(size)}")
    else:
        lines.append("- none")

    lines.extend(["", "## Manual Backlog", ""])
    if manual_rows:
        for row in manual_rows[:20]:
            lines.append(f"- `{row['source_name']}`: {row['reason']} ({row['url']})")
    else:
        lines.append("- none")

    lines.extend(["", "## Output Files", ""])
    lines.append(f"- [candidate_manifest.jsonl]({CANDIDATES})")
    lines.append(f"- [downloaded_manifest.jsonl]({DOWNLOADED})")
    lines.append(f"- [manual_backlog.csv]({MANUAL})")
    lines.append(f"- [intake_manifest.jsonl]({INTAKE})")
    lines.append(f"- [quality_manifest.jsonl]({QUALITY})")
    lines.append(f"- [labeling_queue_manifest.jsonl]({LABELING_QUEUE})")

    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote summary to {SUMMARY}")


if __name__ == "__main__":
    main()
