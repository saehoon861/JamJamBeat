# download.py - Step 0 검증 후 file_direct/repack_sources만 소량 다운로드
from __future__ import annotations

import argparse
import json
import tempfile
import time
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gdown
import requests
import yaml

import repack
from common import (
    ROOT,
    SEED_PROFILE,
    VIDEO_SUFFIXES,
    append_jsonl,
    ensure_seed_profile_current,
    load_jsonl,
)


DEFAULT_CANDIDATES = ROOT / "manifests" / "candidate_manifest.jsonl"
DEFAULT_CATALOG = ROOT / "source_catalog.yaml"
DEFAULT_DOWNLOADED = ROOT / "manifests" / "downloaded_manifest.jsonl"
DEFAULT_LOG = ROOT / "logs" / "download_log.jsonl"
RUN_LOG_DIR = ROOT / "logs" / "runs"
INBOX_ROOT = ROOT / "inbox_videos"
DEFAULT_MAX_FILE_GB = 3.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return cleaned.strip("._") or "item"


def gib_to_bytes(value: float) -> int:
    return int(value * 1024 ** 3)


def compute_existing_downloads(path: Path) -> tuple[int, list[dict[str, Any]]]:
    rows = load_jsonl(path)
    total = 0
    valid_rows: list[dict[str, Any]] = []
    for row in rows:
        file_path = Path(row["path"])
        if file_path.exists():
            size = file_path.stat().st_size
            row["filesize_bytes"] = size
            total += size
            valid_rows.append(row)
    return total, valid_rows


def log_event(log_path: Path, event: dict[str, Any]) -> None:
    append_jsonl(log_path, [event])


def write_log_pointer(pointer_path: Path, payload: dict[str, Any]) -> None:
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def head_size(url: str) -> int | None:
    try:
        response = requests.head(url, allow_redirects=True, timeout=20)
        if response.ok:
            header = response.headers.get("Content-Length")
            if header and header.isdigit():
                return int(header)
    except requests.RequestException:
        return None
    return None


def stream_http_download(url: str, output_path: Path, hard_limit_remaining: int) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        bytes_written = 0
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                bytes_written += len(chunk)
                if bytes_written > hard_limit_remaining:
                    handle.close()
                    output_path.unlink(missing_ok=True)
                    raise RuntimeError("Download would exceed remaining hard cap")
                handle.write(chunk)
    return output_path.stat().st_size


def download_gdrive_file(url: str, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = gdown.download(url=url, output=str(output_path), quiet=True, fuzzy=True, use_cookies=False)
    if not downloaded:
        raise RuntimeError("No file downloaded from Google Drive file source")
    return Path(downloaded).stat().st_size


def extract_tgz_video_sample(
    *,
    archive_path: Path,
    output_dir: Path,
    source_name: str,
    sample_limit: int,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_rows: list[dict[str, Any]] = []
    with tarfile.open(archive_path, mode="r:gz") as handle:
        video_members = [
            member
            for member in handle.getmembers()
            if member.isfile() and Path(member.name).suffix.lower() in VIDEO_SUFFIXES
        ]
        selected = sorted(video_members, key=lambda member: member.name)[:sample_limit]
        for index, member in enumerate(selected, start=1):
            suffix = Path(member.name).suffix.lower() or ".bin"
            basename = Path(member.name).stem
            output_path = output_dir / f"{safe_name(source_name)}__{index:02d}__{safe_name(basename)}{suffix}"
            with handle.extractfile(member) as src:
                if src is None:
                    continue
                with output_path.open("wb") as dst:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
            extracted_rows.append(
                {
                    "path": str(output_path),
                    "filesize_bytes": output_path.stat().st_size,
                    "derived_from": str(archive_path),
                    "source_member": member.name,
                }
            )
    return extracted_rows


def tier_order(candidates: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    ordered_tiers = ["repack_sources", "file_direct"]
    grouped = {tier: [] for tier in ordered_tiers}
    for candidate in candidates:
        grouped.setdefault(candidate["source_tier"], []).append(candidate)
    return [
        (
            tier,
            sorted(
                grouped.get(tier, []),
                key=lambda row: (
                    int(row.get("filesize_bytes") or row.get("size_estimate_bytes") or 10 ** 18),
                    -float(row.get("score", 0.0)),
                    row["candidate_id"],
                ),
            ),
        )
        for tier in ordered_tiers
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download low-risk candidates under hard caps.")
    parser.add_argument("--seed-profile", type=Path, default=SEED_PROFILE)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--downloaded-manifest", type=Path, default=DEFAULT_DOWNLOADED)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--max-gb", type=float, default=49.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tiers", nargs="*", default=["repack_sources", "file_direct"])
    parser.add_argument("--sleep-between-downloads", type=float, default=10.0)
    args = parser.parse_args()

    ensure_seed_profile_current(args.seed_profile)
    catalog = yaml.safe_load(args.catalog.read_text(encoding="utf-8"))
    video_cap = int(catalog.get("global", {}).get("hard_cap_videos", 10))
    max_file_gb = float(catalog.get("global", {}).get("max_file_gb", DEFAULT_MAX_FILE_GB))
    max_file_bytes = gib_to_bytes(max_file_gb)
    effective_limit = min(args.limit, video_cap) if args.limit is not None else video_cap
    candidates = [
        row
        for row in load_jsonl(args.candidates)
        if row.get("status") == "candidate"
        and row.get("eligible_for_auto_download", True)
        and row.get("source_tier") in set(args.tiers)
    ]
    hard_cap_bytes = int(args.max_gb * 1024 * 1024 * 1024)
    existing_total, existing_rows = compute_existing_downloads(args.downloaded_manifest)
    existing_video_count = len(existing_rows)
    existing_candidate_ids = {row.get("candidate_id") for row in existing_rows if row.get("candidate_id")}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_log = RUN_LOG_DIR / f"download_{run_id}.jsonl"
    write_log_pointer(
        args.log,
        {
            "event": "latest_run",
            "run_id": run_id,
            "run_log": str(run_log),
            "dry_run": args.dry_run,
            "tiers": args.tiers,
            "max_gb": args.max_gb,
            "max_file_gb": max_file_gb,
            "started_at": utc_now(),
        },
    )
    log_event(
        run_log,
        {
            "event": "run_started",
            "run_id": run_id,
            "dry_run": args.dry_run,
            "tiers": args.tiers,
            "max_gb": args.max_gb,
            "max_file_gb": max_file_gb,
            "at": utc_now(),
        },
    )

    downloaded_rows: list[dict[str, Any]] = []
    processed = 0

    if not candidates:
        log_event(
            run_log,
            {
                "event": "no_eligible_candidates",
                "tiers": args.tiers,
                "at": utc_now(),
            },
        )

    for tier, grouped_candidates in tier_order(candidates):
        if tier not in args.tiers:
            continue
        for candidate in grouped_candidates:
            if processed >= effective_limit or existing_video_count >= video_cap:
                break

            if candidate["candidate_id"] in existing_candidate_ids:
                log_event(
                    run_log,
                    {
                        "event": "skip_already_downloaded",
                        "candidate_id": candidate["candidate_id"],
                        "source_tier": candidate["source_tier"],
                        "at": utc_now(),
                    },
                )
                continue

            processed += 1
            remaining = hard_cap_bytes - existing_total
            if remaining <= 0:
                log_event(
                    run_log,
                    {
                        "event": "hard_cap_reached",
                        "candidate_id": candidate["candidate_id"],
                        "at": utc_now(),
                    },
                )
                break

            expected_size = candidate.get("filesize_bytes") or candidate.get("size_estimate_bytes")
            if not expected_size and candidate["source_kind"] == "http_file":
                expected_size = head_size(candidate["original_url"])

            if not expected_size:
                log_event(
                    run_log,
                    {
                        "event": "skip_unknown_size",
                        "candidate_id": candidate["candidate_id"],
                        "source_tier": candidate["source_tier"],
                        "at": utc_now(),
                    },
                )
                continue

            if int(expected_size) >= max_file_bytes:
                log_event(
                    run_log,
                    {
                        "event": "skip_over_file_cap",
                        "candidate_id": candidate["candidate_id"],
                        "source_tier": candidate["source_tier"],
                        "expected_size": expected_size,
                        "max_file_bytes": max_file_bytes,
                        "at": utc_now(),
                    },
                )
                continue

            if expected_size and expected_size > remaining:
                log_event(
                    run_log,
                    {
                        "event": "skip_over_cap",
                        "candidate_id": candidate["candidate_id"],
                        "expected_size": expected_size,
                        "remaining": remaining,
                        "at": utc_now(),
                    },
                )
                continue

            if args.dry_run:
                log_event(
                    run_log,
                    {
                        "event": "dry_run_candidate",
                        "candidate_id": candidate["candidate_id"],
                        "source_tier": candidate["source_tier"],
                        "expected_size": expected_size,
                        "at": utc_now(),
                    },
                )
                continue

            try:
                if candidate["source_tier"] == "file_direct":
                    output_dir = INBOX_ROOT / "official" / safe_name(candidate["source_name"])
                    output_dir.mkdir(parents=True, exist_ok=True)
                    if candidate["source_kind"] == "gdrive_file":
                        extract_mode = candidate.get("extract_mode")
                        if extract_mode == "tgz_video_sample":
                            remaining_slots = video_cap - existing_video_count
                            if remaining_slots <= 0:
                                break
                            with tempfile.TemporaryDirectory(prefix="web_collect_gdrive_") as temp_dir_str:
                                temp_dir = Path(temp_dir_str)
                                archive_filename = candidate.get("archive_filename") or f"{safe_name(candidate['source_name'])}.tgz"
                                archive_path = temp_dir / archive_filename
                                download_gdrive_file(candidate["original_url"], archive_path)
                                extracted = extract_tgz_video_sample(
                                    archive_path=archive_path,
                                    output_dir=output_dir,
                                    source_name=safe_name(candidate["source_name"]),
                                    sample_limit=min(int(candidate.get("sample_limit") or 2), remaining_slots),
                                )
                            batch_total = sum(int(row["filesize_bytes"]) for row in extracted)
                            downloaded_rows.extend(
                                [
                                    {
                                        "candidate_id": candidate["candidate_id"],
                                        "source_name": candidate["source_name"],
                                        "source_tier": candidate["source_tier"],
                                        "path": row["path"],
                                        "filesize_bytes": row["filesize_bytes"],
                                        "original_url": candidate["original_url"],
                                        "title": Path(row["path"]).name,
                                        "license_note": candidate.get("license_note"),
                                        "downloaded_at": utc_now(),
                                        "needs_manual_review": candidate.get("needs_manual_review", True),
                                        "notes": candidate.get("notes", ""),
                                        "derived_from": row["derived_from"],
                                        "source_member": row["source_member"],
                                    }
                                    for row in extracted
                                ]
                            )
                            existing_total += batch_total
                            existing_video_count += len(extracted)
                            existing_candidate_ids.add(candidate["candidate_id"])
                            output_path = output_dir
                            actual_size = batch_total
                        else:
                            archive_name = candidate.get("archive_filename") or f"{safe_name(candidate['source_name'])}.bin"
                            output_path = output_dir / archive_name
                            actual_size = download_gdrive_file(candidate["original_url"], output_path)
                            downloaded_rows.append(
                                {
                                    "candidate_id": candidate["candidate_id"],
                                    "source_name": candidate["source_name"],
                                    "source_tier": candidate["source_tier"],
                                    "path": str(output_path),
                                    "filesize_bytes": actual_size,
                                    "original_url": candidate["original_url"],
                                    "title": output_path.name,
                                    "license_note": candidate.get("license_note"),
                                    "downloaded_at": utc_now(),
                                    "needs_manual_review": candidate.get("needs_manual_review", True),
                                    "notes": candidate.get("notes", ""),
                                }
                            )
                            existing_total += actual_size
                            existing_video_count += 1
                            existing_candidate_ids.add(candidate["candidate_id"])
                    elif candidate["source_kind"] == "http_file":
                        ext = Path(candidate["original_url"]).suffix or ".bin"
                        output_path = output_dir / f"{safe_name(candidate['source_name'])}{ext}"
                        actual_size = stream_http_download(candidate["original_url"], output_path, remaining)
                        downloaded_rows.append(
                            {
                                "candidate_id": candidate["candidate_id"],
                                "source_name": candidate["source_name"],
                                "source_tier": candidate["source_tier"],
                                "path": str(output_path),
                                "filesize_bytes": actual_size,
                                "original_url": candidate["original_url"],
                                "title": output_path.name,
                                "license_note": candidate.get("license_note"),
                                "downloaded_at": utc_now(),
                                "needs_manual_review": candidate.get("needs_manual_review", True),
                                "notes": candidate.get("notes", ""),
                            }
                        )
                        existing_total += actual_size
                        existing_video_count += 1
                        existing_candidate_ids.add(candidate["candidate_id"])
                    else:
                        raise RuntimeError(f"Unsupported file_direct kind: {candidate['source_kind']}")

                elif candidate["source_tier"] == "repack_sources":
                    remaining_slots = video_cap - existing_video_count
                    if remaining_slots <= 0:
                        break
                    with tempfile.TemporaryDirectory(prefix="web_collect_") as temp_dir_str:
                        temp_dir = Path(temp_dir_str)
                        archive_ext = Path(candidate["original_url"]).suffix or ".zip"
                        archive_path = temp_dir / f"{safe_name(candidate['source_name'])}{archive_ext}"
                        stream_http_download(candidate["original_url"], archive_path, remaining)
                        repacked = repack.repack_archive_to_mp4s(
                            source_path=archive_path,
                            output_dir=INBOX_ROOT / "repacked",
                            source_name=safe_name(candidate["source_name"]),
                            fps=int(candidate.get("fps", 30)),
                            clip_duration_sec=float(candidate.get("clip_duration_sec", 1.0)),
                            sample_limit=min(int(candidate.get("sample_limit", 2)), remaining_slots),
                            remaining_bytes=hard_cap_bytes - existing_total,
                        )
                        batch_total = sum(int(row["filesize_bytes"]) for row in repacked)
                        downloaded_rows.extend(
                            [
                                {
                                    "candidate_id": candidate["candidate_id"],
                                    "source_name": candidate["source_name"],
                                    "source_tier": candidate["source_tier"],
                                    "path": row["path"],
                                    "filesize_bytes": row["filesize_bytes"],
                                    "original_url": candidate["original_url"],
                                    "title": Path(row["path"]).name,
                                    "license_note": candidate.get("license_note"),
                                    "downloaded_at": utc_now(),
                                    "needs_manual_review": True,
                                    "notes": candidate.get("notes", ""),
                                    "derived_from": row["derived_from"],
                                    "source_image": row["source_image"],
                                }
                                for row in repacked
                            ]
                        )
                        existing_total += batch_total
                        existing_video_count += len(repacked)
                        existing_candidate_ids.add(candidate["candidate_id"])
                else:
                    raise RuntimeError(f"Unsupported source tier: {candidate['source_tier']}")

                log_event(
                    run_log,
                    {
                        "event": "downloaded",
                        "candidate_id": candidate["candidate_id"],
                        "source_tier": candidate["source_tier"],
                        "total_downloaded_bytes": existing_total,
                        "total_videos": existing_video_count,
                        "at": utc_now(),
                    },
                )
            except Exception as exc:
                log_event(
                    run_log,
                    {
                        "event": "download_failed",
                        "candidate_id": candidate["candidate_id"],
                        "source_tier": candidate["source_tier"],
                        "error": str(exc),
                        "at": utc_now(),
                    },
                )

            if args.sleep_between_downloads > 0:
                time.sleep(args.sleep_between_downloads)

        if processed >= effective_limit or existing_video_count >= video_cap:
            break

    append_jsonl(args.downloaded_manifest, downloaded_rows)
    print(
        json.dumps(
            {
                "downloaded_records": len(downloaded_rows),
                "dry_run": args.dry_run,
                "total_bytes_now": existing_total,
                "total_videos_now": existing_video_count,
                "video_cap": video_cap,
                "effective_limit": effective_limit,
                "max_file_gb": max_file_gb,
                "run_log": str(run_log),
                "manifest": str(args.downloaded_manifest),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
