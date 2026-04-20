# discover.py - Step 0 검증 후 저위험 수집 후보만 candidate manifest로 생성
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import requests
import yaml

from common import SEED_PROFILE, ensure_seed_profile_current, write_jsonl


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_CATALOG = ROOT / "source_catalog.yaml"
DEFAULT_CANDIDATES = ROOT / "manifests" / "candidate_manifest.jsonl"
DEFAULT_MANUAL = ROOT / "manifests" / "manual_backlog.csv"
DEFAULT_MAX_FILE_GB = 3.0


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate_catalog_schema(catalog: dict[str, Any]) -> None:
    sources = catalog.get("sources")
    if not isinstance(sources, dict):
        raise RuntimeError("source_catalog.yaml 형식 오류: top-level `sources` dict가 필요합니다.")
    required = {"file_direct", "bulk_direct", "manual_pages", "repack_sources"}
    missing = sorted(required - set(sources))
    if missing:
        raise RuntimeError(
            "source_catalog.yaml 형식 오류: "
            f"v2 필수 키가 없습니다: {', '.join(missing)}"
        )


def probe_http_size(url: str) -> int | None:
    try:
        response = requests.head(url, allow_redirects=True, timeout=20)
        if response.ok:
            header = response.headers.get("Content-Length")
            if header and header.isdigit():
                return int(header)
    except requests.RequestException:
        return None
    return None


def gib_to_bytes(value: float) -> int:
    return int(value * 1024 ** 3)


def build_record(
    *,
    candidate_id: str,
    source_tier: str,
    source_name: str,
    source_kind: str,
    url: str,
    title: str,
    status: str,
    license_note: str,
    estimated_semantic: str,
    score: float,
    notes: str = "",
    needs_manual_review: bool = False,
    manual_reason: str | None = None,
    filesize_bytes: int | None = None,
    size_estimate_bytes: int | None = None,
    extra: dict[str, Any] | None = None,
    eligible_for_auto_download: bool = True,
    rejection_reason: str | None = None,
) -> dict[str, Any]:
    payload = {
        "candidate_id": candidate_id,
        "source_tier": source_tier,
        "source_name": source_name,
        "source_kind": source_kind,
        "original_url": url,
        "title": title,
        "status": status,
        "license_note": license_note,
        "estimated_semantic": estimated_semantic,
        "score": score,
        "notes": notes,
        "needs_manual_review": needs_manual_review,
        "manual_reason": manual_reason,
        "filesize_bytes": filesize_bytes,
        "size_estimate_bytes": size_estimate_bytes,
        "eligible_for_auto_download": eligible_for_auto_download,
        "rejection_reason": rejection_reason,
    }
    if extra:
        payload.update(extra)
    return payload


def manual_row(source: dict[str, Any]) -> dict[str, str]:
    return {
        "source_name": source["name"],
        "source_tier": source.get("source_tier", "unknown"),
        "url": source["url"],
        "estimated_semantic": source.get("estimated_semantic", ""),
        "license_note": source.get("license_note", ""),
        "reason": source.get("manual_reason", "manual verification required"),
        "notes": source.get("notes", ""),
        "filesize_bytes": str(source.get("filesize_bytes", "")),
    }


def push_manual_backlog(
    manual_backlog: list[dict[str, str]],
    *,
    source_tier: str,
    source_name: str,
    url: str,
    estimated_semantic: str,
    license_note: str,
    reason: str,
    notes: str = "",
    filesize_bytes: int | None = None,
) -> None:
    manual_backlog.append(
        {
            "source_name": source_name,
            "source_tier": source_tier,
            "url": url,
            "estimated_semantic": estimated_semantic,
            "license_note": license_note,
            "reason": reason,
            "notes": notes,
            "filesize_bytes": str(filesize_bytes or ""),
        }
    )


def candidate_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    size = row.get("filesize_bytes")
    if size is None:
        size = row.get("size_estimate_bytes")
    if size is None:
        size = 10 ** 18
    return (int(size), -float(row.get("score", 0.0)), row["candidate_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build low-risk candidates from local source catalog.")
    parser.add_argument("--seed-profile", type=Path, default=SEED_PROFILE)
    parser.add_argument("--source-catalog", type=Path, default=DEFAULT_SOURCE_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--manual-output", type=Path, default=DEFAULT_MANUAL)
    args = parser.parse_args()

    seed = ensure_seed_profile_current(args.seed_profile)
    catalog = load_yaml(args.source_catalog)
    validate_catalog_schema(catalog)
    video_cap = int(catalog.get("global", {}).get("hard_cap_videos", 10))
    max_file_gb = float(catalog.get("global", {}).get("max_file_gb", DEFAULT_MAX_FILE_GB))
    max_file_bytes = gib_to_bytes(max_file_gb)

    records: list[dict[str, Any]] = []
    manual_backlog: list[dict[str, str]] = []

    for source in catalog["sources"].get("file_direct", []):
        size_bytes = source.get("filesize_bytes") or source.get("size_estimate_bytes")
        if size_bytes is None and source["kind"] == "http_file":
            size_bytes = probe_http_size(source["url"])

        rejection_reason: str | None = None
        if not source.get("auto_download", True):
            rejection_reason = source.get("manual_reason", "manual_only_license")
        elif size_bytes is None:
            rejection_reason = "unknown_size"
        elif int(size_bytes) >= max_file_bytes:
            rejection_reason = "over_3gb"

        record = build_record(
            candidate_id=f"file_direct__{source['name']}",
            source_tier="file_direct",
            source_name=source["name"],
            source_kind=source["kind"],
            url=source["url"],
            title=source["name"],
            status="candidate" if rejection_reason is None else "rejected_auto",
            license_note=source.get("license_note", ""),
            estimated_semantic=source.get("estimated_semantic", ""),
            score=80.0,
            notes=source.get("notes", ""),
            needs_manual_review=bool(source.get("needs_manual_review", False)),
            filesize_bytes=int(size_bytes) if size_bytes is not None else None,
            size_estimate_bytes=source.get("size_estimate_bytes"),
            eligible_for_auto_download=rejection_reason is None,
            rejection_reason=rejection_reason,
            extra={
                "archive_filename": source.get("archive_filename"),
                "extract_mode": source.get("extract_mode"),
                "sample_limit": source.get("sample_limit"),
            },
        )
        records.append(record)
        if rejection_reason is not None:
            push_manual_backlog(
                manual_backlog,
                source_tier="file_direct",
                source_name=source["name"],
                url=source["url"],
                estimated_semantic=source.get("estimated_semantic", ""),
                license_note=source.get("license_note", ""),
                reason=rejection_reason,
                notes=source.get("notes", ""),
                filesize_bytes=int(size_bytes) if size_bytes is not None else None,
            )

    for source in catalog["sources"].get("bulk_direct", []):
        source["source_tier"] = "bulk_direct"
        manual_backlog.append(manual_row(source))

    for source in catalog["sources"].get("manual_pages", []):
        source["source_tier"] = "manual_pages"
        manual_backlog.append(manual_row(source))

    include_queries = [term.lower() for term in seed.get("include_queries", [])]
    exclude_terms = [term.lower() for term in seed.get("exclude_terms", [])]

    for source in catalog["sources"].get("repack_sources", []):
        blob = f"{source['name']} {source.get('estimated_semantic', '')} {source.get('notes', '')}".lower()
        score = 40.0
        score += sum(8.0 for term in include_queries if term in blob)
        score -= sum(20.0 for term in exclude_terms if term in blob)
        filesize_bytes = probe_http_size(source["url"])
        rejection_reason: str | None = None
        if not source.get("auto_download", True):
            rejection_reason = source.get("manual_reason", "manual_only_license")
        elif filesize_bytes is None:
            rejection_reason = "unknown_size"
        elif int(filesize_bytes) >= max_file_bytes:
            rejection_reason = "over_3gb"

        record = build_record(
            candidate_id=f"repack_sources__{source['name']}",
            source_tier="repack_sources",
            source_name=source["name"],
            source_kind=source["kind"],
            url=source["url"],
            title=source["name"],
            status="candidate" if rejection_reason is None else "rejected_auto",
            license_note=source.get("license_note", ""),
            estimated_semantic=source.get("estimated_semantic", ""),
            score=round(score, 2),
            notes=source.get("notes", ""),
            needs_manual_review=bool(source.get("needs_manual_review", False)),
            filesize_bytes=filesize_bytes,
            eligible_for_auto_download=rejection_reason is None,
            rejection_reason=rejection_reason,
            extra={
                "repack_mode": source.get("repack_mode"),
                "fps": source.get("fps", seed.get("fps_estimate", 30)),
                "clip_duration_sec": source.get("clip_duration_sec", 1.0),
                "sample_limit": source.get("sample_limit", 2),
            },
        )
        records.append(record)
        if rejection_reason is not None:
            push_manual_backlog(
                manual_backlog,
                source_tier="repack_sources",
                source_name=source["name"],
                url=source["url"],
                estimated_semantic=source.get("estimated_semantic", ""),
                license_note=source.get("license_note", ""),
                reason=rejection_reason,
                notes=source.get("notes", ""),
                filesize_bytes=filesize_bytes,
            )

    eligible = sorted(
        [row for row in records if row.get("eligible_for_auto_download")],
        key=candidate_sort_key,
    )
    selected = eligible[:video_cap]
    overflow = eligible[video_cap:]
    for row in overflow:
        row["status"] = "rejected_auto"
        row["eligible_for_auto_download"] = False
        row["rejection_reason"] = "over_video_cap"
        push_manual_backlog(
            manual_backlog,
            source_tier=row["source_tier"],
            source_name=row["source_name"],
            url=row["original_url"],
            estimated_semantic=row.get("estimated_semantic", ""),
            license_note=row.get("license_note", ""),
            reason="over_video_cap",
            notes=row.get("notes", ""),
            filesize_bytes=row.get("filesize_bytes"),
        )

    rejected = sorted(
        [row for row in records if not row.get("eligible_for_auto_download")],
        key=lambda row: (row.get("rejection_reason") or "", row["candidate_id"]),
    )
    write_jsonl(args.output, selected + rejected)

    args.manual_output.parent.mkdir(parents=True, exist_ok=True)
    with args.manual_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_name",
                "source_tier",
                "url",
                "estimated_semantic",
                "license_note",
                "reason",
                "notes",
                "filesize_bytes",
            ],
        )
        writer.writeheader()
        for row in manual_backlog:
            writer.writerow(row)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "manual_output": str(args.manual_output),
                "candidate_count": len(selected),
                "rejected_auto_count": len(rejected),
                "manual_backlog_count": len(manual_backlog),
                "video_cap": video_cap,
                "max_file_gb": max_file_gb,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
