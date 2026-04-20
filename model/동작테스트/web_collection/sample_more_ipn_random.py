"""sample_more_ipn_random.py - IPN archive에서 아직 안 받은 다른 번호 비디오를 랜덤 추출."""

from __future__ import annotations

import argparse
import json
import random
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

from common import ROOT, VIDEO_SUFFIXES, append_jsonl
from download import download_gdrive_file, safe_name


CATALOG = ROOT / "source_catalog.yaml"
DOWNLOADED = ROOT / "manifests" / "downloaded_manifest.jsonl"
IPN_CACHE_DIR = ROOT / "IPN_Hand_probe"
OFFICIAL_INBOX = ROOT / "inbox_videos" / "official"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_member_id(member_name: str) -> str:
    stem = Path(member_name).stem
    return stem.replace("_#", "__")


def existing_ipn_ids() -> set[str]:
    ids: set[str] = set()
    for path in OFFICIAL_INBOX.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        if "ipn_hand_videos" not in path.name:
            continue
        parts = path.stem.split("__")
        if len(parts) >= 4:
            ids.add(parts[-2] + "__" + parts[-1])
    return ids


def load_ipn_sources() -> list[dict]:
    catalog = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    sources = catalog.get("sources", {}).get("file_direct", [])
    return [row for row in sources if str(row.get("name", "")).startswith("ipn_hand_videos")]


def ensure_archive_cached(source: dict) -> Path:
    archive_name = source["archive_filename"]
    archive_path = IPN_CACHE_DIR / archive_name
    if archive_path.exists():
        return archive_path
    IPN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    download_gdrive_file(source["url"], archive_path)
    return archive_path


def extract_random_members(
    *,
    source: dict,
    archive_path: Path,
    sample_count: int,
    rng: random.Random,
    existing_ids_set: set[str],
) -> list[dict]:
    output_dir = OFFICIAL_INBOX / source["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_rows: list[dict] = []
    with tarfile.open(archive_path, mode="r:gz") as handle:
        members = [
            member
            for member in handle.getmembers()
            if member.isfile() and Path(member.name).suffix.lower() in VIDEO_SUFFIXES
        ]
        available = [member for member in members if normalize_member_id(member.name) not in existing_ids_set]
        if not available:
            return []

        chosen = rng.sample(available, k=min(sample_count, len(available)))
        existing_count = len(list(output_dir.glob("*")))
        for offset, member in enumerate(sorted(chosen, key=lambda item: item.name), start=1):
            normalized_id = normalize_member_id(member.name)
            suffix = Path(member.name).suffix.lower() or ".bin"
            basename = Path(member.name).stem
            output_path = output_dir / (
                f"{safe_name(source['name'])}__R{existing_count + offset:02d}__{safe_name(basename)}{suffix}"
            )
            with handle.extractfile(member) as src:
                if src is None:
                    continue
                with output_path.open("wb") as dst:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)

            existing_ids_set.add(normalized_id)
            extracted_rows.append(
                {
                    "candidate_id": f"random_extract__{source['name']}__{normalized_id}",
                    "source_name": source["name"],
                    "source_tier": "file_direct",
                    "path": str(output_path),
                    "filesize_bytes": output_path.stat().st_size,
                    "original_url": source["url"],
                    "derived_from": str(archive_path),
                    "source_member": member.name,
                    "downloaded_at": utc_now(),
                    "notes": "Random extra IPN sample; chosen to avoid previously extracted video ids.",
                }
            )
    return extracted_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample more random IPN videos from cached/downloading archives.")
    parser.add_argument("--per-archive", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260322)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    sources = load_ipn_sources()
    existing_ids_set = existing_ipn_ids()
    all_rows: list[dict] = []

    for source in sources:
        archive_path = ensure_archive_cached(source)
        rows = extract_random_members(
            source=source,
            archive_path=archive_path,
            sample_count=args.per_archive,
            rng=rng,
            existing_ids_set=existing_ids_set,
        )
        all_rows.extend(rows)

    if all_rows:
        append_jsonl(DOWNLOADED, all_rows)
    print(json.dumps({"new_random_ipn_videos": len(all_rows), "per_archive": args.per_archive}, ensure_ascii=False))


if __name__ == "__main__":
    main()
