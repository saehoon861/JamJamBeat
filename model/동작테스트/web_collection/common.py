"""common.py - 동작테스트 web_collection 공용 경로/시드 검증/직렬화 도구."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
RAW_DATA_DIR = WORKSPACE_ROOT / "raw_data"
LABELED_DATA_DIR = WORKSPACE_ROOT / "labeled_data"
LANDMARK_DATA_DIR = WORKSPACE_ROOT / "landmark_data"
SEED_VIDEO = RAW_DATA_DIR / "7_grab.mp4"
SEED_LABELS = LABELED_DATA_DIR / "7_grab.csv"
SEED_PROFILE = ROOT / "manifests" / "seed_profile.json"
HAND_LANDMARKER_TASK = WORKSPACE_ROOT / "hand_landmarker.task"
LANDMARK_QUEUE_DIR = ROOT / "landmark_queue"


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_file_stamp(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "mtime_ns": stat.st_mtime_ns,
        "filesize_bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_seed_profile_current(profile_path: Path = SEED_PROFILE) -> dict[str, Any]:
    if not profile_path.exists():
        raise RuntimeError(
            "Step 0 missing: manifests/seed_profile.json 이 없습니다. "
            "먼저 `uv run python profile_seed.py` 를 실행하세요."
        )

    profile = load_json(profile_path)
    expected_video = profile.get("seed_video", {})
    expected_labels = profile.get("seed_labels", {})

    if Path(expected_video.get("path", "")).resolve() != SEED_VIDEO.resolve():
        raise RuntimeError(
            "Step 0 stale: seed video path mismatch 입니다. "
            "먼저 `uv run python profile_seed.py` 를 다시 실행하세요."
        )
    if Path(expected_labels.get("path", "")).resolve() != SEED_LABELS.resolve():
        raise RuntimeError(
            "Step 0 stale: seed label path mismatch 입니다. "
            "먼저 `uv run python profile_seed.py` 를 다시 실행하세요."
        )

    current_video = build_file_stamp(SEED_VIDEO)
    current_labels = build_file_stamp(SEED_LABELS)
    if current_video != expected_video or current_labels != expected_labels:
        raise RuntimeError(
            "Step 0 stale: seed input 파일이 변경됐습니다. "
            "먼저 `uv run python profile_seed.py` 를 다시 실행하세요."
        )
    return profile
