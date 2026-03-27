# repack.py - 이미지 zip 또는 디렉터리를 짧은 mp4 클립으로 변환
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sample_evenly(items: list[Path], limit: int) -> list[Path]:
    if limit <= 0 or len(items) <= limit:
        return items
    if limit == 1:
        return [items[0]]
    step = (len(items) - 1) / float(limit - 1)
    sampled: list[Path] = []
    seen: set[int] = set()
    for idx in range(limit):
        pos = int(round(idx * step))
        pos = min(pos, len(items) - 1)
        if pos in seen:
            continue
        seen.add(pos)
        sampled.append(items[pos])
    return sampled


def collect_images(source: Path, temp_dir: Path) -> list[Path]:
    if source.is_dir():
        return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)

    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            zf.extractall(temp_dir)
        return sorted(p for p in temp_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)

    raise ValueError(f"Unsupported repack input: {source}")


def write_still_clip(
    image_path: Path,
    output_path: Path,
    fps: int,
    clip_duration_sec: float,
) -> int:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Failed to read image: {image_path}")
    height, width = frame.shape[:2]
    frame_count = max(1, int(round(fps * clip_duration_sec)))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")
    try:
        for _ in range(frame_count):
            writer.write(frame)
    finally:
        writer.release()
    return output_path.stat().st_size


def repack_archive_to_mp4s(
    source_path: Path,
    output_dir: Path,
    source_name: str,
    fps: int,
    clip_duration_sec: float,
    sample_limit: int,
    remaining_bytes: int | None = None,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="repack_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        images = collect_images(source_path, temp_dir)
        sampled_images = sample_evenly(images, sample_limit)
        results: list[dict] = []
        consumed = 0
        for idx, image_path in enumerate(sampled_images, start=1):
            out_name = f"{source_name}__{idx:05d}.mp4"
            out_path = output_dir / out_name
            size_bytes = write_still_clip(
                image_path=image_path,
                output_path=out_path,
                fps=fps,
                clip_duration_sec=clip_duration_sec,
            )
            if remaining_bytes is not None and consumed + size_bytes > remaining_bytes:
                out_path.unlink(missing_ok=True)
                break
            consumed += size_bytes
            results.append(
                {
                    "path": str(out_path),
                    "filesize_bytes": size_bytes,
                    "derived_from": str(source_path),
                    "source_image": str(image_path),
                    "fps": fps,
                    "clip_duration_sec": clip_duration_sec,
                }
            )
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Repack image archives into short mp4 clips.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-name", type=str, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--clip-duration-sec", type=float, default=1.0)
    parser.add_argument("--sample-limit", type=int, default=200)
    parser.add_argument("--remaining-bytes", type=int, default=None)
    args = parser.parse_args()

    results = repack_archive_to_mp4s(
        source_path=args.input,
        output_dir=args.output_dir,
        source_name=args.source_name,
        fps=args.fps,
        clip_duration_sec=args.clip_duration_sec,
        sample_limit=args.sample_limit,
        remaining_bytes=args.remaining_bytes,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
