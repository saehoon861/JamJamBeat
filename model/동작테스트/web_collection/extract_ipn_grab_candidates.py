"""extract_ipn_grab_candidates.py - IPN Hand에서 grab 유사 후보 구간만 잘라 별도 클립으로 저장."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2

from common import ROOT, VIDEO_SUFFIXES, write_jsonl


IPN_ANNOTATIONS = ROOT / "IPN_Hand" / "annotations" / "Annot_List.txt"
IPN_CLIP_DIR = ROOT / "grab_candidate_clips" / "ipn_hand"
IPN_MANIFEST = ROOT / "manifests" / "ipn_grab_candidate_manifest.jsonl"
OFFICIAL_INBOX = ROOT / "inbox_videos" / "official"

TARGET_LABELS = ("G07", "G10", "G11")
LABEL_DESCRIPTIONS = {
    "G07": "open_twice",
    "G10": "zoom_in",
    "G11": "zoom_out",
}
PRE_CONTEXT_FRAMES = 15
POST_CONTEXT_FRAMES = 15
OUTPUT_FOURCC = "MJPG"


def local_ipn_videos() -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in sorted(OFFICIAL_INBOX.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        if "ipn_hand_" not in path.name:
            continue
        parts = path.stem.split("__")
        if len(parts) < 4:
            continue
        mapping[f"{parts[-2]}__{parts[-1]}"] = path
    return mapping


def load_target_segments(video_ids: set[str]) -> dict[str, list[dict[str, int | str]]]:
    segments: dict[str, list[dict[str, int | str]]] = {video_id: [] for video_id in video_ids}
    with IPN_ANNOTATIONS.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            video_key = row["video"].replace("_#", "__")
            label = row["label"]
            if video_key not in segments or label not in TARGET_LABELS:
                continue
            segments[video_key].append(
                {
                    "label_code": label,
                    "label_name": LABEL_DESCRIPTIONS[label],
                    "t_start": int(row["t_start"]),
                    "t_end": int(row["t_end"]),
                }
            )
    return segments


def extract_clip(
    source_path: Path,
    output_path: Path,
    frame_start: int,
    frame_end: int,
    fps: float,
    width: int,
    height: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video: {source_path}")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*OUTPUT_FOURCC),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open writer for: {output_path}")

    written = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    current = frame_start
    while current <= frame_end:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
        current += 1

    writer.release()
    cap.release()
    return written


def main() -> None:
    IPN_CLIP_DIR.mkdir(parents=True, exist_ok=True)

    source_videos = local_ipn_videos()
    target_segments = load_target_segments(set(source_videos))
    manifest_rows: list[dict[str, object]] = []

    total_clips = 0
    for video_key, source_path in sorted(source_videos.items()):
        segments = target_segments.get(video_key, [])
        if not segments:
            continue

        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            print(f"skip unreadable source: {source_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        if width <= 0 or height <= 0 or total_frames <= 0:
            print(f"skip invalid metadata: {source_path}")
            continue

        safe_video_key = video_key.replace("__", "_")
        for idx, segment in enumerate(segments, start=1):
            t_start = int(segment["t_start"])
            t_end = int(segment["t_end"])
            frame_start = max(0, (t_start - 1) - PRE_CONTEXT_FRAMES)
            frame_end = min(total_frames - 1, (t_end - 1) + POST_CONTEXT_FRAMES)
            output_name = (
                f"{safe_video_key}__{segment['label_code']}__"
                f"{segment['label_name']}__clip{idx:02d}.avi"
            )
            output_path = IPN_CLIP_DIR / output_name
            frames_written = extract_clip(
                source_path=source_path,
                output_path=output_path,
                frame_start=frame_start,
                frame_end=frame_end,
                fps=fps,
                width=width,
                height=height,
            )
            total_clips += 1
            manifest_rows.append(
                {
                    "source_video_id": video_key,
                    "source_video_path": str(source_path),
                    "output_clip_path": str(output_path),
                    "label_code": segment["label_code"],
                    "label_name": segment["label_name"],
                    "annotation_start_frame_1based": t_start,
                    "annotation_end_frame_1based": t_end,
                    "clip_frame_start_0based": frame_start,
                    "clip_frame_end_0based": frame_end,
                    "frames_written": frames_written,
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "note": "IPN candidate only; not an official grab label.",
                }
            )

    write_jsonl(IPN_MANIFEST, manifest_rows)
    print(
        f"Extracted {total_clips} IPN candidate clips "
        f"from {len(source_videos)} local IPN videos into {IPN_CLIP_DIR}"
    )


if __name__ == "__main__":
    main()
