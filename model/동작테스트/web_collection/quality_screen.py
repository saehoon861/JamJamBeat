# quality_screen.py - intake 비디오를 샘플링해 품질/hand detect ratio 기반으로 분류
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from common import HAND_LANDMARKER_TASK, ROOT, SEED_PROFILE, ensure_seed_profile_current, load_jsonl, write_jsonl


DEFAULT_INPUT = ROOT / "manifests" / "intake_manifest.jsonl"
DEFAULT_OUTPUT = ROOT / "manifests" / "quality_manifest.jsonl"


def sample_hand_detect_ratio(video_path: Path, sample_every_n_frames: int = 10, max_samples: int = 60) -> tuple[int, int]:
    import mediapipe as mp

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_LANDMARKER_TASK)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    sampled = 0
    detected = 0
    with HandLandmarker.create_from_options(options) as landmarker:
        for frame_idx in range(0, total_frames, sample_every_n_frames):
            if sampled >= max_samples:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(frame_idx / max(fps, 1e-6) * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            sampled += 1
            if getattr(result, "hand_landmarks", None):
                detected += 1

    cap.release()
    return sampled, detected


def classify_quality(row: dict, sampled: int, detected: int, detection_error: str | None = None) -> tuple[str, str]:
    if not row.get("readable"):
        return "rejected", "video_not_readable"
    if float(row.get("duration_sec", 0.0)) < 1.0 or float(row.get("duration_sec", 0.0)) > 120.0:
        return "rejected", "duration_out_of_range"
    if float(row.get("fps", 0.0)) < 15.0:
        return "rejected", "fps_too_low"
    if int(row.get("width", 0)) < 320 or int(row.get("height", 0)) < 240:
        return "rejected", "resolution_too_small"
    if detection_error:
        return "needs_manual_review", f"mediapipe_error:{detection_error}"

    ratio = detected / sampled if sampled > 0 else 0.0
    if ratio >= 0.30:
        return "accepted", "hand_detect_ratio_ok"
    if ratio >= 0.10:
        return "needs_manual_review", "hand_detect_ratio_borderline"
    return "rejected", "hand_detect_ratio_too_low"


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen inbox videos by readability and hand detect ratio.")
    parser.add_argument("--seed-profile", type=Path, default=SEED_PROFILE)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    ensure_seed_profile_current(args.seed_profile)
    intake_rows = load_jsonl(args.input)
    quality_rows: list[dict] = []

    for row in intake_rows:
        video_path = Path(row["path"])
        sampled = 0
        detected = 0
        detection_error = None
        if row.get("readable"):
            try:
                sampled, detected = sample_hand_detect_ratio(video_path)
            except Exception as exc:
                detection_error = str(exc)
        status, reason = classify_quality(row, sampled, detected, detection_error)
        ratio = detected / sampled if sampled > 0 else 0.0
        quality_rows.append(
            {
                **row,
                "sampled_frame_count": sampled,
                "sampled_hand_frames": detected,
                "hand_detect_ratio": round(ratio, 4),
                "quality_status": status,
                "quality_reason": reason,
                "accepted_for_queue": status == "accepted",
            }
        )

    write_jsonl(args.output, quality_rows)
    counts = {
        "accepted": sum(1 for row in quality_rows if row["quality_status"] == "accepted"),
        "needs_manual_review": sum(1 for row in quality_rows if row["quality_status"] == "needs_manual_review"),
        "rejected": sum(1 for row in quality_rows if row["quality_status"] == "rejected"),
    }
    print(json.dumps({"output": str(args.output), "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
