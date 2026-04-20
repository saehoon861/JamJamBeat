# run_viewer_probe.py - viewer training-aligned semantics를 비디오 기준으로 JSON 추출
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path
import sys

import cv2
import mediapipe as mp
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from shared.bundle_runtime import load_bundle_runtime
from shared.probe_common import (
    DEFAULT_BUNDLE_DIR,
    DEFAULT_TASK_MODEL_PATH,
    DEFAULT_VIDEO_PATH,
    build_delta_feature_sequence,
    frame_timestamp_ms,
    neutral_probs,
    pos_scale_frame,
    summarize_array,
    topk_from_probs,
    write_json,
)


BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump viewer-style frame predictions to JSON.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "artifacts" / "viewer_probe_default.json",
    )
    parser.add_argument("--tau", type=float, default=0.90)
    parser.add_argument("--task-model", type=Path, default=DEFAULT_TASK_MODEL_PATH)
    return parser.parse_args()


def make_landmarker(task_model_path: Path) -> HandLandmarker:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(task_model_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def main() -> None:
    args = parse_args()
    bundle = load_bundle_runtime(args.bundle_dir)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    seq_buffer: deque[np.ndarray] = deque(maxlen=bundle.seq_len)
    records: list[dict[str, object]] = []

    with make_landmarker(args.task_model) as landmarker:
        for frame_idx in range(total_frames):
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_ms = frame_timestamp_ms(frame_idx, fps)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if not result.hand_landmarks:
                seq_buffer.clear()
                probs = neutral_probs(len(bundle.class_names), bundle.neutral_index)
                records.append(
                    {
                        "frame_idx": frame_idx,
                        "timestamp_ms": timestamp_ms,
                        "hand_detected": False,
                        "status": "no_hand",
                        "frames_collected": 0,
                        "raw_pred_idx": bundle.neutral_index,
                        "raw_pred_label": bundle.class_names[bundle.neutral_index],
                        "final_pred_idx": bundle.neutral_index,
                        "final_pred_label": bundle.class_names[bundle.neutral_index],
                        "confidence": 0.0,
                        "top3": topk_from_probs(bundle.class_names, probs),
                        "tau_applied": args.tau,
                        "tau_neutralized": False,
                        "elapsed_ms": 0.0,
                        "normalized_landmarks_summary": None,
                    }
                )
                continue

            raw_landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]],
                dtype=np.float32,
            )
            normalized = pos_scale_frame(raw_landmarks).reshape(63).astype(np.float32)
            seq_buffer.append(normalized)

            if len(seq_buffer) < bundle.seq_len:
                probs = neutral_probs(len(bundle.class_names), bundle.neutral_index)
                records.append(
                    {
                        "frame_idx": frame_idx,
                        "timestamp_ms": timestamp_ms,
                        "hand_detected": True,
                        "status": "warmup",
                        "frames_collected": len(seq_buffer),
                        "raw_pred_idx": bundle.neutral_index,
                        "raw_pred_label": bundle.class_names[bundle.neutral_index],
                        "final_pred_idx": bundle.neutral_index,
                        "final_pred_label": bundle.class_names[bundle.neutral_index],
                        "confidence": 0.0,
                        "top3": topk_from_probs(bundle.class_names, probs),
                        "tau_applied": args.tau,
                        "tau_neutralized": False,
                        "elapsed_ms": 0.0,
                        "normalized_landmarks_summary": summarize_array("viewer_pos_scale_63", normalized),
                    }
                )
                continue

            features = build_delta_feature_sequence(np.stack(list(seq_buffer), axis=0))
            started = time.perf_counter()
            _, probs_arr = bundle.predict_feature_sequence(features)
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            raw_pred_idx = int(np.argmax(probs_arr))
            confidence = float(probs_arr[raw_pred_idx])
            final_pred_idx = raw_pred_idx
            status = "ready"
            tau_neutralized = False
            if raw_pred_idx != bundle.neutral_index and confidence < float(args.tau):
                final_pred_idx = bundle.neutral_index
                status = "tau_neutralized"
                tau_neutralized = True

            records.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp_ms": timestamp_ms,
                    "hand_detected": True,
                    "status": status,
                    "frames_collected": len(seq_buffer),
                    "raw_pred_idx": raw_pred_idx,
                    "raw_pred_label": bundle.class_names[raw_pred_idx],
                    "final_pred_idx": final_pred_idx,
                    "final_pred_label": bundle.class_names[final_pred_idx],
                    "confidence": confidence,
                    "top3": topk_from_probs(bundle.class_names, probs_arr),
                    "tau_applied": args.tau,
                    "tau_neutralized": tau_neutralized,
                    "elapsed_ms": elapsed_ms,
                    "normalized_landmarks_summary": summarize_array("viewer_pos_scale_63", normalized),
                }
            )

    cap.release()
    payload = {
        "probe": "viewer",
        "video_path": str(args.video.resolve()),
        "bundle_dir": str(Path(args.bundle_dir).resolve()),
        "bundle_id": bundle.bundle_id,
        "model_id": bundle.model_id,
        "checkpoint_fingerprint": bundle.config.get("checkpoint_fingerprint"),
        "tau": float(args.tau),
        "seq_len": bundle.seq_len,
        "fps": fps,
        "total_frames": total_frames,
        "records": records,
    }
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
