# run_frontend_probe.py - frontend sequence runtime semantics를 비디오 기준으로 JSON 추출
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import cv2
import mediapipe as mp
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from shared.bundle_runtime import load_bundle_runtime
from shared.gesture_resolution import ModelGestureResolver
from shared.probe_common import (
    DEFAULT_BUNDLE_DIR,
    DEFAULT_TASK_MODEL_PATH,
    DEFAULT_VIDEO_PATH,
    build_delta_feature_sequence,
    build_frontend_hand_keys,
    frame_timestamp_ms,
    neutral_probs,
    normalize_handedness_labels,
    pos_scale_frame,
    select_frontend_hand_index,
    summarize_array,
    topk_from_probs,
    write_json,
)


BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions


class FrontendModelState:
    def __init__(self, bundle, tau: float, model_interval_ms: float) -> None:
        self.bundle = bundle
        self.tau = float(tau)
        self.model_interval_ms = float(model_interval_ms)
        self.joint_buffer: list[np.ndarray] = []
        self.last_prediction: dict[str, object] | None = None
        self.last_request_at = float("-inf")

    def push_no_hand(self, timestamp_ms: int) -> dict[str, object]:
        self.joint_buffer = []
        probs = neutral_probs(len(self.bundle.class_names), self.bundle.neutral_index)
        self.last_prediction = {
            "label": self.bundle.class_names[self.bundle.neutral_index],
            "confidence": 0.0,
            "classId": self.bundle.neutral_index,
            "source": "onnx-sequence",
            "status": "no_hand",
            "tau_applied": self.tau,
            "tau_neutralized": False,
            "raw_pred_index": self.bundle.neutral_index,
            "raw_pred_label": self.bundle.class_names[self.bundle.neutral_index],
            "framesCollected": 0,
            "elapsed_ms": 0.0,
            "probs": probs,
            "ts_ms": timestamp_ms,
        }
        return dict(self.last_prediction)

    def push_landmarks(self, raw_landmarks: np.ndarray, timestamp_ms: int) -> tuple[dict[str, object] | None, bool, np.ndarray]:
        returned_prediction = dict(self.last_prediction) if self.last_prediction else None
        model_update_ran = False
        normalized = pos_scale_frame(raw_landmarks).reshape(63).astype(np.float32)

        if (timestamp_ms - self.last_request_at) < self.model_interval_ms:
            return returned_prediction, model_update_ran, normalized

        self.last_request_at = float(timestamp_ms)
        model_update_ran = True
        self.joint_buffer.append(normalized)
        if len(self.joint_buffer) > self.bundle.seq_len:
            self.joint_buffer.pop(0)

        if len(self.joint_buffer) < self.bundle.seq_len:
            probs = neutral_probs(len(self.bundle.class_names), self.bundle.neutral_index)
            self.last_prediction = {
                "label": self.bundle.class_names[self.bundle.neutral_index],
                "confidence": 0.0,
                "classId": self.bundle.neutral_index,
                "source": "onnx-sequence",
                "status": "warmup",
                "tau_applied": self.tau,
                "tau_neutralized": False,
                "raw_pred_index": self.bundle.neutral_index,
                "raw_pred_label": self.bundle.class_names[self.bundle.neutral_index],
                "framesCollected": len(self.joint_buffer),
                "elapsed_ms": 0.0,
                "probs": probs,
                "ts_ms": timestamp_ms,
            }
            return returned_prediction, model_update_ran, normalized

        features = build_delta_feature_sequence(np.stack(self.joint_buffer, axis=0))
        started = time.perf_counter()
        _, probs_arr = self.bundle.predict_feature_sequence(features)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        raw_pred_idx = int(np.argmax(probs_arr))
        confidence = float(probs_arr[raw_pred_idx])
        final_idx = raw_pred_idx
        status = "ready"
        tau_neutralized = False
        if raw_pred_idx != self.bundle.neutral_index and confidence < self.tau:
            final_idx = self.bundle.neutral_index
            status = "tau_neutralized"
            tau_neutralized = True

        self.last_prediction = {
            "label": self.bundle.class_names[final_idx],
            "confidence": confidence,
            "classId": final_idx,
            "source": "onnx-sequence",
            "status": status,
            "tau_applied": self.tau,
            "tau_neutralized": tau_neutralized,
            "raw_pred_index": raw_pred_idx,
            "raw_pred_label": self.bundle.class_names[raw_pred_idx],
            "framesCollected": len(self.joint_buffer),
            "elapsed_ms": elapsed_ms,
            "probs": probs_arr.astype(np.float64).tolist(),
            "ts_ms": timestamp_ms,
        }
        return returned_prediction, model_update_ran, normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump frontend-style frame predictions to JSON.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "artifacts" / "frontend_probe_default.json",
    )
    parser.add_argument("--tau", type=float, default=0.85)
    parser.add_argument("--task-model", type=Path, default=DEFAULT_TASK_MODEL_PATH)
    parser.add_argument("--infer-fps", type=float, default=15.0)
    parser.add_argument("--model-interval-ms", type=float, default=150.0)
    parser.add_argument("--infer-width", type=int, default=96)
    parser.add_argument("--hand", default="right")
    return parser.parse_args()


def make_landmarker(task_model_path: Path) -> HandLandmarker:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(task_model_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25,
        min_tracking_confidence=0.25,
    )
    return HandLandmarker.create_from_options(options)


def downscale_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)


def format_raw_for_resolver(prediction: dict[str, object] | None) -> dict[str, object] | None:
    if prediction is None:
        return None
    return {
        "label": prediction.get("label"),
        "confidence": prediction.get("confidence"),
        "classId": prediction.get("classId"),
    }


def main() -> None:
    args = parse_args()
    bundle = load_bundle_runtime(args.bundle_dir)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    infer_interval_ms = 1000.0 / max(args.infer_fps, 1.0)
    next_infer_due_ms = 0.0

    model_state = FrontendModelState(bundle=bundle, tau=args.tau, model_interval_ms=args.model_interval_ms)
    gesture_resolver = ModelGestureResolver()
    last_resolved: dict[str, object] | None = None
    last_raw: dict[str, object] | None = None
    records: list[dict[str, object]] = []

    with make_landmarker(args.task_model) as landmarker:
        for frame_idx in range(total_frames):
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_ms = frame_timestamp_ms(frame_idx, fps)
            mp_inference_ran = False
            model_update_ran = False
            selected_info = {
                "preferred_hand": args.hand,
                "selected_index": None,
                "selected_hand_key": None,
                "selected_handedness": None,
                "fallback_selected": False,
                "detected_handedness": [],
                "detected_hand_keys": [],
            }
            normalized_summary = None
            hand_detected = False
            actual_frontend_disabled = False
            reason = "held"

            if timestamp_ms + 1e-6 >= next_infer_due_ms:
                mp_inference_ran = True
                next_infer_due_ms += infer_interval_ms

                resized = downscale_frame(frame, args.infer_width)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                selected_index, selected_info = select_frontend_hand_index(result, preferred_hand=args.hand)
                hand_detected = selected_index is not None
                if not hand_detected:
                    last_raw = model_state.push_no_hand(timestamp_ms)
                    gesture_resolver.reset()
                    last_resolved = {
                        "label": "None",
                        "confidence": 0.0,
                        "source": "model",
                        "isV": False,
                        "isPaper": False,
                    }
                    reason = "no_hand"
                else:
                    hand_keys = build_frontend_hand_keys(result)
                    labels = normalize_handedness_labels(result)
                    selected_hand_key = hand_keys[selected_index]
                    selected_label = labels[selected_index] if selected_index < len(labels) else None
                    actual_frontend_disabled = selected_hand_key == "left"
                    raw_landmarks = np.array(
                        [[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[selected_index]],
                        dtype=np.float32,
                    )
                    previous_raw, model_update_ran, normalized = model_state.push_landmarks(raw_landmarks, timestamp_ms)
                    normalized_summary = summarize_array("frontend_pos_scale_63", normalized)
                    last_raw = previous_raw
                    if actual_frontend_disabled:
                        gesture_resolver.reset()
                        last_resolved = {
                            "label": "None",
                            "confidence": 0.0,
                            "source": "disabled",
                            "isV": False,
                            "isPaper": False,
                        }
                        reason = f"disabled:{selected_hand_key}:{selected_label or '-'}"
                    else:
                        last_resolved = gesture_resolver.push(format_raw_for_resolver(previous_raw))
                        reason = "model"

            record = {
                "frame_idx": frame_idx,
                "timestamp_ms": timestamp_ms,
                "mp_inference_ran": mp_inference_ran,
                "model_update_ran": model_update_ran,
                "hand_detected": hand_detected,
                "reason": reason,
                "selected_info": selected_info,
                "actual_frontend_disabled": actual_frontend_disabled,
                "raw_model": last_raw,
                "final_gesture": last_resolved,
                "normalized_landmarks_summary": normalized_summary,
            }
            records.append(record)

    cap.release()
    payload = {
        "probe": "frontend",
        "video_path": str(args.video.resolve()),
        "bundle_dir": str(Path(args.bundle_dir).resolve()),
        "bundle_id": bundle.bundle_id,
        "model_id": bundle.model_id,
        "checkpoint_fingerprint": bundle.config.get("checkpoint_fingerprint"),
        "tau": float(args.tau),
        "seq_len": bundle.seq_len,
        "infer_fps": float(args.infer_fps),
        "model_interval_ms": float(args.model_interval_ms),
        "infer_width": int(args.infer_width),
        "hand": args.hand,
        "fps": fps,
        "total_frames": total_frames,
        "records": records,
    }
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
