#!/usr/bin/env python3
"""full_source_inference.py - On-demand landmark-frame inference for the eval dashboard."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = ImageDraw = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = Path(__file__).resolve().parent
MODEL_PIPELINES_ROOT = PROJECT_ROOT / "model" / "model_pipelines"
DEFAULT_SEQ_LEN = 16
DEFAULT_IMAGE_SIZE = 96
FORCED_DEVICE = torch.device("cpu")
HAND_CONNECTIONS = [
    (0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]
RAW_JOINT_DIM = 63
RAW_JOINT_XY_DIM = 42
RAW_JOINT_Z_DIM = 21

if str(MODEL_PIPELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_PIPELINES_ROOT))

RUNTIME_CACHE: dict[str, "RuntimeModel"] = {}
INFERENCE_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}


@dataclass(slots=True)
class RunInfo:
    model_id: str
    run_dir: Path
    checkpoint_path: Path
    mode: str


@dataclass(slots=True)
class RuntimeModel:
    run_info: RunInfo
    model: torch.nn.Module
    device: torch.device
    model_id: str
    mode: str
    class_names: list[str]
    neutral_idx: int
    seq_len: int
    image_size: int
    input_dim: int | None
    aux_input_dim: int | None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_dashboard_relative(path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate.resolve()
    return (APP_ROOT / candidate).resolve()


def dashboard_data_root() -> Path:
    return APP_ROOT / "data" / "suites"


def detail_json_path(suite_name: str, model_id: str) -> Path:
    path = (dashboard_data_root() / suite_name / "models" / f"{model_id}.json").resolve()
    try:
        path.relative_to(dashboard_data_root().resolve())
    except ValueError as exc:
        raise ValueError("Invalid suite/model path") from exc
    if not path.exists():
        raise FileNotFoundError(f"Model detail JSON not found: {path}")
    return path


def format_timestamp(frame_idx: int, fps: float) -> str:
    total_ms = int((frame_idx / max(fps, 1e-6)) * 1000)
    minutes = total_ms // 60000
    seconds = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{minutes:02d}:{seconds:02d}:{ms:03d}"


def safe_torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint format: {path}")
    return checkpoint


def detect_neutral_idx(class_names: list[str]) -> int:
    for idx, name in enumerate(class_names):
        if str(name).strip().lower() in {"neutral", "none", "background"}:
            return idx
    return 0


def infer_num_classes(state_dict: dict[str, Any]) -> int:
    for key in ("head.2.weight", "head.6.weight", "head.weight", "net.6.weight", "net.4.weight"):
        tensor = state_dict.get(key)
        if tensor is not None and hasattr(tensor, "shape") and len(tensor.shape) == 2:
            return int(tensor.shape[0])
    raise ValueError("Could not infer num_classes from checkpoint state_dict.")


def instantiate_model(
    model_id: str,
    state_dict: dict[str, Any],
    num_classes: int,
    seq_len_hint: int,
) -> tuple[torch.nn.Module, int, int | None, int | None]:
    mod = importlib.import_module(f"{model_id}.model")

    if model_id == "mlp_original":
        input_dim = int(state_dict["net.0.weight"].shape[1])
        model = mod.GestureMLP(input_dim=input_dim, num_classes=num_classes)
        return model, DEFAULT_SEQ_LEN, input_dim, None

    if model_id in {"mlp_baseline", "mlp_baseline_full"}:
        input_dim = int(state_dict["net.0.weight"].shape[1])
        model = mod.MLPBaseline(input_dim=input_dim, num_classes=num_classes)
        return model, DEFAULT_SEQ_LEN, input_dim, None

    if model_id == "mlp_baseline_seq8":
        flat_dim = int(state_dict["net.0.weight"].shape[1])
        seq_len = 8
        input_dim = flat_dim // seq_len
        model = mod.MLPBaselineSeq(seq_len=seq_len, input_dim=input_dim, num_classes=num_classes)
        return model, seq_len, input_dim, None

    if model_id == "mlp_sequence_joint":
        input_dim = int(state_dict["net.1.weight"].shape[1])
        seq_len = seq_len_hint or max(input_dim // 63, 1)
        feature_dim = input_dim // max(seq_len, 1)
        model = mod.SequenceJointMLP(seq_len=seq_len, input_dim=feature_dim, num_classes=num_classes)
        return model, seq_len, feature_dim, None

    if model_id == "mlp_temporal_pooling":
        input_dim = int(state_dict["frame_embed.1.weight"].shape[1])
        model = mod.TemporalPoolingMLP(input_dim=input_dim, num_classes=num_classes)
        return model, seq_len_hint or DEFAULT_SEQ_LEN, input_dim, None

    if model_id == "mlp_sequence_delta":
        flat_dim = int(state_dict["net.1.weight"].shape[1])
        seq_len = seq_len_hint or max(flat_dim // 126, 1)
        input_dim = flat_dim // seq_len
        model = mod.SequenceDeltaMLP(seq_len=seq_len, input_dim=input_dim, num_classes=num_classes)
        return model, seq_len, input_dim, None

    if model_id == "mlp_embedding":
        input_dim = int(state_dict["embed.0.weight"].shape[1])
        model = mod.MLPEmbedding(input_dim=input_dim, num_classes=num_classes)
        return model, DEFAULT_SEQ_LEN, input_dim, None

    if model_id == "two_stream_mlp":
        joint_dim = int(state_dict["joint_stream.net.0.weight"].shape[1])
        bone_dim = int(state_dict["bone_stream.net.0.weight"].shape[1])
        model = mod.TwoStreamMLP(joint_dim=joint_dim, bone_dim=bone_dim, num_classes=num_classes)
        return model, DEFAULT_SEQ_LEN, joint_dim, bone_dim

    if model_id == "cnn1d_tcn":
        input_dim = int(state_dict["net.0.weight"].shape[1])
        model = mod.TCN1DClassifier(input_dim=input_dim, num_classes=num_classes)
        return model, DEFAULT_SEQ_LEN, input_dim, None

    if model_id == "transformer_embedding":
        input_dim = int(state_dict["frame_embed.weight"].shape[1])
        seq_len = int(state_dict["pos_embed"].shape[1])
        model = mod.TemporalTransformer(seq_len=seq_len, input_dim=input_dim, num_classes=num_classes)
        return model, seq_len, input_dim, None

    if model_id == "mobilenetv3_small":
        model = mod.MobileNetLike(num_classes=num_classes)
        return model, DEFAULT_SEQ_LEN, None, None

    if model_id == "shufflenetv2_x0_5":
        model = mod.ShuffleNetLike(num_classes=num_classes)
        return model, DEFAULT_SEQ_LEN, None, None

    if model_id == "efficientnet_b0":
        model = mod.EfficientNetLike(num_classes=num_classes)
        return model, DEFAULT_SEQ_LEN, None, None

    raise ValueError(f"Unsupported model_id: {model_id}")


def load_runtime_model(run_info: RunInfo) -> RuntimeModel:
    checkpoint = safe_torch_load(run_info.checkpoint_path, FORCED_DEVICE)
    state_dict = checkpoint["model_state_dict"]
    class_names = list(checkpoint.get("class_names") or [])
    if not class_names:
        class_names = [str(i) for i in range(infer_num_classes(state_dict))]

    num_classes = len(class_names)
    model_id = str(checkpoint.get("model_id") or run_info.model_id)
    mode = str(checkpoint.get("mode") or run_info.mode)
    seq_len_hint = int(checkpoint.get("seq_len") or DEFAULT_SEQ_LEN)
    image_size = int(checkpoint.get("image_size") or DEFAULT_IMAGE_SIZE)
    model, seq_len, input_dim, aux_input_dim = instantiate_model(model_id, state_dict, num_classes, seq_len_hint)
    model.load_state_dict(state_dict)
    model.to(FORCED_DEVICE)
    model.eval()

    return RuntimeModel(
        run_info=run_info,
        model=model,
        device=FORCED_DEVICE,
        model_id=model_id,
        mode=mode,
        class_names=class_names,
        neutral_idx=detect_neutral_idx(class_names),
        seq_len=seq_len,
        image_size=image_size,
        input_dim=input_dim,
        aux_input_dim=aux_input_dim,
    )


def row_to_raw_landmarks(row: dict[str, str]) -> np.ndarray:
    return np.array(
        [
            [
                float(row.get(f"x{i}") or 0.0),
                float(row.get(f"y{i}") or 0.0),
                float(row.get(f"z{i}") or 0.0),
            ]
            for i in range(21)
        ],
        dtype=np.float32,
    )


def raw_feature_candidates(raw_landmarks: np.ndarray) -> dict[int, np.ndarray]:
    return {
        RAW_JOINT_DIM: raw_landmarks.reshape(-1).astype(np.float32),
        RAW_JOINT_XY_DIM: raw_landmarks[:, :2].reshape(-1).astype(np.float32),
        RAW_JOINT_Z_DIM: raw_landmarks[:, 2].reshape(-1).astype(np.float32),
    }


def select_raw_feature_vector(raw_landmarks: np.ndarray, expected_dim: int) -> np.ndarray:
    candidates = raw_feature_candidates(raw_landmarks)
    vector = candidates.get(int(expected_dim))
    if vector is None:
        supported = ", ".join(str(dim) for dim in sorted(candidates))
        raise ValueError(f"Unsupported raw feature dim: {expected_dim}. Supported dims: {supported}")
    return vector


def render_skeleton_image(raw_landmarks: np.ndarray, size: int) -> np.ndarray:
    pts = raw_landmarks.reshape(21, 3).astype(np.float32)
    x = np.clip(pts[:, 0], 0.0, 1.0) * (size - 1)
    y = (size - 1) - (np.clip(pts[:, 1], 0.0, 1.0) * (size - 1))

    if Image is not None and ImageDraw is not None:
        img = Image.new("L", (size, size), color=0)
        draw = ImageDraw.Draw(img)
        for u, v in HAND_CONNECTIONS:
            draw.line(
                [(float(x[u]), float(y[u])), (float(x[v]), float(y[v]))],
                fill=160,
                width=2,
            )
        for i in range(21):
            radius = 2
            draw.ellipse(
                [
                    (float(x[i] - radius), float(y[i] - radius)),
                    (float(x[i] + radius), float(y[i] + radius)),
                ],
                fill=255,
            )
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr[None, :, :]

    canvas = np.zeros((size, size), dtype=np.float32)
    pix_pts = np.stack([x, y], axis=1).astype(np.int32)
    for u, v in HAND_CONNECTIONS:
        cv2.line(
            canvas,
            tuple(int(val) for val in pix_pts[u]),
            tuple(int(val) for val in pix_pts[v]),
            color=160 / 255.0,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    for point in pix_pts:
        cv2.circle(canvas, tuple(int(val) for val in point), 2, color=1.0, thickness=-1, lineType=cv2.LINE_AA)
    return canvas[None, :, :].astype(np.float32)


def neutral_probs(class_count: int, neutral_idx: int) -> list[float]:
    probs = [0.0] * class_count
    probs[neutral_idx] = 1.0
    return probs


def add_runtime_delta_features(seq: np.ndarray) -> np.ndarray:
    delta = np.zeros_like(seq, dtype=np.float32)
    delta[1:, :] = seq[1:, :] - seq[:-1, :]
    return np.concatenate([seq.astype(np.float32), delta], axis=1)


@torch.inference_mode()
def predict_from_features(runtime: RuntimeModel, raw_landmarks: np.ndarray, seq_buffer: deque[np.ndarray]) -> tuple[str, int, float, list[float]]:
    class_count = len(runtime.class_names)

    if runtime.mode == "frame":
        if runtime.input_dim is None:
            raise ValueError(f"Missing input_dim for frame model: {runtime.model_id}")
        vector = select_raw_feature_vector(raw_landmarks, runtime.input_dim)
        logits = runtime.model(torch.from_numpy(vector).unsqueeze(0).to(runtime.device))

    elif runtime.mode == "two_stream":
        if runtime.input_dim is None or runtime.aux_input_dim is None:
            raise ValueError(f"Missing stream dims for two_stream model: {runtime.model_id}")
        joint = torch.from_numpy(select_raw_feature_vector(raw_landmarks, runtime.input_dim)).unsqueeze(0).to(runtime.device)
        aux = torch.from_numpy(select_raw_feature_vector(raw_landmarks, runtime.aux_input_dim)).unsqueeze(0).to(runtime.device)
        logits = runtime.model(joint, aux)

    elif runtime.mode == "sequence":
        if runtime.input_dim is None:
            raise ValueError(f"Missing input_dim for sequence model: {runtime.model_id}")
        if runtime.model_id == "mlp_sequence_delta":
            if runtime.input_dim % 2 != 0:
                raise ValueError(f"Unexpected delta input_dim for {runtime.model_id}: {runtime.input_dim}")
            base_vec = select_raw_feature_vector(raw_landmarks, runtime.input_dim // 2)
        else:
            base_vec = select_raw_feature_vector(raw_landmarks, runtime.input_dim)

        seq_buffer.append(base_vec)
        if len(seq_buffer) < runtime.seq_len:
            return "warmup", runtime.neutral_idx, 0.0, neutral_probs(class_count, runtime.neutral_idx)

        seq = np.stack(list(seq_buffer), axis=0).astype(np.float32)
        if runtime.model_id == "mlp_sequence_delta":
            seq = add_runtime_delta_features(seq)
        logits = runtime.model(torch.from_numpy(seq).unsqueeze(0).to(runtime.device))

    elif runtime.mode == "image":
        image = render_skeleton_image(raw_landmarks, runtime.image_size)
        logits = runtime.model(torch.from_numpy(image).unsqueeze(0).to(runtime.device))

    else:
        raise ValueError(f"Unsupported mode: {runtime.mode}")

    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(float).tolist()
    pred_idx = int(np.argmax(probs))
    return "ready", pred_idx, float(probs[pred_idx]), probs


def build_run_info(detail_payload: dict[str, Any]) -> RunInfo:
    run_dir = Path(str(detail_payload["run_dir"])).resolve()
    checkpoint_path = run_dir / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    comparison_row = detail_payload.get("comparison_row") or {}
    return RunInfo(
        model_id=str(detail_payload["model_id"]),
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        mode=str(detail_payload.get("mode") or comparison_row.get("mode") or "unknown"),
    )


def get_runtime(detail_payload: dict[str, Any]) -> RuntimeModel:
    run_dir = str(detail_payload["run_dir"])
    cached = RUNTIME_CACHE.get(run_dir)
    if cached is not None:
        return cached
    runtime = load_runtime_model(build_run_info(detail_payload))
    RUNTIME_CACHE[run_dir] = runtime
    return runtime


def find_video(detail_payload: dict[str, Any], source_file: str) -> dict[str, Any]:
    for video in detail_payload.get("videos") or []:
        if video.get("source_file") == source_file:
            return video
    raise ValueError(f"Source not found in model detail payload: {source_file}")


def load_dataset_row_map(video_meta: dict[str, Any]) -> dict[int, dict[str, str]]:
    dataset_csv_path = resolve_dashboard_relative(str(video_meta.get("dataset_csv_path") or ""))
    if not dataset_csv_path.exists():
        return {}
    row_map: dict[int, dict[str, str]] = {}
    with dataset_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        has_source_file = bool(reader.fieldnames and "source_file" in reader.fieldnames)
        for row in reader:
            if has_source_file and str(row.get("source_file") or "").strip() != str(video_meta["source_file"]).strip():
                continue
            frame_idx_raw = str(row.get("frame_idx") or "").strip()
            if frame_idx_raw == "":
                continue
            row_map[int(frame_idx_raw)] = row
    return row_map


def load_landmark_rows(video_meta: dict[str, Any]) -> list[dict[str, str]]:
    landmark_path = resolve_dashboard_relative(str(video_meta.get("landmark_path") or ""))
    dataset_row_map = load_dataset_row_map(video_meta)
    source_file = str(video_meta["source_file"])

    def has_landmarks(row: dict[str, str]) -> bool:
        return str(row.get("x0") or "").strip() != ""

    def merge_with_dataset(row: dict[str, str], dataset_row: dict[str, str] | None) -> dict[str, str]:
        enriched = dict(row)
        enriched.setdefault("source_file", source_file)
        if dataset_row:
            for key in ("gesture", "gesture_name", "timestamp"):
                if str(enriched.get(key) or "").strip() == "" and str(dataset_row.get(key) or "").strip() != "":
                    enriched[key] = str(dataset_row.get(key))
            for idx in range(21):
                for axis in ("x", "y", "z"):
                    coord_key = f"{axis}{idx}"
                    if str(enriched.get(coord_key) or "").strip() == "" and str(dataset_row.get(coord_key) or "").strip() != "":
                        enriched[coord_key] = str(dataset_row.get(coord_key))
        return enriched

    dedicated_rows_by_frame: dict[int, dict[str, str]] = {}
    if landmark_path.exists():
        with landmark_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            has_source_file = bool(reader.fieldnames and "source_file" in reader.fieldnames)
            for row in reader:
                if has_source_file and str(row.get("source_file") or "").strip() != source_file:
                    continue
                frame_idx_raw = str(row.get("frame_idx") or "").strip()
                if frame_idx_raw == "":
                    continue
                frame_idx = int(frame_idx_raw)
                dedicated_rows_by_frame[frame_idx] = merge_with_dataset(row, dataset_row_map.get(frame_idx))

    rows_by_frame: dict[int, dict[str, str]] = {}
    for frame_idx, dataset_row in dataset_row_map.items():
        if has_landmarks(dataset_row):
            rows_by_frame[frame_idx] = merge_with_dataset(dataset_row, dedicated_rows_by_frame.get(frame_idx))
            continue
        dedicated_row = dedicated_rows_by_frame.get(frame_idx)
        if dedicated_row is None:
            continue
        if not has_landmarks(dedicated_row):
            continue
        rows_by_frame[frame_idx] = dedicated_row

    for frame_idx, dedicated_row in dedicated_rows_by_frame.items():
        if frame_idx in rows_by_frame:
            continue
        if not has_landmarks(dedicated_row):
            continue
        rows_by_frame[frame_idx] = dedicated_row

    return [rows_by_frame[frame_idx] for frame_idx in sorted(rows_by_frame)]


def class_name(class_names: list[str], index: int | None) -> str:
    if index is None:
        return "-"
    if 0 <= int(index) < len(class_names):
        return class_names[int(index)]
    return str(index)


def infer_landmark_frames(detail_payload: dict[str, Any], source_file: str) -> dict[str, Any]:
    cache_key = (str(detail_payload["suite_name"]), str(detail_payload["model_id"]), source_file)
    cached = INFERENCE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    video_meta = find_video(detail_payload, source_file)
    runtime = get_runtime(detail_payload)
    landmark_rows = load_landmark_rows(video_meta)
    stored_events_by_frame = {
        int(event["frame_idx"]): event
        for event in (video_meta.get("events") or [])
        if str(event.get("frame_idx") or "").strip() != ""
    }
    seq_buffer: deque[np.ndarray] = deque(maxlen=runtime.seq_len)
    events: list[dict[str, Any]] = []

    for row in landmark_rows:
        frame_idx = int(row.get("frame_idx") or 0)
        raw_landmarks = row_to_raw_landmarks(row)
        status, pred_idx, confidence, probs = predict_from_features(runtime, raw_landmarks, seq_buffer)
        gt_idx = int(row["gesture"]) if str(row.get("gesture") or "").strip() not in {"", "None"} else None
        gt_name = class_name(runtime.class_names, gt_idx)
        pred_name = class_name(runtime.class_names, pred_idx)
        event = {
            "frame_idx": frame_idx,
            "timestamp": row.get("timestamp") or format_timestamp(frame_idx, float(video_meta.get("fps") or 30.0)),
            "ground_truth": gt_name,
            "ground_truth_idx": gt_idx,
            "predicted": pred_name,
            "predicted_idx": pred_idx,
            "confidence": round(float(confidence), 6),
            "probabilities": [{"label": runtime.class_names[idx], "value": round(float(value), 6)} for idx, value in enumerate(probs)],
            "status": status,
            "is_mismatch": gt_idx is not None and int(pred_idx) != int(gt_idx),
        }
        stored_event = stored_events_by_frame.get(frame_idx)
        if stored_event is not None:
            event.update(
                {
                    "timestamp": stored_event.get("timestamp") or event["timestamp"],
                    "ground_truth": stored_event.get("ground_truth") or event["ground_truth"],
                    "predicted": stored_event.get("predicted") or event["predicted"],
                    "confidence": float(stored_event.get("confidence") or event["confidence"]),
                    "probabilities": stored_event.get("probabilities") or event["probabilities"],
                    "status": "stored_test",
                    "is_mismatch": bool(stored_event.get("is_mismatch")),
                    "stored_prediction_available": True,
                    "runtime_predicted": pred_name,
                    "runtime_confidence": round(float(confidence), 6),
                }
            )
        else:
            event["stored_prediction_available"] = False
        events.append(event)

    payload = {
        "suite_name": detail_payload["suite_name"],
        "model_id": detail_payload["model_id"],
        "source_file": source_file,
        "frame_count": len(events),
        "landmark_source_kind": video_meta.get("landmark_source_kind"),
        "events": events,
    }
    INFERENCE_CACHE[cache_key] = payload
    return payload


def infer_from_suite_model_source(suite_name: str, model_id: str, source_file: str) -> dict[str, Any]:
    detail_payload = read_json(detail_json_path(suite_name, model_id))
    return infer_landmark_frames(detail_payload, source_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full landmark-frame inference for one suite/model/source.")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = infer_from_suite_model_source(args.suite, args.model, args.source)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
