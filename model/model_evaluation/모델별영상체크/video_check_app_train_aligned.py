#!/usr/bin/env python3
"""video_check_app_train_aligned.py - Training-aligned viewer variant that preserves raw landmark feature inputs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

try:
    from . import video_check_app as base
    from .dataset_variant_runtime import apply_dataset_variant, infer_dataset_variant
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    import video_check_app as base
    from dataset_variant_runtime import apply_dataset_variant, infer_dataset_variant


@dataclass(slots=True)
class TrainingAlignedFeaturePack:
    """Runtime feature bundle that prefers the same raw landmark schema used at training time."""

    raw_landmarks: np.ndarray
    train_landmarks: np.ndarray
    normalized_train_landmarks: np.ndarray
    joint: np.ndarray
    joint_xy: np.ndarray
    joint_z: np.ndarray
    bone: np.ndarray
    angle: np.ndarray
    bone_angle: np.ndarray
    full: np.ndarray


_ORIGINAL_LOAD_RUNTIME_MODEL = base.load_runtime_model
_ACTIVE_DATASET_VARIANT = "baseline"


def load_runtime_model_training_aligned(run_info: base.RunInfo) -> base.RuntimeModel:
    """Resolve run-specific dataset variant and attach it to the runtime model."""
    global _ACTIVE_DATASET_VARIANT

    runtime = _ORIGINAL_LOAD_RUNTIME_MODEL(run_info)
    resolution = infer_dataset_variant(run_info.run_dir, run_info.summary_path)
    runtime.dataset_variant = resolution.variant
    _ACTIVE_DATASET_VARIANT = resolution.variant

    print(
        f"[viewer] dataset_variant={resolution.variant} "
        f"(source={resolution.source}, model={runtime.model_id}, run={run_info.run_dir.name})"
    )
    if resolution.warning:
        print(f"[viewer] dataset_variant_warning={resolution.warning}")

    return runtime


def extract_feature_pack_training_aligned(
    raw_landmarks: np.ndarray,
    dataset_variant: str | None = None,
) -> TrainingAlignedFeaturePack:
    """Build runtime features in the same landmark coordinate system used during training."""
    variant = dataset_variant or _ACTIVE_DATASET_VARIANT
    raw = raw_landmarks.astype(np.float32)
    train_landmarks = apply_dataset_variant(raw, variant)
    normalized_train = base.normalize_landmarks(train_landmarks)

    train_joint = train_landmarks.reshape(-1).astype(np.float32)
    train_joint_xy = train_landmarks[:, :2].reshape(-1).astype(np.float32)
    train_joint_z = train_landmarks[:, 2].reshape(-1).astype(np.float32)

    bone = base.compute_bone_features(normalized_train).reshape(-1).astype(np.float32)
    angle = base.compute_angle_features(normalized_train).astype(np.float32)
    bone_angle = np.concatenate([bone, angle]).astype(np.float32)
    full = np.concatenate([normalized_train.reshape(-1).astype(np.float32), bone, angle]).astype(np.float32)

    return TrainingAlignedFeaturePack(
        raw_landmarks=raw,
        train_landmarks=train_landmarks,
        normalized_train_landmarks=normalized_train,
        joint=train_joint,
        joint_xy=train_joint_xy,
        joint_z=train_joint_z,
        bone=bone,
        angle=angle,
        bone_angle=bone_angle,
        full=full,
    )


def render_train_skeleton_image(train_landmarks: np.ndarray, size: int) -> np.ndarray:
    """Render image-model input from dataset-variant landmarks to match training-time skeleton images."""
    canvas = np.zeros((size, size), dtype=np.float32)
    pts = train_landmarks.reshape(21, 3)
    clipped = np.clip(pts[:, :2], 0.0, 1.0)
    px = clipped[:, 0] * (size - 1)
    py = (size - 1) - (clipped[:, 1] * (size - 1))
    pix_pts = np.stack([px, py], axis=1).astype(np.int32)

    for u, v in base.HAND_CONNECTIONS:
        p1 = tuple(int(x) for x in pix_pts[u])
        p2 = tuple(int(x) for x in pix_pts[v])
        cv2.line(canvas, p1, p2, color=160 / 255.0, thickness=2, lineType=cv2.LINE_AA)

    for p in pix_pts:
        cv2.circle(canvas, tuple(int(x) for x in p), 2, color=1.0, thickness=-1, lineType=cv2.LINE_AA)

    return canvas[None, :, :].astype(np.float32)


def select_feature_vector_training_aligned(
    features: TrainingAlignedFeaturePack,
    expected_dim: int,
) -> np.ndarray:
    """Select raw feature vectors first, then fall back to normalized derived features for legacy checkpoints."""
    raw_candidates = {
        int(features.joint.shape[0]): features.joint,
        int(features.joint_xy.shape[0]): features.joint_xy,
        int(features.joint_z.shape[0]): features.joint_z,
    }
    if int(expected_dim) in raw_candidates:
        return raw_candidates[int(expected_dim)]

    fallback_candidates = {
        int(features.bone.shape[0]): features.bone,
        int(features.angle.shape[0]): features.angle,
        int(features.bone_angle.shape[0]): features.bone_angle,
        int(features.full.shape[0]): features.full,
    }
    vector = fallback_candidates.get(int(expected_dim))
    if vector is None:
        supported = ", ".join(str(dim) for dim in sorted(set(raw_candidates) | set(fallback_candidates)))
        raise ValueError(
            f"Unsupported runtime feature dim: {expected_dim}. "
            f"Supported dims from training-aligned feature pack: {supported}"
        )
    return vector


@torch.inference_mode()
def predict_from_features_training_aligned(
    runtime: base.RuntimeModel,
    features: TrainingAlignedFeaturePack,
    seq_buffer: deque[np.ndarray],
) -> tuple[str, int, float, list[float]]:
    """Predict with feature selection that matches the training dataset schema whenever possible."""
    class_count = len(runtime.class_names)

    if runtime.mode == "frame":
        if runtime.input_dim is None:
            raise ValueError(f"Missing runtime input_dim for frame model: {runtime.model_id}")
        vector = select_feature_vector_training_aligned(features, runtime.input_dim)
        tensor = torch.from_numpy(vector).unsqueeze(0).to(runtime.device)
        logits = runtime.model(tensor)

    elif runtime.mode == "two_stream":
        if runtime.input_dim is None or runtime.aux_input_dim is None:
            raise ValueError(f"Missing runtime stream dims for two_stream model: {runtime.model_id}")
        joint_vec = select_feature_vector_training_aligned(features, runtime.input_dim)
        aux_vec = select_feature_vector_training_aligned(features, runtime.aux_input_dim)
        joint = torch.from_numpy(joint_vec).unsqueeze(0).to(runtime.device)
        aux = torch.from_numpy(aux_vec).unsqueeze(0).to(runtime.device)
        logits = runtime.model(joint, aux)

    elif runtime.mode == "sequence":
        if runtime.input_dim is None:
            raise ValueError(f"Missing runtime input_dim for sequence model: {runtime.model_id}")

        if runtime.model_id == "mlp_sequence_delta":
            if runtime.input_dim % 2 != 0:
                raise ValueError(f"Unexpected delta input_dim for {runtime.model_id}: {runtime.input_dim}")
            base_vec = select_feature_vector_training_aligned(features, runtime.input_dim // 2)
        else:
            base_vec = select_feature_vector_training_aligned(features, runtime.input_dim)

        seq_buffer.append(base_vec)
        if len(seq_buffer) < runtime.seq_len:
            return "warmup", runtime.neutral_idx, 0.0, base.neutral_probs(class_count, runtime.neutral_idx)

        seq = np.stack(list(seq_buffer), axis=0).astype(np.float32)
        if runtime.model_id == "mlp_sequence_delta":
            seq = base.add_runtime_delta_features(seq)
            if seq.shape[1] != runtime.input_dim:
                raise ValueError(
                    f"Delta feature mismatch for {runtime.model_id}: "
                    f"got {seq.shape[1]}, expected {runtime.input_dim}"
                )
        tensor = torch.from_numpy(seq).unsqueeze(0).to(runtime.device)
        logits = runtime.model(tensor)

    elif runtime.mode == "image":
        image = render_train_skeleton_image(features.train_landmarks, runtime.image_size)
        tensor = torch.from_numpy(image).unsqueeze(0).to(runtime.device)
        logits = runtime.model(tensor)

    else:
        raise ValueError(f"Unsupported mode: {runtime.mode}")

    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(float).tolist()
    pred_idx = int(np.argmax(probs))
    return "ready", pred_idx, float(probs[pred_idx]), probs


def main() -> None:
    """Run the original viewer UI/CLI with training-aligned feature extraction patched in."""
    print("[viewer] using training-aligned runtime feature mapping")
    base.main()


base.load_runtime_model = load_runtime_model_training_aligned
base.extract_feature_pack = extract_feature_pack_training_aligned
base.select_feature_vector = select_feature_vector_training_aligned
base.predict_from_features = predict_from_features_training_aligned


if __name__ == "__main__":
    main()
