#!/usr/bin/env python3
"""video_check_app_train_aligned.py - Training-aligned viewer variant that preserves raw landmark feature inputs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tkinter.scrolledtext as scrolledtext

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

from checkpoint_verification import summarize_array


@dataclass(slots=True)
class TrainingAlignedFeaturePack:
    """Runtime feature bundle that prefers the same raw landmark schema used at training time."""

    raw_landmarks: np.ndarray
    train_landmarks: np.ndarray
    normalized_train_landmarks: np.ndarray
    joint: np.ndarray
    bone: np.ndarray
    angle: np.ndarray
    bone_angle: np.ndarray
    full: np.ndarray


_ORIGINAL_LOAD_RUNTIME_MODEL = base.load_runtime_model
_ACTIVE_DATASET_VARIANT = "baseline"


def maybe_log_input_verification(
    runtime: base.RuntimeModel,
    *,
    raw_landmarks: np.ndarray,
    train_landmarks: np.ndarray,
    final_input: np.ndarray,
    input_kind: str,
) -> None:
    """Log one representative runtime input to prove checkpoint/variant application."""
    if runtime.input_verification_logged:
        return

    payload = {
        "model_id": runtime.model_id,
        "mode": runtime.mode,
        "dataset_variant": runtime.dataset_variant,
        "checkpoint_fingerprint": (
            runtime.checkpoint_verification or {}
        ).get("checkpoint_fingerprint"),
        "input_kind": input_kind,
        "raw_landmarks": summarize_array("raw_landmarks", raw_landmarks),
        "variant_landmarks": summarize_array("variant_landmarks", train_landmarks),
        "final_model_input": summarize_array("final_model_input", final_input),
    }
    runtime.input_verification = payload
    runtime.input_verification_logged = True
    print("[runtime] input_verification=" + json.dumps(payload, ensure_ascii=False, sort_keys=True))


def load_inference_source_groups(run_info: base.RunInfo) -> tuple[list[str], str | None]:
    """Read hold-out inference source_file names from the run summary if available."""
    if run_info.summary_path is None or not run_info.summary_path.exists():
        return [], "Inference split metadata unavailable."

    try:
        summary = json.loads(run_info.summary_path.read_text(encoding="utf-8"))
        source_groups = (
            summary.get("dataset_info", {})
            .get("split", {})
            .get("inference", {})
            .get("source_groups", [])
        )
    except Exception:
        return [], "Inference split metadata unavailable."

    if not isinstance(source_groups, list):
        return [], "Inference split metadata unavailable."

    cleaned = [str(value).strip() for value in source_groups if str(value).strip()]
    if not cleaned:
        return [], "Inference split metadata unavailable."
    return cleaned, None


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

    bone = base.compute_bone_features(normalized_train).reshape(-1).astype(np.float32)
    angle = base.compute_angle_features(normalized_train).astype(np.float32)
    bone_angle = np.concatenate([bone, angle]).astype(np.float32)
    full = np.concatenate([normalized_train.reshape(-1).astype(np.float32), bone, angle]).astype(np.float32)

    return TrainingAlignedFeaturePack(
        raw_landmarks=raw,
        train_landmarks=train_landmarks,
        normalized_train_landmarks=normalized_train,
        joint=train_joint,
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
        maybe_log_input_verification(
            runtime,
            raw_landmarks=features.raw_landmarks,
            train_landmarks=features.train_landmarks,
            final_input=vector,
            input_kind="frame_vector",
        )
        tensor = torch.from_numpy(vector).unsqueeze(0).to(runtime.device)
        logits = runtime.model(tensor)

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
        maybe_log_input_verification(
            runtime,
            raw_landmarks=features.raw_landmarks,
            train_landmarks=features.train_landmarks,
            final_input=seq,
            input_kind="sequence_tensor",
        )
        tensor = torch.from_numpy(seq).unsqueeze(0).to(runtime.device)
        logits = runtime.model(tensor)

    elif runtime.mode == "image":
        image = render_train_skeleton_image(features.train_landmarks, runtime.image_size)
        maybe_log_input_verification(
            runtime,
            raw_landmarks=features.raw_landmarks,
            train_landmarks=features.train_landmarks,
            final_input=image,
            input_kind="image_tensor",
        )
        tensor = torch.from_numpy(image).unsqueeze(0).to(runtime.device)
        logits = runtime.model(tensor)

    else:
        raise ValueError(f"Unsupported mode: {runtime.mode}")

    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(float).tolist()
    pred_idx = int(np.argmax(probs))
    return "ready", pred_idx, float(probs[pred_idx]), probs


class TrainingAlignedVideoCheckApp(base.VideoCheckApp):
    """Base viewer UI with an extra panel for run-specific inference split guidance."""

    def __init__(self, root: base.tk.Tk):
        super().__init__(root)
        self.root.title("JamJamBeat Video Check (Train Aligned)")
        self.root.geometry("1080x620")

    def _build_ui(self) -> None:
        frame = base.ttk.Frame(self.root, padding=16)
        frame.pack(fill=base.tk.BOTH, expand=True)

        self.inference_hint_var = base.tk.StringVar(value="Inference split metadata unavailable.")

        base.ttk.Label(frame, text="Trained Run").grid(row=0, column=0, sticky="w")
        self.run_combo = base.ttk.Combobox(frame, textvariable=self.run_var, state="readonly", width=100)
        self.run_combo.grid(row=1, column=0, sticky="ew", pady=(4, 12))
        self.run_combo.bind("<<ComboboxSelected>>", lambda _: self.update_info())

        base.ttk.Label(frame, text="Video").grid(row=2, column=0, sticky="w")
        self.video_combo = base.ttk.Combobox(frame, textvariable=self.video_var, state="readonly", width=100)
        self.video_combo.grid(row=3, column=0, sticky="ew", pady=(4, 12))
        self.video_combo.bind("<<ComboboxSelected>>", lambda _: self.update_info())

        button_row = base.ttk.Frame(frame)
        button_row.grid(row=4, column=0, sticky="w", pady=(0, 12))
        base.ttk.Button(button_row, text="Refresh", command=self.refresh_options).pack(side=base.tk.LEFT, padx=(0, 8))
        base.ttk.Button(button_row, text="Analyze And Play", command=self.on_play).pack(side=base.tk.LEFT, padx=(0, 8))
        base.ttk.Button(button_row, text="Quit", command=self.root.destroy).pack(side=base.tk.LEFT)

        base.ttk.Label(frame, textvariable=self.info_var, justify=base.tk.LEFT).grid(row=5, column=0, sticky="ew")

        inference_panel = base.ttk.LabelFrame(frame, text="Inference Videos", padding=10)
        inference_panel.grid(row=6, column=0, sticky="nsew", pady=(12, 0))
        base.ttk.Label(
            inference_panel,
            textvariable=self.inference_hint_var,
            justify=base.tk.LEFT,
            wraplength=980,
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.inference_text = scrolledtext.ScrolledText(
            inference_panel,
            height=10,
            wrap=base.tk.WORD,
            state="disabled",
        )
        self.inference_text.grid(row=1, column=0, sticky="nsew")

        base.ttk.Label(frame, textvariable=self.status_var, foreground="#005f99").grid(row=7, column=0, sticky="w", pady=(12, 0))

        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(6, weight=1)
        inference_panel.columnconfigure(0, weight=1)
        inference_panel.rowconfigure(1, weight=1)

    def _set_inference_text(self, text: str) -> None:
        self.inference_text.configure(state="normal")
        self.inference_text.delete("1.0", base.tk.END)
        self.inference_text.insert("1.0", text)
        self.inference_text.configure(state="disabled")

    def update_info(self) -> None:
        super().update_info()

        run = self.run_lookup.get(self.run_var.get())
        video = self.video_lookup.get(self.video_var.get())
        if run is None:
            self.inference_hint_var.set("Inference split metadata unavailable.")
            self._set_inference_text("")
            return

        source_groups, warning = load_inference_source_groups(run)
        if warning is not None:
            self.inference_hint_var.set(warning)
            details = ""
            if run.summary_path is not None:
                details = (
                    f"Run summary: {run.summary_path}\n"
                    "This usually means the selected run was created before inference split metadata was recorded."
                )
            self._set_inference_text(details)
            return

        current_video_stem = video.stem if video is not None else None
        in_split = current_video_stem in set(source_groups) if current_video_stem is not None else None

        self.inference_hint_var.set(f"{len(source_groups)} video(s) are in the inference split for this run.")

        lines = [f"{len(source_groups)} video(s) in the inference split:"]
        lines.extend(f"- {source_file}" for source_file in source_groups)

        if current_video_stem is not None:
            lines.append("")
            lines.append(f"Current video: {current_video_stem}")
            if in_split:
                lines.append("This video is part of the inference split.")
            else:
                lines.append("This video is not in the inference split for the selected run.")

        self._set_inference_text("\n".join(lines))


def main() -> None:
    """Run the original viewer UI/CLI with training-aligned feature extraction patched in."""
    print("[viewer] using training-aligned runtime feature mapping")
    base.main()


base.load_runtime_model = load_runtime_model_training_aligned
base.extract_feature_pack = extract_feature_pack_training_aligned
base.select_feature_vector = select_feature_vector_training_aligned
base.predict_from_features = predict_from_features_training_aligned
base.VideoCheckApp = TrainingAlignedVideoCheckApp


if __name__ == "__main__":
    main()
