#!/usr/bin/env python3
"""video_check_app_train_aligned.py - Training-aligned viewer variant that preserves raw landmark feature inputs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tkinter.scrolledtext as scrolledtext
from typing import Any

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
VIEWER_DEFAULT_TAU = 0.90
NOT_RECORDED = "Not recorded"


def load_run_summary(run_info: base.RunInfo) -> tuple[dict[str, Any] | None, str | None]:
    """Load run_summary.json once and return a friendly warning instead of raising."""
    if run_info.summary_path is None or not run_info.summary_path.exists():
        return None, "Run summary metadata unavailable."

    try:
        summary = json.loads(run_info.summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"Failed to parse run_summary.json: {exc}"

    if not isinstance(summary, dict):
        return None, "Run summary metadata unavailable."
    return summary, None


def summary_value(summary: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = summary
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def run_dataset_key(summary: dict[str, Any] | None) -> str | None:
    raw = summary_value(summary, "dataset_info", "dataset_key") or summary_value(summary, "dataset_key")
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def resolve_tau_threshold(summary: dict[str, Any] | None) -> float:
    raw = summary_value(summary, "hyperparameters", "tau")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return VIEWER_DEFAULT_TAU


def display_value(value: Any, *, fallback: str = NOT_RECORDED) -> str:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        if len(value) <= 4:
            return ", ".join(str(item) for item in value)
        head = ", ".join(str(item) for item in value[:4])
        return f"{head}, ... ({len(value)} total)"
    text = str(value).strip()
    return text if text else fallback


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
    summary, warning = load_run_summary(run_info)
    if summary is None:
        return [], warning or "Inference split metadata unavailable."

    source_groups = summary_value(summary, "dataset_info", "split", "inference", "source_groups")

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
    summary, summary_warning = load_run_summary(run_info)
    resolution = infer_dataset_variant(run_info.run_dir, run_info.summary_path)
    runtime.run_summary = summary
    runtime.dataset_variant = resolution.variant
    runtime.dataset_variant_source = resolution.source
    runtime.dataset_variant_warning = resolution.warning or summary_warning
    runtime.tau_threshold = resolve_tau_threshold(summary)
    _ACTIVE_DATASET_VARIANT = resolution.variant

    dataset_key = run_dataset_key(summary) or resolution.dataset_key
    print(
        f"[viewer] dataset_variant={resolution.variant} "
        f"(source={resolution.source}, dataset_key={dataset_key or '-'}, "
        f"model={runtime.model_id}, run={run_info.run_dir.name}, tau={runtime.tau_threshold:.2f})"
    )
    if runtime.dataset_variant_warning:
        print(f"[viewer] dataset_variant_warning={runtime.dataset_variant_warning}")

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
    confidence = float(probs[pred_idx])
    tau_threshold = runtime.tau_threshold
    if tau_threshold is not None and pred_idx != runtime.neutral_idx and confidence < float(tau_threshold):
        return "tau_neutralized", runtime.neutral_idx, confidence, probs
    return "ready", pred_idx, confidence, probs


class TrainingAlignedVideoCheckApp(base.VideoCheckApp):
    """Base viewer UI with an extra panel for run-specific inference split guidance."""

    def __init__(self, root: base.tk.Tk):
        self.summary_cache: dict[str, dict[str, Any] | None] = {}
        self.summary_warning_cache: dict[str, str | None] = {}
        self.runtime_error_cache: dict[str, str] = {}
        self.current_tau_run_key: str | None = None
        super().__init__(root)
        self.root.title("JamJamBeat Video Check (Train Aligned)")
        self.root.geometry("1360x760")

    def _build_ui(self) -> None:
        frame = base.ttk.Frame(self.root, padding=16)
        frame.pack(fill=base.tk.BOTH, expand=True)

        self.inference_hint_var = base.tk.StringVar(value="Inference split metadata unavailable.")
        self.tau_var = base.tk.StringVar(value=f"{VIEWER_DEFAULT_TAU:.2f}")

        base.ttk.Label(frame, text="Run Search").grid(row=0, column=0, sticky="w")
        self.run_search_entry = base.ttk.Entry(frame, textvariable=self.run_search_var)
        self.run_search_entry.grid(row=1, column=0, sticky="ew", pady=(4, 8))
        self.run_search_var.trace_add("write", lambda *_: self.apply_run_filter())

        base.ttk.Label(frame, text="Trained Run").grid(row=2, column=0, sticky="w")
        self.run_combo = base.ttk.Combobox(frame, textvariable=self.run_var, state="readonly", width=100, height=20)
        self.run_combo.grid(row=3, column=0, sticky="ew", pady=(4, 12))
        self.run_combo.bind("<<ComboboxSelected>>", lambda _: self.update_info())

        base.ttk.Label(frame, text="Video").grid(row=4, column=0, sticky="w")
        self.video_combo = base.ttk.Combobox(frame, textvariable=self.video_var, state="readonly", width=100)
        self.video_combo.grid(row=5, column=0, sticky="ew", pady=(4, 12))
        self.video_combo.bind("<<ComboboxSelected>>", lambda _: self.update_info())

        button_row = base.ttk.Frame(frame)
        button_row.grid(row=6, column=0, sticky="w", pady=(0, 12))
        base.ttk.Button(button_row, text="Refresh", command=self.refresh_options).pack(side=base.tk.LEFT, padx=(0, 8))
        base.ttk.Button(button_row, text="Clear Search", command=self.clear_run_search).pack(side=base.tk.LEFT, padx=(0, 8))
        base.ttk.Button(button_row, text="Analyze And Play", command=self.on_play).pack(side=base.tk.LEFT, padx=(0, 8))
        base.ttk.Label(button_row, text="Tau override").pack(side=base.tk.LEFT, padx=(16, 6))
        self.tau_entry = base.ttk.Entry(button_row, textvariable=self.tau_var, width=8)
        self.tau_entry.pack(side=base.tk.LEFT, padx=(0, 8))
        base.ttk.Button(button_row, text="Quit", command=self.root.destroy).pack(side=base.tk.LEFT)

        base.ttk.Label(frame, textvariable=self.info_var, justify=base.tk.LEFT).grid(row=7, column=0, sticky="ew")

        panels = base.ttk.Frame(frame)
        panels.grid(row=8, column=0, sticky="nsew", pady=(12, 0))

        details_panel = base.ttk.LabelFrame(panels, text="Run Details", padding=10)
        details_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.details_text = scrolledtext.ScrolledText(
            details_panel,
            height=18,
            wrap=base.tk.WORD,
            state="disabled",
        )
        self.details_text.grid(row=0, column=0, sticky="nsew")

        inference_panel = base.ttk.LabelFrame(panels, text="Inference Videos", padding=10)
        inference_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        base.ttk.Label(
            inference_panel,
            textvariable=self.inference_hint_var,
            justify=base.tk.LEFT,
            wraplength=620,
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.inference_text = scrolledtext.ScrolledText(
            inference_panel,
            height=18,
            wrap=base.tk.WORD,
            state="disabled",
        )
        self.inference_text.grid(row=1, column=0, sticky="nsew")

        base.ttk.Label(frame, textvariable=self.status_var, foreground="#005f99").grid(row=9, column=0, sticky="w", pady=(12, 0))

        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(8, weight=1)
        panels.columnconfigure(0, weight=1)
        panels.columnconfigure(1, weight=1)
        panels.rowconfigure(0, weight=1)
        details_panel.columnconfigure(0, weight=1)
        details_panel.rowconfigure(0, weight=1)
        inference_panel.columnconfigure(0, weight=1)
        inference_panel.rowconfigure(1, weight=1)

    def refresh_options(self) -> None:
        self.summary_cache.clear()
        self.summary_warning_cache.clear()
        self.current_tau_run_key = None
        super().refresh_options()

    def _set_inference_text(self, text: str) -> None:
        self.inference_text.configure(state="normal")
        self.inference_text.delete("1.0", base.tk.END)
        self.inference_text.insert("1.0", text)
        self.inference_text.configure(state="disabled")

    def _set_details_text(self, text: str) -> None:
        self.details_text.configure(state="normal")
        self.details_text.delete("1.0", base.tk.END)
        self.details_text.insert("1.0", text)
        self.details_text.configure(state="disabled")

    def _summary_for_run(self, run: base.RunInfo) -> tuple[dict[str, Any] | None, str | None]:
        cache_key = str(run.run_dir)
        if cache_key not in self.summary_cache:
            summary, warning = load_run_summary(run)
            self.summary_cache[cache_key] = summary
            self.summary_warning_cache[cache_key] = warning
        return self.summary_cache[cache_key], self.summary_warning_cache.get(cache_key)

    def _parse_tau_override(self) -> float:
        raw = self.tau_var.get().strip()
        if not raw:
            raise ValueError("Tau override cannot be empty.")
        try:
            tau = float(raw)
        except ValueError as exc:
            raise ValueError(f"Tau override must be a number, got: {raw}") from exc
        if not 0.0 < tau <= 1.0:
            raise ValueError("Tau override must be in the range (0, 1].")
        return tau

    def _sync_tau_for_run(self, run: base.RunInfo, summary: dict[str, Any] | None) -> None:
        run_key = str(run.run_dir)
        if self.current_tau_run_key == run_key:
            return
        self.tau_var.set(f"{resolve_tau_threshold(summary):.2f}")
        self.current_tau_run_key = run_key

    def _build_details_text(
        self,
        run: base.RunInfo,
        *,
        summary: dict[str, Any] | None,
        summary_warning: str | None,
        runtime: base.RuntimeModel | None = None,
    ) -> str:
        resolution = infer_dataset_variant(run.run_dir, run.summary_path)
        checkpoint_info = dict(summary_value(summary, "checkpoint_verification") or {})
        if runtime is not None and runtime.checkpoint_verification:
            checkpoint_info.update(runtime.checkpoint_verification)

        dataset_variant = runtime.dataset_variant if runtime is not None else resolution.variant
        dataset_variant_source = (
            runtime.dataset_variant_source if runtime is not None else resolution.source
        )
        dataset_variant_warning = (
            runtime.dataset_variant_warning if runtime is not None else resolution.warning
        )
        dataset_key = run_dataset_key(summary) or resolution.dataset_key
        hparams = dict(summary_value(summary, "hyperparameters") or {})
        tau_recorded = hparams.get("tau")
        tau_line = display_value(tau_recorded)
        if tau_recorded is None:
            tau_line = f"{NOT_RECORDED} (viewer default {resolve_tau_threshold(summary):.2f})"

        lines = [
            f"Run dir: {run.run_dir}",
            f"Checkpoint: {run.checkpoint_path}",
            f"Run summary: {run.summary_path if run.summary_path else NOT_RECORDED}",
            "",
            f"Dataset key: {display_value(dataset_key)}",
            f"Normalization family: {display_value(summary_value(summary, 'normalization_family'))}",
            f"Mode: {display_value(summary_value(summary, 'mode') or run.mode)}",
            f"Dataset variant: {dataset_variant}",
            f"Variant source: {dataset_variant_source}",
            f"Loss type: {display_value(hparams.get('loss_type'))}",
            f"Weighted sampler: {display_value(hparams.get('use_weighted_sampler'))}",
            f"Alpha enabled: {display_value(hparams.get('use_alpha'))}",
            f"Label smoothing enabled: {display_value(hparams.get('use_label_smoothing'))}",
            f"Focal gamma: {display_value(hparams.get('focal_gamma'))}",
            f"Label smoothing: {display_value(hparams.get('label_smoothing'))}",
            f"Tau: {tau_line}",
            f"Vote N: {display_value(hparams.get('vote_n'))}",
            f"Debounce K: {display_value(hparams.get('debounce_k'))}",
            f"Fallback FPS: {display_value(hparams.get('fallback_fps'))}",
            f"UI tau override: {display_value(self.tau_var.get().strip(), fallback=f'{resolve_tau_threshold(summary):.2f}')}",
            "",
            f"Checkpoint fingerprint: {display_value(checkpoint_info.get('checkpoint_fingerprint'))}",
            f"Strict load verified: {display_value(checkpoint_info.get('strict_load_verified'))}",
            f"Stored matches loaded state: {display_value(checkpoint_info.get('stored_matches_loaded_state'))}",
            f"Missing keys: {display_value(checkpoint_info.get('missing_keys'))}",
            f"Unexpected keys: {display_value(checkpoint_info.get('unexpected_keys'))}",
        ]

        runtime_error = self.runtime_error_cache.get(str(run.run_dir))
        if summary_warning:
            lines.extend(["", f"Summary warning: {summary_warning}"])
        if dataset_variant_warning:
            lines.extend(["", f"Variant warning: {dataset_variant_warning}"])
        if runtime_error:
            lines.extend(["", f"Last load error: {runtime_error}"])
        return "\n".join(lines)

    def update_info(self) -> None:
        super().update_info()

        run = self.run_lookup.get(self.run_var.get())
        video = self.video_lookup.get(self.video_var.get())
        if run is None:
            self.inference_hint_var.set("Inference split metadata unavailable.")
            self._set_inference_text("")
            self._set_details_text("No run selected.")
            return

        summary, summary_warning = self._summary_for_run(run)
        self._sync_tau_for_run(run, summary)
        cached_runtime = self.runtime_cache.get(str(run.run_dir))
        self._set_details_text(
            self._build_details_text(
                run,
                summary=summary,
                summary_warning=summary_warning,
                runtime=cached_runtime,
            )
        )

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
            if not run.checkpoint_path.exists():
                self.status_var.set(f"Checkpoint missing: {run.checkpoint_path}")
            else:
                self.status_var.set("Run selected. Inference split metadata unavailable.")
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
        if not run.checkpoint_path.exists():
            self.status_var.set(f"Checkpoint missing: {run.checkpoint_path}")
        else:
            self.status_var.set(
                f"Checkpoint ready | variant={cached_runtime.dataset_variant if cached_runtime else infer_dataset_variant(run.run_dir, run.summary_path).variant} "
                f"| tau={self.tau_var.get().strip() or f'{resolve_tau_threshold(summary):.2f}'}"
            )

    def on_play(self) -> None:
        """Analyze/play selected run with training-aligned variant logic and tau override."""
        run = self.run_lookup.get(self.run_var.get())
        video = self.video_lookup.get(self.video_var.get())

        if run is None:
            base.messagebox.showerror("Missing Run", "Select a trained run first.")
            return
        if video is None:
            base.messagebox.showerror("Missing Video", "Select a video first.")
            return

        try:
            tau_override = self._parse_tau_override()
            self.runtime_error_cache.pop(str(run.run_dir), None)

            self.status_var.set("Loading model...")
            self.root.update_idletasks()
            runtime = self.get_runtime(run)
            runtime.tau_threshold = tau_override
            self._set_details_text(
                self._build_details_text(
                    run,
                    summary=runtime.run_summary,
                    summary_warning=self.summary_warning_cache.get(str(run.run_dir)),
                    runtime=runtime,
                )
            )

            cache_key = (str(run.run_dir), str(video), f"{tau_override:.6f}")
            analyzed = self.analysis_cache.get(cache_key)
            if analyzed is None:
                self.status_var.set(
                    f"Analyzing video with MediaPipe + model inference... (tau={tau_override:.2f})"
                )
                self.root.update_idletasks()
                analyzed = base.analyze_video(runtime, video)
                self.analysis_cache[cache_key] = analyzed

            self.status_var.set(
                f"Checkpoint loaded | strict={display_value((runtime.checkpoint_verification or {}).get('strict_load_verified'))} "
                f"| variant={runtime.dataset_variant} | tau={tau_override:.2f}"
            )
            self.root.update_idletasks()
            self.root.withdraw()
            try:
                base.playback(runtime, analyzed)
            finally:
                self.root.deiconify()
                self.root.update()
                self.root.lift()
                self.root.focus_force()
                self.status_var.set(f"Playback finished | tau={tau_override:.2f}")
        except Exception as exc:
            self.runtime_error_cache[str(run.run_dir)] = str(exc)
            self.status_var.set("Error")
            summary, summary_warning = self._summary_for_run(run)
            cached_runtime = self.runtime_cache.get(str(run.run_dir))
            self._set_details_text(
                self._build_details_text(
                    run,
                    summary=summary,
                    summary_warning=summary_warning,
                    runtime=cached_runtime,
                )
            )
            base.messagebox.showerror("Viewer Error", str(exc))


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
