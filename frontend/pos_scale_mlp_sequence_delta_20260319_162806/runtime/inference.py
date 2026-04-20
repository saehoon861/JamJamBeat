# inference.py - bundle-local 추론 API
from __future__ import annotations

from typing import Any

import numpy as np

from .loader import LoadedBundle, load_bundle
from .preprocess import prepare_features as _prepare_raw_features


def _softmax(logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float32)
    shifted = arr - np.max(arr)
    exp = np.exp(shifted)
    return (exp / np.sum(exp)).astype(np.float32)


def prepare_features(bundle: LoadedBundle, raw_sequence: list[list[float]] | np.ndarray) -> np.ndarray:
    return _prepare_raw_features(raw_sequence, seq_len=int(bundle.config["seq_len"]))


def _prepare_feature_input(bundle: LoadedBundle, feature_sequence: np.ndarray) -> np.ndarray:
    arr = np.asarray(feature_sequence, dtype=np.float32)
    expected_shape = (int(bundle.config["seq_len"]), int(bundle.config["input_dim"]))
    if arr.shape != expected_shape:
        raise ValueError(f"Expected feature sequence shape {expected_shape}, got {arr.shape}")
    return arr


def predict_features(bundle: LoadedBundle, feature_sequence: np.ndarray, tau: float | None = None) -> dict[str, Any]:
    feats = _prepare_feature_input(bundle, feature_sequence)
    logits = bundle.run_logits(feats)
    probs_arr = _softmax(logits)

    probs = probs_arr.astype(np.float64).tolist()
    raw_pred_index = int(np.argmax(probs_arr))
    confidence = float(probs_arr[raw_pred_index])
    pred_index = raw_pred_index
    pred_label = bundle.class_names[pred_index]
    tau_applied = tau
    tau_neutralized = False

    neutral_index = int(bundle.config.get("neutral_index", 0))
    if tau is not None and confidence < float(tau):
        pred_index = neutral_index
        pred_label = bundle.class_names[pred_index]
        tau_neutralized = pred_index != raw_pred_index

    return {
        "pred_index": pred_index,
        "pred_label": pred_label,
        "confidence": confidence,
        "probs": probs,
        "raw_pred_index": raw_pred_index,
        "raw_pred_label": bundle.class_names[raw_pred_index],
        "tau_applied": tau_applied,
        "tau_neutralized": tau_neutralized,
    }


def predict(bundle: LoadedBundle, raw_sequence: list[list[float]] | np.ndarray, tau: float | None = None) -> dict[str, Any]:
    feats = prepare_features(bundle, raw_sequence)
    return predict_features(bundle, feats, tau=tau)


__all__ = ["LoadedBundle", "load_bundle", "prepare_features", "predict_features", "predict"]
