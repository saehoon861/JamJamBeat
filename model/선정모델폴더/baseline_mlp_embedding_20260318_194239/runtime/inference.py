# inference.py - bundle-local 추론 API
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .loader import LoadedBundle, load_bundle


def _prepare_input(raw_joint63: list[float] | np.ndarray, expected_dim: int) -> np.ndarray:
    arr = np.asarray(raw_joint63, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D input, got shape={arr.shape}")
    if arr.shape[0] != expected_dim:
        raise ValueError(f"Expected input_dim={expected_dim}, got {arr.shape[0]}")
    return arr


def predict(bundle: LoadedBundle, raw_joint63: list[float] | np.ndarray, tau: float | None = None) -> dict[str, Any]:
    x = _prepare_input(raw_joint63, int(bundle.config["input_dim"]))
    x_tensor = torch.from_numpy(x).to(bundle.device).unsqueeze(0)

    with torch.no_grad():
        logits = bundle.model(x_tensor)
        probs_tensor = F.softmax(logits, dim=-1).squeeze(0).detach().cpu()

    probs = probs_tensor.numpy().astype(np.float64).tolist()
    raw_pred_index = int(probs_tensor.argmax().item())
    confidence = float(probs_tensor[raw_pred_index].item())

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


__all__ = ["LoadedBundle", "load_bundle", "predict"]
