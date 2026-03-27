# checkpoint_verification.py - Shared checkpoint and runtime-input fingerprint helpers.
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def safe_torch_load(path: Path, device: torch.device | str) -> dict[str, Any]:
    """Load checkpoint dicts across PyTorch versions."""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint format: {path}")
    return checkpoint


def _to_numpy_array(data: Any) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return np.ascontiguousarray(data)
    if torch.is_tensor(data):
        return data.detach().cpu().contiguous().numpy()
    return np.ascontiguousarray(np.asarray(data))


def _array_bytes(arr: np.ndarray) -> bytes:
    """Return deterministic raw bytes for arrays, including 0d tensors."""
    return np.ascontiguousarray(arr).tobytes(order="C")


def summarize_array(name: str, data: Any) -> dict[str, Any]:
    """Create a deterministic fingerprint plus lightweight stats for numeric inputs."""
    arr = _to_numpy_array(data)
    raw = _array_bytes(arr)

    payload = {
        "name": name,
        "shape": [int(dim) for dim in arr.shape],
        "dtype": str(arr.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "numel": int(arr.size),
    }

    if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
        payload.update({"mean": None, "std": None, "min": None, "max": None})
        return payload

    numeric = arr.astype(np.float64, copy=False)
    payload.update(
        {
            "mean": float(numeric.mean()),
            "std": float(numeric.std()),
            "min": float(numeric.min()),
            "max": float(numeric.max()),
        }
    )
    return payload


def fingerprint_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Fingerprint a state_dict deterministically on CPU."""
    hasher = hashlib.sha256()
    keys = sorted(str(key) for key in state_dict.keys())
    tensor_count = 0
    total_numel = 0
    total_bytes = 0

    hasher.update(f"keys:{len(keys)}".encode("utf-8"))

    for key in keys:
        value = state_dict[key]
        hasher.update(key.encode("utf-8"))
        if not torch.is_tensor(value):
            hasher.update(f"non_tensor:{type(value).__name__}".encode("utf-8"))
            continue

        tensor = value.detach().cpu().contiguous()
        arr = tensor.numpy()
        raw = _array_bytes(arr)

        tensor_count += 1
        total_numel += int(tensor.numel())
        total_bytes += len(raw)

        hasher.update(str(tuple(int(dim) for dim in tensor.shape)).encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(raw)

    return {
        "checkpoint_fingerprint": hasher.hexdigest(),
        "key_count": int(len(keys)),
        "tensor_count": int(tensor_count),
        "total_numel": int(total_numel),
        "total_bytes": int(total_bytes),
    }


def strict_load_state_dict(model: torch.nn.Module, state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly load a checkpoint and return explicit verification metadata."""
    try:
        incompatible = model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"Strict checkpoint load failed for {type(model).__name__}: {exc}") from exc

    missing_keys = list(getattr(incompatible, "missing_keys", []))
    unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))
    return {
        "strict": True,
        "strict_load_verified": not missing_keys and not unexpected_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }


def infer_num_classes_from_state_dict(state_dict: Mapping[str, Any]) -> int:
    """Infer classifier output count from common head tensor names."""
    for key in ("head.2.weight", "head.6.weight", "head.weight", "net.6.weight", "net.4.weight"):
        tensor = state_dict.get(key)
        if tensor is not None and hasattr(tensor, "shape") and len(tensor.shape) == 2:
            return int(tensor.shape[0])
    raise ValueError("Could not infer num_classes from checkpoint state_dict.")


def instantiate_model_from_state_dict(
    model_id: str,
    state_dict: Mapping[str, Any],
    num_classes: int,
    seq_len_hint: int,
    *,
    default_seq_len: int = 16,
) -> tuple[torch.nn.Module, int, int | None, int | None]:
    """Rebuild a model instance from its checkpoint tensor shapes."""
    mod = importlib.import_module(f"{model_id}.model")

    if model_id == "mlp_original":
        input_dim = int(state_dict["net.0.weight"].shape[1])
        model = mod.GestureMLP(input_dim=input_dim, num_classes=num_classes)
        return model, default_seq_len, input_dim, None

    if model_id == "mlp_baseline":
        input_dim = int(state_dict["net.0.weight"].shape[1])
        model = mod.MLPBaseline(input_dim=input_dim, num_classes=num_classes)
        return model, default_seq_len, input_dim, None

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
        return model, seq_len_hint or default_seq_len, input_dim, None

    if model_id == "mlp_sequence_delta":
        flat_dim = int(state_dict["net.1.weight"].shape[1])
        seq_len = seq_len_hint or max(flat_dim // 126, 1)
        input_dim = flat_dim // seq_len
        model = mod.SequenceDeltaMLP(seq_len=seq_len, input_dim=input_dim, num_classes=num_classes)
        return model, seq_len, input_dim, None

    if model_id == "mlp_embedding":
        input_dim = int(state_dict["embed.0.weight"].shape[1])
        model = mod.MLPEmbedding(input_dim=input_dim, num_classes=num_classes)
        return model, default_seq_len, input_dim, None

    if model_id == "cnn1d_tcn":
        input_dim = int(state_dict["net.0.weight"].shape[1])
        model = mod.TCN1DClassifier(input_dim=input_dim, num_classes=num_classes)
        return model, default_seq_len, input_dim, None

    if model_id == "transformer_embedding":
        input_dim = int(state_dict["frame_embed.weight"].shape[1])
        seq_len = int(state_dict["pos_embed"].shape[1])
        model = mod.TemporalTransformer(seq_len=seq_len, input_dim=input_dim, num_classes=num_classes)
        return model, seq_len, input_dim, None

    if model_id == "mobilenetv3_small":
        model = mod.MobileNetLike(num_classes=num_classes)
        return model, default_seq_len, None, None

    if model_id == "shufflenetv2_x0_5":
        model = mod.ShuffleNetLike(num_classes=num_classes)
        return model, default_seq_len, None, None

    if model_id == "efficientnet_b0":
        model = mod.EfficientNetLike(num_classes=num_classes)
        return model, default_seq_len, None, None

    if model_id == "frame_spatial_transformer":
        model = mod.LandmarkSpatialTransformer(num_classes=num_classes)
        return model, default_seq_len, None, None

    raise ValueError(f"Unsupported model_id: {model_id}")
