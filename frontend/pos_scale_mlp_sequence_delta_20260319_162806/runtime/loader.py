# loader.py - bundle-local checkpoint 복원과 검증
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import SequenceDeltaMLP


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_torch_load(path: Path, device: torch.device | str) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint format: {path}")
    return checkpoint


def _array_bytes(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr).tobytes(order="C")


def fingerprint_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
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


def strict_load_state_dict(model: torch.nn.Module, state_dict: dict[str, Any]) -> dict[str, Any]:
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


@dataclass
class LoadedBundle:
    bundle_dir: Path
    model: Any
    class_names: list[str]
    feature_order: list[str]
    input_spec: dict[str, Any]
    config: dict[str, Any]
    checkpoint_verification: dict[str, Any]
    device: str
    backend: str

    def run_logits(self, feature_sequence: np.ndarray) -> np.ndarray:
        seq_len = int(self.config["seq_len"])
        input_dim = int(self.config["input_dim"])
        feats = np.asarray(feature_sequence, dtype=np.float32)
        if feats.shape != (seq_len, input_dim):
            raise ValueError(
                f"Expected feature sequence shape ({seq_len}, {input_dim}), got {feats.shape}"
            )

        if self.backend == "torch":
            x_tensor = torch.from_numpy(feats).to(self.device).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(x_tensor).squeeze(0).detach().cpu().numpy()
            return np.asarray(logits, dtype=np.float32)

        if self.backend == "onnx":
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: feats[None, :, :]})
            logits = np.asarray(outputs[0], dtype=np.float32)
            return logits.squeeze(0)

        raise ValueError(f"Unsupported backend: {self.backend}")

    def predict(self, raw_sequence: list[list[float]] | np.ndarray, tau: float | None = None) -> dict[str, Any]:
        from .inference import predict

        return predict(self, raw_sequence, tau=tau)

    def predict_features(self, feature_sequence: np.ndarray, tau: float | None = None) -> dict[str, Any]:
        from .inference import predict_features

        return predict_features(self, feature_sequence, tau=tau)


def load_bundle(bundle_dir: str | Path, device: str = "cpu", backend: str = "torch") -> LoadedBundle:
    bundle_dir = Path(bundle_dir).resolve()
    runtime_dir = bundle_dir / "runtime"

    config = _read_json(runtime_dir / "config.json")
    class_names = list(_read_json(runtime_dir / "class_names.json"))
    feature_order = list(_read_json(runtime_dir / "feature_order.json"))
    input_spec = dict(_read_json(runtime_dir / "input_spec.json"))

    input_dim = int(config["input_dim"])
    num_classes = int(config["num_classes"])
    if len(feature_order) != input_dim:
        raise ValueError(
            f"feature_order length mismatch: expected {input_dim}, got {len(feature_order)}"
        )
    if len(class_names) != num_classes:
        raise ValueError(
            f"class_names length mismatch: expected {num_classes}, got {len(class_names)}"
        )

    checkpoint_path = runtime_dir / "model.pt"
    checkpoint = safe_torch_load(checkpoint_path, device)
    state_dict = checkpoint["model_state_dict"]

    seq_len = int(config["seq_len"])

    torch_model = SequenceDeltaMLP(seq_len=seq_len, input_dim=input_dim, num_classes=num_classes)
    strict_info = strict_load_state_dict(torch_model, state_dict)
    state_verification = fingerprint_state_dict(state_dict)
    stored_verification = dict(checkpoint.get("checkpoint_verification") or {})

    stored_class_names = list(checkpoint.get("class_names") or [])
    if stored_class_names and stored_class_names != class_names:
        raise ValueError(
            "Bundle class_names.json does not match checkpoint class_names."
        )

    checkpoint_verification = {
        "checkpoint_path": str(checkpoint_path),
        **state_verification,
        "stored_checkpoint_fingerprint": stored_verification.get("checkpoint_fingerprint"),
        "stored_matches_loaded_state": (
            stored_verification.get("checkpoint_fingerprint")
            == state_verification["checkpoint_fingerprint"]
        ),
        **strict_info,
    }

    expected_fingerprint = config.get("checkpoint_fingerprint")
    if expected_fingerprint and expected_fingerprint != checkpoint_verification["checkpoint_fingerprint"]:
        raise ValueError(
            "Checkpoint fingerprint mismatch between config.json and loaded checkpoint."
        )

    backend_model: Any
    if backend == "torch":
        torch_model.to(device)
        torch_model.eval()
        backend_model = torch_model
    elif backend == "onnx":
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "ONNX backend requested but onnxruntime is not installed. "
                "Install it with the bundle requirements first."
            ) from exc
        backend_model = ort.InferenceSession(str(runtime_dir / "model.onnx"))
    else:
        raise ValueError(f"Unsupported backend: {backend}. Expected 'torch' or 'onnx'.")

    return LoadedBundle(
        bundle_dir=bundle_dir,
        model=backend_model,
        class_names=class_names,
        feature_order=feature_order,
        input_spec=input_spec,
        config=config,
        checkpoint_verification=checkpoint_verification,
        device=str(device),
        backend=backend,
    )
