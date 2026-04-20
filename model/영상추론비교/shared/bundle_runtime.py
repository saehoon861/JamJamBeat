# bundle_runtime.py - 비교 도구에서 공통으로 쓰는 ONNX 번들 로더
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_runtime_dir(bundle_dir_or_runtime_dir: str | Path) -> Path:
    candidate = Path(bundle_dir_or_runtime_dir).resolve()
    runtime_dir = candidate / "runtime"
    if (runtime_dir / "config.json").exists():
        return runtime_dir
    if (candidate / "config.json").exists():
        return candidate
    raise FileNotFoundError(
        f"Could not resolve runtime directory from: {candidate}. "
        "Expected either .../runtime/config.json or .../config.json."
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float32)
    shifted = arr - np.max(arr)
    exp = np.exp(shifted)
    return (exp / np.sum(exp)).astype(np.float32)


@dataclass(slots=True)
class BundleRuntime:
    runtime_dir: Path
    config: dict[str, Any]
    class_names: list[str]
    session: ort.InferenceSession

    @property
    def bundle_id(self) -> str:
        return str(self.config.get("bundle_id", "unknown"))

    @property
    def model_id(self) -> str:
        return str(self.config.get("model_id", "unknown"))

    @property
    def seq_len(self) -> int:
        return int(self.config["seq_len"])

    @property
    def input_dim(self) -> int:
        return int(self.config["input_dim"])

    @property
    def neutral_index(self) -> int:
        return int(self.config.get("neutral_index", 0))

    def predict_feature_sequence(self, feature_sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        feats = np.asarray(feature_sequence, dtype=np.float32)
        expected_shape = (self.seq_len, self.input_dim)
        if feats.shape != expected_shape:
            raise ValueError(f"Expected feature sequence shape {expected_shape}, got {feats.shape}")

        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: feats[None, :, :]})
        logits = np.asarray(outputs[0], dtype=np.float32).squeeze(0)
        probs = softmax(logits)
        return logits, probs


def load_bundle_runtime(bundle_dir_or_runtime_dir: str | Path) -> BundleRuntime:
    runtime_dir = resolve_runtime_dir(bundle_dir_or_runtime_dir)
    config = dict(_read_json(runtime_dir / "config.json"))
    class_names = list(_read_json(runtime_dir / "class_names.json"))

    session = ort.InferenceSession(
        str(runtime_dir / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    return BundleRuntime(
        runtime_dir=runtime_dir,
        config=config,
        class_names=class_names,
        session=session,
    )
