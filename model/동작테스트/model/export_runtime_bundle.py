# export_runtime_bundle.py - 학습 run 체크포인트를 브라우저용 ONNX runtime bundle로 변환
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model import SequenceDeltaMLP


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
FRONTEND_PUBLIC = REPO_ROOT / "frontend" / "public"
RUNTIME_TEMPLATE = FRONTEND_PUBLIC / "runtime_sequence"
DEFAULT_RUN_DIR = ROOT / "runs" / "20260323_005909"
DEFAULT_BUNDLE_NAME = "runtime_sequence_grab_20260323_005909"
DEFAULT_CLASS_NAMES = [
    "neutral",
    "fist",
    "open_palm",
    "V",
    "pinky",
    "animal",
    "k-heart",
    "grab",
]


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


def safe_torch_load(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint format: {path}")
    return checkpoint


def export_model_onnx(model: SequenceDeltaMLP, seq_len: int, input_dim: int, out_path: Path) -> None:
    dummy = torch.zeros(1, seq_len, input_dim, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        dynamo=False,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
    )


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_bundle(run_dir: Path, bundle_name: str) -> Path:
    checkpoint_path = run_dir / "model.pt"
    run_summary_path = run_dir / "run_summary.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = safe_torch_load(checkpoint_path)
    state_dict = checkpoint["model_state_dict"]
    seq_len = int(checkpoint.get("seq_len", 8))
    input_dim = int(checkpoint.get("input_dim", 126))
    num_classes = int(checkpoint.get("num_classes", len(DEFAULT_CLASS_NAMES)))
    if num_classes != len(DEFAULT_CLASS_NAMES):
        raise ValueError(
            f"Expected {len(DEFAULT_CLASS_NAMES)} classes for browser bundle, got {num_classes}"
        )

    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8")) if run_summary_path.exists() else {}
    hyper = dict(run_summary.get("hyperparameters") or {})
    verification = fingerprint_state_dict(state_dict)

    bundle_dir = FRONTEND_PUBLIC / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    model = SequenceDeltaMLP(seq_len=seq_len, input_dim=input_dim, num_classes=num_classes)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    onnx_path = bundle_dir / "model.onnx"
    export_model_onnx(model, seq_len, input_dim, onnx_path)
    # 현재 브라우저 로더가 같은 경로를 참조하므로 placeholder 파일을 함께 둡니다.
    (bundle_dir / "model.onnx.data").write_bytes(b"")

    bundle_checkpoint = {
        **checkpoint,
        "class_names": list(DEFAULT_CLASS_NAMES),
        "checkpoint_verification": verification,
    }
    torch.save(bundle_checkpoint, bundle_dir / "model.pt")

    shutil.copy2(RUNTIME_TEMPLATE / "feature_order.json", bundle_dir / "feature_order.json")
    input_spec = json.loads((RUNTIME_TEMPLATE / "input_spec.json").read_text(encoding="utf-8"))
    input_spec["default_tau"] = 0.85
    write_json(bundle_dir / "input_spec.json", input_spec)
    write_json(bundle_dir / "class_names.json", DEFAULT_CLASS_NAMES)

    config = {
        "bundle_id": bundle_name,
        "model_id": "mlp_sequence_delta",
        "mode": "sequence",
        "dataset_key": "pos_scale_grab",
        "normalization_family": "pos_scale",
        "input_type": "model_feature_sequence_joint63_delta63",
        "caller_input_type": "raw_mediapipe_landmark_frame_stream",
        "input_dim": input_dim,
        "seq_len": seq_len,
        "num_classes": num_classes,
        "neutral_index": 0,
        "default_tau": 0.85,
        "supported_backends": ["onnx-web"],
        "streaming_supported": True,
        "no_hand_resets_buffer": True,
        "class_names": list(DEFAULT_CLASS_NAMES),
        "checkpoint_fingerprint": verification["checkpoint_fingerprint"],
        "source_checkpoint_run_dir": str(run_dir),
        "focal_gamma": hyper.get("focal_gamma"),
        "preprocess": {
            "normalization": "pos_scale",
            "delta": True,
            "delta_order": "first",
            "seq_len": seq_len,
            "allowed_frame_shapes": [[63], [21, 3]],
            "formula": "(pts - pts[0]) / ||pts[9] - pts[0]||",
            "eps": 1e-8,
        },
    }
    write_json(bundle_dir / "config.json", config)

    return bundle_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="학습 run 체크포인트를 frontend runtime bundle로 export")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--bundle-name", type=str, default=DEFAULT_BUNDLE_NAME)
    args = parser.parse_args()

    bundle_dir = build_bundle(args.run_dir.resolve(), args.bundle_name)
    print(json.dumps({"bundle_dir": str(bundle_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
