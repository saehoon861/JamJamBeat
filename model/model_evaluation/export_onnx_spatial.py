#!/usr/bin/env python3
"""
export_onnx_spatial.py - Landmark_Spatial_Transformer 모델을 ONNX로 내보냅니다.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.onnx
import numpy as np

# --- 경로 설정 ---
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
MODEL_PIPELINES_DIR = PROJECT_ROOT / "model" / "model_pipelines"

if str(MODEL_PIPELINES_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_PIPELINES_DIR))

from checkpoint_verification import safe_torch_load, instantiate_model_from_state_dict, infer_num_classes_from_state_dict

def export_to_onnx(run_dir: Path, output_name: str = "model.onnx", opset_version: int = 12):
    run_dir = run_dir.resolve()
    checkpoint_path = run_dir / "model.pt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[Export] Loading checkpoint: {checkpoint_path}")
    
    # 1. 모델 로드
    device = torch.device("cpu")
    checkpoint = safe_torch_load(checkpoint_path, device)
    
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        class_names = checkpoint.get("class_names", [])
        model_id = checkpoint.get("model_id", "Landmark_Spatial_Transformer")
    else:
        state_dict = checkpoint
        class_names = []
        model_id = "Landmark_Spatial_Transformer"

    if not class_names:
        num_classes = infer_num_classes_from_state_dict(state_dict)
        class_names = [str(i) for i in range(num_classes)]
    else:
        num_classes = len(class_names)

    print(f"[Export] Model ID: {model_id}, Classes: {num_classes}")

    # 모델 인스턴스화 (Landmark_Spatial_Transformer는 frame 모델이므로 seq_len_hint=1)
    model, seq_len, input_dim, _ = instantiate_model_from_state_dict(
        model_id, state_dict, num_classes, seq_len_hint=1 
    )
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    # 2. 더미 입력 생성
    # Landmark_Spatial_Transformer는 (Batch, 21, 3) 입력을 기대함
    batch_size = 1
    num_landmarks = 21
    coord_dim = 3
    dummy_input = torch.randn(batch_size, num_landmarks, coord_dim, dtype=torch.float32)

    # 3. ONNX Export
    output_path = run_dir / output_name
    
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }

    print(f"[Export] Exporting to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )
    
    print(f"[Export] ONNX model saved size: {output_path.stat().st_size / 1024:.2f} KB")

    # 4. 메타데이터 저장 (Frontend용)
    class_names_path = run_dir / "class_names.json"
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2, ensure_ascii=False)

    # 학습모델_웹배포_정리.md 반영: pos_scale 전처리 명시
    config = {
        "model_id": model_id,
        "type": "spatial_transformer",
        "input_shape": [21, 3],
        "input_desc": "Normalized landmarks (pos_scale applied), shape (21, 3)",
        "classes": class_names,
        "preprocessing": "pos_scale",
        "normalization": {
            "origin_idx": 0,       # wrist
            "scale_idxs": [0, 9]   # wrist to middle_mcp
        }
    }
    config_path = run_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"[Export] Metadata saved: class_names.json, config.json")

def main():
    parser = argparse.ArgumentParser(description="Export Landmark_Spatial_Transformer to ONNX")
    parser.add_argument("--run-dir", type=str, required=True, help="Directory containing model.pt")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output ONNX filename")
    args = parser.parse_args()
    export_to_onnx(Path(args.run_dir), args.output)

if __name__ == "__main__":
    main()