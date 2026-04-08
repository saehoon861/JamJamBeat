import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure offline_pipeline is in sys.path
current_dir = Path(__file__).resolve().parent
offline_dir = current_dir.parent
if str(offline_dir) not in sys.path:
    sys.path.insert(0, str(offline_dir))

import config
from modules.normalizer import apply_position_normalization, apply_distance_normalization

# ========== 정규화 옵션별 시나리오 정의 ==========
NORMALIZE_SCENARIOS = {
    "baseline": {
        "description": "정규화 없음 (원본)",
        "apply_position": False,
        "apply_distance": False,
    },
    "pos_only": {
        "description": "위치 정규화만 (wrist 기준)",
        "apply_position": True,
        "apply_distance": False,
        "origin_idx": 0,  # wrist
    },
    "scale_only": {
        "description": "거리 정규화만 (wrist-middle_mcp 기준)",
        "apply_position": False,
        "apply_distance": True,
        "origin_idx": 0,  # wrist
        "scale_idx_list": [0, 9],  # wrist, middle_mcp
    },
    "pos_scale": {
        "description": "위치 + 거리 정규화 둘 다",
        "apply_position": True,
        "apply_distance": True,
        "origin_idx": 0,  # wrist
        "scale_idx_list": [0, 9],  # wrist, middle_mcp
    },
}


def normalize_landmarks(df: pd.DataFrame, scenario_name: str, scenario_params: dict) -> pd.DataFrame:
    """
    Normalize landmarks according to the scenario.
    
    Args:
        df: DataFrame with 'source_file', 'frame_idx', 'timestamp', 'gesture', and x0-z20 columns
        scenario_name: name of the scenario
        scenario_params: parameters for the scenario
    
    Returns:
        DataFrame with normalized landmarks
    """
    print(f"\n  Processing scenario: {scenario_name}")
    print(f"    Description: {scenario_params['description']}")
    
    df_result = df.copy()
    
    # Extract coordinate columns
    coord_cols = []
    for i in range(21):
        coord_cols.extend([f'x{i}', f'y{i}', f'z{i}'])
    
    # Reshape to (N, 21, 3)
    landmarks = df_result[coord_cols].values.reshape(-1, 21, 3)
    
    # Apply position normalization if requested
    if scenario_params.get("apply_position", False):
        origin_idx = scenario_params["origin_idx"]
        print(f"    Applying position normalization (origin: landmark {origin_idx})")
        landmarks = apply_position_normalization(landmarks, origin_idx)
    
    # Apply distance normalization if requested
    if scenario_params.get("apply_distance", False):
        origin_idx = scenario_params["origin_idx"]
        scale_idx_list = scenario_params["scale_idx_list"]
        print(f"    Applying distance normalization (origin: {origin_idx}, scale: {scale_idx_list})")
        landmarks = apply_distance_normalization(landmarks, origin_idx, scale_idx_list)
    
    # Reshape back to (N, 63) and update dataframe
    df_result[coord_cols] = landmarks.reshape(-1, 63)
    
    return df_result


def main():
    # Load raw data
    total_csv_path = config.TOTAL_DIR / "total_data_test.csv"
    if not total_csv_path.exists():
        raise FileNotFoundError(f"File not found: {total_csv_path}")
    
    print(f"Loading raw data from {total_csv_path}...")
    df_raw = pd.read_csv(total_csv_path)
    print(f"  Loaded {len(df_raw)} rows")
    
    # Create output directory if needed
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each scenario
    for scenario_name, scenario_params in NORMALIZE_SCENARIOS.items():
        print(f"\n{'='*60}")
        print(f"Processing: {scenario_name}")
        print(f"{'='*60}")
        
        # Normalize landmarks
        df_normalized = normalize_landmarks(df_raw, scenario_name, scenario_params)
        
        # Save to processed_scenarios directory
        output_filename = f"total_data_test_{scenario_name}.csv"
        output_path = config.PROCESSED_DIR / output_filename
        
        df_normalized.to_csv(output_path, index=False)
        print(f"  ✓ Saved to: {output_path}")
        print(f"    Rows: {len(df_normalized)}")
    
    print(f"\n{'='*60}")
    print(f"✓ All scenarios processed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
