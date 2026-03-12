import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure offline_pipeline is in sys.path
current_dir = Path(__file__).resolve().parent
offline_dir = current_dir.parent
if str(offline_dir) not in sys.path:
    sys.path.insert(0, str(offline_dir))

import config
from modules import apply_downsampling, apply_position_normalization, apply_distance_normalization

def main():
    # 1. Load the total dataset into memory
    total_csv_path = config.TOTAL_DIR / "total_data.csv"
    if not total_csv_path.exists():
        csvs = list(config.TOTAL_DIR.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in {config.TOTAL_DIR}")
        total_csv_path = csvs[0]
        
    print(f"Loading raw data from {total_csv_path}...")
    df_raw = pd.read_csv(total_csv_path)

    coord_cols = []
    for i in range(21):
        coord_cols.extend([f'x{i}', f'y{i}', f'z{i}'])
        
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Iterate through all 12 configured scenarios
    for scenario_name, params in config.SCENARIOS.items():
        print(f"\n--- Processing Scenario: {scenario_name} ---")
        
        # 3. String to Index Mappings
        origin_str = params.get("origin")
        scale_strs = params.get("scale")
        
        # None -> None, String -> Int, List[String] -> List[Int]
        origin_idx = config.LANDMARK_IDX[origin_str] if origin_str else None
        
        if isinstance(scale_strs, list):
            scale_idx_list = [config.LANDMARK_IDX[s] for s in scale_strs]
        elif scale_strs is not None:
            scale_idx_list = [config.LANDMARK_IDX[scale_strs]]
        else:
            scale_idx_list = None
            
        # 4. Downsampling Application
        ratio = params.get("downsample_ratio", "origin")
        df_processed = apply_downsampling(
            df=df_raw, 
            target_ratio=ratio, 
            margin_frames=config.MARGIN_FRAMES
        )
        print(f"[{scenario_name}] Downsampling ({ratio}): {len(df_raw)} -> {len(df_processed)} rows")
        
        # 5. Extract 3D Tensors
        landmarks_3d = df_processed[coord_cols].to_numpy().reshape(-1, 21, 3)
        
        # 6. Apply sequence of requested normalizations based on the current scenario
        # 6-1. Position Normalization (If origin is requested)
        if origin_idx is not None:
            landmarks_3d = apply_position_normalization(landmarks_3d, origin_idx)
            
        # 6-2. Distance (Scale) Normalization (If scales are requested)
        if scale_idx_list is not None and origin_idx is not None:
            landmarks_3d = apply_distance_normalization(landmarks_3d, origin_idx, scale_idx_list)
        elif scale_idx_list is not None and origin_idx is None:
            # Plan implies scaling needs origin distance, if origin is None, assume wrist for distance anchor
            landmarks_3d = apply_distance_normalization(landmarks_3d, config.LANDMARK_IDX["wrist"], scale_idx_list)
        
        # 7. Masking back to Pandas DataFrame inplace
        df_processed.loc[:, coord_cols] = landmarks_3d.reshape(-1, 63)
        
        # 8. Save output
        out_path = config.PROCESSED_DIR / f"{scenario_name}.csv"
        df_processed.to_csv(out_path, index=False)
        print(f"[{scenario_name}] Saved successfully to {out_path.name}")

if __name__ == "__main__":
    main()
