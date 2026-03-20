import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import wandb
import argparse

# Ensure offline_pipeline is in sys.path
current_dir = Path(__file__).resolve().parent
offline_dir = current_dir.parent
if str(offline_dir) not in sys.path:
    sys.path.insert(0, str(offline_dir))

import config
from modules import apply_downsampling, apply_position_normalization, apply_distance_normalization

def main():
    # 1. 실행할 모드 arg로 받기
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["seq", "frame", "both"],
        default="both",
        help="seq: 시퀀스 모드만 | frame: 프레임 단위 모드만 | both: 둘 다 (default)"
    )
    args = parser.parse_args()

    # MARGIN_FRAMES_DROP=None인데 frame 모드 요청 시 경고 후 seq로 fallback
    if config.MARGIN_FRAMES_DROP is None and args.mode in ("frame", "both"):
        print("⚠️  MARGIN_FRAMES_DROP이 None이라 frame 모드 실행 불가. seq 모드만 실행합니다.")
        args.mode = "seq"

    if args.mode == "seq":
        modes = [None]
    elif args.mode == "frame":
        modes = [config.MARGIN_FRAMES_DROP]
    else:
        modes = [None, config.MARGIN_FRAMES_DROP]

    # 2. Load the total dataset into memory
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

    # 3. 시나리오 × 모드 이중 루프
    for scenario_name, params in config.SCENARIOS.items():
        for margin_drop in modes:
            mode_suffix = "seq" if margin_drop is None else "frame"
            run_name = f"{scenario_name}_{mode_suffix}"
            print(f"\n--- Processing Scenario: {run_name} ---")

            with wandb.init(
                project="JamJamBeat",
                job_type="preprocess",
                name=run_name,
                config={**params, "mode": mode_suffix},
            ) as run:

                # 4. String to Index Mappings
                origin_str = params.get("origin")
                scale_strs = params.get("scale")
                
                origin_idx = config.LANDMARK_IDX[origin_str] if origin_str else None
                
                if isinstance(scale_strs, list):
                    scale_idx_list = [config.LANDMARK_IDX[s] for s in scale_strs]
                elif scale_strs is not None:
                    scale_idx_list = [config.LANDMARK_IDX[scale_strs]]
                else:
                    scale_idx_list = None
                    
                # 5. Downsampling Application
                ratio = params.get("downsample_ratio", "origin")
                df_processed = apply_downsampling(
                    df=df_raw, 
                    target_ratio=ratio, 
                    margin_drop=margin_drop,                  # ← 모드별로 다르게
                    margin_collect=config.MARGIN_FRAMES_COLLECT,
                )
                print(f"[{run_name}] Downsampling ({ratio}): {len(df_raw)} -> {len(df_processed)} rows")
                
                # 6. Extract 3D Tensors
                landmarks_3d = df_processed[coord_cols].to_numpy().reshape(-1, 21, 3)
                
                # 7. Apply normalizations
                if origin_idx is not None:
                    landmarks_3d = apply_position_normalization(landmarks_3d, origin_idx)
                    
                if scale_idx_list is not None and origin_idx is not None:
                    landmarks_3d = apply_distance_normalization(landmarks_3d, origin_idx, scale_idx_list)
                elif scale_idx_list is not None and origin_idx is None:
                    landmarks_3d = apply_distance_normalization(landmarks_3d, config.LANDMARK_IDX["wrist"], scale_idx_list)
                
                # 8. Masking back to Pandas DataFrame inplace
                df_processed.loc[:, coord_cols] = landmarks_3d.reshape(-1, 63)
                
                # 9. Save output
                out_path = config.PROCESSED_DIR / f"{run_name}.csv"
                df_processed.to_csv(out_path, index=False)
                print(f"[{run_name}] Saved to {out_path.name}")

                # 10. W&B Artifact
                artifact = wandb.Artifact(
                    name=run_name,
                    type="dataset",
                    description=(
                        f"downsample={params['downsample_ratio']} | "
                        f"position_norm={params['origin']} | "
                        f"scale_norm={params['scale']} | "
                        f"mode={mode_suffix}"
                    ),
                    metadata={
                        "downsample_ratio": params["downsample_ratio"],
                        "position_norm":    params["origin"],
                        "scale_norm":       params["scale"],
                        "mode":             mode_suffix,
                        "row_count":        len(df_processed),
                    },
                )
                artifact.add_file(str(out_path))
                run.log_artifact(artifact)
                print(f"[{run_name}] W&B Artifact 업로드 완료 ✅")

if __name__ == "__main__":
    main()