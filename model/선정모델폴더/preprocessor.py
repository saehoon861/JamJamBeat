import pandas as pd
import numpy as np

def apply_downsampling(df: pd.DataFrame, target_ratio: str, margin_frames: int = 5) -> pd.DataFrame:
    """
    Downsamples Class 0 frames based on the specified target_ratio.
    Preserves all 0_neutral and 0_hardneg frames (100%) EXCEPT when target_ratio == "1:1".
    In the 1:1 scheme, 0_neutral and 0_hardneg data are proportional sampled using the same drop ratio 
    so that the final 0 count exactly matches the average 1~6 count without overwhelming the dataset.
    Excludes Class 0 frames near gesture transitions (0->1 or 1->0).
    """
    if target_ratio == "origin":
        return df.copy()

    # Find transition frames in the entire DataFrame to identify danger zones
    drop_indices = set()
    for _, group in df.groupby('source_file'):
        gestures = group['gesture'].values
        transitions = np.where(gestures[:-1] != gestures[1:])[0]
        
        for t_idx in transitions:
            start_idx = max(0, t_idx - margin_frames)
            end_idx = min(len(group), t_idx + 1 + margin_frames)
            original_indices = group.index[start_idx:end_idx].tolist()
            drop_indices.update(original_indices)

    # Phase 1: Separate unaffected data
    group_a_mask = df['gesture'] != 0
    group_a = df[group_a_mask]

    group_b_mask = (df['gesture'] == 0) & (df['source_file'].str.startswith('0_'))
    group_b = df[group_b_mask]

    group_c_mask = (df['gesture'] == 0) & (~df['source_file'].str.startswith('0_'))
    group_c = df[group_c_mask]

    # Phase 2: Exclude danger transitions from Group C
    c_drop_mask = group_c.index.isin(drop_indices)
    group_c_safe = group_c[~c_drop_mask]

    # Phase 3: Calculate target sizes
    class_counts = group_a['gesture'].value_counts()
    mean_count = class_counts.mean() if not class_counts.empty else 0
    
    if target_ratio == "4:1":
        target_total_0 = int(mean_count * 4)
    elif target_ratio == "1:1":
        target_total_0 = int(mean_count * 1)
    else:
        raise ValueError(f"Unknown target_ratio: {target_ratio}")
        
    num_b = len(group_b)
    num_c = len(group_c_safe)
    total_available_0 = num_b + num_c
    
    # 1:1 Special Proportional Downsampling
    if target_ratio == "1:1" and total_available_0 > target_total_0:
        # We need to drop frames, and since B is disproportionately large (3:1 to mean), 
        # we apply an equal survival fraction to both B and C to reach the global target.
        survival_fraction = target_total_0 / total_available_0
        
        target_b = int(num_b * survival_fraction)
        target_c = target_total_0 - target_b  # Remainder belongs to C
        
        group_b_final = group_b.sample(n=target_b, random_state=42)
        group_c_final = group_c_safe.sample(n=target_c, random_state=42)
    else:
        # Standard downsampling (4:1 or when we have fewer frames than needed)
        # Protect all of B 100%
        group_b_final = group_b
        
        remaining_quota = max(0, target_total_0 - len(group_b_final))
        if remaining_quota < len(group_c_safe):
            group_c_final = group_c_safe.sample(n=remaining_quota, random_state=42)
        else:
            group_c_final = group_c_safe

    # Phase 4: Concat and sort
    final_df = pd.concat([group_a, group_b_final, group_c_final])
    final_df = final_df.sort_index()

    return final_df
