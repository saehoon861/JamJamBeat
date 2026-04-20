import numpy as np

def apply_position_normalization(landmarks_3d: np.ndarray, origin_idx: int) -> np.ndarray:
    """
    Translates the landmarks so that the landmark at origin_idx becomes (0, 0, 0).
    landmarks_3d: shape (N, 21, 3)
    Returns: shape (N, 21, 3)
    """
    return landmarks_3d - landmarks_3d[:, origin_idx:origin_idx+1, :]

def apply_distance_normalization(landmarks_3d: np.ndarray, origin_idx: int, scale_idx_list: list) -> np.ndarray:
    """
    Scales the landmarks so that the Euclidean distance between origin_idx and a representative scale_idx is 1.0.
    Includes unconditional defensive coding to prevent division by zero.
    landmarks_3d: shape (N, 21, 3)
    Returns: shape (N, 21, 3)
    """
    if not scale_idx_list:
        return landmarks_3d
        
    scale_idx = scale_idx_list[0] if len(scale_idx_list) == 1 else scale_idx_list[1]
    
    # Calculate Euclidean distance: shape (N,)
    dist = np.linalg.norm(landmarks_3d[:, scale_idx, :] - landmarks_3d[:, origin_idx, :], axis=1)
    
    # Unconditional defensive coding to prevent NaN/Inf 
    safe_dist = np.maximum(dist, 1e-6)
    
    # Scale adjustment: (N, 21, 3) / (N, 1, 1)
    return landmarks_3d / safe_dist[:, np.newaxis, np.newaxis]
