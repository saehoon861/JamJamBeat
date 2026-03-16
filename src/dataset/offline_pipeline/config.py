from pathlib import Path

# --- 경로 관리 ---
ROOT = Path(__file__).resolve().parents[3]  # /home/user/JamJamBeat

DATA_DIR = ROOT / "data"
TOTAL_DIR = DATA_DIR / "total_data"
PROCESSED_DIR = DATA_DIR / "processed_scenarios"

# --- 파라미터 ---
MARGIN_FRAMES = 5  # 전이 구간 근방의 위험군 프레임 배제 마진

# --- MediaPipe 손 랜드마크 구조 매핑 ---
LANDMARK_IDX = {
    "wrist": 0,
    "thumb_cmc": 1, "thumb_mcp": 2, "thumb_ip": 3, "thumb_tip": 4,
    "index_mcp": 5, "index_pip": 6, "index_dip": 7, "index_tip": 8,
    "middle_mcp": 9, "middle_pip": 10, "middle_dip": 11, "middle_tip": 12,
    "ring_mcp": 13, "ring_pip": 14, "ring_dip": 15, "ring_tip": 16,
    "pinky_mcp": 17, "pinky_pip": 18, "pinky_dip": 19, "pinky_tip": 20,
}

# --- 시나리오 설정 (총 12개 조합: 비율 3 x 위치 정규화 2 x 스케일 정규화 2) ---
SCENARIOS = {
    # 1. 아무것도 안 함 (X / X)
    "baseline": { "downsample_ratio": "origin", "origin": None, "scale": None, "augment": False },
    "ds_4_none": { "downsample_ratio": "4:1", "origin": None, "scale": None, "augment": False },
    "ds_1_none": { "downsample_ratio": "1:1", "origin": None, "scale": None, "augment": False },
    
    # 2. 위치만 (O / X)
    "pos_only": { "downsample_ratio": "origin", "origin": "wrist", "scale": None, "augment": False },
    "ds_4_pos": { "downsample_ratio": "4:1", "origin": "wrist", "scale": None, "augment": False },
    "ds_1_pos": { "downsample_ratio": "1:1", "origin": "wrist", "scale": None, "augment": False },
    
    # 3. 스케일만 (X / O)
    "scale_only": { "downsample_ratio": "origin", "origin": None, "scale": ["wrist", "middle_mcp"], "augment": False },
    "ds_4_scale": { "downsample_ratio": "4:1", "origin": None, "scale": ["wrist", "middle_mcp"], "augment": False },
    "ds_1_scale": { "downsample_ratio": "1:1", "origin": None, "scale": ["wrist", "middle_mcp"], "augment": False },
    
    # 4. 위치 + 스케일 (O / O)
    "pos_scale": { "downsample_ratio": "origin", "origin": "wrist", "scale": ["wrist", "middle_mcp"], "augment": False },
    "ds_4_pos_scale": { "downsample_ratio": "4:1", "origin": "wrist", "scale": ["wrist", "middle_mcp"], "augment": False },
    "ds_1_pos_scale": { "downsample_ratio": "1:1", "origin": "wrist", "scale": ["wrist", "middle_mcp"], "augment": False },
}
