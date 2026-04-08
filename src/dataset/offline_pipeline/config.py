from pathlib import Path
import numpy as np

# --- 경로 관리 ---
ROOT = Path(__file__).resolve().parents[3]  # /home/user/JamJamBeat

DATA_DIR = ROOT / "data"
TOTAL_DIR = DATA_DIR / "total_data"
PROCESSED_DIR = DATA_DIR / "processed_scenarios"
DIR_AUGMENTED = DATA_DIR / "augmented_scenarios"  # 증강 결과물 출력 디렉토리

# --- 증강 파이프라인 파라미터 ---
AUG_RANDOM_SEED = 42  # 재현성을 위한 글로벌 난수 시드

AUG_PARAMS = {
    # 증강 기법별 적용 확률 (결합 파이프라인: Mirror → BLP → Noise)
    "prob": {
        "mirror": 0.5,          # 50% 확률로 좌우 반전
        "blp": 1.0,             # 100% 전량 뼈 길이 축소 적용
        "gaussian_noise": 1.0,  # 100% 전량 위치 노이즈 적용
    },
    # 노이즈 σ 범위 (파일명 기반 분기)
    "noise_sigma_range": {
        "non_scale": (0.003, 0.005),  # scale 미포함 파일
        "scale":     (0.020, 0.030),  # scale 포함 파일
    },
    # BLP 마디별 축소 스케일 범위 (축소만 허용, 확대 배제)
    "blp_scales": {
        "proximal": (0.90, 0.98),  # MCP → PIP
        "middle":   (0.85, 0.93),  # PIP → DIP
        "distal":   (0.80, 0.88),  # DIP → Tip
    }
}

# 손가락 Kinematic Chain (계층적 BLP 연산용 트리 구조)
FINGER_CHAINS = [
    [1, 2, 3, 4],     # 엄지 (CMC → MCP → IP → Tip)
    [5, 6, 7, 8],     # 검지 (MCP → PIP → DIP → Tip)
    [9, 10, 11, 12],  # 중지
    [13, 14, 15, 16], # 약지
    [17, 18, 19, 20]  # 소지
]

# --- 파라미터 ---
MARGIN_FRAMES_DROP = 2      # 정수형이면 frame+seq 모드같이 돌림, 
                            # None이면 seq 모드만 돌림
MARGIN_FRAMES_COLLECT = 8   # 시퀀스 길이에 맞춤
# MARGIN_FRAMES = 5           # 전이 구간 근방의 위험군 프레임 배제 마진

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
