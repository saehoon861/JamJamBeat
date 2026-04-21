"""
augmentor.py — 결합 증강 벡터 연산 모듈
=========================================
전처리 및 정규화 완료 데이터에 대해 아래 3단계 증강을 순차 적용:
  1) Mirroring (50% 확률)
  2) BLP — Bone Length Perturbation (100%)
  3) Gaussian Noise (100%)

모든 함수는 **전 클래스(0 포함)**에 동일하게 적용되며,
원본 DataFrame을 직접 수정(in-place)하지 않고 copy본 위에서 작업한다.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# config 임포트를 위한 경로 추가
_offline_dir = Path(__file__).resolve().parent.parent
if str(_offline_dir) not in sys.path:
    sys.path.insert(0, str(_offline_dir))

import config


# ──────────────────────────────────────────────
# 유틸: 좌표 컬럼 리스트 생성 헬퍼
# ──────────────────────────────────────────────
def _get_coord_cols():
    """x0,y0,z0 ~ x20,y20,z20 순서의 좌표 컬럼 이름 리스트 반환 (총 63개)"""
    cols = []
    for i in range(21):
        cols.extend([f'x{i}', f'y{i}', f'z{i}'])
    return cols


def _get_x_cols():
    """x좌표 컬럼만 반환 (x0 ~ x20, 총 21개)"""
    return [f'x{i}' for i in range(21)]


# ──────────────────────────────────────────────
# 1단계: 좌우 반전 (Mirroring)
# ──────────────────────────────────────────────
def apply_mirroring(df: pd.DataFrame, prob: float) -> np.ndarray:
    """
    x좌표를 부호 반전하여 좌우 미러링을 수행한다.
    - 전체 샘플 중 prob 확률(기본 50%)에 해당하는 행만 반전 적용.
    - z축(깊이)은 그대로 보존하여 3D 구조적 무결성 유지.

    Parameters
    ----------
    df : pd.DataFrame
        증강 대상 DataFrame (in-place 수정됨).
    prob : float
        미러링이 적용될 확률 (0.0 ~ 1.0).

    Returns
    -------
    mirror_mask : np.ndarray (bool)
        각 행에 미러링이 적용되었는지를 나타내는 불리언 배열.
        → aug_mirror 컬럼 기입용.
    """
    n = len(df)
    # 샘플별 독립적인 확률 마스크 생성
    mirror_mask = np.random.rand(n) < prob

    # x좌표 컬럼만 선택하여 마스크된 행의 부호 반전
    x_cols = _get_x_cols()
    df.loc[mirror_mask, x_cols] = df.loc[mirror_mask, x_cols] * -1

    return mirror_mask


# ──────────────────────────────────────────────
# 2단계: 뼈 길이 섭동 (Bone Length Perturbation)
# ──────────────────────────────────────────────
def apply_blp(df: pd.DataFrame, blp_scales: dict) -> None:
    """
    Kinematic Chain 계층적 스케일링으로 손가락 뼈 길이를 축소한다.
    - 정규화 기준축(손목(0) ~ 중지MCP(9))은 절대 건드리지 않음.
    - 각 손가락의 MCP를 원점으로, 하위 마디(PIP→DIP→Tip)를 연쇄적으로 축소.
    - 수식: 자식_new = 부모_old + (자식_old - 부모_old) × scale

    Parameters
    ----------
    df : pd.DataFrame
        증강 대상 DataFrame (in-place 수정됨).
    blp_scales : dict
        마디별 축소 범위. {"proximal": (min, max), "middle": ..., "distal": ...}
    """
    n = len(df)

    # 마디 레벨 이름과 스케일 범위를 순서대로 매핑
    # 각 FINGER_CHAIN = [root, proximal_child, middle_child, distal_child]
    # 연산 대상 엣지: root→proximal_child (proximal), proximal_child→middle_child (middle), middle_child→distal_child (distal)
    level_names = ["proximal", "middle", "distal"]

    for chain in config.FINGER_CHAINS:
        # chain 예시: [5, 6, 7, 8] → 엣지: 5→6(proximal), 6→7(middle), 7→8(distal)
        for edge_idx, level_name in enumerate(level_names):
            parent_idx = chain[edge_idx]
            child_idx = chain[edge_idx + 1]

            # 부모/자식 좌표 컬럼명
            parent_cols = [f'x{parent_idx}', f'y{parent_idx}', f'z{parent_idx}']
            child_cols = [f'x{child_idx}', f'y{child_idx}', f'z{child_idx}']

            # 샘플마다 독립적인 스케일 계수 벡터 생성 (축소만: 범위가 0.8~0.98)
            scale_range = blp_scales[level_name]
            scales = np.random.uniform(scale_range[0], scale_range[1], size=n)

            # Kinematic Chain 수식 적용: 자식_new = 부모 + (자식 - 부모) × scale
            # 부모 좌표는 이전 루프에서 이미 업데이트된 상태 (연쇄 전파)
            parent_vals = df[parent_cols].values  # shape: (n, 3)
            child_vals = df[child_cols].values    # shape: (n, 3)

            # scales를 (n, 1)로 reshape하여 xyz 3축에 동일 적용
            new_child = parent_vals + (child_vals - parent_vals) * scales[:, np.newaxis]
            df[child_cols] = new_child


# ──────────────────────────────────────────────
# 3단계: 가우시안 노이즈
# ──────────────────────────────────────────────
def apply_gaussian_noise(df: pd.DataFrame, sigma_range: tuple) -> np.ndarray:
    """
    좌표에 가우시안 노이즈를 추가하여 센서 측정 오차에 대한 강건성을 확보한다.
    - 손목 원점 컬럼(x0, y0, z0)에는 노이즈를 강제로 0으로 마스킹하여 원점 고정.
    - 각 샘플마다 독립적인 σ를 sigma_range 범위에서 균일 추출.

    Parameters
    ----------
    df : pd.DataFrame
        증강 대상 DataFrame (in-place 수정됨).
    sigma_range : tuple
        (σ_min, σ_max) — 샘플별 σ 균일 분포 추출 범위.

    Returns
    -------
    sigmas : np.ndarray (float)
        각 샘플에 실제 적용된 σ 값 배열.
        → aug_noise_sigma 컬럼 기입용.
    """
    n = len(df)
    coord_cols = _get_coord_cols()

    # 샘플마다 독립적인 σ 추출
    sigmas = np.random.uniform(sigma_range[0], sigma_range[1], size=n)

    # 노이즈 행렬 생성: (n, 63) — 각 행마다 해당 행의 σ로 정규분포 샘플링
    noise_matrix = np.random.randn(n, len(coord_cols)) * sigmas[:, np.newaxis]

    # ★ 방어로직: 손목 원점(x0, y0, z0) 컬럼의 노이즈를 0으로 강제 마스킹
    # coord_cols에서 x0, y0, z0은 인덱스 0, 1, 2에 위치
    noise_matrix[:, 0] = 0.0  # x0
    noise_matrix[:, 1] = 0.0  # y0
    noise_matrix[:, 2] = 0.0  # z0

    # 노이즈 적용
    df[coord_cols] = df[coord_cols].values + noise_matrix

    return sigmas
