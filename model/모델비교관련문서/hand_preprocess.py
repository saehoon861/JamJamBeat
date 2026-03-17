"""
Hand Gesture 데이터 전처리 모듈
================================
MediaPipe 손 랜드마크 21점 CSV → 정규화 + Bone/Angle 피처 추출

입력 CSV 형식:
  source_file, frame_idx, timestamp, gesture, x0,y0,z0, ..., x20,y20,z20

출력 CSV 형식:
  source_file, frame_idx, timestamp, gesture,
  nx0,ny0,nz0,...,nx20,ny20,nz20,          (정규화 좌표 63개)
  bx0,by0,bz0,bl0,...,bx20,by20,bz20,bl20, (bone vector+length 84개)
  flex_thumb,...,flex_pinky,                 (flexion 5개)
  abd_th_idx,...,abd_ring_pinky              (abduction 4개)

사용법:
  python hand_gesture_preprocess.py input.csv output.csv
"""

import numpy as np
import pandas as pd
import sys
from typing import Tuple


# ============================================================================
# 1. 상수 정의
# ============================================================================

HAND_CONNECTIONS = [
    # Palm (6)
    (0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
    # Thumb (3)
    (1, 2), (2, 3), (3, 4),
    # Index (3)
    (5, 6), (6, 7), (7, 8),
    # Middle (3)
    (9, 10), (10, 11), (11, 12),
    # Ring (3)
    (13, 14), (14, 15), (15, 16),
    # Pinky (3)
    (17, 18), (18, 19), (19, 20),
]

FINGER_CHAINS = {
    "thumb":  [0, 1, 2, 3, 4],
    "index":  [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring":   [0, 13, 14, 15, 16],
    "pinky":  [0, 17, 18, 19, 20],
}

FINGER_PAIRS = [
    ("thumb", "index"),
    ("index", "middle"),
    ("middle", "ring"),
    ("ring", "pinky"),
]

# 원본 좌표 컬럼명
RAW_COORD_COLS = []
for i in range(21):
    RAW_COORD_COLS.extend([f"x{i}", f"y{i}", f"z{i}"])

# 메타데이터 컬럼
META_COLS = ["source_file", "frame_idx", "timestamp", "gesture"]


# ============================================================================
# 2. 정규화 (Translation → Scale → Rotation)
# ============================================================================

def normalize_landmarks(pts: np.ndarray) -> np.ndarray:
    """
    21×3 랜드마크 정규화.

    1) Translation: index(5)/middle(9)/pinky(17) knuckle 평균을 원점으로
    2) Scale: center에서 가장 먼 knuckle까지 거리로 나눔
    3) Rotation: alignment vector를 기준축으로 2D 회전 정렬

    Args:
        pts: (21, 3) 원본 좌표
    Returns:
        (21, 3) 정규화 좌표
    """
    pts = pts.copy()

    # --- Translation ---
    center = pts[[5, 9, 17]].mean(axis=0)
    pts -= center

    # --- Scale ---
    knuckle_dists = np.linalg.norm(pts[[1, 5, 9, 13, 17]], axis=1)
    max_dist = knuckle_dists.max()
    if max_dist > 1e-8:
        pts /= max_dist

    # --- Rotation (xy 평면) ---
    # alignment = (middle_knuckle→wrist) + (index_knuckle→pinky_knuckle)
    alignment = (pts[0] - pts[9]) + (pts[17] - pts[5])
    norm_xy = np.linalg.norm(alignment[:2])
    if norm_xy > 1e-8:
        cos_t = alignment[1] / norm_xy
        sin_t = alignment[0] / norm_xy
        rot = np.array([
            [ cos_t, sin_t, 0],
            [-sin_t, cos_t, 0],
            [     0,     0, 1],
        ])
        pts = pts @ rot.T

    return pts


# ============================================================================
# 3. Bone 피처 (21 edges × 4 = 84차원)
# ============================================================================

def compute_bone_features(pts: np.ndarray) -> np.ndarray:
    """
    HAND_CONNECTIONS 각 엣지에 대해 bone vector(3) + length(1) 계산.

    Returns:
        (21, 4) — [dx, dy, dz, length] per edge
    """
    bones = np.zeros((len(HAND_CONNECTIONS), 4), dtype=np.float32)
    for i, (u, v) in enumerate(HAND_CONNECTIONS):
        diff = pts[v] - pts[u]
        bones[i, :3] = diff
        bones[i, 3] = np.linalg.norm(diff)
    return bones


# ============================================================================
# 4. Angle 피처 (5 flexion + 4 abduction = 9차원)
# ============================================================================

def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """두 벡터 사이 각도(radian)"""
    cos_val = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.arccos(np.clip(cos_val, -1.0, 1.0))


def compute_angle_features(pts: np.ndarray) -> np.ndarray:
    """
    손가락 굽힘(flexion) + 인접 손가락 벌어짐(abduction) 각도.

    - Flexion: 각 손가락 체인의 첫 segment와 나머지 segments 사이 최대 각도
    - Abduction: 인접 손가락 base→tip 방향 사이 각도

    Returns:
        (9,) — [flex×5, abd×4]
    """
    angles = []

    # Flexion (5)
    for chain in FINGER_CHAINS.values():
        segs = np.diff(pts[chain], axis=0)
        if len(segs) < 2:
            angles.append(0.0)
            continue
        max_ang = max(_angle_between(segs[0], s) for s in segs[1:])
        angles.append(max_ang)

    # Abduction (4)
    for name_a, name_b in FINGER_PAIRS:
        ca, cb = FINGER_CHAINS[name_a], FINGER_CHAINS[name_b]
        dir_a = pts[ca[-1]] - pts[ca[1]]
        dir_b = pts[cb[-1]] - pts[cb[1]]
        angles.append(_angle_between(dir_a, dir_b))

    return np.array(angles, dtype=np.float32)


# ============================================================================
# 5. 단일 프레임 전체 피처 추출
# ============================================================================

def extract_features(landmarks_21x3: np.ndarray) -> dict:
    """
    원본 (21,3) → 정규화 → joint/bone/angle 피처 딕셔너리.

    Returns:
        {
            'joint': (63,),  정규화 좌표 flatten
            'bone':  (84,),  bone vector+length flatten
            'angle': (9,),   flexion+abduction
        }
    """
    # run_pipeline.py가 기대하는 입력 계약은 결국 이 세 feature 묶음에 의해 결정된다.
    norm = normalize_landmarks(landmarks_21x3)
    return {
        "joint": norm.flatten().astype(np.float32),
        "bone":  compute_bone_features(norm).flatten().astype(np.float32),
        "angle": compute_angle_features(norm).astype(np.float32),
    }


# ============================================================================
# 6. CSV 전체 처리
# ============================================================================

def build_output_columns() -> list:
    """출력 CSV 컬럼명 생성"""
    cols = list(META_COLS)

    # 정규화 좌표 (63)
    for i in range(21):
        cols.extend([f"nx{i}", f"ny{i}", f"nz{i}"])

    # Bone (84)
    for i in range(len(HAND_CONNECTIONS)):
        cols.extend([f"bx{i}", f"by{i}", f"bz{i}", f"bl{i}"])

    # Angle (9)
    for name in FINGER_CHAINS:
        cols.append(f"flex_{name}")
    for na, nb in FINGER_PAIRS:
        cols.append(f"abd_{na[:3]}_{nb[:3]}")

    # 최종 컬럼 순서는 학습 / 평가 / 런타임 feature slicing과 일치해야 한다.
    return cols


def preprocess_csv(
    input_path: str,
    output_path: str,
    drop_missing: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    CSV 전처리 메인 함수.

    1) 원본 CSV 로드
    2) 결측(NaN) 프레임 처리 (제거 or 유지)
    3) 프레임별 정규화 + bone/angle 피처 추출
    4) 결과 CSV 저장

    Args:
        input_path: 입력 CSV 경로
        output_path: 출력 CSV 경로
        drop_missing: True면 결측 프레임 제거, False면 NaN 행 유지(피처도 NaN)
        verbose: 진행 상황 출력
    """
    df = pd.read_csv(input_path)

    if verbose:
        print(f"원본 로드: {len(df)}행")
        print(f"클래스 분포:\n{df['gesture'].value_counts().sort_index().to_string()}")

    # 결측 판별
    coord_values = df[RAW_COORD_COLS].values
    has_landmark = ~np.isnan(coord_values).any(axis=1)

    if verbose:
        n_valid = has_landmark.sum()
        n_missing = (~has_landmark).sum()
        print(f"\n랜드마크 유효: {n_valid}행, 결측(미검출): {n_missing}행")

    # 출력 준비
    out_cols = build_output_columns()
    out_rows = []
    skipped = 0

    for idx in range(len(df)):
        row = df.iloc[idx]

        if not has_landmark[idx]:
            if drop_missing:
                skipped += 1
                continue
            else:
                # 후속 분석에서 결측 프레임 위치가 필요하면 메타는 유지하고 feature만 비운다.
                meta = [row.get(c, np.nan) for c in META_COLS]
                feat_nan = [np.nan] * (len(out_cols) - len(META_COLS))
                out_rows.append(meta + feat_nan)
                continue

        # 좌표 추출 → (21, 3)
        coords = coord_values[idx].astype(np.float32).reshape(21, 3)

        # 이 단계에서 63 + 84 + 9 구조가 고정되며 이후 모든 모델 비교 실험이 이 포맷을 전제로 한다.
        feats = extract_features(coords)

        meta = [row.get(c, np.nan) for c in META_COLS]
        feat_values = np.concatenate([
            feats["joint"],   # 63
            feats["bone"],    # 84
            feats["angle"],   # 9
        ]).tolist()

        out_rows.append(meta + feat_values)

    result_df = pd.DataFrame(out_rows, columns=out_cols)

    if verbose:
        print(f"\n결과: {len(result_df)}행 (제거: {skipped}행)")
        print(f"피처 차원: joint(63) + bone(84) + angle(9) = 156")
        print(f"컬럼 수: {len(out_cols)} (메타 {len(META_COLS)} + 피처 156)")

        if drop_missing:
            print(f"\n전처리 후 클래스 분포:")
            print(result_df["gesture"].value_counts().sort_index().to_string())

    result_df.to_csv(output_path, index=False)

    if verbose:
        print(f"\n저장 완료: {output_path}")

    return result_df


# ============================================================================
# 7. 실행
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python hand_gesture_preprocess.py <input.csv> [output.csv]")
        print("  output 미지정 시 input_preprocessed.csv 로 저장")
        sys.exit(1)

    input_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        output_path = input_path.rsplit(".", 1)[0] + "_preprocessed.csv"

    # 기본 실행은 단일 CSV를 읽어 같은 위치에 *_preprocessed.csv를 생성한다.
    preprocess_csv(input_path, output_path)
