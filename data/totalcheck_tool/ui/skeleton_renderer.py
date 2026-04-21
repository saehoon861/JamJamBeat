"""ui/skeleton_renderer.py - [View] MediaPipe 핸드 랜드마크 스켈레톤 시각화 모듈

landmark_data CSV에서 읽어온 정규화 좌표를 픽셀 좌표로 변환하여
원본 프레임 위에 손 뼈대(Skeleton)를 덧그립니다.

스타일 (하드코딩):
  - 관절(Joint): 녹색 (0, 255, 0), 반지름 3px, 채움(-1)
  - 뼈대(Bone):  파란색 (255, 0, 0), 두께 2px

MediaPipe 공식 HAND_CONNECTIONS 튜플 (21개 랜드마크 연결망) 하드코딩:
  https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/hands.py
"""

import cv2
import numpy as np
import pandas as pd

# --- 관절/뼈대 스타일 상수 ---
JOINT_COLOR = (0, 255, 0)    # 녹색 (BGR)
BONE_COLOR = (255, 0, 0)     # 파란색 (BGR)
JOINT_RADIUS = 3
BONE_THICKNESS = 2

# --- MediaPipe HAND_CONNECTIONS 하드코딩 ---
# (랜드마크 인덱스 쌍: 연결할 두 관절의 인덱스)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # 엄지
    (0, 5), (5, 6), (6, 7), (7, 8),         # 검지
    (5, 9), (9, 10), (10, 11), (11, 12),    # 중지
    (9, 13), (13, 14), (14, 15), (15, 16),  # 약지
    (13, 17), (17, 18), (18, 19), (19, 20), # 새끼
    (0, 17),                                 # 손바닥 외곽
]


def draw_landmarks(frame: np.ndarray, landmark_row: pd.Series) -> np.ndarray:
    """현재 프레임에 핸드 랜드마크 스켈레톤을 덧그려 반환합니다.

    Args:
        frame: 원본 프레임 numpy 배열 (BGR, in-place 수정됨)
        landmark_row: landmark_data CSV의 1개 행 (pandas Series).
                      x0~z20 컬럼이 포함되어 있어야 합니다.

    Returns:
        스켈레톤이 덧그려진 프레임 배열 (동일 객체).
    """
    # 손 미감지 프레임 처리: x0가 NaN이면 그리지 않고 즉시 반환
    try:
        x0_val = landmark_row["x0"]
        if pd.isna(x0_val):
            return frame
    except (KeyError, TypeError):
        return frame

    h, w = frame.shape[:2]

    # 21개 랜드마크 픽셀 좌표 계산
    pts = []
    for i in range(21):
        x_norm = landmark_row.get(f"x{i}", float("nan"))
        y_norm = landmark_row.get(f"y{i}", float("nan"))
        if pd.isna(x_norm) or pd.isna(y_norm):
            pts.append(None)
        else:
            px = int(float(x_norm) * w)
            py = int(float(y_norm) * h)
            pts.append((px, py))

    # 뼈대(Bone) 그리기
    for (start_idx, end_idx) in HAND_CONNECTIONS:
        p1 = pts[start_idx]
        p2 = pts[end_idx]
        if p1 is not None and p2 is not None:
            cv2.line(frame, p1, p2, BONE_COLOR, BONE_THICKNESS, cv2.LINE_AA)

    # 관절(Joint) 그리기
    for pt in pts:
        if pt is not None:
            cv2.circle(frame, pt, JOINT_RADIUS, JOINT_COLOR, -1, cv2.LINE_AA)

    return frame
