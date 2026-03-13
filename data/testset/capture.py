"""
capture.py - 제스처 테스트셋 데이터 수집 도구
==============================================
관련 계획서: data/docs/plan_testset.md (섹션 1, 5.1)

실행 방법:
    uv run python capture.py

기능 개요:
    - 웹캠 영상에서 MediaPipe Hand Landmarker로 손 랜드마크 추론 및 오버레이
    - Condition(촬영 상태) 선택 후 0~6 키로 제스처 이미지/랜드마크 수집
    - 종료(q/Esc) 시 landmarks_{gesture}.csv 저장
"""

import os
import re
import sys
import time
import csv

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# 상수 정의
# ─────────────────────────────────────────────

# 프로젝트 루트 기준 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

TESTDATA_DIR = os.path.join(SCRIPT_DIR, "testdata")
IMAGES_DIR   = os.path.join(TESTDATA_DIR, "images")

# hand_landmarker.task 모델 파일 위치 (프로젝트 루트)
TASK_FILE = os.path.join(PROJECT_ROOT, "hand_landmarker.task")

# 제스처 클래스 번호 목록 (0: No Gesture ~ 6: KHeart)
GESTURE_CLASSES = list(range(7))

# Condition 단축키 → condition 문자열 매핑
CONDITION_KEYS = {
    ord('b'): "BaseP",
    ord('r'): "RollP",
    ord('p'): "PitchP",
    ord('y'): "YawP",
    ord('n'): "NoneNetural",
    ord('o'): "NoneOther",
}

# CSV 컬럼 순서 정의
LANDMARK_COUNT = 21
CSV_COLUMNS = (
    ["frame_idx", "timestamp", "gesture"]
    + [f"{ax}{i}" for i in range(LANDMARK_COUNT) for ax in ("x", "y", "z")]
)

# 웹캠 해상도
CAM_WIDTH  = 1280
CAM_HEIGHT = 720

# HUD 색상/폰트
COLOR_WHITE  = (255, 255, 255)
COLOR_GREEN  = (0, 220, 0)
COLOR_RED    = (0, 0, 220)
COLOR_YELLOW = (0, 220, 220)
COLOR_ORANGE = (0, 165, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────
# 초기화 유틸
# ─────────────────────────────────────────────

def ensure_directories() -> None:
    """
    역할: images/0~6/ 하위 폴더를 포함한 필수 디렉토리를 모두 생성한다.
          이미 존재할 경우 무시한다.
    """
    os.makedirs(TESTDATA_DIR, exist_ok=True)
    for g in GESTURE_CLASSES:
        os.makedirs(os.path.join(IMAGES_DIR, str(g)), exist_ok=True)


def get_user_name() -> str:
    """
    역할: 터미널에서 user_name을 입력받고 유효성을 검증한다.
    반환: 검증된 user_name 문자열 (영문자/숫자, 10자 이내)
    """
    pattern = re.compile(r'^[A-Za-z0-9]+$')
    while True:
        name = input("사용자 이름을 입력하세요 (영문/숫자, 10자 이내): ").strip()
        if not name:
            print("  [오류] 이름이 비어있습니다. 다시 입력해 주세요.")
        elif len(name) > 10:
            print("  [오류] 10자를 초과했습니다. 다시 입력해 주세요.")
        elif not pattern.match(name):
            print("  [오류] 영문자/숫자만 사용할 수 있습니다. 다시 입력해 주세요.")
        else:
            return name


def load_initial_indices() -> dict[int, int]:
    """
    역할: 각 gesture 클래스별 현재 최대 frame_idx를 CSV 마지막 행에서 읽어 초기화한다.
          행 수가 아닌 마지막 행의 frame_idx 값을 직접 읽어 순번 어긋남을 방지한다.
    반환: {gesture: next_frame_idx} 형태의 딕셔너리
    """
    indices = {}
    for g in GESTURE_CLASSES:
        csv_path = os.path.join(TESTDATA_DIR, f"landmarks_{g}.csv")
        if os.path.exists(csv_path):
            try:
                # 마지막 행만 효율적으로 읽기
                with open(csv_path, 'r') as f:
                    reader = csv.reader(f)
                    last_row = None
                    for row in reader:
                        last_row = row
                if last_row and last_row[0] != "frame_idx":
                    # 헤더가 아닌 마지막 데이터 행의 frame_idx + 1
                    indices[g] = int(last_row[0]) + 1
                else:
                    indices[g] = 1
            except (ValueError, IndexError):
                indices[g] = 1
        else:
            # CSV 없으면 1부터 시작
            indices[g] = 1
    return indices


def load_landmarker() -> mp_vision.HandLandmarker:
    """
    역할: hand_landmarker.task 파일을 로드하여 HandLandmarker 인스턴스를 반환한다.
    반환: 초기화된 HandLandmarker 객체
    """
    if not os.path.exists(TASK_FILE):
        print(f"[오류] hand_landmarker.task 파일을 찾을 수 없습니다: {TASK_FILE}")
        sys.exit(1)

    base_options = mp_python.BaseOptions(model_asset_path=TASK_FILE)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,                          # max_num_hands = 1
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


# ─────────────────────────────────────────────
# 시각화 유틸
# ─────────────────────────────────────────────

# MediaPipe Hand의 관절 연결선 정의
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # 엄지
    (0,5),(5,6),(6,7),(7,8),          # 검지
    (0,9),(9,10),(10,11),(11,12),     # 중지
    (0,13),(13,14),(14,15),(15,16),   # 약지
    (0,17),(17,18),(18,19),(19,20),   # 소지
    (5,9),(9,13),(13,17),             # 손바닥 가로 연결
]


def draw_landmarks(frame: np.ndarray, landmarks) -> None:
    """
    역할: 손 랜드마크와 관절 연결선을 프레임에 그린다.
    파라미터:
        frame     - OpenCV BGR 이미지 (in-place 수정)
        landmarks - MediaPipe HandLandmarker 결과의 hand_landmarks[0] 리스트
    """
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # 연결선 그리기
    for (a, b) in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], COLOR_GREEN, 2)
    # 랜드마크 점 그리기
    for pt in pts:
        cv2.circle(frame, pt, 4, COLOR_WHITE, -1)
        cv2.circle(frame, pt, 4, COLOR_GREEN, 1)


def draw_hud(
    frame: np.ndarray,
    user_name: str,
    condition: str | None,
    no_hand_ts: float,
    save_fail_ts: float,
    captured_counts: dict[int, int],
) -> None:
    """
    역할: 화면 상단 HUD(상태 정보)와 경고 메시지를 렌더링한다.
    파라미터:
        frame          - OpenCV BGR 이미지
        user_name      - 사용자 이름
        condition      - 현재 선택된 Condition 문자열 (None이면 미선택)
        no_hand_ts     - 손 미검출 이벤트 발생 시각 (time.time())
        save_fail_ts   - imwrite 실패 이벤트 발생 시각 (time.time())
        captured_counts - {gesture: 캡처된 프레임 수} (이번 세션)
    """
    now = time.time()
    y = 30

    # 사용자 이름 표시
    cv2.putText(frame, f"User: {user_name}", (10, y), FONT, 0.7, COLOR_WHITE, 2)
    y += 30

    # Condition 상태 표시
    if condition:
        cv2.putText(frame, f"Condition: {condition}", (10, y), FONT, 0.7, COLOR_GREEN, 2)
    else:
        cv2.putText(frame, "Condition을 선택해 주세요 (b/r/p/y/n/o)", (10, y), FONT, 0.65, COLOR_YELLOW, 2)
    y += 30

    # 키 도움말
    help_text = "b:BaseP r:RollP p:PitchP y:YawP n:NoneNetural o:NoneOther | 0-6:capture | q/Esc:quit"
    cv2.putText(frame, help_text, (10, y), FONT, 0.45, (180, 180, 180), 1)
    y += 25

    # 이번 세션 캡처 수 표시
    count_text = " | ".join([f"G{g}:{captured_counts.get(g, 0)}" for g in GESTURE_CLASSES])
    cv2.putText(frame, count_text, (10, y), FONT, 0.45, (150, 220, 255), 1)

    # 손 미검출 경고 (2초 표시)
    if now - no_hand_ts < 2.0:
        cv2.putText(frame, "No hand detected", (10, CAM_HEIGHT - 60), FONT, 1.0, COLOR_RED, 3)

    # 저장 실패 경고 (1초 표시)
    if now - save_fail_ts < 1.0:
        cv2.putText(frame, "저장 실패!", (10, CAM_HEIGHT - 30), FONT, 1.0, COLOR_ORANGE, 3)


# ─────────────────────────────────────────────
# CSV 저장
# ─────────────────────────────────────────────

def save_to_csv(memory_buffer: list[dict]) -> None:
    """
    역할: 세션 동안 수집한 메모리 버퍼를 gesture별로 CSV 파일에 저장한다.
          기존 파일이 있으면 Append, 없으면 헤더 포함 신규 생성한다.
    파라미터:
        memory_buffer - 캡처된 데이터 딕셔너리 리스트 (CSV_COLUMNS와 동일 키)
    """
    if not memory_buffer:
        print("[저장] 이번 세션에 캡처된 데이터가 없습니다.")
        return

    # gesture 별로 분류
    gesture_groups: dict[int, list[dict]] = {}
    for row in memory_buffer:
        g = row["gesture"]
        gesture_groups.setdefault(g, []).append(row)

    for g, rows in gesture_groups.items():
        df = pd.DataFrame(rows, columns=CSV_COLUMNS)
        csv_path = os.path.join(TESTDATA_DIR, f"landmarks_{g}.csv")
        if os.path.exists(csv_path):
            # 기존 파일에 Append (헤더 없이)
            df.to_csv(csv_path, mode='a', header=False, index=False)
            print(f"[저장] landmarks_{g}.csv 에 {len(rows)}행 추가 (Append)")
        else:
            # 신규 파일 생성 (헤더 포함)
            df.to_csv(csv_path, mode='w', header=True, index=False)
            print(f"[저장] landmarks_{g}.csv 신규 생성 ({len(rows)}행)")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main() -> None:
    # 0. 환경 확인 및 디렉토리 생성
    ensure_directories()

    # 1. 초기화
    user_name = get_user_name()
    indices   = load_initial_indices()   # {gesture: next_frame_idx}
    landmarker = load_landmarker()

    print(f"\n[시작] user={user_name}")
    print("[시작] 각 클래스 시작 frame_idx:", {g: indices[g] for g in GESTURE_CLASSES})
    print("[안내] 웹캠이 열립니다. q 또는 Esc로 종료하세요.\n")

    # 상태 변수
    condition:    str | None = None   # 현재 선택된 Condition
    memory_buffer: list[dict] = []    # 세션 랜드마크 버퍼
    no_hand_ts   = -999.0             # 손 미검출 이벤트 시각
    save_fail_ts = -999.0             # imwrite 실패 이벤트 시각
    captured_counts: dict[int, int] = {g: 0 for g in GESTURE_CLASSES}  # 이번 세션 캡처 수
    prev_key     = -1                 # 이전 프레임 키 (key-repeat 방지)

    # 웹캠 열기 (720p 고정)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        print("[오류] 웹캠을 열 수 없습니다.")
        sys.exit(1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[경고] 프레임을 읽을 수 없습니다.")
                break

            frame = cv2.flip(frame, 1)  # 좌우 반전 (거울 모드)

            # MediaPipe 추론
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            hand_detected = bool(result.hand_landmarks)

            if hand_detected:
                # 랜드마크 오버레이
                draw_landmarks(frame, result.hand_landmarks[0])
            else:
                # 손 미검출 → 현재 키 입력으로 캡처 시도 시 차단
                pass  # no_hand_ts 는 캡처 시도 시점에 갱신

            # HUD 렌더링
            draw_hud(frame, user_name, condition, no_hand_ts, save_fail_ts, captured_counts)

            # 화면 출력
            cv2.imshow("JamJamBeat Capture Tool", frame)

            # 키 이벤트 처리 (key down 최초 이벤트만, key-repeat 무시)
            key = cv2.waitKey(1) & 0xFF

            # key-repeat 방지: 직전과 동일한 키가 연속으로 들어오면 무시
            # waitKey(1)은 매 프레임마다 새로 읽으므로 충분하지만,
            # 실제로 누름이 유지되는 경우 연속 프레임에서 동일 key가 반복될 수 있음.
            # OpenCV에서 key-repeat을 완벽히 막으려면 별도 state 관리 필요.
            # 여기서는 released key 감지를 위해 prev_key를 활용한다.
            if key != 0xFF and key == prev_key:
                # 같은 키가 연속으로 눌리면 key-repeat으로 간주하고 무시
                key = 0xFF
            prev_key = key if key != 0xFF else -1

            # 종료
            if key in (ord('q'), 27):  # 27 = Esc
                break

            # Condition 선택 키
            if key in CONDITION_KEYS:
                condition = CONDITION_KEYS[key]
                print(f"[Condition] {condition} 선택됨")
                continue

            # 캡처 키 (0~6)
            if ord('0') <= key <= ord('6'):
                gesture = key - ord('0')

                # Condition 미선택 차단
                if not condition:
                    print("[캡처 차단] Condition이 선택되지 않았습니다.")
                    continue

                # 손 미검출 차단
                if not hand_detected:
                    no_hand_ts = time.time()
                    print("[캡처 차단] 손이 감지되지 않았습니다.")
                    continue

                # 현재 클래스의 frame_idx
                fidx = indices[gesture]
                filename = f"{gesture}_{fidx}_{condition}_{user_name}.jpg"
                img_path = os.path.join(IMAGES_DIR, str(gesture), filename)

                # 이미지 즉시 저장 시도
                success = cv2.imwrite(img_path, frame)
                if not success:
                    # imwrite 실패: 해당 프레임 discard, 1초 경고 표시
                    save_fail_ts = time.time()
                    print(f"[오류] 이미지 저장 실패: {img_path}")
                    continue

                # 랜드마크 normalized 좌표 추출
                lm_list = result.hand_landmarks[0]
                coords = {}
                for i, lm in enumerate(lm_list):
                    coords[f"x{i}"] = round(lm.x, 6)
                    coords[f"y{i}"] = round(lm.y, 6)
                    coords[f"z{i}"] = round(lm.z, 6)

                # 메모리 버퍼에 누적
                row = {
                    "frame_idx": fidx,
                    "timestamp": None,   # 현 단계 null 처리
                    "gesture":   gesture,
                    **coords,
                }
                memory_buffer.append(row)

                # 인덱스 증가 및 카운트 갱신
                indices[gesture] += 1
                captured_counts[gesture] += 1
                print(f"[캡처] G{gesture} | frame_idx={fidx} | condition={condition} | file={filename}")

    finally:
        # 2. 종료 시 CSV 저장
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

        print("\n[종료] CSV 저장 중...")
        save_to_csv(memory_buffer)
        total = sum(captured_counts.values())
        print(f"[종료 완료] 총 {total}장 캡처. 클래스별: {captured_counts}")


if __name__ == "__main__":
    main()
