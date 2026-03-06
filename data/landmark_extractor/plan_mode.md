# 📋 dual mode 지원 계획서 (IMAGE / VIDEO 전환 가능)

## 목표

`landmark_extractor/main.py` 상단의 **상수 1개**(`RUNNING_MODE`)만 바꾸면, IMAGE 모드와 VIDEO 모드가 자동으로 전환되도록 수정한다.

---

## 1. 라이브러리 호환성 확인

| 라이브러리 | 현재 명세 (`pyproject.toml`) | 확인 내용 |
|---|---|---|
| `mediapipe` | `>=0.10.32` | `HandLandmarkerOptions`에 `running_mode`, `min_tracking_confidence` 파라미터 존재. `HandLandmarker.detect(image)` (IMAGE), `HandLandmarker.detect_for_video(image, timestamp_ms)` (VIDEO) 모두 사용 가능. |
| `opencv-python` | `>=4.13.0.92` | `cap.set(cv2.CAP_PROP_POS_FRAMES, idx)` + `cap.read()` 사용 가능. |

### API 시그니처 확인 (mediapipe `>=0.10.32`)

```python
# IMAGE 모드
HandLandmarkerOptions(
    running_mode=VisionRunningMode.IMAGE,
    # min_tracking_confidence 설정 불가 (설정 시 ValueError)
)
result = landmarker.detect(mp_image)  # 인자: mp.Image 1개만

# VIDEO 모드
HandLandmarkerOptions(
    running_mode=VisionRunningMode.VIDEO,
    min_tracking_confidence=0.5,  # 0.0~1.0, 기본값 0.5
)
result = landmarker.detect_for_video(mp_image, timestamp_ms)  # 인자: mp.Image + int(ms)
```

> **주의**: `detect()`에 `timestamp_ms`를 넘기면 `TypeError`, `detect_for_video()`를 IMAGE 모드 인스턴스에서 호출하면 `RuntimeError`.

---

## 2. 설계 방침

1. 파일 상단에 `RUNNING_MODE` 상수를 선언한다.
2. `process_video()` 함수 내부에서 `RUNNING_MODE` 값에 따라:
   - **옵션 생성**: `min_tracking_confidence` 포함 여부 분기
   - **추론 호출**: `detect()` / `detect_for_video()` 분기
3. 프레임 읽기: **`cap.set()` + `cap.read()`** (seek 방식)는 모든 모드에서 **동일하게 유지**. (debug.md 3차 수정 사항 보존)
4. 타임스탬프: VIDEO 모드에서도 수학적 계산(`frame_idx / fps * 1000`)을 사용하되, MediaPipe에 넘기는 `timestamp_ms`는 **단조 증가(monotonically increasing)가 보장**되어야 함.

---

## 3. 변경 상세

### 수정 대상

`data/landmark_extractor/main.py` — 총 3곳 수정

---

### 3.1 상수 추가 (파일 상단, 29줄 부근)

```python
# BEFORE (현재: 29줄)
OUTPUT_DIR = str(PROJECT_ROOT / "data" / "landmark_data")

# AFTER
OUTPUT_DIR = str(PROJECT_ROOT / "data" / "landmark_data")

# --- 실행 모드 설정 ---
# VisionRunningMode.IMAGE : 프레임 단위 독립 추론 (추적 없음, Ground Truth 추출에 적합)
# VisionRunningMode.VIDEO : 연속 프레임 추적 기반 추론 (스무딩 적용, 부드러운 결과)
RUNNING_MODE = VisionRunningMode.IMAGE
```

**주의사항**:
- `RUNNING_MODE` 변수명은 기존 코드의 어떤 변수와도 겹치지 않음 ✅
- `VisionRunningMode`는 23줄에서 이미 import되어 있으므로 추가 import 불필요 ✅

---

### 3.2 옵션 생성 분기 (122~130줄)

```python
# BEFORE (현재: 122~130줄)
    # HandLandmarker 옵션 설정 (IMAGE 모드: 프레임 단위 독립 추론, 추적/예측 없음)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        # 주의: IMAGE 모드에서는 min_tracking_confidence를 설정하면 에러 발생하므로 삭제
    )

# AFTER
    # HandLandmarker 옵션 설정 (RUNNING_MODE에 따라 자동 분기)
    options_kwargs = dict(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RUNNING_MODE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    # VIDEO 모드에서만 min_tracking_confidence 추가 (IMAGE 모드에서 설정 시 에러)
    if RUNNING_MODE == VisionRunningMode.VIDEO:
        options_kwargs["min_tracking_confidence"] = 0.5
    
    options = HandLandmarkerOptions(**options_kwargs)
```

**주의사항**:
- `options_kwargs`는 새로 생기는 지역 변수. 기존 `options`를 덮어쓰므로 네이밍 충돌 없음 ✅
- `dict()` → `HandLandmarkerOptions(**kwargs)` 패턴은 Python 표준 문법 ✅
- `RUNNING_MODE`는 3.1에서 추가한 모듈 레벨 상수 ✅
- `VisionRunningMode.VIDEO`는 23줄에서 이미 import한 `VisionRunningMode`의 멤버 ✅
- `min_tracking_confidence`의 기본 값은 `0.5` (MediaPipe 공식 기본값) ✅

---

### 3.3 추론 호출 분기 (145~150줄)

```python
# BEFORE (현재: 145~150줄)
            # OSD UI 표시용 수학적 타임스탬프 (라벨링 파일과 동일한 계산식)
            timestamp_ms = int(frame_idx / actual_fps * 1000)

            # MediaPipe Image 생성 및 추론 (IMAGE 모드: 타임스탬프 인자 없음)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect(mp_image)

# AFTER
            # 수학적 타임스탬프 (라벨링 파일과 동일한 계산식)
            timestamp_ms = int(frame_idx / actual_fps * 1000)

            # MediaPipe Image 생성 및 추론
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            if RUNNING_MODE == VisionRunningMode.IMAGE:
                result = landmarker.detect(mp_image)
            else:
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
```

**주의사항**:
- `timestamp_ms`는 `int(frame_idx / actual_fps * 1000)`로 이미 `int`형이므로 `detect_for_video`에 넘길 때 추가 캐스팅 불필요 ✅
- `timestamp_ms`는 `frame_idx`가 0부터 순차 증가하므로 **단조 증가 자동 보장됨** ✅ (별도 `last_timestamp_ms` 가드 불필요)
  - 단, `actual_fps`가 극도로 높은 경우(예: 1000fps+) `int()` 절삭으로 같은 ms 값이 나올 수 있음. 현재 영상은 30fps이므로 매 프레임마다 최소 33ms 이상 증가하여 **충돌 가능성 0** ✅
- `else` 분기에서 `detect_for_video`를 쓰는데, 이 메서드는 VIDEO 모드 인스턴스에서만 호출 가능. RUNNING_MODE가 VIDEO일 때만 이 분기에 도달하므로 **안전** ✅
- `RUNNING_MODE`의 비교 대상: `VisionRunningMode.IMAGE`만 사용. LIVE_STREAM 모드는 고려 불필요. ✅
- `landmarker` 변수: `with` 블록에서 생성된 것 그대로. 재선언 없음 ✅
- `mp_image`, `mp.Image()`, `mp.ImageFormat.SRGB`는 모드와 무관하게 동일 ✅

---

## 4. 건드리지 않는 것들

| 항목 | 이유 |
|---|---|
| `cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)` | 3차 수정에서 추가한 seek 방식. 모드와 무관하게 유지. |
| `format_timestamp()` | CSV 타임스탬프 포맷. 모드와 무관. |
| `extract_landmarks()` | `HandLandmarkerResult` 구조는 IMAGE/VIDEO 동일. |
| `format_timestamp_from_ms()` | 미사용 함수이나 삭제하지 않음 (향후 활용 가능). |
| `get_pending_videos()` | 파일 스캐닝 로직. 변경 불필요. |
| `main()` | 진입점 로직. 변경 불필요. |

---

## 5. 사용 방법

모드를 전환하고 싶을 때 **`main.py` 상단의 `RUNNING_MODE` 상수 한 줄만 수정**:

```python
# 이미지 모드 (Ground Truth 추출용)
RUNNING_MODE = VisionRunningMode.IMAGE

# 비디오 모드 (스무딩 적용, 부드러운 결과)
RUNNING_MODE = VisionRunningMode.VIDEO
```

그 후 기존과 동일하게 실행:

```bash
rm -f data/landmark_data/*.csv
cd data/landmark_extractor
uv run python main.py
```

---

## 6. 검증 방법

### 자동 테스트
별도 유닛 테스트 없음 (기존에도 없음). 실행 후 CSV가 정상 생성되는지 확인.

### 수동 검증
1. `RUNNING_MODE = VisionRunningMode.IMAGE`로 설정 → 추출 실행 → `totalcheck_tool`로 랜드마크 1:1 매칭 확인
2. `RUNNING_MODE = VisionRunningMode.VIDEO`로 변경 → 추출 실행 → `totalcheck_tool`로 결과 비교
3. 두 결과의 차이를 육안으로 비교하여, 사용자가 최종 모드를 결정
