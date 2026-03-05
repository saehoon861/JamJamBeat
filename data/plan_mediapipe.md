# MediaPipe 핸드 랜드마크 자동 추출 도구 기획서

## 1. 개요 및 목표
* **목표**: `data/raw_data/` 폴더 내 핸드 제스처 동영상 파일들에서, MediaPipe Hand Landmarker (Tasks API)를 이용하여 각 프레임마다 21개 손 랜드마크의 x, y, z 좌표를 자동 추출한 뒤, CSV 파일로 저장하는 파이썬 배치 처리 도구 개발
* **입력**: `{클래스ID}_{템포}_{왼손오른손}_{참여자}.mp4` 형식의 동영상 파일
* **출력**: 프레임별 랜드마크 좌표가 기록된 CSV 파일

## 2. MediaPipe 버전 및 모델 명세

> **[중요] 레거시(Legacy) API를 사용하지 않습니다.**
> `mp.solutions.hands`(레거시 Hands API)가 아닌, 최신 **MediaPipe Tasks API**(`mediapipe.tasks.vision.HandLandmarker`)를 사용합니다. 두 API는 import 경로, 초기화 방식, 결과 객체 구조가 전부 다릅니다.

### 사용할 모델 파일
* **경로**: `/home/user/JamJamBeat/hand_landmarker.task` (사전 다운로드 완료, 7.8MB)
* **코드에서 모델 로드 시** 이 절대 경로 또는 프로젝트 루트 기준 상대 경로로 참조

### 핵심 API (공식 문서 기반)
```python
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# VIDEO 모드로 생성 (프레임 단위 동기 처리)
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,                         # 촬영 영상이 한 손만 사용
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

with HandLandmarker.create_from_options(options) as landmarker:
    # 프레임 루프 내부에서:
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
```

### `detect_for_video` 주의사항
* **VIDEO 모드 전용 메서드**입니다. `running_mode=VisionRunningMode.VIDEO`가 아니면 에러가 발생합니다.
* 두 번째 인자 `frame_timestamp_ms`는 **밀리초 단위 정수**이며, 이전 호출보다 반드시 증가해야 합니다.
* OpenCV에서 읽은 프레임은 BGR 포맷이므로, **`cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`로 RGB 변환 후** `mp.Image`에 전달해야 합니다.

### 좌표 종류 명시
* 추출 대상은 `result.hand_landmarks` (**Landmarks**, 이미지 좌표 정규화 값)입니다.
* `result.hand_world_landmarks` (WorldLandmarks, 실제 미터 단위 3D 좌표)는 사용하지 않습니다.
* **Landmarks**의 x, y는 이미지의 너비/높이 대비 정규화된 `0.0~1.0` 값이며, z는 손목 기준 상대 깊이입니다.

### ⚠️ 손 미감지 프레임 안전 처리 (IndexError 방지)
`result.hand_landmarks`는 손이 감지되지 않으면 **빈 리스트**입니다. `result.hand_landmarks[0]`으로 바로 접근하면 `IndexError`가 발생하여 **배치 전체가 중단**됩니다. 반드시 길이 체크를 먼저 수행합니다:
```python
if len(result.hand_landmarks) > 0:
    landmarks = result.hand_landmarks[0]  # 첫 번째 손의 21개 랜드마크
    # x0, y0, z0, ..., x20, y20, z20 추출
else:
    # 해당 프레임의 x0~z20 컬럼을 전부 NaN으로 채움
```

### ⚠️ HandLandmarker 인스턴스 수명 원칙
HandLandmarker는 **영상 1개당 `with` 블록 1개**로 생성/소멸해야 합니다. 여러 영상을 하나의 인스턴스로 처리하면, 이전 영상의 마지막 `frame_timestamp_ms`가 내부에 누적되어 다음 영상의 첫 프레임(타임스탬프 0ms)에서 **"timestamp must be monotonically increasing"** 에러가 발생합니다.
```python
# ❌ 잘못된 예시: 루프 바깥에서 한 번만 생성
with HandLandmarker.create_from_options(options) as landmarker:
    for video in videos:        # 두 번째 영상에서 에러!
        ...

# ✅ 올바른 예시: 영상마다 새로 생성
for video in videos:
    with HandLandmarker.create_from_options(options) as landmarker:
        ...
```

## 3. FPS 처리 원칙

> 30fps 고정이라 가정하지 않습니다.

* 각 동영상 파일을 `cv2.VideoCapture`로 열 때 `cap.get(cv2.CAP_PROP_FPS)`로 **실제 FPS를 읽어냅니다.**
* `frame_timestamp_ms`는 `int(frame_idx / actual_fps * 1000)`으로 계산합니다.
* CSV에 기록되는 `timestamp` 컬럼도 동일한 수식(`frame_idx / actual_fps * 1000`)을 기반으로 `mm:ss:ms` 포맷으로 변환합니다.
* 이를 통해 15fps, 30fps, 60fps 등 어떤 영상이 들어와도 정확한 시간 매핑이 보장됩니다.

## 4. 결과물 CSV 포맷

### 컬럼 구성
| 컬럼명 | 타입 | 설명 |
|---|---|---|
| `frame_idx` | int | 프레임 인덱스 (0부터 시작) |
| `timestamp` | str | `mm:ss:ms` 포맷 (예: `00:00:866`) |
| `gesture` | str/null | **항상 빈 값(None)**. MediaPipe는 제스처를 판별하지 않으므로 비워둠. 수동 라벨링 결과와 추후 merge 시 사용 |
| `x0` ~ `x20` | float | 21개 **Landmarks**의 x 좌표 (이미지 너비 기준 정규화, 0.0~1.0) |
| `y0` ~ `y20` | float | 21개 **Landmarks**의 y 좌표 (이미지 높이 기준 정규화, 0.0~1.0) |
| `z0` ~ `z20` | float | 21개 **Landmarks**의 z 좌표 (깊이, 손목 기준 상대값) |

### 컬럼 순서
```
frame_idx, timestamp, gesture, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20
```

총 컬럼 수: 3 + (21 × 3) = **66개**

### 손 미감지 프레임 처리
특정 프레임에서 손이 감지되지 않을 경우, 해당 행의 `x0`~`z20` 컬럼을 전부 **빈 값(NaN/공백)**으로 남깁니다. `frame_idx`와 `timestamp`는 정상 기록하여 프레임 연속성을 유지합니다.

## 5. 저장 경로 및 파일 매핑

### 출력 폴더
* **경로**: `data/landmark_data/`
* 라벨링 결과(`labeled_data/`)와 혼동을 피하기 위해 별도 폴더를 사용합니다.
* **첫 실행 시 자동 생성**: 코드에서 `Path(output_dir).mkdir(parents=True, exist_ok=True)`를 호출하여, 폴더가 없으면 자동으로 생성합니다. 수동으로 미리 만들 필요 없습니다.

### 파일명 매칭 규칙
원본 영상의 파일명 stem(확장자 제거)에 `.csv`를 붙여 저장합니다.
* `1_slow_right_woman1.mp4` → `1_slow_right_woman1.csv`

### 스킵 로직
이미 동일 stem의 `.csv` 파일이 `landmark_data/`에 존재하면, 해당 영상은 건너뜁니다 (재작업 방지).

## 6. 디렉토리 구조 및 코드 구성

이 도구는 라벨링 툴과 달리 GUI가 없는 단순 배치 처리이므로, **`main.py` 단일 파일**에 모든 로직을 담습니다.

```text
JamJamBeat/
├── hand_landmarker.task              # 사전 다운로드된 MediaPipe 모델 파일
├── data/
│   ├── raw_data/                     # 원본 .mp4 동영상
│   ├── labeled_data/                 # 수동 라벨링 CSV (기존 도구)
│   ├── landmark_data/                # ✨ MediaPipe 추출 CSV (본 도구 출력, 자동 생성)
│   ├── labeling_tool/                # 기존 수동 라벨링 도구 (별개)
│   │   └── ...
│   └── landmark_extractor/           # ✨ 본 도구 소스코드
│       ├── pyproject.toml            # uv 프로젝트 의존성
│       └── main.py                   # 진입점 (단일 파일 구성)
```

### `main.py` 내부 함수 구성

| 함수명 | 역할 |
|---|---|
| `get_pending_videos(raw_dir, output_dir) -> list[str]` | `.mp4` 스캔 시 반드시 **`sorted()`** 사용 (OS별 정렬 불일치 방지) → `.csv` stem 대조 → 미처리 영상 목록 반환 |
| `format_timestamp(frame_idx, fps) -> str` | 프레임 인덱스를 `mm:ss:ms` 문자열로 변환 |
| `extract_landmarks(result) -> list[float \| None]` | `HandLandmarkerResult`에서 21개 랜드마크 x/y/z 63개 값 추출. `len(result.hand_landmarks) == 0`이면 `[None] * 63` 반환 |
| `process_video(video_path, output_dir, model_path) -> str` | 영상 1개를 처리하는 핵심 함수. `with HandLandmarker` 블록 안에서 프레임 루프 실행, CSV 저장, 저장 경로 반환 |
| `main()` | 진입점. 미처리 영상 목록 확보 → `process_video` 순회 호출 → 전체 완료 메시지 |

### `process_video` 상세 동작
```python
def process_video(video_path, output_dir, model_path):
    cap = cv2.VideoCapture(video_path)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
    )

    rows = []
    with HandLandmarker.create_from_options(options) as landmarker:  # 영상 1개당 1인스턴스
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(frame_idx / actual_fps * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            landmark_values = extract_landmarks(result)  # 63개 float 또는 None
            row = [frame_idx, format_timestamp(frame_idx, actual_fps), None] + landmark_values
            rows.append(row)

            # 진행률 출력
            if (frame_idx + 1) % 30 == 0 or frame_idx == total_frames - 1:
                pct = (frame_idx + 1) / total_frames * 100
                print(f"  Frame {frame_idx + 1}/{total_frames} ({pct:.1f}%)")

    cap.release()
    # DataFrame 조립 및 CSV 저장
    ...
```

## 7. 처리 흐름 요약

```
1. data/raw_data/*.mp4 파일 목록 스캔 (OS 호환성을 위해 `sorted()` 사용 필수)
2. data/landmark_data/*.csv 와 stem 대조 → 미처리 목록 필터링
3. data/landmark_data/ 폴더가 없으면 자동 생성 (mkdir)
4. for each 미처리 영상:
   a. cv2.VideoCapture로 영상 오픈, actual_fps 확보
   b. HandLandmarker를 VIDEO 모드로 생성 (with 블록 — 영상마다 새 인스턴스)
   c. for each frame:
      i.   cap.read() → BGR→RGB 변환
      ii.  frame_timestamp_ms = int(frame_idx / actual_fps * 1000)
      iii. result = landmarker.detect_for_video(mp_image, timestamp_ms)
      iv.  len(result.hand_landmarks) > 0 체크 후 좌표 추출 또는 NaN 채움
      v.   진행률 출력: "Frame 150/900 (16.7%)"
   d. pd.DataFrame 조립 → CSV 저장
   e. 완료 로그 출력
5. 전체 완료 메시지 출력
```

## 8. 의존성

```text
mediapipe>=0.10.9
opencv-python>=4.8.0
pandas>=2.0.0
numpy>=1.24.0
```

## 9. 실행 방법 (예상)
```bash
cd data/landmark_extractor
uv sync
uv run python main.py
```

## 10. 참고: 공식 문서 출처
* [MediaPipe Hand Landmarker Python Guide](https://ai.google.dev/mediapipe/solutions/vision/hand_landmarker/python)
* `running_mode`: `VisionRunningMode.VIDEO` 사용
* `detect_for_video(mp_image, frame_timestamp_ms)` 메서드 사용
* 결과 객체: `HandLandmarkerResult.hand_landmarks` → 21개 `NormalizedLandmark(x, y, z)`
* **WorldLandmarks는 사용하지 않음** (Landmarks만 사용)

## 11. 향후 참고: labeled_data와 merge 시 주의사항
> 지금 당장 구현할 사항은 아니지만, 미리 인지해두어야 할 문제입니다.

수동 라벨링 CSV(`labeled_data/`)와 랜드마크 CSV(`landmark_data/`)를 `frame_idx` 기준으로 합칠 때, 같은 영상이라도 OpenCV가 총 프레임 수를 **1~2개 다르게 보고할 수 있습니다** (디코더 차이, 마지막 불완전 프레임 등). 따라서:
* **inner join이 아닌 outer join**으로 합칠 것
* 불일치 행(한쪽에만 존재하는 frame_idx)은 **로그로 기록**하여 데이터 누락을 추적할 것
