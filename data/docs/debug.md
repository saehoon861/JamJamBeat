# 🐛 랜드마크 프레임 불일치 (Sync Issue) 디버깅 기록

---

## 1. 1차 수정 — 실제 타임스탬프 적용 시도

### 원인 파악
`totalcheck_tool`에서 확인 시, 랜드마크 뼈대가 영상의 손 위치보다 1~2 프레임 앞서 위치하는 현상 발견.
`landmark_extractor`가 MediaPipe에 넘기는 타임스탬프를 수학적 계산(`frame_idx / 30 * 1000`)으로 만들고 있었는데, 실제 동영상 코덱(H.264)의 프레임 간격은 정확히 등간격이 아니기 때문에 MediaPipe `VIDEO` 모드의 내부 추적기가 시간을 잘못 인식, 랜드마크 위치를 앞당겨 예측한 것으로 판단.

### 해결 방안
수학적 계산 대신 OpenCV의 실제 코덱 타임스탬프(`cv2.CAP_PROP_POS_MSEC`)를 MediaPipe에 전달하도록 수정.

### 수정 사항
- `format_timestamp_from_ms()` 함수 추가
- `cap.read()` 직후 `cap.get(cv2.CAP_PROP_POS_MSEC)`으로 코덱 실제 PTS 획득
- `last_timestamp_ms` 변수로 단조 증가 보장
- `detect_for_video(mp_image, timestamp_ms)` 호출은 유지

### 결과: ❌ 실패
프레임 격차가 **4프레임 이상으로 더 크게 벌어짐.** 동영상 코덱의 PTS가 들쭉날쭉하게 요동치면서, MediaPipe `VIDEO` 모드의 추적기(Kalman Filter)가 "시간이 순간적으로 크게 건너뛰었으니 손이 매우 빠르게 움직였다"고 판단해 오버슈팅(Overshooting).

또한 1056번째 프레임에서 라벨링 데이터(수학적 계산)는 `35.166초`, 랜드마크 데이터(실제 코덱 시간)는 `35.536초`로 **0.37초(약 10프레임)의 시간 격차**가 발생.

---

## 2. 2차 수정 — IMAGE 모드로 전면 전환

### 원인 파악
MediaPipe `VIDEO` 모드 자체가 앞뒤 프레임의 연속성을 기반으로 손 위치를 "예측"하는 구조. 타임스탬프를 어떻게 넣든 추적(Tracking) 알고리즘이 개입하여 현재 프레임 사진 속 실제 손 위치와 다른 좌표가 나올 수밖에 없음.

### 해결 방안
MediaPipe의 실행 모드를 `VisionRunningMode.VIDEO` → `VisionRunningMode.IMAGE`로 교체하여, 프레임 1장을 전후 문맥 없이 독립적인 사진으로 처리하도록 변경.

### 수정 사항
- `running_mode=VisionRunningMode.IMAGE`로 변경
- `min_tracking_confidence=0.5` 삭제 (IMAGE 모드에서 설정 시 에러)
- `detect_for_video(mp_image, timestamp_ms)` → `detect(mp_image)` (타임스탬프 인자 없음)
- `last_timestamp_ms`, `cap.get(CAP_PROP_POS_MSEC)` 등 추적 관련 로직 전면 삭제
- CSV timestamp를 수학적 계산식(`format_timestamp(frame_idx, actual_fps)`)으로 원복

### 결과: ❌ 여전히 3~4프레임 밀림
IMAGE 모드는 추적/예측을 일절 하지 않으므로 MediaPipe의 문제가 아님이 확인됨. **프레임을 가져오는 방식 자체에 문제가 있음을 시사.**

---

## 3. 원인 파악 (진짜 근본 원인)

코드를 샅샅이 뒤진 결과, **`labeling_tool`/`totalcheck_tool`과 `landmark_extractor`가 동영상에서 프레임을 가져오는 OpenCV API 호출 방식이 근본적으로 다릅니다.**

| 도구 | 프레임 읽기 방식 | 코드 위치 |
|---|---|---|
| **labeling_tool** (수동 라벨링) | `cap.set(CAP_PROP_POS_FRAMES, idx)` → `cap.read()` | `labeling_tool/core/video_handler.py:47` |
| **totalcheck_tool** (검수 도구) | `cap.set(CAP_PROP_POS_FRAMES, idx)` → `cap.read()` | `totalcheck_tool/core/video_handler.py:40` |
| **landmark_extractor** (랜드마크 추출) | for 루프 안에서 순차적으로 `cap.read()` 반복 | `landmark_extractor/main.py:137` |

### 왜 이 차이가 프레임 밀림을 일으키는가?

MP4(H.264) 코덱은 모든 프레임을 독립적으로 저장하지 않습니다. **키프레임(I-frame)**만 완전한 이미지이고, 나머지 프레임(P-frame, B-frame)은 키프레임과의 차이만 저장합니다.

* **순차 읽기 (`cap.read()` 반복)**: OpenCV가 내부 디코더 버퍼를 유지하면서 프레임 0부터 순서대로 디코딩.
* **랜덤 접근 (`cap.set(idx)` → `cap.read()`)**: 요청한 `idx`에 가장 가까운 이전 키프레임으로 점프한 뒤, 거기서부터 순차 디코딩. H.264 코덱의 B-frame 재배열(reordering) 특성에 따라 **순차 읽기와 다른 프레임이 반환될 수 있음.**

즉, 같은 `frame_idx = 100`을 요청해도:
* `labeling_tool`이 라벨링 시 **`cap.set(100)` → `cap.read()`**로 본 프레임과,
* `landmark_extractor`가 추출 시 **100번째 순차 `cap.read()`**로 본 프레임이 **다릅니다.**

**이것이 IMAGE 모드로 바꿔도 여전히 3~4프레임이 밀리는 진짜 원인입니다.**

---

## 4. 해결 방안

**`landmark_extractor`도 `labeling_tool`/`totalcheck_tool`과 동일한 프레임 접근 방식(`cap.set() + cap.read()`)을 사용하도록 수정.**

---

## 5. 수정 사항

### 수정 대상 파일

`data/landmark_extractor/main.py` — `process_video()` 함수 내부 (134~161줄)

### BEFORE (현재 코드, 순차 읽기)

```python
# landmark_extractor/main.py:134-156 (현재)
    with HandLandmarker.create_from_options(options) as landmarker:
        for frame_idx in range(total_frames):
            ret, frame = cap.read()  # ⚠️ 순차 읽기
            if not ret:
                break
```

### AFTER (수정 코드, seek 기반 읽기)

```python
# landmark_extractor/main.py (수정 후)
    with HandLandmarker.create_from_options(options) as landmarker:
        for frame_idx in range(total_frames):
            # ✅ labeling_tool/totalcheck_tool과 동일한 seek 방식으로 프레임 읽기
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
```

### 변경 요약 (diff)

```diff
    with HandLandmarker.create_from_options(options) as landmarker:
        for frame_idx in range(total_frames):
+           cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
```

**단 1줄 추가.** 기존 변수는 전부 그대로 유지.

### 수정 시 주의사항 체크리스트

# 🐛 수정 시 주의사항 체크리스트

| 분류 | # | 체크 포인트 |
|:---:|:---:|---|
| **A. OpenCV Seek 제어** | 1 | `cap.set()` 인자 타입 (`int`) |
| | 2 | `frame_idx` 범위 준수 (`0` ~ `total-1`) |
| | 3 | `ret` 실패 시 루프 탈출 조건 |
| | 4 | `CAP_PROP_POS_FRAMES` 상수 사용 가능 여부 |
| **B. MediaPipe 모드 정합성** | 5 | `detect()` 메서드 호출 (IMAGE 모드 전용) |
| | 6 | `min_tracking_confidence` 인자 삭제 확인 |
| | 7 | `detect_for_video()` 복귀 금지 |
| **C. 데이터 싱크 일관성** | 8 | `actual_fps` 유지 (Labeling Tool과 통일) |
| | 9 | `format_timestamp` 계산식 보존 |
| **D. 구조 및 성능** | 10 | `cap` 객체 중복 선언 및 네이밍 충돌 방지 |

### 실행 방법

```bash
# JamJamBeat 루트 디렉토리 기준

# 1. 1차 시도로 오염된 landmark_data 전체 삭제
rm -f data/landmark_data/*.csv

# 2. 코드 수정 (IMAGE 모드로 변경)

# 3. 재추출 실행
cd data/landmark_extractor
uv run python main.py

# 4. totalcheck_tool로 검수하여 오버레이 1:1 매칭 최종 검증
cd ../totalcheck_tool
uv run python main.py
```

## 6. 구현 기록 (Implementation Log)

### [1차 수정 시도 보관] 2026-03-06 실제 타임스탬프 적용 시도

계획서와 동일하게 `landmark_extractor/main.py`를 수정.

| 항목 | 계획 | 실제 구현 | 차이 |
|---|---|---|---|
| `format_timestamp_from_ms()` 추가 | 계획 있음 | ✅ 추가됨 | 동일 |
| `cap.get(CAP_PROP_POS_MSEC)` 사용 | 계획 있음 | ✅ 적용됨 | 동일 |
| `last_timestamp_ms` 초기화 위치 | `with` 블록 직전 | ✅ `with` 블록 직전에 `-1`로 초기화 | 동일 |
| `cap.get` 호출 순서 | `cap.read()` 직후 | ✅ `cap.read()` 바로 다음 줄 | 동일 |
| `detect_for_video` 타입 | `int` | ✅ `int(pos_msec)` 그대로 전달 | 동일 |

### [2차 수정] 2026-03-06 IMAGE 모드 전환

| 항목 | 수정 내용 |
|---|---|
| `running_mode` | `VIDEO` → `IMAGE` |
| `min_tracking_confidence` | 삭제 |
| `detect_for_video()` | `detect()`로 변경 |
| 타임스탬프 로직 | `last_timestamp_ms`, `POS_MSEC` 전면 삭제, 수학적 계산 원복 |

### [3차 수정] 2026-03-06 landmark_extractor 의 프레임 읽는 방식 변경

| 항목 | 수정내용|
|---|---|
| `landmark_extractor/main.py:137` | `cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)` | ✅ 추가 |

---

## 7. 추가 발견 사항

### FPS 설정 불일치

| 도구 | FPS 출처 | 값 |
|---|---|---|
| `labeling_tool` | `label_config.yaml`에 하드코딩 | `fps: 30` (고정) |
| `landmark_extractor` | `cap.get(cv2.CAP_PROP_FPS)` | 영상마다 다름 (예: 29.97, 30.0 등) |

이로 인해 같은 1056번째 프레임의 타임스탬프가 라벨링 데이터는 `35.166초`, 랜드마크 데이터는 `35.536초`로 다르게 찍힘. 현재 타임스탬프는 UI 표시 목적으로만 사용되고, 데이터 join은 `frame_idx` 기준으로 하므로 당장 치명적이지는 않으나, 향후 통일 검토 필요.