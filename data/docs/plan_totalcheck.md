# 통합 검수 도구 (Total Check Tool) 기획서

## 1. 개요 및 목표
* **목표**: `raw_data`의 동영상, `landmark_data`의 랜드마크 좌표, `labeled_data`의 수동 라벨링 결과를 한 화면에 통합하여 보여주고, 잘못된 라벨링을 보정한 뒤 결과를 저장하는 시각적 검수 도구 개발.
* **주요 기능**: 
  1. 영상 위에 MediaPipe로 추출된 손 랜드마크 뼈대(Skeleton) 오버레이 렌더링
  2. 현재 프레임의 라벨링 상태(클래스명, 범위) 표시
  3. 라벨링 도구와 동일한 단축키로 제스처 라벨 수정 (수퍼키릿 및 텐키패드 모두 지원)
  4. Enter(저장), n(스킵), Esc(종료) 등 세밀한 저장 플로우 제공

## 2. 디렉토리 구조 및 위치 설계

이 도구는 기존 `labeling_tool`(수동 라벨링) 설계 사상(MVC 패턴, uv 환경 격리)을 계승하되, 검수 목적에 맞게 확장된 구조를 가집니다. 독립된 폴더에 구성합니다.

```text
JamJamBeat/
├── data/
│   ├── raw_data/                     # 원본 동영상 (.mp4)
│   ├── labeled_data/                 # 수동/수정 라벨링 파일 (.csv)
│   ├── landmark_data/                # 추출된 랜드마크 파일 (.csv)
│   ├── labeling_tool/                # (기존) 수동 라벨링 도구
│   ├── landmark_extractor/           # (기존) 랜드마크 추출 도구
│   └── totalcheck_tool/              # ✨ (신규) 통합 검수 도구
│       ├── pyproject.toml
│       ├── config/
│       │   └── (기존 라벨링툴의 label_config.yaml 만 재사용, style_config는 사용 안함)
│       ├── core/
│       │   ├── annotation_state.py   # [Model] 기존 UI 로직 + labeled.csv 읽어오기 기능
│       │   ├── class_pipeline.py     # [Singleton] 라벨 클래스 파싱
│       │   └── video_handler.py      # MP4 및 두 CSV 병합 관리
│       ├── ui/
│       │   ├── osd_renderer.py       # [View] OSD, 타임라인 렌더러
│       │   └── skeleton_renderer.py  # [View] 랜드마크 스켈레톤 시각화 모듈 (신규)
│       ├── file_io/
│       │   ├── data_loader.py        # 3종 파일(mp4, labeled, landmark) 로드 및 유효성 검증
│       │   └── data_exporter.py      # 변경된 라벨을 덮어쓰기 저장
│       └── main.py                   # [Controller] 이벤트 루프 (Enter/n/Esc 제어)
```

## 3. 핵심 플로우 및 예외 처리

### 데이터 입력 로드 및 검증 (✨ Phase 2 변경사항 안내)

> ⚠️ **중요**: 아래 내용은 1차 구현 시의 명세입니다. 빈 라벨부터 시작할 수 있도록 로드 로직이 [6.2 모듈별 구체적 수정 계획 (A. data_loader.py)](#a-file_iodata_loaderpy-로드-로직-유연화)에서 변경되었습니다.

`data_loader.py`에서 영상 1개를 처리할 때 다음 3개가 모두 존재해야 합니다(Phase 1 기준):
1. `raw_data / {stem}.mp4`
2. `labeled_data / {stem}.csv` (라벨링 수정 대상)
3. `landmark_data / {stem}.csv` (읽기 전용, 시각화 용도)

* **에러 및 필터링 기준**:
  * 3개 중 하나라도 누락되면 해당 영상은 작업 대상에서 **제외(Skip)** 합니다.
  * 콘솔(print) 창에 예외 발생을 알립니다. `[WARN] {stem} 파일 로드 실패 (labeled_data 누락) -> 스킵합니다.`
  * 또한, 본 도구는 `labeled_data/`의 파일이 이미 성공적으로 존재하는 것들만 검수하는 것이므로, 아직 라벨링하지 않은(미처리) 영상은 아예 대상 목록에 오르지 않아야 합니다.

### 저장 및 스킵 메커니즘
* **Enter (저장 및 다음)**: 현재 수정한 상태(`AnnotationState` 내부 배열)를 `labeled_data/{stem}.csv` 파일에 **덮어쓰기(Overwrite)**로 내보냅니다.
* **'n' 키 (스킵)**: 수정한 내용을 메모리 상에서만 파기하고, 원본 CSV 파일엔 쓰기 작업을 일절 호출하지 않은 채 루프를 `break`하여 다음 영상으로 넘어갑니다.
* **Esc 키 (종료)**: 수정한 내용을 저장하지 않고 전체 프로그램 세션을 즉시 `sys.exit()`으로 종료합니다.

### 상태 관리 (State) 초기화 원칙
새 영상을 열 때 `AnnotationState`의 0으로 가득 찬 초기 배열 대신, **`labeled_data/{stem}.csv`의 `gesture` 컬럼 값을 파싱하여 상태 배열의 초기값으로 덮어씌웁니다.**
* 사용자가 'n' 또는 'Esc'를 누르면 메모리(State)의 변경점만 휘발되며, 원본 CSV 파일에는 물리적 쓰기(Write) 연산이 발생하지 않으므로 데이터가 안전하게 보존됩니다.

### 라벨링 조작 단축키 (숫자 패드 확장)
* **0~6 (숫자키/텐키패드 병행 지원)**: 
  * 상단 숫자키: `48`(0) ~ `54`(6)
  * 우측 텐키패드: WSL(Ubuntu) 등 특정 환경에서는 텐키패드도 상단 숫자키와 동일한 키코드(`48`~`54`)로 입력되므로, 복잡한 분기 없이 `48~54` 및 `176~182`를 모두 포괄하여 수용하도록 처리합니다.
* **S / E**: 구간 설정(Start / End) (`labeling_tool`의 S/F 규칙을 동일 적용)
* **탐색 조작**: 스페이스바(단일 프레임 토글), A/D(1프레임 전후), Q/E(10프레임 전후)

## 4. UI 렌더링 명세

### 랜드마크(Skeleton) 시각화 논리 (`skeleton_renderer.py`)
`landmark_data` CSV는 X, Y가 정규화(`0.0~1.0`)되어 있습니다.
* **스타일 하드코딩 명세**:
  복잡도를 줄이기 위해 `style_config.yaml`을 따로 두지 않고 `skeleton_renderer.py` 내부에 상수로 하드코딩합니다.
  * **관절(Joint/Circle)**: 녹색(`(0, 255, 0)`), 두께 `-1`(채움), 반지름 `3` 픽셀.
  * **뼈대(Bone/Line)**: 파란색(`(255, 0, 0)`), 두께 `2` 픽셀.
* **렌더링 흐름**:
  1. 현재 프레임 인덱스에 해당하는 행을 `landmark_data` DataFrame에서 조회.
  2. `x0, y0, ...` 좌표값이 `NaN` (또는 빈 문자열)이면 화면에 손을 그리지 않고 즉시 반환(손 미감지 프레임 안전처리).
  3. 존재할 경우, 각 `x * frame_width`, `y * frame_height`로 실제 Pixel 좌표로 강제 형변환(`int()`).
  4. 리스트 순회: MediaPipe의 공식 `HAND_CONNECTIONS` 정적 튜플 명세(0-1, 1-2 등 21개 연결망)를 소스코드 에 그대로 복사(하드코딩)하여 `cv2.line`과 `cv2.circle`로 원본 프레임 위에 덮어 그립니다.

### 프레임 불일치 (데이터 Merge 문제) 대응
OpenCV 영상 총 프레임 수와, 두 CSV의 행 개수가 1~2개 다를 수 있습니다.
* **Controller** (`main.py`) 단에서 파일 로드 시 3개 길이 중 `min()` 값을 구해 `limit_frames` 변수로 삼습니다.
* 슬라이더 제어 및 State 배열 크기는 이 `limit_frames`에 맞춰 생성되며, 이를 초과하는 프레임(배그나 비디오 끝자락 찌꺼기)은 화면에 진입하지 못하게 차단(`IndexError` 원천 봉쇄)합니다.

## 5. 단계별 모듈 세부 구현 명세


### 1) 폴더 및 의존성 (`pyproject.toml`)
* 독립 폴더: `data/totalcheck_tool`
* 의존성: `opencv-python-headless>=4.13.0`, `pandas>=2.0.0`, `numpy>=1.24.0`, `pyyaml>=6.0`

### 2) `file_io.data_loader` & `file_io.data_exporter`
> **패키지명 `file_io/`**: 파이썬 내장 `io` 모듈과의 이름 충돌을 방지하기 위해 `io/` 대신 `file_io/`를 사용합니다. (`labeling_tool`과 동일)

* `data_loader.py` 내 `load_session_data(stem)`:
  * 3개 경로 존재 확인 후 누락 시 `None` 리턴.
  * 존재 시 `cv2.VideoCapture()`, `pd.read_csv(labeled)`, `pd.read_csv(landmark)` 객체를 묶어서 반환.
* `data_exporter.py` 내 `overwrite_labels(stem, labels)`:
  * 기존 `labeled_data/{stem}.csv`를 읽어와 `gesture` 컬럼만 통째로 교체한 뒤 저장.

### 3) `core.annotation_state` (Model 확장)
* 초기화 시 인자로 기존 `labeled_data`의 `gesture` 리스트를 받아와 내부 멤버 `self.labels`를 구성.
* 나머지는 덮어쓰기(`toggle`, `set_range`) 로직 기존과 100% 동일.

### 4) `ui.skeleton_renderer` (View 확장)
* `draw_landmarks(frame_buffer, landmark_row_series)`: 프레임 배열 위 뼈대 덧그리기.
  * 뼈대 연결선 네트워크 튜플 상수 리스트 배정
  * 색상 관절 녹색, 라인 파란색 하드코딩.

### 5) `main.py` (Controller)
* **대상 큐 확보**: (✨ [6.2-A](#a-file_iodata_loaderpy-로드-로직-유연화) 참고) 기존에는 `labeled_data` 폴더를 기준으로 했으나, Phase 2부터는 `raw_data` ∩ `landmark_data`의 교집합을 기준으로 스캔합니다.
* 영상별 `while True:` 블록:
   * **키 입력 분기**:
     * `0~6` + Numpad `0~6`: 현재 프레임 덮어쓰기. (키코드 하드코딩 혹은 `cv2` 매핑)
      * `Enter(13)`: `data_exporter` 덮어쓰기 수행 → `break` 
      * `n(110 / 78)`: 아무 IO 작업 없이 `break` (다음 영상)
      * `Esc(27)`: `sys.exit(0)`
    * **렌더링 흐름**: `frame_buffer` = 원본이미지 복사 → `skeleton_renderer`로 뼈대 드로잉 → `osd_renderer`로 타임라인과 메뉴 출력 → `cv2.imshow()`.

---

## 6. 추가적인 기능 구현 및 수정 (Phase 2)

### 6.1 요구사항 분석 및 핵심 해결 방향

1. **결과물 통합 저장 (`total_data` 폴더 신설) 및 원본 유지** 
   - 기존에는 `labeled_data`의 라벨만 수정/저장했음.
   - 변경 후: 
     1) 수정한 내용으로 **기존 `labeled_data` 파일을 덮어씌워 최신화**함 (라벨 데이터 버전을 일치시키기 위함).
     2) 이 최신화된 라벨을 원본 좌표 데이터(`landmark_data`)의 빈 `gesture` 컬럼에 결합한 병합본을 `total_data/` 디렉토리에 **동시 생성**.

2. **`labeled_data`가 없는 초기 상태 라벨링 지원 (백지 라벨링)**
   - 기존에는 3종 파일(`raw_data`, `labeled_data`, `landmark_data`)이 모두 있어야만 검수 툴이 켜졌음.
   - 변경 후: `raw_data`와 `landmark_data`만 존재한다면, 모든 프레임의 라벨을 `0`(Neutral)으로 꽉 채운 상태로 라벨링 창을 띄움. 작업을 마치고 Enter(저장)를 누르면 `labeled_data`와 `total_data` 두 곳 모두에 새로운 CSV 파일을 동시 생성.
   

3. **구간 라벨링의 3단계 State Machine UX (완전 개편 및 안전망 확보)**
   - **1단계 (시작 마킹)**: `S` 키 입력 시 시작점 지정.
     - 🌟 화면 상단 경고창(Warning Msg)에 `[Range Start: frame N] Press F to set end, or S to cancel` 표시.
   - **2단계 (취소 기능 추가)**: 이미 `S`가 눌린 1단계 상태나 아래의 3단계 상태에서, 또다시 **`S`**를 누르면 진행 중이던 구간 설정이 즉시 **전면 취소(Cancel)**되고 평시(0단계)로 복귀.
   - **3단계 (종료 마킹 및 클래스 확정 대기)**: `F`를 눌러 종료점을 지정하면, 라벨이 바로 칠해지지 않고 **확정 대기 상태(Waiting)**로 진입.
     - 🌟 화면 상단에 `[WAITING CLASS] Press 0~6 to fill range, or S to cancel` 경고창 고정 출력.
     - 이 대기 상태에서는 **0~6 숫자 키**를 눌러야만 해당 구간이 해당 제스처로 일괄 덮어씌워지며 평시로 복귀함.
     - ⚠️ **키보드 입력 방어 상세**: 
       - **1단계(S 찍힌 상태)**: `F` 또는 `S`(취소) 외의 키 중, 탐색키(`A, D, Q, E, ←, →`)는 허용. 나머지(`스페이스바, Enter, 0~6, n`)는 무시.
       - **3단계(대기 상태)**: `0~6` 확정 키와 `S`(취소)만 허용. 탐색키(`A, D, Q, E, ←, →`), `스페이스바`, `Enter`, `n` 예약어 등은 모두 무시 (실수 원천 차단).
       - 만약 1단계/3단계에서 `n`이 눌렸다면, `[WARNING: Cannot skip! Finish range or press S to cancel]` 경고를 띄웁니다.
     - ⚠️ (단, `Esc`는 입력 가능. `Esc`는 모든 작업을 중단하고 저장 없이 프로그램을 즉시 종료하는 절대키이기 때문.)

4. **스페이스바 토글 규칙 유지**
   - 평시 상태일 때 스페이스바를 누르면 단일 프레임만 `Target_Class ↔ 0`으로 계속 토글됨. (구간 대기 상태에서는 안 눌림)
   - 🌟 이때 `Target_Class`는 파일명에서 추출하는 기존 로직을 무조건 100% 동일하게 유지함.

---

### 6.2 모듈별 상세 수정 및 리팩토링 계획

수정 계획을 직관적이고 알아보게 쉽게 개조식과 표를 섞어 재정리했습니다. 코드를 완전히 새로 짜지 않고 기존 로직을 최대한 이식합니다.

#### 🛠️ A. `file_io/data_loader.py` (로드 시 에러 무시)

* **어디를 수정할건지**: `get_checkable_stems()`, `load_session_data()`
* **수정 내용**:
  1. `get_checkable_stems()`: 
     - 3개 교집합(`labeled_stems & landmark_stems & raw_stems_normalized`)에서 **`labeled_stems`를 제외**.
     - 오직 `raw`와 `landmark` 파일의 교집합 기준으로만 stem을 추출.
  2. `load_session_data()`:
     - `labeled_file.exists()` 검사 통과 시 예외 처리 제거.
     - 파일이 있으면 기존대로 `pd.read_csv`, 없으면 조용히 `labeled_df = None` 할당.

#### 🛠️ B. `file_io/data_exporter.py` (일괄 듀얼 저장소)

* **어디를 수정할건지**: `overwrite_labels()` → `save_all_data(...)`로 전면 개편.
* **수정 내용**:
  * 파라미터: `stem`, `labels`(작업 완료된 1D 배열), `raw_fps`, `landmark_df`(원본 복사본), 경로 2개(`labeled_dir`, `total_dir`)
  * **저장 1 (`labeled_data` 복원/생성)**: 
    - CSV가 이미 있으면: 불러와서 `gesture` 컬럼만 교체. (이때 `landmark_data`에서 뽑은 `frame_idx`와 `timestamp`를 그대로 사용하여 행을 맞춥니다).
    - CSV가 아예 없으면: 빈 DataFrame 생성 후 `frame_idx`, `timestamp`, `gesture = labels` 열을 조립시켜서 새로 저장.
      *(참고: 백지 라벨링 시 `timestamp` 열을 새로 만들 때 쓰는 `format_timestamp(idx, raw_fps)` 유틸 함수는 `landmark_extractor/main.py`에 있던 로직을 `totalcheck_tool/file_io/data_exporter.py` 상단에 동일하게 정의하여 사용합니다.)*
  * **저장 2 (`total_data` 융합 저장)**: 
    - 미리 확보한 `landmark_df` 본사본에 접근.
    - `landmark_df["gesture"] = labels` 코드로 덮어씌운 뒤 `total_data/{stem}.csv`로 통째로 저장(기존 파일이 이미 있다면 전체 덮어쓰기 진행).

#### 🛠️ C. `core/annotation_state.py` (라벨링 State Machine 확장)

* **어디를 수정할건지**: `AnnotationState` 초기화 및 구간 처리 메서드
* **수정 내용**:
  | 메서드명 | 변경 내용 (상태 기계 동작) |
  |---|---|
  | `__init__()` | `existing_labels`가 `None` 또는 빈 리스트로 들어오면, 주어진 `total_frames` 횟수만큼 `[0, 0, ...]` 생성 배정. |
  | `set_range_start()` | 기존 로직 유지 (1단계 마킹) |
  | `set_range_end()` | **(로직 변경)** 기존처럼 덮어쓰지 않고 `self.end_marker = index`만 지정 (3단계 대기 모드 진입). |
  | `cancel_range()` | **(신규)** `start_marker`, `end_marker` 둘 다 `None`으로 날려버림 (S 취소용). |
  | `fill_range_with_class(id)` | **(신규)** `start_marker`부터 `end_marker` 사이를 넘겨받은 `id`로 쫙 칠하고 두 마커 모두 취소. |
  | `is_waiting_for_class()` | **(신규)** 명확한 대기 상태 판별: `return (self.start_marker is not None) and (self.end_marker is not None)` 이어야 True 반환. |


#### 🛠️ D. `main.py` (이벤트 컨트롤러 & OSD 메세지 분기)

* **어디를 수정할건지**: `process_video()` 이벤트 분기문 및 초기화 로직
* **수정 내용**:
  * `limit_frames` 계산 시 방어 로직 (3개 비교): 
    ```python
    if getattr(session, "labeled_df", None) is not None:
        limit_frames = min(video_frames, len(session.landmark_df), len(session.labeled_df))
    else:
        limit_frames = min(video_frames, len(session.landmark_df))
    ```
  * 결과 저장 대상 경로에 상단에 선언해둔 `TOTAL_DIR` 상수 활용.
  * 이벤트 루프 진입 즉시 **OSD Warning 세팅**:
    - `state.is_waiting_for_class()`가 True면: `"[WAITING CLASS] Press 0~6 to fill range, or S to cancel"`
    - `state.is_range_active()`만 True면: `f"[Range Start: frame {state.start_marker}] Press F to set end, or S to cancel"`
  * **키보드 입력 방어 로직 재설계 (우선순위 부여)**:
    1. **`Esc`**: 상태 불문 즉시 강제 종료. (`sys.exit(0)`)
    2. **대기 모드 방어**: 현재 `is_waiting_for_class()` 상태라면 오직 `0~6` 숫자키, `S`(취소), `Esc` 세 가지만 허용. 이외의 입력(←/→ 화살표, A, D, E, Q, 스페이스바, Enter 등)은 모두 무시(`continue`).
    3. **`S`** 키: 진행 중(대기 모드 포함)이면 `state.cancel_range()`, 평시면 `state.set_range_start()`.
    4. **`F`** 키: 평시면 무시, `S`가 찍힌 상태면 `state.set_range_end()`.
    5. **`0~6`** 키: 대기 상태면 `fill_range_with_class(id)`, 평시면 1개 프레임 변경. 
       *(참고: 1 프레임 개별 변경 시 호출하는 `state.set_label(idx, id)`는 기존 `AnnotationState` 클래스 내부에 이미 정의되어 있는 메서드입니다. 재사용합니다)*
    6. **스페이스바**: 평시에만 동작토록 하되 로직(`toggle_frame()`)은 기존 그대로 보존 (`target_class` 유지).
---

### 6.3 구현 시 보호 및 주의사항 체크리스트

- [x] **Data Exporter 행(Row) 길이 불일치 방어**: `save_all_data()`에서 `landmark_df.iloc[:limit].copy()`로 길이를 잘라낸 뒤 `gesture` 컬럼을 삽입. `labels[:limit]` 슬라이싱도 적용.
- [x] **`fps` 파라미터 전달 확인**: `main.py`에서 `video.fps`를 `save_all_data(raw_fps=video.fps)`로 전달. `format_timestamp()`는 `data_exporter.py` 상단에 동일 로직으로 정의.
- [x] **Data Loader 안전망 유지**: `SessionData.labeled_df` 타입 힌트를 `pd.DataFrame | None`으로 변경 완료.
- [x] **동시성 문제와 Side Effect 차단**: `main.py`에서 `save_all_data` 호출 시 `landmark_df.copy()`를 넘겨 in-memory 오염 방지.

---

## 7. Phase 2 구현 기록

### 수정된 파일 4개

| 파일 | 핵심 변경 |
|---|---|
| `file_io/data_loader.py` | `get_checkable_stems`: `raw ∩ landmark` 교집합으로 변경. `load_session_data`: `labeled_df = None` 허용. `SessionData.labeled_df` 타입 → `pd.DataFrame \| None`. |
| `file_io/data_exporter.py` | `overwrite_labels()` 삭제 → `save_all_data()` 신규. `format_timestamp()` 복제 추가. labeled + total 듀얼 저장. |
| `core/annotation_state.py` | `end_marker` 멤버 추가. `set_range_end()` 로직 변경(대기만). `cancel_range()`, `fill_range_with_class()`, `is_waiting_for_class()` 신규. `__init__`에서 `None` 시 백지 초기화. |
| `main.py` | `TOTAL_DIR` 상수 추가. `limit_frames` 3중 방어. 이벤트 루프 3단계 State Machine 분기. 1단계 탐색 허용/기타 차단. 3단계 0~6+S+Esc만 허용. `n`키 경고. `save_all_data` 연동. |

### 기존(Phase 1) 대비 핵심 차이점
- **Phase 1**: labeled_data 3종 파일 전부 필요, Enter 시 labeled_data만 덮어쓰기, S→F 즉시 target_class로 칠함
- **Phase 2**: labeled_data 없이 백지 라벨링 가능, Enter 시 labeled+total 듀얼 저장, S→F→0~6 확정 3단계 UX, S 재입력 취소, OSD 경고, n키 방어

### 추가 수정 (화살표 키 보강)
- 탐색키 목록에 **좌/우 화살표(←/→, 키코드 81/83)** 추가.
- 1단계(S 상태)에서는 화살표로 탐색 허용, 3단계(대기 상태)에서는 차단.
- `main.py`의 1단계 블록에서 기존 `ord('d')/ord('a')` 분기에 이미 `83`/`81`을 포함하고 있어 코드 변경 불필요.