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

### 데이터 입력 로드 및 검증
`data_loader.py`에서 영상 1개를 처리할 때 다음 3개가 모두 존재해야 합니다:
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
* **대상 큐 확보**: `labeled_data` 폴더의 `.csv` 목록을 기준으로 스캔합니다.
* 영상별 `while True:` 블록:
   * **키 입력 분기**:
     * `0~6` + Numpad `0~6`: 현재 프레임 덮어쓰기. (키코드 하드코딩 혹은 `cv2` 매핑)
     * `Enter(13)`: `data_exporter` 덮어쓰기 수행 → `break` 
     * `n(110 / 78)`: 아무 IO 작업 없이 `break` (다음 영상)
     * `Esc(27)`: `sys.exit(0)`
   * **렌더링 흐름**: `frame_buffer` = 원본이미지 복사 → `skeleton_renderer`로 뼈대 드로잉 → `osd_renderer`로 타임라인과 메뉴 출력 → `cv2.imshow()`.