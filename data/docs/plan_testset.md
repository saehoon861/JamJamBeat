# 제스처 데이터셋 수집 및 검수 시스템 설계 계획서

본 문서는 제스처 인식을 위한 테스트셋 데이터 수집기(`capture.py`) 및 수집된 데이터의 품질을 검수하는 도구(`review.py`)의 설계 및 구현 계획을 정의한다.

## 1. 캡처 툴 (`capture.py`)

### 1.1 하드웨어 및 기본 설정
- **카메라 장치**: 웹캠 (Logitech C922 혹은 동급 추천)
- **해상도**: 720p (1280x720) 고정
- **카메라 속성**:
    - 자동 초점 (Auto Focus): ON
    - 자동 화이트 밸런스 (Auto White Balance): ON
    - *비고*: 카메라는 항상 동일한 품질의 이미지를 얻기 위해 고정된 세팅으로 실행된다.

### 1.2 사용자 입력 및 초기화
- **사용자 이름 입력**:
    - 프로그램 실행 시 터미널을 통해 `user_name`을 입력받는다.
    - **제약 사항**: 10자 이내의 영문/숫자 조합.
    - **방어 로직**: 10자 초과 시 재입력 유도 및 특수문자 필터링.
- **Condition (촬영 상태) 선택**:
    - 캡처 직전, 현재 촬영 환경/동작의 특이사항을 반드시 명시해야 한다.
    - 단축키: `b`(BaseP), `r`(RollP), `p`(PitchP), `y`(YawP), `n`(NoneNetural), `o`(NoneOther)
    - 툴 화면 상단에 현재 선택된 `Condition`을 실시간으로 표시한다.
    - **방어 로직**: `Condition` 미선택 시 캡처 기능(`0~6`)을 비활성화하고, 상단에 "Condition을 선택해 주세요" 문구를 출력한다.

### 1.3 데이터 수집 및 저장

> **⚠️ 비정상 종료 시 데이터 유실 주의**
> - 이미지(`cv2.imwrite`)는 캡처 즉시 디스크에 저장되므로, Ctrl+C, 프로그램 Crash, OS 강제 종료 등 비정상 종료 상황에서도 **이미지는 보존**된다.
> - 반면, 랜드마크 데이터는 프로그램 종료 시점에 일괄 저장(`memory_buffer → CSV`)하므로, **비정상 종료 시 해당 세션의 랜드마크 CSV는 저장되지 않는다.**
> - 따라서 비정상 종료 후에는 저장된 이미지와 CSV의 `frame_idx`가 불일치할 수 있으며, 이 경우 해당 이미지는 수동으로 삭제해야 한다.

#### 손 미검출 시 처리 정책
- MediaPipe가 손을 감지하지 못한 경우, 캡처 및 랜드마크 저장을 **즉시 차단**한다.
- 화면에 "No hand detected" 메시지를 출력하고, **2초 후 자동으로 사라지게** 처리한다.
    - 구현: `time.time()`으로 메시지 발생 시각(`no_hand_ts`)을 기록하고, 매 프레임마다 `time.time() - no_hand_ts < 2.0` 조건을 만족할 때만 텍스트를 렌더링한다.
- *이유*: 손 미검출 데이터가 CSV에 섞이는 것을 원천 차단하기 위함이다.

#### 캡처 방식
- `0~6` 숫자 키 입력 시 해당 프레임을 즉시 캡처한다.
- **Key Down 이벤트만 처리**: key-repeat(키 누름 지속) 이벤트는 무시하고 최초 눌림 이벤트만 처리한다.
    - *이유*: 키를 누르고 있는 동안 중복 캡처가 발생하는 것을 방지하기 위함이다.

#### 이미지 저장
- **저장 시점**: 캡처 동시에 즉시 저장한다.
- **저장 경로**: `data/testset/testdata/images/{gesture}/`
- **파일명 규칙**: `{gesture}_{frame_idx}_{condition}_{user_name}.jpg`
    - 예시: `1_23_BaseP_man1.jpg`

#### 순번 관리 (Indexing)
- **원칙**: 글로벌 인덱스(전체 통합 순번)가 아닌, **각 제스처 클래스별 독립적인 `frame_idx`** 를 사용한다.
    - `gesture=0`의 `frame_idx`와 `gesture=1`의 `frame_idx`는 서로 독립적으로 카운팅된다.
- **초기화 방법**: 프로그램 실행 시 기존 `landmarks_{gesture}.csv` 파일의 **마지막 행**을 읽어 `frame_idx` 값을 파악하고, 그 다음 번호부터 순번을 증가시킨다.
    - *주의*: 행의 길이(행 수)를 기준으로 카운팅하면 검수 과정에서 행이 삭제되었을 때 순번이 어긋날 수 있으므로, 반드시 마지막 행의 `frame_idx` 값을 직접 읽어야 한다.
    - CSV가 존재하지 않을 경우 `frame_idx`는 1부터 시작한다.

#### 랜드마크 저장
- **엔진**: MediaPipe Tasks API의 최신 Hand Landmarker를 사용한다. 구체적으로 `hand_landmarker.task` 모델 파일을 직접 로드하는 방식으로 구현한다.
- **설정값**:
    - `max_num_hands = 1`
    - `min_hand_detection_confidence = 0.7`
    - `min_tracking_confidence = 0.7`
- 캡처 시점의 21개 랜드마크 좌표(x, y, z)와 제스처 번호를 메모리 내 리스트(`memory_buffer`)에 누적한다.
- **좌표 기준**: MediaPipe가 반환하는 **normalized 좌표**를 그대로 사용한다.
    - x, y: 이미지 너비/높이를 기준으로 0.0 ~ 1.0 범위로 정규화된 값.
    - z: 손목을 기준으로 한 상대적 깊이 값 (단위: 이미지 너비를 기준으로 정규화).
    - *비고*: 픽셀 좌표로 변환하지 않고 normalized 값을 직접 CSV에 기록한다.

### 1.4 파일 출력
- 프로그램 종료(`q` 혹은 `Esc`) 시, 메모리에 누적된 데이터를 클래스별 `.csv` 파일로 저장한다.
- **저장 정책**:
    - 기존 `landmarks_{gesture}.csv` 파일이 **존재하면** 누적 데이터를 파일 끝에 Append(`mode='a', header=False`) 한다.
    - 기존 파일이 **존재하지 않으면** 헤더를 포함하여 새 파일로 생성한다.
- **CSV 파일 구조**:
    - 컬럼명: `frame_idx`, `timestamp`, `gesture`, `x0`, `y0`, `z0`, ..., `x20`, `y20`, `z20`
    - `frame_idx`: 각 제스처 클래스별 고유 순번.
    - `timestamp`: 현재 단계에서는 `null`로 처리한다.
    - `gesture`: 0~6 중 선택된 값.

### 1.5 코드 주석 가이드
모든 함수와 주요 로직 블록에는 **한국어 주석**을 작성하여 팀원이 코드를 처음 보더라도 바로 파악할 수 있도록 한다.
- **함수 단위**: 함수 상단에 역할, 파라미터, 반환값을 명시한다.
- **주요 로직**: 조건 분기, 인덱싱 처리, 파일 저장 등 핵심 구간에 목적을 설명하는 인라인 주석을 붙인다.
- 예시:
    ```python
    # 각 클래스별 frame_idx 초기값을 CSV 마지막 행에서 읽어 초기화
    # 파일 없을 시 1부터 시작
    def load_initial_indices(gestures: list[int]) -> dict[int, int]:
        ...
    ```

---

## 2. 검수 툴 (`review.py`)

- **검수 단위**: 제스처 클래스별로 CSV 파일을 기준으로 독립적인 검수를 진행한다.
    - 실행 예시: `python review.py --gesture 1`
    - `--gesture` 인자는 **필수**이며, 생략할 경우 에러 메시지를 출력하고 즉시 종료한다 (`argparse`의 `required=True` 설정).
    - 한 번에 하나의 클래스만 검수하며, 결과는 해당 클래스의 CSV에만 반영된다.
- **실행 전제 조건 및 예외 처리**:
    - `landmarks_{gesture}.csv` 파일이 **없을 경우**: 검수 툴을 실행할 수 없다. 에러 메시지를 출력하고 즉시 종료한다.
    - CSV에는 `frame_idx`가 존재하지만 해당 **이미지 파일이 없는 경우**: 이미지를 표시하지 않고 3D 랜드마크만 렌더링한다. 이미지 영역은 빈 영역 또는 "이미지 없음" 플레이스홀더로 표시한다.

### 2.1 UI/UX 구성 (Dash + Plotly)
- **2분할 화면 구성**:
    - **좌측**: 촬영된 실제 이미지 (`.jpg`) 표시 (`html.Img`).
        - **이미지 캐시 버스팅**: Dash의 `html.Img`는 `src` URL이 같으면 브라우저가 이전 이미지를 캐싱하여 다른 프레임임에도 동일한 이미지가 표시되는 문제가 있다. 이를 방지하기 위해 이미지 `src` 경로에 `?t={int(time.time()*1000)}` 형태의 쿼리스트링을 붙여 브라우저가 매번 강제로 이미지를 새로 로드하도록 처리한다.
    - **우측**: 해당 프레임의 21개 핸드 랜드마크 3D 시각화 (`go.Scatter3d`).
- **동기화**: 선택된 제스처의 `frame_idx`를 기준으로 이미지와 랜드마크 데이터를 1:1 일치시켜 표시한다.
    - **이미지 검색 패턴**: `{gesture}_{frame_idx}_*.jpg` glob 패턴으로 이미지 파일을 탐색한다.
    - condition이 다르더라도 `gesture`와 `frame_idx`, `user_name`이 일치하면 해당 이미지를 표시한다.
- **3D 시각화 설정**:
    - **축 비율 고정**: `layout.scene.aspectmode = 'cube'`로 설정하여 손 모양에 따라 그래프가 길쭉하게 왜곡되는 현상을 방지한다.
    - **범위 고정**: x, y, z 각 축의 범위를 동일한 고정값(예: `-1 ~ 1`)으로 설정하여, 어떤 손 모양이든 항상 동일한 정사각형 무대 안에 표시되도록 한다.
    - **y축 반전**: MediaPipe의 y좌표는 이미지 상단이 0, 하단이 1 (아래로 갈수록 증가)이다. 3D 그래프에서는 y축을 반전(`autorange='reversed'`)하여 시각적으로 자연스럽게 손이 위를 향하도록 표시한다.
    - **관절 연결선(Edges)**: MediaPipe Hand의 21개 랜드마크를 아래 연결 순서에 따라 `go.Scatter3d`의 선분으로 시각화한다.
        ```
        # 손목(0)에서 각 손가락 뿌리까지
        HAND_EDGES = [
            (0,1),(1,2),(2,3),(3,4),          # 엄지
            (0,5),(5,6),(6,7),(7,8),           # 검지
            (0,9),(9,10),(10,11),(11,12),      # 중지
            (0,13),(13,14),(14,15),(15,16),    # 약지
            (0,17),(17,18),(18,19),(19,20),    # 소지
            (5,9),(9,13),(13,17),              # 손바닥 가로 연결
        ]
        ```

### 2.2 검수 로직
- **탐색 버튼**: 화면 **중앙 하단**에 "Prev" / "Keep" / "Drop" 버튼을 순서대로 배치한다.
    - **Prev**: 이전 프레임으로 이동. 현재 인덱스가 0일 경우 동작하지 않는다 (음수 인덱스 방지 방어 로직).
        - **재결정 지원**: Prev로 돌아간 후 Keep/Drop을 다시 선택할 수 있어야 한다. 이전에 해당 프레임을 Drop 했다면 `drop_list`에서 해당 `(gesture, frame_idx)`를 **제거**하여 결정을 취소한다.
    - **Keep**: 현재 프레임을 유지하고(drop 하지 않고) 다음 프레임으로 이동.
    - **Drop**: 현재 프레임을 `drop_list`에 `(gesture, frame_idx)` 형태로 추가하고 다음 프레임으로 이동.
- **기본 상태**: 기본값은 "Keep"으로 간주한다. 사용자가 명시적으로 "Drop"을 누른 프레임만 기록한다.
- **동작 방식**: 데이터를 순차적으로 탐색하며, 랜드마크 추론이 부정확하거나 제스처가 잘못된 프레임을 "Drop" 처리한다.

### 2.3 데이터 후처리
- **후처리 트리거**: 마지막 프레임(인덱스 끝)에서 "Keep" 혹은 "Drop" 버튼을 누르는 순간 후처리를 자동으로 실행한다. 별도의 "완료" 버튼은 두지 않는다.
- 후처리는 아래 순서로 수행한다.
1. **CSV 업데이트**: `drop_list`에 포함된 `frame_idx`에 해당하는 행을 `landmarks_{gesture}.csv`에서 영구 삭제하고 덮어쓴다.
2. **이미지 이동**: `drop_list`에 포함된 이미지를 `data/testset/testdata/drop_images/` 폴더로 이동시킨다 (`shutil.move`).
    - **사전 준비**: `review.py` 실행 시작 시점에 `drop_images/` 폴더의 존재 여부를 확인하고, 없으면 `os.makedirs`로 자동 생성한다.
    - 이동 후 파일명은 원본 그대로 유지: `{gesture}_{frame_idx}_{condition}_{user_name}.jpg`
3. **최종 요약 리포트 출력**:
    - 총 제거된 프레임 수.
    - 클래스별 제거된 `frame_idx` 리스트.

### 2.4 코드 주석 가이드
`capture.py`와 동일하게 **한국어 주석**을 작성한다.
- Dash 콜백(`@app.callback`) 함수에는 반드시 트리거 조건, 입출력 데이터의 의미를 명시한다.
- CSV 필터링, 이미지 이동 로직 등 부수 효과(Side-effect)가 있는 구간에는 동작 목적을 명확히 기술한다.

---

## 3. 예상 폴더 구조

```text
JamJamBeat/
├── data/
│   └── testset/             # 테스트셋 관련 루트
│       ├── capture.py       # 데이터 수집 스크립트
│       ├── review.py        # 데이터 검수 스크립트
│       └── testdata/        # 수집 데이터 저장소
│           ├── images/
│           │   ├── 0/       # Gesture 0 (No Gesture)
│           │   │   ├── 0_1_NoneNetural_man1.jpg
│           │   │   └── ...
│           │   ├── 1/       # Gesture 1 (Fist) 이미지들
│           │   ├── 2/       # Gesture 2 (OpenPlam)
│           │   ├── 3/       # Gesture 3 (V)
│           │   ├── 4/       # Gesture 4 (Pinky)
│           │   ├── 5/       # Gesture 5 (Fox)
│           │   └── 6/       # Gesture 6 (KHeart) 이미지들
│           ├── landmarks_0.csv
│           ├── landmarks_1.csv
│           ├── ...
│           ├── landmarks_6.csv
│           └── drop_images/ # 검수 시 Drop된 이미지들 (자동 생성)
│               ├── 0_1_NoneNetural_man1.jpg
│               └── ...
└── ...
```

---

## 4. 실행 환경 및 의존성 명세

### 4.1 실행 환경
- **Python 버전**: `>=3.11` (`pyproject.toml` 기준)
- **패키지 관리**: `uv` 가상환경 사용
    - 실행: `uv run python capture.py` / `uv run python review.py --gesture 1`

### 4.2 사용 라이브러리 및 버전 검증

| 라이브러리 | `pyproject.toml` 요구 버전 | 계획에서 사용하는 주요 기능 | 지원 여부 |
|------------|---------------------------|-----------------------------|-----------|
| `mediapipe` | `>=0.10.32` | `HandLandmarker`, `.task` 파일 로드 API (Tasks API v2) | ✅ 지원 |
| `opencv-python` | `>=4.8.0` | `VideoCapture`, `imwrite`, `putText`, `waitKey` | ✅ 지원 |
| `pandas` | `>=3.0.1` | `read_csv`, `DataFrame.to_csv`, `mode='a'` | ✅ 지원 |
| `numpy` | `>=1.26.0` | 랜드마크 좌표 배열 처리 | ✅ 지원 |
| `dash` | **미설치** | `html.Img`, `@app.callback`, Dash 레이아웃 구성 | ❌ **설치 필요** |
| `plotly` | **미설치** | `go.Scatter3d`, `layout.scene` 설정 | ❌ **설치 필요** |

### 4.3 패키지 설치 계획
- `dash`와 `plotly`는 `review.py`에서만 사용되며, 현재 가상환경에 설치되어 있지 않다.
- `pyproject.toml`에 의존성을 추가하고 `uv sync`로 설치한다.
    ```toml
    # pyproject.toml dependencies에 추가
    "dash>=2.18.0",
    "plotly>=6.0.0",
    ```
    ```bash
    uv sync
    ```

---

## 5. 파일별 구현 계획

### 5.1 `capture.py` 파이프라인

```
0. 시작 시 환경 확인
   → data/testset/testdata/images/0~6/ 폴더 존재 여부 확인
   → 없으면 os.makedirs로 자동 생성

1. 초기화
   → user_name 입력 및 검증 (10자 이내, 영문/숫자만 허용)
   → 각 gesture 클래스별 landmarks_{gesture}.csv의 마지막 행에서
      frame_idx를 읽어 다음 순번으로 초기화 (행 수 기준 X, 마지막 행 값 O)
   → CSV 없을 시 해당 클래스의 frame_idx는 1부터 시작
   → MediaPipe HandLandmarker 로드 (hand_landmarker.task, IMAGE 모드)
      설정값: max_num_hands=1, min_hand_detection_confidence=0.7,
              min_tracking_confidence=0.7

2. 메인 루프 (cv2.VideoCapture)
   → 프레임 획득 → MediaPipe 추론 → 랜드마크 오버레이
   → 화면 상단에 현재 user_name, condition, 키 도움말 표시
   → 손 미검출 시 "No hand detected" 출력 (no_hand_ts 기록, 2초 후 소멸)
   → 키 이벤트 처리 (key down 최초 이벤트만, key-repeat 무시):
       b / r / p / y / n / o → condition 변수 업데이트
       0 ~ 6                 → condition 체크 → 손 감지 체크 → 캡처 실행
                               cv2.imwrite() 즉시 저장 시도
                               imwrite 성공 → memory_buffer에 랜드마크 누적
                               imwrite 실패 → 해당 프레임 discard
                                             (메모리에 추가하지 않음)
                                             화면에 "저장 실패" 경고 1초 표시
                                             (save_fail_ts 기록, 1초 후 소멸)

3. 종료 (q / Esc)
   → memory_buffer를 pandas DataFrame으로 변환
   → 기존 landmarks_{gesture}.csv 존재 시: mode='a', header=False로 Append
   → 없을 시: 헤더 포함 신규 파일 생성
```

### 5.2 `review.py` 파이프라인

```
0. 인자 처리
   → argparse로 --gesture 인자 수신 (required=True)
   → --gesture 없이 실행 시 에러 메시지 출력 후 즉시 종료

1. 데이터 로드 및 유효성 검증
   → landmarks_{gesture}.csv 존재 확인 → 없으면 에러 종료
   → CSV 로드 (pandas)
   → images/{gesture}/ 폴더에서 이미지 경로 리스트 구성
   → frame_idx 기준으로 CSV 행과 이미지 파일을 매핑
      (이미지 없는 frame_idx는 이미지 영역을 빈 플레이스홀더로 처리)
   → drop_images/ 폴더 존재 여부 확인 → 없으면 os.makedirs로 자동 생성

2. Dash 앱 구동
   → 2분할 레이아웃: 좌측(html.Img 이미지) / 우측(go.Scatter3d 3D 랜드마크)
   → 이미지 src URL에 ?t={timestamp}(ms) 쿼리스트링 추가 (캐시 버스팅)
   → Plotly 3D 설정:
       - layout.scene.aspectmode = 'cube'
       - 각 축 range = [-1, 1] 고정
       - y축 autorange='reversed' (MediaPipe y축 방향 보정)
       - 손가락 HAND_EDGES 연결선 포함 시각화
   → 하단 중앙에 "Prev" / "Keep" / "Drop" 버튼 순서대로 배치
   → Prev 클릭:
       - 인덱스 0이면 동작 안 함
       - 이전 프레임에 drop 결정이 있었으면 drop_list에서 제거 (재결정 가능)
   → Drop 클릭: drop_list에 (gesture, frame_idx) 추가, 다음으로 이동
   → Keep/Drop 클릭 시 마지막 프레임이면 자동으로 후처리 실행

3. 검수 완료 후처리 (마지막 프레임에서 Keep/Drop 클릭 시 자동 실행)
   ① drop_list 기반으로 CSV 해당 행 필터링 후 덮어쓰기
   ② drop된 이미지를 shutil.move로 drop_images/ 폴더에 이동 (원본 파일명 유지)
   ③ 최종 리포트 출력:
       - 총 Drop 수
       - 클래스별 Drop된 frame_idx 리스트
```
