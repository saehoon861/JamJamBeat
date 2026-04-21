# 오프라인 데이터 증강 플랫폼 구현 계획 (Implementation Plan)

본 문서는 `src/dataset` 디렉터리 내부에서 오프라인 데이터 증강 및 시각화를 수행하기 위한 구체적인 구현 계획입니다. 유저의 요구사항에 따라 디렉터리 구조를 약간만 분리하여 직관적인 진입점을 마련하고, 데이터 추적성을 확보하며, 기존 코드 의존성을 100% 배제한 완전히 독립적인 파이프라인으로 설계했습니다.

---

## 1. 파이프라인 개요 및 데이터 저장 전략

전체 데이터 처리 파이프라인은 다음과 같이 3단계로 나뉩니다.

**1단계: 정제 및 정규화 (Cleaning & Normalization)**
- **입력**: `raw_data.csv` (원본 그대로의 결측치가 포함된 데이터)
- **과정**: 결측치(NaN) 행을 버리는 정제(`dropna`)를 수행한 뒤, 랜드마크의 원점을 손목으로 옮기고 손 크기에 비례해 스케일을 맞추는 **정규화(Normalization)** 작업을 수행합니다.
**1.5단계: 스마트 언더샘플링 (Smart Undersampling)**
- **입력**: `normalized_data.csv`
- **문제점**: Class 0(No Gesture) 데이터가 약 25,043개로 압도적으로 많습니다. 무작정 Random 1:1 비율로 자르면, 오분류 방지를 위해 의도적으로 수집한 소중한 하드 네거티브(Hard Negative) 패치들마저 날아가는 치명적 손실이 발생합니다.
- **과정**: Class 0의 `source_file` 출처 텍스트를 파싱하여 가중치를 둔 계층적(Stratified) 언더샘플링을 적용합니다. 
  - ① `hardneg` 텍스트 포함 파일: **전량 100% 유지** (모델 강건성의 핵심이므로 절대 버리지 않음)
  - ③ 기타 제스처 전이 구간 및 명확하지 않은 동작 구간: **최소화하여 남은 Target 수량만 채움**
  - 이를 통해 Class 0은 `target_majority`(결과값), 나머지 제스처 클래스는 `target_minority`(결과값) 수준으로 스마트하게 조율됩니다.
- **출력**: 데이터의 가치(하드 네거티브)를 보존하면서도 불균형을 해방시킨 `balanced_normalized_data.csv` 로 저장. 증강은 오직 이 데이터만을 입력으로 받게 됩니다.

**2단계: 오프라인 데이터 증강 (Offline Augmentation)**
- **입력**: `balanced_normalized_data.csv` (클래스 불균형이 해소된 정규화 데이터)
- **시나리오 정의 (총 4개로 5배 증강, 데이터의 다양성과 MVP 수준의 필요량을 고려)**:
  - **버전 A (실제 환경 노이즈 모사)**: `Gaussian Noise` 조작 (카메라 흔들림 및 랜드마크 픽셀 떨림 특화)
  - **버전 B (카메라 촬영 각도 및 거리 모사)**: `3D Rotation (x,y,z축)` + `Global Scale Variation` 조작
  - **버전 C (손의 해부학적 신체 특징 다양화)**: `Bone Length Perturbation` + `Joint Angle Perturbation` 조작
  - **버전 D (극한 환경 / Hard 셋)**: 위 A, B, C를 모두 약하게 적용 (조합)
- **증강 시나리오 기하학(Geometry) 피드백 반영 사항**:
  - ❌ `Wrist Shake` **제거**: 이미 1단계 정규화에서 손목을 원점(0,0,0)으로 Translation 했기 때문에, 여기서 다시 손목 위치를 흔드는 것은 원점 정규화의 의미를 퇴색시키므로 파이프라인에서 완전히 제외합니다.
  - ❌ `Perspective Distortion` **제거**: MediaPipe의 랜드마크는 이미 Camera Projection이 완료된 정규화 좌표계입니다. 이 상태에서 원근 왜곡을 다시 먹이면 손 구조 자체가 기형적으로 뒤틀릴 극심한 위험이 있습니다.
  - ✅ `Mirroring (Flip)` **추가**: 현재 수집된 데이터가 오른손(`man1_right`) 전용이긴 하나, 3D 공간상 랜드마크의 X좌표를 반전시키면 기하학적 왼손 데이터가 탄생합니다. 미래 확장성 및 좌우 대칭 학습 강건성을 위해 **수평 뒤집기(Horizontal Flip)** 기법을 노이즈 위주의 **시나리오 A**, 카메라 조작 위주의 **시나리오 B**, 그리고 한계 테스트용인 **시나리오 D**에 확률적으로 편입하여 적용합니다.

- **출력 컬럼 상세 정의 (총 67개 컬럼)**:
  - 기존 `man1_right_for_poc.csv`를 참고하여 **최좌측에 `aug_version` 컬럼을 1개 추가**합니다.
  - 시간 정보는 이미 `source_file`과 `frame_idx` 두 컬럼의 조합만으로 완벽하게 순차적 추적이 가능하므로, 헷갈림을 유발하는 **`timestamp` 컬럼은 삭제(Drop)** 합니다.
  - `[aug_version(예: original, aug_A, aug_B), source_file, frame_idx, gesture, x0, y0, z0, ..., x20, y20, z20]`
- **결과물 저장 전략 (추적성 확보)**:
  - 파일을 하나로 뭉치지 않고 시나리오별로 나누어 저장합니다. (`csv_data/augmented/aug_A.csv`, `aug_B.csv` ...) 
  - 이를 통해 사용자가 특정 시나리오만 시각화하거나 문제가 생긴 시나리오를 특정하여 버리기가 매우 쉬워집니다. 시각화 시에는 해당 파일만 가져와서 검토하면 됩니다.

**3단계: 육안 검증 및 시각화 (Visualization)**
- **입력**: `augmented` 폴더에 저장된 시나리오별 CSV 파일
- **사용 툴 및 버전**: `plotly.express` (버전 최신 5.x) 또는 Python 기본 `matplotlib`의 `mplot3d`를 사용합니다.
- **시각화 방식 (Plotly 적극 권장)**: 
  - `plotly`를 권장하는 이유는 브라우저에 팝업을 띄워서 마우스로 3D 공간을 이리저리 **드래그해서 회전, 줌(Zoom-in/out)** 해볼 수 있기 때문입니다. 손가락이 꼬여있는지 면밀히 보려면 시점 변환이 필수적입니다.
  - 각 프레임 단위로 원본과 증강버전의 랜드마크를 두 개의 색상(예: 원본 파란색, 증강 빨간색)으로 스켈레톤의 뼈대를 선분으로 이어서 동시에 그려 비교합니다.
  - **💡 Plotly 버전 및 기능 지원에 대한 답변**:
    - **3D 인터랙션 지원 시기**: Plotly의 3D Scatter (드래그하여 회전, 줌인/아웃, 패닝(Pan)) 기능은 최신 기능이 아니라, **2017년(버전 2.x 시절)부터 기본으로 탑재되어 온 Plotly 3D 그래프의 Core 정체성**입니다. 따라서 어떤 버전을 써도 해당 인터랙션은 완벽하게 100% 지원됩니다.
    - **버전 추천 (`plotly==5.24.1` 권장)**: 파이썬 생태계에서는 메이저 변경(예: 3에서 4, 4에서 5) 직후 `.0` 버전은 피하는 것이 정규 관례입니다. 그러나 Plotly 5.x 시리즈는 이미 수년 전에 출시되어 현재 `5.24.x` 대역에 이르렀으며, 스택오버플로우 레퍼런스가 가장 빵빵하고 버그가 모두 잡힌 매우 안정적인 황금기(Stable) 버전입니다. 따라서 가장 최신의 검증된 5.x 버전을 사용하는 것이 가장 안전하고 호환성이 높습니다. (명시적으로 `uv add "plotly<6.0.0"` 형태로 추가합니다)
  - **❗ 강화된 검증 로직 (Random Sample Auto-Vis) 및 검증 산출물 처리 프로세스**:
    - **무작위 추출(Random Sampling)의 수학적 근거**: 수만에서 수십만 장에 달하는 증강 프레임을 육안으로 일일이 전수 검사하는 것은 물리적으로 불가능합니다. 무작위로 추출한 표본(예: 50개)을 통해 전체 모집단의 증강 로직(회전, 스케일링, 역학 변형 등)에 체계적, 기하학적 결함이 없는지 통계적으로 검증(Sample Test)하는 것이 목적입니다.
    - **자동화된 시각화 보드**: 시각화 스크립트는 매 실행 시 지정된 수량 단위의 무작위 프레임을 한 화면에 Subplots 형태로 띄워, 전문 작업자가 직관적이고 빠르게 '정상/비정상' 패턴을 일괄적으로 비교 대조할 수 있도록 돕습니다.
    - **검증 실패 시나리오 및 해결 프로세스**:
      - 만약 표본 검사 중 '손가락 관절이 기형적으로 꼬이는 현상' 등 심각한 증강 실패(Augmentation Failure)가 다수 목격된다면, 이는 특정 프레임의 문제이기 보다 **해당 시나리오의 파라미터(예: 회전 범위, 노이즈 분산치 등)가 과도하게 설정되었거나 수학적 로직에 오류가 있음을 시사**합니다.
      - **제거 및 재실행 전략**: 잘못된 프레임만 개별적으로 솎아내는(Row-level drop) 방식이 아니라, 불량률이 확인된 해당 증강 결과 파일(예: `aug_C_anatomy.csv`) 전체를 폐기합니다. 직후 `augmenters.py`의 설정값을 조정한 뒤 스크립트(`02_run_augmentation.py`)를 재구동(Re-run)하여 무결성을 확보합니다.
      - **최종 승인**: 교차 표본 검증을 통해 시각적 오류가 발견되지 않는다면, 해당 증강 데이터 배치는 학습을 위한 안전 세트로 공식 인계됩니다.

---

## 2. 디렉터리 구조 설계 (Depth 1.5 - 직관적 분리)

유저님의 "진입점이 한 곳에 묶여있고 이름 구분이 안 되어서 구분하기 어렵다"는 의견 및 "기존 코드와 독립성" 피드백을 적극 수용하여 구조를 재설계했습니다.

- **독립성 원칙**: 기존 학습 과정에 묶여있던 `src/preprocessing/`과 `src/dataset/gesture_dataset.py` 등의 코드는 의존성을 피하기 위해 **전부 무시**합니다. 해당 파이프라인 외부로 어떤 Import도 하지 않고 완전히 안전 지대(Safe Zone)에서 재창조합니다.
- **실행 분리 원칙**: 진입점(`uv run` 할 파일)과 핵심 로직 함수를 잘 모를 수 없도록 분리하고, 넘버링(`01_`, `02_`)을 추가하여 워크플로우 순서를 강제합니다.

```text
src/dataset/
├── README.md               # 파이프라인 사용법 설명
├── plan.md                 
├── research.md             
│
├── offline_pipeline/           # 묶음: 오프라인 파이프라인 메인 폴더
│   ├── 01_run_preprocess.py    # (진입점) 1단계: raw -> normalized CSV 저장 스크립트
│   ├── 02_run_augmentation.py  # (진입점) 2단계: normalized -> 시나리오별 aug_A, B CSV 저장
│   ├── 03_run_visualization.py # (진입점) 3단계: Plotly 기반 시각화 팝업 스크립트
│   │
│   └── modules/                # 핵심 로직 함수들 (진입점 스크립트에서만 import됨)
│       ├── __init__.py
│       ├── normalizer.py       # 원점 이동, 스케일링 함수
│       └── augmenters.py       # Rotation, Noise, Perturbation 등 개별 함수
│
└── csv_data/                   # 데이터 저장소
    ├── raw/
    │   └── man1_right_for_poc.csv  # 유저가 제공한 원본 데이터
    ├── normalized/
    │   └── normalized_data.csv     # 1단계 출력물
    └── augmented/                  # 2단계 출력물 (시나리오별 분리 저장)
        ├── aug_A_noise.csv
        ├── aug_B_rotation.csv
        ├── aug_C_anatomy.csv
        └── aug_D_hard.csv
```

---

## 3. 개발 환경(uv) 및 라이브러리 스펙 / 함수 상세 설계

### 3.1. 가상환경 (uv) 및 라이브러리 사용 전략

**💡 가상환경 전략에 대한 답변 (🔥 기존 `.venv` 환경 유지/공유 적극 추천 🔥)**
- 현재 프로젝트 루트의 `pyproject.toml`을 확인해 본 결과, `numpy>=1.26.0`, `pandas>=3.0.1`, `torch>=2.10.0` 등 증강 데이터 처리에 필요한 무거운 코어 라이브러리들이 이미 아주 잘 정리되어 설치되어 있습니다. 
- **오프라인 증강을 위해 새로운 가상환경을 파는 것은 강하게 비추천합니다.**
  - **이유 1**: 추후 증강된 데이터를 결국 `src/training`에 있는 MLP 모델(PyTorch)에서 불러와야 하는데, 데이터를 만들 때의 환경(Numpy, Pandas 버전)과 학습시킬 때의 환경이 다르면 의도치 않은 패키지 충돌이나 데이터 직렬화 버그가 발생할 수 있습니다.
  - **이유 2**: 팀원들이 이미 공유하는 메인 환경이 잘 잡혀있다면(`uv.lock` 존재), 여기에 단지 시각화용 패키지인 `plotly`만 가볍게 주입(`uv add plotly`)하여 하나의 무결점 환경으로 관리하는 것이 실무적으로 압도적으로 편리하고 안전합니다.

**✅ 사용 라이브러리 및 메서드 검증 (의존성 스펙)**
- **`numpy (>=1.26.0)`**: (현재 pyproject.toml 기준) 벡터 연산. `np.linalg.norm`, `np.dot` (회전 행렬 적용) 등 핵심 수학 메서드 안정성 O.
- **`pandas (>=3.0.1)`**: (현재 pyproject.toml 기준) DataFrame 조작. `pd.read_csv`, `dropna`, `to_csv` 메서드 완벽 지원. 최신 3.x 버전의 문법 활용.
- **`plotly (>=5.24.0, <6.0.0)`**: `plotly.graph_objects.Scatter3d` 사용. (위 가상환경 공유 결정을 따라, 현재 환경에 `uv add plotly` 명령어로 추가하여 진행)

---

### 3.2. 모듈/함수 수준의 상세 아키텍처 및 구현 스펙

코드를 짜기 전, `modules/` 내부에 어떠한 함수들이 작성되고, 어떻게 연결되는지 파라미터 수준으로 명세합니다. 아래 명세된 메서드는 위에서 정의한 라이브러리 버전 스펙과 100% 문법이 일치합니다.

### 3.1. `modules/normalizer.py`
전처리 및 데이터 밸런싱(언더샘플링) 로직을 전담합니다. 

1. **`clean_missing_values(df: pd.DataFrame) -> pd.DataFrame`**
   - **기능**: 프레임 중 하나라도 좌표 결측치가 있는 Row를 판별하여 삭제(`dropna`).
2. **`normalize_landmarks(landmarks: np.ndarray) -> np.ndarray`**
   - **기능**: 손목을 원점(0,0,0)으로 이동시키고, 손목~중지 첫 마디 길이를 1로 스케일링합니다.
3. **`apply_undersampling(df: pd.DataFrame, target_minority_ratio: float = 1.0, hardneg_keyword: str = 'hardneg', neutral_keyword: str = 'neutral') -> pd.DataFrame`**
   - **기능**: 단순 1:1 랜덤 컷팅 구조가 아니라, `majority_class`(Class 0)의 내부 구성을 파악하고 보존 가치에 따라 차등 컷오프(Cut-off)를 수행합니다.
   - **로직 개선 (하드코딩 제거 및 무한 확장 대응)**: 
     - 6000, 2000 같은 고정된 숫자를 파라미터에서 완전히 제거했습니다. 데이터가 늘어날 때마다 코드를 고쳐야 하는 병목을 없애기 위함입니다.
     - **Min Class 탐색**: 전체 클래스 중 가장 개수가 적은 클래스의 샘플 수(`min_count`)를 먼저 동적으로 찾습니다.
     - **비율 적용**: 다른 모든 제스처 클래스들은 이 `min_count * target_minority_ratio`(기본 1.0)에 맞춰 샘플링됩니다.
     - **Class 0 전략**: `source_file`의 텍스트 파싱을 함수 외부에서 키워드 추출용 정규식이나 설정 파일(Config)로 빼내어 데이터셋 규칙 변경에 대응하게 짰습니다. (현재는 직관성을 위해 기본 키워드 파라미터로 제공). 파싱 후 `hardneg` 샘플은 무조건 100% 살리고, 남은 할당량(`(min_count * ratio) - hardneg_count`)의 70%는 `neutral`에서, 30%는 찌꺼기 구간에서 동적 확률로 채워넣는 수학적 분배 수식을 적용합니다.
   - **입력**: (N, 21, 3) 차원의 Numpy 배열 (N=행 개수, 21=랜드마크, 3=x,y,z 스칼라)
   - **수식/구현 로직**:
     1. 손목 원점 맞추기: 모든 랜드마크에서 손목(`Index 0`)의 좌표 벡터를 뺍니다. `landmarks = landmarks - landmarks[:, 0:1, :]`
     2. 글로벌 스케일 맞추기: 손목(`Index 0`)과 중지 첫 번째 마디(`Index 9`) 사이의 유클리드 거리를 구합니다. 이 거리가 1이 되도록 모든 랜드마크 좌표를 나눕니다. `scale = ||landmarks[:, 9, :] - landmarks[:, 0, :]||`, `landmarks = landmarks / scale`
   - **출력**: 손목이 (0,0,0)에 오고 손바닥 크기가 1인 (N, 21, 3) 랜드마크.

### 3.2. `modules/augmenters.py`
각각의 증강 기법을 독립적인 순수 Numpy 수학 함수로 구현합니다. 모든 변환 함수들은 `gesture_class` 파라미터를 선택적으로 받아 클래스별로 변형 강도를 다르게 스케일링할 수 있도록 유연하게 설계됩니다.

**💡 스켈레톤 증강 수치(Range) 및 난수 분포 명확화 기준**
- 본 데이터셋 좌표계는 손목~중지 길이가 약 `1.0` 스케일로 정규화된 3D 로컬 공간입니다.
- 타 3D Hand Pose 논문들의 데이터 어그멘테이션 휴리스틱을 차용하여, 형태 붕괴를 일으키지 않는 통계적 안전 범위를 각 함수 초기 파라미터(Default)로 명확히 고정합니다.

1. **`add_gaussian_noise(landmarks: np.ndarray, gesture_class: int = -1, std_dev: float = 0.005) -> np.ndarray`**
   - **방식**: 기존 좌표에 평균 0, `std_dev=0.005`인 정규분포(Gaussian) `np.random.normal(0, std_dev)` 난수 벡터를 더합니다.
   - **범위 제한**: 노이즈가 손가락 구조를 뚫고 튀어나가지 않게 `[-0.015, 0.015]` (즉, 3시그마 컷오프) 사이로 클리핑(`np.clip`) 처리합니다. 이는 실제 렌즈 Jittering 오차율인 1.5% 수준에 해당합니다.
2. **`apply_3d_rotation(landmarks: np.ndarray, gesture_class: int = -1, max_angle_x: float = 15.0, max_angle_y: float = 15.0, max_angle_z: float = 15.0) -> np.ndarray`**
   - **방식**: 주어진 최대 각도 내에서 균등 분포(`np.random.uniform(-max_angle, max_angle)`)로 `x, y, z` 축 각각의 회전 각도를 독립적으로 추출합니다. 추출 후 라디안 변환을 거쳐 3D 회전 행렬(`Rotation Matrix`)을 점곱합니다.
   - **설정 근거**: 경험상 단일 카메라 기준 15도를 넘어가면 오큘루전(손가락 가림) 현상이 MediaPipe 추론 범위를 완전히 이탈해 노이즈를 양산하므로 `[-15도, 15도]`를 한계선으로 잡습니다.
3. **`apply_global_scale_variation(landmarks: np.ndarray, gesture_class: int = -1, min_scale: float = 0.9, max_scale: float = 1.1) -> np.ndarray`**
   - **방식**: (폐기된 Perspective Distortion과 Wrist Shake를 대체하는 핵심 함수) 전체 랜드마크 배열 좌표에 `np.random.uniform(min_scale, max_scale)`로 추출한 단일 스칼라 크기(Scale) 값을 곱합니다.
   - **효과**: 카메라와 손의 거리가 일시적으로 멀어지거나 가까워지는 줌인/줌아웃 현상을 완벽하게 기하학적 형태의 붕괴 없이 모사합니다. 정규화된 스케일(1.0)에 위배되지 않는 선인 `±10%` 조작을 수행합니다.
4. **`apply_horizontal_flip(landmarks: np.ndarray, gesture_class: int = -1, apply_prob: float = 0.5) -> np.ndarray`**
   - **방식**: `apply_prob`(기본 50% 확률)에 당첨되면, 랜드마크의 X좌표 부호를 반전시켜(`landmarks[:, 0] *= -1`) 기하학적으로 완벽한 반대쪽 손(Mirroring) 모양을 생성합니다. 좌우 대칭성을 학습시켜 손 방향에 대한 강건성을 극대화합니다.
5. **`apply_bone_length_perturbation(landmarks: np.ndarray, gesture_class: int, min_ratio: float = 0.9, max_ratio: float = 1.1) -> np.ndarray`**
   - **방식**: 사람마다 다른 손가락 마디 길이를 고려해 부모-자식 벡터 길이를 `0.9배(~ -10%)`에서 `1.1배(~ +10%)` 사이의 균등 분포 난수로 스케일링하여 자식 좌표를 업데이트합니다. 
   - **클래스 보호 제어**: 주먹 쥐기(`Fist`) 클래스의 경우 오차 허용이 매우 타이트하므로 파이프라인 외부에서 범위를 `[0.95, 1.05]`로 대폭 축소 오버라이드하여 호출합니다.
6. **`apply_joint_angle_perturbation(landmarks: np.ndarray, gesture_class: int, max_angle: float = 5.0) -> np.ndarray`**
   - **[핵심 수학 로직 - Vector Rotation 근사 기반]**: 복잡한 정확 기구학(Kinematic rotation)을 계산하지 않습니다. 대신 각 joint pair의 **Bone Vector(뼈 방향 벡터)**를 추출하여 여기에 무작위 회전 행렬(Rotation Matrix)을 스칼라 내적 방식으로 연산하여, 관절 위치(joint position)가 변화되는 대체 모델을 적용합니다. (이는 Skeleton-based action recognition 연구에서 널리 통용되는 보편적 기법으로 연산 병목을 예방합니다)
   - **방식**: 사람의 손가락 관절 각도 편차를 모사합니다. `[-5도, +5도]` 구간에서 균등 분포 난수를 추출하고, 부모-자식 방향 벡터에 국소 행렬 연산을 선형 대입합니다.
   - **클래스 보호 제어**: 활짝 펼친 손바닥이나 고유의 방향성을 갖는 제스처(예: 'V' 모양의 각도)는 변경 범위를 절반(`[-2.5도, 2.5도]`)으로 제한하여, V 제스처가 의미상 W 제스처로 오인되는 기하학적 붕괴를 예방합니다.

**✅ [Strategy Pattern] 시나리오 파이프라인 디자인 패턴**
단순한 if/else 분기가 아닌 Dictionary 기반의 전략 패턴(Strategy Pattern)을 사용하여 파이프라인을 구축합니다. 특정 증강 버전 추가/삭제 시 딕셔너리 리스트만 수정하면 되는 매우 깔끔하고 확장성 높은 구조입니다.

```python
# augmenters.py
# 시나리오별 파이프라인 전략 매핑
AUGMENTATION_SCENARIOS = {
    "aug_A": [add_gaussian_noise, apply_horizontal_flip],  # 플립 추가
    "aug_B": [apply_3d_rotation, apply_global_scale_variation, apply_horizontal_flip],  
    "aug_C": [apply_bone_length_perturbation, apply_joint_angle_perturbation],
    "aug_D": [add_gaussian_noise, apply_3d_rotation, apply_bone_length_perturbation, apply_horizontal_flip]
}



def scenario_pipeline(landmarks: np.ndarray, version: str, gesture_class: int) -> np.ndarray:
    """
    주어진 전략 버전에 따라 체인 형태로 함수들을 관통시켜 증강을 수행합니다.
    gesture_class 인자를 통해 클래스의 정체성을 무너뜨리지 않도록 강도를 조절합니다.
    """
    pipeline = AUGMENTATION_SCENARIOS[version]
    for fn in pipeline:
        # inspect.signature 등을 활용하거나, kwargs를 통해 선택적 인자 넘기기
        landmarks = fn(landmarks, gesture_class=gesture_class)
    return landmarks
```

### 3.3. 진입점 스크립트 연결 로직 및 파이프라인 상세 아키텍처

진입점 스크립트들은 `modules`에서 정의된 핵심 수학 로직을 가져다 쓰는(Import) 컨트롤러 형태를 띄며, IO(파일 읽고 쓰기) 및 워크플로우 제어를 담당합니다.

1. **`01_run_preprocess.py` (전처리 및 언더샘플링 파이프라인)**
   - **로드**: `pandas`로 `raw/man1_right_for_poc.csv`를 읽어들입니다. `dropna`로 결측치를 즉각 제거합니다.
   - **언더샘플링**: 데이터프레임을 `normalizer.apply_undersampling()`에 통과시킵니다. 내부적으로 `hardneg` 텍스트 출처는 절대 보존되며 클래스 간 `1:1:1` 비율이 맞춰진 쪼그라든(하지만 치명상은 없는) DF가 반환됩니다.
   - **행렬 변환**: DataFrame에서 순수 학습 좌표계(`x0~z20` 컬럼)를 추출해 `(N, 21, 3)` 포맷의 넘파이 3D 텐서로 Reshape합니다.
   - **정규화**: 이 텐서를 `normalizer.normalize_landmarks()`에 통과시켜 손목이 좌표 (0,0,0)에 위치하도록 좌표 이동 브로드캐스팅 수식을 수행합니다.
   - **저장**: 다시 Reshape하여 원래의 DataFrame 형식으로 변환합니다. 이 과정 중 타임스탬프 열은 Drop되고, 최좌측에 `aug_version="original"` 컬럼을 Insert하여 `csv_data/normalized/balanced_normalized_data.csv`로 최종 세이브합니다.

2. **`02_run_augmentation.py` (증강 메인 파이프라인)**
   - **로드**: 앞서 추출해낸 하드 네거티브가 보존된 균형 데이터(`balanced_normalized_data.csv`)를 호출하여 메모리에 적재합니다. (잉여 샘플은 이미 필터링 된 엑기스 상태)
   - **시나리오 브랜치**: `A, B, C, D` 4개의 시나리오에 대해 거대한 `For loop` 블럭을 돕니다.
   - **증강 전략 주입**: 각 루프 블록에서 `augmenters.scenario_pipeline(landmarks, version='aug_A', gesture_class=c)` 를 호출합니다. (Strategy Pattern을 통해 내부 함수들이 연쇄적으로 관통하면서 스케일, 회전, 좌우반전, 뼈단축 등이 차례로 적용됩니다)
   - **결과 분할 저장**: 증강 처리를 맞고 나온 텐서를 다시 DataFrame으로 감싸고 뱃지를 `"aug_A"`로 바꿉니다. 그리고 `csv_data/augmented/aug_A_noise.csv`, `aug_B_rotation.csv` 등 파일명을 명확히 쪼개서 개별 CSV로 저장하고 터미널에 `tqdm` 진행상황을 출력합니다.

3. **`03_run_visualization.py` (랜덤 50 샘플 오토 비전 파이프라인)**
   - **목적**: 기하학적 붕괴가 발생한 증강 실패 사례(`Augmentation Failure`)를 직관적으로 식별 및 표본 측정하기 위한 자동 검수 시스템(QA).
   - **로드**: `argparse` 등 인자 해석기를 통해 대상 파일명(예: `aug_A_noise.csv`)을 지정받아 로드하고, 이와 1:1 대응되는 원본 데이터 텐서를 중복 없이 맵핑합니다.
   - **무작위 추출(Random Sampling)**: 두 파일에서 인덱스가 완벽히 매칭된 동일 연속 프레임(Row) 및 산발적 프레임을 무작위로 **50개(Random 50 Samples)** 추출 표집합니다.
   - **Plotly 렌더링 아키텍처**:
     - `plotly.subplots.make_subplots` (Grid Layout) 모듈을 이용해 50개의 독립 좌표 축 평면 창(예: 5행 10열)을 한 인터랙티브 화면에 생성합니다.
     - 순회 순서마다 "원본 스켈레톤(푸른 선분)"과 "증강 결과 좌표(붉은 선분)"를 다중 오버레이 방식으로 시분할 출력합니다.
   - **출력 및 QA 피드백 환원 루프**: 사용자는 브라우저의 전방위 인터페이스(Orbit)를 이용해 즉각적인 검사 작업을 실시합니다.
   - 만약 표본에서 꼬임 등 기하학적 결함이 무더기로 검출될 경우, 해당 프레임 하나를 폐기하는 것이 의미가 없으므로 해당 증강 세트 뭉치(`.csv`) 전체를 파기합니다. 이 후 오프라인 증강 엔진 스크립트의 수치를 보수(`augmenters.py`)하고 다시 증강 스크립트를 재호출하는 선순환 개발 체계를 반복해 강건성을 확보합니다. 
