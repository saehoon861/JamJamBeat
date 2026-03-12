# 데이터 전처리 및 정규화 파이프라인 구현 계획서 (Implementation Plan)

본 문서는 `src/dataset` 디렉터리 내부에서 오프라인 데이터 증강 전 단계인 **데이터 전처리(다운샘플링) 및 정규화(위치, 거리 정규화)** 파이프라인을 구축하기 위한 기획 및 아키텍처 문서입니다. 증강(Augmentation)을 배제하고, 순수 모델 성과를 파악하기 위한 단계별 Ablation Study 시나리오를 구성합니다.

---

## 1. 파이프라인 개요 및 데이터 보존 원칙

이 파이프라인은 설정 파일(`config.py`)에 정의된 시나리오 딕셔너리를 기반으로 작동하며, 각 시나리오의 키(Key) 이름이 곧 최종 추출되는 데이터셋 파일명(`.csv`)이 됩니다.
**⚠️ 중요 원칙**: 이 모든 과정에서 처리 전/후 속성 데이터(Column)는 추가나 병합 없이 **원본 파일과 100% 동일한 열 구조를 유지**해야만 합니다.

* **다운샘플링(Downsampling)** 로직 상세 규칙:
  1. `0_neutral` 및 `0_hardneg` 출처의 영상 프레임 데이터는 어떠한 경우에도 버리지 않고 **100% 보존**합니다.
  2. 그 외 동작 비디오(`1_fast_...`, `2_slow_...` 등) 내부의 Class 0 프레임은 무작위 추출(샘플링) 계산 대상이 됩니다.
  3. **전이 구간 마진(Transition Margin) 방어**: 영상 모델의 오분류를 막기 위해, 제스처가 0에서 1로 변하거나 그 반대인 변화 지점 근방에 위치한 Class 0 프레임은 무조건 샘플링에서 탈락 및 배제됩니다. 이 마진 값은 `config.py`의 `MARGIN_FRAMES` 변수로 중앙 제어됩니다.
  4. **수치 비례 목표량 명확화**: 지정된 `downsample_ratio` (`"origin"`, `"4:1"`, `"1:1"` 세 가지 방식 중 택일) 옵션에 따라 0번 클래스의 데이터 개수가 축소 조정됩니다. 
     - 0번을 제외한 1~6번 나머지 클래스 상호간의 비율 차이나 갯수 차이는 전혀 신경 쓰지 않으며, 이들에 대해서는 **절대 어떠한 다운샘플링이나 조작도 진행하지 않고 원본 그대로 둡니다.** 
     - 목표 수치 예시 (`4:1`): `(0번 클래스) : (나머지 1~6번 클래스의 평균 개수)` 가 대략 `4:1` 이 되도록 0번 클래스만 무작위 추출합니다.
     - 목표 수치 예시 (`1:1`): 일반적인 관측 결과 나머지 클래스들의 평균 프레임 수와, `0_neutral` 및 `0_hardneg` 절대 보존 프레임 수의 비율이 대략 `3:1` 이기 때문에, 1:1 매칭 시에는 `0_` 보존 체인 내부에서도 추가적으로 동일 비율만큼 무작위 샘플링 하락을 허용하는 특별 로직을 구현에 포함시킵니다.

* **정규화(Normalization)** 로직 상세 규칙:
  1. **위치 정규화(Position Normalization)**: `config.py`에 정의된 `origin` 랜드마크(예: `"wrist"`)를 원점 `(0,0,0)`으로 이동(`Translation`) 시킵니다.
  2. **거리/스케일 정규화(Scale Normalization)**: `config.py`에 정의된 `scale` 랜드마크(예: `"middle_mcp"`)와 `origin` 간의 좌표 거리를 기준 스케일 `1.0`으로 축소/확대시킵니다.
     * **무조건적 방어 로직 적용 (`scale_defense` 불필요)**: 거리 계산의 결과값이 비정상적으로 0에 한없이 가까워질 경우 텐서 나눗셈에서 발산(NaN) 에러가 발생할 수 있으므로, 어떤 시나리오에서든 수식 연산 시 무조건 스케일 분모에 `max(distance, 1e-6)` 등 하한선 락(Lock)을 씌웁니다.

---

## 2. 시각적 검토 도구 및 성능 점검 파이프라인

검수 도구는 통계 숫자를 점검하는 스크립트와, 3D 가시화를 하는 스크립트로 구성됩니다.

1. **다운샘플링 통계 검증 툴 (`tests/test_downsampled.py`)** 
   * 다운샘플링 처리된 CSV 파일을 CLI 상에서 로드하고 터미널 콘솔 로그(`print` / `logger`)로 생존 통계(Statistics)를 출력합니다.
   * `0_neutral`, `0_hardneg` 프레임 생존율 출력 (무조건 100% 보존 확인).
   * 각 비디오 클래스별로 잔여 0번 동작 프레임 생존 수량 명세 출력 및 비율 검산 증명. 다른 클래스들 간의 데이터 불균형 내역은 단지 로그로 명시만 해줍니다.
2. **3D 랜드마크 렌더링 검증 툴 (`tests/test_normalization.py`)**
   * 특정 프레임 혹은 랜덤 서브샘플링된 프레임들의 스켈레톤을 팝업 브라우저(Plotly 등) 상에 오버레이 렌더링.
   * "위치 정규화가 된 버전"은 기준점이 `(0,0,0)` 지점에 시각적으로 맞물리는지 마우스 드래그를 통해 3D 환경에서 확인 가능.
   * "스케일 정규화가 된 버전"은 손 구조 스케일이 모두 동일한 체적 내로 일정하게 들어오는지 원근으로 직접 관측 가능.

---

## 3. 디렉토리 구조 및 파이썬 환경 구성 (`config.py` 연동)

파이썬 패키지 버전을 엄격히 통제하되, 타 프로세스들과의 연계를 고려해 직관적이고 분리된 폴더를 구성합니다.

```text
JamJamBeat/
├── src/
│   └── dataset/
│       ├── README.md               # 전처리 파이프라인 사용법
│       ├── plan.md                 # 오프라인 데이터 증강 계획
│       ├── research.md             # 리서치 노트
│       ├── plan_ds_norm.md         # 데이터 전처리 및 정규화 구현 계획서 (현재 문서)
│       │
│       └── offline_pipeline/       # ✨ 오프라인 파이프라인 작업 디렉터리
│           ├── config.py           # 경로 관리, MediaPipe 매핑, 파라미터 및 SCENARIOS 보관
│           │
│           ├── runners/            
│           │   └── run_preprocess.py  # config.py의 시나리오를 바탕으로 CSV 일괄 추출하는 메인 실행기
│           │
│           ├── modules/            
│           │   ├── __init__.py
│           │   ├── preprocessor.py    # 다운샘플링 및 DataFrame 조작 로직
│           │   └── normalizer.py      # 넘파이(Numpy) 텐서 정규화 수학 모델
│           │
│           └── tests/              
│               ├── test_downsampled.py    # 터미널 통계 뷰어 스크립트
│               └── test_normalization.py  # Plotly 3D 육안 검증 스크립트
│
└── data/                   
    ├── raw_data/                 # 동영상 원본 (.mp4)
    ├── total_data/               # 랜드마크+라벨링 추출본 통합 데이터 (.csv)
    └── processed_scenarios/      # ✨ SCENARIOS 키 명칭으로 새롭게 도출되는 데이터셋 결과물 저장소
        ├── baseline.csv
        ├── downsample_3.csv 
        └── ...
```

---

## 4. 라이브러리 및 의존성 구성 (`pyproject.toml`)

루트 디렉토리의 메인 환경(`JamJamBeat/pyproject.toml`)에는 파이프라인 구동에 가장 필수적인 코어 모듈들(`numpy>=1.26.0`, `pandas>=3.0.1`, `torch>=2.10.0`)이 호환성 문제 없이 이미 설치되어 있습니다. 
따라서 별도의 가상환경을 복잡하게 구성할 필요 없이, 시각적 검증(`test_normalization.py`)을 위한 3D 렌더링 브라우저 출력 패키지인 **`plotly` (>=5.24.0)** 하나만 기존 가상환경에 추가(`uv add plotly`)하여 가볍게 활용하도록 합니다.

---

## 5. 단계별 모듈 세부 구현 명세

유지보수와 분산 처리에 용이하도록 코어 함수(Model)와 제어(Controller)를 명확히 분리하여 개발합니다.

### 5.1. `config.py` 코어 명세 및 시나리오 맵핑 로직

모든 경로, 랜드마크 인덱스 번역기, 시나리오 조합법, 오염 데이터 방지용 마진 frame 갯수는 `offline_pipeline/config.py` 한 곳에서 중앙 통제됩니다.

```python
from pathlib import Path

# --- 경로 관리 ---
ROOT = Path(__file__).resolve().parents[4]  # 프로젝트 루트 디렉터리 자동 탐색

DATA_DIR = ROOT / "data"
TOTAL_DIR = DATA_DIR / "total_data"
PROCESSED_DIR = DATA_DIR / "processed_scenarios"

# --- 파라미터 ---
MARGIN_FRAMES = 5  # 전이 구간 근방의 위험군 프레임 배제 마진

# --- MediaPipe 손 랜드마크 구조 매핑 ---
LANDMARK_IDX = {
    "wrist": 0,
    "thumb_cmc": 1, "thumb_mcp": 2, "thumb_ip": 3, "thumb_tip": 4,
    "index_mcp": 5, "index_pip": 6, "index_dip": 7, "index_tip": 8,
    "middle_mcp": 9, "middle_pip": 10, "middle_dip": 11, "middle_tip": 12,
    "ring_mcp": 13, "ring_pip": 14, "ring_dip": 15, "ring_tip": 16,
    "pinky_mcp": 17, "pinky_pip": 18, "pinky_dip": 19, "pinky_tip": 20,
}

# --- 시나리오 설정 (총 12개 조합: 비율 3 x 위치 정규화 2 x 스케일 정규화 2) ---
SCENARIOS = {
    # 1. 아무것도 안 함 (X / X)
    "baseline": { "downsample_ratio": "origin", "origin": None, "scale": None, "augment": False },
    "ds_4_none": { "downsample_ratio": "4:1", "origin": None, "scale": None, "augment": False },
    "ds_1_none": { "downsample_ratio": "1:1", "origin": None, "scale": None, "augment": False },
    
    # 2. 위치만 (O / X)
    "pos_only": { "downsample_ratio": "origin", "origin": "wrist", "scale": None, "augment": False },
    "ds_4_pos": { "downsample_ratio": "4:1", "origin": "wrist", "scale": None, "augment": False },
    "ds_1_pos": { "downsample_ratio": "1:1", "origin": "wrist", "scale": None, "augment": False },
    
    # 3. 스케일만 (X / O)
    "scale_only": { "downsample_ratio": "origin", "origin": None, "scale": ["wrist", "middle_mcp"], "augment": False },
    "ds_4_scale": { "downsample_ratio": "4:1", "origin": None, "scale": ["wrist", "middle_mcp"], "augment": False },
    "ds_1_scale": { "downsample_ratio": "1:1", "origin": None, "scale": ["wrist", "middle_mcp"], "augment": False },
    
    # 4. 위치 + 스케일 (O / O)
    "pos_scale": { "downsample_ratio": "origin", "origin": "wrist", "scale": ["wrist", "middle_mcp"], "augment": False },
    "ds_4_pos_scale": { "downsample_ratio": "4:1", "origin": "wrist", "scale": ["wrist", "middle_mcp"], "augment": False },
    "ds_1_pos_scale": { "downsample_ratio": "1:1", "origin": "wrist", "scale": ["wrist", "middle_mcp"], "augment": False },
}
```

### 5.2. `modules/preprocessor.py` (다운샘플링 로직 - Pandas Model)
* **목표**: DataFrame을 Input 받아 조건에 맞춰 추출된 DataFrame을 Return 합니다.
* **주요 함수 `apply_downsampling(df: pd.DataFrame, target_ratio: str, margin_frames: int = 5) -> pd.DataFrame`**:
  * `target_ratio` 아규먼트는 `"origin"`, `"4:1"`, `"1:1"` 세 상태를 받습니다. `"origin"`일 때는 그대로 `return df`하여 조기 종료합니다.
  * **Phase 1 (불구속 데이터 분리)**: 전체 DF 중 `gesture != 0` 인 행(A 그룹)과, `gesture == 0` 중 `source_file` 속성이 `0_`으로 시작하는 것(B 그룹, `0_neutral`, `0_hardneg`)을 별도 보존 영역으로 분리 추출합니다.
  * **Phase 2 (위험군 전이 마진 배제)**: 남은 `gesture == 0` 데이터 중(C 그룹), 각 데이터별 `source_file`을 기준으로 순회합니다. 제스처 값이 0→1, 1→0 으로 전환되는 프레임 좌표를 잡아내고, 기준점 앞뒤의 ±`margin_frames` 구간 영역을 모두 `.drop()`으로 삭제합니다.
  * **Phase 3 (비율 기반 스케일 다운)**: 타겟 보존 그룹(A 그룹) 내의 `1~6` 제스처 각각 데이터 개수들의 산술 평균값(`mean`)을 산출합니다. 그 후 수신된 스펙 `"4:1"` 일 시 해당 평균값의 x4배, `"1:1"` 일 시 해당 평균값의 x1배 처리해 목표 할당량 수치를 계산합니다. 살아남은 C 그룹 데이터 세트 안에서 잔여 제한 할당량 수치만큼 무작위 난수 샘플링(`pd.DataFrame.sample`)을 실행합니다.
  * **Phase 4 (결합 및 정렬)**: 분산했던 A, 100% 데이터 보호된 B, 전처리 샘플링이 끝난 C를 모두 합치고(`.concat`), 기존 인덱스 행 순서 혹은 `source_file`+`frame_idx`에 맞춰 다시(`.sort_values()`) 원형 구조 형태로 완벽히 되돌려줍니다.

### 5.3. `modules/normalizer.py` (수학적 3D 정규화 로직 - Numpy Model)
* **목표**: 랜드마크 좌표계 (x, y, z) 원본 열을 가공하여 평행이동 및 비율 수식 연산을 넘파이 텐서 단위로 이룹니다.
* **주요 함수 `apply_position_normalization(landmarks_3d: np.ndarray, origin_idx: int) -> np.ndarray`**:
  * 입력 차원: `(데이터_로우_단위, 21, 3)`
  * 브로드캐스팅 수식 연산: `landmarks_3d - landmarks_3d[:, origin_idx:origin_idx+1, :]`
  * 모든 프레임에서 `config`에서 넘어오는 `origin_idx`(예: 0번 손목) 좌표를 전체 좌표 텐서 값들에 상대 위치로 마이너스 처리합니다.
* **주요 함수 `apply_distance_normalization(landmarks_3d: np.ndarray, origin_idx: int, scale_idx: int) -> np.ndarray`**:
  * **수식 연산 (유클리디안 거리 측정 및 무조건 방어 결합)**:
    ```python
    dist = np.linalg.norm(landmarks_3d[:, scale_idx, :] - landmarks_3d[:, origin_idx, :], axis=1, keepdims=True)
    # 텐서 붕괴 발산(NaN)을 막기 위한 무조건적 방어 설계 적용 완료
    safe_dist = np.maximum(dist, 1e-6)
    ```
  * **수식 연산 2 (스케일 조정)**: 전체 좌표를 분모로 나누고(`landmarks_3d / np.expand_dims(safe_dist, axis=1)`) 0위 스케일 클리핑 리턴합니다. (위 코드와 직관적으로 합병 구현)

### 5.4. `runners/run_preprocess.py` (시나리오 마스터 컨트롤러)
* **목표**: 전체 파이프라인 진입 메인 파일(`__main__`)로서 작동하며, `config.py`의 `SCENARIOS` 딕셔너리를 활용해 다중 옵션 콤비네이션들을 배치 처리합니다.
* **주요 로직 흐름**:
  1. `pd.read_csv()`를 활용해 `total_data` 폴더의 전체 통파일 원본 메모리로 깊은 복사(`copy`)로드.
  2. `for scenario_name, params in SCENARIOS.items():` 형식으로 루프 회전.
  3. 파이프라인 초반부 맵핑: 루프 진입 즉시 `LANDMARK_IDX` 사전을 참조하여 `origin`(문자열)과 향후 다중 참조가 가능하도록 리스트로 관리되는 `scale_points`(리스트) 파라미터 값들을 사전에 정수형 인덱스 리스트 등으로 일괄 파싱 변환해 줍니다.
  4. 반복문 내부: 각 루프마다 `params['downsample_ratio']`에 따라 전처리 단계를 넘기며, Numpy 객체 파싱 후 변환된 인덱스 번호들을 바탕으로 위치/거리 정규화를 순차적 연쇄 호출 적용.
  5. 가공된 넘파이 텐서 결과물을 다시 원형 DataFrame인 패키지의 `x0 ~ z20` 숫자 컬럼 속성에 제자리 덮어씌움.
  6. 1개 시나리오 완료 시마다 `processed_scenarios/` 물리 디스크 폴더 내에 해당 딕셔너리 key 값인 `scenario_name` 형식(`downsample_4.csv` 등)으로 파일 일괄 추출 저장. 무한 동작.

### 5.5. `tests/test_downsampled.py` & `tests/test_normalization.py` (검수 컨트롤러)
* **`test_downsampled.py`**:
  * 산출된 처리 CSV 경로를 인자로 받습니다.
  * DataFrame 로드 후 `df.groupby('source_file')['gesture'].count()` 등의 GroupBy 통계 연산을 활용하여 기존 원본 파일 통계 대비 변동량을 연산합니다.
  * 특히 `0_neutral`/`0_hardneg` 보호 그룹이 100% 생존했는지 점검하며, 0번 클래스가 최종 목표 스펙("1:1", "4:1" 등)에 맞추어 얼마나 안전하게 배분되었는지 확인하는 그룹 출력 로직을 전개합니다.
* **`test_normalization.py`**:
  * `plotly.graph_objects` 라이브러리의 `Scatter3d` 인스턴스를 통해 각 뼈대점(21개 마디) 선을 정의하고 UI 캔버스 브라우저에 그립니다.
  * 사용자의 마우스 Orbit 이벤트 등에 실시간 반응하며 3D 시각적 진단을 돕습니다.

---

## 6. 구현 기록 (Implementation Log)

- **작업 내용 (2026.03.12)**: 추가된 주석 기반 요구사항 (1:1 비율 특별 제어 규칙 및 12개 시나리오 세트 등)을 반영하여, `config.py`, `preprocessor.py`, `normalizer.py`, `run_preprocess.py` 및 통계 점검과 시각 리포트 테스트 모듈을 재작성 및 생성 완료했습니다.
- **성능 검증 완료 및 결과**:
  1. `baseline.csv` 등 4종 (origin 비율): 원본 14,887행 프레임 개수 보호 그룹 100% 생존 유지 완료. (15.79:1 비율 자연관측)
  2. `ds_4_none.csv` 등 4종 (4:1 비율): 0_보호 프레임들(총 2,108개) 100% 생존 확인 완료 및, C그룹에서 목표 수량대로 단축이 이루어져 최종 0번 분류 프레임이 총 2,732개(`4.00 : 1` 비율)로 정확하게 처리됨을 확인.
  3. **`ds_1_none.csv` 등 4종 (1:1 비율 특별 처리)**: 보호 그룹의 기존 비중이 지나치게 커 1:1 목표를 초과하던 기존 이슈가 해결되었음을 확인. 0_보호 데이터와 일반 C데이터 그룹 양쪽에 동률의 생존 컷오프(Fraction) 연산을 가하여, 보호그룹 생존수 159개 / 전체 0번 그룹 생존수 683개로 **정확하게 1~6 그룹 전체 평균(683.2개) 대비 `1.00 : 1` 도달을 달성**했습니다.
- **계획과 달라진 실제 변동/보완 사항**:
  - `config.py`의 `ROOT` 경로 변수 지정 시, 모듈의 트리 깊이를 반영하여 설계서 논의 단계의 `parents[4]` 구문에서 실제 폴더 뎁스에 맞춘 `parents[3]`으로 수정하여 반영했습니다.
  - 좌표 텐서 정규화 수행 시 `config.py`에서 의도적으로 스케일에 빈 리스트(`None`)가 할당된 조합형 옵션 시그널(`X`)들을 정상적으로 Pass하기 위한 분기 검사가 `runner` 쪽에 추가되었습니다.
