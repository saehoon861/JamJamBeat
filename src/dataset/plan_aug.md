# 오프라인 데이터 증강 파이프라인 구현 계획서

본 문서는 전처리 및 정규화가 완료된 오프라인 데이터에 대해 데이터 증강(Data Augmentation)을 수행하기 위한 구현 계획서입니다. 작업자가 어떠한 추가 결정을 내릴 필요 없이 코드로 즉각 변환할 수 있도록 디렉토리 구조, 모듈별 역할, 세부 구현 로직 및 방어 로직을 매우 구체적으로 명시합니다.

## 1. 개요 및 목표
* **목표**: 일괄 전처리된 시나리오 데이터를 사용자 선택에 따라 증강하여 별도 저장 공간에 산출하는 파이프라인 구축.
* **코어 파이프라인**: 
  `data/processed_scenarios/` (원본 시나리오) 
  → **증강 처리 (기본 1배수)** 
  → `data/augmented_scenarios/` (증강된 시나리오)
* **메타데이터 추적**: 모든 산출 데이터프레임에 `aug_type` 컬럼을 삽입하여 추후 분석 시 데이터의 출처(`original`, `gaussian_noise`, `blp`, `mirror`)를 식별 가능토록 구현합니다.

---

## 2. 증강 기법 상세

현재 데이터 상황을 고려하여 기본 **1배수(1x) 증강**을 타겟으로 합니다. (**전 클래스 대상, 클래스 0 포함**)
이 파라미터 제어는 `config.py`에서 담당합니다.

### 증강 파이프라인 수행 순서: Mirroring → BLP → Gaussian Noise

각 기법은 독립적으로 원본 데이터에 적용되어 별도 증강 DF를 만들지만, **조합 증강을 추가 구현할 때를 대비해 수행 순서를 다음과 같이 정의한다.**

| 단계 | 기법 | 이유 |
|---|---|---|
| 1 | Mirroring | 가장 큰 단위의 기하학적 변환. 데이터의 방향(L/R)을 먼저 확정 |
| 2 | BLP (Scaling) | 구조적 변형 단계. 방향이 정해진 상태에서 뼈 비율 조정. 노이즈 전에 수행해야 벡터 방향 오염 방지 |
| 3 | Gaussian Noise | 최종 오염 단계. 모든 구조적 변형 완료 후 센서 오차를 덧씌움. 선행 스케일링 후에 노이즈를 더해야 분산 크기가 의도치 않게 줄어드는 사태를 방지 |

---

### 2.1 좌우 반전 (Mirroring)
* **목적**: 왼손/오른손 등장 형상 변동 대응.
* **구현 핵심**: Numpy를 통해 `x_`로 시작하는 모든 좌표 칼럼의 부호를 반전(`value = value * -1`)시킵니다.
* **방어 로직**: 3D 공간 상에서 x축 부호 반전만으로 좌우반전을 이뤄내며, z축(상대적 깊이)은 그대로 보존합니다. 제스처 클래스 라벨은 일반화 목적으로 변경하지 않습니다.

### 2.2 뼈 길이 섭동 (Bone Length Perturbation, BLP)
* **목적**: 아동의 짧고 통통한 손 비율 특성을 데이터셋에 부여하여 서비스 타겟 강건성 향상.
* **증강 제약 사항**: 길이에 대한 **축소만 허용**하며 역방향 확대는 배제합니다.
  * Proximal (MCP → PIP) 스케일 범위: `0.90 ~ 0.98`
  * Middle (PIP → DIP) 스케일 범위: `0.85 ~ 0.93`
  * Distal (DIP → Tip) 스케일 범위: `0.80 ~ 0.88`
* **구현 핵심 (Kinematic Chain Hierarchical Scaling)**:
  * 부모 마디 위치를 고정한 뒤, 자식 마디를 향하는 방향 벡터의 길이를 스케일링 계수만큼 감소시킵니다.
  * *수식*: $자식_{new} = 부모_{old} + (자식_{old} - 부모_{old}) \times 스케일$
  * 축소가 적용된 자식 마디를 다시 부모로 취급하여 그 다음 마디의 위치를 연쇄적으로 재계산합니다.
* **핵심 방어로직**: 정규화 기준축인 `손목(0) ~ 중지MCP(9)` 뼈대는 절대 건드리지 않으며, 모든 손가락의 스케일링 원점은 각각의 `MCP` (또는 엄지의 경우 1번 조인트)로 간주합니다.

### 2.3 가우시안 노이즈 (Gaussian Noise)
* **목적**: 센서 측정 오차의 강건성 확보.
* **적용 패러다임**: 모든 (x, y, z) 좌표 칼럼에 대해 `np.random.normal(0, σ)` 를 더합니다.
* **σ (표준편차) 결정 기준**: 파일명(`_scale` 포함 여부)에 따라 자동 분기합니다.
  * **Scale 전 데이터** (`파일명에 _scale 없음`): `σ = U(0.003, 0.005)` 범위에서 샘플마다 랜덤 적용
    * 근거: MediaPipe 정규화 좌표계 기준 자연 지터링 오차(~0.001~0.003) 수준의 약간 상위
  * **Scale 후 데이터** (`파일명에 _scale 포함`): `σ = U(0.02, 0.03)` 범위에서 샘플마다 랜덤 적용
    * 근거: Scale 전후 데이터셋 평균 크기 비율이 약 4.8배 → `0.005 × 4.8 ≈ 0.024`
  * 구현: `run_augment.py`에서 파일명을 `'_scale' in filename` 조건으로 판별하여 `apply_gaussian_noise(df, sigma_range)` 호출 시 sigma_range를 넘겨줌.
* **핵심 방어로직 (마스킹)**: 정규화의 영점 기준이 되는 **손목(0번 랜드마크) 표본 컬럼(`x_0, y_0, z_0`)에는 노이즈를 0으로 강제하는 마스크 행렬을 적용**하여, 부모 노드의 좌표가 흔들리는 현상을 원천 차단합니다.

---

## 3. 구조 및 모듈화 계획

uv 가상환경 하에 실행될 수 있도록 현대적 패키지 구조를 준수하며, 파이프라인과 비즈니스 로직을 명확히 분리합니다.

```text
JamJamBeat/
├── src/
│   └── dataset/
│       └── offline_pipeline/
│           ├── config.py               # (수정) 하이퍼파라미터 및 경로 상수 추가 
│           ├── runners/
│           │   ├── run_preprocess.py
│           │   └── run_augment.py      # ✨ (신규) 파이프라인 진입점 (CLI 기반 구동)
│           ├── modules/
│           │   ├── __init__.py
│           │   ├── preprocessor.py
│           │   ├── normalizer.py
│           │   └── augmentor.py        # ✨ (신규) Numpy 기반 증강 벡터 연산
│           └── tests/
│               ├── test_downsampled.py
│               ├── test_normalization.py  # Plotly 3D 육안 검증 스크립트
│               └── test_augment.py     # ✨ (신규) 방어 로직 단언(Assert) 테스트
│
└── data/
    ├── processed_scenarios/            # (입력) 전처리본 시나리오 데이터
    └── augmented_scenarios/            # (출력) 증강된 시나리오 최종 출력
```

---

## 4. 코드 수준 세부 구현안
코드를 구현할 때 주석을 한국어로 팀프로젝트라는 기반하에 다른 팀원이 기능과 코드를 바로 파악할 수 있는 수준의 주석을 작성할 것.

### 4.1. `config.py` 업데이트 사항
```python
import numpy as np

# 증강 입출력 디렉토리
DIR_AUGMENTED = PROJECT_ROOT / "data" / "augmented_scenarios"

# 재현성을 위한 글로벌 난수 시드 (항상 고정)
AUG_RANDOM_SEED = 42

# 증강 기능별 적용 배수 및 BLP 파라미터
AUG_PARAMS = {
    "multipliers": {
        "mirror": 1.0,          # 순서 반영: Mirroring이 1번
        "blp": 1.0,             # BLP가 2번
        "gaussian_noise": 1.0,  # Gaussian Noise가 마지막
    },
    # 노이즈 σ 범위 (파일명 기반 분기)
    "noise_sigma_range": {
        "non_scale": (0.003, 0.005),  # _scale 미포함 파일
        "scale":     (0.020, 0.030),  # _scale 포함 파일
    },
    "blp_scales": {
        "proximal": (0.90, 0.98),
        "middle":   (0.85, 0.93),
        "distal":   (0.80, 0.88),
    }
}

# 계층 구조 (수직적 Kinematic 연산을 위한 트리)
FINGER_CHAINS = [
    [1, 2, 3, 4],     # 엄지
    [5, 6, 7, 8],     # 검지
    [9, 10, 11, 12],  # 중지
    [13, 14, 15, 16], # 약지
    [17, 18, 19, 20]  # 소지
]
```

### 4.2. `modules/augmentor.py`
Numpy 벡터 연산을 활용해 속도를 극대화하고 DataFrame의 `SettingWithCopyWarning`을 우회하기 위해 `df.copy()`를 적극 사용합니다. **모든 함수는 클래스 0 포함 전 클래스에 동일하게 적용합니다.**

* **`apply_mirroring(df)`**:
  1. `df_aug = df.copy()`
  2. `x_cols = [c for c in df.columns if c.startswith('x_')]`
  3. `df_aug[x_cols] = df_aug[x_cols] * -1`
  4. `df_aug['aug_type'] = 'mirror'` 후 반환.

* **`apply_blp(df, blp_scales)`**:
  1. `df_aug = df.copy()`
  2. `FINGER_CHAINS`를 순회. 각 손가락 마디 연결선(예: `5->6`, `6->7`, `7->8`)에 대해 순차적으로 루프.
  3. `sizes = len(df_aug)` 만큼 각 마디에 할당된 튜플 범위에서 `np.random.uniform(min, max, size)` 백터 추출.
  4. 넘파이 브로드캐스팅 수식: `df_aug[자식_cols] = df_aug[부모_cols] + (df_aug[자식_cols] - df_aug[부모_cols]).multiply(스케일_벡터, axis=0)`
  5. 부모 좌표가 업데이트되면 그 좌표가 다음 자식의 원점이 되는 연쇄적 로직 구성.
  6. `df_aug['aug_type'] = 'blp'` 반환.

* **`apply_gaussian_noise(df, sigma_range: tuple)`**: 
  1. `df_aug = df.copy()`
  2. `sigma = np.random.uniform(*sigma_range)` — 해당 파일에 적용할 단일 σ 값을 결정.
  3. 좌표 열들만 선택하여 `np.random.normal(0, sigma, shape)` 행렬 생성.
  4. 손목인 `x_0`, `y_0`, `z_0` 칼럼에 삽입될 난수 셀은 모두 강제로 `0.0`으로 덮어씌움.
  5. `df_aug[coord_cols] += noise_matrix`
  6. `df_aug['aug_type'] = 'gaussian_noise'` 후 반환.

### 4.3. `runners/run_augment.py` (CLI & 파이프라인 엔트리)
* **초기화**: `np.random.seed(AUG_RANDOM_SEED)` 호출로 난수 테이블 고정.
* **콘솔 UI**: `os.listdir`로 `processed_scenarios/` 의 항목들을 탐색하여 콘솔에 1번, 2번 등으로 선택지 제시(`input()`).
* **파일명 기반 σ 분기**:
  * `sigma_range = AUG_PARAMS['noise_sigma_range']['scale'] if '_scale' in filename else AUG_PARAMS['noise_sigma_range']['non_scale']`
* **파이프라인 결합 (순서 엄수: Mirror → BLP → Noise)**:
  1. 선택된 CSV를 로딩: `df_origin = pd.read_csv(filepath)`
  2. 원본 데이터 라벨링: `df_origin['aug_type'] = 'original'`
  3. `df_mirror = apply_mirroring(df_origin)`
  4. `df_blp = apply_blp(df_origin, AUG_PARAMS['blp_scales'])`
  5. `df_noise = apply_gaussian_noise(df_origin, sigma_range)`
  6. `pd.concat([df_origin, df_mirror, df_blp, df_noise], ignore_index=True)`로 하나의 DataFrame으로 합병.
* **I/O 안전장치**: 
  1. 출력 디렉토리가 없다면 `os.makedirs(DIR_AUGMENTED, exist_ok=True)` 로 안전망 확보.
  2. 최종 `df_final.to_csv(DIR_AUGMENTED / f"{scenario_name}_aug.csv", index=False)` 로 덤프 및 데이터 저장 완료.

---

## 5. 방어로직 및 예외사항 체크리스트
* [x] **손목 원점 기준 유지**: 가우시안 노이즈 시 `x_0, y_0, z_0`은 절대 오염시키지 않을 것. 
* [x] **정규화 기준선 보호**: BLP 작동 시 `손목(0) ~ 중지MCP(9)` 뼈대는 절대 건드리지 않고, 손가락 MCP 이후 마디들만 건드릴 것.
* [x] **전 클래스 동일 처리**: 클래스 0 포함 모든 클래스에 동일하게 증강을 적용할 것.
* [x] **파일명 기반 σ 자동 분기**: `_scale` 포함 여부로 노이즈 분산 레인지를 자동 결정할 것.
* [x] **무결성 및 메모리 보호**: 증강 모듈 내에서 `df`를 변경 시 `.copy()` 메서드를 활용하여 원본 데이터프레임 내부 참조 변화(SettingWithCopyWarning)를 막을 것.
* [x] **재현성 보장**: 파이프라인 구동 시 맨 처음 글로벌 랜덤 시드를 할당할 것.