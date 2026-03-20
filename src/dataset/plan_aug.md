# 오프라인 데이터 증강 파이프라인 구현 계획서

본 문서는 전처리 및 정규화가 완료된 오프라인 데이터에 대해 데이터 증강(Data Augmentation)을 수행하기 위한 구현 계획서입니다. 작업자가 어떠한 추가 결정을 내릴 필요 없이 코드로 즉각 변환할 수 있도록 디렉토리 구조, 모듈별 역할, 세부 구현 로직 및 방어 로직을 매우 구체적으로 명시합니다.

## 1. 개요 및 목표
* **목표**: 일괄 전처리된 시나리오 데이터를 사용자 선택에 따라 증강하여 별도 저장 공간에 산출하는 파이프라인 구축.
* **코어 파이프라인**: 
  `data/processed_scenarios/` (원본 시나리오) 
  → **조합 증강 처리 (기본 1배수)** 
  → `data/augmented_scenarios/` (원본 1배수 + 조합 증강된 데이터 1배수 결합)
* **메타데이터 추적**: 모든 증강은 파이프라인을 통과하면서 중첩(Combined) 적용되므로, 데이터별로 어떤 증강이 어떻게 가해졌는지 추적 기록하기 위해 개별 컬럼 3개(`aug_mirror`, `aug_blp`, `aug_noise_sigma`)를 추가하여 불리언 혹은 수치값으로 상태를 표기합니다.

---

## 2. 증강 기법 상세

현재 데이터 상황을 고려하여 기본 **1배수(1x) 증강**을 타겟으로 합니다. (**전 클래스 대상, 클래스 0 포함**)
즉, 원본 데이터와 동일한 크기의 증강 데이터 세트를 1개 만들어 원본과 병합합니다. 결합 증강 파라미터 제어는 `config.py`에서 담당합니다.

### 조합 증강 파이프라인 수행 순서: Mirroring(50% 확률) → BLP(100% 무조건) → Gaussian Noise(100% 무조건)

단일 증강 방식에서 벗어나, 데이터당 여러 증강 확률을 조합하여 하나의 결합 파이프라인으로 구성합니다.

| 순서 | 기법 | 적용 확률 | 이유 |
|---|---|---|---|
| 1 | Mirroring | 50% | 가장 큰 단위의 기하학적 변환. 무작위 50%의 샘플만을 대상으로 L/R 방향 전환 확정 |
| 2 | BLP (Scaling) | 100% | 구조적 변형 단계. 방향이 정해진 손가락 뼈의 비율을 무조건 축소 조정. 노이즈 전에 수행해야 벡터 방향의 구조적 오염 방지 |
| 3 | Gaussian Noise | 100% | 최종 오염 단계. 모든 구조적 변형 완료 후 샘플마다 각기 다른 $\sigma$의 센서 오차를 무조건 덧씌움 |

---

### 2.1 좌우 반전 (Mirroring)
* **목적**: 왼손/오른손 등장 형상 변동 대응.
* **적용 비율**: 50% 확률 부분 적용 (개별 샘플 대상 난수 마스킹).
* **구현 핵심**: Numpy를 통해 `x`로 시작하는 모든 좌표 칼럼의 부호를 반전(`value = value * -1`)시킵니다.
* **방어 로직**: 3D 공간 상에서 x축 부호 반전만으로 좌우반전을 이뤄내며, z축(상대적 깊이)은 그대로 보존합니다. 제스처 클래스 라벨은 일반화 목적으로 변경하지 않습니다.

### 2.2 뼈 길이 섭동 (Bone Length Perturbation, BLP)
* **목적**: 아동의 짧고 통통한 손 비율 특성을 데이터셋에 부여하여 서비스 타겟 강건성 향상.
* **적용 비율**: 100% 파이프라인 대상 프레임 전체 적용.
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
* **적용 비율**: 100% 파이프라인 대상 프레임 전체 적용.
* **적용 패러다임**: 모든 (x, y, z) 좌표 칼럼에 대해 `np.random.normal(0, σ)` 를 더합니다.
* **σ (표준편차) 결정 기준**: 파이프라인을 통과하는 시점에 대상 파일명(`scale` 포함 여부)에 따라 자동 분기합니다.
  * **Scale 전 데이터** (`파일명에 scale 없음`): `σ = U(0.003, 0.005)` 범위에서 개별 샘플마다 각기 다른 $\sigma$를 랜덤 추출하여 적용.
  * **Scale 후 데이터** (`파일명에 scale 포함`): `σ = U(0.02, 0.03)` 범위에서 개별 샘플마다 각기 다른 $\sigma$를 랜덤 추출하여 적용.
* **핵심 방어로직 (마스킹)**: 정규화의 영점 기준이 되는 **손목(0번 랜드마크) 표본 컬럼(`x0, y0, z0`)에는 노이즈를 0으로 강제하는 마스크 행렬을 적용**하여, 부모 노드의 좌표가 흔들리는 현상을 원천 차단합니다.

---

## 3. 구조 및 모듈화 계획

uv 가상환경 하에 실행될 수 있도록 현대적 패키지 구조를 준수하며, 파이프라인과 비즈니스 로직을 명확히 분리합니다.

```text
JamJamBeat/
├── src/
│   └── dataset/
│       └── offline_pipeline/
│           ├── config.py               # (수정) 결합 증강 하이퍼파라미터 및 경로 수정
│           ├── runners/
│           │   ├── run_preprocess.py
│           │   └── run_augment.py      # ✨ (신규) 파이프라인 진입점 (CLI 기반 구동)
│           ├── modules/
│           │   ├── __init__.py
│           │   ├── preprocessor.py
│           │   ├── normalizer.py
│           │   └── augmentor.py        # ✨ (신규) Numpy 기반 결합 증강 벡터 연산 모듈
│           └── tests/
│               ├── test_downsampled.py
│               ├── test_normalization.py
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

# 증강 기능별 적용 확률(Probability) 및 파라미터
AUG_PARAMS = {
    "prob": {
        "mirror": 0.5,          # 50% 확률로 좌우 반전
        "blp": 1.0,             # 1배수(100%) 전량 뼈 길이 축소 적용
        "gaussian_noise": 1.0,  # 1배수(100%) 전량 위치 노이즈 적용
    },
    # 노이즈 σ 범위 (파일명 기반 분기)
    "noise_sigma_range": {
        "non_scale": (0.003, 0.005),  # scale 미포함 파일
        "scale":     (0.020, 0.030),  # scale 포함 파일
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
Numpy 벡터 연산을 활용해 속도를 극대화하고 DataFrame의 `SettingWithCopyWarning`을 우회합니다. 데이터프레임을 직접 반환하지 않고 인플레이스(In-place) 텐서 연산을 진행하며, 메타데이터 컬럼 산출용 벡터를 별도 반환합니다.

* **`apply_mirroring(df, prob)`**:
  1. `sizes = len(df)`
  2. `mirror_mask = np.random.rand(sizes) < prob` 논리 배열로 반전 대상(50%) 인덱스 추출.
  3. `x_cols = [c for c in df.columns if c.startswith('x') and c[1:].isdigit()]`
  4. 인덱스 마스크를 통해 타겟 row들의 x좌표들에 대해서만 `*= -1` 처리 적용.
  5. `mirror_mask` 불리언 배열 반환 (데이터프레임 `aug_mirror` 칼럼 기입용).

* **`apply_blp(df, blp_scales)`**:
  1. `FINGER_CHAINS`를 순회. 각 손가락 마디 연결선(예: `5->6`, `6->7`, `7->8`)에 대해 순차적으로 루프.
  2. `sizes = len(df)` 만큼 각 마디에 할당된 튜플 범위에서 `np.random.uniform(min, max, sizes)` 벡터 연속 생성.
  3. 넘파이 브로드캐스팅 수식: `df[자식_cols] = df[부모_cols] + (df[자식_cols] - df[부모_cols]).multiply(스케일_벡터, axis=0)`
  4. 부모 좌표 업데이트 결과를 기반으로 자식의 원점이 갱신되는 트리 연쇄적 로직 구성. (결과 반환 없음, df 제자리 수정)

* **`apply_gaussian_noise(df, sigma_range: tuple)`**: 
  1. `sizes = len(df)`
  2. `sigmas = np.random.uniform(*sigma_range, sizes)` — 샘플 단위로 제각기 다른 σ 매트릭스 백터 계산.
  3. 좌표 열들만 선택하여 `np.random.normal(0, sigmas[:, None], shape)` 행렬곱 브로드캐스팅으로 텐서 노이즈 생성.
  4. 손목 원점인 `x0`, `y0`, `z0` 칼럼에 삽입될 노이즈 셀은 모두 백터 슬라이싱을 통해 강제로 `0.0` 스왑 처리.
  5. `df[coord_cols] += noise_matrix` 수행.
  6. `sigmas` 넘파이 배열 반환 (`aug_noise_sigma` 칼럼 기입용).

### 4.3. `runners/run_augment.py` (CLI & 파이프라인 엔트리)
* **초기화**: `np.random.seed(AUG_RANDOM_SEED)` 호출로 난수 테이블 고정.
* **콘솔 UI**: `os.listdir`로 `processed_scenarios/` 의 항목들을 탐색하여 콘솔에 1번, 2번 등으로 선택지 제시(`input()`).
* **파일명 기반 σ 분기 판별**:
  * `sigma_range = AUG_PARAMS['noise_sigma_range']['scale'] if 'scale' in filename else AUG_PARAMS['noise_sigma_range']['non_scale']`
* **파이프라인 결합 (순서 엄수: Mirror 50% → BLP 100% → Noise 100% 연속 조합)**:
  1. 선택된 시나리오 CSV 로딩: `df_origin = pd.read_csv(filepath)`
  2. 원본 데이터 더미 메타 라벨링 초기화: 
     `df_origin['aug_mirror'] = False`
     `df_origin['aug_blp'] = False`
     `df_origin['aug_noise_sigma'] = 0.0`
  3. 파이프라인 돌입용 딥카피: `df_aug = df_origin.copy()`
  4. 미러링 파이프라인: `mirror_mask = apply_mirroring(df_aug, AUG_PARAMS['prob']['mirror'])`
  5. BLP 파이프라인: `apply_blp(df_aug, AUG_PARAMS['blp_scales'])`
  6. 노이즈 파이프라인: `sigmas = apply_gaussian_noise(df_aug, sigma_range)`
  7. 증강 데이터 결론 메타 라벨링:
     `df_aug['aug_mirror'] = mirror_mask`
     `df_aug['aug_blp'] = True`
     `df_aug['aug_noise_sigma'] = sigmas.round(6)`
  8. 최종 합병 스택다운: `pd.concat([df_origin, df_aug], ignore_index=True)`로 2배수 병합(원본 1 : 증강 1).
* **I/O 안전장치**: 
  1. 출력 디렉토리가 없다면 `os.makedirs(DIR_AUGMENTED, exist_ok=True)` 로 생성자 확보.
  2. 최종 `df_final.to_csv(DIR_AUGMENTED / f"{scenario_name}_aug.csv", index=False)` 로 덤프 및 데이터 저장 완료.

---

## 5. 방어로직 및 예외사항 체크리스트
* [x] **손목 원점 기준 유지**: 가우시안 노이즈 시 `x0, y0, z0`은 절대 오염시키지 않을 것. 
* [x] **정규화 기준선 보호**: BLP 작동 시 `손목(0) ~ 중지MCP(9)` 뼈대는 절대 건드리지 않고, 손가락 MCP 이후 마디들만 건드릴 것.
* [x] **전 클래스 동일 처리**: 클래스 0 포함 모든 클래스에 동일하게 증강을 적용할 것.
* [x] **파일명 기반 σ 자동 분기**: `scale` 포함 여부로 노이즈 분산 레인지를 자동 결정할 것.
* [x] **무결성 및 메모리 보호**: 증강 모듈 내에서 원본 `df`의 카피본을 통해 인플레이스 업데이트(`copy()` 및 `+=` 등)를 진행하여 참조 변화로 인한 메모리 오염(SettingWithCopyWarning)을 완벽 차단할 것.
* [x] **재현성 보장**: 파이프라인 구동 시 맨 처음 글로벌 랜덤 시드를 할당할 것.