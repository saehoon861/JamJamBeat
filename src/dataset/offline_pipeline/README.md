# 오프라인 데이터 파이프라인 (Offline Pipeline)

이 디렉토리는 모델 학습 전 로컬에서 수행되는 **데이터 전처리 및 증강** 파이프라인입니다.
파이프라인은 크게 **2개의 독립적인 단계(Step)** 로 나뉘어 실행 가능합니다. 
상세한 기법이나 수식 등 기획 관련 내용은 `data/docs/` 내의 문서를 참고하세요.

---

## 🌊 과정 1: 전처리 (Downsampling & Normalization)

클래스 간 불균형 해소(다운샘플링)와 좌표 강건성 확보(정규화)를 수행합니다.

### 실행 방법
```bash
uv run python src/dataset/offline_pipeline/runners/run_preprocess.py
```
* **입력**: `data/total_data/total_data.csv`
* **출력**: `data/processed_scenarios/` 에 통제 조건별 12개의 시나리오 파일 생성

### 시나리오 명칭 규칙 (총 12개)

파일 이름은 **[다운샘플링 비율]**과 **[정규화 범주]**의 조합으로 결정됩니다.

| 파일명 (`.csv`) | 다운샘플링 비율 | 위치 정규화 | 스케일 정규화 |
|---|:---:|:---:|:---:|
| **baseline** | 원본 유지 | X | X |
| **ds_4_none** | 4:1 | X | X |
| **ds_1_none** | 1:1 | X | X |
| **pos_only** | 원본 유지 | O | X |
| **ds_4_pos** | 4:1 | O | X |
| **ds_1_pos** | 1:1 | O | X |
| **scale_only** | 원본 유지 | X | O |
| **ds_4_scale** | 4:1 | X | O |
| **ds_1_scale** | 1:1 | X | O |
| **pos_scale** | 원본 유지 | O | O |
| **ds_4_pos_scale** | 4:1 | O | O |
| **ds_1_pos_scale** | 1:1 | O | O |

> **TIPS**: 다운샘플링 없이 지정한 파일에 대해 정규화만 수행하고 싶다면 `run_normalization.py`를 사용하세요.

> 실행 방법
```bash
uv run python src/dataset/offline_pipeline/runners/run_normalization.py
```


---

## 🌊 과정 2: 데이터 증강 (Augmentation)

1단계에서 생성된 전처리 시나리오 데이터를 기반으로 오프라인 증강을 수행합니다.
한 번에 전부 증강하는 방식이 아니라, 필요한 시나리오만 선택해 증강 후 결합합니다. (Mirroring 50% → BLP 100% → Gaussian Noise 100%)

### 실행 방법
```bash
uv run python src/dataset/offline_pipeline/runners/run_augment.py
```
* **동작 방식**: CLI 목록에서 대상 파일을 고르면 즉시 해당 시나리오 증강
* **입력**: `data/processed_scenarios/` 디렉토리에 있는 시나리오 CSV
* **출력**: `data/augmented_scenarios/` 디렉토리에 원본 대비 **2배수**(원본 1배수 + 증강 1배수 결합)로 데이터가 결합된 `{시나리오명}_aug.csv` 생성

---

## ⚙️ 설정 가이드 (`config.py`)

파이프라인의 핵심 파라미터와 경로는 모두 `config.py`에서 중앙 제어됩니다. 상황에 맞게 값을 수정하여 파이프라인의 동작을 세밀하게 조정할 수 있습니다.

### 과정 1 (샘플링 + 정규화) 제어
* **작업 폴더 위치**: `TOTAL_DIR`(입력), `PROCESSED_DIR`(출력) 경로 지정.
* **제작 시나리오 정의 (`SCENARIOS`)**: 추출할 시나리오 딕셔너리. 각 시나리오별 다운샘플링 비율과 정규화 옵션(위치/스케일) 적용 여부를 켜고 끌 수 있습니다.
* **정규화/샘플링 방어 기준 (`MARGIN_FRAMES`, `THRESHOLD_DIST`)**: 손실 방지를 위한 마진 프레임과 거리 임계값을 조정합니다.

### 과정 2 (데이터 증강) 제어
* **경로 및 난수 (`DIR_AUGMENTED`, `AUG_RANDOM_SEED`)**: 증강 출력 폴더 및 실험 재현성을 위한 글로벌 랜덤 시드 고정.
* **적용 확률 (`AUG_PARAMS['prob']`)**: Mirroring, BLP, Gaussian Noise 각 기법별로 파이프라인에서 작동할 확률 퍼센티지 조정.
* **노이즈 스케일 (`AUG_PARAMS['noise_sigma_range']`)**: 스케일 정규화 적용 유무(파일명 기반)에 따라 부여될 가우시안 노이즈의 $\sigma$ 최소~최대 범위.
* **뼈 길이 축소 (`AUG_PARAMS['blp_scales']`)**: 손가락 마디(근위, 중위, 원위) 계층별로 적용할 길이 축소(Scaling) 계수 레인지 제어.

---

## 🧪 기타 유틸리티 및 검증 도구

### 학습 파이프라인 연동 예시
```python
# W&B Artifact 로 특정 시나리오 데이터셋만 불러오기
with wandb.init(project="JamJamBeat", job_type="train") as run:
    artifact = run.use_artifact("ds_1_pos_scale_aug:latest") # 시나리오명 매칭
    data_dir = artifact.download()
    df = pd.read_csv(f"{data_dir}/ds_1_pos_scale_aug.csv")
```

### 파이프라인 결과 자체 검증 도구
```bash
# 다운샘플링 데이터 비율 통계 확인 (processed_scenarios 대상)
uv run python src/dataset/offline_pipeline/tests/test_downsampled.py

# 정규화 좌표가 틀어지지 않았는지 확인하기 위한 시각화 플레이어
uv run python src/dataset/offline_pipeline/tests/test_normalization.py data/processed_scenarios/ds_1_pos_scale.csv
```
