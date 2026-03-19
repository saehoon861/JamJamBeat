# 데이터 전처리 파이프라인 (Offline Pipeline)

> ⚠️ **주의:** 본 전처리 도구는 정규화와 다운샘플링만 수행하며, 본격적인 Feature Engineering 단계는 포함하지 않습니다.

---

## 📂 결과 데이터 사용 방법

파이프라인 실행 결과물은 `data/processed_scenarios/` 디렉터리 내에 `.csv` 파일들로 추출됩니다. 팀원 분들은 실험하고자 하는 시나리오에 맞는 파일을 그대로 가져다 모델 학습 및 분석에 사용하시면 됩니다.

### 시나리오 명칭 규칙 (총 12개 데이터셋)

파일 이름은 **[다운샘플링 비율]**과 **[정규화 범주]**의 조합으로 결정됩니다. 각 조건 통제(Ablation)를 위해 총 12가지 버전이 준비되어 있습니다.

| 파일명 (`.csv`) | 다운샘플링 비율 | 위치 정규화 | 스케일 정규화 | 설명 |
|---|:---:|:---:|:---:|---|
| **baseline** | 원본 유지 | X | X | 아무런 처리도 하지 않은 순정 원본 데이터 |
| **ds_4_none** | 4:1 | X | X | 0번 단일 클래스만 4:1 다운샘플링 |
| **ds_1_none** | 1:1 | X | X | 0번 단일 클래스만 1:1 다운샘플링 |
| **pos_only** | 원본 유지 | O | X | 위치 정규화만 적용 (스케일 원본 유지) |
| **ds_4_pos** | 4:1 | O | X | 4:1 다운샘플링 + 위치 정규화 |
| **ds_1_pos** | 1:1 | O | X | 1:1 다운샘플링 + 위치 정규화 |
| **scale_only** | 원본 유지 | X | O | 스케일 정규화만 적용 (위치 좌표 원본 유지) |
| **ds_4_scale** | 4:1 | X | O | 4:1 다운샘플링 + 스케일 정규화 |
| **ds_1_scale** | 1:1 | X | O | 1:1 다운샘플링 + 스케일 정규화 |
| **pos_scale** | 원본 유지 | O | O | 위치와 스케일 정규화 모두 적용 |
| **ds_4_pos_scale** | 4:1 | O | O | 4:1 샘플링 + 전체 정규화 적용 풀-패키지 |
| **ds_1_pos_scale** | 1:1 | O | O | 1:1 샘플링 + 전체 정규화 적용 풀-패키지 |

### W&B Artifact로 시나리오 데이터셋 불러오기

학습 코드에서 아래 패턴을 사용하면 **"이 실험이 어떤 데이터셋을 썼는지"** 가 W&B에 자동으로 연결됩니다.
`use_artifact()` 호출이 없으면 데이터-모델 간 lineage가 기록되지 않으니 반드시 포함해 주세요.

```python
import wandb
import pandas as pd

# ✅ wandb.init()은 학습 전체를 감싸야 lineage가 기록됩니다
with wandb.init(project="JamJamBeat", job_type="train") as run:

    # 1. 사용할 시나리오 이름을 아래에서 골라 교체하세요
    #    (예: "baseline", "ds_1_pos_scale", "pos_only" ...)
    artifact = run.use_artifact("ds_1_pos_scale:latest")
    data_dir = artifact.download()

    # 2. 이후 기존 방식 그대로 데이터 로드
    df = pd.read_csv(f"{data_dir}/ds_1_pos_scale.csv")

    # 3. 학습 코드 ...
```

> 시나리오 이름 목록은 위 테이블의 **파일명** 열을 참고하세요.
---

## ⚙️ 주요 파이프라인 전처리 로직

### 1. 데이터 다운샘플링 (Downsampling)
제스처가 없는 `0`번 클래스의 데이터 불균형을 해소하기 위해, **나머지 액션 클래스(1~6)들의 평균 데이터 수**를 기준으로 0번 클래스의 수를 조절합니다.
* **적용 비율**: `원본(관측상 약 10:1)`, `4:1`, `1:1` 세 가지 강도 분리.
* **안전장치 방어**: 오분류를 막기 위해 제스처 전이 구간(마진) 비디오 프레임은 무조건 배제하며, 핵심 정지 동작인 `0_neutral`,`0_hardneg`  등의 프레임들은 보호 및 분산 샘플링되어 보존됩니다.

### 2. 데이터 정규화 (Normalization)
오브젝트가 카메라로부터 멀어지거나 화면 귀퉁이에 위치하더라도 손 모양 자체가 같으면 동일하게 인식하도록 강건성을 부여합니다.
* **위치 정규화 (Position)**: `wrist(손목, 0번 마디)` 랜드마크를 절대 원점인 `(0, 0, 0)`으로 일괄 평행 이동시킵니다.
* **거리/스케일 정규화 (Scale/Distance)**: `wrist(0번)`부터 손바닥 중앙부인 `middle_mcp(중지 첫 마디, 9번)`까지의 유클리드 거리를 측정하고, 전체 손 뼈대 스케일이 화면 크기와 무관하게 `1.0` 배율 안에 들어오도록 리스케일링합니다.

---

## 🚀 전처리 파이프라인 재실행 방법

원본 데이터가 업데이트되어 12개 시나리오 CSV 파일을 새로 생성해야 할 경우 아래 순서를 따릅니다.

### 1단계: 데이터 준비
* `data/total_data/` 위치에 `total_data.csv`라는 이름으로 통합 데이터 파일을 준비합니다.
* 파일명이 다를 경우 `runners/run_preprocess.py`의 18라인 인근에서 로드할 파일명을 직접 수정할 수 있습니다.

### 2단계: config 설정
* `SCENARIOS` 딕셔너리를 수정하여 원하는 전처리/정규화 조합만 선택하거나 새 시나리오를 추가할 수 있습니다.
* `PROCESSED_DIR`를 수정해 전처리 작업 후 저장될 경로를 지정할 수 있습니다.
* `MARGIN_FRAMES_DROP` 변수를 수정해 다운샘플링 시 제스처 전환점 주변 제거할 프레임 수(frame_idx 기준)를 조절할 수 있습니다.
* `MARGIN_FRAMES_COLLECT` 변수를 수정해 제스처 전환점 주변에서 Class 0으로 수집할 최대 프레임 범위(frame_idx 기준)를 조절할 수 있습니다.

### 3단계: 파이프라인 실행
프로젝트 루트(`JamJamBeat/`) 경로에서 아래 명령어를 실행합니다.
```bash
uv run python src/dataset/offline_pipeline/runners/run_preprocess.py
```




### 추가단계: 검증
비율 통계 확인 - outputdir에 존재하는 모든 데이터들에 대한 비율 통계확인
```bash
uv run python src/dataset/offline_pipeline/tests/test_downsampled.py
```

정규화 시각화 확인 - 지정 데이터에 대해 랜덤으로 1~6클래스의 프레임을 시각화.
```bash
uv run python src/dataset/offline_pipeline/tests/test_normalization.py data/processed_scenarios/ds_1_pos_scale.csv
```



## 추가기능 - 정규화만 하는 파이프라인

### 사용 방법

```bash
uv run python src/dataset/offline_pipeline/runners/run_normalization.py
```


1. src/dataset/offline_pipeline/config.py:7 에 있는 `TOTAL_DIR`(default: data/total_data/)를 원하는 **데이터 경로**로 변경하고 

2. 작업을 하기 위한 **원본 파일 명**을 src/dataset/offline_pipeline/runners/run_normalize_only.py:91 의 `total_csv_path = config.TOTAL_DIR / "total_data_test.csv"`와 일치하도록 변경하거나, run_normalize_only.py:91 라인을 수정해서 실행하면 됩니다.

> 시나리오를 변경하고 싶다면, src/dataset/offline_pipeline/runners/run_normalize_only.py:16 에 있는 **`NORMALIZE_SCENARIOS` 딕셔너리를 수정**해서 원하는 시나리오로 작업하면 됩니다.
