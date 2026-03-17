# README.md - JamJamBeat data_fusion 데이터셋 구조 안내

`model/data_fusion`은 JamJamBeat 학습/평가에 쓰는 landmark CSV를 역할별로 정리한 폴더다.

현재 기준으로 이 폴더는 아래 4개 묶음으로 이해하면 된다.

- `기존데이터셋/`
  - 학습 원천 CSV
  - `source_file` 56개를 포함한 통합 landmark 세트
- `테스트데이터셋/`
  - 공식 비교용 외부 test CSV
  - `total_data_test_*` 4종이 핵심
- `학습데이터셋/`
  - 실제 파이프라인 입력용 explicit split CSV
  - `train / val / inference / test` 역할이 이미 나뉜 상태
- `추론용데이터셋/`
  - 이미지/추론 실험용 별도 입력 폴더

## 1. 공통 컬럼 구조

대부분의 CSV는 아래 컬럼을 공통으로 가진다.

- `source_file`
- `frame_idx`
- `timestamp`
- `gesture`
- `x0,y0,z0 ... x20,y20,z20`

의미:

- `source_file`: 원본 비디오 단위 식별자
- `gesture`: 제스처 클래스 id
- `x*, y*, z*`: 21개 손 랜드마크 좌표

## 2. 폴더별 역할

### 2.1 `기존데이터셋/`

학습 원천으로 쓰는 통합 CSV가 들어 있다.

대표 파일:

- `baseline.csv`
- `pos_only.csv`
- `scale_only.csv`
- `pos_scale.csv`
- `baseline_ds_1_none.csv`
- `pos_only_ds_1_pos.csv`
- `scale_only_ds_4_scale.csv`
- `pos_scale_ds_4_pos_scale.csv`

중요한 점:

- 현재 표준 학습 흐름에서는 이 CSV를 직접 `run_pipeline.py`에 넣지 않는다
- 대신 [build_training_datasets.py](/home/user/projects/JamJamBeat/model/data_fusion/build_training_datasets.py)가 이 원천 CSV를 읽어
  `학습데이터셋/`의 역할형 CSV를 만든다

### 2.2 `테스트데이터셋/`

외부 평가용 test CSV와 그 원본 조각들이 들어 있다.

핵심 파일:

- `total_data_test_baseline.csv`
- `total_data_test_pos_only.csv`
- `total_data_test_scale_only.csv`
- `total_data_test_pos_scale.csv`

이 4개는 same-normalization 기준 공식 test 세트다.

보조 파일:

- `0_test_right_man1.csv` 같은 개별 제스처 test 파일
- `man1_right_fortest.csv` 같은 사용자별 통합 test 파일

현재 표준 파이프라인에서는 개별 조각보다 `total_data_test_*` 4종을 공식 test로 사용한다.

### 2.3 `학습데이터셋/`

현재 실제 파이프라인 입력은 이 폴더 기준이다.

파일명 규칙:

- `{dataset_key}_train.csv`
- `{dataset_key}_val.csv`
- `{dataset_key}_inference.csv`
- `{dataset_key}_test.csv`

여기서:

- `train`: 학습용
- `val`: early stopping / 모델 선택용
- `inference`: hold-out 추론 검증용
- `test`: 공식 외부 평가용

즉 이제는 예전 `trainval.csv` 구조가 아니라, 역할이 이미 분리된 CSV만 사용한다.

## 3. 현재 표준 dataset key 체계

현재는 학습 원천 12세트를 사용한다.

### 3.1 baseline family

- `baseline`
- `baseline_ds_1_none`
- `baseline_ds_4_none`

모두 같은 test family를 쓴다.

- test source: `테스트데이터셋/total_data_test_baseline.csv`

### 3.2 pos_only family

- `pos_only`
- `pos_only_ds_1_pos`
- `pos_only_ds_4_pos`

공통 test:

- `테스트데이터셋/total_data_test_pos_only.csv`

### 3.3 scale_only family

- `scale_only`
- `scale_only_ds_1_scale`
- `scale_only_ds_4_scale`

공통 test:

- `테스트데이터셋/total_data_test_scale_only.csv`

### 3.4 pos_scale family

- `pos_scale`
- `pos_scale_ds_1_pos_scale`
- `pos_scale_ds_4_pos_scale`

공통 test:

- `테스트데이터셋/total_data_test_pos_scale.csv`

## 4. `학습데이터셋/`는 어떻게 생성되나

[build_training_datasets.py](/home/user/projects/JamJamBeat/model/data_fusion/build_training_datasets.py)가 아래 규칙으로 explicit split CSV를 만든다.

- train source CSV는 위 12개 dataset key mapping을 사용
- `source_file` 단위 video-level split
- 고정 seed: `42`
- source 개수:
  - `train = 40`
  - `val = 6`
  - `inference = 10`
- `test`는 대응하는 `total_data_test_*`를 복사

실행:

```bash
cd /home/user/projects/JamJamBeat/model
uv run python data_fusion/build_training_datasets.py
```

산출물 예시:

```text
data_fusion/학습데이터셋/
├── baseline_train.csv
├── baseline_val.csv
├── baseline_inference.csv
├── baseline_test.csv
├── baseline_ds_1_none_train.csv
├── baseline_ds_1_none_val.csv
├── ...
└── dataset_manifest.csv
```

## 5. 현재 파이프라인과의 연결 방식

이제 학습은 explicit 역할 CSV만 받는다.

[run_pipeline.py](/home/user/projects/JamJamBeat/model/model_pipelines/run_pipeline.py) 입력:

- `--train-csv`
- `--val-csv`
- `--test-csv`
- `--inference-csv`

예:

```bash
cd /home/user/projects/JamJamBeat/model
uv run python model_pipelines/run_pipeline.py \
  --model-id mlp_baseline \
  --train-csv data_fusion/학습데이터셋/baseline_ds_1_none_train.csv \
  --val-csv data_fusion/학습데이터셋/baseline_ds_1_none_val.csv \
  --test-csv data_fusion/학습데이터셋/baseline_ds_1_none_test.csv \
  --inference-csv data_fusion/학습데이터셋/baseline_ds_1_none_inference.csv
```

주의:

- 예전 `--csv-path` 단일 통합 CSV 흐름은 deprecated 상태다
- 예전 `trainval/test` 파일명 기준 문서는 현재 구조와 맞지 않는다

## 6. 배치 실행

전체 dataset key를 자동 인식해서 비-pretrained 모델 11개를 순차 실행하려면:

```bash
cd /home/user/projects/JamJamBeat/model
uv run python model_pipelines/run_no_pretrained.py
```

전체 14개 모델까지 포함하려면:

```bash
uv run python model_pipelines/run_no_pretrained.py --include-pretrained
```

## 7. 데이터셋 해석시 기억할 점

- `inference`는 hold-out 세트다
  - 학습/검증/공식 test ranking에는 섞지 않는다
- `test`는 기존 `total_data_test_*` 기반 외부 test다
- same-normalization test만 유지한다
- 12개 dataset key는 교차 48조합이 아니라, 12개 학습원천 세트다

## 8. 추천 확인 포인트

새 세트를 다시 만든 뒤에는 [dataset_manifest.csv](/home/user/projects/JamJamBeat/model/data_fusion/학습데이터셋/dataset_manifest.csv)에서 아래를 확인하면 된다.

- dataset key별 파일 4종이 모두 있는지
- `train / val / inference` source 수가 `40 / 6 / 10`인지
- 같은 family 내부 3세트가 동일한 test family를 쓰는지
