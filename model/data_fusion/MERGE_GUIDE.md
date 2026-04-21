# MERGE_GUIDE.md - 현재 기준 data_fusion 세트 생성 규칙

이 문서는 예전 `merged_*_trainval.csv` 흐름 대신, 현재 사용하는 explicit split 체계를 짧게 정리한 가이드다.

## 1. 지금은 무엇을 직접 merge하지 않는가

현재 표준 흐름에서는 아래를 수동으로 만들지 않는다.

- `merged_baseline_trainval.csv`
- `merged_baseline_test.csv`
- 기타 `merged_*_trainval.csv`, `merged_*_test.csv`

이 방식은 예전 문맥이고, 현재 파이프라인 구조와 맞지 않는다.

## 2. 현재 표준 출력 형식

이제는 dataset key마다 아래 4파일을 만든다.

- `{dataset_key}_train.csv`
- `{dataset_key}_val.csv`
- `{dataset_key}_inference.csv`
- `{dataset_key}_test.csv`

예:

- `baseline_train.csv`
- `baseline_val.csv`
- `baseline_inference.csv`
- `baseline_test.csv`

## 3. 분할 원칙

핵심 원칙은 다음과 같다.

- row 단위 분할을 하지 않는다
- `source_file` 단위로 video-level split 한다
- seed는 `42`
- source 개수는 고정:
  - train `40`
  - val `6`
  - inference `10`
- `test`는 기존 `테스트데이터셋/total_data_test_*`를 그대로 사용한다

즉 `inference`는 학습/검증/공식 test와 완전히 분리된 hold-out 세트다.

## 4. 현재 dataset key 체계

현재는 총 12개 학습원천 세트를 사용한다.

### baseline family

- `baseline`
- `baseline_ds_1_none`
- `baseline_ds_4_none`

공통 test:

- `total_data_test_baseline.csv`

### pos_only family

- `pos_only`
- `pos_only_ds_1_pos`
- `pos_only_ds_4_pos`

공통 test:

- `total_data_test_pos_only.csv`

### scale_only family

- `scale_only`
- `scale_only_ds_1_scale`
- `scale_only_ds_4_scale`

공통 test:

- `total_data_test_scale_only.csv`

### pos_scale family

- `pos_scale`
- `pos_scale_ds_1_pos_scale`
- `pos_scale_ds_4_pos_scale`

공통 test:

- `total_data_test_pos_scale.csv`

## 5. 실제 생성은 무엇이 담당하나

[build_training_datasets.py](/home/user/projects/JamJamBeat/model/data_fusion/build_training_datasets.py)가 위 규칙을 그대로 적용해 파일을 생성한다.

실행:

```bash
cd /home/user/projects/JamJamBeat/model
uv run python data_fusion/build_training_datasets.py
```

산출물:

- `model/data_fusion/학습데이터셋/`
- `model/data_fusion/학습데이터셋/dataset_manifest.csv`

## 6. 현재 파이프라인 입력 방식

[run_pipeline.py](/home/user/projects/JamJamBeat/model/model_pipelines/run_pipeline.py)는 더 이상 단일 `--csv-path`를 표준 입력으로 사용하지 않는다.

현재 표준 입력:

- `--train-csv`
- `--val-csv`
- `--test-csv`
- `--inference-csv`

즉 `학습데이터셋/`의 역할형 파일을 그대로 넣는 구조다.

## 7. 한 줄 권장안

현재 기준으로는:

1. `기존데이터셋/`과 `테스트데이터셋/`는 원천 데이터로 유지
2. `build_training_datasets.py`로 `학습데이터셋/`를 재생성
3. 학습은 `학습데이터셋/`의 explicit split CSV만 사용

이 흐름이 가장 일관되고 안전하다.
