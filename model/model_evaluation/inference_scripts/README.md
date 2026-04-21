# inference_scripts - checkpoint / variant 진단 도구

이 폴더는 현재 `run_pipeline.py`나 `run_all.py`의 공식 평가를 대체하지 않습니다.
주 용도는 `여러 run이 정말 다른 checkpoint를 읽었는지`, `dataset_variant가 실제 입력을 바꾸는지`를
빠르게 확인하는 디버깅입니다.

현재 유지하는 스크립트는 아래 하나입니다.

- `compare_runtime_variants.py`
  - 여러 `run_dir`를 한 번에 비교
  - 같은 raw landmark 또는 같은 video frame을 각 run에 넣음
  - checkpoint fingerprint, variant 적용 후 입력 hash, 최종 모델 입력 hash, logits/probs, top-k를 비교

## 언제 쓰나

- 모델을 바꿨는데도 같은 프레임에서 같은 오탐이 반복될 때
- checkpoint가 제대로 로드되는지 의심될 때
- `baseline / pos_only / scale_only / pos_scale` 전처리 분기가 실제로 먹는지 확인하고 싶을 때

## 입력 방식

둘 중 하나를 선택합니다.

- `--video` + `--frame-idx`
  - 실제 raw video의 특정 프레임에서 landmark를 한 번 추출한 뒤 여러 run에 공통 입력으로 사용
- `--raw-landmarks-json` 또는 `--raw-landmarks-npz`
  - 이미 뽑아둔 `(21, 3)` raw landmark를 직접 비교 입력으로 사용

## 출력 내용

run별로 아래를 비교합니다.

- checkpoint fingerprint
- model id / mode / dataset variant
- raw landmarks hash
- variant 적용 후 landmarks hash
- final model input hash
- logits hash
- probs hash
- top-k 예측

필요하면 JSON/CSV 리포트로 저장할 수 있습니다.

## 실행 예시

```bash
python model/model_evaluation/inference_scripts/compare_runtime_variants.py \
  --run-dir \
  model/model_evaluation/pipelines/20260318_170030__baseline/mlp_baseline/20260318_170101 \
  model/model_evaluation/pipelines/20260318_170910__pos_only_ds_4_pos/mlp_baseline/20260318_170937 \
  --raw-landmarks-json /tmp/jjb_raw_landmarks.json \
  --output-json /tmp/jjb_compare_report.json \
  --output-csv /tmp/jjb_compare_report.csv
```

video frame를 바로 비교할 때:

```bash
python model/model_evaluation/inference_scripts/compare_runtime_variants.py \
  --run-dir \
  model/model_evaluation/pipelines/20260318_170030__baseline/mlp_baseline/20260318_170101 \
  model/model_evaluation/pipelines/20260318_170910__pos_only_ds_4_pos/mlp_baseline/20260318_170937 \
  --video data/raw_data/2_test_right_woman1.mp4 \
  --frame-idx 1
```

## 참고

- 공식 학습/평가는 계속 `run_pipeline.py`, `run_all.py`, 각 run의 `evaluation/` 산출물을 기준으로 봅니다.
- 예전의 `full_inference` 재생성 스크립트는 현재 운영 흐름에서 제외했습니다.
