# inference_scripts - pipeline run별 전체 landmark frame 추론 결과 생성

이 폴더의 스크립트는 `model/model_evaluation/pipelines/` 아래의 각 run 폴더에 대해,
브라우저 대시보드나 `preds_test.csv`보다 `jamjambeat-viewer`에 더 가까운 방식으로
`전체 landmark frame 추론 결과`를 생성한다.

핵심 차이:

- `preds_test.csv`는 test split 프레임만 포함한다.
- 이 스크립트는 `model.pt + raw video`를 사용해 전체 비디오를 다시 돌린다.
- 출력은 `손이 검출된 frame` 기준으로 저장되며, sequence 모델은 `no_hand` 구간에서 buffer reset까지 viewer와 같은 규칙을 따른다.

## 출력 위치

각 run 폴더 안에 아래가 생성된다.

```text
model/model_evaluation/pipelines/{suite}/{model_id}/{run_id}/
└── full_inference/
    ├── full_preds_landmark_frames.csv
    └── summary.json
```

## CSV 컬럼

- `source_file`
- `frame_idx`
- `timestamp`
- `gesture`, `gesture_name`
- `pred_class`, `pred_label`, `p_max`
- `status`
- `gt_available`, `is_mismatch`
- `stored_test_available`
- `stored_test_pred_class`, `stored_test_pred_label`, `stored_test_p_max`
- `runtime_matches_stored_test`
- `p0 ... pN`

설명:

- `pred_*`는 새로 생성한 전체-frame runtime 추론 결과다.
- `stored_test_*`는 기존 `preds_test.csv`의 원본 값이 같은 frame에 존재할 때만 채워진다.

## 실행 예시

repo 루트에서:

```bash
model/.venv/bin/python model/model_evaluation/inference_scripts/generate_full_landmark_inference.py \
  --suite 20260313_175343__ds_4_scale \
  --model mlp_embedding \
  --run-id 20260313_175551
```

suite 전체:

```bash
model/.venv/bin/python model/model_evaluation/inference_scripts/generate_full_landmark_inference.py \
  --suite 20260313_175343__ds_4_scale
```

전체 pipelines:

```bash
model/.venv/bin/python model/model_evaluation/inference_scripts/generate_full_landmark_inference.py
```

기존 결과 덮어쓰기:

```bash
model/.venv/bin/python model/model_evaluation/inference_scripts/generate_full_landmark_inference.py \
  --suite 20260313_175343__ds_4_scale \
  --force
```

## 주의

- 이 스크립트는 `data/raw_data/` 비디오를 실제로 다시 읽고 MediaPipe로 손을 검출하므로, `preds_test.csv` 생성보다 느리다.
- 대신 viewer와 동일한 `raw video -> landmark -> feature -> checkpoint inference` 흐름을 따르므로, 브라우저용 full inference artifact를 만들 때 더 적합하다.
