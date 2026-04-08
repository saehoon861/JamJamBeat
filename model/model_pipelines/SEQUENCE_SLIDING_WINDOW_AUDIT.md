# SEQUENCE_SLIDING_WINDOW_AUDIT.md - Sequence sliding window 설정 점검 기록

## Current Baseline

- Audit date: `2026-03-17`
- Dataset key: `baseline`
- Output root: `model/model_evaluation/pipelines_sliding_audit`
- Official test policy:
  - `*_test.csv = static_images_63d`
  - sequence official test = `independent_repeat`
- Runtime / inference policy:
  - `train/val = sliding window`
  - `inference = sliding`
  - runtime video step = `1 frame`
- Current default config:
  - `seq_len = 8`
  - `train/val seq_stride = 2`

## Why This Audit Exists

frame 모델은 현재 구조가 비교적 단순하지만, sequence 모델은 아래 mismatch 가능성이 있다.

- 학습은 `stride=2`인데 runtime video 추론은 `step=1`
- `seq_len`이 반응성, warmup, 안정성에 직접 영향을 준다
- 공식 `*_test.csv`는 정지사진 기반이라, 실제 online behavior는 `preds_inference.csv`와 viewer/runtime에서 따로 봐야 한다

이번 audit의 목적은 **구조를 바꾸지 않고**, 현재 기본값이 타당한지 실험으로 기록하는 것이다.

## Experiment Matrix

공통 조건:

- models:
  - `mlp_baseline_seq8`
  - `mlp_temporal_pooling`
  - `mlp_sequence_delta`
  - `cnn1d_tcn`
  - `transformer_embedding`
- device: `cuda`
- dataset key: `baseline`
- epochs: pipeline default

실험 조합:

| Label | seq_len | seq_stride | Suite |
|------|---------|------------|------|
| Baseline | `8` | `2` | `20260317_145730__baseline` |
| Candidate A | `8` | `1` | `20260317_150038__baseline` |
| Candidate B | `12` | `1` | `20260317_150634__baseline` |

실행 명령:

```bash
# Baseline
model/.venv/bin/python model/model_pipelines/run_all.py \
  --dataset-key baseline \
  --models mlp_baseline_seq8 mlp_temporal_pooling mlp_sequence_delta cnn1d_tcn transformer_embedding \
  --device cuda \
  --output-root model/model_evaluation/pipelines_sliding_audit

# Candidate A
model/.venv/bin/python model/model_pipelines/run_all.py \
  --dataset-key baseline \
  --models mlp_baseline_seq8 mlp_temporal_pooling mlp_sequence_delta cnn1d_tcn transformer_embedding \
  --device cuda \
  --seq-len 8 \
  --seq-stride 1 \
  --output-root model/model_evaluation/pipelines_sliding_audit

# Candidate B
model/.venv/bin/python model/model_pipelines/run_all.py \
  --dataset-key baseline \
  --models mlp_baseline_seq8 mlp_temporal_pooling mlp_sequence_delta cnn1d_tcn transformer_embedding \
  --device cuda \
  --seq-len 12 \
  --seq-stride 1 \
  --output-root model/model_evaluation/pipelines_sliding_audit
```

## Per-Run Results

### Baseline (`seq_len=8`, `seq_stride=2`)

| model_id | test_acc | test_macro_f1 | inference_acc | inference_macro_f1 | latency_p50_ms | epochs | verdict |
|------|------:|------:|------:|------:|------:|------:|------|
| `transformer_embedding` | 0.7671 | 0.7763 | 0.6092 | 0.6459 | 0.02 | 17 | baseline best official test |
| `mlp_temporal_pooling` | 0.7460 | 0.7469 | 0.5610 | 0.6137 | 0.01 | 15 | stable baseline |
| `mlp_sequence_delta` | 0.7300 | 0.7369 | 0.4572 | 0.5291 | 0.01 | 20 | weak on inference |
| `mlp_baseline_seq8` | 0.7249 | 0.7229 | 0.4254 | 0.4896 | 0.01 | 20 | weakest baseline |
| `cnn1d_tcn` | 0.7139 | 0.7310 | 0.6406 | 0.6703 | 0.02 | 20 | best baseline inference |

### Candidate A (`seq_len=8`, `seq_stride=1`)

| model_id | test_acc | test_macro_f1 | inference_acc | inference_macro_f1 | latency_p50_ms | epochs | verdict |
|------|------:|------:|------:|------:|------:|------:|------|
| `transformer_embedding` | 0.7570 | 0.7620 | 0.5973 | 0.6409 | 0.02 | 17 | slightly worse than baseline |
| `mlp_baseline_seq8` | 0.7536 | 0.7606 | 0.5307 | 0.5757 | 0.01 | 20 | clearly better than baseline |
| `cnn1d_tcn` | 0.7502 | 0.7571 | 0.5942 | 0.6304 | 0.02 | 11 | better test, worse inference |
| `mlp_temporal_pooling` | 0.6667 | 0.6443 | 0.4427 | 0.5237 | 0.02 | 9 | clearly worse |
| `mlp_sequence_delta` | failure | failure | failure | failure | failure | failure | process exited with code `-9` |

### Candidate B (`seq_len=12`, `seq_stride=1`)

| model_id | test_acc | test_macro_f1 | inference_acc | inference_macro_f1 | latency_p50_ms | epochs | verdict |
|------|------:|------:|------:|------:|------:|------:|------|
| `transformer_embedding` | 0.7840 | 0.7927 | 0.5846 | 0.6209 | 0.02 | 13 | best official test, worse inference than baseline |
| `mlp_sequence_delta` | 0.7570 | 0.7650 | 0.5634 | 0.6005 | 0.02 | 20 | much better than baseline |
| `mlp_baseline_seq8` | 0.7536 | 0.7606 | 0.5307 | 0.5757 | 0.01 | 20 | unchanged; see note below |
| `cnn1d_tcn` | 0.7426 | 0.7502 | 0.6224 | 0.6559 | 0.02 | 13 | slightly better than baseline, below 8/2 inference |
| `mlp_temporal_pooling` | 0.7409 | 0.7430 | 0.5469 | 0.5968 | 0.01 | 9 | below baseline |

## Interim Findings

### 1. `8/1` is not a universal upgrade

- `mlp_baseline_seq8`는 `8/1`에서 크게 좋아졌다
- 반면 `mlp_temporal_pooling`은 크게 나빠졌고
- `transformer_embedding`은 baseline보다 오히려 약간 떨어졌다
- `cnn1d_tcn`은 official test는 좋아졌지만 inference는 내려갔다

현재 기준으로 `8/1`을 공통 정책으로 바로 바꾸기에는 증거가 부족하다.

### 2. `12/1` is mixed

- `transformer_embedding`의 official test는 가장 좋았다
- `mlp_sequence_delta`도 baseline 대비 의미 있게 좋아졌다
- 하지만 `transformer_embedding`의 inference는 baseline보다 내려갔다
- `mlp_temporal_pooling`은 좋아지지 않았다

즉 `12/1`도 일부 모델에는 이득이 있지만 공통 정책으로 바로 채택할 정도로 일관적이지 않다.

### 3. `mlp_baseline_seq8` is a special case

`mlp_baseline_seq8`는 이름 그대로 dataset builder가 `SEQ_LEN=8` 고정이다.

- file: `model/model_pipelines/mlp_baseline_seq8/dataset.py`
- `seq_len` CLI 인자는 무시되고 항상 `8`을 사용한다

실제로 Candidate A와 Candidate B에서:

- checkpoint에는 `seq_len=12` 인자가 저장됐지만
- `state_dict['net.0.weight'].shape == [128, 504]`
- 즉 실제 학습 입력은 `8 x 63` 그대로였다

따라서 `mlp_baseline_seq8`의 `12/1` 결과는 **진짜 12-frame 실험으로 해석하면 안 된다**.

### 4. Runtime observation on one inference video

분석 대상:

- video: `data/raw_data/3_fast_right_woman1.mp4`
- representative models:
  - `mlp_temporal_pooling` from Candidate A (`8/1`)
  - `transformer_embedding` from Candidate B (`12/1`)

관찰 결과:

| model | seq_len | total_frames | no_hand | warmup_leading | first_ready | first_non_neutral_ready | first_stable_non_neutral |
|------|------:|------:|------:|------:|------:|------:|------:|
| `mlp_temporal_pooling` | 8 | 918 | 84 | 7 | 25 | 48 | 57 |
| `transformer_embedding` | 12 | 918 | 84 | 11 | 29 | 55 | 57 |

해석:

- `seq_len=12`는 예상대로 warmup 길이와 첫 non-neutral 예측 시점을 늦췄다
- 하지만 이 샘플에서는 first stable non-neutral frame이 둘 다 `57`로 같았다
- 즉 더 긴 window가 반드시 더 빠르게 안정화되는 것은 아니었다

## Decision Log

- `2026-03-17`: 구조 변경은 보류
- `2026-03-17`: 공식 기본값 `8/2`는 유지
- `2026-03-17`: `8/1`은 일부 모델에만 이득이 있어 공통 정책 채택 보류
- `2026-03-17`: `12/1`은 transformer / delta에는 가능성이 있지만 반응 지연과 model variance가 있어 공통 정책 채택 보류
- `2026-03-17`: `mlp_baseline_seq8`는 `seq_len` audit에서 예외 모델로 취급해야 함

## Next Recommended Test

다음 단계도 구조 변경 없이 실험만 이어간다.

1. `mlp_sequence_delta`의 `8/1` 실패를 단독 재실행해서 `-9`가 메모리 압박인지 확인
2. 같은 3개 config를 `baseline` 외 다른 dataset key 하나에서 반복
   - 추천: `baseline_ds_4_none` 또는 `scale_only_ds_4_scale`
3. 공통 정책 후보를 다시 볼 때는 아래 우선순위로 판단
   - `inference_acc / inference_macro_f1`
   - runtime warmup 및 first_non_neutral_ready
   - 그 다음 `official test` 수치

현재 임시 결론:

- **지금은 기본값을 바꾸지 않는 것이 가장 안전하다**
- 다만 `transformer_embedding`, `mlp_sequence_delta`는 `12/1` 추가 검토 가치가 있다
- `mlp_baseline_seq8`는 별도 예외로 두고 보아야 한다
