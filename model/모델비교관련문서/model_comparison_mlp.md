# JamJamBeat MLP 계열 파이프라인 비교

기준 suite:

- [comparison_suite.json](/home/user/projects/JamJamBeat/model/model_evaluation/pipelines/20260312_125119__man1_right_for_poc_notnull__man2_right_for_poc_notnull__plus2__ab316159/comparison_suite.json)
- [comparison_results.csv](/home/user/projects/JamJamBeat/model/model_evaluation/pipelines/20260312_125119__man1_right_for_poc_notnull__man2_right_for_poc_notnull__plus2__ab316159/comparison_results.csv)

기준일: 2026-03-12

이 문서는 현재 코드베이스의 **MLP 관련 분류 파이프라인**을 실제 구현 기준으로 정리한 문서다.  
중요한 점은, 일부 모델은 파일명/주석의 의도와 실제 입력 형식이 다르므로, 여기서는 **이름이나 설계 의도보다 실제 `dataset.py`/checkpoint가 쓰는 입력**을 기준으로 정리한다.

대상 모델:

- `mlp_baseline`
- `mlp_baseline_full`
- `mlp_baseline_seq8`
- `mlp_sequence_joint`
- `mlp_temporal_pooling`
- `mlp_sequence_delta`
- `mlp_embedding`
- `two_stream_mlp`

## 1. 실험 설정

- 입력 CSV: `man1_right_for_poc_notnull.csv`, `man2_right_for_poc_notnull.csv`, `man3_right_for_poc_notnull.csv`, `woman1_right_for_poc_notnull.csv`
- split: source group 기준 hold-out split
- train: `man1 + man2` (24,330 rows)
- val: `man3` (8,843 rows)
- test: `woman1` (12,310 rows)
- frame 모델 sample 수: train `24,330`, test `12,310`
- sequence 모델 sample 수: train `12,158`, test `6,152`
- 공통 학습: `epochs=20`, `patience=6`, `optimizer=AdamW`, `loss=FocalLoss`, train loader에 `WeightedRandomSampler`
- 디바이스: CPU

해석할 때 주의할 점:

- 이 split은 사실상 **사람 단위 일반화 성능**을 보는 테스트다.
- accuracy는 neutral 비중 영향을 크게 받는다.
- `fp_per_min`은 raw frame 예측이 아니라 **threshold/voting/debounce 이후의 trigger 기준**이라, `class0_fnr`과 바로 일치하지 않을 수 있다.

## 2. 지표가 말하는 것

| 지표 | 의미 | 이 프로젝트에서 읽는 법 |
|---|---|---|
| `accuracy` | 전체 프레임(또는 시퀀스 윈도우) 중 정답 비율 | neutral이 많으면 높아지기 쉬움 |
| `macro_f1` | 클래스별 F1 평균 | 클래스 불균형 영향을 덜 받는 주 비교 지표 |
| `macro_precision` | 클래스별 precision 평균 | gesture를 너무 많이 찍는지 보는 데 도움 |
| `macro_recall` | 클래스별 recall 평균 | gesture를 전반적으로 얼마나 놓치지 않는지 |
| `class0_fnr` | neutral 프레임을 gesture로 잘못 본 비율 | 낮을수록 안정적, 오발동 억제에 유리 |
| `fp_per_min` | 후처리 이후 분당 오발동 수 | 실제 trigger 관점의 운영 지표 |
| `latency_p50_ms` | 추론 지연 중간값 | MLP 계열은 거의 차별점이 아님 |

핵심적으로는 이렇게 보면 된다.

- **모델 비교 1순위:** `macro_f1`
- **실제 사용 안정성:** `class0_fnr`, `fp_per_min`
- **영상 체감상 "neutral을 너무 많이 깨는가":** `class0_fnr`
- **"정확도는 높은데 체감이 별로" 문제 확인:** `accuracy` 단독 해석 금지

## 3. 구현 파일

- [mlp_baseline/dataset.py](/home/user/projects/JamJamBeat/model/model_pipelines/mlp_baseline/dataset.py)
- [mlp_baseline/model.py](/home/user/projects/JamJamBeat/model/model_pipelines/mlp_baseline/model.py)
- [mlp_baseline_full/dataset.py](/home/user/projects/JamJamBeat/model/model_pipelines/mlp_baseline_full/dataset.py)
- [mlp_baseline_seq8/dataset.py](/home/user/projects/JamJamBeat/model/model_pipelines/mlp_baseline_seq8/dataset.py)
- [mlp_baseline_seq8/model.py](/home/user/projects/JamJamBeat/model/model_pipelines/mlp_baseline_seq8/model.py)
- [mlp_sequence_joint/dataset.py](/home/user/projects/JamJamBeat/model/model_pipelines/mlp_sequence_joint/dataset.py)
- [mlp_sequence_joint/model.py](/home/user/projects/JamJamBeat/model/model_pipelines/mlp_sequence_joint/model.py)
- [mlp_temporal_pooling/dataset.py](/home/user/projects/JamJamBeat/model/model_pipelines/mlp_temporal_pooling/dataset.py)
- [mlp_temporal_pooling/model.py](/home/user/projects/JamJamBeat/model/model_pipelines/mlp_temporal_pooling/model.py)
- [mlp_sequence_delta/dataset.py](/home/user/projects/JamJamBeat/model/model_pipelines/mlp_sequence_delta/dataset.py)
- [mlp_sequence_delta/model.py](/home/user/projects/JamJamBeat/model/model_pipelines/mlp_sequence_delta/model.py)
- [mlp_embedding/dataset.py](/home/user/projects/JamJamBeat/model/model_pipelines/mlp_embedding/dataset.py)
- [mlp_embedding/model.py](/home/user/projects/JamJamBeat/model/model_pipelines/mlp_embedding/model.py)
- [two_stream_mlp/dataset.py](/home/user/projects/JamJamBeat/model/model_pipelines/two_stream_mlp/dataset.py)
- [two_stream_mlp/model.py](/home/user/projects/JamJamBeat/model/model_pipelines/two_stream_mlp/model.py)

## 4. 파이프라인 구조 요약

| model_id | mode | 현재 실제 입력 | 구조 핵심 | 요약 |
|---|---|---|---|---|
| `mlp_baseline` | frame | raw joint `63d` | `63 -> 128 -> 64 -> class` | 가장 단순한 기준선 |
| `mlp_baseline_full` | frame | **실제 구현은 raw joint `63d`** | baseline과 동일 MLP | 이름과 달리 현재는 baseline과 사실상 같은 실험 |
| `mlp_embedding` | frame | **실제 구현은 raw joint `63d`** | `63 -> 128` learnable embedding 후 head | 같은 63d라도 projection/normalization을 추가한 변형 |
| `two_stream_mlp` | two_stream | `xy 42d` + `z 21d` | 두 스트림 독립 인코딩 후 late fusion | 현재 MLP 계열 중 표현 분리가 가장 명확함 |
| `mlp_baseline_seq8` | sequence | `8 x 63` | 8프레임 flatten 후 baseline head | 시간축을 그냥 길게 펴는 가장 단순한 sequence baseline |
| `mlp_sequence_joint` | sequence | `16 x 63` | 16프레임 flatten MLP | 더 긴 window를 보지만 temporal bias는 거의 없음 |
| `mlp_temporal_pooling` | sequence | `16 x 63` | frame embedding 후 last/max/std pooling | 순서 전체보다 구간 요약 통계를 보는 구조 |
| `mlp_sequence_delta` | sequence | `16 x 126` | `joint + delta`를 flatten | 움직임 차분을 직접 넣는 sequence MLP |

해석 포인트:

- `mlp_baseline_full`과 `mlp_embedding`은 주석상 `156d full feature`처럼 보이지만, **현재 `dataset.py`는 둘 다 `RAW_JOINT_COLS`만 사용**한다.
- `two_stream_mlp`도 bone/angle late fusion이 아니라, 현재 구현은 **`xy`와 `z` 분리**다.
- 즉 이번 suite 결과를 보고 곧바로 "bone/angle이 효과 없다"라고 결론 내리면 안 된다. 현재 실험은 그 비교를 깨끗하게 하지 못하고 있다.

## 5. 현재 결과

| model_id | accuracy | macro_f1 | macro_precision | macro_recall | class0_fnr | fp_per_min | latency_p50_ms | 비고 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `mlp_embedding` | `0.5833` | `0.5734` | `0.5442` | `0.6871` | `0.5118` | `20.224` | `0.00` | accuracy 최고 |
| `two_stream_mlp` | `0.4981` | `0.5761` | `0.5492` | `0.7512` | `0.7389` | `9.193` | `0.01` | macro_f1 최고, fp/min 최저 |
| `mlp_temporal_pooling` | `0.4426` | `0.4780` | `0.4552` | `0.6769` | `0.7765` | `0.000` | `0.02` | sequence MLP 중 최고 |
| `mlp_baseline` | `0.3582` | `0.4208` | `0.4096` | `0.6331` | `0.8982` | `18.386` | `0.00` | 단순 63d frame baseline |
| `mlp_baseline_full` | `0.3582` | `0.4208` | `0.4096` | `0.6331` | `0.8982` | `18.386` | `0.00` | baseline과 완전 동일 |
| `mlp_sequence_joint` | `0.2983` | `0.3786` | `0.4195` | `0.5978` | `0.9829` | `0.000` | `0.00` | 긴 flatten sequence |
| `mlp_baseline_seq8` | `0.2734` | `0.3453` | `0.3869` | `0.5525` | `0.9899` | `0.000` | `0.00` | 짧은 flatten sequence |
| `mlp_sequence_delta` | `0.2578` | `0.3290` | `0.4045` | `0.5238` | `0.9938` | `0.000` | `0.01` | delta 추가했지만 이번 suite에선 가장 약함 |

순위만 정리하면:

1. accuracy 기준: `mlp_embedding > two_stream_mlp > mlp_temporal_pooling > mlp_baseline`
2. macro_f1 기준: `two_stream_mlp > mlp_embedding > mlp_temporal_pooling > mlp_baseline`
3. sequence MLP 내부: `mlp_temporal_pooling > mlp_sequence_joint > mlp_baseline_seq8 > mlp_sequence_delta`

## 6. 클래스별 패턴

| model_id | neutral recall | 가장 잘 맞는 non-neutral class | 가장 약한 non-neutral class |
|---|---:|---|---|
| `mlp_baseline` | `0.1018` | `pinky (F1 0.6148)` | `animal (F1 0.2637)` |
| `mlp_baseline_full` | `0.1018` | `pinky (F1 0.6148)` | `animal (F1 0.2637)` |
| `mlp_baseline_seq8` | `0.0101` | `pinky (F1 0.6245)` | `fist (F1 0.2569)` |
| `mlp_sequence_joint` | `0.0171` | `pinky (F1 0.6294)` | `fist (F1 0.2993)` |
| `mlp_temporal_pooling` | `0.2235` | `pinky (F1 0.6690)` | `open_palm (F1 0.3833)` |
| `mlp_sequence_delta` | `0.0062` | `V (F1 0.5359)` | `fist (F1 0.2619)` |
| `mlp_embedding` | `0.4882` | `pinky (F1 0.7456)` | `animal (F1 0.3416)` |
| `two_stream_mlp` | `0.2611` | `pinky (F1 0.7356)` | `animal (F1 0.2746)` |

이 표가 보여주는 것은 명확하다.

- 대부분의 MLP 계열에서 **`pinky`는 상대적으로 쉬운 클래스**다.
- 반대로 **`animal`은 거의 공통적으로 약한 클래스**다.
- 많은 sequence MLP는 neutral recall이 매우 낮다. 즉 neutral 구간을 gesture로 깨는 경향이 강하다.
- `mlp_embedding`은 neutral recall을 가장 많이 회복해서 accuracy가 올라갔다.
- `two_stream_mlp`는 neutral에는 조금 더 공격적이지만 gesture 쪽 recall이 높아서 macro_f1이 조금 더 높다.

## 7. 결과 이유

### 7.1 `mlp_embedding`이 accuracy 1위인 이유

- 현재 구현 기준으로는 `156d full feature` 모델이 아니라, **동일한 raw joint 63d에 learnable projection을 한 번 더 넣은 MLP**다.
- baseline과 입력은 사실상 같은데, 첫 단계 `Linear + LayerNorm + GELU`가 representation을 정리해준다.
- 그 결과 neutral recall이 `0.1018 -> 0.4882`로 크게 올라갔다.
- 이 실험에서 accuracy 차이의 상당 부분은 neutral 처리에서 나왔다고 보는 게 맞다.

### 7.2 `two_stream_mlp`가 macro_f1 1위인 이유

- 현재 구현은 bone/angle이 아니라 **`xy`와 `z`를 분리**한다.
- 손 모양의 평면 구조와 깊이 신호를 따로 읽고 마지막에 합치기 때문에, gesture class recall이 전체적으로 올라간다.
- 그래서 `macro_recall=0.7512`로 MLP 계열 중 가장 높다.
- 대신 neutral recall은 `0.2611`로 `mlp_embedding`보다 낮아서 accuracy는 뒤진다.
- 즉, `two_stream_mlp`는 **gesture를 더 적극적으로 잡는 대신 neutral 안정성이 조금 떨어지는 모델**이다.

### 7.3 `mlp_temporal_pooling`이 sequence MLP 중 가장 나은 이유

- 이번 suite에서는 flatten 계열보다 pooling 계열이 더 잘 나왔다.
- 이유는 sequence를 길게 펼친 모델들이 사람별 pose shift와 frame alignment noise를 그대로 먹는 반면,
- pooling은 프레임별 embedding 뒤에 last/max/std 요약을 쓰므로, 시퀀스 전체를 하나의 거대한 좌표 벡터로 보는 방식보다 덜 민감하기 때문이다.
- neutral recall도 sequence MLP 중 가장 높은 `0.2235`라서 accuracy가 같이 올라갔다.

### 7.4 `mlp_sequence_delta`가 이번엔 안 나온 이유

- 이전에는 delta가 유리할 수 있다고 봤지만, 이번 suite에서는 오히려 제일 약했다.
- delta는 motion 정보를 주는 대신 작은 landmark 흔들림도 같이 키운다.
- 사람 단위 hold-out split에서는 이런 frame-to-frame noise가 style difference와 합쳐져 오히려 일반화를 해칠 수 있다.
- neutral recall이 `0.0062`라는 점이 특히 치명적이다.

### 7.5 `mlp_baseline_full` 결과를 해석할 때의 함정

- 이름만 보면 `joint + bone + angle = 156d` 실험처럼 보인다.
- 하지만 현재 [mlp_baseline_full/dataset.py](/home/user/projects/JamJamBeat/model/model_pipelines/mlp_baseline_full/dataset.py) 는 `RAW_JOINT_COLS`만 사용한다.
- 그래서 `mlp_baseline`과 수치가 완전히 같은 것은 "full feature가 무의미"의 증거가 아니라,
- **현재 구현상 같은 입력/같은 모델을 두 번 비교한 결과**라고 보는 게 맞다.

## 8. 지금 문서에서 내려야 할 결론

### 8.1 가장 실용적인 frame MLP 계열

- **accuracy/neutral 안정성 우선:** `mlp_embedding`
- **macro_f1/gesture recall 우선:** `two_stream_mlp`
- 둘 중 어느 쪽이 더 맞는지는 서비스 목적에 따라 다르다.

정리하면:

- false trigger를 더 줄이고 싶으면 `mlp_embedding`
- gesture 자체를 더 적극적으로 살리고 싶으면 `two_stream_mlp`

### 8.2 sequence MLP 계열

- 현재 suite 기준으로는 `mlp_temporal_pooling`만 상대적으로 의미가 있다.
- `mlp_baseline_seq8`, `mlp_sequence_joint`, `mlp_sequence_delta`는 strong frame MLP보다 분명히 뒤진다.
- 즉 "MLP에 시간축을 억지로 넣는 것"보다, 현재는 **frame 표현을 더 잘 만드는 것**이 더 중요했다.

### 8.3 이 결과를 과하게 해석하면 안 되는 부분

- 현재 suite는 사람 4명 수준의 작은 source-group split이다.
- 게다가 MLP 계열 일부는 이름/주석과 실제 구현이 다르다.
- 따라서 이 문서로 확실하게 말할 수 있는 것은
  "현재 구현 상태의 MLP 파이프라인 중 무엇이 더 낫나"까지다.
- "bone/angle이 무가치하다" 같은 강한 결론은 아직 이르다.

## 9. 추천 액션

1. 운영 관점 우선 비교는 `mlp_embedding` vs `two_stream_mlp` 두 개에 집중한다.
2. sequence MLP를 계속 볼 거면 `mlp_temporal_pooling`만 남기고 나머지는 우선순위를 낮춘다.
3. `mlp_baseline_full`, `mlp_embedding`의 실제 입력을 의도한 대로 `156d`로 고친 뒤 다시 비교해야 feature ablation 해석이 가능하다.
4. `animal` 클래스는 공통적으로 약하므로, 데이터 품질 또는 class definition 자체를 재검토하는 게 맞다.

## 10. 한 줄 요약

- **현재 구현 기준 MLP 최고 accuracy:** `mlp_embedding`
- **현재 구현 기준 MLP 최고 macro_f1:** `two_stream_mlp`
- **현재 구현 기준 sequence MLP 최고:** `mlp_temporal_pooling`
- **가장 중요한 caveat:** `mlp_baseline_full`과 `mlp_embedding`은 이름과 달리 지금은 full 156d 실험이 아니다
