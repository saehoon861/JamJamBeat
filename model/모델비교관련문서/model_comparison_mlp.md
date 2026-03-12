# JamJamBeat MLP Sequence Variant Comparison

기준일: 2026-03-10

이 문서는 `frame MLP`가 강한 현재 프로젝트에서, "시간 정보를 MLP 계열에 어떻게 넣을 것인가"를 보기 위해 만든 3개 sequence-MLP 파이프라인을 정리한 문서다.

대상 파이프라인:

- `mlp_sequence_joint`
- `mlp_temporal_pooling`
- `mlp_sequence_delta`

공통 실험 조건:

- 입력 소스: `model/data_fusion/man1_right_for_poc_output.csv`, `man2`, `man3`, `woman1`
- split: group split (`train 2명 / val 1명 / test 1명`)
- `seq_len=16`
- `seq_stride=2`
- `seed=42`
- 실행 디바이스: `cpu`

## 1. 파이프라인별 구조

### 1.1 `mlp_sequence_joint`

파일:

- [model.py](/home/user/projects/JamJamBeat/model/model_pipelines/mlp_sequence_joint/model.py)
- [dataset.py](/home/user/projects/JamJamBeat/model/model_pipelines/mlp_sequence_joint/dataset.py)

입력:

- `B x 16 x 63`
- feature는 정규화된 `joint(63d)`만 사용

구조:

- 시퀀스를 그대로 flatten
- `16 x 63 = 1008`
- `1008 -> 256 -> 128 -> class`

의도:

- 가장 단순한 sequence baseline
- "복잡한 temporal 모델 없이, 여러 프레임을 한 번에 넣는 것만으로 이득이 있는가"를 확인

장점:

- 구현이 가장 단순함
- 뷰어/배포도 쉬움

약점:

- 시간축 구조를 전혀 명시적으로 반영하지 않음
- 위치가 조금만 흔들려도 flatten된 큰 벡터가 쉽게 과적합될 수 있음

### 1.2 `mlp_temporal_pooling`

파일:

- [model.py](/home/user/projects/JamJamBeat/model/model_pipelines/mlp_temporal_pooling/model.py)
- [dataset.py](/home/user/projects/JamJamBeat/model/model_pipelines/mlp_temporal_pooling/dataset.py)

입력:

- `B x 16 x 63`
- feature는 정규화된 `joint(63d)`만 사용

구조:

- 프레임별 MLP embedding
- 각 프레임을 `63 -> 128 -> 128`로 임베딩
- 시간축에서 `mean`, `max`, `std` pooling
- pooled vector를 concat해서 head에 전달
- 최종 head: `384 -> 256 -> 128 -> class`

의도:

- frame 단위 feature는 유지하면서
- 순서 자체보다 "구간 전체의 통계량"을 보도록 설계

장점:

- 시퀀스 길이에 덜 민감함
- flatten보다 temporal shift에 조금 더 강함

약점:

- 순서를 거의 버림
- 이 태스크에서 중요한 짧은 모양 변화가 pooling 과정에서 뭉개질 수 있음

### 1.3 `mlp_sequence_delta`

파일:

- [model.py](/home/user/projects/JamJamBeat/model/model_pipelines/mlp_sequence_delta/model.py)
- [dataset.py](/home/user/projects/JamJamBeat/model/model_pipelines/mlp_sequence_delta/dataset.py)

입력:

- 원본 joint sequence: `B x 16 x 63`
- 추가 delta: `x_t - x_(t-1)`
- 최종 입력: `B x 16 x 126`

구조:

- 각 프레임에서 `joint(63) + delta(63)`를 결합
- 전체 시퀀스를 flatten
- `16 x 126 = 2016`
- `2016 -> 256 -> 128 -> class`

의도:

- 정적인 손 모양뿐 아니라 "이전 프레임 대비 얼마나 변했는가"를 같이 넣기
- TCN/Transformer 수준의 복잡도 없이 motion 신호를 MLP에 제공

장점:

- motion 정보가 가장 직접적으로 들어감
- 구현이 여전히 단순함

약점:

- sequence flatten 특유의 취약점은 그대로 남음
- delta가 noise도 같이 증폭할 수 있음

## 2. 현재 실험 결과

실행 결과:

- [mlp_sequence_joint run_summary.json](/home/user/projects/JamJamBeat/model/model_evaluation/pipelines/mlp_sequence_joint/20260310_104416/run_summary.json)
- [mlp_temporal_pooling run_summary.json](/home/user/projects/JamJamBeat/model/model_evaluation/pipelines/mlp_temporal_pooling/20260310_105241/run_summary.json)
- [mlp_sequence_delta run_summary.json](/home/user/projects/JamJamBeat/model/model_evaluation/pipelines/mlp_sequence_delta/20260310_105305/run_summary.json)

| model_id | input | best_val_loss | epochs | accuracy | macro_f1 | class0_fnr | fp_per_min | latency_p50_ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_sequence_joint` | `16 x 63` | `0.0951` | `10` | `0.3256` | `0.3639` | `0.8679` | `1.8556` | `0.0031` |
| `mlp_temporal_pooling` | `16 x 63` | `0.1230` | `7` | `0.0899` | `0.0957` | `1.0000` | `0.0000` | `0.0148` |
| `mlp_sequence_delta` | `16 x 126` | `0.1063` | `12` | `0.3682` | `0.3863` | `0.7815` | `0.0000` | `0.0046` |

현재 3개 중 순위:

1. `mlp_sequence_delta`
2. `mlp_sequence_joint`
3. `mlp_temporal_pooling`

## 3. 해석

### 3.1 `mlp_sequence_delta`가 3개 중 가장 나은 이유

- 단순 flatten MLP의 구조는 유지하면서도
- `delta`를 넣어 최소한의 temporal signal을 제공했기 때문으로 해석할 수 있다
- 즉, 이 데이터에서는 "시간 정보가 전혀 필요 없는 것"은 아니지만
- 그 정보를 복잡한 sequence model보다 아주 약하게 주는 편이 더 안정적이었다

### 3.2 `mlp_sequence_joint`가 기대보다 낮은 이유

- `16 x 63`을 그대로 flatten하면 사람별 스타일 차이, 위치 흔들림, 프레임 정렬 오차까지 한 벡터에 같이 들어간다
- sequence 구조를 쓰지만 temporal inductive bias는 없어서
- 실제로는 "긴 입력을 받는 큰 frame-MLP"처럼 동작한다

### 3.3 `mlp_temporal_pooling`이 가장 안 나온 이유

- mean/max/std pooling은 순서를 거의 버린다
- 이 태스크가 "구간 전체 통계"보다 "특정 프레임/짧은 구간의 손 모양"에 더 의존한다면 성능이 바로 무너질 수 있다
- 현재 결과상 `class0_fnr = 1.0`이라 neutral 처리도 거의 못 하고 있다
- 즉, 이 데이터셋에는 pooling 기반 요약이 지나치게 거칠었던 것으로 보는 편이 맞다

## 4. `frame MLP` 대비 의미

참고:

- [mlp_baseline run_summary.json](/home/user/projects/JamJamBeat/model/model_evaluation/pipelines/mlp_baseline/20260310_082223/run_summary.json)
- [mlp_embedding run_summary.json](/home/user/projects/JamJamBeat/model/model_evaluation/pipelines/mlp_embedding/20260310_082237/run_summary.json)

비교하면:

- `mlp_baseline` macro_f1: `0.5857`
- `mlp_embedding` macro_f1: `0.7051`
- `mlp_sequence_delta` macro_f1: `0.3863`

즉, 현재 데이터와 split 조건에서는 "sequence MLP를 붙이는 것"보다 "frame feature를 더 잘 표현하는 것"이 훨씬 중요하다.

이건 아래 둘 중 하나를 시사한다.

1. 제스처 판별 신호의 대부분이 프레임 단위 손 모양에 있다
2. 현재 sequence 모델들이 사람별 스타일 차이와 frame alignment noise를 더 크게 먹고 있다

## 5. 내가 내리는 결론

현재 상태에서 우선순위는 아래가 맞다.

1. `mlp_embedding`
2. `two_stream_mlp`
3. `mlp_baseline`
4. sequence MLP 계열은 보류

sequence MLP 계열 내부에서는:

1. `mlp_sequence_delta`
2. `mlp_sequence_joint`
3. `mlp_temporal_pooling`

## 6. 다음 개선안

sequence 쪽을 더 보려면 아래가 더 유망하다.

### A. `frame embedding + attention-free weighted pooling`

- 단순 mean/max/std 대신
- 프레임 중요도를 학습하는 soft weighting 추가
- temporal pooling의 정보 손실을 줄일 수 있음

### B. `joint + delta + delta norm`

- `delta`뿐 아니라 프레임별 이동량 norm을 별도 채널로 추가
- motion magnitude를 더 안정적으로 전달 가능

### C. `two_stream_mlp + delta`

- 현재 가장 강한 MLP 계열은 frame 표현 쪽이다
- 따라서 sequence를 무리하게 붙이기보다
- `joint stream`과 `bone/angle stream`에 motion 신호를 추가하는 편이 더 현실적일 수 있다

### D. 사람 기준 split 확대

- 현재 source group 수가 작아서 sequence 모델이 일반화하기 어렵다
- 사람 수나 촬영 조건 수가 늘어나면 sequence 계열의 가치가 다시 올라갈 수 있다

## 7. 요약

- `mlp_sequence_joint`: 가장 단순한 sequence baseline이지만 현재 데이터에서는 약했다
- `mlp_temporal_pooling`: pooling으로 시간 정보를 요약했지만 정보 손실이 커서 가장 안 좋았다
- `mlp_sequence_delta`: 세 모델 중 가장 나았지만, 여전히 `frame MLP` 계열보다 크게 뒤처졌다
- 현재 프로젝트에서는 sequence를 억지로 넣는 것보다 `frame 표현력 개선`이 우선이다
