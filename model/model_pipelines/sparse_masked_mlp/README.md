# sparse_masked_mlp

`sparse_masked_mlp`는 손 해부학 adjacency를 선형층 가중치에 직접 반영하는 frame 분류 파이프라인이다. 목적은 "연결된 관절만 서로 영향을 줄 수 있게" 만드는 topology-aware baseline을 만드는 것이며, 입력은 raw 63D landmark만 사용한다.

## 핵심 아이디어

- MediaPipe hand landmark는 21개 landmark 인덱스를 고정된 형태로 제공한다.
- 손 skeleton의 물리 연결을 adjacency로 만들고, 이를 `masked linear`의 block mask로 확장한다.
- adjacency가 0인 joint pair에 대응하는 weight block은 항상 0이므로:
  - forward에서 정보가 섞이지 않고
  - backward에서도 해당 weight gradient가 0으로 차단된다.

## 입력과 출력

- 입력:
  - 모델 직접 호출 시 `(B, 63)` 또는 `(B, 21, 3)`
  - dataset 샘플은 `(21, 3)`
- 출력:
  - `(B, num_classes)` logits

## 모델 구조

```text
X (21x3 raw landmarks)
 -> MaskedStage 1: AnatomicalMaskedLinear(3 -> 64)
 -> MaskedStage 2: AnatomicalMaskedLinear(64 -> 64) + residual
 -> MaskedStage 3: AnatomicalMaskedLinear(64 -> 64) + residual
 -> MeanPool over 21 joints
 -> MLP head: 64 -> 128 -> num_classes
```

각 `MaskedStage`는 아래 순서를 따른다.

```text
AnatomicalMaskedLinear
 -> GELU
 -> LayerNorm
 -> Dropout(0.1)
```

## adjacency와 masked linear

- adjacency는 `_shared.HAND_CONNECTIONS`에서 가져온 undirected edge를 사용한다.
- self-loop를 포함하므로 각 관절은 자기 자신의 feature도 참조할 수 있다.
- `expand_block_mask(A, out_block, in_block)`는 `(21, 21)` adjacency를 `(21*out_block, 21*in_block)` mask로 확장한다.
- 실제 연산은 `W_eff = W * M` 형태로 이루어진다.

즉, 이 모델은 graph convolution처럼 adjacency를 aggregation 규칙으로 쓰는 대신, 선형층 가중치의 허용 연결 자체를 해부학 prior로 제한한다.

다만 현재 구현은 sparse 연산 커널이 아니라 `dense weight * binary mask` 방식이므로, 유효 연결은 sparse여도 저장 파라미터 수와 메모리 사용량은 dense block matrix에 가깝다. 기본 설정(`hidden_dim=64`, `num_classes=7`) 기준 파라미터 수는 약 `3.71M`이다.

## 필요한 데이터 컬럼

최소한 아래 컬럼이 필요하다.

- `source_file`
- `frame_idx`
- `timestamp`
- `gesture`
- `x0..x20`
- `y0..y20`
- `z0..z20`

이 파이프라인은 `_shared.RAW_JOINT_COLS`만 읽고, 추가 angle feature는 사용하지 않는다.

## 전처리 정책

- raw 63D landmark를 그대로 사용한다.
- `(N, 63)`을 `(N, 21, 3)`으로 reshape만 한다.
- root-relative normalization 없음
- scale normalization 없음
- handedness reflection 없음

## 웹 검증 기반 적합성 판단

### 적합

- 단일 프레임 hand landmark 분류가 목표일 때
- 모델 경량화와 구현 단순성이 중요할 때
- 해부학 prior를 강하게 넣고 싶을 때
- "손 뼈대 연결을 벗어난 임의 상호작용"보다 local anatomical consistency가 더 중요한 baseline이 필요할 때

### 조건부 적합

- 정적 제스처 중심이지만 손가락 간 비인접 상호작용도 어느 정도 필요한 경우
- 이 경우 `sparse_masked_mlp`는 좋은 시작점이지만, 표현력 부족이 보이면 hidden width를 `64 -> 96`으로 키우거나 masked stage를 1개 더 쌓는 편이 자연스럽다.

### 부적합 또는 baseline 전용

- 시간축 정보가 중요한 dynamic gesture
- 인접하지 않은 joint 간 implicit correlation이 성능에 큰 영향을 주는 경우
- sample-specific topology나 channel-wise topology refinement가 필요한 경우

### 왜 이렇게 판단하는가

- MediaPipe 공식 문서는 hand landmark를 21개 index로 제공하고, handedness와 world landmarks도 함께 제공한다. 즉, 손 skeleton prior를 landmark graph로 정리하는 전제 자체는 자연스럽다.
- MADE는 masked weights가 간단하고 GPU에서 빠르게 구현 가능하다고 보여준다. 그래서 adjacency mask를 dense linear에 얹는 방식은 구현 난도가 낮고 디버깅이 쉽다.
- 반면 2s-AGCN은 manually fixed topology가 모든 샘플과 모든 layer에 최적은 아닐 수 있다고 지적한다.
- AS-GCN은 fixed skeleton graph가 local physical dependency만 잡고 implicit correlation을 놓칠 수 있다고 설명한다.
- CTR-GCN은 channel-wise topology refinement가 representation을 더 강하게 만든다고 주장한다.
- SiT-MLP는 topology를 더 유연하게 학습하는 gating 기반 MLP 대안이 경쟁력 있음을 보여준다.

정리하면, `sparse_masked_mlp`는 좋은 topology-aware baseline이지만 표현력의 상한은 고정 adjacency가 결정한다. 따라서 "해부학 prior가 강한 경량 baseline"으로는 적합하지만, 더 복잡한 latent topology를 직접 학습하는 모델의 대체재로 보는 것은 무리다.

## 학습 레시피

- Loss:
  - 기본 `CrossEntropyLoss`
  - 필요 시 label smoothing `0.05 ~ 0.1`
- Optimizer:
  - `AdamW`
  - weight decay `1e-3 ~ 1e-2`
- LR schedule:
  - cosine decay
  - warmup `5%`
- Regularization:
  - dropout `0.1 ~ 0.2`
- 표현력이 부족하면:
  - hidden dim `64 -> 96`
  - masked stage 1개 추가

## 구현 범위

- 현재 폴더 내부에만 파이프라인을 추가한다.
- `run_pipeline.py` 등록은 하지 않는다.
- 웹 검증 로직은 runtime code가 아니라, 본 README 안의 정적 판단 규칙으로 정리한다.

## 1차 출처 링크

- MediaPipe Hand Landmarker guide:
  - https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
- MediaPipe HandLandmark enum:
  - https://ai.google.dev/edge/api/mediapipe/java/com/google/mediapipe/tasks/vision/handlandmarker/HandLandmark
- MADE (PMLR 2015):
  - https://proceedings.mlr.press/v37/germain15.html
- 2s-AGCN (CVPR 2019):
  - https://openaccess.thecvf.com/content_CVPR_2019/html/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.html
- AS-GCN (CVPR 2019):
  - https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Actional-Structural_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.html
- CTR-GCN (ICCV 2021):
  - https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Channel-Wise_Topology_Refinement_Graph_Convolution_for_Skeleton-Based_Action_Recognition_ICCV_2021_paper.html
- SiT-MLP (arXiv / TCSVT 2024):
  - https://arxiv.org/abs/2308.16018
