# lappe_dist_mixer

`lappe_dist_mixer`는 Laplacian positional encoding과 shortest-path distance 기반 token mixing을 결합한 frame 분류 파이프라인이다. 이 구조는 특정 단일 논문의 직접 재현이 아니라, LapPE + SPD structural encoding + MLP-Mixer형 token/channel mixing을 static gesture 분류용으로 조합한 repo-specific adaptation이다.

## 핵심 아이디어

- LapPE는 각 관절이 그래프 전체에서 어떤 구조적 역할을 가지는지 나타내는 absolute positional encoding 역할을 한다.
- SPD는 두 관절 사이의 상대적 그래프 거리를 나타내는 structural encoding 역할을 한다.
- self-attention 대신, 거리 버킷별로 학습되는 channel-wise mixing 계수 `alpha[d]`를 사용해 token mixing을 수행한다.
- 그 뒤 joint별 channel MLP를 적용해 MLP-Mixer와 유사한 token/channel 분리를 유지한다.

## 입력과 출력

- 입력:
  - 모델 직접 호출 시 `(B, 63)` 또는 `(B, 21, 3)`
  - dataset 샘플은 `(21, 3)`
- 출력:
  - `(B, num_classes)` logits

## 아키텍처 다이어그램

```text
Precompute on hand graph:
  adjacency A (with self-loops)
  SPD matrix S in {0..d_max}^{21x21}
  LapPE E in R^{21x8}

X (21x3 raw landmarks)
 -> concat LapPE: [x, y, z, E] -> (21x11)
 -> Shared Linear(11 -> 64)

 -> DistMixerBlock x 3
    - Token mixing:
        H = H + DistTokenMix(LN(H))
        DistTokenMix(H) = sum_d (B_d @ H) * alpha[d]
    - Channel mixing:
        H = H + MLP(LN(H))
        MLP = 64 -> 128 -> 64

 -> LayerNorm
 -> GlobalAvgPool over 21 joints
 -> Linear(64 -> G)
```

## 내부 구조

### LapPE 계산

- adjacency는 `_shared.HAND_CONNECTIONS` 기반 undirected hand graph에 self-loop를 더해 만든다.
- normalized Laplacian:
  - `L = I - D^{-1/2} A D^{-1/2}`
- `torch.linalg.eigh(L)`로 고유분해하고, 첫 번째 trivial eigenvector를 제외한 `k=8`개를 사용한다.

### SPD 계산

- undirected hand edges에 대해 BFS를 반복해 `21x21` shortest-path distance 행렬을 만든다.
- self-loop는 SPD 계산에서 거리 0의 diagonal로 반영된다.

### DistTokenMix

- 각 거리 버킷 `d`에 대해 `B_d = 1{SPD == d}`를 만든다.
- token mixing 식:
  - `H_out = sum_d (B_d @ H) * alpha[d]`
- `alpha[d]`는 `(64,)` shape의 learnable channel-wise 계수다.

### Channel mixing

- 각 joint별 MLP:
  - `64 -> 128 -> 64`

## sign flip과 alpha 규제 helper

- Laplacian eigenvector는 부호가 임의로 뒤집힐 수 있으므로, 모델은 `train()`일 때만 LapPE에 random sign flip을 적용할 수 있다.
- 기본값은 `lappe_sign_flip=True`다.
- `alpha_l2_penalty()` helper는 모든 DistMixer block의 `alpha`에 대한 평균 L2 penalty scalar를 반환한다.
- 이번 범위에서는 training loop를 바꾸지 않으므로, sign flip과 alpha 규제 helper는 모델 API 차원에서만 제공한다.

## 필요한 데이터 컬럼

최소한 아래 컬럼이 필요하다.

- `source_file`
- `frame_idx`
- `timestamp`
- `gesture`
- `x0..x20`
- `y0..y20`
- `z0..z20`

이 파이프라인은 `_shared.RAW_JOINT_COLS`만 읽고, 실제 dataset은 raw 63D landmark만 사용한다.

## 전처리 정책

- 실제 dataset 코드는 raw 63D landmark를 그대로 사용한다.
- `(N, 63)`을 `(N, 21, 3)`으로 reshape만 한다.
- 아래 전처리는 권장사항으로만 문서화하며, 이번 구현에는 넣지 않는다.
  - root-relative normalization
  - scale normalization
  - world landmark 기준 회전 정규화

## 웹 검증 기반 적합성 판단

### 적합

- static single-hand gesture 분류가 목표일 때
- global structural role과 relative graph distance를 함께 쓰고 싶을 때
- attention보다 가벼운 구조 인코딩 + mixer형 블록을 원할 때

### 조건부 적합

- 구조 인코딩은 중요하지만 full graph attention까지는 과하다고 판단될 때
- 이 경우 LapPE + SPD는 좋은 출발점이지만, 더 강한 pairwise adaptation이 필요하면 attention이나 adaptive graph block이 더 유리할 수 있다.

### 부적합 또는 baseline 전용

- temporal cue가 중요한 dynamic gesture
- 양손 상호작용이 중요한 경우
- sample-specific graph adaptation이 핵심일 때
- 좌표 기하 정규화가 성능의 핵심인데 아직 raw landmark만 쓰는 설정을 유지해야 할 때

### 왜 이렇게 판단하는가

- Graph positional encoding 벤치마크 계열 연구는 Laplacian eigenvector PE가 그래프 위치 정보를 주는 강한 baseline임을 보여준다.
- Graph transformer 연구는 Laplacian PE 같은 구조 인코딩이 그래프 입력에 유용하다고 제안한다.
- Graphormer는 SPD를 structural relation으로 쓰고, shortest-path distance를 bias로 넣는 방식이 효과적임을 보여준다.
- MLP-Mixer는 token mixing과 channel mixing의 분리 설계가 충분히 강력할 수 있음을 보여준다.
- Graph ViT / MLP-Mixer to Graphs는 mixer류 구조를 그래프에 일반화할 수 있음을 보여준다.

정리하면, `lappe_dist_mixer`는 static hand graph에서 구조 인코딩을 강하게 활용하는 가벼운 adaptation으로는 타당하지만, attention 기반 pairwise modeling이나 temporal modeling을 직접 대체하는 구조로 보기는 어렵다. 이 평가는 아래 1차 출처를 바탕으로 한 명시적 추론이다.

## 구현 범위

- 현재 폴더 내부에만 파이프라인을 추가한다.
- `run_pipeline.py` 등록은 하지 않는다.
- 전처리 권장사항은 README에만 적고, dataset/runner 쪽 실제 코드 변경은 하지 않는다.

## 1차 출처 링크

- MediaPipe Hand Landmarker guide:
  - https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
- MediaPipe HandLandmark enum:
  - https://ai.google.dev/edge/api/mediapipe/java/com/google/mediapipe/tasks/vision/handlandmarker/HandLandmark
- Laplacian PE / graph positional encoding benchmark context:
  - https://jmlr.org/papers/v24/22-0567.html
- Graph transformer with Laplacian PE:
  - https://arxiv.org/abs/2012.09699
- Graphormer, SPD as structural bias:
  - https://arxiv.org/abs/2106.05234
- MLP-Mixer:
  - https://arxiv.org/abs/2105.01601
- Graph ViT / MLP-Mixer to Graphs:
  - https://arxiv.org/abs/2212.13350
