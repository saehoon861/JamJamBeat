# hierarchical_tree_mlp

`hierarchical_tree_mlp`는 손 관절을 트리로 보고, wrist에서 시작해 parent→child 순서로 hidden state를 전달하는 frame 분류 파이프라인이다. 이 구조는 특정 단일 논문의 직접 재현이 아니라, hand-topology/tree-routing 아이디어를 static gesture 분류용으로 조합한 repo-specific adaptation이다.

## 핵심 아이디어

- root인 wrist에서 먼저 hidden feature를 만든다.
- 각 자식 관절은 자기 좌표만 보지 않고, 부모 hidden과 부모 대비 상대 좌표를 함께 입력받는다.
- palm branch와 finger branch에 서로 다른 edge MLP 템플릿을 두어, 손바닥 라우팅과 손가락 체인 라우팅을 분리한다.
- 마지막에는 wrist와 5개 fingertip의 hidden을 읽어 분류 head로 보낸다.

## 입력과 출력

- 입력:
  - 모델 직접 호출 시 `(B, 63)` 또는 `(B, 21, 3)`
  - dataset 샘플은 `(21, 3)`
- 출력:
  - `(B, num_classes)` logits

## 아키텍처 다이어그램

```text
Root MLP:
  wrist x0 -> [3 -> 32 -> 64]

Edge MLP templates:
  input = [h_parent || x_node || (x_node - x_parent)] = 64 + 3 + 3 = 70
  palm template   : [70 -> 32 -> 64]
  finger template : [70 -> 32 -> 64]

Readout head:
  [wrist(0), tip(4), tip(8), tip(12), tip(16), tip(20)] concat = 6 * 64 = 384
  head = [384 -> 128 -> G]
```

## 트리 정의

### Parent map

```text
1:0, 5:1, 9:5, 13:9, 17:13,
2:1, 3:2, 4:3,
6:5, 7:6, 8:7,
10:9, 11:10, 12:11,
14:13, 15:14, 16:15,
18:17, 19:18, 20:19
```

### Traversal order

```text
[0, 1, 5, 9, 13, 17, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]
```

### Palm nodes

```text
{1, 5, 9, 13, 17}
```

### Readout nodes

- wrist:
  - `0`
- fingertips:
  - `4, 8, 12, 16, 20`

위 fingertip/wrist 인덱스는 MediaPipe 공식 `HandLandmark` enum과 일치한다.

## forward 규칙

1. `x[:, 0, :]`에서 wrist root feature를 만든다.
2. `order[1:]`를 순회하면서 각 node `v`에 대해:
   - parent `p = parents[v]`
   - `feat = [h_p || x_v || (x_v - x_p)]`
   - palm node면 `edge_mlp_palm`
   - 그 외면 `edge_mlp_finger`
3. `h_0, h_4, h_8, h_12, h_16, h_20`를 concat해서 classification head에 넣는다.

## LapPE 옵션

- 모델 API에는 `use_lappe`, `lappe_dim`, `forward(x, lappe=None)`가 포함된다.
- 기본값은 `use_lappe=False`, `lappe_dim=0`이다.
- 이번 파이프라인에서는 dataset이나 preprocessing에서 LapPE를 생성하지 않는다.
- `use_lappe=True`일 때만:
  - root 입력은 `3 + lappe_dim`
  - edge 입력은 `64 + 3 + 3 + lappe_dim`
- `lappe`는 고정 positional encoding `(21, k)`만 받으며 batch로 broadcast 한다.

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

- static single-hand gesture 분류가 목표일 때
- 해부학적 routing prior를 강하게 넣고 싶을 때
- 구현 단순성과 디버깅 용이성이 중요한 경우
- palm에서 finger tip으로 이어지는 정보 흐름을 명시적으로 모델링하고 싶은 경우

### 조건부 적합

- local parent-child routing은 맞지만, 손가락 사이 sibling interaction이 조금 필요한 경우
- 이 경우 baseline으로는 유효하지만, finger 간 직접 상호작용을 별도 블록으로 보강할 여지가 있다.

### 부적합 또는 baseline 전용

- temporal cue가 중요한 dynamic gesture
- 양손 상호작용이 중요한 경우
- 비인접 joint correlation이 핵심일 때
- sample-specific topology refinement가 필요한 경우

### 왜 이렇게 판단하는가

- MediaPipe 공식 문서는 hand landmark를 21개로 고정하고, handedness와 world landmarks도 제공한다. 따라서 손 구조를 root와 finger chain 중심으로 다루는 전제 자체는 자연스럽다.
- DGNN 논문은 skeleton을 directed acyclic graph로 보고, 인접 노드와 edge를 따라 정보를 업데이트하는 설계가 유효하다고 설명한다.
- Pose-REN은 hand joints topology를 따라 tree-structured fully connections를 사용해 계층적으로 feature를 결합한다.
- HMTNet은 hand morphological topology를 따라 tree-like network structure를 사용해 고차 의존성을 반영한다고 설명한다.
- Hierarchical RNN 계열은 skeleton action recognition에서 body hierarchy를 따라 구조적 분해가 유용하다는 관점을 보여준다.
- 반면 DGNN과 후속 graph 논문들은 고정 구조만으로는 task-optimal topology가 아닐 수 있고, 비인접 dependency나 temporal cue가 중요해질 수 있다고 지적한다.
- MLPHand는 hand modeling에서도 효율적인 MLP 설계가 경쟁력이 있음을 보여주지만, 그 구조는 mesh reconstruction 목적이다. 여기서는 그 "효율적인 hand-centric MLP"라는 방향성만 차용한다.

정리하면, `hierarchical_tree_mlp`는 static single-hand gesture에 맞춘 명시적 routing baseline으로는 타당하지만, 더 풍부한 cross-finger interaction이나 temporal modeling이 필요한 문제에서는 상한이 분명하다. 이 평가는 아래 1차 출처를 바탕으로 한 명시적 추론이다.

## 구현 범위

- 현재 폴더 내부에만 파이프라인을 추가한다.
- `run_pipeline.py` 등록은 하지 않는다.
- 웹 검증 로직은 runtime code가 아니라, 본 README 안의 정적 판단 규칙으로 정리한다.

## 1차 출처 링크

- MediaPipe Hand Landmarker guide:
  - https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
- MediaPipe HandLandmark enum:
  - https://ai.google.dev/edge/api/mediapipe/java/com/google/mediapipe/tasks/vision/handlandmarker/HandLandmark
- Directed Graph Neural Networks, directed acyclic graph skeleton modeling:
  - https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_Skeleton-Based_Action_Recognition_With_Directed_Graph_Neural_Networks_CVPR_2019_paper.pdf
- Hierarchical RNN for skeleton action recognition:
  - https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Du_Hierarchical_Recurrent_Neural_2015_CVPR_paper.html
- Pose-REN, tree-structured fully connections for hand-joint topology:
  - https://arxiv.org/abs/1708.03416
- HMTNet, tree-like network based on hand morphological topology:
  - https://arxiv.org/abs/1911.04930
- MLPHand, hand modeling with efficient MLP design:
  - https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09503.pdf
