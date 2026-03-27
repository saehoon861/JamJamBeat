# README.md - Edge-STGU MLP pipeline notes and data contract

# `edge_stgu_mlp`

`edge_stgu_mlp` is a hand-landmark classification pipeline that keeps the input as raw 63D landmarks and replaces heavy graph aggregation with an edge-wise gated message-passing block.

This design follows the STGU motivation from SiT-MLP:

- Paper: https://arxiv.org/abs/2308.16018
- Official code repository: https://github.com/BUPTSJZhang/SiT-MLP

The paper and official implementation report a full skeleton sequence model at about `0.6M` parameters. This hand-only single-frame variant intentionally stays smaller, but it is not squeezed into a toy-sized regime. The implementation here uses `d_model=192`, `gate_hidden=96`, and `3` Edge-STGU blocks, which places it around the low-`0.23M` parameter range for small class counts.

## 목적

- GCN-style spatial reasoning의 장점을 일부 유지하면서도 복잡한 adaptive adjacency 설계를 피한다.
- 손 관절 topology를 edge-wise gate로 직접 반영한다.
- raw 63D landmark를 바로 입력받는 frame pipeline으로 사용한다.

## 입력 / 출력

- accepted input tensor:
  - `(B, 63)`
  - `(B, 21, 3)`
- internal representation:
  - raw landmark를 `(21, 3)`으로 reshape
  - shared linear projection `3 -> 192`
- output:
  - `(B, num_classes)` logits

## 아키텍처

1. Raw landmark input `(21, 3)`
2. Shared joint projection `Linear(3 -> 192)`
3. Learnable joint embedding `Embedding(21, 192)`
4. `EdgeSTGUBlock x 3`
5. `LayerNorm(192)`
6. Mean pool over 21 joints
7. `Linear(192 -> num_classes)`

Each `EdgeSTGUBlock` uses:

- `LayerNorm(192)`
- source value projection `Linear(192 -> 192)`
- gate MLP `Linear(384 -> 96) -> GELU -> Linear(96 -> 1) -> sigmoid`
- gated message `m_ij = g_ij * v_i`
- target aggregation by sum
- residual update `H = H + agg`

## Edge 구성

The block uses a directed hand graph built from:

- all pairs in `_shared.HAND_CONNECTIONS`
- reverse direction for every hand connection
- self-loop for all 21 joints

So the final directed edge count is:

- `2 * len(HAND_CONNECTIONS) + 21`

With the current shared hand topology, that is `63` directed edges.

## 데이터 계약

This pipeline expects the same frame-level CSV contract used by other `model_pipelines` builders.

Required columns:

- `source_file`
- `frame_idx`
- `timestamp`
- `gesture`
- `x0..z20`

The dataset builder reads only raw landmark columns:

- `x0, y0, z0, ..., x20, y20, z20`

## 전처리 정책

This pipeline does not apply any extra preprocessing inside the dataset builder.

- raw 63D landmarks are read as-is
- no root-relative normalization
- no scale normalization
- no handedness reflection

The only transformation is:

- reshape from `(63,)` to `(21, 3)`

## 권장 사용 상황

- hand topology를 완전히 무시하고 싶지는 않지만, 큰 GCN 계열을 쓰기엔 과하다고 느낄 때
- single-frame raw landmark baseline보다 topology-aware interaction을 추가하고 싶을 때
- sequence model로 가기 전에 spatial-only block을 먼저 비교하고 싶을 때

## 현재 scope 제한

- this folder only
- no `run_pipeline.py` registration
- no checkpoint loader wiring
- no extra docs outside `model/model_pipelines/edge_stgu_mlp/`

So the module is importable and self-contained, but it is not yet selectable from the shared top-level runner.
