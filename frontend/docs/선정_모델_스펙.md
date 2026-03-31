# 선정 모델 스펙 — mlp_embedding / baseline

## 아키텍처

```
입력: 63d (x0,y0,z0,...,x20,y20,z20) — raw MediaPipe, 정규화 없음

embed: Linear(63→128) → LayerNorm(128) → GELU → Dropout(0.1)
head:  Linear(128→128) → ReLU → Dropout(0.1) → Linear(128→64) → ReLU → Linear(64→7)

파라미터 수: 33,671
mode: frame (단일 프레임)
```

## 학습 설정

| 항목 | 값 |
|------|-----|
| dataset | baseline (변환 없음, raw 좌표) |
| optimizer | AdamW |
| lr | 1e-3 |
| weight_decay | 1e-4 |
| batch_size | 32 |
| epochs | 20 (early stopping patience=6) |
| best_val_loss | 0.01205 |
| seed | 42 |
| scheduler | CosineAnnealingLR |

## 손실함수 / 샘플링

| 항목 | 값 |
|------|-----|
| loss | FocalLoss |
| focal_gamma | 2.0 |
| label_smoothing | 비활성 (0.0) |
| sampler | WeightedRandomSampler ✅ |
| alpha (class weight) | 활성 ✅ |

> ⚠️ sampler + alpha 동시 활성 → neutral 이중 보정 구조 (class0_fnr 높은 원인)

## 데이터 분할

| split | rows | source 수 |
|-------|------|-----------|
| train | 32,522 | 40 |
| val | 7,188 | 9 |
| test | 1,185 | 21 |
| inference | 5,773 | 7 |

분할 방식: 영상(source_file) 단위 block split, 겹침 없음

## 평가 결과 (test split 기준)

| 지표 | 값 |
|------|-----|
| accuracy | 0.7747 |
| macro_f1 | 0.7948 |
| class0_fnr | 0.5929 |
| class0_fpr | 0.0884 |
| latency_p50 | 0.005ms |
| latency_p95 | 0.010ms |

### 클래스별 recall

| class | recall |
|-------|--------|
| 0 neutral | 0.407 |
| 1 fist | 0.803 |
| 2 open_palm | 0.925 |
| 3 V | 0.833 |
| 4 pinky | 0.901 |
| 5 animal | 0.972 |
| 6 k-heart | 0.888 |

## 후처리 (post-processing)

| 항목 | 값 |
|------|-----|
| tau | 0.85 (기본값) — p_max < tau 이면 neutral(0) 강제 |
| vote_n | 7 |
| debounce_k | 3 |

> tau 상향 시 neutral FNR 개선되나 gesture recall 저하 trade-off 존재
