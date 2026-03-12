# JamJamBeat — Learning Rate & Training Curves 비교 분석

[TOC]

## 실험 개요

- **입력**: raw MediaPipe landmark 63채널 (`x0~x20, y0~y20, z0~z20`)
- **모델 수**: 13개
- **옵티마이저**: AdamW (weight_decay=0.0001)
- **초기 LR**: 0.001 (cosine annealing with warmup)
- **최대 epochs**: 20, early stopping patience=6
- **평가 기준**: macro F1 ≥ 0.80

---

## Learning Rate 스케줄 개요

모든 모델은 동일한 cosine annealing LR 스케줄러를 사용하지만, **early stopping으로 조기 종료된 epoch 수에 따라 최종 LR이 달라집니다**.

| 모델 | 학습 epochs | 최종 LR | LR 감소율 |
|------|------------|---------|----------|
| mlp_baseline_seq8 | 16 | 0.0000950 | 90.4% |
| transformer_embedding | 10 | 0.0005000 | 49.7% |
| mlp_baseline | 9 | 0.0005780 | 41.9% |
| mlp_baseline_full | 9 | 0.0005780 | 41.9% |
| mlp_sequence_joint | 9 | 0.0005780 | 41.9% |
| mlp_sequence_delta | 9 | 0.0005780 | 41.9% |
| mlp_embedding | 8 | 0.0006550 | 34.1% |
| mlp_temporal_pooling | 8 | 0.0006550 | 34.1% |
| two_stream_mlp | 8 | 0.0006550 | 34.1% |
| shufflenetv2_x0_5 | 8 | 0.0006550 | 34.1% |
| cnn1d_tcn | 7 | 0.0007270 | 26.9% |
| efficientnet_b0 | 7 | 0.0007270 | 26.9% |
| mobilenetv3_small | 7 | 0.0007270 | 26.9% |

> **인사이트**: `mlp_baseline_seq8`은 가장 긴 16 epoch 학습으로 LR이 90% 이상 감소. 반면 CNN/Image 계열(7 epoch)은 early stopping이 빨리 걸려 LR 감소 폭이 작음.

---

## 모델별 LR 그래프 (wandb Workspace에서 확인)

각 모델 run의 **`lr` vs `epoch`** 그래프는 wandb Project Workspace의 아래 경로에서 확인할 수 있습니다:

> **[JamJamBeat Workspace →](https://wandb.ai/sgim49697-hancom/JamJamBeat)**
> - 좌측 패널 → Runs → 모델명 클릭 → Charts 탭 → `lr` 선택

### 전체 LR 통합 비교

Workspace에서 전체 run을 선택 후 `lr` 메트릭을 선택하면 13개 모델의 LR 곡선을 한 화면에서 비교할 수 있습니다.

---

## Training Loss & Val Loss 비교

### macro F1 기준 성능 순위

| 순위 | 모델 | macro F1 | accuracy | best val_loss | epochs |
|------|------|---------|---------|--------------|--------|
| 1 | transformer_embedding | **0.4548** | 0.4343 | 0.0800 | 10 |
| 2 | two_stream_mlp | **0.4548** | 0.3704 | 0.0446 | 8 |
| 3 | mlp_embedding | 0.4387 | 0.3813 | 0.0864 | 8 |
| 4 | mlp_temporal_pooling | 0.3880 | 0.3485 | 0.0798 | 8 |
| 5 | mlp_baseline | 0.3284 | 0.2689 | 0.0796 | 9 |
| 6 | mlp_baseline_full | 0.3284 | 0.2689 | 0.0796 | 9 |
| 7 | cnn1d_tcn | 0.3025 | 0.2354 | 0.1007 | 7 |
| 8 | mlp_sequence_joint | 0.3012 | 0.2450 | 0.1069 | 9 |
| 9 | mlp_baseline_seq8 | 0.2774 | 0.2289 | 0.1074 | 16 |
| 10 | mlp_sequence_delta | 0.2624 | 0.2144 | 0.1193 | 9 |
| 11 | shufflenetv2_x0_5 | 0.2289 | 0.1833 | 0.1096 | 8 |
| 12 | efficientnet_b0 | 0.2068 | 0.1637 | 0.1318 | 7 |
| 13 | mobilenetv3_small | 0.1926 | 0.1512 | 0.1181 | 7 |

> ⚠️ **전체 성능 미달**: 모든 모델이 PoC 기준 macro F1 ≥ 0.80 미달. raw 63채널 기준 첫 실험 결과로, 전처리(정규화+bone+angle 156ch) 적용 후 재실험 필요.

---

## LR vs 성능 분석

### LR 감소량과 성능 관계

| LR 최종값 | 해당 모델들 | 평균 macro F1 |
|----------|-----------|--------------|
| 0.000095 (90% 감소) | mlp_baseline_seq8 | 0.277 |
| 0.000500 (50% 감소) | transformer_embedding | 0.455 |
| 0.000578 (42% 감소) | mlp_baseline, mlp_baseline_full, mlp_sequence_joint, mlp_sequence_delta | 0.314 |
| 0.000655 (34% 감소) | mlp_embedding, mlp_temporal_pooling, two_stream_mlp, shufflenetv2_x0_5 | 0.382 |
| 0.000727 (27% 감소) | cnn1d_tcn, efficientnet_b0, mobilenetv3_small | 0.234 |

> **인사이트**: LR 감소율이 반드시 성능과 비례하지 않음. `mlp_baseline_seq8`은 가장 많이 훈련됐음에도 성능이 낮아, 모델 복잡도 대비 데이터 부족 가능성 시사. Transformer와 two_stream_mlp가 moderate LR에서 최고 성능.

---

## 결론 및 다음 단계

### 관찰
1. **raw 63ch 기준**: 모든 모델이 PoC 기준 미달 → 전처리 피처 추가 실험 필요
2. **LR 스케줄**: cosine annealing이 전반적으로 안정적으로 작동
3. **early stopping**: patience=6 설정으로 대부분 7~10 epoch에서 조기 종료
4. **최고 성능**: transformer_embedding / two_stream_mlp (macro F1 ≈ 0.455)

### 권장 다음 단계
- [ ] 전처리 CSV (156ch: joint+bone+angle) 기준으로 재실험
- [ ] LR warmup 구간 조정 또는 더 긴 max_epochs 설정
- [ ] transformer_embedding / two_stream_mlp 하이퍼파라미터 튜닝
- [ ] 데이터 증강 적용 (flip, jitter 등)
