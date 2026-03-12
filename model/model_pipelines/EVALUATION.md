# JamJamBeat 모델 평가 가이드

> 관련 코드: `model/model_evaluation/evaluation_runtime.py`
> 평가 실행: `run_model_pipeline.py` → `evaluate_predictions()` 자동 호출

---

## 1. 평가가 언제 실행되나

`run_model_pipeline.py` 또는 `run_all.py`를 실행하면 학습 완료 후 **자동으로** 평가가 돌아간다.
별도 스크립트를 따로 실행할 필요 없다.

```
학습 완료
  → preds_test.csv 생성 (모델 예측 결과)
    → evaluate_predictions() 호출
      → evaluation/ 폴더에 결과 저장
```

---

## 2. 출력 파일 구조

`run_all.py` 배치 실행 시 아래 경로에 파일이 생성된다.

```
model/model_evaluation/pipelines/
├── latest_suite.json
└── {yyyymmdd_HHMMSS}__{dataset_tag}/
    ├── comparison_suite.json      ← 이번 배치 입력 CSV / 모델 목록 메타데이터
    ├── comparison_results.csv     ← 이번 배치 전체 비교표
    └── {model_id}/
        └── {yyyymmdd_HHMMSS}/
            ├── preds_test.csv          ← 프레임별 예측 원본
            ├── model.pt                ← 학습된 모델 가중치
            ├── train_history.csv       ← epoch별 loss/acc
            ├── run_summary.json        ← 전체 메트릭 요약
            └── evaluation/
                ├── dataset_info.json      ← 입력 CSV / split 정보
                ├── confusion_matrix.csv   ← 혼동 행렬 (수치)
                ├── confusion_matrix.png   ← 혼동 행렬 (시각화)
                ├── per_class_report.csv   ← 클래스별 P/R/F1
                ├── latency_cdf.png        ← 추론 지연 분포
                └── metrics_summary.json   ← 핵심 지표 JSON
```

전체 실험 비교 결과:
```
model/model_evaluation/pipelines/{suite_name}/comparison_results.csv
```

`run_pipeline.py` 단독 실행은 기존처럼 `model/model_evaluation/pipelines/{model_id}/{timestamp}/` 구조를 유지한다.

---

## 3. 평가 지표 상세

### 3.1 Macro F1 (주 비교 지표)

```
macro_f1 = 클래스별 F1의 단순 평균
```

- **Class 0 (neutral)이 데이터의 대부분**이라 accuracy는 "모두 0 예측"만 해도 높게 나온다.
- Macro F1은 소수 클래스(Fist, V, K-Heart 등)를 동등하게 반영하므로 주 비교 지표로 사용한다.

`per_class_report.csv` 예시:
```
class       precision  recall   f1     support
neutral     0.91       0.88     0.895  1200
fist        0.84       0.79     0.814  180
open_palm   0.78       0.82     0.800  160
V           0.80       0.75     0.774  150
pinky       0.72       0.68     0.699  120
animal      0.76       0.71     0.734  110
k-heart     0.69       0.65     0.670   95
macro avg   0.79       0.75     0.769
```

### 3.2 Confusion Matrix

`confusion_matrix.png` — **행 = 실제 클래스, 열 = 예측 클래스**
각 셀은 해당 행 기준 비율로 정규화된다 (행 합 = 1.0).

읽는 법:
- **대각선**: 맞게 분류된 비율 → 높을수록 좋다
- **같은 행에서 대각선 외 셀이 크면**: 해당 클래스를 다른 클래스로 오분류
- **같은 열에서 대각선 외 셀이 크면**: 다른 클래스를 해당 클래스로 잘못 예측

예시 해석:
```
             neutral  fist  open_palm  ...
neutral        0.88   0.05     0.04   ← neutral을 fist로 5% 오분류
fist           0.09   0.79     0.06   ← fist를 neutral로 9% 누락
open_palm      0.03   0.05     0.82   ← 비교적 안정적
```

### 3.3 Class 0 FP/FN Rate

Class 0 = `neutral` (아무 제스처도 아님)

| 지표 | 의미 | 실서비스 영향 |
|------|------|-------------|
| **FPR** (False Positive Rate) | 실제 제스처를 neutral로 놓치는 비율 | 제스처 인식 누락 |
| **FNR** (False Negative Rate) | neutral인데 제스처로 오판하는 비율 | 오발동 |

`metrics_summary.json` 내 위치:
```json
"class0_metrics": {
    "false_positive_rate": 0.12,   ← 제스처 12% 누락
    "false_negative_rate": 0.07,   ← neutral 7% 오발동
    "false_positive_count": 48,
    "false_negative_count": 84
}
```

### 3.4 FP/min (False Positive per Minute)

```
후처리 파이프라인: threshold(τ) → voting(N프레임) → debounce(K회)
```

neutral 구간에서 잘못 발동된 횟수를 **분당 기준**으로 측정한다.

기본 설정:
```
tau        = 0.85   (신뢰도 임계값: p_max < 0.85이면 neutral로 처리)
vote_n     = 7      (최근 7프레임 다수결)
debounce_k = 3      (같은 제스처가 3회 연속 감지돼야 트리거)
```

`metrics_summary.json` 내 위치:
```json
"fp_per_min_metrics": {
    "fp_per_min": 0.8,       ← 분당 0.8회 오발동
    "fp_count": 3,
    "duration_min": 3.7,
    "trigger_count_total": 3
}
```

PoC 목표: **< 2 FP/min**

### 3.5 Latency CDF

`latency_cdf.png` — 프레임별 추론 시간의 누적 분포 그래프

읽는 법:
- **X축**: 지연 시간 (ms)
- **Y축**: 해당 시간 이하인 샘플 비율 (0~1)
- **p50**: 중간값 (전체의 50%가 이 시간 이하)
- **p95**: 상위 5% 지연 (이상치 기준)
- **빨간 수직선**: 목표 200ms 기준선

`metrics_summary.json` 내 위치:
```json
"latency": {
    "mean_ms": 0.31,
    "p50_ms":  0.28,
    "p95_ms":  0.55,
    "p99_ms":  0.89
}
```

> 현재 `t_mlp_ms`만 기록되므로 순수 모델 추론 시간 기준.
> MediaPipe 전처리 시간은 별도 측정 필요.

---

## 4. 전체 모델 비교 (`comparison_results.csv`)

`run_all.py` 실행 후 생성되는 비교 테이블. 컬럼 설명:

| 컬럼 | 설명 |
|------|------|
| `model_id` | 모델 이름 |
| `mode` | 입력 모드 (frame / two_stream / sequence / image) |
| `accuracy` | 전체 정확도 |
| `macro_f1` | **주 비교 지표** |
| `macro_precision` | 클래스별 precision 평균 |
| `macro_recall` | 클래스별 recall 평균 |
| `class0_fpr` | 제스처 누락률 |
| `class0_fnr` | neutral 오발동률 |
| `fp_per_min` | 분당 오발동 횟수 (후처리 포함) |
| `latency_p50_ms` | 추론 시간 중간값 |
| `latency_p95_ms` | 추론 시간 p95 |
| `best_val_loss` | validation loss 최솟값 |
| `epochs_ran` | 실제 학습 에폭 수 (early stop 반영) |

---

## 5. 평가 결과 해석 기준 (PoC)

```
macro_f1     ≥ 0.80   → 양호
macro_f1     ≥ 0.70   → 개선 필요
macro_f1     < 0.70   → 재설계 필요

class0_fnr   < 0.10   → neutral 오발동 수용 범위
fp_per_min   < 2.0    → 실서비스 수용 범위

latency_p95  < 200ms  → 온디바이스 배포 가능
```

---

## 6. 실행 방법 요약

```bash
# 전체 9개 모델 순차 실행
cd /home/user/projects/JamJamBeat
python model/model_pipelines/run_all.py

# 일부 모델만 실행
python model/model_pipelines/run_all.py \
    --models mlp_baseline mlp_embedding two_stream_mlp

# 에폭 수 조정
python model/model_pipelines/run_all.py --epochs 50

# 결과 확인
cat model/model_evaluation/pipelines/latest_suite.json
cat model/model_evaluation/pipelines/{suite_name}/comparison_results.csv
```

---

## 7. 자주 보는 패턴

**Macro F1은 높은데 특정 클래스 recall이 낮은 경우**
→ 해당 클래스 학습 데이터 부족. `data/labeled_data/` 추가 수집 필요.

**FNR이 높은 경우 (neutral을 제스처로 오판)**
→ `tau` 값 올리거나 `debounce_k` 높이기.

**두 모델의 macro_f1이 비슷한 경우**
→ `fp_per_min`과 `latency_p50_ms`로 실서비스 적합성 비교.

**이미지 모델(mobilenetv3_small 등)의 F1이 낮은 경우**
→ 스켈레톤 이미지 렌더링 해상도(`--image-size`) 조정 또는 랜드마크 기반 모델이 이 데이터에 더 적합함을 의미.
