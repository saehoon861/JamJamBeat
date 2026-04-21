# 분류기 출력 포맷 & 평가 시각화 가이드

> evaluation_guide.md 의 평가를 실제로 돌리기 위한 사전 조건 정리

---

## 1. MLP + Softmax 출력 구조

MLP + Softmax는 추론 시 자동으로 아래 값을 만들어냅니다.

```
input  : [x0,y0,z0, x1,y1,z1, ... x20,y20,z20]  ← 63차원 랜드마크

hidden : ReLU(W1 @ input + b1)
logits : W2 @ hidden + b2                          ← 7개 raw score
probs  : softmax(logits)                           ← [p0, p1, p2, p3, p4, p5, p6] 합=1.0

pred_class = argmax(probs)   ← 가장 높은 확률의 클래스 번호
p_max      = max(probs)      ← 그 확률값
```

**softmax 출력 자체가 이미 p0~p6** 입니다.
분류기 내부는 건드리지 않고, 추론 코드에 래퍼를 씌워 이 값들을 CSV로 저장하면 됩니다.

---

## 2. 추론 래퍼 코드 (predict_with_log)

실시간 환경의 지연 시간(Latency) 병목을 분석하기 위해, 전체 지연뿐만 아니라 **파이프라인 단계별 소요 시간**을 분리하여 측정하고 로깅합니다.

```python
# src/training/predict.py

import time
import numpy as np
import pandas as pd
import torch

# 예시용 더미 함수 (실제 환경에서는 모듈 호출로 대체)
def mock_mediapipe_process(frame): time.sleep(0.020) # 약 20ms
def mock_feature_extraction(landmarks): time.sleep(0.002) # 약 2ms
def mock_postprocess(probs): time.sleep(0.001) # 약 1ms

def predict_with_log(model, landmark_df: pd.DataFrame) -> pd.DataFrame:
    """
    학습된 모델을 사용하여 데이터를 평가하고, 병목 분석을 위해 
    파이프라인 각 단계(Stage)별 지연 시간 로깅을 포함하여 CSV로 저장하는 래퍼 함수입니다.
    
    landmark_df : frame_idx + 63개 랜드마크 컬럼 (landmark_data CSV)
    반환        : 평가용 preds CSV (pred_class, p_max, p0~p6, t_mp_ms, t_feat_ms, t_mlp_ms, t_post_ms, latency_total_ms)
    """
    records = []

    for _, row in landmark_df.iterrows():
        # 전체 파이프라인 시작 타임스탬프
        t_pipeline_start = time.perf_counter()

        # -----------------------------------------------------
        # [Stage 1] MediaPipe 랜드마크 추출 (더미 처리)
        # -----------------------------------------------------
        t_mp_start = time.perf_counter()
        mock_mediapipe_process(frame=None) # 실제로는 웹캠 프레임 입력
        t_mp_ms = (time.perf_counter() - t_mp_start) * 1000

        # -----------------------------------------------------
        # [Stage 2] 피처 추출 및 정규화
        # -----------------------------------------------------
        t_feat_start = time.perf_counter()
        x = row[[f"{c}{i}" for i in range(21) for c in ["x","y","z"]]].to_numpy(dtype=float)
        mock_feature_extraction(x) # 스케일링, 중심 정렬 등 연산
        t_feat_ms = (time.perf_counter() - t_feat_start) * 1000

        # -----------------------------------------------------
        # [Stage 3] 모델 추론 (MLP + Softmax)
        # -----------------------------------------------------
        t_mlp_start = time.perf_counter()
        with torch.no_grad(): # 평가 단계이므로 그래디언트 연산 비활성화
            logits = model(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
            probs  = torch.softmax(logits, dim=1).squeeze().numpy()  # [p0..p6]
        t_mlp_ms = (time.perf_counter() - t_mlp_start) * 1000

        # -----------------------------------------------------
        # [Stage 4] 후처리 및 최종 판정 (Voting / Debounce)
        # -----------------------------------------------------
        t_post_start = time.perf_counter()
        mock_postprocess(probs)
        t_post_ms = (time.perf_counter() - t_post_start) * 1000

        # End-to-End 전체 지연 시간 계산
        latency_total_ms = (time.perf_counter() - t_pipeline_start) * 1000

        # 결과 레코드 생성
        record = {"frame_idx": int(row["frame_idx"])}
        record["pred_class"] = int(np.argmax(probs))         # 최종 예측 클래스
        record["p_max"]      = round(float(probs.max()), 4)  # 최대 확률값 (신뢰도)
        
        # 클래스별 세부 확률 기록
        for i, p in enumerate(probs):
            record[f"p{i}"]  = round(float(p), 4)
            
        # 단계별 지연 시간(Stage-wise Latency) 기록
        record["t_mp_ms"] = round(t_mp_ms, 3)
        record["t_feat_ms"] = round(t_feat_ms, 3)
        record["t_mlp_ms"] = round(t_mlp_ms, 3)
        record["t_post_ms"] = round(t_post_ms, 3)
        record["latency_total_ms"] = round(latency_total_ms, 3)

        records.append(record)

    return pd.DataFrame(records)


# 실행 예시
if __name__ == "__main__":
    landmark_df = pd.read_csv("data/landmark_data/3_fast_right_man3.csv")
    preds_df    = predict_with_log(model, landmark_df)
    preds_df.to_csv("runs/preds/3_fast_right_man3_pred.csv", index=False)
```

---

## 3. preds CSV 출력 포맷

```
frame_idx, pred_class, p_max,  p0,    p1,    p2,    p3,    p4,    p5,    p6,    latency_ms
0,          2,         0.934,  0.012, 0.031, 0.934, 0.008, 0.007, 0.005, 0.003, 18.2
1,          0,         0.612,  0.612, 0.094, 0.103, 0.071, 0.058, 0.039, 0.023, 17.8
2,          2,         0.971,  0.003, 0.008, 0.971, 0.005, 0.006, 0.004, 0.003, 19.1
```

| 컬럼 | 설명 | 필요한 평가 |
|------|------|------------|
| `pred_class` | argmax 예측 클래스 | Confusion Matrix, F1 |
| `p_max` | 최대 확률값 | FP/min (threshold τ 적용) |
| `p0` ~ `p6` | 클래스별 softmax 확률 | PR curve (선택) |
| `latency_ms` | 추론 소요 시간 | Latency CDF |

> **최소 요건**: `pred_class + p_max` 만 있어도 Confusion Matrix, F1, FP/min 전부 가능.
> `latency_ms` 는 Latency CDF를 볼 때만 필요.

---

## 4. 평가 흐름 전체

```
landmark_data/*.csv          labeled_data/*.csv
(63컬럼 랜드마크)              (frame_idx, gesture)
        │                            │
        ▼                            │
  predict_with_log(model)            │
        │                            │
        ▼                            ▼
  runs/preds/*_pred.csv ─── frame_idx join ──→ 평가 스크립트
```

---

## 5. 시각화 방법

### 5.1 Confusion Matrix — 히트맵

**무엇을 보나**: 실제 클래스가 어느 클래스로 오분류되는지 패턴

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

labels     = pd.read_csv("labeled_data/3_fast_right_man1.csv")
preds      = pd.read_csv("runs/preds/3_fast_right_man3_pred.csv")
df         = labels.merge(preds, on="frame_idx")

CLASS_NAMES = ["neutral","fist","open_palm","V","pinky","animal","k-heart"]
cm = confusion_matrix(df["gesture"], df["pred_class"],
                      labels=list(range(7)), normalize="true")

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
fig, ax = plt.subplots(figsize=(9, 7))
disp.plot(ax=ax, values_format=".2f", cmap="Blues")
ax.set_title("Confusion Matrix (행 정규화 — 각 실제 클래스별 비율)")
plt.tight_layout()
plt.savefig("runs/reports/confusion_matrix.png", dpi=150)
```

```
읽는 법:
  대각선 높음      → 잘 맞힘
  [0][1..6] 높음  → neutral을 제스처로 오탐 (가장 위험)
  [1..6][0] 높음  → 제스처를 neutral로 미탐 (인식 실패)
```

---

### 5.2 클래스별 Precision / Recall / F1 — 막대 그래프

**무엇을 보나**: 특정 클래스가 약한지, precision/recall 중 어느 쪽이 문제인지

```python
from sklearn.metrics import classification_report

report = classification_report(
    df["gesture"], df["pred_class"],
    labels=list(range(7)), target_names=CLASS_NAMES,
    output_dict=True, digits=4
)

df_report = pd.DataFrame(report).T.loc[CLASS_NAMES, ["precision","recall","f1-score"]]

ax = df_report.plot(kind="bar", figsize=(11, 4), ylim=(0, 1.05))
ax.axhline(0.9, linestyle="--", color="gray", alpha=0.6, label="0.9 기준선")
ax.set_title("Per-class Precision / Recall / F1")
ax.set_xlabel("Gesture Class")
ax.legend()
plt.tight_layout()
plt.savefig("runs/reports/per_class_f1.png", dpi=150)
```

---

### 5.3 FP/min τ 스윕 — 꺾은선 그래프

**무엇을 보나**: threshold를 높일수록 오탐이 얼마나 줄고, recall은 얼마나 희생되는지

```python
import matplotlib.pyplot as plt

# τ를 바꾸면서 FP/min과 recall 변화 기록
tau_list, fp_list, recall_list = [], [], []

for tau in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    # neutral 구간에서 FP/min 계산 (evaluate_triggers.py 활용)
    fp  = calc_fp_per_min("labels.csv", "preds.csv", tau=tau, vote_n=7, debounce_k=3)
    rec = (df[df["gesture"] != 0]
           .assign(pred_ok=lambda d: (d["p_max"] >= tau) & (d["pred_class"] == d["gesture"]))
           ["pred_ok"].mean())
    tau_list.append(tau); fp_list.append(fp); recall_list.append(rec)

fig, ax1 = plt.subplots(figsize=(8, 4))
ax2 = ax1.twinx()

ax1.plot(tau_list, fp_list,    "o-", color="red",   label="FP/min")
ax2.plot(tau_list, recall_list,"s--", color="blue",  label="Recall")

ax1.set_xlabel("Threshold τ")
ax1.set_ylabel("FP/min", color="red")
ax2.set_ylabel("Recall",  color="blue")
ax1.set_title("Threshold Sweep — FP/min vs Recall")
fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
plt.tight_layout()
plt.savefig("runs/reports/threshold_sweep.png", dpi=150)
```

---

### 5.4 Latency CDF + Stage Box Plot

**무엇을 보나**: 지연 스파이크 존재 여부, 병목 스테이지 파악

```python
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- 좌: Stage-wise Box Plot ---
stage_cols = {
    "MediaPipe": "t_mp_ms", "Feature": "t_feat_ms",
    "MLP":       "t_mlp_ms", "PostProc": "t_post_ms",
}
data   = [preds[c].dropna().to_numpy() for c in stage_cols.values() if c in preds]
labels = [k for k, v in stage_cols.items() if v in preds]

axes[0].boxplot(data, labels=labels)
axes[0].axhline(33, linestyle="--", color="red", label="33ms (30fps)")
axes[0].set_ylabel("ms"); axes[0].set_title("Stage-wise Latency")
axes[0].legend()

# --- 우: Total Latency CDF ---
x  = np.sort(preds["latency_ms"].dropna().to_numpy())
y  = np.arange(1, len(x)+1) / len(x)
p50, p95, p99 = np.percentile(x, [50, 95, 99])

axes[1].plot(x, y, linewidth=2)
for p, lbl in [(p50,"p50"),(p95,"p95"),(p99,"p99")]:
    axes[1].axvline(p, linestyle="--", label=f"{lbl}={p:.1f}ms")
axes[1].set_xlabel("Total Latency (ms)"); axes[1].set_ylabel("CDF")
axes[1].set_title("End-to-End Latency CDF"); axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("runs/reports/latency_cdf.png", dpi=150)
print(f"p50={p50:.1f}ms | p95={p95:.1f}ms | p99={p99:.1f}ms")
```

---

### 5.5 학습 곡선 (Learning Curve)

**무엇을 보나**: Epoch에 따른 Loss와 정확도(F1) 변화를 통해 모델의 과적합(Overfitting) 여부 진단

```python
# df_history는 학습 과정에서 저장된 Epoch별 지표 (예: CSV 또는 텐서보드 로그 변환)
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss Curve
axes[0].plot(df_history['epoch'], df_history['train_loss'], label='Train Loss')
axes[0].plot(df_history['epoch'], df_history['val_loss'], label='Val Loss')
axes[0].set_title('Loss Curve')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].legend()

# Metric Curve
axes[1].plot(df_history['epoch'], df_history['train_f1'], label='Train F1')
axes[1].plot(df_history['epoch'], df_history['val_f1'], label='Val F1')
axes[1].set_title('F1 Score Curve')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('F1 Score')
axes[1].legend()

plt.tight_layout()
plt.savefig("runs/reports/learning_curve.png", dpi=150)
```

---

### 5.6 시간축 확률 변동 (Temporal Jitter)

**무엇을 보나**: 특정 동작 전환 구간이나 유지 구간에서 클래스별 확률($p_0 \sim p_6$)이 요동치는지(Flickering) 시각적으로 디버깅

```python
import matplotlib.pyplot as plt

# 테스트할 특정 클립 1개만 추출 (예: 3_fast_right_man3)
clip_df = df[df['session_id'] == '3_fast_right_man3'].sort_values('frame_idx')

fig, ax = plt.subplots(figsize=(14, 5))
for i in range(7):
    ax.plot(clip_df['frame_idx'], clip_df[f'p{i}'], label=f'{CLASS_NAMES[i]} (p{i})', alpha=0.8)

# 동작이 전환되는 시점을 세로선으로 표시
gt_changes = clip_df[clip_df['gesture'].diff() != 0]
for _, row in gt_changes.iterrows():
    ax.axvline(x=row['frame_idx'], color='gray', linestyle='--', alpha=0.5)

ax.set_title("Temporal Probability Fluctuation (3_fast_right_man3)")
ax.set_xlabel("Frame Index")
ax.set_ylabel("Softmax Probability")
ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
plt.tight_layout()
plt.savefig("runs/reports/temporal_jitter.png", dpi=150)
```

---

### 5.7 모델 신뢰도 교정 (Reliability Diagram & ECE)

**무엇을 보나**: 모델이 내놓는 확률값(p_max)이 실제 정답률과 얼마나 일치하는지 (과잉 확신 방지)

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# 예시: 2번(open_palm) 클래스에 대한 이진 교정 곡선 확인
class_idx = 2
y_true_binary = (df['gesture'] == class_idx).astype(int)
y_prob = df[f'p{class_idx}']

prob_true, prob_pred = calibration_curve(y_true_binary, y_prob, n_bins=10)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(prob_pred, prob_true, marker='o', label=f'{CLASS_NAMES[class_idx]}')
ax.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly Calibrated')

ax.set_title("Reliability Diagram (Calibration Curve)")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.legend()
plt.tight_layout()
plt.savefig("runs/reports/reliability_diagram.png", dpi=150)
```

---

## 6. 시각화 요약

| 시각화 | 입력 | 그래프 형태 | 확인 포인트 |
|--------|------|------------|------------|
| Confusion Matrix | pred_class | **2D 히트맵** | 어떤 클래스끼리 헷갈리는가 |
| Per-class F1 | pred_class | **클래스별 막대 그래프** | precision/recall 중 어느 쪽이 약한가 |
| τ 스윕 | pred_class + p_max | **이중 축 꺾은선** | threshold 조정 시 FP/min vs Recall 변화 |
| Latency CDF | latency_ms | **CDF 곡선 + Box Plot** | 스파이크 유무, 병목 스테이지 |
| Learning Curve | epoch, loss/f1 | **에포크별 꺾은선** | 학습 과정에서의 과적합 여부 |
| Temporal Jitter | p0~p6, frame_idx | **시간축 꺾은선** | 프레임 간 확률 요동(Flickering) 패턴 |
| Reliability Diag. | p_max, gesture | **교정 곡선** | 예측 확률의 실제 신뢰성(과잉 확신) 검증 |
