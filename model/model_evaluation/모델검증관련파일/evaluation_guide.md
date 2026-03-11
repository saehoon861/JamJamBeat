# JamJamBeat 제스처 분류기 성능평가 가이드

> 대상 모델: MediaPipe 21 랜드마크(63차원) → MLP Softmax (7클래스)
> 데이터 포맷: `frame_idx, timestamp, gesture` CSV

---

## 평가 구조 개요

```
[프레임 단위 평가]   → 모델 자체 품질
  Confusion Matrix + F1-score

[이벤트 단위 평가]  → 사운드 UX 품질  ← 이 프로젝트의 최우선 지표
  FP/min (오탐 트리거 빈도)

[지연 단위 평가]    → 실시간성 검증
  Trigger Delay + Jitter CDF
```

> **핵심 원칙**: 오디오 트리거 앱에서 **FP(오탐)가 UX를 망가뜨리는 주범**이다.
> accuracy가 높아도 FP/min이 높으면 실사용 불가. 세 레이어를 모두 측정해야 한다.

---

## 0. 평가용 테스트셋 구성 원칙 (먼저 확인)

본 가이드는 성능 평가(Evaluation) 방법론에 집중합니다. 측정된 평가 지표가 실사용 환경을 대변하기 위해서는 평가에 사용되는 데이터(Test Set)가 다음 원칙을 반드시 충족해야 합니다.

* **프레임 단위 섞기(Shuffle) 절대 금지**: 연속된 영상 프레임 간 상관관계가 매우 높으므로, 프레임을 무작위로 섞어서 평가하면 성능이 심각하게 과대평가됩니다.
* **독립된 환경/참가자 기반 평가**: 실제 앱 배포 시 처음 보는 사용자의 환경과 손을 인식해야 하므로, 평가 데이터는 **분석 대상 시스템이 사전에 경험하지 못한 독립된 참가자나 세션의 데이터**로만 구성해야 합니다.

**[데이터 확장 단계별 테스트 데이터 구성]**
* **초기 단계 (1인)**: 다른 세션이나 속도(예: fast)로 촬영된 영상을 테스트용으로 지정하여 최소한의 지표를 평가합니다.
* **팀원 4명 단계 (우선 목표)**: 평가 시점에는 반드시 특정 **1명의 데이터 전체를 철저히 분리**하여 평가 전용 테스트셋으로 지정하고 지표를 추출합니다. (평가 대상은 다른 팀원으로 교체하며 여러 번 평가 가능)
* **서비스 확장 단계 (20명+)**: 완전히 독립적인 별도의 평가 전용 참가자 그룹(예: 4~5명)을 사전에 구축해 두고 최종 성능을 검증합니다.

---

## 1. 기본 평가 — Confusion Matrix + F1-score

### 1.1 클래스 정의

| ID | 제스처 | 평가 비고 |
|----|--------|----------|
| 0 | Neutral / Hard Negative | Precision 최우선 (FP 방지) |
| 1 | fist | |
| 2 | open palm | |
| 3 | V | |
| 4 | pinky | |
| 5 | animal | |
| 6 | k-heart | |

### 1.2 구현 코드

```python
# evaluate_frames.py
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

CLASS_NAMES = ["neutral", "fist", "open_palm", "V", "pinky", "animal", "k-heart"]
LABELS = [0, 1, 2, 3, 4, 5, 6]

def evaluate_frames(labels_csv: str, preds_csv: str, out_dir: str = "runs/reports"):
    # 1. 정답 라벨과 예측 결과 CSV 파일 로드
    labels = pd.read_csv(labels_csv)   # 프레임 인덱스, 타임스탬프, 정답 제스처 ID
    preds  = pd.read_csv(preds_csv)    # 프레임 인덱스, 예측 클래스, 최대 확률(p_max) 등

    # 2. frame_idx를 기준으로 정답과 예측 데이터 병합 (동일 프레임 매칭)
    df = labels.merge(preds, on="frame_idx", how="inner")
    y_true = df["gesture"]
    y_pred = df["pred_class"]

    # 3. 텍스트 형태의 분류 리포트 출력 (Precision, Recall, F1-Score)
    # digits=4로 소수점 넷째 자리까지 표기하여 미세한 차이 분석
    print(classification_report(y_true, y_pred, labels=LABELS,
                                target_names=CLASS_NAMES, digits=4))

    # 4. Confusion Matrix (혼동 행렬) 생성
    # normalize="true" 옵션으로 각 실제 클래스(행 기준)의 재현율(Recall) 비율을 계산
    cm = confusion_matrix(y_true, y_pred, labels=LABELS, normalize="true")
    
    # 5. 생성된 혼동 행렬을 히트맵 형태로 시각화 및 저장
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(9, 7))
    disp.plot(ax=ax, values_format=".2f", cmap="Blues") # 비율을 소수점 둘째 자리까지 표기
    ax.set_title("Normalized Confusion Matrix (row = true class)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_matrix.png", dpi=150)
    plt.show()
```

### 1.3 읽는 법

```
confusion matrix [i][j] = 실제 i인데 j로 예측한 비율

대각선 높음 → 잘 맞힘
[0][1..6] 높음 → neutral을 제스처로 오탐 (가장 위험!)
[1..6][0] 높음 → 제스처를 neutral로 미탐 (인식 실패)
```

### 1.4 PoC 확인 기준

> PoC 단계 목표: "이 파이프라인이 동작하는가" 확인. 수치 기준보다 **혼동 패턴 파악**이 우선.

| 지표 | PoC 확인 포인트 | 비고 |
|------|----------------|------|
| 클래스 1~6 Precision | 어느 클래스에서 오탐이 많은지 파악 | 절대 수치보다 패턴 확인 |
| 클래스 1~6 Recall | 어느 클래스가 잘 안 잡히는지 파악 | |
| Macro F1 | 0.80 이상이면 파이프라인 동작 확인 | 1인 데이터 기준 참고치 |
| Class 0 Precision | Neutral → 제스처 오탐 비율 확인 | 높으면 hardneg 보강 필요 |

> Precision을 Recall보다 우선 확인하는 이유: 오탐(잘못된 소리)이 미탐(인식 실패)보다 UX 파괴적이다.

---

## 2. 응용 평가 ① — FP/min (오탐 트리거 빈도)

프레임 F1이 0.98이어도 FP가 Neutral 구간에서 빈발하면 제품이 망가진다.
**사운드 UX의 실질적 품질 지표**.

### 2.1 개념

```
postprocess 파이프라인:
  raw argmax → threshold(τ) → N-frame voting → K-frame debounce → 트리거 이벤트

FP/min = (Neutral 구간에서 발생한 비정상 트리거 수) / (Neutral 구간 시간(분))
```

### 2.2 postprocess 파라미터

| 파라미터 | 설명 | 초기 권장값 |
|---------|------|------------|
| `τ` (threshold) | `p_max < τ`면 Neutral 강제 | 0.80 ~ 0.90 |
| `N` (vote_window) | 최근 N프레임 다수결 | 5 ~ 10 |
| `K` (debounce) | K프레임 연속 조건 만족 시 확정 | 3 ~ 5 |

> K=5, 30fps → 최소 167ms 지연 발생. 지연과 FP는 trade-off.

### 2.3 구현 코드

```python
# evaluate_triggers.py
import pandas as pd
import numpy as np
from collections import deque

def postprocess(df: pd.DataFrame, tau: float, vote_n: int, debounce_k: int):
    """
    프레임 예측값을 실제 사운드 트리거 이벤트로 변환하는 후처리 로직입니다.
    Threshold -> Voting -> Debounce 순서로 노이즈를 필터링합니다.
    """
    events = []
    window = deque(maxlen=vote_n) # 최근 N개 프레임의 예측을 저장하는 큐
    debounce_count = 0            # 연속으로 동일한 제스처가 판정된 횟수
    last_trigger = -1             # 가장 최근에 트리거(확정)된 제스처

    for _, row in df.iterrows():
        # 1. 임계값(tau) 미만이면 Neutral(0) 처리하여 불확실한 오탐 방지
        pred = row["pred_class"] if row["p_max"] >= tau else 0
        window.append(pred)

        # 윈도우가 가득 찰 때까지는 판정 보류
        if len(window) < vote_n:
            continue

        # 2. 다수결(Voting): 최근 N개 프레임 중 가장 많이 나온 클래스 선정
        voted = max(set(window), key=list(window).count)

        if voted != 0:
            # 3. 디바운스(Debounce): 의미 있는 제스처일 때 카운트 누적
            debounce_count += 1
            
            # K번 연속 유지되었고, 직전 트리거와 다른 제스처일 때만 이벤트 확정
            if debounce_count >= debounce_k and voted != last_trigger:
                events.append({
                    "frame_idx": row["frame_idx"],
                    "timestamp": row["timestamp"],
                    "triggered_class": voted,
                    "p_max": row["p_max"],
                })
                last_trigger = voted
        else:
            # Neutral로 돌아오면 카운트 및 마지막 트리거 초기화
            debounce_count = 0
            last_trigger = -1

    return pd.DataFrame(events)


def calc_fp_per_min(labels_csv: str, preds_csv: str,
                    tau=0.85, vote_n=7, debounce_k=3):
    """실제 평상시(Neutral) 구간에서 발생하는 오탐(FP) 트리거 빈도를 계산합니다."""
    labels = pd.read_csv(labels_csv)
    preds  = pd.read_csv(preds_csv)
    df = labels.merge(preds, on="frame_idx", how="inner")

    # 1. 정답이 Neutral(0)인 구간만 분리 (오탐 테스트용)
    neutral_df = df[df["gesture"] == 0].copy()
    if neutral_df.empty:
        print("Neutral 구간 없음")
        return

    # 2. 분리된 구간에 후처리 로직을 돌려 잘못 발생한 이벤트를 추출
    triggers = postprocess(neutral_df, tau, vote_n, debounce_k)

    # 3. 타임스탬프(mm:ss:msec)를 초(sec)로 변환하는 함수
    def ts_to_sec(ts):
        parts = ts.split(":")
        return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 1000

    neutral_df = neutral_df.copy()
    neutral_df["sec"] = neutral_df["timestamp"].apply(ts_to_sec)
    
    # 분석에 사용된 Neutral 구간의 총 시간(분) 도출
    duration_min = (neutral_df["sec"].max() - neutral_df["sec"].min()) / 60

    # 4. 제스처(0이 아닌 클래스)로 잘못 트리거된 횟수 산출
    fp_count = len(triggers[triggers["triggered_class"] != 0]) if not triggers.empty else 0
    
    # 5. 분당 오탐 발생 횟수 (FP/min) 계산
    fp_per_min = fp_count / duration_min if duration_min > 0 else float("inf")

    print(f"[τ={tau}, N={vote_n}, K={debounce_k}]")
    print(f"  Neutral 구간: {duration_min:.2f}분")
    print(f"  FP 트리거 수: {fp_count}")
    print(f"  FP/min     : {fp_per_min:.3f}")
    return fp_per_min
```

### 2.4 τ 스윕 — FP/min vs Recall Trade-off

```python
import matplotlib.pyplot as plt

results = []
for tau in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    fp = calc_fp_per_min("labels.csv", "preds.csv", tau=tau, vote_n=7, debounce_k=3)
    # recall은 frame-level에서 별도 계산
    results.append({"tau": tau, "fp_per_min": fp})

df_res = pd.DataFrame(results)
plt.plot(df_res["tau"], df_res["fp_per_min"], marker="o")
plt.axhline(0.5, linestyle="--", color="red", label="목표: FP/min ≤ 0.5")
plt.xlabel("Threshold τ")
plt.ylabel("FP/min @ Neutral")
plt.title("Threshold Sweep — FP/min Trade-off")
plt.legend()
plt.tight_layout()
plt.show()
```

### 2.5 PoC 확인 기준

> PoC 단계에서는 절대 수치보다 **τ / N / K 조합에 따라 FP/min이 어떻게 변하는지 경향 파악**이 목적.

| 구간 | PoC 확인 포인트 |
|------|----------------|
| FP/min @ neutral | τ 스윕 그래프에서 FP/min이 감소하는 패턴 확인 |
| FP/min @ hardneg | neutral 대비 얼마나 더 오탐이 나는지 비교 |

---

## 3. 응용 평가 ② — 트리거 지연 + Jitter CDF

실시간성과 리듬 품질의 핵심. 평균만 보면 스파이크를 놓친다.

### 3.1 측정 대상 (Stage-wise)

전체 지연 시간만 측정해서는 어느 부분에서 랙(Lag)이 걸리는지 파악할 수 없으므로, 카메라에서 영상을 받아 음악이 나오기까지의 과정을 **5개의 단계(Stage)로 쪼개어 부분별 지연시간을 측정**해야 병목을 식별할 수 있습니다.

```text
L_total = L_mp + L_feat + L_mlp + L_post + L_audio

L_mp   : MediaPipe Hand Landmarker 처리 시간 (웹캠 프레임 → 좌표 도출)
         *병목 예상 지점 1순위 (기기 스펙 의존)*
L_feat : 좌표 정규화 + 피처 벡터 생성 연산 시간 (스케일링, 각도 계산 등)
L_mlp  : MLP forward + softmax (분류기 통과 시간)
         *1~3ms 내외로 매우 짧아야 정상*
L_post : threshold + voting + debounce (후처리 확정 연산 시간)
L_audio: Web Audio 스케줄링 지연 (baseLatency + outputLatency)
         *소리가 실제 스피커로 출력되기까지의 OS 레벨 지연*
```

> **실무 적용 팁**: Python 오프라인 테스트에서는 영상 프레임 입출력이 없으므로 주로 `L_mlp`와 `L_post`만 측정됩니다. 향후 브라우저단 실시간 런타임(JavaScript) 구현 시, 각 모듈 앞뒤에 `performance.now()` 타이머를 삽입하여 이 5가지 지표를 별도 수집하고 로깅해야 완벽한 병목 분석이 가능합니다.
> 30fps 기준 프레임당 가용 시간은 33ms입니다. K=5 debounce 적용 시 확정까지 최소 +167ms의 논리적 지연이 추가됩니다.

### 3.2 preds CSV 권장 스키마

파이프라인 각 단계에서 측정한 부분 지연 시간을 CSV에 함께 기록합니다.

```
frame_idx, pred_class, p_max, p0, p1, p2, p3, p4, p5, p6,
t_mp_ms, t_feat_ms, t_mlp_ms, t_post_ms, latency_total_ms
```

### 3.3 Latency CDF 및 Stage-wise Box Plot 코드

각 단계별 지연 시간의 분포를 한눈에 비교할 수 있는 **Box Plot**과 전체 시스템 지연(Total Latency)의 꼬리(Spike) 현상을 분석하는 **CDF 플롯**을 함께 생성합니다.

```python
# evaluate_latency.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_latency_cdf(preds_csv: str):
    """
    각 파이프라인 스테이지별 지연 시간(Latency) 분포를 박스 플롯으로 확인하고,
    전체 엔드투엔드 지연 시간의 누적 분포 함수(CDF)를 그려 실시간성을 평가합니다.
    """
    df = pd.read_csv(preds_csv)

    # 파이프라인 단계별 소요 시간 컬럼 매핑
    stages = {
        "MediaPipe":  "t_mp_ms",      # 랜드마크 추출 시간
        "Feature":    "t_feat_ms",    # 좌표 정규화 및 피처 연산 시간
        "MLP":        "t_mlp_ms",     # 모델 추론 시간
        "PostProcess":"t_post_ms",    # 후처리 (Voting, Debounce) 연산 시간
        "Total":      "latency_total_ms", # 전체 지연 시간 합계
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- 좌: 스테이지별 Box Plot ---
    ax = axes[0]
    # 결측치(NaN)를 제거하고 유효한 데이터만 추출하여 박스 플롯 생성
    # 각 컴포넌트 중 어느 것이 시간을 가장 많이 소비하는지(병목) 파악
    stage_data = [df[col].dropna().to_numpy()
                  for col in stages.values() if col in df.columns]
    ax.boxplot(stage_data, labels=[k for k, v in stages.items() if v in df.columns])
    
    # 30fps 기준 프레임당 가용 시간(약 33ms)을 빨간 점선으로 표시
    ax.axhline(33, linestyle="--", color="red", label="33ms (30fps 기준)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Stage-wise Latency Distribution")
    ax.legend()

    # --- 우: Total Latency CDF ---
    ax = axes[1]
    if "latency_total_ms" in df.columns:
        # 전체 지연 시간 데이터를 오름차순 정렬
        x = np.sort(df["latency_total_ms"].dropna().to_numpy())
        # CDF의 y축(누적 확률) 계산
        y = np.arange(1, len(x) + 1) / len(x)
        
        # 주요 퍼센타일(50%, 95%, 99%) 지연 시간 추출 (스파이크/이상치 파악 용도)
        # p99 지표가 급격히 높다면 순간적인 랙(Lag)이 발생하고 있음을 의미
        p50 = np.percentile(x, 50)
        p95 = np.percentile(x, 95)
        p99 = np.percentile(x, 99)

        ax.plot(x, y, linewidth=2)
        # 퍼센타일 값을 세로선으로 표시
        for p, label in [(p50, "p50"), (p95, "p95"), (p99, "p99")]:
            ax.axvline(p, linestyle="--", label=f"{label}={p:.1f}ms")
            
        # 목표 지연 시간(200ms)을 기준선으로 표시
        ax.axvline(200, linestyle="-", color="red", alpha=0.5, label="목표 200ms")
        ax.set_xlabel("Total Latency (ms)")
        ax.set_ylabel("CDF")
        ax.set_title(f"End-to-End Latency CDF")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("runs/reports/latency_cdf.png", dpi=150)
    plt.show()
    print(f"Latency — p50: {p50:.1f}ms | p95: {p95:.1f}ms | p99: {p99:.1f}ms")
```

### 3.4 Jitter (트리거 간격 변동) 측정

```python
def calc_trigger_jitter(events_df: pd.DataFrame):
    """
    연속으로 발생한 사운드 트리거 이벤트 사이의 시간 간격 변동성(Jitter)을 측정합니다.
    지터가 높으면 사용자가 '리듬이 밀리거나 흔들린다'고 체감하게 됩니다.
    """
    if len(events_df) < 2:
        print("트리거 이벤트 부족 (< 2)")
        return

    # 타임스탬프(mm:ss:msec) 문자열을 초(second) 단위의 float로 변환
    def ts_to_sec(ts):
        parts = ts.split(":")
        return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 1000

    events_df = events_df.copy()
    events_df["sec"] = events_df["timestamp"].apply(ts_to_sec)
    
    # diff()를 사용해 연속된 이벤트 간의 시간 차이(간격)를 계산하고 ms 단위로 변환
    intervals_ms = events_df["sec"].diff().dropna() * 1000

    # 지터 산출 (간격의 표준편차를 사용하거나 p95-p50의 편차를 사용)
    jitter = intervals_ms.std()
    p95_p50 = np.percentile(intervals_ms, 95) - np.percentile(intervals_ms, 50)

    print(f"Jitter (std)   : {jitter:.1f}ms")
    print(f"Jitter (p95-p50): {p95_p50:.1f}ms")
    return jitter
```

### 3.5 PoC 확인 기준

> PoC 단계에서는 절대 수치 통과 여부보다 **병목 스테이지 파악 + K 조합별 지연 분포 확인**이 목적.

| 지표 | PoC 확인 포인트 |
|------|----------------|
| Stage-wise Box Plot | 어느 단계가 가장 느린지 확인 (MediaPipe vs MLP vs PostProcess) |
| Total Latency p95 | debounce K 값 변화에 따른 지연 증가 경향 파악 |
| Jitter (p95 - p50) | 지연이 일정한지 or 스파이크가 있는지 확인 |

---

## 4. 전체 평가 실행 순서

실제 프로젝트 디렉토리 구조(`src/training/evaluation.py`)에 맞춘 CLI 실행 예시입니다. 단일 스크립트에 파라미터(`--mode`)를 넘기는 구조를 권장합니다.

```bash
# Step 1: 프레임 단위 평가 (모델 품질)
python src/training/evaluation.py --mode frames \
  --labels data/labeled_data/3_fast_right_man1.csv \
  --preds  runs/preds/3_fast_right_man1_pred.csv \
  --out    runs/reports/

# Step 2: 오탐 트리거 평가 (UX 품질)
python src/training/evaluation.py --mode triggers \
  --labels data/labeled_data/0_neutral_right_man1.csv \
  --preds  runs/preds/0_neutral_right_man1_pred.csv \
  --tau 0.85 --vote_n 7 --debounce_k 3 \
  --out    runs/reports/

# Step 3: 지연 CDF
python src/training/evaluation.py --mode latency \
  --preds  runs/preds/all_pred.csv \
  --out    runs/reports/
```

---

## 5. 평가 결과 리포트 템플릿

```
=== JamJamBeat 평가 리포트 ===

[데이터 분할]
  train: {파일 목록}
  test : {파일 목록}  ← 학습 미사용 세션/영상

[1. 프레임 단위 — Confusion Matrix & F1]
  Macro F1    : 0.XXX  (목표: ≥ 0.93)
  Class 0 Precision: 0.XXX  (목표: ≥ 0.99)
  Class 1~6 Precision (min): 0.XXX  (목표: ≥ 0.98)
  Class 1~6 Recall   (min): 0.XXX  (목표: ≥ 0.90)
  → confusion_matrix.png 참조

[2. 이벤트 단위 — FP/min]
  τ=0.85, N=7, K=3
  FP/min @ neutral: X.XXX  (목표: ≤ 0.5)
  FP/min @ hardneg: X.XXX  (목표: ≤ 1.0)

[3. 지연 단위 — Latency CDF]
  p50: XX.Xms | p95: XX.Xms | p99: XX.Xms  (p95 목표: ≤ 200ms)
  Jitter (p95-p50): XX.Xms  (목표: ≤ 100ms)
  → latency_cdf.png 참조

[판정]
  □ 기준 미달 항목: ...
  □ 다음 액션: τ 조정 / Neutral 데이터 추가 / debounce K 증가 / ...
```

---

## 6. 데이터 확장 단계별 평가 주의사항

| 항목 | 현재 (1인) | 팀원 4명 단계 (우선 목표) | 서비스 준비 단계 (추후) |
|------|------------|------------------------|-------------------------|
| **평가 지표 (F1, FP/min)** | 특정 1인의 특성에 과대평가될 수 있으므로 절대 수치보다는 **파이프라인 동작 확인용**으로만 사용 | 4명 간의 개인차(손 크기, 속도, 수행 방식)가 반영되어 유의미한 평가 시작. 단, 팀원들의 비슷한 환경에 편향될 수 있음 | 조명, 배경, 연령대(아동) 등 다양한 환경이 반영된 **신뢰도 높은 서비스 품질 기준**으로 활용 |
| **테스트셋 구성** | 다른 세션이나 속도(예: fast) 영상을 테스트용으로 지정 | 특정 1명의 데이터 전체를 테스트 전용으로 고정 (평가 시마다 대상 교체 가능) | 학습과 완전히 독립된 별도 참가자 그룹(예: 4~5명)을 테스트 전용으로 구축 |
| **일반화 검증** | 불가 (기본 동작 여부만 확인) | 초기 MVP 수준의 일반화 및 런타임 안정성 확인 가능 | **서비스급: 20명 이상 분포에서의 검증 필수** |
| **오탐(FP) 평가** | 해당 1인의 Neutral/Hardneg 환경에서의 오탐 빈도 측정 | 4명 모두의 헷갈리는 제스처 패턴을 취합하여 **공통적인 오탐에 대한 방어력 측정** | 실사용 환경에서 발생할 수 있는 예상치 못한 행동에 대한 강건성 종합 평가 |
