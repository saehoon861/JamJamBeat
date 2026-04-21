# JamJamBeat 통합 실험 설계 가이드 (Loss & Dataset Variant)

이 문서는 Class 0(Neutral/Hard negative)에서 발생하는 오탐지(False Positive) 및 제스처에 대한 과도한 확신(Overconfidence) 문제를 해결하기 위해 고안된 **격리된 실험 시나리오**를 안내합니다.

> **🚨 중요 경고 (Tau 튜닝에 대하여):**
> 현재 `class0_fnr > 0.5` 상태는 모델의 Decision Boundary 자체가 잘못 형성된 **학습 단계의 문제**입니다. 이 상태에서 사후 처리용 임계값인 `tau`만 극단적으로 상향(`0.95` 이상)하면, 실제 제스처(Class 1~6)의 Recall이 완전히 붕괴되어 전체 `macro_f1`이 나빠지는 치명적인 Trade-off가 발생합니다.
> 
> **해결 원칙:** 사후 처리(`tau`)를 만지기 전에, 반드시 아래 안내된 "학습 단계 수정(Loss 튜닝)"을 먼저 수행하여 모델의 근본적인 추론 능력을 고쳐야 합니다.

---

## 1단계: 학습 단계 수정 (우선 시도)

가장 유력한 원인인 **"이중 가중치(Sampler+Alpha) 및 Focal Loss 부작용"**을 제거하고, Label Smoothing으로 과신을 막는 실험입니다.

### 실험 E (1순위 권고안)
* **설정:** `CrossEntropyLoss` + `WeightedRandomSampler` + `label_smoothing=0.1` (`alpha` 미사용)
* **목적:** 데이터 불균형은 Sampler로만 잡고(이중 가중치 제거), Focal Loss 대신 표준 CE를 쓰며, Smoothing으로 과도한 확신을 꺾습니다.
* **명령어:**
```bash
uv run python model_pipelines/run_all.py \
  --dataset-key baseline \
  --models mlp_embedding \
  --device auto \
  --loss-type cross_entropy \
  --no-use-alpha \
  --use-label-smoothing
```

> **결과 확인:** 위 커맨드 실행 후 `comparison_results.csv`에서 `class0_fnr`이 0.5 밑으로 유의미하게 떨어지는지, 동시에 `macro_f1`이 유지되는지 먼저 확인하십시오.

---

## 2단계: 최적 Loss 기반 Dataset Variant 비교

1단계를 통해 최적의 Loss 설정(예: CE+Smoothing+Sampler)이 확정되면, 해당 Loss 설정을 고정한 상태에서 입력 데이터의 기하학적 정규화 방식(Variant)을 테스트합니다.

* **유력 후보 (`pos_scale`):** 손목 기준으로 이동하고 스케일을 정규화하여, 카메라 거리나 사람마다 다른 손 크기에 가장 강건(Robust)할 것으로 예상됩니다.

**실행 명령 (1단계 설정이 최적이라고 가정했을 때):**
```bash
uv run python model_pipelines/run_all.py \
  --dataset-key baseline pos_only scale_only pos_scale \
  --models mlp_embedding \
  --device auto \
  --loss-type cross_entropy \
  --no-use-alpha \
  --use-label-smoothing
```

---

## 3단계: 전체 아키텍처 비교 및 후처리(Tau) 미세 조정

최적의 Loss와 최적의 Dataset Variant(예: `pos_scale`)가 확정되면, 시간축(Temporal) 정보를 활용하는 모델들을 포함하여 최종 비교를 수행합니다.
이 단계에서 필요하다면 `tau` 값을 `0.85`에서 `0.90` 정도로만 아주 살짝 올려서 미세 튜닝을 시도해 볼 수 있습니다.

**실행 명령 (예시):**
```bash
uv run python model_pipelines/run_all.py \
  --dataset-key pos_scale \
  --models mlp_original mlp_embedding mlp_sequence_delta cnn1d_tcn transformer_embedding \
  --device auto \
  --loss-type cross_entropy \
  --no-use-alpha \
  --use-label-smoothing \
  --tau 0.90
```

---

## 평가 및 해석 기준

전체 Accuracy(정확도)보다 **실제 서비스 품질에 직결되는 아래 3가지 지표를 최우선으로 확인**해야 합니다. (`comparison_results.csv` 참고)

1. **`class0_fnr` (Neutral 오발동률):** Neutral 또는 Hard negative 구간을 뚫고 들어온 실패율. (**목표: < 0.10**)
2. **`fp_per_min` (분당 오발동 빈도):** 시간축(Video) 추론 시 1분에 오작동하는 횟수. (**목표: < 2.0**)
3. **`macro_f1` (전체 균형 성능):** 다수 클래스(Neutral)에 가려지지 않은 진짜 제스처 분류 성능. (**목표: ≥ 0.80**)
