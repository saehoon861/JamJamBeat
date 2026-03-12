# JamJamBeat 모델 파이프라인 실행 가이드

## 0. 환경 준비

```bash
cd /home/user/projects/JamJamBeat/model
```

최초 1회: 의존성 설치
```bash
uv sync
```

> 이후 모든 명령은 `/home/user/projects/JamJamBeat/model` 기준으로 실행한다.

---

## 1. 데이터 전처리 (Data Fusion)

원시 랜드마크 CSV → 모델 학습용 피처 CSV 변환 단계.
`*_output.csv`가 이미 있으면 이 단계는 건너뛰어도 된다.

### 단일 파일 전처리

```bash
uv run python data_fusion/hand_preprocess.py \
    ../data/landmark_data/man1_right.csv \
    data_fusion/man1_right_for_poc_output.csv
```

출력 파일명 자동 지정 (input 이름 + `_preprocessed.csv`):
```bash
uv run python data_fusion/hand_preprocess.py ../data/landmark_data/man1_right.csv
```

### 4개 파일 일괄 전처리

```bash
for name in man1 man2 man3 woman1; do
  uv run python data_fusion/hand_preprocess.py \
    ../data/landmark_data/${name}_right_for_poc.csv \
    data_fusion/${name}_right_for_poc_output.csv
done
```

**입력 CSV 필수 컬럼:** `source_file, frame_idx, timestamp, gesture, x0,y0,z0, ..., x20,y20,z20`

**출력 CSV 추가 컬럼:**
```
nx*, ny*, nz*   정규화 joint 좌표  (63개)
bx*, by*, bz*, bl*   bone vector + length  (84개)
flex_*, abd_*   손가락 굽힘/벌어짐 각도  (9개)
합계: 156개 피처
```

---

## 2. 모델 학습 (전체 순차 실행, 권장)

```bash
uv run python model_pipelines/run_all.py
```

13개 모델을 순서대로 학습 → 평가 → 저장한다.
완료 후 비교 결과가 아래 경로에 저장된다:

```
model_evaluation/pipelines/{suite_name}/comparison_results.csv
```

---

## 3. 특정 모델만 실행

```bash
uv run python model_pipelines/run_pipeline.py --model-id mlp_baseline
```

**선택 가능한 model-id:**
```
mlp_baseline          mlp_baseline_full     mlp_baseline_seq8
mlp_sequence_joint    mlp_sequence_delta    mlp_temporal_pooling
mlp_embedding         two_stream_mlp        cnn1d_tcn
transformer_embedding mobilenetv3_small     shufflenetv2_x0_5
efficientnet_b0
```

---

## 4. 여러 모델만 골라서 실행

```bash
uv run python model_pipelines/run_all.py \
    --models mlp_baseline mlp_baseline_full two_stream_mlp
```

---

## 5. 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--epochs` | `20` | 최대 학습 에폭 수 |
| `--batch-size` | `32` | 배치 크기 (이미지 모델 CPU OOM 방지 기준) |
| `--lr` | `1e-3` | 학습률 |
| `--patience` | `6` | Early stopping 기준 에폭 수 |
| `--device` | `cpu` | `cpu` / `cuda` / `auto` |
| `--seed` | `42` | 재현성 시드 |

```bash
# 예: 에폭 50, 배치 64로 실행
uv run python model_pipelines/run_all.py --epochs 50 --batch-size 64
```

---

## 6. 실행 결과 위치

`run_all.py` 배치 실행 기준 구조:

```
model_evaluation/pipelines/
├── latest_suite.json            ← 가장 최근 run_all suite 포인터
└── {yyyymmdd_HHMMSS}__{dataset_tag}/
    ├── comparison_suite.json    ← 이번 배치의 입력 CSV / 모델 목록 메타데이터
    ├── comparison_results.csv   ← 이번 배치의 전체 비교표
    └── {model_id}/
        ├── latest.json          ← 이 suite 안에서의 최신 실험 포인터
        └── {yyyymmdd_HHMMSS}/
            ├── model.pt             ← 학습된 가중치
            ├── preds_test.csv       ← 테스트 예측값
            ├── train_history.csv    ← epoch별 loss/acc
            ├── run_summary.json     ← 전체 메트릭 요약
            └── evaluation/
                ├── dataset_info.json    ← 입력 CSV / split source group 정보
                ├── metrics_summary.json
                ├── confusion_matrix.png
                ├── per_class_report.csv
                └── latency_cdf.png
```

`run_pipeline.py` 단독 실행은 기존처럼 `model_evaluation/pipelines/{model_id}/{timestamp}/` 아래에 바로 저장된다.

---

## 7. 결과 빠르게 확인

```bash
# 가장 최근 batch 확인
cat model_evaluation/pipelines/latest_suite.json

# 해당 suite 안의 비교표 확인
cat model_evaluation/pipelines/{suite_name}/comparison_results.csv

# 특정 모델 메트릭
cat model_evaluation/pipelines/{suite_name}/mlp_baseline/latest.json
```

---

## 8. 모델별 영상 체크 (추론 시각화)

학습된 모델을 실제 영상에 돌려보고 예측 결과를 오버레이로 확인한다.

**사전 조건:** 프로젝트 루트에 `hand_landmarker.task` 파일이 있어야 한다.

### UI 모드 (드롭다운으로 run/영상 선택)

```bash
uv run python "model_evaluation/모델별영상체크/video_check_app.py"
```

실행하면 학습된 run과 `../data/raw_data/` 영상을 자동으로 탐색해 드롭다운으로 선택할 수 있다.

### 직접 지정 모드

```bash
uv run python "model_evaluation/모델별영상체크/video_check_app.py" \
  --run-dir model_evaluation/pipelines/{suite_name}/mlp_baseline \
  --video ../data/raw_data/4_slow_right_man3.mp4
```

`--run-dir`에는 모델 폴더(`latest.json` 기준)나 실제 timestamp run 폴더 둘 다 사용할 수 있다.

### 재생 컨트롤

| 키 | 동작 |
|----|------|
| `Space` | 일시정지 / 재생 |
| `A` / `←` | 이전 프레임 |
| `D` / `→` | 다음 프레임 |
| `R` | 처음부터 재시작 |
| `Q` / `Esc` | 종료 |

> 영상 전체를 먼저 분석한 뒤 재생 창이 열린다.
> `cnn1d_tcn`, `transformer_embedding` 등 sequence 모델은 초반 몇 프레임은 워밍업 구간이라 예측이 불안정할 수 있다.

---

## 9. PoC 수용 기준

| 지표 | 기준 |
|------|------|
| `macro_f1` | ≥ 0.80 |
| `class0_fnr` | < 0.10 |
| `fp_per_min` | < 2.0 |
| `latency_p95_ms` | < 200 |
