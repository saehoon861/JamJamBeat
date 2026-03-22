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

## 2. 역할형 학습데이터셋 생성

새 파이프라인은 `학습데이터셋` 폴더의 역할형 CSV를 기준으로 동작한다.

```bash
uv run python data_fusion/build_training_datasets.py
```

생성 결과 예시:
```
data_fusion/학습데이터셋/
├── baseline_train.csv
├── baseline_val.csv
├── baseline_inference.csv
├── baseline_test.csv
├── ...
└── dataset_manifest.csv
```

---

## 3. 모델 학습 (전체 순차 실행, 권장)

```bash
uv run python model_pipelines/run_all.py
```

기본적으로 12개 dataset key를 자동 인식하고, core 9개 모델을 dataset별로 순차 실행한다.
image 모델 3종까지 포함해 12개를 돌리려면:

```bash
uv run python model_pipelines/run_all.py --include-image-models
```

---

## 4. 특정 dataset / 특정 모델만 실행

```bash
uv run python model_pipelines/run_all.py \
    --dataset-key baseline_ds_1_none \
    --models mlp_baseline mlp_embedding
```

---

## 5. 단일 모델만 직접 실행

`run_pipeline.py`는 이제 역할형 CSV를 직접 받는다.

```bash
uv run python model_pipelines/run_pipeline.py \
    --model-id mlp_baseline \
    --train-csv data_fusion/학습데이터셋/baseline_ds_1_none_train.csv \
    --val-csv data_fusion/학습데이터셋/baseline_ds_1_none_val.csv \
    --test-csv data_fusion/학습데이터셋/baseline_ds_1_none_test.csv \
    --inference-csv data_fusion/학습데이터셋/baseline_ds_1_none_inference.csv
```

---

## 6. 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--epochs` | `20` | 최대 학습 에폭 수 |
| `--batch-size` | `32` | 배치 크기 (이미지 모델 CPU OOM 방지 기준) |
| `--lr` | `1e-3` | 학습률 |
| `--patience` | `6` | Early stopping 기준 에폭 수 |
| `--loss-type` | `focal` | `cross_entropy` 또는 `focal` |
| `--use-weighted-sampler` | `true` | train loader에 weighted sampler 사용 여부 |
| `--use-alpha` | `true` | class alpha(weight) 사용 여부 |
| `--use-label-smoothing` | `false` | label smoothing 사용 여부 |
| `--device` | `cpu` | `cpu` / `cuda` / `auto` |
| `--seed` | `42` | 재현성 시드 |

```bash
# 예: 에폭 50, 배치 64로 실행
uv run python model_pipelines/run_all.py --epochs 50 --batch-size 64

# 예: plain cross entropy 실험
uv run python model_pipelines/run_all.py \
  --loss-type cross_entropy \
  --no-use-weighted-sampler \
  --no-use-alpha \
  --no-use-label-smoothing
```

손실함수 관련 기본 수치는 `model_pipelines/_shared.py` 상단 상수에서 관리한다.
- `DEFAULT_FOCAL_GAMMA`
- `DEFAULT_LABEL_SMOOTHING`

---

## 7. 실행 결과 위치

`run_all.py` 배치 실행 기준 구조:

```
model_evaluation/pipelines/
├── latest_suite.json            ← 가장 최근 dataset suite 포인터
└── {yyyymmdd_HHMMSS}__{dataset_key}/
    ├── comparison_suite.json    ← dataset key / 입력 역할 CSV / 모델 목록 메타데이터
    ├── comparison_results.csv   ← 이번 배치의 전체 비교표
    └── {model_id}/
        ├── latest.json          ← 이 suite 안에서의 최신 실험 포인터
        └── {yyyymmdd_HHMMSS}/
            ├── model.pt             ← 학습된 가중치
            ├── preds_test.csv       ← 테스트 예측값
            ├── preds_inference.csv  ← hold-out inference 예측값
            ├── train_history.csv    ← epoch별 loss/acc
            ├── run_summary.json     ← 전체 메트릭 요약
            └── evaluation/
                ├── dataset_info.json    ← 입력 역할 CSV / split 정보
                ├── metrics_summary.json
                ├── confusion_matrix.png
                ├── per_class_report.csv
                └── latency_cdf.png
```

`run_pipeline.py` 단독 실행은 `--output-root` 아래 `{model_id}/{timestamp}/`에 저장된다.

---

## 8. 결과 빠르게 확인

```bash
# 가장 최근 batch 확인
cat model_evaluation/pipelines/latest_suite.json

# 해당 suite 안의 비교표 확인
cat model_evaluation/pipelines/{suite_name}/comparison_results.csv

# 특정 모델 메트릭
cat model_evaluation/pipelines/{suite_name}/mlp_baseline/latest.json
```

---

<<<<<<< HEAD
## 9. 모델별 영상/이미지 체크

현재 기본 viewer는 `video_check_app_train_aligned.py`다.  
run의 `dataset_variant`를 읽어서 학습 때와 같은 landmark 좌표계로 맞춘 뒤 추론한다.
UI 모드에서는 raw 영상 전체를 계속 보여주되, 화면의 `Inference Videos` 패널에서
선택한 run의 hold-out inference 영상 목록과 현재 선택 영상의 포함 여부를 바로 확인할 수 있다.

**사전 조건:** 프로젝트 루트에 `hand_landmarker.task` 파일이 있어야 한다.

### 9-1. 기본 영상 viewer

```bash
# UI 모드
uv run python "model_evaluation/모델별영상체크/video_check_app_train_aligned.py"
```

```bash
# direct run
uv run python "model_evaluation/모델별영상체크/video_check_app_train_aligned.py" \
  --run-dir model_evaluation/pipelines/mlp_baseline/20260318_152800 \
  --video ../data/raw_data/4_slow_right_man3.mp4
```

```bash
# suite latest.json
uv run python "model_evaluation/모델별영상체크/video_check_app_train_aligned.py" \
  --run-dir model_evaluation/pipelines/{suite_name}/mlp_baseline \
  --video ../data/raw_data/4_slow_right_man3.mp4
```

`--run-dir`는 아래를 모두 지원한다.
- `model_evaluation/pipelines/{model_id}/{run_id}/`
- `model_evaluation/pipelines/{model_id}/latest.json`
- `model_evaluation/pipelines/{suite_name}/{model_id}/{run_id}/`
- `model_evaluation/pipelines/{suite_name}/{model_id}/latest.json`

재생 컨트롤:
=======
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
  --run-dir model_evaluation/pipelines/mlp_baseline/20260310_082223 \
  --video ../data/raw_data/4_slow_right_man3.mp4
```

### 재생 컨트롤
>>>>>>> develop

| 키 | 동작 |
|----|------|
| `Space` | 일시정지 / 재생 |
| `A` / `←` | 이전 프레임 |
| `D` / `→` | 다음 프레임 |
| `R` | 처음부터 재시작 |
| `Q` / `Esc` | 종료 |

<<<<<<< HEAD
### 9-2. 영상 export

```bash
uv run python "model_evaluation/모델별영상체크/video_check_app_train_aligned_export.py" \
  --run-dir model_evaluation/pipelines/mlp_baseline/20260318_152800 \
  --video ../data/raw_data/4_slow_right_man3.mp4
```

### 9-3. 이미지셋 추론 체크

```bash
uv run python "model_evaluation/모델별영상체크/image_check_app_train_aligned.py" \
  --run-dir model_evaluation/pipelines/mlp_baseline/20260318_152800 \
  --images-root data_fusion/추론용데이터셋
```

sequence 모델을 이미지셋에 넣을 때는 기본적으로 `independent` 모드를 사용한다.

```bash
uv run python "model_evaluation/모델별영상체크/image_check_app_train_aligned.py" \
  --run-dir model_evaluation/pipelines/mlp_baseline/20260318_152800 \
  --images-root data_fusion/추론용데이터셋 \
  --image-sequence-mode independent
```

기본 출력:

```text
run_dir/image_inference/{dataset_name}/
├── preds_images.csv
├── dataset_info.json
├── inference_summary.json
└── evaluation/
```

---

## 10. 오류 프레임 분석 (예측 vs Ground Truth 비교)

`error_frame_viewer.py`는 현재 `train_aligned` 런타임을 사용한다.  
즉 `pos_only`, `scale_only`, `pos_scale` run도 학습 때 쓴 variant에 맞춰 GT와 비교한다.

### UI 모드

```bash
uv run python "model_evaluation/모델별영상체크/error_frame_viewer.py"
```

기본 GT CSV 탐색 순서:
- `data_fusion/학습데이터셋/*_test.csv`
- `data_fusion/학습데이터셋/*_inference.csv`
- 예전 POC CSV는 보조 호환 입력

### CLI 모드

```bash
# 공식 test split 사용
uv run python "model_evaluation/모델별영상체크/error_frame_viewer.py" \
  --run-dir model_evaluation/pipelines/mlp_baseline/20260318_152800 \
  --csv data_fusion/학습데이터셋/baseline_test.csv
```

```bash
# hold-out inference split 사용
uv run python "model_evaluation/모델별영상체크/error_frame_viewer.py" \
  --run-dir model_evaluation/pipelines/mlp_baseline \
  --csv data_fusion/학습데이터셋/baseline_inference.csv \
  --context-frames 5
```

```bash
# 공식 split CSV 내부 source_file만 선택
uv run python "model_evaluation/모델별영상체크/error_frame_viewer.py" \
  --run-dir model_evaluation/pipelines/mlp_baseline/20260318_152800 \
  --csv data_fusion/학습데이터셋/baseline_test.csv \
  --source-filter 3_fast_right_man1 3_slow_right_man1
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--run-dir` | (필수) | run 폴더 또는 latest.json 있는 모델 폴더 |
| `--csv` | (필수) | ground truth CSV. 반복 사용 가능 |
| `--context-frames` | `0` | 오류 프레임 앞뒤로 포함할 프레임 수 |
| `--source-filter` | 전체 | 분석할 `source_file` 이름 목록 |

### 출력 결과

```text
run_dir/error_analysis/
├── error_frames_{model_id}.mp4
└── error_summary.csv
```

영상 오버레이 기호:
- **빨간 테두리** = 실제 오류 프레임
- **노란 테두리** = context 프레임
- 하단 좌 = `GT`, 하단 우 = `PRED`

---

## 11. PoC 수용 기준
=======
> 영상 전체를 먼저 분석한 뒤 재생 창이 열린다.
> `cnn1d_tcn`, `transformer_embedding` 등 sequence 모델은 초반 몇 프레임은 워밍업 구간이라 예측이 불안정할 수 있다.

---

## 9. PoC 수용 기준
>>>>>>> develop

| 지표 | 기준 |
|------|------|
| `macro_f1` | ≥ 0.80 |
| `class0_fnr` | < 0.10 |
<<<<<<< HEAD
| `latency_p50_ms` | < 200 |

> 현재 공식 랭킹용 `*_test.csv`는 독립 정지사진 기반 landmark 세트이므로
> `fp_per_min`은 공식 비교에 사용하지 않는다. 필요하면 video / inference runtime 분석에서만 본다.
=======
| `fp_per_min` | < 2.0 |
| `latency_p95_ms` | < 200 |
>>>>>>> develop
