# JamJamBeat — 프로젝트 가이드 (Claude용)

## 프로젝트 개요

손 제스처 인식 PoC 시스템. MediaPipe 21-keypoint 랜드마크를 입력으로 받아 7개 제스처를 분류하는 모델 비교 파이프라인.

**7개 제스처 클래스 (class ID 순서 고정):**
```
0: neutral  1: fist  2: open_palm  3: V  4: pinky  5: animal  6: k-heart
```

---

## 핵심 디렉토리 구조

```
JamJamBeat/
├── model/
│   ├── data_fusion/           ← 전처리 스크립트 + 입력 CSV
│   ├── model_pipelines/       ← 파이프라인 핵심 (주 작업 영역)
│   │   ├── _shared.py         ← 공통 Dataset 클래스 + 상수 (모든 파이프라인이 임포트)
│   │   ├── run_pipeline.py    ← 단일 모델 실행기 (학습 → 평가 → 저장)
│   │   ├── run_all.py         ← 전체 모델 순차 실행기
│   │   ├── EVALUATION.md      ← 평가 지표 설명
│   │   ├── mlp_original/      ← 각 파이프라인: model.py + dataset.py
│   │   ├── mlp_baseline/
│   │   ├── mlp_baseline_seq8/
│   │   ├── mlp_sequence_joint/
│   │   ├── mlp_sequence_delta/
│   │   ├── mlp_temporal_pooling/
│   │   ├── mlp_embedding/
│   │   ├── cnn1d_tcn/
│   │   ├── transformer_embedding/
│   │   ├── mobilenetv3_small/
│   │   ├── shufflenetv2_x0_5/
│   │   └── efficientnet_b0/
│   ├── model_evaluation/
│   │   ├── 모델검증관련파일/
│   │   │   └── evaluation_runtime.py  ← 평가 함수 모음 (run_pipeline.py에서 호출)
│   │   └── pipelines/             ← 실험 결과 저장 (git-ignored)
│   └── cron/                  ← 크론 스크립트 (레거시, 현재는 run_all.py 사용 권장)
├── data/                      ← 원시 데이터 (landmark CSV 등)
├── frontend/                  ← 프론트엔드 코드
└── src/                       ← 기타 소스
```

---

## 모듈 간 호출 관계

### 전체 실행 흐름

```
[CLI 또는 run_all.py]
        │
        ▼ subprocess 호출 (--train-csv --val-csv --test-csv [--inference-csv])
run_pipeline.py  ──imports──▶  _shared.py
        │                          (JOINT_COLS, BONE_COLS, RAW_JOINT_COLS, SplitData,
        │                           FrameDataset, SequenceDataset, etc.)
        │
        ├── load_preprocessed_data() × 3  ← 학습데이터셋/{key}_{train|val|test}.csv 읽기
        │   (split_by_group()은 코드에 존재하지만 현재 run()에서 호출 안 함 —
        │    분리는 학습데이터셋/ CSV가 이미 source_file 단위로 완료된 상태)
        │
        ├── build_experiment()  ──importlib──▶  {model_id}/dataset.py
        │                                            │
        │                                            ├── imports: _shared.*
        │                                            ├── imports: .model.{ModelClass}
        │                                            └── returns: (model, mode, train_ds, val_ds, test_ds)
        │
        ├── train_one_epoch() / validate_one_epoch()
        ├── predict_dataset()  → preds_test.csv
        ├── predict_dataset()  → preds_inference.csv  (--inference-csv 지정 시)
        │
        └── evaluate_predictions()  ──imports──▶  evaluation_runtime.py
                                                      └── metrics, plots, JSON 저장
```

### run_all.py 흐름

```
run_all.py
  └── scan_dataset_registry(학습데이터셋/)  ← *_train.csv 자동 탐색
  └── subprocess.run([run_pipeline.py, --model-id, {model_id},
                      --train-csv, --val-csv, --test-csv, --inference-csv, ...])  × 12회 순차
        ↓ 각 실행 완료 후
  └── latest.json → run_summary.json 로드 → comparison_results.csv 생성
```

---

## 핵심 파일 역할

### `model/model_pipelines/_shared.py`
모든 파이프라인이 공통으로 임포트하는 파일. **절대 독립적으로 수정하면 안 됨** — 변경 시 전체 12개 파이프라인에 영향.

| 제공 항목 | 설명 |
|----------|------|
| `JOINT_COLS` | 정규화 좌표 63개 (`nx0,ny0,nz0,...`) |
| `BONE_COLS` | Bone vector 84개 (`bx0,by0,bz0,bl0,...`) |
| `SplitData` | train/val/test DataFrame을 묶는 dataclass |
| `detect_angle_cols(df)` | flex_*/abd_* 컬럼 자동 감지 |
| `frame_arrays(df, cols)` | (N, D) numpy 배열 반환 |
| `sequence_arrays(df, cols, seq_len, stride)` | (N, T, D) sliding window 배열 |
| `FrameDataset` | mode="frame" 모델용 |
| `TwoStreamDataset` | mode="two_stream" 모델용 |
| `SequenceDataset` | mode="sequence" 모델용 |
| `LandmarkImageDataset` | mode="image" 모델용 (스켈레톤 렌더링) |

### `model/model_pipelines/run_pipeline.py`
단일 모델 학습/평가 실행기.

- **입력**: `--train-csv --val-csv --test-csv [--inference-csv]` (역할별 CSV 명시 필수)
- `--train-csv`는 반드시 `{dataset_key}_train.csv` 형식이어야 함 (`infer_dataset_key()` 파싱)
- `--model-id` 에서 받은 이름으로 `importlib.import_module(f"{model_id}.dataset")` 동적 임포트
- `MODEL_CHOICES` 리스트: 허용 모델 목록 (새 파이프라인 추가 시 여기에 등록 필수)
- 주요 하이퍼파라미터 기본값: epochs=20, batch=32, lr=1e-3, patience=6, seq_len=8
- `forward_batch()`가 지원하는 mode: `"frame"` / `"sequence"` / `"image"` (three modes only)

### `model/model_pipelines/run_all.py`
전체 모델 순차 실행기.

- `학습데이터셋/` 디렉토리를 자동 스캔해 `*_train.csv` 기준으로 dataset registry 구성
- `CORE_MODELS` + `IMAGE_MODELS` = `ALL_MODELS` (새 파이프라인 추가 시 여기에도 등록 필수)
- 각 모델을 subprocess로 격리 실행 → GPU 메모리 완전 해제
- 결과: `model/model_evaluation/pipelines/{timestamp}__{dataset_key}/comparison_results.csv`

### `model/model_evaluation/모델검증관련파일/evaluation_runtime.py`
평가 함수 모음. `run_pipeline.py`에서만 호출됨. **직접 실행 불필요**.

---

## 새 파이프라인 추가 시 체크리스트

1. `model/model_pipelines/{model_id}/` 디렉토리 생성
2. `__init__.py` (빈 파일)
3. `model.py` — `nn.Module` 서브클래스
4. `dataset.py` — 반드시 아래 시그니처의 `build()` 함수 포함:
   ```python
   def build(split, angle_cols, seq_len, seq_stride, image_size, num_classes)
       -> (model, mode, train_ds, val_ds, test_ds)
   ```
   `mode` 값: `"frame"` / `"sequence"` / `"image"` 중 하나
   (`"two_stream"` 은 `forward_batch()`에 미구현 — 사용 불가)
5. `run_pipeline.py`의 `MODEL_CHOICES`에 추가
6. `run_all.py`의 `CORE_MODELS` 또는 `IMAGE_MODELS`에 추가

---

## 입력 데이터 규격

**학습용 역할 CSV 경로 (`model/data_fusion/학습데이터셋/`):**
```
{dataset_key}_train.csv      ← --train-csv
{dataset_key}_val.csv        ← --val-csv
{dataset_key}_test.csv       ← --test-csv
{dataset_key}_inference.csv  ← --inference-csv (선택)
```

dataset_key 예시: `baseline`, `pos_only`, `scale_only`, `pos_scale`,
`baseline_ds_4_none`, `pos_only_ds_1_pos`, ...

**학습 피처: RAW_JOINT_COLS (raw x/y/z, 63d)**
```
RAW_JOINT_COLS: x0,y0,z0,...,x20,y20,z20  (63d, MediaPipe 원본 좌표)
```
모든 모델이 `RAW_JOINT_COLS`를 기본 입력으로 사용한다.
`JOINT_COLS`(nx*/ny*/nz*, 정규화), `BONE_COLS`(bx*/by*/bz*/bl*)은 일부 모델에서 추가 사용.

**전체 피처 차원 (full feature set 모델용):**
```
RAW_JOINT_COLS: 63d  (21 joints × x/y/z, raw)
BONE_COLS:      84d  (21 bones × bx/by/bz/bl)
angle_cols:      9d  (flex_thumb~pinky + abd × 4)
합계:           156d
```

**variant별 좌표 변환 (학습데이터셋 생성 시 적용됨):**
```
baseline  : 변환 없음 (raw MediaPipe 좌표)
pos_only  : pts - pts[0]  (손목 기준 이동)
scale_only: pts / ‖pts[9]-pts[0]‖  (중지MCP-손목 거리로 스케일)
pos_scale : (pts - pts[0]) / ‖pts[9]-pts[0]‖  (이동 + 스케일)
```

---

## 실험 결과 출력 구조

```
model/model_evaluation/pipelines/
└── {timestamp}__{dataset_key}/        ← run_all.py 실행 단위 suite
    └── {model_id}/
        ├── latest.json                ← 최신 실험 경로 포인터
        └── {yyyymmdd_HHMMSS}/
            ├── model.pt               ← 학습된 가중치 (+ class_names, mode, seq_len 등 메타)
            ├── preds_test.csv         ← test split 예측 결과
            ├── preds_inference.csv    ← inference split 예측 결과 (--inference-csv 지정 시)
            ├── train_history.csv      ← epoch별 loss/acc
            ├── run_summary.json       ← 전체 메트릭 + 메타데이터
            └── evaluation/
                ├── metrics_summary.json
                ├── confusion_matrix.csv / .png
                ├── per_class_report.csv
                └── latency_cdf.png
```

---

## 12개 파이프라인 현황

| model_id | mode | 입력 차원 | 설명 |
|----------|------|----------|------|
| `mlp_original` | frame | 63d | 초기 GestureMLP 구현체 (raw 63d) |
| `mlp_baseline` | frame | 63d | 기준선 MLP |
| `mlp_baseline_seq8` | sequence | 8×63→504 | 동일 아키텍처, T=8 temporal context 효과 검증 |
| `mlp_sequence_joint` | sequence | T×63 | MLP + flatten (큰 hidden dim) |
| `mlp_sequence_delta` | sequence | T×126 | joint + 1차 차분 |
| `mlp_temporal_pooling` | sequence | T×63 | mean/max/std pooling |
| `mlp_embedding` | frame | 156d | LayerNorm+GELU embedding head |
| `cnn1d_tcn` | sequence | T×156 | Temporal Convolutional Network |
| `transformer_embedding` | sequence | T×156 | Transformer encoder |
| `mobilenetv3_small` | image | 1×96×96 | 스켈레톤 이미지 → CNN |
| `shufflenetv2_x0_5` | image | 1×96×96 | 스켈레톤 이미지 → CNN |
| `efficientnet_b0` | image | 1×96×96 | 스켈레톤 이미지 → CNN |

> `mlp_baseline_full`, `two_stream_mlp`은 현재 미구현 (코드 없음, MODEL_CHOICES 미등록)

---

## 주요 실행 명령

```bash
# 가상환경 활성화
source model/.venv/bin/activate

# 단일 모델 실행 (역할 CSV 명시 필수)
cd /home/user/projects/JamJamBeat
uv run python model/model_pipelines/run_pipeline.py \
  --model-id mlp_baseline \
  --train-csv model/data_fusion/학습데이터셋/baseline_train.csv \
  --val-csv   model/data_fusion/학습데이터셋/baseline_val.csv \
  --test-csv  model/data_fusion/학습데이터셋/baseline_test.csv \
  --inference-csv model/data_fusion/학습데이터셋/baseline_inference.csv

# 전체 모델 순차 실행 (학습데이터셋/ 자동 스캔)
uv run python model/model_pipelines/run_all.py

# 특정 dataset으로 실행
uv run python model/model_pipelines/run_all.py --dataset-key pos_only

# 일부 모델만
uv run python model/model_pipelines/run_all.py --models mlp_baseline mlp_baseline_seq8

# 결과 확인
ls model/model_evaluation/pipelines/
```

---

## 문서 작성 규칙

모델 관련 문서는 아래 두 파일로 관리한다. 변경사항 발생 시 반드시 해당 파일에 반영한다.

| 파일 | 경로 | 작성 대상 |
|------|------|----------|
| `model_comparison_total.md` | `model/모델비교관련문서/model_comparison_total.md` | **파이프라인 전체 구조 변경** 시 업데이트 — 실험 매트릭스, 피처 체계, 평가 기준, 모델군 추가/제거, 실험 순서 변경 등 |
| `model_comparison_mlp.md` | `model/모델비교관련문서/model_comparison_mlp.md` | **MLP 계열 모델 정보** 업데이트 — mlp_baseline, mlp_embedding, mlp_sequence_*, mlp_temporal_pooling, mlp_baseline_full, mlp_baseline_seq8 관련 실험 결과, 해석, 구조 변경 |

기타 참고 문서:
- `model/모델비교관련문서/model_comparison_v1.md` — 구버전 비교 (읽기 전용)
- `model/모델비교관련문서/모델비교문서.pdf` — 초기 설계 문서 (읽기 전용)

---

## PoC 수용 기준

```
macro_f1     ≥ 0.80   → 양호
class0_fnr   < 0.10   → neutral 오발동 수용
fp_per_min   < 2.0    → 실서비스 수용
latency_p95  < 200ms  → 온디바이스 배포 가능
```

---

## 개발 주의사항

- **`_shared.py` 수정 시**: 모든 12개 파이프라인에 영향 → backward compatibility 반드시 확인
- **새 파이프라인 추가 시**: `run_pipeline.py:MODEL_CHOICES` + `run_all.py:CORE_MODELS` 또는 `IMAGE_MODELS` 양쪽 등록 필수
- **`dataset.py`의 `build()` 반환 타입**: `(nn.Module, str, Dataset, Dataset, Dataset)` 순서 고정
- **mode 문자열**: `"frame"` / `"sequence"` / `"image"` 외 값 사용 불가 (`forward_batch`에서 ValueError) — `"two_stream"` 미구현
- **train CSV 파일명 규칙**: `{dataset_key}_train.csv` 형식 필수 (`infer_dataset_key()` 파싱 기준)
- **sequence_arrays**: 레이블은 윈도우 마지막 프레임 기준 (`ys[e-1]`) — 경계 구간 노이즈 존재
- **venv 경로**: `model/.venv/` — model_pipelines와 model_evaluation venv를 하나로 통합. `model/pyproject.toml`로 관리
