# 모델별 영상 체크 도구

학습된 모델 체크포인트를 로드해 실제 영상에 추론하고 결과를 시각화하는 도구 모음.

---

## 파일 목록

| 파일 | 용도 |
|------|------|
| `video_check_app.py` | UI 드롭다운 / 직접 지정으로 실시간 추론 재생 |
| `error_frame_viewer.py` | ground truth와 비교해 **틀린 프레임만** 추출·영상 저장 |

---

## 1. video_check_app.py — 실시간 추론 뷰어

학습된 run과 영상을 선택해 프레임별 예측을 오버레이로 보여준다.

### 실행

```bash
# model/ 디렉토리 기준
# UI 모드 (드롭다운으로 run/영상 선택)
uv run python "model_evaluation/모델별영상체크/video_check_app.py"

# 직접 지정 모드
uv run python "model_evaluation/모델별영상체크/video_check_app.py" \
  --run-dir model_evaluation/pipelines/{suite_name}/mlp_baseline \
  --video ../data/raw_data/4_slow_right_man3.mp4
```

`--run-dir`에는 실제 timestamp 폴더 또는 `latest.json`이 있는 model 폴더 둘 다 사용 가능.

### 재생 컨트롤

| 키 | 동작 |
|----|------|
| `Space` | 일시정지 / 재생 |
| `A` / `←` | 이전 프레임 |
| `D` / `→` | 다음 프레임 |
| `R` | 처음부터 재시작 |
| `Q` / `Esc` | 종료 |

---

## 2. error_frame_viewer.py — 오류 프레임 추출기

ground truth CSV와 모델 추론을 프레임 단위로 비교해, **예측이 틀린 프레임만** 오버레이 영상으로 저장한다.

### 입력

| 항목 | 설명 |
|------|------|
| `--run-dir` | 학습된 run 폴더 (model.pt 위치 또는 latest.json 있는 model 폴더) |
| `--csv` | ground truth CSV (`source_file`, `frame_idx`, `gesture` 컬럼 필수). 반복 사용 가능 |

**ground truth CSV 예시:**
```
data_fusion/man1_right_for_poc.csv
data_fusion/man2_right_for_poc.csv
data_fusion/man3_right_for_poc.csv
data_fusion/woman1_right_for_poc.csv
```
각 CSV는 `source_file` 컬럼으로 영상 파일명과 매핑된다 (`0_hardneg_right_man1` → `0_hardneg_right_man1.mp4`).

### 실행 예시

```bash
# model/ 디렉토리 기준

# 단일 CSV, 기본 설정
uv run python "model_evaluation/모델별영상체크/error_frame_viewer.py" \
  --run-dir model_evaluation/pipelines/mlp_baseline/20260313_120557 \
  --csv data_fusion/man1_right_for_poc.csv

# 4개 CSV 전체 분석 + 오류 앞뒤 5프레임 context 포함
uv run python "model_evaluation/모델별영상체크/error_frame_viewer.py" \
  --run-dir model_evaluation/pipelines/mlp_baseline/20260313_120557 \
  --csv data_fusion/man1_right_for_poc.csv \
  --csv data_fusion/man2_right_for_poc.csv \
  --csv data_fusion/man3_right_for_poc.csv \
  --csv data_fusion/woman1_right_for_poc.csv \
  --context-frames 5

# latest.json 포인터 사용 (가장 최근 run 자동 선택)
uv run python "model_evaluation/모델별영상체크/error_frame_viewer.py" \
  --run-dir model_evaluation/pipelines/mlp_baseline \
  --csv data_fusion/man1_right_for_poc.csv

# 특정 source_file만 분석
uv run python "model_evaluation/모델별영상체크/error_frame_viewer.py" \
  --run-dir model_evaluation/pipelines/mlp_baseline/20260313_120557 \
  --csv data_fusion/man1_right_for_poc.csv \
  --source-filter 3_fast_right_man1 3_slow_right_man1
```

### 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--run-dir` | (필수) | model.pt 위치 또는 latest.json 있는 폴더 |
| `--csv` | (필수) | ground truth CSV. `--csv a.csv --csv b.csv` 반복 가능 |
| `--output-dir` | `run_dir/error_analysis/` | 출력 저장 폴더 |
| `--video-root` | `data/raw_data/` | 영상 파일이 있는 폴더 |
| `--context-frames` | `0` | 오류 프레임 앞뒤로 추가할 프레임 수 |
| `--source-filter` | 전체 | 분석할 source_file 이름 목록 |
| `--fps` | `10.0` | 출력 영상 FPS |

### 출력

```
run_dir/error_analysis/
├── error_frames_{model_id}.mp4   ← 오류 프레임 모음 영상
└── error_summary.csv             ← 오류 프레임 요약표
```

**error_summary.csv 컬럼:**

| 컬럼 | 설명 |
|------|------|
| `source_file` | 영상 파일명 (확장자 제외) |
| `frame_idx` | 영상 내 프레임 번호 |
| `gt` | ground truth 제스처 이름 |
| `pred` | 모델 예측 제스처 이름 |
| `confidence` | 예측 확률 |
| `is_context` | `True` = 주변 context 프레임, `False` = 실제 오류 프레임 |

**영상 오버레이:**
- 빨간 테두리 = 실제 오류 프레임
- 노란 테두리 = context 프레임 (주변)
- 상단: source_file / frame_idx / timestamp
- 하단 좌: `GT: {클래스명}` (색상 코딩)
- 하단 우: `PRED: {클래스명} (확률)`
- 우측 상단: 전체 클래스 확률 바
- 랜드마크 스켈레톤 오버레이

---

## 공통 사전 조건

- 프로젝트 루트에 `hand_landmarker.task` 파일 필요
- PyTorch 추론은 CPU 고정
- sequence 모델(`cnn1d_tcn`, `transformer_embedding` 등)은 초반 `seq_len` 프레임이 warmup 구간 — 해당 프레임은 neutral로 처리됨
