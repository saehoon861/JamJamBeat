# 모델별 영상 체크 도구

현재 파이프라인 기준 기본 도구는 `train_aligned` 계열이다.  
학습 run이 어떤 dataset variant(`baseline / pos_only / scale_only / pos_scale`)로 만들어졌는지 읽어서, 영상/이미지 추론 때 같은 좌표계로 맞춰본다.

관련 문서:
- 전체 평가 구조: [EVALUATION.md](../../model_pipelines/EVALUATION.md)
- 평가 산출물 읽는 법: [evaluation_guide.md](../모델검증관련파일/evaluation_guide.md)

## 파일 목록

| 파일 | 용도 | 권장 여부 |
|------|------|-----------|
| `video_check_app_train_aligned.py` | 현재 파이프라인 기준 기본 영상 뷰어 | 기본 |
| `video_check_app_train_aligned_export.py` | 위 뷰어 오버레이를 MP4로 저장 | 기본 |
| `image_check_app_train_aligned.py` | 이미지 폴더/zip landmark 추론 + 평가 산출물 생성 | 기본 |
| `error_frame_viewer.py` | GT CSV와 비교해 틀린 프레임만 분석 | 기본 |
| `video_check_app.py` | 저수준 코어/호환용 viewer | 보조 |

## run 경로 규칙

`--run-dir`는 아래 네 형태를 모두 지원한다.

- direct run: `model/model_evaluation/pipelines/{model_id}/{run_id}/`
- direct latest pointer: `model/model_evaluation/pipelines/{model_id}/latest.json`
- suite run: `model/model_evaluation/pipelines/{suite_name}/{model_id}/{run_id}/`
- suite latest pointer: `model/model_evaluation/pipelines/{suite_name}/{model_id}/latest.json`

공식 GT CSV 예시는 아래를 우선 사용한다.

- `model/data_fusion/학습데이터셋/*_test.csv`
- `model/data_fusion/학습데이터셋/*_inference.csv`

이 CSV들은 이미 `source_file`, `frame_idx`, `gesture`를 포함하므로 `error_frame_viewer.py`에서 바로 사용할 수 있다.

## 1. 기본 영상 체크

현재 기본 viewer는 `video_check_app_train_aligned.py`다.

UI 모드에서는 raw 영상 드롭다운은 계속 전체를 보여주고, 화면 아래 `Inference Videos` 패널에서
선택한 run의 hold-out inference 영상 목록과 현재 선택 영상이 그 split에 포함되는지 여부를 같이 안내한다.

```bash
# model/ 디렉토리 기준
uv run python "model_evaluation/모델별영상체크/video_check_app_train_aligned.py"
```

```bash
# direct run 폴더 지정
uv run python "model_evaluation/모델별영상체크/video_check_app_train_aligned.py" \
  --run-dir model_evaluation/pipelines/mlp_baseline/20260318_152800 \
  --video ../data/raw_data/4_slow_right_man3.mp4
```

```bash
# suite 안의 latest.json 포인터 사용
uv run python "model_evaluation/모델별영상체크/video_check_app_train_aligned.py" \
  --run-dir model_evaluation/pipelines/{suite_name}/mlp_baseline \
  --video ../data/raw_data/4_slow_right_man3.mp4
```

재생 전 영상 전체를 먼저 분석하고, sequence 모델은 초반 `seq_len` 프레임에서 warmup 상태가 나타날 수 있다.

## 2. 영상 export

뷰어와 같은 오버레이를 MP4로 저장한다.

```bash
uv run python "model_evaluation/모델별영상체크/video_check_app_train_aligned_export.py" \
  --run-dir model_evaluation/pipelines/mlp_baseline/20260318_152800 \
  --video ../data/raw_data/4_slow_right_man3.mp4
```

출력 파일을 직접 지정하지 않으면 원본 영상 옆에 `__{model_id}__annotated.mp4` 형식으로 저장된다.

## 3. 이미지셋 추론 체크

`추론용데이터셋` 같은 라벨된 이미지 폴더나 zip을 읽어 추론하고, `preds_images.csv`와 `evaluation/` 산출물을 만든다.

```bash
uv run python "model_evaluation/모델별영상체크/image_check_app_train_aligned.py" \
  --run-dir model_evaluation/pipelines/mlp_baseline/20260318_152800 \
  --images-root data_fusion/추론용데이터셋
```

sequence 모델을 이미지셋에 넣을 때는 기본적으로 `independent` 모드를 쓴다.

```bash
uv run python "model_evaluation/모델별영상체크/image_check_app_train_aligned.py" \
  --run-dir model_evaluation/pipelines/mlp_baseline/20260318_152800 \
  --images-root data_fusion/추론용데이터셋 \
  --image-sequence-mode independent
```

기본 출력 위치:

```text
run_dir/image_inference/{dataset_name}/
├── preds_images.csv
├── dataset_info.json
├── inference_summary.json
└── evaluation/
```

## 4. 오류 프레임 분석

`error_frame_viewer.py`는 현재 `train_aligned` 런타임을 사용한다.  
즉 non-baseline run도 `dataset_variant`를 맞춘 상태로 GT와 비교한다.

```bash
# UI 모드
uv run python "model_evaluation/모델별영상체크/error_frame_viewer.py"
```

```bash
# 공식 split CSV를 그대로 사용
uv run python "model_evaluation/모델별영상체크/error_frame_viewer.py" \
  --run-dir model_evaluation/pipelines/mlp_baseline/20260318_152800 \
  --csv data_fusion/학습데이터셋/baseline_test.csv
```

```bash
# hold-out inference split 분석
uv run python "model_evaluation/모델별영상체크/error_frame_viewer.py" \
  --run-dir model_evaluation/pipelines/mlp_baseline \
  --csv data_fusion/학습데이터셋/baseline_inference.csv \
  --context-frames 5
```

```bash
# 공식 split CSV 내부 source_file만 골라서 분석
uv run python "model_evaluation/모델별영상체크/error_frame_viewer.py" \
  --run-dir model_evaluation/pipelines/mlp_baseline/20260318_152800 \
  --csv data_fusion/학습데이터셋/baseline_test.csv \
  --source-filter 3_fast_right_man1 3_slow_right_man1
```

기본 출력:

```text
run_dir/error_analysis/
├── error_frames_{model_id}.mp4
└── error_summary.csv
```

UI에서는 `*_test.csv`, `*_inference.csv`를 우선 탐색하고, 예전 POC CSV는 보조 호환 입력으로만 남겨둔다.

## 5. 코어/호환 viewer

`video_check_app.py`는 저수준 코어 viewer다.  
현재 구조에서는 `video_check_app_train_aligned.py`를 먼저 쓰고, 호환 확인이나 내부 동작 점검이 필요할 때만 직접 사용하면 된다.

## 공통 사전 조건

- 프로젝트 루트에 `hand_landmarker.task` 파일 필요
- viewer 계열 추론은 CPU 기준
- `source_file`은 `data/raw_data/{source_file}.mp4`와 연결된다
- 공식 split CSV는 `source_file`, `frame_idx`, `gesture`를 포함해야 한다
