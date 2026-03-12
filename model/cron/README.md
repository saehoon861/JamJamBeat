# JamJamBeat Model Comparison Cron

`model_comparison_v2.md` 기준 8개 파이프라인을 cron으로 분리 실행하는 스크립트 모음입니다.

## 구성 파일

- `run_model_pipeline.py`  
  단일 모델 파이프라인 실행 + 평가 지표 산출
- `install_model_comparison_cron.sh`  
  8개 cron job 설치
- `uninstall_model_comparison_cron.sh`  
  설치된 cron block 제거

## 기본 입력 데이터

다음 4개 전처리 CSV(`*_output.csv`)를 기본 사용합니다.

- `model/data_fusion/man1_right_for_poc_output.csv`
- `model/data_fusion/man2_right_for_poc_output.csv`
- `model/data_fusion/man3_right_for_poc_output.csv`
- `model/data_fusion/woman1_right_for_poc_output.csv`

## 실행 모델 (8개)

1. `mlp_baseline`
2. `mlp_embedding`
3. `two_stream_mlp`
4. `cnn1d_tcn`
5. `transformer_embedding`
6. `mobilenetv3_small`
7. `shufflenetv2_x0_5`
8. `efficientnet_b0`

## 평가 출력

각 실행 결과는 기본적으로 아래 경로에 저장됩니다.

- `model/model_evaluation/pipelines/<model_id>/<timestamp>/`

포함 파일:

- `preds_test.csv`
- `train_history.csv`
- `model.pt`
- `run_summary.json`
- `evaluation/metrics_summary.json`
- `evaluation/confusion_matrix.csv`
- `evaluation/confusion_matrix.png`
- `evaluation/per_class_report.csv`
- `evaluation/latency_cdf.png`

## cron 설치/삭제

```bash
# 설치
bash model/cron/install_model_comparison_cron.sh

# 제거
bash model/cron/uninstall_model_comparison_cron.sh
```

## 의존성

- Python 3.11+
- `torch`
- `pandas`
- `numpy`
- `matplotlib`
- `Pillow`
