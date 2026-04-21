#!/usr/bin/env bash
set -euo pipefail

# JamJamBeat v2 9개 모델 파이프라인 cron 파일 생성기
# 참고: model/model_comparison_v2.md
#
# 생성만 수행합니다. 설치는 install_cron_jobs.sh에서 수행하세요.

PROJECT_ROOT="${PROJECT_ROOT:-/home/user/projects/JamJamBeat}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
RUNNER="${RUNNER:-$PROJECT_ROOT/model/model_pipelines/run_pipeline.py}"
CRON_FILE="${CRON_FILE:-$PROJECT_ROOT/model/cron/jamjambeat_9jobs.crontab}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/model/cron/logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/model/model_evaluation/pipelines}"

# 학습 파라미터 오버라이드
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"

# 스케줄 파라미터 (매일 실행)
START_HOUR="${START_HOUR:-0}"
START_MINUTE="${START_MINUTE:-5}"
INTERVAL_MIN="${INTERVAL_MIN:-75}"

DATA_1="${PROJECT_ROOT}/model/data_fusion/man1_right_for_poc_output.csv"
DATA_2="${PROJECT_ROOT}/model/data_fusion/man2_right_for_poc_output.csv"
DATA_3="${PROJECT_ROOT}/model/data_fusion/man3_right_for_poc_output.csv"
DATA_4="${PROJECT_ROOT}/model/data_fusion/woman1_right_for_poc_output.csv"

MODELS=(
  "mlp_baseline"
  "mlp_embedding"
  "two_stream_mlp"
  "cnn1d_tcn"
  "transformer_embedding"
  "mediapipe_hand_landmarker"
  "mobilenetv3_small"
  "shufflenetv2_x0_5"
  "efficientnet_b0"
)

# 결과 경로와 로그 경로를 먼저 만들어 두고 이후 cron line을 deterministic하게 생성한다.
mkdir -p "$(dirname "$CRON_FILE")"
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_ROOT"

# runner / 입력 CSV가 없으면 잘못된 cron만 만들게 되므로 생성 전에 바로 중단한다.
for req in "$RUNNER" "$DATA_1" "$DATA_2" "$DATA_3" "$DATA_4"; do
  if [[ ! -e "$req" ]]; then
    echo "[ERROR] Required file not found: $req" >&2
    exit 1
  fi
done

{
  echo "# ------------------------------------------------------------------"
  echo "# JamJamBeat model comparison v2 (9 jobs)"
  echo "# Generated: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "# ------------------------------------------------------------------"
  echo "SHELL=/bin/bash"
  echo "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  echo

  minute=$START_MINUTE
  hour=$START_HOUR

  for model_id in "${MODELS[@]}"; do
    # interval 간격으로 모델별 실행 시각을 순차 배치한다.
    m=$((minute % 60))
    h=$((hour + (minute / 60)))
    h=$((h % 24))

    lock_file="/tmp/jamjambeat_${model_id}.lock"
    log_file="$LOG_DIR/${model_id}.log"

    # flock 으로 동일 모델 중복 실행을 막고, 로그는 모델별 파일로 분리한다.
    echo "$m $h * * * cd $PROJECT_ROOT && /usr/bin/flock -n $lock_file $PYTHON_BIN $RUNNER --model-id $model_id --csv-path $DATA_1 --csv-path $DATA_2 --csv-path $DATA_3 --csv-path $DATA_4 --output-root $OUTPUT_ROOT --epochs $EPOCHS --batch-size $BATCH_SIZE >> $log_file 2>&1"

    minute=$((minute + INTERVAL_MIN))
  done

  echo
  echo "# 설치: $PROJECT_ROOT/model/cron/install_cron_jobs.sh"
  echo "# 확인: crontab -l"
} > "$CRON_FILE"

echo "[OK] Generated crontab file: $CRON_FILE"
sed -n '1,200p' "$CRON_FILE"
