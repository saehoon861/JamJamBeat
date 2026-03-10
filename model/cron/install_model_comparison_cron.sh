#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# JamJamBeat v2 model comparison cron installer
# -----------------------------------------------------------------------------
# Creates 9 cron jobs (one job per model pipeline from model_comparison_v2.md):
#   1) mlp_baseline
#   2) mlp_embedding
#   3) two_stream_mlp
#   4) cnn1d_tcn
#   5) transformer_embedding
#   6) mediapipe_hand_landmarker
#   7) mobilenetv3_small
#   8) shufflenetv2_x0_5
#   9) efficientnet_b0
#
# Usage:
#   bash model/cron/install_model_comparison_cron.sh
#
# Optional env overrides:
#   PROJECT_ROOT=/home/user/projects/JamJamBeat
#   PYTHON_BIN=/home/user/projects/JamJamBeat/.venv/bin/python
#   OUTPUT_ROOT=/home/user/projects/JamJamBeat/model/model_evaluation/runs
#   EPOCHS=20
#   BATCH_SIZE=128
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="/home/user/projects/JamJamBeat"

if [[ -d "${DEFAULT_PROJECT_ROOT}" ]]; then
  PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
else
  PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
fi

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
RUNNER="${PROJECT_ROOT}/model/cron/run_model_pipeline.py"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/model/model_evaluation/runs}"
LOG_DIR="${PROJECT_ROOT}/model/cron/logs"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"

DATA_1="${PROJECT_ROOT}/model/data_fusion/man1_right_for_poc_output.csv"
DATA_2="${PROJECT_ROOT}/model/data_fusion/man2_right_for_poc_output.csv"
DATA_3="${PROJECT_ROOT}/model/data_fusion/man3_right_for_poc_output.csv"
DATA_4="${PROJECT_ROOT}/model/data_fusion/woman1_right_for_poc_output.csv"

for req in "$PYTHON_BIN" "$RUNNER" "$DATA_1" "$DATA_2" "$DATA_3" "$DATA_4"; do
  if [[ ! -e "$req" ]]; then
    echo "[ERROR] Required file not found: $req" >&2
    exit 1
  fi
done

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_ROOT"

START_MARK="# >>> JAMJAMBEAT_MODEL_COMPARISON_CRON >>>"
END_MARK="# <<< JAMJAMBEAT_MODEL_COMPARISON_CRON <<<"

TMP_OLD="$(mktemp)"
TMP_CLEAN="$(mktemp)"
TMP_NEW="$(mktemp)"

crontab -l > "$TMP_OLD" 2>/dev/null || true

# Remove old managed block (if exists)
awk -v start="$START_MARK" -v end="$END_MARK" '
  $0 == start {skip=1; next}
  $0 == end   {skip=0; next}
  skip != 1   {print}
' "$TMP_OLD" > "$TMP_CLEAN"

# Build managed block
{
  cat "$TMP_CLEAN"
  echo "$START_MARK"
  echo "# Auto-generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "# JamJamBeat model comparison v2 (9 pipelines)"

  # Daily staggered schedule (UTC)
  # minute hour day month weekday command
  echo "5 0 * * * cd ${PROJECT_ROOT} && ${PYTHON_BIN} ${RUNNER} --model-id mlp_baseline --csv-path ${DATA_1} --csv-path ${DATA_2} --csv-path ${DATA_3} --csv-path ${DATA_4} --output-root ${OUTPUT_ROOT} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} >> ${LOG_DIR}/mlp_baseline.log 2>&1"
  echo "20 1 * * * cd ${PROJECT_ROOT} && ${PYTHON_BIN} ${RUNNER} --model-id mlp_embedding --csv-path ${DATA_1} --csv-path ${DATA_2} --csv-path ${DATA_3} --csv-path ${DATA_4} --output-root ${OUTPUT_ROOT} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} >> ${LOG_DIR}/mlp_embedding.log 2>&1"
  echo "35 2 * * * cd ${PROJECT_ROOT} && ${PYTHON_BIN} ${RUNNER} --model-id two_stream_mlp --csv-path ${DATA_1} --csv-path ${DATA_2} --csv-path ${DATA_3} --csv-path ${DATA_4} --output-root ${OUTPUT_ROOT} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} >> ${LOG_DIR}/two_stream_mlp.log 2>&1"
  echo "50 3 * * * cd ${PROJECT_ROOT} && ${PYTHON_BIN} ${RUNNER} --model-id cnn1d_tcn --csv-path ${DATA_1} --csv-path ${DATA_2} --csv-path ${DATA_3} --csv-path ${DATA_4} --output-root ${OUTPUT_ROOT} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} >> ${LOG_DIR}/cnn1d_tcn.log 2>&1"
  echo "5 5 * * * cd ${PROJECT_ROOT} && ${PYTHON_BIN} ${RUNNER} --model-id transformer_embedding --csv-path ${DATA_1} --csv-path ${DATA_2} --csv-path ${DATA_3} --csv-path ${DATA_4} --output-root ${OUTPUT_ROOT} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} >> ${LOG_DIR}/transformer_embedding.log 2>&1"
  echo "20 6 * * * cd ${PROJECT_ROOT} && ${PYTHON_BIN} ${RUNNER} --model-id mediapipe_hand_landmarker --csv-path ${DATA_1} --csv-path ${DATA_2} --csv-path ${DATA_3} --csv-path ${DATA_4} --output-root ${OUTPUT_ROOT} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} >> ${LOG_DIR}/mediapipe_hand_landmarker.log 2>&1"
  echo "35 7 * * * cd ${PROJECT_ROOT} && ${PYTHON_BIN} ${RUNNER} --model-id mobilenetv3_small --csv-path ${DATA_1} --csv-path ${DATA_2} --csv-path ${DATA_3} --csv-path ${DATA_4} --output-root ${OUTPUT_ROOT} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} >> ${LOG_DIR}/mobilenetv3_small.log 2>&1"
  echo "50 8 * * * cd ${PROJECT_ROOT} && ${PYTHON_BIN} ${RUNNER} --model-id shufflenetv2_x0_5 --csv-path ${DATA_1} --csv-path ${DATA_2} --csv-path ${DATA_3} --csv-path ${DATA_4} --output-root ${OUTPUT_ROOT} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} >> ${LOG_DIR}/shufflenetv2_x0_5.log 2>&1"
  echo "5 10 * * * cd ${PROJECT_ROOT} && ${PYTHON_BIN} ${RUNNER} --model-id efficientnet_b0 --csv-path ${DATA_1} --csv-path ${DATA_2} --csv-path ${DATA_3} --csv-path ${DATA_4} --output-root ${OUTPUT_ROOT} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} >> ${LOG_DIR}/efficientnet_b0.log 2>&1"

  echo "$END_MARK"
} > "$TMP_NEW"

crontab "$TMP_NEW"

rm -f "$TMP_OLD" "$TMP_CLEAN" "$TMP_NEW"

echo "[OK] Installed JamJamBeat model comparison cron jobs."
echo "[INFO] Project root: $PROJECT_ROOT"
echo "[INFO] Logs: $LOG_DIR"
echo "[INFO] Results: $OUTPUT_ROOT"
