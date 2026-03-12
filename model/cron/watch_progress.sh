#!/usr/bin/env bash
# watch_progress.sh - 모델 파이프라인 진행 상황을 2분마다 출력
#
# Usage:
#   bash model/cron/watch_progress.sh
#
# Optional env overrides:
#   PROJECT_ROOT=/home/user/projects/JamJamBeat
#   INTERVAL=120   (초 단위, 기본 2분)

set -euo pipefail

INTERVAL="${INTERVAL:-120}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/user/projects/JamJamBeat}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/model/cron/logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/model/model_evaluation/pipelines}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

MODELS=(
  "mlp_baseline"
  "mlp_embedding"
  "two_stream_mlp"
  "cnn1d_tcn"
  "transformer_embedding"
  "mobilenetv3_small"
  "shufflenetv2_x0_5"
  "efficientnet_b0"
)

# 상태 판정은 latest.json, lock file, log tail 순으로 확인해 DONE/RUNNING/ERROR/PENDING을 구분한다.
status_of() {
  local model_id="$1"
  local lock_file="/tmp/jamjambeat_${model_id}.lock"
  local log_file="$LOG_DIR/${model_id}.log"
  local latest_json="$OUTPUT_ROOT/${model_id}/latest.json"

  # latest.json이 있으면 가장 신뢰도 높은 완료 신호로 본다.
  if [[ -f "$latest_json" ]]; then
    local run_dir=""
    run_dir=$("$PYTHON_BIN" -c "
import json, sys
try:
    d = json.load(open('$latest_json'))
    print(d.get('latest_run', ''))
except Exception:
    print('')
" 2>/dev/null || true)

    if [[ -n "$run_dir" && -f "${run_dir}/run_summary.json" ]]; then
      local f1_str=""
      f1_str=$("$PYTHON_BIN" -c "
import json
try:
    s = json.load(open('${run_dir}/run_summary.json'))
    m = s.get('metrics', {})
    val = m.get('macro_f1', m.get('f1', None))
    if val is not None:
        print(f'macro_f1={float(val):.4f}')
    else:
        print('macro_f1=N/A')
except Exception:
    print('')
" 2>/dev/null || true)
      echo "DONE   ${f1_str}"
      return
    fi
  fi

  # lock file이 있으면 현재 cron이 실행 중인 것으로 간주한다.
  if [[ -f "$lock_file" ]]; then
    local epoch_info=""
    if [[ -f "$log_file" ]]; then
      # "epoch 012/020 train_loss=..." 패턴에서 진행 epoch 추출
      epoch_info=$(grep -oE 'epoch [0-9]+/[0-9]+' "$log_file" 2>/dev/null | tail -1 || true)
    fi
    if [[ -n "$epoch_info" ]]; then
      echo "RUNNING  (${epoch_info})"
    else
      echo "RUNNING  (starting...)"
    fi
    return
  fi

  # lock은 없고 로그만 있으면 직전 실행이 끝난 상태이므로 마지막 로그 줄로 상태를 추정한다.
  if [[ -f "$log_file" && -s "$log_file" ]]; then
    local last_line=""
    last_line=$(tail -1 "$log_file" 2>/dev/null || true)
    if echo "$last_line" | grep -qi "error\|traceback\|exception"; then
      echo "ERROR    ($(echo "$last_line" | cut -c1-40))"
    elif echo "$last_line" | grep -qi "done\.\|output:"; then
      echo "ENDED    (summary missing — check log)"
    else
      echo "ENDED    ($(echo "$last_line" | cut -c1-40))"
    fi
    return
  fi

  # 아무 흔적도 없으면 아직 스케줄이 돌지 않은 상태다.
  echo "PENDING"
}

# 완료된 모델은 latest run의 macro_f1을 바로 읽어 간단한 progress bar로 보여준다.
get_f1_bar() {
  local model_id="$1"
  local latest_json="$OUTPUT_ROOT/${model_id}/latest.json"
  [[ -f "$latest_json" ]] || return

  "$PYTHON_BIN" -c "
import json
try:
    d = json.load(open('$latest_json'))
    run_dir = d.get('latest_run', '')
    if not run_dir:
        raise ValueError
    s = json.load(open(run_dir + '/run_summary.json'))
    val = s.get('metrics', {}).get('macro_f1', None)
    if val is None:
        raise ValueError
    f = float(val)
    bars = int(f * 20)
    print(f'  F1  [' + '█' * bars + '░' * (20 - bars) + f'] {f:.4f}')
except Exception:
    print('')
" 2>/dev/null || true
}

# 전체 모델 상태를 한 번에 표 형태로 출력한다.
print_report() {
  local now
  now=$(date '+%Y-%m-%d %H:%M:%S')
  local done_count=0
  local running_count=0
  local error_count=0

  # 먼저 상태를 모두 계산해 요약 카운트와 상세 표에 재사용한다.
  declare -a statuses=()
  for model_id in "${MODELS[@]}"; do
    statuses+=("$(status_of "$model_id")")
  done

  for s in "${statuses[@]}"; do
    if   [[ "$s" == DONE*    ]]; then done_count=$((done_count + 1));
    elif [[ "$s" == RUNNING* ]]; then running_count=$((running_count + 1));
    elif [[ "$s" == ERROR*   ]]; then error_count=$((error_count + 1));
    fi
  done

  local total_count=${#MODELS[@]}
  local pending_count=$(( total_count - done_count - running_count - error_count ))

  echo "┌─────────────────────────────────────────────────────────────────┐"
  printf "│  JamJamBeat 모델 비교 진행 현황  [%s]  │\n" "$now"
  echo "├────┬──────────────────────────────┬────────────────────────────┤"
  printf "│ %-2s │ %-28s │ %-26s │\n" " #" "Model" "Status"
  echo "├────┼──────────────────────────────┼────────────────────────────┤"

  for i in "${!MODELS[@]}"; do
    local model_id="${MODELS[$i]}"
    local s="${statuses[$i]}"
    local icon

    case "$s" in
      DONE*)    icon="✅" ;;
      RUNNING*) icon="🔄" ;;
      ERROR*)   icon="❌" ;;
      ENDED*)   icon="⚠️ " ;;
      *)        icon="⏳" ;;
    esac

    printf "│ %-2s │ %-28s │ %s %-24s │\n" \
      "$((i+1))" "$model_id" "$icon" "${s:0:24}"

    # 완료된 경우 F1 bar 출력
    if [[ "$s" == DONE* ]]; then
      local bar
      bar=$(get_f1_bar "$model_id")
      if [[ -n "$bar" ]]; then
        printf "│    │ %-28s │ %-26s │\n" "" "$bar"
      fi
    fi
  done

  echo "├────┴──────────────────────────────┴────────────────────────────┤"
  printf "│  완료: %d/%d  실행중: %d  오류: %d  대기: %d  (다음 갱신: %ds 후)%s│\n" \
    "$done_count" "$total_count" "$running_count" "$error_count" "$pending_count" \
    "$INTERVAL" "         "
  echo "└─────────────────────────────────────────────────────────────────┘"
  echo
}

# ──────────────────────────────────────────────────────
# 메인 루프
# ──────────────────────────────────────────────────────
echo "[watch_progress] 시작. ${INTERVAL}초(2분)마다 진행 현황 출력합니다."
echo "[watch_progress] 종료: Ctrl+C"
echo

while true; do
  print_report
  sleep "$INTERVAL"
done
