#!/usr/bin/env bash
# install_cron_jobs.sh - generate_9_cron_jobs.sh 로 생성된 9개 job을
# 기존 crontab에 managed block 방식으로 병합 설치합니다.
# (기존 crontab의 다른 항목을 보존합니다)
#
# Usage:
#   bash model/cron/install_cron_jobs.sh
#
# uninstall:
#   bash model/cron/uninstall_model_comparison_cron.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/user/projects/JamJamBeat}"
CRON_FILE="${CRON_FILE:-$PROJECT_ROOT/model/cron/jamjambeat_9jobs.crontab}"
GENERATOR="${GENERATOR:-$PROJECT_ROOT/model/cron/generate_9_cron_jobs.sh}"

START_MARK="# >>> JAMJAMBEAT_MODEL_COMPARISON_CRON >>>"
END_MARK="# <<< JAMJAMBEAT_MODEL_COMPARISON_CRON <<<"

# crontab source 파일이 없으면 먼저 생성기부터 실행해 managed block 입력을 준비한다.
if [[ ! -f "$CRON_FILE" ]]; then
  echo "[INFO] cron 파일이 없어 먼저 생성합니다: $CRON_FILE"
  if [[ ! -x "$GENERATOR" ]]; then
    chmod +x "$GENERATOR" 2>/dev/null || true
  fi
  if [[ ! -f "$GENERATOR" ]]; then
    echo "[ERROR] generator not found: $GENERATOR" >&2
    exit 1
  fi
  bash "$GENERATOR"
fi

TMP_OLD="$(mktemp)"
TMP_CLEAN="$(mktemp)"
TMP_NEW="$(mktemp)"

# 시스템 crontab 전체를 임시 파일로 받고, 우리 block만 교체하는 방식으로 병합한다.
crontab -l > "$TMP_OLD" 2>/dev/null || true

# 기존 managed block 제거 (재설치 시 중복 방지)
awk -v start="$START_MARK" -v end="$END_MARK" '
  $0 == start {skip=1; next}
  $0 == end   {skip=0; next}
  skip != 1   {print}
' "$TMP_OLD" > "$TMP_CLEAN"

# 생성된 cron 파일에서는 실제 cron 행만 뽑아 managed block 안에 넣는다.
{
  cat "$TMP_CLEAN"
  echo "$START_MARK"
  echo "# Auto-generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  # jamjambeat_9jobs.crontab 에서 실제 cron 행만 추출 (주석·빈줄·SHELL/PATH 제외)
  grep -E '^[0-9]' "$CRON_FILE" || true
  echo "$END_MARK"
} > "$TMP_NEW"

crontab "$TMP_NEW"

rm -f "$TMP_OLD" "$TMP_CLEAN" "$TMP_NEW"

echo "[OK] crontab 설치 완료 (managed block 방식, 기존 항목 보존)"
echo "----------------------------------------"
crontab -l
