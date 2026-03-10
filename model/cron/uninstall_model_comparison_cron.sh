#!/usr/bin/env bash
set -euo pipefail

START_MARK="# >>> JAMJAMBEAT_MODEL_COMPARISON_CRON >>>"
END_MARK="# <<< JAMJAMBEAT_MODEL_COMPARISON_CRON <<<"

TMP_OLD="$(mktemp)"
TMP_NEW="$(mktemp)"

crontab -l > "$TMP_OLD" 2>/dev/null || true

awk -v start="$START_MARK" -v end="$END_MARK" '
  $0 == start {skip=1; next}
  $0 == end   {skip=0; next}
  skip != 1   {print}
' "$TMP_OLD" > "$TMP_NEW"

crontab "$TMP_NEW"
rm -f "$TMP_OLD" "$TMP_NEW"

echo "[OK] Removed JamJamBeat model comparison cron block."
