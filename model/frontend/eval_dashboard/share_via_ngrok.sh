#!/usr/bin/env bash
# share_via_ngrok.sh - Run the eval dashboard locally and expose it with ngrok.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PORT="${1:-8123}"
HOST="${HOST:-127.0.0.1}"
NGROK_DOMAIN="${NGROK_DOMAIN:-}"
SERVER_PID=""
MODEL_PYTHON="${ROOT_DIR}/model/.venv/bin/python"

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

if ! command -v ngrok >/dev/null 2>&1; then
  echo "ngrok CLI is not installed. Install it first and run 'ngrok config add-authtoken <TOKEN>'." >&2
  exit 1
fi

if [[ ! -x "${MODEL_PYTHON}" ]]; then
  echo "model/.venv python is not available at ${MODEL_PYTHON}." >&2
  echo "Create the model virtual environment first." >&2
  exit 1
fi

cd "${ROOT_DIR}"

echo "[1/3] Starting eval dashboard server on http://${HOST}:${PORT}/ ..."
"${MODEL_PYTHON}" model/frontend/eval_dashboard/serve_dashboard.py --host "${HOST}" --port "${PORT}" &
SERVER_PID=$!

echo "[2/3] Waiting for local server to respond ..."
for _ in $(seq 1 30); do
  if curl -fsS "http://${HOST}:${PORT}/" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -fsS "http://${HOST}:${PORT}/" >/dev/null 2>&1; then
  echo "Local dashboard server did not start correctly." >&2
  exit 1
fi

echo "[3/3] Opening ngrok tunnel ..."
echo "Local dashboard: http://${HOST}:${PORT}/model/frontend/eval_dashboard/"
echo "Press Ctrl+C to stop both the local server and ngrok."

if [[ -n "${NGROK_DOMAIN}" ]]; then
  echo "Using reserved ngrok domain: ${NGROK_DOMAIN}"
  ngrok http --url="${NGROK_DOMAIN}" "${HOST}:${PORT}"
else
  ngrok http "${HOST}:${PORT}"
fi
