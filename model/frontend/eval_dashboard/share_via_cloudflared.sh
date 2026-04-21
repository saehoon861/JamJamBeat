#!/usr/bin/env bash
# share_via_cloudflared.sh - Run the eval dashboard locally and expose it with a Cloudflare Quick Tunnel.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PORT="${1:-8123}"
HOST="${HOST:-127.0.0.1}"
SERVER_PID=""
MODEL_PYTHON="${ROOT_DIR}/model/.venv/bin/python"

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

cd "${ROOT_DIR}"

if [[ ! -x "${MODEL_PYTHON}" ]]; then
  echo "model/.venv python is not available at ${MODEL_PYTHON}." >&2
  echo "Create the model virtual environment first." >&2
  exit 1
fi

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

echo "[3/3] Opening Cloudflare Quick Tunnel ..."
echo "Local dashboard: http://${HOST}:${PORT}/model/frontend/eval_dashboard/"
echo "Press Ctrl+C to stop both the local server and the tunnel."

for attempt in 1 2 3; do
  if [[ "${attempt}" -gt 1 ]]; then
    echo "Retrying Cloudflare Quick Tunnel (${attempt}/3) ..."
  fi
  if cloudflared tunnel --no-autoupdate --url "http://${HOST}:${PORT}"; then
    exit 0
  fi
  if [[ "${attempt}" -lt 3 ]]; then
    sleep 2
  fi
done

echo "Cloudflare Quick Tunnel did not come up after 3 attempts." >&2
echo "The local dashboard server is healthy, but TryCloudflare returned an invalid 500/1101 response." >&2
echo "This usually means a Quick Tunnel outage on api.trycloudflare.com rather than a problem in this repo." >&2
echo "If this keeps happening, retry later or switch to a named tunnel with a fixed hostname." >&2
exit 1
