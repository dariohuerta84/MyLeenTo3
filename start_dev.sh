#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PORT="${MYLEENTO3_BACKEND_PORT:-8085}"
FRONTEND_PORT="${MYLEENTO3_FRONTEND_PORT:-5173}"

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "${BACKEND_PID}" 2>/dev/null || true
    wait "${BACKEND_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting MyLeenTo3 FastAPI backend on http://127.0.0.1:${BACKEND_PORT} ..."
(
  cd "${ROOT_DIR}/backend"
  exec .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port "${BACKEND_PORT}"
) &
BACKEND_PID=$!

for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${BACKEND_PORT}/api/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.3
done

echo "MyLeenTo3 Web UI: http://127.0.0.1:${FRONTEND_PORT}"
echo "MyLeenTo3 API:    http://127.0.0.1:${BACKEND_PORT}/docs"

cd "${ROOT_DIR}/frontend"
VITE_API_TARGET="http://127.0.0.1:${BACKEND_PORT}" npm run dev -- --host 127.0.0.1 --port "${FRONTEND_PORT}"
