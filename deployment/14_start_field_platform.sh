#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d "${AGRIBOT_VENV:-$HOME/agribot_venv}" ]; then
  source "${AGRIBOT_VENV:-$HOME/agribot_venv}/bin/activate"
fi

PORT="${1:-8080}"
PID_FILE="runtime_reports/platform/platform.pid"
mkdir -p "$(dirname "$PID_FILE")"

if curl -fsS "http://127.0.0.1:${PORT}/api/status" >/dev/null 2>&1; then
  echo "Agribot platform is already running."
  echo "Open: http://$(hostname -I | awk '{print $1}'):${PORT}"
  exit 0
fi

python3 agribot_platform/run_platform.py \
  --host 0.0.0.0 \
  --port "$PORT" \
  --data-root agribot_inference_data
