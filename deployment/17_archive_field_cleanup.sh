#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d "${AGRIBOT_VENV:-$HOME/agribot_venv}" ]; then
  source "${AGRIBOT_VENV:-$HOME/agribot_venv}/bin/activate"
fi

python3 44_farmer_one_touch.py archive
