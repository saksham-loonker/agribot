#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d "${AGRIBOT_VENV:-$HOME/agribot_venv}" ]; then
  source "${AGRIBOT_VENV:-$HOME/agribot_venv}/bin/activate"
fi

echo "Installing Agribot autostart service. You may be asked for the Pi password."
sudo "$(command -v python3)" 44_farmer_one_touch.py install-autostart --config farmer_config.json --user pi
