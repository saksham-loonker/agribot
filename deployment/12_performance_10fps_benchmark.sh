#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d "${AGRIBOT_VENV:-$HOME/agribot_venv}" ]; then
  source "${AGRIBOT_VENV:-$HOME/agribot_venv}/bin/activate"
fi

if [ ! -d runtime_exports/classifier_openvino_model ]; then
  echo "[ERROR] Missing runtime_exports/classifier_openvino_model"
  echo "Copy/export the calibrated OpenVINO classifier before benchmarking."
  exit 1
fi

python3 43_low_power_5v3a_experiment.py \
  --mode benchmark \
  --test clf_dataset/test \
  --clf-model runtime_exports/classifier_openvino_model \
  --threads 4 \
  --fps 10 \
  --limit 0 \
  --max-edge 256 \
  --crop-mode mask \
  --crop-scale 0.55 \
  --crop-pad 0.05 \
  --conf 0.765 \
  --high-conf 0.765 \
  --nice 5 \
  --cpu-affinity 0,1,2,3 \
  --sample-power-every 0 \
  --progress-every 0 \
  --report runtime_reports/low_power/performance_10fps_report.json
vcgencmd get_throttled || true
vcgencmd measure_temp || true
