#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d "${AGRIBOT_VENV:-$HOME/agribot_venv}" ]; then
  source "${AGRIBOT_VENV:-$HOME/agribot_venv}/bin/activate"
fi

if [ ! -d runtime_exports/classifier_openvino_model ]; then
  echo "[ERROR] Missing runtime_exports/classifier_openvino_model"
  echo "Copy/export the calibrated OpenVINO classifier before running live inference."
  exit 1
fi

CAMERA="${1:-picamera2}"
RUN_ID="${AGRIBOT_RUN_ID:-field_$(date +%Y%m%d_%H%M%S)}"

python3 43_low_power_5v3a_experiment.py \
  --mode live \
  --camera "$CAMERA" \
  --camera-width 640 \
  --camera-height 480 \
  --picamera-raw-width 2304 \
  --picamera-raw-height 1296 \
  --clf-model runtime_exports/classifier_openvino_model \
  --threads 4 \
  --fps 10 \
  --max-edge 256 \
  --crop-mode mask \
  --crop-scale 0.55 \
  --crop-pad 0.05 \
  --conf 0.765 \
  --high-conf 0.765 \
  --jsonl runtime_reports/low_power/plant_loop_10fps.jsonl \
  --data-root agribot_inference_data \
  --run-id "$RUN_ID" \
  --field-id "Field 1" \
  --row-id "A" \
  --start-plant 1 \
  --plant-step 1 \
  --plant-cooldown-sec 2.0 \
  --nice 5 \
  --cpu-affinity 0,1,2,3
