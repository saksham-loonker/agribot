#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Agribot final field setup"
echo "This installs boot automation. Field-safe default is direct phone hotspot."
echo "Optional farm Wi-Fi can be supplied with AGRIBOT_WIFI_SSID and AGRIBOT_WIFI_PASSWORD."
echo

MODE="${AGRIBOT_NETWORK_MODE:-hotspot}"
WIFI_SSID="${AGRIBOT_WIFI_SSID:-}"
WIFI_PASSWORD="${AGRIBOT_WIFI_PASSWORD:-}"
WIFI_STATIC_IP="${AGRIBOT_WIFI_STATIC_IP:-}"
WIFI_GATEWAY="${AGRIBOT_WIFI_GATEWAY:-}"
WIFI_DNS="${AGRIBOT_WIFI_DNS:-1.1.1.1 8.8.8.8}"
HOTSPOT_SSID="${AGRIBOT_HOTSPOT_SSID:-Agribot_Field}"
HOTSPOT_PASSWORD="${AGRIBOT_HOTSPOT_PASSWORD:-}"
HOTSPOT_IP="${AGRIBOT_HOTSPOT_IP:-10.42.0.1/24}"
PLATFORM_TOKEN="${AGRIBOT_PLATFORM_TOKEN:-}"
START_NOW="${AGRIBOT_START_NOW:-0}"

if [ -t 0 ] && [ -z "${AGRIBOT_NONINTERACTIVE:-}" ]; then
  read -rp "Access mode [hotspot/auto/wifi/existing] (default ${MODE}): " mode_input
  MODE="${mode_input:-$MODE}"
  if [ "$MODE" = "auto" ] || [ "$MODE" = "wifi" ]; then
    if [ -z "$WIFI_SSID" ]; then
      read -rp "Farm/lab Wi-Fi SSID (Enter to use hotspot only): " WIFI_SSID
    fi
    if [ -n "$WIFI_SSID" ] && [ -z "$WIFI_PASSWORD" ]; then
      read -rsp "Wi-Fi password for ${WIFI_SSID} (Enter if already saved on Pi): " WIFI_PASSWORD
      echo
    fi
  fi
fi

if [ "$MODE" = "wifi" ] && [ -z "$WIFI_SSID" ]; then
  echo "[ERROR] Wi-Fi mode requires AGRIBOT_WIFI_SSID." >&2
  exit 2
fi
if [ "$MODE" = "hotspot" ] && [ "${#HOTSPOT_PASSWORD}" -lt 8 ]; then
  echo "[ERROR] Set AGRIBOT_HOTSPOT_PASSWORD to a device-specific password of at least 8 characters." >&2
  exit 2
fi
if [ "$MODE" = "auto" ] && [ -z "$WIFI_SSID" ]; then
  echo "No Wi-Fi SSID provided; using hotspot mode."
  MODE="hotspot"
fi

python3 - "$MODE" "$WIFI_SSID" "$WIFI_STATIC_IP" "$WIFI_GATEWAY" "$WIFI_DNS" "$HOTSPOT_SSID" "$HOTSPOT_PASSWORD" "$HOTSPOT_IP" <<'PY'
import json
import sys
from pathlib import Path

mode, wifi_ssid, wifi_static_ip, wifi_gateway, wifi_dns, hotspot_ssid, hotspot_password, hotspot_ip = sys.argv[1:9]
path = Path("farmer_config.json")
cfg = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
net = cfg.setdefault("network", {})
net.update({
    "mode": mode,
    "wifi_ssid": wifi_ssid,
    "wifi_static_ip": wifi_static_ip,
    "wifi_gateway": wifi_gateway,
    "wifi_dns": wifi_dns,
    "hotspot_ssid": hotspot_ssid,
    "hotspot_password": hotspot_password,
    "hotspot_ip": hotspot_ip,
    "auto_wifi_wait_sec": int(net.get("auto_wifi_wait_sec") or 12),
})
path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
PY

tmp_env="$(mktemp)"
python3 - "$MODE" "$WIFI_SSID" "$WIFI_PASSWORD" "$WIFI_STATIC_IP" "$WIFI_GATEWAY" "$WIFI_DNS" "$HOTSPOT_SSID" "$HOTSPOT_PASSWORD" "$HOTSPOT_IP" "$PLATFORM_TOKEN" >"$tmp_env" <<'PY'
import shlex
import sys

mode, wifi_ssid, wifi_password, wifi_static_ip, wifi_gateway, wifi_dns, hotspot_ssid, hotspot_password, hotspot_ip, platform_token = sys.argv[1:11]
values = {
    "AGRIBOT_NETWORK_MODE": mode,
    "AGRIBOT_WIFI_SSID": wifi_ssid,
    "AGRIBOT_WIFI_PASSWORD": wifi_password,
    "AGRIBOT_WIFI_STATIC_IP": wifi_static_ip,
    "AGRIBOT_WIFI_GATEWAY": wifi_gateway,
    "AGRIBOT_WIFI_DNS": wifi_dns,
    "AGRIBOT_HOTSPOT_SSID": hotspot_ssid,
    "AGRIBOT_HOTSPOT_PASSWORD": hotspot_password,
    "AGRIBOT_HOTSPOT_IP": hotspot_ip,
    "AGRIBOT_PLATFORM_TOKEN": platform_token,
}
for key, value in values.items():
    if value:
        print(f"{key}={shlex.quote(value)}")
PY
sudo mkdir -p /etc/agribot
sudo install -m 600 -o root -g root "$tmp_env" /etc/agribot/access.env
rm -f "$tmp_env"

PYTHON_BIN="${AGRIBOT_VENV:-$HOME/agribot_venv}/bin/python3"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3)"
fi

install_args=(44_farmer_one_touch.py install-autostart --config farmer_config.json --user pi)
if [ "$START_NOW" = "1" ]; then
  install_args+=(--start-now)
fi
sudo "$PYTHON_BIN" "${install_args[@]}"

echo
echo "Setup complete."
echo "Boot access mode: $MODE"
echo "Hotspot fallback : $HOTSPOT_SSID / $HOTSPOT_PASSWORD / http://${HOTSPOT_IP%/*}:8080"
if [ "$START_NOW" != "1" ]; then
  echo "Services are installed and will start on next boot. Set AGRIBOT_START_NOW=1 to start/switch network immediately."
fi
echo "Check:"
echo "  systemctl status agribot-access.service agribot-farmer.service"
echo "  cat agribot_inference_data/dashboard_url.txt"
