#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SSID="${AGRIBOT_WIFI_SSID:-}"
PASSWORD="${AGRIBOT_WIFI_PASSWORD:-}"
STATIC_IP="${AGRIBOT_WIFI_STATIC_IP:-}"
GATEWAY="${AGRIBOT_WIFI_GATEWAY:-}"
DNS="${AGRIBOT_WIFI_DNS:-1.1.1.1 8.8.8.8}"

if [ -z "$SSID" ] && [ -t 0 ]; then
  read -rp "Wi-Fi SSID: " SSID
fi
if [ -z "$SSID" ]; then
  echo "[ERROR] Set AGRIBOT_WIFI_SSID or enter an SSID." >&2
  exit 2
fi

if [ -z "$PASSWORD" ] && [ -t 0 ]; then
  read -rsp "Wi-Fi password for ${SSID}: " PASSWORD
  echo
fi

sudo nmcli radio wifi on
sudo nmcli con delete "$SSID" >/dev/null 2>&1 || true
if [ -n "$PASSWORD" ]; then
  sudo nmcli dev wifi connect "$SSID" password "$PASSWORD" ifname wlan0 name "$SSID"
else
  sudo nmcli con add type wifi ifname wlan0 con-name "$SSID" autoconnect yes ssid "$SSID"
fi

sudo nmcli con mod "$SSID" connection.autoconnect yes connection.autoconnect-priority 100 ipv6.method ignore
if [ -n "$STATIC_IP" ] && [ -n "$GATEWAY" ]; then
  sudo nmcli con mod "$SSID" ipv4.method manual ipv4.addresses "$STATIC_IP" ipv4.gateway "$GATEWAY" ipv4.dns "$DNS"
else
  sudo nmcli con mod "$SSID" ipv4.method auto ipv4.dns "$DNS"
fi
sudo nmcli con up "$SSID"

python3 - "$SSID" "$STATIC_IP" "$GATEWAY" "$DNS" <<'PY'
import json
import sys
from pathlib import Path

ssid, static_ip, gateway, dns = sys.argv[1:5]
path = Path("farmer_config.json")
cfg = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
net = cfg.setdefault("network", {})
net.update({
    "mode": "auto",
    "wifi_ssid": ssid,
    "wifi_static_ip": static_ip,
    "wifi_gateway": gateway,
    "wifi_dns": dns,
})
path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
PY

tmp_env="$(mktemp)"
python3 - "$SSID" "$PASSWORD" "$STATIC_IP" "$GATEWAY" "$DNS" >"$tmp_env" <<'PY'
import shlex
import sys

ssid, password, static_ip, gateway, dns = sys.argv[1:6]
values = {
    "AGRIBOT_NETWORK_MODE": "auto",
    "AGRIBOT_WIFI_SSID": ssid,
    "AGRIBOT_WIFI_PASSWORD": password,
    "AGRIBOT_WIFI_STATIC_IP": static_ip,
    "AGRIBOT_WIFI_GATEWAY": gateway,
    "AGRIBOT_WIFI_DNS": dns,
}
for key, value in values.items():
    if value:
        print(f"{key}={shlex.quote(value)}")
PY
sudo mkdir -p /etc/agribot
sudo install -m 600 -o root -g root "$tmp_env" /etc/agribot/access.env
rm -f "$tmp_env"

echo "Connected to $SSID."
echo "Saved this Wi-Fi as the preferred boot network; if it is unavailable, Agribot can still fall back to hotspot mode."
