#!/usr/bin/env bash
set -euo pipefail

SSID="${AGRIBOT_HOTSPOT_SSID:-Agribot_Field}"
PASSWORD="${AGRIBOT_HOTSPOT_PASSWORD:-}"
IP_CIDR="${AGRIBOT_HOTSPOT_IP:-10.42.0.1/24}"

if [ "${#PASSWORD}" -lt 8 ]; then
  echo "[ERROR] Set AGRIBOT_HOTSPOT_PASSWORD to a device-specific password of at least 8 characters." >&2
  exit 2
fi

sudo nmcli radio wifi on
sudo nmcli con delete "$SSID" >/dev/null 2>&1 || true
sudo nmcli con add type wifi ifname wlan0 con-name "$SSID" autoconnect yes ssid "$SSID"
sudo nmcli con mod "$SSID" \
  802-11-wireless.mode ap \
  802-11-wireless.band bg \
  ipv4.method shared \
  ipv4.addresses "$IP_CIDR" \
  ipv6.method ignore \
  wifi-sec.key-mgmt wpa-psk \
  wifi-sec.psk "$PASSWORD"
sudo nmcli con up "$SSID"

echo "Agribot direct phone hotspot is active."
echo "  Wi-Fi : $SSID"
echo "  Pass  : configured from AGRIBOT_HOTSPOT_PASSWORD (not printed)"
echo "  URL   : http://${IP_CIDR%/*}:8080"
echo "Set the Android app Pi URL to http://${IP_CIDR%/*}:8080"
