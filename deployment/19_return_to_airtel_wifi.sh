#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
echo "[INFO] This helper is now generic. Use deployment/19_connect_configured_wifi.sh for new docs."
exec ./deployment/19_connect_configured_wifi.sh "$@"
