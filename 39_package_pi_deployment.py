#!/usr/bin/env python3
"""
Package the calibrated Raspberry Pi deployment bundle.

The package keeps the normal deployment default at 4 threads, but also writes
safe test/run scripts that force 1 thread for today's undervoltage-limited Pi.
It includes the calibrated fast-crop classifier, runtime scripts, optional test
datasets, and helper scripts for export, benchmarking, power checks, and live
operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent

RUNTIME_FILES = [
    "inference_rpi.py",
    "pi_plant_loop.py",
    "benchmark_rpi_pi.py",
    "runtime_benchmark_common.py",
    "runtime_suite_common.py",
    "runtime_export_common.py",
    "24_benchmark_runtime_openvino.py",
    "34_benchmark_tomato_crop_disease.py",
    "37_train_deployment_fastcrop_classifier.py",
    "38_sweep_classifier_confidence_gate.py",
    "40_pi_end_to_end_validation.py",
    "41_pi_tomato_crop_disease_validation.py",
    "43_low_power_5v3a_experiment.py",
    "44_farmer_one_touch.py",
    "45_build_android_apk.py",
    "farmer_config.json",
    "requirements-runtime-pi.txt",
    "RUNTIME_TESTING.md",
]


def parse_args():
    p = argparse.ArgumentParser(description="Build Pi deployment package")
    p.add_argument("--out-dir", default="dist")
    p.add_argument("--name", default="agribot_pi_deploy")
    p.add_argument("--no-stamp", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--pi-ip", default="192.168.1.25")

    p.add_argument("--classifier-pt", default="models/classifier_deploy_fastcrop.pt")
    p.add_argument("--classifier-openvino", default="runtime_exports/classifier_openvino_model")
    p.add_argument("--include-openvino", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-test-set", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-tomato-dataset", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-nano-detector", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--nano-detector", default="models/detector_nano_256.pt")

    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--deploy-threads", type=int, default=4)
    p.add_argument("--safe-test-threads", type=int, default=1)
    p.add_argument("--low-power-threads", type=int, default=2)
    p.add_argument("--low-power-cpu-affinity", default="0,1")
    p.add_argument("--performance-fps", type=float, default=10.0)
    p.add_argument("--performance-threads", type=int, default=4)
    p.add_argument("--performance-cpu-affinity", default="0,1,2,3")
    p.add_argument("--conf", type=float, default=0.765)
    p.add_argument("--high-conf", type=float, default=0.765)
    p.add_argument("--max-edge", type=int, default=256)
    p.add_argument("--crop-mode", choices=["mask", "center", "none"], default="mask")
    p.add_argument("--crop-scale", type=float, default=0.55)
    p.add_argument("--crop-pad", type=float, default=0.05)
    p.add_argument("--safe-benchmark-limit", type=int, default=40)
    p.add_argument("--normal-benchmark-limit", type=int, default=80)
    p.add_argument("--camera", default="0")
    p.add_argument("--no-archive", action="store_true")
    return p.parse_args()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(src: Path, dst: Path, manifest: list[dict], package_root: Path, *, required: bool = True):
    if not src.exists():
        if required:
            raise SystemExit(f"[ERROR] Required file missing: {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    manifest.append(
        {
            "type": "file",
            "source": rel(src),
            "path": str(dst.relative_to(package_root)),
            "bytes": dst.stat().st_size,
            "sha256": sha256(dst),
        }
    )


def ignore_tree(dir_name, names):
    ignored = {"__pycache__", ".pytest_cache", ".DS_Store", "node_modules", ".gradle", "build"}
    return [name for name in names if name in ignored or name.endswith(".cache")]


def copy_tree(src: Path, dst: Path, manifest: list[dict], package_root: Path, *, required: bool = True):
    if not src.exists():
        if required:
            raise SystemExit(f"[ERROR] Required directory missing: {src}")
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore_tree)
    total_bytes = sum(path.stat().st_size for path in dst.rglob("*") if path.is_file())
    file_count = sum(1 for path in dst.rglob("*") if path.is_file())
    manifest.append(
        {
            "type": "directory",
            "source": rel(src),
            "path": str(dst.relative_to(package_root)),
            "files": file_count,
            "bytes": total_bytes,
        }
    )


def write_text(path: Path, content: str, *, executable: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | 0o755)


def shell_header() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d "${AGRIBOT_VENV:-$HOME/agribot_venv}" ]; then
  source "${AGRIBOT_VENV:-$HOME/agribot_venv}/bin/activate"
fi
"""


def quoted_config_args(args, *, threads: int) -> str:
    return (
        f"--threads {threads} "
        f"--max-edge {args.max_edge} "
        f"--conf {args.conf} "
        f"--crop-mode {args.crop_mode} "
        f"--crop-scale {args.crop_scale} "
        f"--crop-pad {args.crop_pad}"
    )


def write_helper_scripts(pkg: Path, args):
    scripts = pkg / "deployment"

    write_text(
        scripts / "00_power_check.sh",
        shell_header()
        + """echo "Power/thermal status:"
vcgencmd get_throttled || true
vcgencmd measure_temp || true
echo
echo "Recent kernel warnings:"
dmesg -T | egrep -i "under-voltage|voltage|thrott|thermal|mmc|i/o|ext4|oom|power|reset" | tail -80 || true
""",
        executable=True,
    )

    write_text(
        scripts / "01_export_classifier_openvino.sh",
        shell_header()
        + """if [ -d runtime_exports/classifier_openvino_model ]; then
  echo "OpenVINO classifier already exists: runtime_exports/classifier_openvino_model"
  exit 0
fi

python3 37_train_deployment_fastcrop_classifier.py \\
  --skip-prepare --skip-train --skip-eval \\
  --base-model models/classifier_deploy_fastcrop.pt \\
  --copy-best-to models/classifier_deploy_fastcrop.pt \\
  --export-openvino \\
  --openvino-out runtime_exports/classifier_openvino_model \\
  --imgsz 224
""",
        executable=True,
    )

    safe_args = quoted_config_args(args, threads=args.safe_test_threads)
    normal_args = quoted_config_args(args, threads=args.deploy_threads)

    write_text(
        scripts / "02_safe_benchmark_threads1.sh",
        shell_header()
        + f"""./deployment/01_export_classifier_openvino.sh
python3 24_benchmark_runtime_openvino.py \\
  --test clf_dataset/test \\
  --limit {args.safe_benchmark_limit} \\
  {safe_args}
vcgencmd get_throttled || true
""",
        executable=True,
    )

    write_text(
        scripts / "03_full_benchmark_threads1.sh",
        shell_header()
        + f"""./deployment/01_export_classifier_openvino.sh
python3 24_benchmark_runtime_openvino.py \\
  --test clf_dataset/test \\
  --limit 0 \\
  {safe_args}
vcgencmd get_throttled || true
""",
        executable=True,
    )

    write_text(
        scripts / "04_normal_benchmark_threads4_after_power_fix.sh",
        shell_header()
        + f"""./deployment/01_export_classifier_openvino.sh
python3 24_benchmark_runtime_openvino.py \\
  --test clf_dataset/test \\
  --limit {args.normal_benchmark_limit} \\
  {normal_args}
vcgencmd get_throttled || true
""",
        executable=True,
    )

    write_text(
        scripts / "05_live_safe_threads1.sh",
        shell_header()
        + f"""./deployment/01_export_classifier_openvino.sh
CAMERA="${{1:-{args.camera}}}"
python3 pi_plant_loop.py \\
  --camera "$CAMERA" \\
  --clf-model runtime_exports/classifier_openvino_model \\
  --fps {args.fps} \\
  --high-conf {args.high_conf} \\
  --jsonl runtime_reports/plant_loop.jsonl \\
  {safe_args}
""",
        executable=True,
    )

    write_text(
        scripts / "06_live_normal_threads4_after_power_fix.sh",
        shell_header()
        + f"""./deployment/01_export_classifier_openvino.sh
CAMERA="${{1:-{args.camera}}}"
python3 pi_plant_loop.py \\
  --camera "$CAMERA" \\
  --clf-model runtime_exports/classifier_openvino_model \\
  --fps {args.fps} \\
  --high-conf {args.high_conf} \\
  --jsonl runtime_reports/plant_loop.jsonl \\
  {normal_args}
""",
        executable=True,
    )

    write_text(
        scripts / "07_tomato_image_hit_smoke_threads1.sh",
        shell_header()
        + f"""DET_MODEL="${{1:-models/detector_nano_256.pt}}"
if [ ! -e "$DET_MODEL" ]; then
  echo "[ERROR] Detector not found: $DET_MODEL"
  echo "Pass a detector path, for example: ./deployment/07_tomato_image_hit_smoke_threads1.sh runtime_exports/detector_openvino_model"
  exit 1
fi
python3 34_benchmark_tomato_crop_disease.py \\
  --det-model "$DET_MODEL" \\
  --imgsz 256 \\
  --max-det 20 \\
  --threads {args.safe_test_threads} \\
  --limit 10 \\
  --target-metric image_hit_rate \\
  --target-value 0.90
vcgencmd get_throttled || true
""",
        executable=True,
    )

    write_text(
        scripts / "08_full_end_to_end_validation_threads1.sh",
        shell_header()
        + f"""./deployment/01_export_classifier_openvino.sh
python3 40_pi_end_to_end_validation.py \\
  --threads {args.safe_test_threads} \\
  --normal-threads {args.deploy_threads} \\
  --classifier-limit 0 \\
  --conf {args.conf} \\
  --high-conf {args.high_conf} \\
  --max-edge {args.max_edge} \\
  --crop-mode {args.crop_mode} \\
  --crop-scale {args.crop_scale} \\
  --crop-pad {args.crop_pad} \\
  --tomato-limit 10
""",
        executable=True,
    )

    write_text(
        scripts / "09_tomato_crop_disease_full_validation_threads1.sh",
        shell_header()
        + f"""python3 41_pi_tomato_crop_disease_validation.py \\
  --dataset tomato_crop_disease \\
  --det-model models/detector_nano_256.pt \\
  --threads {args.safe_test_threads} \\
  --normal-threads {args.deploy_threads} \\
  --imgsz 256 \\
  --max-det 20 \\
  --limit 0 \\
  --target-metric image_hit_rate \\
  --target-value 0.90
""",
        executable=True,
    )

    write_text(
        scripts / "10_low_power_5v3a_benchmark.sh",
        shell_header()
        + f"""./deployment/01_export_classifier_openvino.sh
python3 43_low_power_5v3a_experiment.py \\
  --mode benchmark \\
  --test clf_dataset/test \\
  --clf-model runtime_exports/classifier_openvino_model \\
  --threads {args.low_power_threads} \\
  --fps {args.fps} \\
  --limit 0 \\
  --max-edge {args.max_edge} \\
  --crop-mode {args.crop_mode} \\
  --crop-scale {args.crop_scale} \\
  --crop-pad {args.crop_pad} \\
  --conf {args.conf} \\
  --high-conf {args.high_conf} \\
  --nice 5 \\
  --cpu-affinity {args.low_power_cpu_affinity} \\
  --sample-power-every 0
vcgencmd get_throttled || true
""",
        executable=True,
    )

    write_text(
        scripts / "11_low_power_5v3a_live.sh",
        shell_header()
        + f"""./deployment/01_export_classifier_openvino.sh
CAMERA="${{1:-{args.camera}}}"
python3 43_low_power_5v3a_experiment.py \\
  --mode live \\
  --camera "$CAMERA" \\
  --camera-width 640 \\
  --camera-height 480 \\
  --picamera-raw-width 2304 \\
  --picamera-raw-height 1296 \\
  --clf-model runtime_exports/classifier_openvino_model \\
  --threads {args.low_power_threads} \\
  --fps {args.fps} \\
  --max-edge {args.max_edge} \\
  --crop-mode {args.crop_mode} \\
  --crop-scale {args.crop_scale} \\
  --crop-pad {args.crop_pad} \\
  --conf {args.conf} \\
  --high-conf {args.high_conf} \\
  --jsonl runtime_reports/low_power/plant_loop_low_power.jsonl \\
  --data-root agribot_inference_data \\
  --nice 5 \\
  --cpu-affinity {args.low_power_cpu_affinity}
""",
        executable=True,
    )

    write_text(
        scripts / "12_performance_10fps_benchmark.sh",
        shell_header()
        + f"""./deployment/01_export_classifier_openvino.sh
echo "[INFO] Fan/heatsink performance profile. Validated with 5V/5A supply."
python3 43_low_power_5v3a_experiment.py \\
  --mode benchmark \\
  --test clf_dataset/test \\
  --clf-model runtime_exports/classifier_openvino_model \\
  --threads {args.performance_threads} \\
  --fps {args.performance_fps} \\
  --limit 0 \\
  --max-edge {args.max_edge} \\
  --crop-mode {args.crop_mode} \\
  --crop-scale {args.crop_scale} \\
  --crop-pad {args.crop_pad} \\
  --conf {args.conf} \\
  --high-conf {args.high_conf} \\
  --nice 5 \\
  --cpu-affinity {args.performance_cpu_affinity} \\
  --sample-power-every 0 \\
  --progress-every 0 \\
  --report runtime_reports/low_power/performance_10fps_report.json
vcgencmd get_throttled || true
vcgencmd measure_temp || true
""",
        executable=True,
    )

    write_text(
        scripts / "13_performance_10fps_live.sh",
        shell_header()
        + f"""./deployment/01_export_classifier_openvino.sh
CAMERA="${{1:-{args.camera}}}"
RUN_ID="${{AGRIBOT_RUN_ID:-field_$(date +%Y%m%d_%H%M%S)}}"
echo "[INFO] Fan/heatsink performance live profile. Validated with 5V/5A supply."
python3 43_low_power_5v3a_experiment.py \\
  --mode live \\
  --camera "$CAMERA" \\
  --camera-width 640 \\
  --camera-height 480 \\
  --picamera-raw-width 2304 \\
  --picamera-raw-height 1296 \\
  --clf-model runtime_exports/classifier_openvino_model \\
  --threads {args.performance_threads} \\
  --fps {args.performance_fps} \\
  --max-edge {args.max_edge} \\
  --crop-mode {args.crop_mode} \\
  --crop-scale {args.crop_scale} \\
  --crop-pad {args.crop_pad} \\
  --conf {args.conf} \\
  --high-conf {args.high_conf} \\
  --jsonl runtime_reports/low_power/plant_loop_10fps.jsonl \\
  --data-root agribot_inference_data \\
  --run-id "$RUN_ID" \\
  --field-id "Field 1" \\
  --row-id "A" \\
  --start-plant 1 \\
  --plant-step 1 \\
  --plant-cooldown-sec 2.0 \\
  --nice 5 \\
  --cpu-affinity {args.performance_cpu_affinity}
""",
        executable=True,
    )

    write_text(
        scripts / "14_start_field_platform.sh",
        shell_header()
        + """PORT="${1:-8080}"
if curl -fsS "http://127.0.0.1:${PORT}/api/status" >/dev/null 2>&1; then
  echo "Agribot platform is already running."
  echo "Open: http://$(hostname -I | awk '{print $1}'):${PORT}"
  exit 0
fi
python3 agribot_platform/run_platform.py \\
  --host 0.0.0.0 \\
  --port "$PORT" \\
  --data-root agribot_inference_data
""",
        executable=True,
    )

    write_text(
        scripts / "15_farmer_one_touch.sh",
        shell_header()
        + """python3 44_farmer_one_touch.py run --config farmer_config.json
""",
        executable=True,
    )

    write_text(
        scripts / "16_install_farmer_autostart.sh",
        shell_header()
        + """echo "Installing Agribot autostart service. You may be asked for the Pi password."
sudo "$(command -v python3)" 44_farmer_one_touch.py install-autostart --config farmer_config.json --user pi
""",
        executable=True,
    )

    write_text(
        scripts / "17_archive_field_cleanup.sh",
        shell_header()
        + """python3 44_farmer_one_touch.py archive
""",
        executable=True,
    )

    write_text(
        scripts / "18_enable_direct_phone_hotspot.sh",
        """#!/usr/bin/env bash
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
sudo nmcli con mod "$SSID" \\
  802-11-wireless.mode ap \\
  802-11-wireless.band bg \\
  ipv4.method shared \\
  ipv4.addresses "$IP_CIDR" \\
  ipv6.method ignore \\
  wifi-sec.key-mgmt wpa-psk \\
  wifi-sec.psk "$PASSWORD"
sudo nmcli con up "$SSID"

echo "Agribot direct phone hotspot is active."
echo "  Wi-Fi : $SSID"
echo "  Pass  : configured from AGRIBOT_HOTSPOT_PASSWORD (not printed)"
echo "  URL   : http://${IP_CIDR%/*}:8080"
echo "Set the Android app Pi URL to http://${IP_CIDR%/*}:8080"
""",
        executable=True,
    )

    write_text(
        scripts / "19_connect_configured_wifi.sh",
        """#!/usr/bin/env bash
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
path.write_text(json.dumps(cfg, indent=2) + "\\n", encoding="utf-8")
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
""",
        executable=True,
    )

    write_text(
        scripts / "19_return_to_airtel_wifi.sh",
        """#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
echo "[INFO] This helper is now generic. Use deployment/19_connect_configured_wifi.sh for new docs."
exec ./deployment/19_connect_configured_wifi.sh "$@"
""",
        executable=True,
    )

    write_text(
        scripts / "20_install_final_field_system.sh",
        """#!/usr/bin/env bash
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
path.write_text(json.dumps(cfg, indent=2) + "\\n", encoding="utf-8")
PY

tmp_env="$(mktemp)"
python3 - "$MODE" "$WIFI_SSID" "$WIFI_PASSWORD" "$WIFI_STATIC_IP" "$WIFI_GATEWAY" "$WIFI_DNS" "$HOTSPOT_SSID" "$HOTSPOT_PASSWORD" "$HOTSPOT_IP" >"$tmp_env" <<'PY'
import shlex
import sys

mode, wifi_ssid, wifi_password, wifi_static_ip, wifi_gateway, wifi_dns, hotspot_ssid, hotspot_password, hotspot_ip = sys.argv[1:10]
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
echo "Hotspot fallback : $HOTSPOT_SSID / password configured on-device / http://${HOTSPOT_IP%/*}:8080"
if [ "$START_NOW" != "1" ]; then
  echo "Services are installed and will start on next boot. Set AGRIBOT_START_NOW=1 to start/switch network immediately."
fi
echo "Check:"
echo "  systemctl status agribot-access.service agribot-farmer.service"
echo "  cat agribot_inference_data/dashboard_url.txt"
""",
        executable=True,
    )


def write_config(pkg: Path, args, manifest: list[dict], included: dict):
    config = {
        "package": {
            "name": pkg.name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "purpose": "Raspberry Pi 5 CPU-only OpenVINO classifier deployment",
        },
        "deployment_defaults": {
            "runtime": "openvino",
            "classifier_model": "runtime_exports/classifier_openvino_model",
            "classifier_pt_fallback": "models/classifier_deploy_fastcrop.pt",
            "threads_normal": args.deploy_threads,
            "threads_safe_test": args.safe_test_threads,
            "fps": args.fps,
            "max_edge": args.max_edge,
            "crop_mode": args.crop_mode,
            "crop_scale": args.crop_scale,
            "crop_pad": args.crop_pad,
            "conf": args.conf,
            "high_conf": args.high_conf,
            "detector_enabled": False,
        },
        "known_good_pi_result": {
            "threads": 1,
            "limit": 40,
            "avg_fps": 9.92,
            "max_latency_ms": 154,
            "status": "PASS under undervoltage-limited testing",
        },
        "low_power_5v3a_experiment": {
            "script": "43_low_power_5v3a_experiment.py",
            "benchmark_helper": "deployment/10_low_power_5v3a_benchmark.sh",
            "live_helper": "deployment/11_low_power_5v3a_live.sh",
            "threads": args.low_power_threads,
            "fps": args.fps,
            "detector_enabled": False,
            "nice": 5,
            "cpu_affinity": args.low_power_cpu_affinity,
            "validated_pi_result": {
                "threads": 2,
                "cpu_affinity": "0,1",
                "wall_fps": 5.0,
                "avg_latency_ms": 115,
                "max_latency_ms": 120,
                "cpu_duty_cycle": 0.576,
                "emitted_accuracy": 0.9732,
                "coverage": 0.9696,
                "status": "PASS",
            },
            "note": "Keeps calibrated classifier/crop/confidence settings but paces inference to 5 FPS for lower CPU duty cycle on a 5V/3A bank.",
        },
        "performance_10fps_fan_profile": {
            "benchmark_script": "deployment/12_performance_10fps_benchmark.sh",
            "live_script": "deployment/13_performance_10fps_live.sh",
            "threads": args.performance_threads,
            "fps": args.performance_fps,
            "cpu_affinity": args.performance_cpu_affinity,
            "validated_pi_result": {
                "images": 461,
                "wall_fps": 10.01,
                "active_fps": 13.66,
                "avg_latency_ms": 73,
                "max_latency_ms": 78,
                "over_200ms_frames": 0,
                "emitted_accuracy": 0.9732,
                "coverage": 0.9696,
                "end_temp_c": 60.4,
                "end_throttled": "0x0",
                "status": "PASS",
            },
            "note": "Use with fan/heatsink and 5V/5A supply. This is now the recommended performance profile.",
        },
        "field_platform": {
            "server_script": "agribot_platform/run_platform.py",
            "helper_script": "deployment/14_start_field_platform.sh",
            "data_root": "agribot_inference_data",
            "default_hotspot_url": "http://10.42.0.1:8080",
            "router_url": "http://<pi-ip>:8080",
            "offline_phone_cache": True,
            "note": "Serves completed and live inference runs over the local network. The browser caches the latest loaded run for phone access after closing/reopening.",
        },
        "android_apk": {
            "source_dir": "agribot_android_app",
            "build_helper": "45_build_android_apk.py",
            "default_pi_url": "http://10.42.0.1:8080",
            "lab_fallback_url": f"http://{args.pi_ip}:8080",
            "apk_output": "agribot_android_app/android/app/build/outputs/apk/debug/app-debug.apk",
            "packaged_debug_apk": "dist/agribot-field-app-debug.apk",
            "note": "Capacitor Android wrapper for the Pi dashboard/API. It caches the latest loaded run on the phone but does not run inference locally.",
        },
        "farmer_one_touch": {
            "script": "44_farmer_one_touch.py",
            "config": "farmer_config.json",
            "run_helper": "deployment/15_farmer_one_touch.sh",
            "autostart_helper": "deployment/16_install_farmer_autostart.sh",
            "archive_helper": "deployment/17_archive_field_cleanup.sh",
            "mapping": "Sequential row scan: the farmer scans Row A Plant 1, waits for a decision, then moves to Plant 2. The dashboard flags sick plants by row and plant number.",
        },
        "included": included,
        "files": manifest,
    }
    write_text(pkg / "deployment" / "deployment_config.json", json.dumps(config, indent=2) + "\n")


def write_readme(pkg: Path, args, archive_name: str | None):
    archive_hint = archive_name or f"{pkg.name}.tar.gz"
    readme = f"""# Agribot Pi Deployment Package

This package is calibrated for Raspberry Pi 5 CPU-only inference.

Normal deployment threading remains **{args.deploy_threads} threads**. Because this Pi currently reports undervoltage (`throttled=0x50000`), today's test scripts force **{args.safe_test_threads} thread**.

## Install On Pi

From your laptop:

```bash
scp {archive_hint} pi@{args.pi_ip}:~/
ssh pi@{args.pi_ip}
```

On the Pi:

```bash
mkdir -p ~/agribot
tar -xzf ~/{archive_hint} -C ~/agribot --strip-components=1
cd ~/agribot
```

## Safe Test Today

```bash
./deployment/00_power_check.sh
./deployment/02_safe_benchmark_threads1.sh
```

For the full classifier test with the safe thread count:

```bash
./deployment/03_full_benchmark_threads1.sh
```

For the full end-to-end software validation:

```bash
./deployment/08_full_end_to_end_validation_threads1.sh
```

For the `tomato_crop_disease` benchmark specifically:

```bash
./deployment/09_tomato_crop_disease_full_validation_threads1.sh
```

## Low-Power 5V/3A Experiment

This path uses the same calibrated classifier stats, but keeps detector off, pins two CPU cores, increases niceness, and paces selected frames to 5 FPS:

```bash
./deployment/10_low_power_5v3a_benchmark.sh
```

Live low-power run:

```bash
./deployment/11_low_power_5v3a_live.sh 0
```

## 10 FPS Fan/Heatsink Performance Profile

With the fan/heatsink and 5V/5A supply, this keeps the same classifier accuracy
and passes timing without throttling:

```bash
./deployment/12_performance_10fps_benchmark.sh
```

Live run:

```bash
./deployment/13_performance_10fps_live.sh picamera2
```

## Farmer Field Platform

Start the platform on the Pi:

```bash
./deployment/14_start_field_platform.sh
```

Open the printed `http://<pi-ip>:8080` URL from a phone on the same network.
Live scripts write organized runs to `agribot_inference_data/`, and the platform
keeps showing the newest completed run as long as the Pi is powered on.

No internet is required, but the phone still needs a local network path to the
Pi. The field-safe default is the Pi's own direct hotspot:

```bash
./deployment/18_enable_direct_phone_hotspot.sh
```

Then connect the phone to the configured hotspot (the password is supplied on-device
via `AGRIBOT_HOTSPOT_PASSWORD`) and set the app URL to
`http://10.42.0.1:8080`. To connect the Pi to any available
Wi-Fi instead:

```bash
./deployment/19_connect_configured_wifi.sh
```

Non-interactive Wi-Fi setup also works:

```bash
AGRIBOT_WIFI_SSID='FarmWifi' AGRIBOT_WIFI_PASSWORD='wifi-password' ./deployment/19_connect_configured_wifi.sh
```

## Android APK Wrapper

The Android wrapper source is included under `agribot_android_app/`. It defaults
to the Pi hotspot URL `http://10.42.0.1:8080`, also tries
`http://{args.pi_ip}:8080` for lab/router testing, caches the latest loaded run
on the phone, and does not run inference locally.

Build from the laptop/workstation:

```bash
python3 45_build_android_apk.py
```

If Java/Android SDK are installed, the debug APK is written to:

```text
agribot_android_app/android/app/build/outputs/apk/debug/app-debug.apk
dist/agribot-field-app-debug.apk
```

## One-Touch Farmer Mode

This is the final field mode. It starts the dashboard and live inference
together, writes mapped plant results, and keeps the software running:

```bash
./deployment/15_farmer_one_touch.sh
```

Install it once so it starts automatically when the Pi boots:

```bash
./deployment/16_install_farmer_autostart.sh
```

For the final field install, use the all-in-one installer. It installs both boot
services and defaults to direct hotspot mode without switching the current SSH
session immediately:

```bash
./deployment/20_install_final_field_system.sh
```

To make the Pi prefer a specific Wi-Fi while still falling back to hotspot:

```bash
AGRIBOT_NETWORK_MODE=auto \\
AGRIBOT_WIFI_SSID='FarmWifi' \\
AGRIBOT_WIFI_PASSWORD='wifi-password' \\
./deployment/20_install_final_field_system.sh
```

Set `AGRIBOT_START_NOW=1` only when you intentionally want the setup command to
switch the Pi network immediately. Otherwise it takes effect on the next boot.

The map uses the farmer-entered field setup from the dashboard/app: field name,
row count, plants per row, and active row. Plant numbering and movement timing
are calculated from defaults. The physical workflow is still sequential: point
the camera at one plant until the app emits a result, then move to the next
plant in the configured active row.

To archive field-unneeded training and benchmark files into a zip:

```bash
./deployment/17_archive_field_cleanup.sh
```

## Live Run Today

```bash
./deployment/05_live_safe_threads1.sh 0
```

## After Power Is Fixed

Use the normal 4-thread scripts:

```bash
./deployment/04_normal_benchmark_threads4_after_power_fix.sh
./deployment/06_live_normal_threads4_after_power_fix.sh 0
```

After that run:

```bash
vcgencmd get_throttled
```

The desired value is `throttled=0x0`. If it returns `0x50000`, power is still bad.

## Tomato Crop Disease

This dataset is YOLO detection, not 8-class classification. The smoke script uses the small detector by default to avoid stressing the current weak power setup:

```bash
./deployment/07_tomato_image_hit_smoke_threads1.sh
```

After power is fixed, you may pass a heavier detector path:

```bash
./deployment/07_tomato_image_hit_smoke_threads1.sh runtime_exports/detector_openvino_model
```
"""
    write_text(pkg / "README_PI_DEPLOY.md", readme)


def create_archive(pkg: Path) -> Path:
    archive = pkg.with_suffix(".tar.gz")
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(pkg, arcname=pkg.name)
    return archive


def main():
    args = parse_args()
    stamp = "" if args.no_stamp else "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (ROOT / args.out_dir).resolve()
    pkg = out_dir / f"{args.name}{stamp}"

    if pkg.exists():
        if not args.overwrite:
            raise SystemExit(f"[ERROR] Package directory exists; use --overwrite: {pkg}")
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)

    manifest: list[dict] = []
    included = {
        "classifier_pt": False,
        "classifier_openvino": False,
        "clf_test_set": False,
        "tomato_crop_disease": False,
        "nano_detector": False,
        "field_platform": False,
        "android_app": False,
        "android_apk_debug": False,
    }

    for file_name in RUNTIME_FILES:
        copy_file(ROOT / file_name, pkg / file_name, manifest, pkg, required=True)

    copy_tree(ROOT / "agribot_platform", pkg / "agribot_platform", manifest, pkg, required=True)
    copy_tree(ROOT / "agribot_inference_data", pkg / "agribot_inference_data", manifest, pkg, required=False)
    included["field_platform"] = True

    copy_tree(ROOT / "agribot_android_app", pkg / "agribot_android_app", manifest, pkg, required=True)
    included["android_app"] = True

    debug_apk = ROOT / "dist" / "agribot-field-app-debug.apk"
    if debug_apk.exists():
        copy_file(debug_apk, pkg / "dist" / "agribot-field-app-debug.apk", manifest, pkg, required=False)
        included["android_apk_debug"] = True

    copy_file(ROOT / args.classifier_pt, pkg / "models" / "classifier_deploy_fastcrop.pt", manifest, pkg, required=True)
    included["classifier_pt"] = True

    openvino = ROOT / args.classifier_openvino
    if args.include_openvino and openvino.exists():
        copy_tree(openvino, pkg / "runtime_exports" / "classifier_openvino_model", manifest, pkg, required=False)
        included["classifier_openvino"] = True

    if args.include_test_set:
        copy_tree(ROOT / "clf_dataset" / "test", pkg / "clf_dataset" / "test", manifest, pkg, required=True)
        included["clf_test_set"] = True

    if args.include_tomato_dataset:
        copy_tree(ROOT / "tomato_crop_disease", pkg / "tomato_crop_disease", manifest, pkg, required=False)
        included["tomato_crop_disease"] = (pkg / "tomato_crop_disease").exists()

    if args.include_nano_detector:
        nano = ROOT / args.nano_detector
        if nano.exists():
            copy_file(nano, pkg / "models" / "detector_nano_256.pt", manifest, pkg, required=False)
            included["nano_detector"] = True

    write_helper_scripts(pkg, args)
    archive = None if args.no_archive else create_archive(pkg)
    write_config(pkg, args, manifest, included)

    if archive is not None:
        # Recreate the archive so README/config are included after they are written.
        archive = create_archive(pkg)
    write_readme(pkg, args, archive.name if archive else None)
    if archive is not None:
        archive = create_archive(pkg)

    print(f"Package dir : {pkg}")
    if archive is not None:
        print(f"Archive     : {archive}")
        print(f"Archive size: {archive.stat().st_size / (1024 * 1024):.1f} MB")
        print()
        print("Copy to Pi:")
        print(f"  scp {archive} pi@{args.pi_ip}:~/")
        print()
        print("Install on Pi:")
        print(f"  mkdir -p ~/agribot && tar -xzf ~/{archive.name} -C ~/agribot --strip-components=1")
        print("  cd ~/agribot && ./deployment/02_safe_benchmark_threads1.sh")
    else:
        print("Archive     : skipped")


if __name__ == "__main__":
    main()
