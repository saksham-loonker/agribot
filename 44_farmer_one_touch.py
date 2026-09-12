#!/usr/bin/env python3
"""
One-touch farmer controller for Agribot.

Default action:
  - start the local dashboard
  - start live camera inference
  - write organized mapped plant data
  - keep the process alive for systemd autostart

Other actions:
  - install-autostart: install and enable a systemd service
  - archive: zip field-unneeded training/benchmark artifacts
  - status: print the farmer URL and current service state
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "farmer_config.json"
STOP = False

ARCHIVE_CANDIDATES = [
    "archive",
    "dist",
    "runs",
    "__pycache__",
    ".pytest_cache",
    "clf_dataset",
    "clf_dataset_deploy_fastcrop",
    "tomato_crop_disease",
    "dataset/runtime_tracking",
    "runtime_reports/accuracy",
    "runtime_reports/benchmark",
]
ARCHIVE_PROTECTED_NAMES = {
    ".git",
    ".env",
    ".env.local",
    "farmer_config.json",
    "agribot_inference_data",
    "field_archives",
}


def handle_signal(signum, frame):
    global STOP
    STOP = True


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


def now_stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def safe_archive_target(raw_path: str | Path) -> Path:
    """Resolve an archive target without allowing path traversal or symlinks."""
    raw = Path(raw_path).expanduser()
    candidate = raw if raw.is_absolute() else ROOT / raw
    root = ROOT.resolve()
    lexical = Path(os.path.abspath(candidate))
    try:
        lexical_relative = lexical.relative_to(root)
    except ValueError as exc:
        raise ValueError("archive path must remain inside the repository") from exc
    current = lexical
    while current != root:
        if current.is_symlink():
            raise ValueError(f"archive path may not contain symlinks: {lexical_relative}")
        current = current.parent
    resolved = lexical.resolve()
    if resolved == root:
        raise ValueError("refusing to archive the repository root")
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("archive path must remain inside the repository") from exc
    if any(part in ARCHIVE_PROTECTED_NAMES for part in relative.parts):
        raise ValueError(f"archive path is protected: {relative}")
    return resolved


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return default


def cfg(args, config: dict, name: str, fallback=None):
    value = getattr(args, name, None)
    if value is not None:
        return value
    return config.get(name, fallback)


def safe_int(value, fallback: int, low: int = 1) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = fallback
    return max(low, number)


def safe_float(value, fallback: float, low: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = fallback
    return max(low, number)


def row_id_for_index(index: int) -> str:
    value = ""
    number = max(0, int(index))
    while True:
        number, rem = divmod(number, 26)
        value = chr(65 + rem) + value
        if number == 0:
            return value
        number -= 1


def row_ids(count: int) -> list[str]:
    return [row_id_for_index(i) for i in range(max(1, int(count or 1)))]


def active_field_layout(layout: dict) -> dict:
    if not isinstance(layout, dict):
        return {}
    fields = layout.get("fields")
    if isinstance(fields, list) and fields:
        active_id = layout.get("active_field_id") or fields[0].get("id")
        for field in fields:
            if isinstance(field, dict) and field.get("id") == active_id:
                return field
        first = fields[0]
        return first if isinstance(first, dict) else {}
    return layout


def ensure_field_rows(field: dict) -> dict:
    row_count = safe_int(field.get("row_count"), 1)
    default_plants = safe_int(field.get("plants_per_row"), 100)
    default_row_spacing = safe_float(field.get("row_spacing_m"), 1.0)
    default_plant_spacing = safe_float(field.get("plant_spacing_m"), 0.5)
    existing = {
        str(row.get("id") or row.get("row_id") or "").strip().upper(): row
        for row in field.get("rows", [])
        if isinstance(row, dict)
    }
    rows = []
    y_m = 0.0
    for index, row_id in enumerate(row_ids(row_count), start=1):
        raw = existing.get(row_id, {})
        row_spacing = safe_float(raw.get("row_spacing_m"), default_row_spacing)
        rows.append(
            {
                "id": row_id,
                "row_id": row_id,
                "row_index": index,
                "plants_per_row": safe_int(raw.get("plants_per_row"), default_plants),
                "plant_spacing_m": safe_float(raw.get("plant_spacing_m"), default_plant_spacing),
                "row_spacing_m": row_spacing,
                "y_m": round(safe_float(raw.get("y_m"), y_m), 4),
            }
        )
        y_m += row_spacing
    field["rows"] = rows
    field["row_ids"] = [row["id"] for row in rows]
    field["planned_total_plants"] = sum(row["plants_per_row"] for row in rows)
    return field


def local_ip() -> str:
    try:
        output = subprocess.check_output(["hostname", "-I"], text=True, timeout=0.5)
        for item in output.split():
            if item and not item.startswith("127."):
                return item
    except Exception:
        pass
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.2)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception:
        return "127.0.0.1"


def url_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1.5) as response:
            return 200 <= response.status < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def network_cfg(config: dict) -> dict:
    defaults = {
        "mode": "hotspot",
        "wifi_ssid": "",
        "wifi_static_ip": "",
        "wifi_gateway": "",
        "wifi_dns": "1.1.1.1 8.8.8.8",
        "hotspot_ssid": "Agribot_Field",
        "hotspot_password": "",
        "hotspot_ip": "10.42.0.1/24",
        "auto_wifi_wait_sec": 12,
    }
    user_cfg = config.get("network") if isinstance(config.get("network"), dict) else {}
    defaults.update(user_cfg)
    env_overrides = {
        "mode": "AGRIBOT_NETWORK_MODE",
        "wifi_ssid": "AGRIBOT_WIFI_SSID",
        "wifi_static_ip": "AGRIBOT_WIFI_STATIC_IP",
        "wifi_gateway": "AGRIBOT_WIFI_GATEWAY",
        "wifi_dns": "AGRIBOT_WIFI_DNS",
        "hotspot_ssid": "AGRIBOT_HOTSPOT_SSID",
        "hotspot_password": "AGRIBOT_HOTSPOT_PASSWORD",
        "hotspot_ip": "AGRIBOT_HOTSPOT_IP",
        "auto_wifi_wait_sec": "AGRIBOT_AUTO_WIFI_WAIT_SEC",
    }
    for key, env_name in env_overrides.items():
        value = os.environ.get(env_name)
        if value is not None and value.strip():
            defaults[key] = value.strip()
    return defaults


def apply_network_arg_overrides(args, net: dict) -> dict:
    arg_to_key = {
        "wifi_ssid": "wifi_ssid",
        "wifi_static_ip": "wifi_static_ip",
        "wifi_gateway": "wifi_gateway",
        "wifi_dns": "wifi_dns",
        "hotspot_ssid": "hotspot_ssid",
        "hotspot_password": "hotspot_password",
        "hotspot_ip": "hotspot_ip",
    }
    for arg_name, key in arg_to_key.items():
        value = getattr(args, arg_name, None)
        if value is not None and str(value).strip():
            net[key] = str(value).strip()
    return net


def run_nmcli(args: list[str], *, check: bool = True):
    return subprocess.run(["nmcli", *args], text=True, capture_output=True, check=check)


def have_nmcli() -> bool:
    return shutil.which("nmcli") is not None


def connection_exists(name: str) -> bool:
    result = run_nmcli(["-t", "-f", "NAME", "connection", "show"], check=False)
    return name in result.stdout.splitlines()


def active_connection_name(name: str) -> bool:
    result = run_nmcli(["-t", "-f", "NAME", "connection", "show", "--active"], check=False)
    return name in result.stdout.splitlines()


def wait_for_ip(expected_prefix: str | None = None, *, timeout_sec: int = 20) -> str:
    deadline = time.monotonic() + max(1, timeout_sec)
    while time.monotonic() < deadline:
        ip = local_ip()
        if ip != "127.0.0.1" and (expected_prefix is None or ip.startswith(expected_prefix)):
            return ip
        time.sleep(1)
    return local_ip()


def configure_wifi(net: dict) -> bool:
    ssid = str(net.get("wifi_ssid") or "").strip()
    if not ssid:
        return False
    password = os.environ.get("AGRIBOT_WIFI_PASSWORD") or str(net.get("wifi_password") or "")

    if connection_exists(ssid):
        run_nmcli(["connection", "modify", ssid, "connection.autoconnect", "yes", "connection.autoconnect-priority", "100"], check=False)
        if password:
            run_nmcli(["connection", "modify", ssid, "wifi-sec.key-mgmt", "wpa-psk", "wifi-sec.psk", password], check=False)
    elif password:
        run_nmcli(["radio", "wifi", "on"], check=False)
        run_nmcli(["device", "wifi", "rescan"], check=False)
        created = run_nmcli(["device", "wifi", "connect", ssid, "password", password, "ifname", "wlan0", "name", ssid], check=False)
        if created.returncode != 0:
            print(f"[WARN] Could not create Wi-Fi connection {ssid}: {created.stderr.strip()}", flush=True)
            return False
    else:
        print(f"[WARN] Wi-Fi connection {ssid} is not saved and AGRIBOT_WIFI_PASSWORD is not set.", flush=True)
        return False

    static_ip = str(net.get("wifi_static_ip") or "").strip()
    gateway = str(net.get("wifi_gateway") or "").strip()
    dns = str(net.get("wifi_dns") or "").strip()
    if static_ip and gateway:
        run_nmcli(
            [
                "connection",
                "modify",
                ssid,
                "ipv4.method",
                "manual",
                "ipv4.addresses",
                static_ip,
                "ipv4.gateway",
                gateway,
                "ipv4.dns",
                dns,
                "ipv6.method",
                "ignore",
            ],
            check=False,
        )
    else:
        run_nmcli(["connection", "modify", ssid, "ipv4.method", "auto", "ipv6.method", "ignore"], check=False)
        if dns:
            run_nmcli(["connection", "modify", ssid, "ipv4.dns", dns], check=False)
    up = run_nmcli(["connection", "up", ssid], check=False)
    if up.returncode != 0:
        print(f"[WARN] Could not bring up Wi-Fi {ssid}: {up.stderr.strip()}", flush=True)
        return False
    wait_for_ip(static_ip.split(".")[0] + "." if static_ip else None, timeout_sec=int(net.get("auto_wifi_wait_sec", 20)))
    return active_connection_name(ssid)


def configure_hotspot(net: dict) -> bool:
    ssid = str(net.get("hotspot_ssid") or "Agribot_Field").strip()
    password = str(net.get("hotspot_password") or "").strip()
    if not password:
        raise SystemExit(
            "[ERROR] Hotspot password is not configured. Set AGRIBOT_HOTSPOT_PASSWORD "
            "or provide it in farmer_config.json on the device."
        )
    ip_cidr = str(net.get("hotspot_ip") or "10.42.0.1/24").strip()
    if len(password) < 8:
        raise SystemExit("[ERROR] Hotspot password must be at least 8 characters.")

    run_nmcli(["radio", "wifi", "on"], check=False)
    run_nmcli(["connection", "delete", ssid], check=False)
    run_nmcli(["connection", "add", "type", "wifi", "ifname", "wlan0", "con-name", ssid, "autoconnect", "yes", "ssid", ssid], check=True)
    run_nmcli(
        [
            "connection",
            "modify",
            ssid,
            "802-11-wireless.mode",
            "ap",
            "802-11-wireless.band",
            "bg",
            "ipv4.method",
            "shared",
            "ipv4.addresses",
            ip_cidr,
            "ipv6.method",
            "ignore",
            "wifi-sec.key-mgmt",
            "wpa-psk",
            "wifi-sec.psk",
            password,
        ],
        check=True,
    )
    up = run_nmcli(["connection", "up", ssid], check=False)
    if up.returncode != 0:
        print(f"[ERROR] Could not start hotspot {ssid}: {up.stderr.strip()}", flush=True)
        return False
    wait_for_ip(ip_cidr.removesuffix("/24").rsplit(".", 1)[0] + ".", timeout_sec=10)
    return active_connection_name(ssid)


def write_access_state(data_root: str, state: dict):
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "network_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    (root / "dashboard_url.txt").write_text(state.get("dashboard_url", "") + "\n", encoding="utf-8")


def prepare_access(args):
    config = load_config(Path(args.config))
    net = apply_network_arg_overrides(args, network_cfg(config))
    port = int(cfg(args, config, "dashboard_port", 8080))
    data_root = str(cfg(args, config, "data_root", "agribot_inference_data"))
    mode = str(getattr(args, "mode", None) or net.get("mode", "auto")).lower()
    if mode not in {"auto", "wifi", "hotspot", "existing"}:
        raise SystemExit(f"[ERROR] Invalid network mode: {mode}")
    if mode != "existing":
        if os.geteuid() != 0:
            raise SystemExit("Run access setup with sudo, or install autostart so systemd runs it as root.")
        if not have_nmcli():
            raise SystemExit("[ERROR] nmcli/NetworkManager is required for automated access setup.")

    selected = "existing"
    if mode == "wifi":
        if not configure_wifi(net):
            raise SystemExit("[ERROR] Wi-Fi setup failed.")
        selected = "wifi"
    elif mode == "hotspot":
        if not configure_hotspot(net):
            raise SystemExit("[ERROR] Hotspot setup failed.")
        selected = "hotspot"
    elif mode == "auto":
        selected = "wifi" if configure_wifi(net) else "hotspot"
        if selected == "hotspot" and not configure_hotspot(net):
            raise SystemExit("[ERROR] Auto network setup failed: Wi-Fi unavailable and hotspot failed.")

    ip = local_ip()
    if selected == "hotspot":
        ip = str(net.get("hotspot_ip", "10.42.0.1/24")).split("/", 1)[0]
    elif selected == "wifi":
        static_ip = str(net.get("wifi_static_ip") or "").strip()
        ip = static_ip.split("/", 1)[0] if static_ip else local_ip()
    state = {
        "mode": selected,
        "dashboard_url": f"http://{ip}:{port}",
        "apk_url": f"http://{ip}:{port}/agribot-field-app-debug.apk",
        "wifi_ssid": net.get("wifi_ssid"),
        "hotspot_ssid": net.get("hotspot_ssid"),
        # Never persist or print the reusable Wi-Fi credential in runtime state.
        "hotspot_password_configured": bool(net.get("hotspot_password")) if selected == "hotspot" else False,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    write_access_state(data_root, state)
    print("Agribot access ready")
    print(f"  mode : {state['mode']}")
    print(f"  app  : {state['dashboard_url']}")
    if selected == "hotspot":
        print(f"  wifi : {state['hotspot_ssid']}")
        print("  pass : configured on device (not printed)")
    return state


def open_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a", encoding="utf-8")


def start_platform(port: int, data_root: str, log_dir: Path):
    url = f"http://127.0.0.1:{port}/api/status"
    if url_ok(url):
        return None
    log = open_log(log_dir / "platform.log")
    return subprocess.Popen(
        [
            sys.executable,
            "agribot_platform/run_platform.py",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--data-root",
            data_root,
            "--quiet",
        ],
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )


def live_command(args, config: dict, run_id: str) -> list[str]:
    layout = config.get("field_layout") if isinstance(config.get("field_layout"), dict) else {}
    active = active_field_layout(layout)
    field_id = getattr(args, "field_id", None) or config.get("field_id") or active.get("field_id") or active.get("name") or layout.get("field_id") or "Field 1"
    row_id = getattr(args, "row_id", None) or config.get("row_id") or active.get("active_row_id") or layout.get("active_row_id") or "A"
    row_count = active.get("row_count") or layout.get("row_count") or config.get("row_count") or 1
    plants_per_row = active.get("plants_per_row") or layout.get("plants_per_row") or config.get("plants_per_row") or 100
    start_plant = getattr(args, "start_plant", None) or config.get("start_plant") or active.get("start_plant") or layout.get("start_plant") or 1
    plant_step = getattr(args, "plant_step", None) or config.get("plant_step") or active.get("plant_step") or layout.get("plant_step") or 1
    row_spacing = active.get("row_spacing_m") or layout.get("row_spacing_m") or config.get("row_spacing_m") or 1.0
    plant_spacing = active.get("plant_spacing_m") or layout.get("plant_spacing_m") or config.get("plant_spacing_m") or 0.5
    plant_cooldown = (
        getattr(args, "plant_cooldown_sec", None)
        if getattr(args, "plant_cooldown_sec", None) is not None
        else config.get("plant_cooldown_sec") or active.get("plant_cooldown_sec") or layout.get("plant_cooldown_sec") or 2.0
    )
    return [
        sys.executable,
        "43_low_power_5v3a_experiment.py",
        "--mode",
        "live",
        "--camera",
        str(cfg(args, config, "camera", "picamera2")),
        "--camera-width",
        "640",
        "--camera-height",
        "480",
        "--picamera-raw-width",
        "2304",
        "--picamera-raw-height",
        "1296",
        "--clf-model",
        str(cfg(args, config, "classifier", "runtime_exports/classifier_openvino_model")),
        "--threads",
        str(cfg(args, config, "threads", 4)),
        "--fps",
        str(cfg(args, config, "fps", 10.0)),
        "--cpu-affinity",
        str(cfg(args, config, "cpu_affinity", "0,1,2,3")),
        "--max-edge",
        "256",
        "--crop-mode",
        "mask",
        "--crop-scale",
        "0.55",
        "--crop-pad",
        "0.05",
        "--conf",
        "0.765",
        "--high-conf",
        "0.765",
        "--jsonl",
        "runtime_reports/field/plant_loop.jsonl",
        "--data-root",
        str(cfg(args, config, "data_root", "agribot_inference_data")),
        "--run-id",
        run_id,
        "--run-note",
        "Farmer one-touch live scan",
        "--field-id",
        str(field_id),
        "--row-id",
        str(row_id),
        "--row-count",
        str(row_count),
        "--plants-per-row",
        str(plants_per_row),
        "--start-plant",
        str(start_plant),
        "--plant-step",
        str(plant_step),
        "--row-spacing-m",
        str(row_spacing),
        "--plant-spacing-m",
        str(plant_spacing),
        "--plant-cooldown-sec",
        str(plant_cooldown),
        "--nice",
        "5",
    ]


def normalized_field_layout(config: dict) -> dict:
    layout = config.get("field_layout") if isinstance(config.get("field_layout"), dict) else {}
    if isinstance(layout.get("fields"), list) and layout["fields"]:
        normalized = {
            "active_field_id": layout.get("active_field_id") or layout["fields"][0].get("id") or "field_1",
            "fields": [],
            "updated_at": layout.get("updated_at") or datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        for index, raw_field in enumerate(layout["fields"], start=1):
            if not isinstance(raw_field, dict):
                continue
            field = dict(raw_field)
            field.setdefault("id", f"field_{index}")
            field.setdefault("name", field.get("field_id") or f"Field {index}")
            field["field_id"] = field.get("field_id") or field["name"]
            field["active_row_id"] = field.get("active_row_id") or "A"
            field["row_count"] = safe_int(field.get("row_count"), 1)
            field["plants_per_row"] = safe_int(field.get("plants_per_row"), 100)
            field["start_plant"] = safe_int(field.get("start_plant"), 1)
            field["plant_step"] = safe_int(field.get("plant_step"), 1)
            field["plant_cooldown_sec"] = safe_float(field.get("plant_cooldown_sec"), 2.0)
            field["row_spacing_m"] = safe_float(field.get("row_spacing_m"), 1.0)
            field["plant_spacing_m"] = safe_float(field.get("plant_spacing_m"), 0.5)
            normalized["fields"].append(ensure_field_rows(field))
        if not normalized["fields"]:
            normalized["fields"].append(ensure_field_rows({"id": "field_1", "name": "Field 1", "field_id": "Field 1"}))
        if normalized["active_field_id"] not in {field["id"] for field in normalized["fields"]}:
            normalized["active_field_id"] = normalized["fields"][0]["id"]
        active = active_field_layout(normalized)
        normalized.update(
            {
                "field_id": active.get("field_id") or active.get("name") or "Field 1",
                "field_name": active.get("name") or active.get("field_id") or "Field 1",
                "active_row_id": active.get("active_row_id") or "A",
                "row_count": active.get("row_count") or 1,
                "plants_per_row": active.get("plants_per_row") or 100,
                "start_plant": active.get("start_plant") or 1,
                "plant_step": active.get("plant_step") or 1,
                "plant_cooldown_sec": active.get("plant_cooldown_sec") or 2.0,
                "row_spacing_m": active.get("row_spacing_m") or 1.0,
                "plant_spacing_m": active.get("plant_spacing_m") or 0.5,
                "rows": active.get("rows") or [],
                "row_ids": active.get("row_ids") or [],
                "planned_total_plants": active.get("planned_total_plants") or 0,
            }
        )
        return normalized
    return {
        "field_id": config.get("field_id") or layout.get("field_id") or "Field 1",
        "active_row_id": config.get("row_id") or layout.get("active_row_id") or "A",
        "row_count": safe_int(layout.get("row_count") or config.get("row_count"), 1),
        "plants_per_row": safe_int(layout.get("plants_per_row") or config.get("plants_per_row"), 100),
        "start_plant": safe_int(config.get("start_plant") or layout.get("start_plant"), 1),
        "plant_step": safe_int(config.get("plant_step") or layout.get("plant_step"), 1),
        "plant_cooldown_sec": safe_float(config.get("plant_cooldown_sec") or layout.get("plant_cooldown_sec"), 2.0),
        "row_spacing_m": safe_float(layout.get("row_spacing_m"), 1.0),
        "plant_spacing_m": safe_float(layout.get("plant_spacing_m"), 0.5),
    }


def prepare_runtime_state(data_root: str, config: dict, run_id: str):
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    layout = normalized_field_layout(config)
    if "fields" in layout:
        active = active_field_layout(layout)
    else:
        active = ensure_field_rows(
            {
                "id": "field_1",
                "name": layout["field_id"],
                "field_id": layout["field_id"],
                **layout,
            }
        )
        layout["rows"] = active["rows"]
        layout["row_ids"] = active["row_ids"]
        layout["planned_total_plants"] = active["planned_total_plants"]
    active_map = {
        **active,
        "field_id": active.get("field_id") or active.get("name") or layout.get("field_id") or "Field 1",
        "active_row_id": active.get("active_row_id") or layout.get("active_row_id") or "A",
    }
    layout["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    (root / "field_layout.json").write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")

    runs_root = root / "runs"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    now = layout["updated_at"]
    metadata = {
        "run_id": run_id,
        "started_at": now,
        "completed_at": None,
        "note": "Boot preload; live inference is starting.",
        "mode": "live",
        "field_map": {
            **active_map,
            "row_id": active_map["active_row_id"],
            "mapping_method": "Sequential row/column scan with automatic row transition after plants_per_row decisions.",
        },
        "farm_layout": layout,
    }
    summary = {
        "run_id": run_id,
        "started_at": now,
        "last_updated_at": now,
        "completed": False,
        "decisions": 0,
        "ok": 0,
        "sick": 0,
        "uncertain": 0,
        "uncertain_rate": 0.0,
        "avg_confidence": 0.0,
        "max_confidence": 0.0,
        "total_late_frames": 0,
        "labels": {},
        "statuses": {},
        "latest_decision": None,
        "paths": {
            "run_dir": str(run_dir),
            "metadata": str(run_dir / "metadata.json"),
            "summary": str(run_dir / "summary.json"),
            "decisions": str(run_dir / "decisions.jsonl"),
            "events_csv": str(run_dir / "events.csv"),
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (run_dir / "decisions.jsonl").touch()
    (root / "latest_run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "updated_at": now,
                "completed": False,
                "run_dir": str(run_dir),
                "summary": str(run_dir / "summary.json"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "boot_ready.json").write_text(
        json.dumps(
            {
                "ready_at": layout["updated_at"],
                "run_id": run_id,
                "field_layout": layout,
                "mapping": "row-major automatic row/plant transition",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return layout


def run_controller(args):
    config = load_config(Path(args.config))
    layout = config.get("field_layout") if isinstance(config.get("field_layout"), dict) else {}
    for key in ("field_id", "row_id", "row_count", "plants_per_row", "start_plant", "plant_step", "row_spacing_m", "plant_spacing_m", "plant_cooldown_sec"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
            layout_key = "active_row_id" if key == "row_id" else key
            layout[layout_key] = value
    if layout:
        config["field_layout"] = layout
    port = int(cfg(args, config, "dashboard_port", 8080))
    data_root = str(cfg(args, config, "data_root", "agribot_inference_data"))
    run_id = getattr(args, "run_id", "") or f"field_{now_stamp()}"
    log_dir = ROOT / "runtime_reports" / "farmer_control" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    field_layout = prepare_runtime_state(data_root, config, run_id)

    if getattr(args, "archive_on_start", False):
        archive_namespace(
            args,
            keep_originals=not getattr(args, "delete_archive_originals", False)
            or getattr(args, "keep_archive_originals", False),
        )

    platform_proc = start_platform(port, data_root, log_dir)
    access_state = read_json(Path(data_root) / "network_state.json", {})
    phone_url = access_state.get("dashboard_url") or f"http://{local_ip()}:{port}"
    (Path(data_root) / "dashboard_url.txt").parent.mkdir(parents=True, exist_ok=True)
    (Path(data_root) / "dashboard_url.txt").write_text(phone_url + "\n", encoding="utf-8")

    print("Agribot farmer controller")
    print(f"  dashboard : {phone_url}")
    print(f"  run id    : {run_id}")
    print(f"  row map   : {field_layout['field_id']} / Row {field_layout['active_row_id']} / {field_layout['row_count']}x{field_layout['plants_per_row']} plants")
    print(f"  logs      : {log_dir}")

    live_log = open_log(log_dir / "live_inference.log")
    live_proc = subprocess.Popen(live_command(args, config, run_id), cwd=ROOT, stdout=live_log, stderr=subprocess.STDOUT, text=True)

    try:
        while not STOP:
            if live_proc.poll() is not None:
                print(f"[WARN] live inference exited with {live_proc.returncode}; restarting in 5s", flush=True)
                time.sleep(5)
                live_log.close()
                live_log = open_log(log_dir / "live_inference.log")
                live_proc = subprocess.Popen(live_command(args, config, run_id), cwd=ROOT, stdout=live_log, stderr=subprocess.STDOUT, text=True)
            if platform_proc is not None and platform_proc.poll() is not None:
                print(f"[WARN] platform exited with {platform_proc.returncode}; restarting", flush=True)
                platform_proc = start_platform(port, data_root, log_dir)
            time.sleep(1)
    finally:
        live_proc.terminate()
        try:
            live_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            live_proc.kill()
        live_log.close()
        if platform_proc is not None:
            platform_proc.terminate()


def archive_namespace(args, *, keep_originals: bool):
    archive_root = ROOT / "field_archives"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_path = archive_root / f"agribot_field_archive_{now_stamp()}.zip"
    try:
        candidates = [safe_archive_target(item) for item in ARCHIVE_CANDIDATES]
        candidates.extend(safe_archive_target(item) for item in getattr(args, "extra_archive", []))
    except ValueError as exc:
        raise SystemExit(f"[ERROR] Unsafe archive target: {exc}") from exc
    existing = [path for path in candidates if path.exists()]
    if getattr(args, "dry_run", False):
        print("Would archive:")
        for path in existing:
            print(f"  {path.relative_to(ROOT)}")
        return archive_path
    if not existing:
        print("No archive candidates found.")
        return archive_path

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in existing:
            if path.is_dir():
                for child in path.rglob("*"):
                    if child.is_symlink():
                        raise SystemExit(f"[ERROR] Refusing to archive symlink: {child.relative_to(ROOT)}")
                    if child.is_file():
                        zf.write(child, child.relative_to(ROOT))
            else:
                if path.is_symlink():
                    raise SystemExit(f"[ERROR] Refusing to archive symlink: {path.relative_to(ROOT)}")
                zf.write(path, path.relative_to(ROOT))
    with zipfile.ZipFile(archive_path) as zf:
        bad = zf.testzip()
        if bad:
            raise SystemExit(f"[ERROR] Archive verification failed at {bad}")

    if not keep_originals:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    print(f"Archive written: {archive_path}")
    print(f"Originals kept : {'yes' if keep_originals else 'no'}")
    return archive_path


def install_autostart(args):
    if os.geteuid() != 0:
        raise SystemExit("Run this once with sudo: sudo python3 44_farmer_one_touch.py install-autostart")
    user = args.user
    user_python = Path(f"/home/{user}/agribot_venv/bin/python3")
    python_path = str(user_python if user_python.exists() else Path(sys.executable))
    root_python = str(Path(sys.executable))
    config_path = Path(args.config).resolve()
    access_env = Path("/etc/agribot/access.env")
    access_env.parent.mkdir(parents=True, exist_ok=True)
    if not access_env.exists():
        access_env.write_text(
            "# Optional network overrides for Agribot boot access\n"
            "# AGRIBOT_NETWORK_MODE=hotspot|auto|wifi|existing\n"
            "# AGRIBOT_WIFI_SSID=FarmWifiName\n"
            "# AGRIBOT_WIFI_PASSWORD=your-wifi-password\n"
            "# AGRIBOT_WIFI_STATIC_IP=192.168.1.25/24\n"
            "# AGRIBOT_WIFI_GATEWAY=192.168.1.1\n"
            "# AGRIBOT_PLATFORM_TOKEN=generate-a-long-random-local-token\n",
            encoding="utf-8",
        )
        access_env.chmod(0o600)
    access_service_path = Path("/etc/systemd/system/agribot-access.service")
    access_service = f"""[Unit]
Description=Agribot local phone access setup
After=NetworkManager.service
Wants=NetworkManager.service
Before=agribot-farmer.service

[Service]
Type=oneshot
WorkingDirectory={ROOT}
Environment=PYTHONUNBUFFERED=1
EnvironmentFile=-/etc/agribot/access.env
ExecStart={root_python} {ROOT / '44_farmer_one_touch.py'} access --config {config_path}
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
"""
    service_path = Path("/etc/systemd/system/agribot-farmer.service")
    service = f"""[Unit]
Description=Agribot farmer one-touch controller
After=agribot-access.service
Wants=agribot-access.service

[Service]
Type=simple
User={user}
WorkingDirectory={ROOT}
Environment=PYTHONUNBUFFERED=1
EnvironmentFile=-/etc/agribot/access.env
ExecStart={python_path} {ROOT / '44_farmer_one_touch.py'} run --config {config_path}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
"""
    access_service_path.write_text(access_service, encoding="utf-8")
    service_path.write_text(service, encoding="utf-8")
    subprocess.run(["systemctl", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "enable", "agribot-access.service"], check=True)
    subprocess.run(["systemctl", "enable", "agribot-farmer.service"], check=True)
    if args.start_now:
        subprocess.run(["systemctl", "restart", "agribot-access.service"], check=True)
        subprocess.run(["systemctl", "restart", "agribot-farmer.service"], check=True)
    print("Installed agribot-access.service and agribot-farmer.service for automatic boot startup")
    print("Optional Wi-Fi password file: /etc/agribot/access.env")
    if args.start_now:
        print("Started Agribot access and farmer services")
    print("Check status: systemctl status agribot-access.service agribot-farmer.service")


def status(args):
    config = load_config(Path(args.config))
    port = int(cfg(args, config, "dashboard_port", 8080))
    data_root = str(cfg(args, config, "data_root", "agribot_inference_data"))
    access_state = read_json(Path(data_root) / "network_state.json", {})
    phone_url = f"http://{local_ip()}:{port}"
    print(f"Dashboard: {access_state.get('dashboard_url') or phone_url}")
    if access_state:
        print(f"Access   : {access_state.get('mode')} {access_state.get('hotspot_ssid') or access_state.get('wifi_ssid')}")
    print(f"Platform : {'up' if url_ok(f'http://127.0.0.1:{port}/api/status') else 'down'}")
    subprocess.run(["pgrep", "-af", "44_farmer_one_touch.py|43_low_power_5v3a_experiment.py|agribot_platform/run_platform.py"], check=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Agribot one-touch farmer controller")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Start dashboard and live inference")
    run.add_argument("--config", default=str(DEFAULT_CONFIG))
    run.add_argument("--run-id", default="")
    run.add_argument("--archive-on-start", action="store_true")
    run.add_argument("--keep-archive-originals", action="store_true")
    run.add_argument("--delete-archive-originals", action="store_true")
    run.add_argument("--dashboard-port", type=int)
    run.add_argument("--camera")
    run.add_argument("--field-id")
    run.add_argument("--row-id")
    run.add_argument("--row-count", type=int)
    run.add_argument("--plants-per-row", type=int)
    run.add_argument("--start-plant", type=int)
    run.add_argument("--plant-step", type=int)
    run.add_argument("--row-spacing-m", type=float)
    run.add_argument("--plant-spacing-m", type=float)
    run.add_argument("--plant-cooldown-sec", type=float)
    run.add_argument("--fps", type=float)
    run.add_argument("--threads", type=int)
    run.add_argument("--cpu-affinity")
    run.add_argument("--classifier")
    run.add_argument("--data-root")

    install = sub.add_parser("install-autostart", help="Install systemd autostart service")
    install.add_argument("--config", default=str(DEFAULT_CONFIG))
    install.add_argument("--user", default="pi")
    install.add_argument("--start-now", action="store_true")

    access = sub.add_parser("access", help="Prepare phone access via saved Wi-Fi or direct hotspot")
    access.add_argument("--config", default=str(DEFAULT_CONFIG))
    access.add_argument("--mode", choices=["auto", "wifi", "hotspot", "existing"])
    access.add_argument("--wifi-ssid")
    access.add_argument("--wifi-static-ip")
    access.add_argument("--wifi-gateway")
    access.add_argument("--wifi-dns")
    access.add_argument("--hotspot-ssid")
    access.add_argument("--hotspot-password")
    access.add_argument("--hotspot-ip")

    archive = sub.add_parser("archive", help="Zip and optionally remove field-unneeded artifacts")
    archive.add_argument("--keep-originals", action="store_true")
    archive.add_argument(
        "--delete-originals",
        action="store_true",
        help="Delete the allow-listed originals only after archive verification",
    )
    archive.add_argument("--dry-run", action="store_true")
    archive.add_argument("--extra-archive", action="append", default=[])

    stat = sub.add_parser("status", help="Print platform and process status")
    stat.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def main():
    args = parse_args()
    command = args.command or "run"
    if command == "run":
        run_controller(args)
    elif command == "access":
        prepare_access(args)
    elif command == "install-autostart":
        install_autostart(args)
    elif command == "archive":
        archive_namespace(args, keep_originals=not args.delete_originals or args.keep_originals)
    elif command == "status":
        status(args)
    else:
        raise SystemExit(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
