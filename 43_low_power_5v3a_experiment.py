#!/usr/bin/env python3
"""
Low-power Raspberry Pi 5 experiment for a 5V/3A battery bank.

This keeps the calibrated deployment behavior:
  - OpenVINO classifier
  - detector disabled
  - mask crop 0.55 with 0.05 pad
  - confidence gate threshold 0.765
  - 5 FPS selected-frame schedule

The key difference from the normal benchmark is that this script deliberately
paces inference to the required 5 FPS and defaults to the proven two-thread
low-power profile. That reduces CPU duty cycle and battery stress while
reporting the same deployment-critical stats: accuracy, emitted accuracy,
coverage, FPS, latency, and pass/fail.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2

from inference_rpi import (
    HIGH_CONF_THRESHOLD,
    MAX_FRAME_LATENCY_MS,
    MIN_FRAME_FPS,
    PREFERRED_FRAME_LATENCY_MS,
    PlantDecisionGate,
    _find_healthy_idx,
    infer_frame,
    load_models,
)
from runtime_suite_common import REPORTS_DIR, write_json


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
STOP = False


def _signal_handler(sig, frame):
    global STOP
    STOP = True
    print("\n[!] Interrupted - finishing report with completed frames...\n", flush=True)


signal.signal(signal.SIGINT, _signal_handler)


def parse_args():
    p = argparse.ArgumentParser(description="Low-power 5V/3A Pi deployment experiment")
    p.add_argument("--mode", choices=["benchmark", "live"], default="benchmark")
    p.add_argument("--test", default="clf_dataset/test", help="Class-folder test split for benchmark mode")
    p.add_argument("--camera", default="0", help="Camera index or video path for live mode")
    p.add_argument("--clf-model", default="runtime_exports/classifier_openvino_model")
    p.add_argument("--det-model", default="runtime_exports/detector_openvino_model")
    p.add_argument("--with-detector", action="store_true", help="Off by default for low-power deployment")

    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--sequential-limit", action="store_true")
    p.add_argument("--no-pace", action="store_true", help="Disable 5 FPS pacing and run as fast as possible")
    p.add_argument("--nice", type=int, default=5, help="Increase process niceness to reduce power contention")
    p.add_argument("--cpu-affinity", default="0,1", help="Comma-separated CPU cores to use; empty disables pinning")

    p.add_argument("--max-edge", type=int, default=256)
    p.add_argument("--crop-mode", choices=["mask", "center", "none"], default="mask")
    p.add_argument("--crop-scale", type=float, default=0.55)
    p.add_argument("--crop-pad", type=float, default=0.05)
    p.add_argument("--conf", type=float, default=0.765)
    p.add_argument("--high-conf", type=float, default=0.765)

    p.add_argument("--det-imgsz", type=int, default=256)
    p.add_argument("--det-conf", type=float, default=0.30)
    p.add_argument("--det-iou", type=float, default=0.45)
    p.add_argument("--max-leaves", type=int, default=1)

    p.add_argument("--camera-width", type=int, default=640)
    p.add_argument("--camera-height", type=int, default=480)
    p.add_argument("--picamera-raw-width", type=int, default=2304)
    p.add_argument("--picamera-raw-height", type=int, default=1296)
    p.add_argument("--max-live-frames", type=int, default=0, help="0 runs live mode until interrupted")
    p.add_argument("--jsonl", default="runtime_reports/low_power/plant_loop_low_power.jsonl")
    p.add_argument("--data-root", default="agribot_inference_data", help="Organized local-first inference data folder")
    p.add_argument("--run-id", default="", help="Optional stable run id; default is timestamped")
    p.add_argument("--run-note", default="", help="Optional note stored with run metadata")
    p.add_argument("--no-organized-data", action="store_true", help="Only write the legacy --jsonl file")
    p.add_argument("--field-id", default="Field 1", help="Human field/block name used for farmer mapping")
    p.add_argument("--row-id", default="A", help="Physical row name used for farmer mapping")
    p.add_argument("--row-count", type=int, default=1, help="Total rows in the field map")
    p.add_argument("--plants-per-row", type=int, default=100, help="Plant columns in each row before advancing to the next row")
    p.add_argument("--start-plant", type=int, default=1, help="First physical plant number in the row")
    p.add_argument("--plant-step", type=int, default=1, help="Plant number increment after each emitted decision")
    p.add_argument("--row-spacing-m", type=float, default=1.0, help="Physical distance between field rows, in meters")
    p.add_argument("--plant-spacing-m", type=float, default=0.5, help="Physical distance between plants in a row, in meters")
    p.add_argument("--plant-cooldown-sec", type=float, default=0.0, help="Pause after each decision so the farmer can move to the next plant")
    p.add_argument("--report", default="")
    p.add_argument("--sample-power-every", type=int, default=0, help="0 disables in-loop vcgencmd sampling")
    p.add_argument("--progress-every", type=int, default=25, help="0 disables progress printing during benchmark")
    return p.parse_args()


def configure_low_power_runtime(threads: int, *, nice: int = 0, cpu_affinity: str = ""):
    threads = max(1, int(threads))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OV_NUM_THREADS",
    ):
        os.environ[name] = str(threads)
    cv2.setNumThreads(threads)

    if cpu_affinity.strip():
        try:
            cores = {int(item) for item in cpu_affinity.split(",") if item.strip()}
            if cores and hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(0, cores)
        except Exception as exc:
            print(f"[WARN] Could not set CPU affinity {cpu_affinity!r}: {exc}")

    if nice > 0:
        try:
            os.nice(nice)
        except Exception as exc:
            print(f"[WARN] Could not set process niceness +{nice}: {exc}")


def vcgencmd(*args: str) -> str:
    try:
        proc = subprocess.run(["vcgencmd", *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return (proc.stdout or "").strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().replace("_", " ").replace("-", " ").split())


def collect_samples(test_dir: Path) -> list[tuple[Path, str]]:
    samples = []
    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMG_EXTS:
                samples.append((image_path, class_dir.name))
    return samples


def limit_samples_balanced(samples: list[tuple[Path, str]], limit: int) -> list[tuple[Path, str]]:
    if limit <= 0 or len(samples) <= limit:
        return samples
    by_label: dict[str, deque[tuple[Path, str]]] = defaultdict(deque)
    for sample in samples:
        by_label[sample[1]].append(sample)

    limited = []
    labels = sorted(by_label)
    while len(limited) < limit:
        added = False
        for label in labels:
            if by_label[label]:
                limited.append(by_label[label].popleft())
                added = True
                if len(limited) >= limit:
                    break
        if not added:
            break
    return limited


def pace_frame(next_slot: float, frame_period: float) -> float:
    now = time.perf_counter()
    if now < next_slot:
        time.sleep(next_slot - now)
    return max(next_slot + frame_period, time.perf_counter())


def print_power(prefix: str = "Power"):
    print(f"{prefix} throttled: {vcgencmd('get_throttled')}")
    print(f"{prefix} temp     : {vcgencmd('measure_temp')}")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def default_run_id() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def safe_run_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return cleaned or default_run_id()


def plant_display(field_id: str, row_id: str, plant_number: int) -> str:
    row = str(row_id).strip() or "A"
    return f"{field_id} / Row {row} / Plant {plant_number}"


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


def active_layout(layout: dict) -> dict:
    if not isinstance(layout, dict):
        return {}
    fields = layout.get("fields")
    if isinstance(fields, list) and fields:
        active_id = layout.get("active_field_id") or fields[0].get("id")
        for field in fields:
            if field.get("id") == active_id:
                return field
        return fields[0]
    return layout


class FieldScanMapper:
    """Deterministic row/column mapper for zero-slip field transitions."""

    def __init__(self, args):
        self.layout_path = Path(args.data_root) / "field_layout.json"
        self.layout_mtime = None
        self._configure(
            field_id=args.field_id,
            active_row_id=args.row_id,
            row_count=args.row_count,
            plants_per_row=args.plants_per_row,
            start_plant=args.start_plant,
            plant_step=args.plant_step,
            row_spacing_m=args.row_spacing_m,
            plant_spacing_m=args.plant_spacing_m,
        )
        self.reload_from_file()

    def _configure(
        self,
        *,
        field_id,
        active_row_id,
        row_count,
        plants_per_row,
        start_plant,
        plant_step,
        row_spacing_m,
        plant_spacing_m,
        rows=None,
    ):
        self.field_id = str(field_id or "Field 1")
        self.start_plant = max(1, int(start_plant or 1))
        self.plant_step = max(1, int(plant_step or 1))
        self.plants_per_row = max(1, int(plants_per_row or 1))
        self.row_spacing_m = max(0.0, float(row_spacing_m or 0.0))
        self.plant_spacing_m = max(0.0, float(plant_spacing_m or 0.0))
        generated = row_ids(max(1, int(row_count or 1)))
        raw_rows = rows if isinstance(rows, list) else []
        row_overrides = {}
        for raw_row in raw_rows:
            if not isinstance(raw_row, dict):
                continue
            row_id = str(raw_row.get("id") or raw_row.get("row_id") or "").strip().upper()
            if row_id:
                row_overrides[row_id] = raw_row
        row_specs = []
        y_m = 0.0
        for row_index, row_id in enumerate(generated, start=1):
            raw_row = row_overrides.get(row_id, {})
            row_spacing = max(0.0, float(raw_row.get("row_spacing_m", self.row_spacing_m) or 0.0))
            row_specs.append(
                {
                    "row_id": row_id,
                    "row_index": row_index,
                    "plants_per_row": max(1, int(raw_row.get("plants_per_row", self.plants_per_row) or 1)),
                    "plant_spacing_m": max(0.0, float(raw_row.get("plant_spacing_m", self.plant_spacing_m) or 0.0)),
                    "y_m": round(float(raw_row.get("y_m", y_m) or 0.0), 4),
                }
            )
            y_m += row_spacing
        active = str(active_row_id or row_specs[0]["row_id"]).strip().upper()
        if active not in {row["row_id"] for row in row_specs}:
            active = row_specs[0]["row_id"]
        start_index = next((idx for idx, row in enumerate(row_specs) if row["row_id"] == active), 0)
        self.row_specs = row_specs[start_index:] + row_specs[:start_index]
        self.rows = [row["row_id"] for row in self.row_specs]
        self.row_count = len(self.row_specs)
        self.active_row_id = active
        self.planned_total = sum(row["plants_per_row"] for row in self.row_specs)

    def reload_from_file(self) -> bool:
        try:
            stat = self.layout_path.stat()
        except OSError:
            return False
        if self.layout_mtime == stat.st_mtime:
            return False
        try:
            layout = json.loads(self.layout_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] Could not reload field layout: {exc}", flush=True)
            return False
        self.layout_mtime = stat.st_mtime
        field = active_layout(layout)
        self._configure(
            field_id=field.get("field_id") or field.get("name") or self.field_id,
            active_row_id=field.get("active_row_id", self.active_row_id),
            row_count=field.get("row_count", self.row_count),
            plants_per_row=field.get("plants_per_row", self.plants_per_row),
            start_plant=field.get("start_plant", self.start_plant),
            plant_step=field.get("plant_step", self.plant_step),
            row_spacing_m=field.get("row_spacing_m", self.row_spacing_m),
            plant_spacing_m=field.get("plant_spacing_m", self.plant_spacing_m),
            rows=field.get("rows"),
        )
        return True

    def position(self, scan_index: int) -> dict:
        zero_based = max(0, int(scan_index) - 1)
        pass_index, field_offset = divmod(zero_based, self.planned_total)
        row_spec = self.row_specs[0]
        plant_col = field_offset
        for candidate in self.row_specs:
            if plant_col < candidate["plants_per_row"]:
                row_spec = candidate
                break
            plant_col -= candidate["plants_per_row"]
        row_id = row_spec["row_id"]
        row_index = row_spec["row_index"]
        plant_number = self.start_plant + plant_col * self.plant_step
        return {
            "field_id": self.field_id,
            "row_id": row_id,
            "row_index": row_index,
            "plant_column": plant_col + 1,
            "plant_number": plant_number,
            "plant_key": f"{self.field_id}|{row_id}|{plant_number}",
            "x_m": round(plant_col * row_spec["plant_spacing_m"], 4),
            "y_m": row_spec["y_m"],
            "scan_pass": pass_index + 1,
            "field_complete": zero_based + 1 >= self.planned_total,
            "planned_total_plants": self.planned_total,
            "plant_display": plant_display(self.field_id, row_id, plant_number),
        }


def action_for_decision(label: str, status: str) -> str:
    normalized = normalize_label(label)
    if status == "uncertain" or normalized == "uncertain":
        return "Rescan this plant"
    if normalized == "healthy":
        return "No action"
    return "Inspect or treat"


class InferenceRunWriter:
    """Append-only local run store consumed by the farmer dashboard."""

    csv_fields = [
        "sequence",
        "timestamp",
        "epoch_time",
        "field_id",
        "row_id",
        "row_index",
        "plant_column",
        "plant_number",
        "plant_key",
        "x_m",
        "y_m",
        "plant_display",
        "scan_index",
        "plant_id",
        "label",
        "confidence",
        "status",
        "action",
        "frames_used",
        "reason",
        "late_frames",
        "gate_avg_latency_ms",
        "gate_max_latency_ms",
        "threads",
        "fps_target",
    ]

    def __init__(self, args):
        self.root = Path(args.data_root)
        self.run_id = safe_run_id(args.run_id)
        self.run_dir = self.root / "runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.root / "runs").mkdir(parents=True, exist_ok=True)
        self.decisions_path = self.run_dir / "decisions.jsonl"
        self.csv_path = self.run_dir / "events.csv"
        self.metadata_path = self.run_dir / "metadata.json"
        self.summary_path = self.run_dir / "summary.json"
        self.latest_path = self.root / "latest_run.json"
        self.started_at = now_iso()
        self.sequence = 0
        self.total_confidence = 0.0
        self.max_confidence = 0.0
        self.total_late_frames = 0
        self.sick = 0
        self.labels = defaultdict(int)
        self.statuses = defaultdict(int)
        self.latest_decision = None
        self.next_scan_index = 1

        existing_records = []
        if self.decisions_path.exists():
            with self.decisions_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        existing_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        existing_metadata = {}
        if self.metadata_path.exists():
            try:
                existing_metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except Exception:
                existing_metadata = {}
        if existing_metadata.get("started_at"):
            self.started_at = existing_metadata["started_at"]

        max_scan_index = 0
        for record in existing_records:
            self.sequence = max(self.sequence, int(record.get("sequence") or 0))
            label = str(record.get("label") or "Unknown")
            status = str(record.get("status") or "unknown")
            confidence = float(record.get("confidence") or 0.0)
            action = str(record.get("action") or action_for_decision(label, status))
            self.labels[label] += 1
            self.statuses[status] += 1
            self.total_confidence += confidence
            self.max_confidence = max(self.max_confidence, confidence)
            self.total_late_frames += int(record.get("late_frames") or 0)
            if action == "Inspect or treat":
                self.sick += 1
            self.latest_decision = record
            max_scan_index = max(max_scan_index, int(record.get("scan_index") or 0))
        if existing_records:
            self.sequence = max(self.sequence, len(existing_records))
            self.next_scan_index = max_scan_index + 1 if max_scan_index else len(existing_records) + 1

        self.decisions_handle = self.decisions_path.open("a", encoding="utf-8")
        new_csv = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        self.csv_handle = self.csv_path.open("a", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(self.csv_handle, fieldnames=self.csv_fields)
        if new_csv:
            self.csv_writer.writeheader()

        saved_farm_layout = {}
        try:
            field_layout_path = self.root / "field_layout.json"
            if field_layout_path.exists():
                saved_farm_layout = json.loads(field_layout_path.read_text(encoding="utf-8"))
        except Exception:
            saved_farm_layout = {}
        active_field = active_layout(saved_farm_layout)
        if active_field:
            active_rows = active_field.get("rows") if isinstance(active_field.get("rows"), list) else []
            active_row_ids = [str(row.get("id") or row.get("row_id")) for row in active_rows if isinstance(row, dict)] or row_ids(max(1, int(active_field.get("row_count") or args.row_count or 1)))
            planned_total = sum(int(row.get("plants_per_row") or active_field.get("plants_per_row") or args.plants_per_row or 1) for row in active_rows) if active_rows else max(1, int(active_field.get("row_count") or args.row_count or 1)) * max(1, int(active_field.get("plants_per_row") or args.plants_per_row or 1))
            field_map = {
                **active_field,
                "field_id": active_field.get("field_id") or active_field.get("name") or args.field_id,
                "active_row_id": active_field.get("active_row_id") or args.row_id,
                "row_id": active_field.get("active_row_id") or args.row_id,
                "row_ids": active_row_ids,
                "planned_total_plants": planned_total,
                "mapping_method": "Sequential row/column scan. Row-level overrides are loaded from field_layout.json, and the mapper advances automatically with no manual row transition.",
            }
        else:
            field_map = {
                "field_id": args.field_id,
                "active_row_id": args.row_id,
                "row_id": args.row_id,
                "row_count": args.row_count,
                "plants_per_row": args.plants_per_row,
                "row_ids": row_ids(max(1, int(args.row_count or 1))),
                "start_plant": args.start_plant,
                "plant_step": args.plant_step,
                "row_spacing_m": args.row_spacing_m,
                "plant_spacing_m": args.plant_spacing_m,
                "plant_cooldown_sec": args.plant_cooldown_sec,
                "mapping_method": "Sequential row/column scan. The farmer scans one plant, waits for a decision, then moves through every plant column before the mapper automatically advances to the next row.",
                "planned_total_plants": max(1, int(args.row_count or 1)) * max(1, int(args.plants_per_row or 1)),
            }

        metadata = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": None,
            "note": args.run_note,
            "mode": "live",
            "model": {
                "classifier": args.clf_model,
                "detector": args.det_model if args.with_detector else None,
                "with_detector": args.with_detector,
            },
            "runtime": {
                "threads": args.threads,
                "fps": args.fps,
                "nice": args.nice,
                "cpu_affinity": args.cpu_affinity,
            },
            "camera": {
                "source": args.camera,
                "width": args.camera_width,
                "height": args.camera_height,
                "picamera_raw_width": args.picamera_raw_width,
                "picamera_raw_height": args.picamera_raw_height,
            },
            "field_map": field_map,
            "farm_layout": saved_farm_layout if saved_farm_layout else None,
            "inference": {
                "max_edge": args.max_edge,
                "crop_mode": args.crop_mode,
                "crop_scale": args.crop_scale,
                "crop_pad": args.crop_pad,
                "conf": args.conf,
                "high_conf": args.high_conf,
                "det_imgsz": args.det_imgsz,
                "det_conf": args.det_conf,
                "det_iou": args.det_iou,
                "max_leaves": args.max_leaves,
            },
            "power": {
                "start_throttled": vcgencmd("get_throttled"),
                "start_temp": vcgencmd("measure_temp"),
                "end_throttled": None,
                "end_temp": None,
            },
            "paths": {
                "decisions_jsonl": str(self.decisions_path),
                "events_csv": str(self.csv_path),
                "summary_json": str(self.summary_path),
            },
        }
        write_json(self.metadata_path, metadata)
        self._write_summary(completed=False)

    def append(self, payload: dict, gate_frames: list[dict]):
        self.sequence += 1
        timestamp = now_iso()
        epoch_time = float(payload.get("time", time.time()))
        latencies = [float(frame.get("latency_ms", 0.0)) for frame in gate_frames if "latency_ms" in frame]
        gate_avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        gate_max_latency = max(latencies) if latencies else 0.0

        row = {
            "sequence": self.sequence,
            "timestamp": timestamp,
            "epoch_time": epoch_time,
            "field_id": payload.get("field_id"),
            "row_id": payload.get("row_id"),
            "row_index": payload.get("row_index"),
            "plant_column": payload.get("plant_column"),
            "plant_number": payload.get("plant_number"),
            "plant_key": payload.get("plant_key"),
            "x_m": payload.get("x_m"),
            "y_m": payload.get("y_m"),
            "plant_display": payload.get("plant_display"),
            "scan_index": payload.get("scan_index"),
            "plant_id": payload.get("plant_id"),
            "label": payload.get("label"),
            "confidence": float(payload.get("confidence", 0.0)),
            "status": payload.get("status"),
            "action": payload.get("action"),
            "frames_used": payload.get("frames_used"),
            "reason": payload.get("reason"),
            "late_frames": int(payload.get("late_frames", 0)),
            "gate_avg_latency_ms": round(gate_avg_latency, 3),
            "gate_max_latency_ms": round(gate_max_latency, 3),
            "threads": payload.get("threads"),
            "fps_target": payload.get("fps_target"),
        }
        record = {
            **payload,
            "run_id": self.run_id,
            "sequence": self.sequence,
            "timestamp": timestamp,
            "gate_avg_latency_ms": row["gate_avg_latency_ms"],
            "gate_max_latency_ms": row["gate_max_latency_ms"],
            "gate_frame_labels": [frame.get("label") for frame in gate_frames],
            "gate_frame_confidences": [round(float(frame.get("confidence", 0.0)), 6) for frame in gate_frames],
        }

        self.decisions_handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.decisions_handle.flush()
        self.csv_writer.writerow(row)
        self.csv_handle.flush()

        label = str(row["label"] or "Unknown")
        status = str(row["status"] or "unknown")
        self.labels[label] += 1
        self.statuses[status] += 1
        self.total_confidence += float(row["confidence"])
        self.max_confidence = max(self.max_confidence, float(row["confidence"]))
        self.total_late_frames += int(row["late_frames"])
        if row["action"] == "Inspect or treat":
            self.sick += 1
        self.latest_decision = record
        self._write_summary(completed=False)

    def close(self):
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        metadata["completed_at"] = now_iso()
        metadata["power"]["end_throttled"] = vcgencmd("get_throttled")
        metadata["power"]["end_temp"] = vcgencmd("measure_temp")
        write_json(self.metadata_path, metadata)
        self._write_summary(completed=True)
        self.decisions_handle.close()
        self.csv_handle.close()

    def _write_summary(self, *, completed: bool):
        avg_confidence = self.total_confidence / self.sequence if self.sequence else 0.0
        ok = int(self.statuses.get("ok", 0))
        uncertain = int(self.statuses.get("uncertain", 0))
        summary = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "last_updated_at": now_iso(),
            "completed": completed,
            "decisions": self.sequence,
            "ok": ok,
            "sick": self.sick,
            "uncertain": uncertain,
            "uncertain_rate": uncertain / self.sequence if self.sequence else 0.0,
            "avg_confidence": avg_confidence,
            "max_confidence": self.max_confidence,
            "total_late_frames": self.total_late_frames,
            "labels": dict(sorted(self.labels.items())),
            "statuses": dict(sorted(self.statuses.items())),
            "latest_decision": self.latest_decision,
            "paths": {
                "run_dir": str(self.run_dir),
                "metadata": str(self.metadata_path),
                "summary": str(self.summary_path),
                "decisions": str(self.decisions_path),
                "events_csv": str(self.csv_path),
            },
        }
        write_json(self.summary_path, summary)
        write_json(
            self.latest_path,
            {
                "run_id": self.run_id,
                "updated_at": summary["last_updated_at"],
                "completed": completed,
                "run_dir": str(self.run_dir),
                "summary": str(self.summary_path),
            },
        )


def load_low_power_models(args):
    configure_low_power_runtime(args.threads, nice=args.nice, cpu_affinity=args.cpu_affinity)
    return load_models(
        det_model_path=args.det_model,
        clf_model_path=args.clf_model,
        threads=args.threads,
        with_detector=args.with_detector,
        det_imgsz=args.det_imgsz,
    )


def benchmark(args):
    test_dir = Path(args.test)
    if not test_dir.is_dir():
        raise SystemExit(f"[ERROR] Test folder not found: {test_dir}")

    samples = collect_samples(test_dir)
    if args.limit > 0:
        samples = samples[:args.limit] if args.sequential_limit else limit_samples_balanced(samples, args.limit)
    if not samples:
        raise SystemExit(f"[ERROR] No images found in: {test_dir}")

    det_model, clf_model = load_low_power_models(args)
    healthy_idx = _find_healthy_idx(clf_model.names)

    frame_period = 1.0 / max(args.fps, 0.001)
    next_slot = time.perf_counter()
    wall_start = time.perf_counter()
    active_time = 0.0
    max_latency_ms = 0.0
    over_preferred = 0
    over_hard = 0
    processed = 0
    raw_correct = 0
    emitted = 0
    emitted_correct = 0
    uncertain = 0
    class_stats = defaultdict(lambda: {"total": 0, "raw_correct": 0, "emitted": 0, "emitted_correct": 0, "uncertain": 0})
    power_samples = []
    start_throttled = vcgencmd("get_throttled")
    start_temp = vcgencmd("measure_temp")

    print()
    print("Low-power deployment benchmark")
    print(f"  classifier : {args.clf_model}")
    print(f"  detector   : {'enabled' if args.with_detector else 'disabled'}")
    print(f"  images     : {len(samples)}")
    print(f"  threads    : {args.threads}")
    print(f"  fps target : {args.fps}")
    print(f"  paced      : {not args.no_pace}")
    print(f"  nice       : +{args.nice}")
    print(f"  affinity   : {args.cpu_affinity or 'default scheduler'}")
    print(f"  crop       : {args.crop_mode} scale={args.crop_scale} pad={args.crop_pad}")
    print(f"  conf gate  : {args.conf}")
    print(f"Start throttled: {start_throttled}")
    print(f"Start temp     : {start_temp}")
    print()

    for image_path, true_label in samples:
        if STOP:
            break
        if not args.no_pace:
            next_slot = pace_frame(next_slot, frame_period)

        img = cv2.imread(str(image_path))
        if img is None:
            continue

        started = time.perf_counter()
        prediction = infer_frame(
            det_model,
            clf_model,
            img,
            with_detector=args.with_detector,
            det_conf=args.det_conf,
            det_iou=args.det_iou,
            det_imgsz=args.det_imgsz,
            max_edge=args.max_edge,
            max_leaves=args.max_leaves,
            crop_mode=args.crop_mode,
            crop_scale=args.crop_scale,
            crop_pad=args.crop_pad,
            clf_conf=args.conf,
            healthy_idx=healthy_idx,
        )
        elapsed = time.perf_counter() - started
        latency_ms = elapsed * 1000.0
        active_time += elapsed
        max_latency_ms = max(max_latency_ms, latency_ms)
        if latency_ms > PREFERRED_FRAME_LATENCY_MS:
            over_preferred += 1
        if latency_ms > MAX_FRAME_LATENCY_MS:
            over_hard += 1

        label = prediction["label"]
        is_uncertain = label == "Uncertain"
        is_correct = normalize_label(label) == normalize_label(true_label)
        processed += 1
        raw_correct += int(is_correct)
        stats = class_stats[true_label]
        stats["total"] += 1
        stats["raw_correct"] += int(is_correct)

        if is_uncertain:
            uncertain += 1
            stats["uncertain"] += 1
        else:
            emitted += 1
            stats["emitted"] += 1
            emitted_correct += int(is_correct)
            stats["emitted_correct"] += int(is_correct)

        if args.sample_power_every > 0 and processed % args.sample_power_every == 0:
            power_samples.append(
                {
                    "frame": processed,
                    "throttled": vcgencmd("get_throttled"),
                    "temp": vcgencmd("measure_temp"),
                }
            )

        if args.progress_every > 0 and (processed == 1 or processed % args.progress_every == 0):
            wall = max(time.perf_counter() - wall_start, 1e-9)
            emitted_acc = emitted_correct / emitted if emitted else 0.0
            print(
                f"  {processed:4d}/{len(samples)} "
                f"wall_fps={processed / wall:5.2f} "
                f"lat={latency_ms:6.1f}ms "
                f"emit_acc={emitted_acc * 100:5.1f}% "
                f"uncertain={uncertain}"
            )

    wall_time = max(time.perf_counter() - wall_start, 1e-9)
    avg_latency_ms = (active_time * 1000.0 / processed) if processed else 0.0
    active_fps = processed / active_time if active_time > 0 else 0.0
    wall_fps = processed / wall_time if wall_time > 0 else 0.0
    duty_cycle = active_time / wall_time if wall_time > 0 else 0.0
    raw_acc = raw_correct / processed if processed else 0.0
    emitted_acc = emitted_correct / emitted if emitted else 0.0
    coverage = emitted / processed if processed else 0.0

    meets_latency = avg_latency_ms <= MAX_FRAME_LATENCY_MS and over_hard == 0
    schedule_target = max(MIN_FRAME_FPS, args.fps)
    meets_schedule = wall_fps >= schedule_target * 0.98 if not args.no_pace else active_fps >= schedule_target
    pi_target = meets_schedule and meets_latency

    per_class = {}
    for label, stats in sorted(class_stats.items()):
        total = stats["total"]
        class_emitted = stats["emitted"]
        per_class[label] = {
            "total": total,
            "frame_accuracy": stats["raw_correct"] / total if total else 0.0,
            "emitted_accuracy": stats["emitted_correct"] / class_emitted if class_emitted else 0.0,
            "coverage": class_emitted / total if total else 0.0,
            "emitted": class_emitted,
            "uncertain": stats["uncertain"],
        }

    report = {
        "mode": "low_power_benchmark",
        "settings": vars(args),
        "power_bank_note": "20,000 mAh 5V/3A is 15W max output; Pi 5 official supply is higher. Keep detector disabled and use the validated paced 2-thread profile for this experiment.",
        "metrics": {
            "images_tested": processed,
            "frame_accuracy": raw_acc,
            "emitted_accuracy": emitted_acc,
            "coverage": coverage,
            "correct": raw_correct,
            "emitted_correct": emitted_correct,
            "emitted": emitted,
            "uncertain": uncertain,
            "active_fps": active_fps,
            "wall_fps": wall_fps,
            "target_fps": args.fps,
            "schedule_target_fps": schedule_target,
            "active_time_sec": active_time,
            "wall_time_sec": wall_time,
            "estimated_cpu_duty_cycle": duty_cycle,
            "avg_latency_ms": avg_latency_ms,
            "max_latency_ms": max_latency_ms,
            "over_preferred_latency_frames": over_preferred,
            "over_hard_latency_frames": over_hard,
            "meets_schedule": meets_schedule,
            "meets_latency_ceiling": meets_latency,
            "pi_target": pi_target,
        },
        "per_class": per_class,
        "power": {
            "start_throttled": start_throttled,
            "start_temp": start_temp,
            "samples": power_samples,
            "end_throttled": vcgencmd("get_throttled"),
            "end_temp": vcgencmd("measure_temp"),
        },
    }

    report_path = Path(args.report) if args.report else REPORTS_DIR / "low_power" / "low_power_benchmark_report.json"
    write_json(report_path, report)

    sep = "=" * 68
    print()
    print(sep)
    print("Low-Power Benchmark Results")
    print(sep)
    print(f"Images tested    : {processed}")
    print(f"Frame accuracy   : {raw_acc * 100:.2f}% ({raw_correct}/{processed})")
    print(f"Emitted accuracy : {emitted_acc * 100:.2f}% ({emitted_correct}/{emitted})")
    print(f"Coverage         : {coverage * 100:.2f}% ({emitted}/{processed})")
    print(f"Uncertain        : {uncertain}/{processed}")
    print(f"Wall FPS         : {wall_fps:.2f}")
    print(f"Active FPS       : {active_fps:.2f}")
    print(f"CPU duty estimate: {duty_cycle * 100:.1f}%")
    print(f"Avg latency      : {avg_latency_ms:.0f} ms/img")
    print(f"Max latency      : {max_latency_ms:.0f} ms/img")
    print(f">180 ms          : {over_preferred}/{processed}")
    print(f">200 ms          : {over_hard}/{processed}")
    print(f"Pi target        : {'PASS' if pi_target else 'FAIL'}")
    print(f"End throttled    : {report['power']['end_throttled']}")
    print(f"End temp         : {report['power']['end_temp']}")
    print(f"Report           : {report_path}")
    print()
    print("Per-class emitted accuracy:")
    for label, stats in per_class.items():
        print(
            f"  {label:<28s} "
            f"emit_acc={stats['emitted_accuracy'] * 100:5.1f}% "
            f"coverage={stats['coverage'] * 100:5.1f}% "
            f"uncertain={stats['uncertain']:3d}/{stats['total']}"
        )
    print(sep)


def open_capture(source: str):
    if source.lower() in {"picamera2", "libcamera", "csi"}:
        return Picamera2Capture()
    try:
        camera_index = int(source)
        cap = cv2.VideoCapture(camera_index)
    except ValueError:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Cannot open camera/video source: {source}")
    return cap


class Picamera2Capture:
    def __init__(self, width: int = 640, height: int = 480, raw_width: int = 2304, raw_height: int = 1296):
        try:
            from picamera2 import Picamera2
        except ModuleNotFoundError:
            dist_packages = "/usr/lib/python3/dist-packages"
            if dist_packages not in sys.path:
                sys.path.append(dist_packages)
            from picamera2 import Picamera2

        self.camera = Picamera2()
        kwargs = {
            "main": {"size": (width, height), "format": "RGB888"},
            "buffer_count": 2,
        }
        if raw_width > 0 and raw_height > 0:
            kwargs["raw"] = {"size": (raw_width, raw_height)}
        config = self.camera.create_video_configuration(**kwargs)
        self.camera.configure(config)
        self.camera.start()
        time.sleep(0.4)

    def isOpened(self):
        return True

    def set(self, prop, value):
        return True

    def read(self):
        frame = self.camera.capture_array()
        if frame is None:
            return False, None
        return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def release(self):
        self.camera.stop()


def live(args):
    det_model, clf_model = load_low_power_models(args)
    healthy_idx = _find_healthy_idx(clf_model.names)
    if args.camera.lower() in {"picamera2", "libcamera", "csi"}:
        cap = Picamera2Capture(
            args.camera_width,
            args.camera_height,
            args.picamera_raw_width,
            args.picamera_raw_height,
        )
    else:
        cap = open_capture(args.camera)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if args.camera_width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    if args.camera_height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    output_path = Path(args.jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_handle = output_path.open("a", encoding="utf-8")
    run_writer = None if args.no_organized_data else InferenceRunWriter(args)

    gate = PlantDecisionGate(high_conf=args.high_conf)
    mapper = FieldScanMapper(args)
    frame_period = 1.0 / max(args.fps, 0.001)
    next_slot = time.perf_counter()
    scan_index = run_writer.next_scan_index if run_writer is not None else 1
    current_position = mapper.position(scan_index)
    late_frames = 0
    live_frames = 0

    print("Low-power live loop")
    print(f"  camera     : {args.camera}")
    print(f"  threads    : {args.threads}")
    print(f"  fps target : {args.fps}")
    print(f"  detector   : {'enabled' if args.with_detector else 'disabled'}")
    print(f"  jsonl      : {output_path}")
    print(f"  map        : {current_position['plant_display']}")
    print(f"  field plan : {mapper.row_count} rows x {mapper.plants_per_row} plants = {mapper.planned_total} mapped positions")
    if args.plant_cooldown_sec > 0:
        print(f"  move wait  : {args.plant_cooldown_sec:.1f}s after each decision")
    if run_writer is not None:
        print(f"  data run   : {run_writer.run_dir}")
    print_power("Start")

    try:
        while not STOP:
            if not args.no_pace:
                next_slot = pace_frame(next_slot, frame_period)
            ok, frame = cap.read()
            if not ok:
                break
            live_frames += 1

            started = time.perf_counter()
            prediction = infer_frame(
                det_model,
                clf_model,
                frame,
                with_detector=args.with_detector,
                det_conf=args.det_conf,
                det_iou=args.det_iou,
                det_imgsz=args.det_imgsz,
                max_edge=args.max_edge,
                max_leaves=args.max_leaves,
                crop_mode=args.crop_mode,
                crop_scale=args.crop_scale,
                crop_pad=args.crop_pad,
                clf_conf=args.conf,
                healthy_idx=healthy_idx,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            prediction["latency_ms"] = latency_ms
            if latency_ms > MAX_FRAME_LATENCY_MS:
                late_frames += 1

            decision = gate.add(prediction)
            if decision is not None:
                gate_frames = list(gate.frames)
                if mapper.reload_from_file():
                    print(
                        f"[MAP] Reloaded field layout: {mapper.row_count} rows x {mapper.plants_per_row} plants",
                        flush=True,
                    )
                position = mapper.position(scan_index)
                display = position["plant_display"]
                action = action_for_decision(decision.label, decision.status)
                payload = {
                    "plant_id": position["plant_number"],
                    "scan_index": scan_index,
                    "field_id": position["field_id"],
                    "row_id": position["row_id"],
                    "row_index": position["row_index"],
                    "plant_column": position["plant_column"],
                    "plant_number": position["plant_number"],
                    "plant_key": position["plant_key"],
                    "x_m": position["x_m"],
                    "y_m": position["y_m"],
                    "scan_pass": position["scan_pass"],
                    "field_complete": position["field_complete"],
                    "planned_total_plants": position["planned_total_plants"],
                    "plant_display": display,
                    "label": decision.label,
                    "confidence": round(decision.confidence, 6),
                    "status": decision.status,
                    "action": action,
                    "frames_used": decision.frames_used,
                    "reason": decision.reason,
                    "late_frames": late_frames,
                    "threads": args.threads,
                    "fps_target": args.fps,
                    "time": time.time(),
                }
                line = json.dumps(payload, separators=(",", ":"))
                print(line, flush=True)
                output_handle.write(line + "\n")
                output_handle.flush()
                if run_writer is not None:
                    run_writer.append(payload, gate_frames)
                print(f"[MAP] {display}: {decision.label} ({action})", flush=True)
                scan_index += 1
                current_position = mapper.position(scan_index)
                late_frames = 0
                gate.reset()
                if args.plant_cooldown_sec > 0:
                    cooldown_until = time.perf_counter() + args.plant_cooldown_sec
                    while not STOP and time.perf_counter() < cooldown_until:
                        remaining = cooldown_until - time.perf_counter()
                        if remaining > 0:
                            time.sleep(min(0.1, remaining))
                    next_slot = time.perf_counter()
                    print(f"[NEXT] {current_position['plant_display']}", flush=True)

            if args.max_live_frames > 0 and live_frames >= args.max_live_frames:
                break
    finally:
        cap.release()
        output_handle.close()
        if run_writer is not None:
            run_writer.close()
        print_power("End")


def main():
    args = parse_args()
    if args.fps < MIN_FRAME_FPS:
        raise SystemExit(f"[ERROR] --fps must be at least {MIN_FRAME_FPS:.1f}")
    if args.threads > 2 and args.fps <= 5.0:
        print("[WARN] Low-power 5V/3A profile was validated with --threads 2.")
    if args.with_detector:
        print("[WARN] Detector mode is not recommended for 5V/3A battery operation.")

    if args.mode == "benchmark":
        benchmark(args)
    else:
        live(args)


if __name__ == "__main__":
    main()
