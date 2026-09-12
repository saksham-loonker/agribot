#!/usr/bin/env python3
"""Import an older Agribot JSONL inference log into agribot_inference_data."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


CSV_FIELDS = [
    "sequence",
    "timestamp",
    "epoch_time",
    "field_id",
    "row_id",
    "plant_number",
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
FORMULA_PREFIX_RE = re.compile(r"^[\t ]*[=+\-@]")


def csv_safe_cell(value):
    text = "" if value is None else str(value)
    return "'" + text if FORMULA_PREFIX_RE.match(text) else text


def iso_from_epoch(value) -> str:
    try:
        return datetime.fromtimestamp(float(value)).astimezone().isoformat(timespec="seconds")
    except Exception:
        return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_run_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return cleaned or datetime.now().astimezone().strftime("import_%Y%m%d_%H%M%S")


def normalize_label(label: str) -> str:
    return " ".join(str(label).strip().lower().replace("_", " ").replace("-", " ").split())


def action_for(label: str, status: str) -> str:
    normalized = normalize_label(label)
    if status == "uncertain" or normalized == "uncertain":
        return "Rescan this plant"
    if normalized == "healthy":
        return "No action"
    return "Inspect or treat"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Import legacy Agribot JSONL into platform data")
    parser.add_argument("jsonl", help="Legacy JSONL path")
    parser.add_argument("--data-root", default="agribot_inference_data")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--note", default="Imported from legacy JSONL")
    parser.add_argument("--set-latest", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    source = Path(args.jsonl)
    if not source.exists():
        raise SystemExit(f"[ERROR] JSONL not found: {source}")
    rows = read_jsonl(source)
    if not rows:
        raise SystemExit(f"[ERROR] No rows found in: {source}")

    run_id = safe_run_id(args.run_id or source.stem)
    data_root = Path(args.data_root)
    run_dir = data_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = iso_from_epoch(rows[0].get("time"))
    completed_at = iso_from_epoch(rows[-1].get("time"))

    labels = defaultdict(int)
    statuses = defaultdict(int)
    total_confidence = 0.0
    max_confidence = 0.0
    total_late_frames = 0
    sick = 0
    latest = None

    decisions_path = run_dir / "decisions.jsonl"
    csv_path = run_dir / "events.csv"
    with decisions_path.open("w", encoding="utf-8") as decisions, csv_path.open("w", newline="", encoding="utf-8") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            timestamp = iso_from_epoch(row.get("time"))
            field_id = row.get("field_id") or "Imported Field"
            row_id = row.get("row_id") or "A"
            plant_number = row.get("plant_number") or row.get("plant_id") or index
            plant_display = row.get("plant_display") or f"{field_id} / Row {row_id} / Plant {plant_number}"
            action = row.get("action") or action_for(row.get("label"), row.get("status"))
            record = {
                **row,
                "run_id": run_id,
                "sequence": index,
                "timestamp": timestamp,
                "field_id": field_id,
                "row_id": row_id,
                "plant_number": plant_number,
                "plant_display": plant_display,
                "scan_index": row.get("scan_index") or index,
                "action": action,
                "gate_avg_latency_ms": row.get("gate_avg_latency_ms", 0.0),
                "gate_max_latency_ms": row.get("gate_max_latency_ms", 0.0),
            }
            decisions.write(json.dumps(record, separators=(",", ":")) + "\n")
            csv_record = {
                "sequence": index,
                "timestamp": timestamp,
                "epoch_time": row.get("time"),
                "field_id": field_id,
                "row_id": row_id,
                "plant_number": plant_number,
                "plant_display": plant_display,
                "scan_index": record["scan_index"],
                "plant_id": row.get("plant_id"),
                "label": row.get("label"),
                "confidence": row.get("confidence", 0.0),
                "status": row.get("status"),
                "action": action,
                "frames_used": row.get("frames_used"),
                "reason": row.get("reason"),
                "late_frames": row.get("late_frames", 0),
                "gate_avg_latency_ms": record["gate_avg_latency_ms"],
                "gate_max_latency_ms": record["gate_max_latency_ms"],
                "threads": row.get("threads"),
                "fps_target": row.get("fps_target"),
            }
            writer.writerow({key: csv_safe_cell(value) for key, value in csv_record.items()})
            label = str(row.get("label") or "Unknown")
            status = str(row.get("status") or "unknown")
            confidence = float(row.get("confidence", 0.0))
            labels[label] += 1
            statuses[status] += 1
            total_confidence += confidence
            max_confidence = max(max_confidence, confidence)
            total_late_frames += int(row.get("late_frames", 0))
            if action == "Inspect or treat":
                sick += 1
            latest = record

    total = len(rows)
    summary = {
        "run_id": run_id,
        "started_at": started_at,
        "last_updated_at": completed_at,
        "completed": True,
        "decisions": total,
        "ok": int(statuses.get("ok", 0)),
        "sick": sick,
        "uncertain": int(statuses.get("uncertain", 0)),
        "uncertain_rate": int(statuses.get("uncertain", 0)) / total if total else 0.0,
        "avg_confidence": total_confidence / total if total else 0.0,
        "max_confidence": max_confidence,
        "total_late_frames": total_late_frames,
        "labels": dict(sorted(labels.items())),
        "statuses": dict(sorted(statuses.items())),
        "latest_decision": latest,
        "paths": {
            "run_dir": str(run_dir),
            "metadata": str(run_dir / "metadata.json"),
            "summary": str(run_dir / "summary.json"),
            "decisions": str(decisions_path),
            "events_csv": str(csv_path),
        },
    }
    metadata = {
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "note": args.note,
        "mode": "legacy_import",
        "source_jsonl": str(source),
        "model": {"classifier": None, "detector": None, "with_detector": False},
        "runtime": {
            "threads": rows[-1].get("threads"),
            "fps": rows[-1].get("fps_target"),
        },
        "paths": summary["paths"],
    }
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "summary.json", summary)
    if args.set_latest:
        write_json(
            data_root / "latest_run.json",
            {
                "run_id": run_id,
                "updated_at": completed_at,
                "completed": True,
                "run_dir": str(run_dir),
                "summary": str(run_dir / "summary.json"),
            },
        )
    print(f"Imported {total} decisions -> {run_dir}")


if __name__ == "__main__":
    main()
