#!/usr/bin/env python3
"""Validate an Android Agribot field export ZIP before field-readiness signoff."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path
from typing import Any


EXPECTED_MODEL_BUNDLE_ID = "agribot-model-bundle-v001"
REQUIRED_ENTRIES = [
    "metadata.json",
    "summary.json",
    "latest_run.json",
    "field_layout.json",
    "decisions.jsonl",
    "events.csv",
    "app_log.jsonl",
    "diagnostics.json",
]
LEGACY_CSV_HEADER = [
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
    "track_id",
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
# Bundles created before track identity was added remain readable. They are
# accepted for migration/diagnostics, while current Android exports always use
# LEGACY_CSV_HEADER (which includes track_id) or the extended CSV_HEADER.
PRE_TRACK_CSV_HEADER = [field for field in LEGACY_CSV_HEADER if field != "track_id"]
CSV_HEADER = LEGACY_CSV_HEADER + [
    "row_side",
    "bbox_px",
    "geometry_confidence",
    "plant_position_confidence",
    "geometry_reason",
    "raw_label",
    "manual_override",
    "evidence_path",
    "evidence_status",
    "measurement_distance_m",
    "measurement_source",
    "measurement_quality",
    "relative_plant_width",
    "relative_plant_height",
    "size_source",
    "size_quality",
    "gps_accuracy_m",
    "gps_fix_age_seconds",
    "top2_margin",
    "prediction_entropy",
    "treatment_status",
    "treatment_note",
]


def validate_export_bundle(path: Path, require_real_field: bool = False) -> dict[str, Any]:
    failures: list[str] = []
    if not path.exists():
        return result(path, failures=[f"field export bundle not found: {path}"])
    if path.suffix.lower() != ".zip":
        failures.append("field export bundle must be a .zip file")

    try:
        with zipfile.ZipFile(path) as zip_file:
            entries = set(zip_file.namelist())
            missing = [entry for entry in REQUIRED_ENTRIES if entry not in entries]
            failures.extend(f"missing {entry}" for entry in missing)
            if missing:
                return result(path, failures=failures)

            metadata = read_json(zip_file, "metadata.json", failures)
            summary = read_json(zip_file, "summary.json", failures)
            latest_run = read_json(zip_file, "latest_run.json", failures)
            field_layout = read_json(zip_file, "field_layout.json", failures)
            diagnostics = read_json(zip_file, "diagnostics.json", failures)
            decisions = read_jsonl(zip_file, "decisions.jsonl", failures)
            app_log = read_jsonl(zip_file, "app_log.jsonl", failures)
            csv_row_count = validate_csv(zip_file, failures)
            validate_app_log(app_log, failures)
            validate_metadata(metadata, summary, latest_run, diagnostics, failures, require_real_field)
            validate_count_consistency(summary, latest_run, len(decisions), csv_row_count, failures)
            validate_field_layout(field_layout, failures)
            evidence_count = validate_decisions(decisions, entries, zip_file, failures)
            return result(path, failures=failures, decision_count=len(decisions), evidence_count=evidence_count)
    except zipfile.BadZipFile:
        failures.append("field export bundle is not a valid ZIP")
    return result(path, failures=failures)


def validate_metadata(
    metadata: dict[str, Any],
    summary: dict[str, Any],
    latest_run: dict[str, Any],
    diagnostics: dict[str, Any],
    failures: list[str],
    require_real_field: bool = False,
) -> None:
    run_id = latest_run.get("run_id") or latest_run.get("latest_run_id")
    if not run_id:
        failures.append("latest_run run_id is required")
    elif require_real_field and str(run_id).startswith("android_test_"):
        failures.append("field export must come from a real field run, not android_test_*")
    if latest_run.get("completed") is not True:
        failures.append("latest_run completed must be true")
    if str(latest_run.get("state") or "").lower() != "completed":
        failures.append("latest_run state must be completed")
    if latest_run.get("model_bundle_id") != EXPECTED_MODEL_BUNDLE_ID:
        failures.append(f"latest_run model_bundle_id must be {EXPECTED_MODEL_BUNDLE_ID}")
    if metadata.get("origin") != "android":
        failures.append("metadata origin must be android")
    if metadata.get("model_bundle_id") != EXPECTED_MODEL_BUNDLE_ID:
        failures.append(f"metadata model_bundle_id must be {EXPECTED_MODEL_BUNDLE_ID}")
    if metadata.get("run_id") and run_id and metadata.get("run_id") != run_id:
        failures.append("metadata run_id must match latest_run")
    if summary.get("run_id") and run_id and summary.get("run_id") != run_id:
        failures.append("summary run_id must match latest_run")
    if summary.get("completed") is not True:
        failures.append("summary completed must be true")
    if diagnostics.get("offline") is not True:
        failures.append("diagnostics offline must be true")
    if diagnostics.get("network_required") is not False:
        failures.append("diagnostics network_required must be false")
    diagnostics_status = diagnostics.get("status")
    if diagnostics_status not in {"captured", "not_captured"}:
        failures.append("diagnostics status must be captured or not_captured")
    if "cpu_default" in diagnostics:
        failures.append("diagnostics cpu_default is obsolete; use cpu_thread_profile")


def validate_count_consistency(
    summary: dict[str, Any],
    latest_run: dict[str, Any],
    decision_count: int,
    csv_row_count: int,
    failures: list[str],
) -> None:
    if latest_run.get("decision_count") != decision_count:
        failures.append("latest_run decision_count must match decisions.jsonl")
    if summary.get("total_decisions") != decision_count:
        failures.append("summary total_decisions must match decisions.jsonl")
    if csv_row_count != decision_count:
        failures.append("events.csv row count must match decisions.jsonl")


def validate_field_layout(field_layout: dict[str, Any], failures: list[str]) -> None:
    fields = field_layout.get("fields")
    if not isinstance(fields, list) or not fields:
        failures.append("field_layout fields must be non-empty")
        return
    rows = fields[0].get("rows") if isinstance(fields[0], dict) else None
    if not isinstance(rows, list) or not rows:
        failures.append("field_layout rows must be non-empty")

def validate_app_log(app_log: list[dict[str, Any]], failures: list[str]) -> None:
    if not app_log:
        failures.append("app_log.jsonl must contain at least one event")
        return
    event_types = {str(event.get("type") or "") for event in app_log}
    if "started" not in event_types:
        failures.append("app_log.jsonl must include started event")
    for index, event in enumerate(app_log, start=1):
        if event.get("source") != "agribot_android":
            failures.append(f"app_log.jsonl event {index} source must be agribot_android")
        if event.get("level") != "info":
            failures.append(f"app_log.jsonl event {index} level must be info")
        if not isinstance(event.get("payload"), dict):
            failures.append(f"app_log.jsonl event {index} payload must be a JSON object")


def validate_decisions(
    decisions: list[dict[str, Any]],
    entries: set[str],
    zip_file: zipfile.ZipFile,
    failures: list[str],
) -> int:
    if not decisions:
        failures.append("decisions.jsonl must contain at least one decision")
        return 0
    evidence_count = 0
    for decision in decisions:
        sequence = decision.get("sequence")
        missing = [name for name in common_required_fields() if decision.get(name) in (None, "")]
        mode = str(decision.get("mode") or "")
        if mode == "FRONT_ROW_OVERVIEW":
            missing.extend(name for name in front_overview_required_fields() if decision.get(name) in (None, ""))
            validate_front_overview_geometry(decision, failures)
        else:
            missing.extend(name for name in side_scan_required_fields() if decision.get(name) in (None, ""))
        if missing:
            failures.append(f"decision {sequence} missing required fields: {', '.join(missing)}")
        if decision.get("model_version") != EXPECTED_MODEL_BUNDLE_ID:
            failures.append(f"decision {sequence} model_version must be {EXPECTED_MODEL_BUNDLE_ID}")
        if requires_evidence(decision):
            evidence_path = str(decision.get("evidence_path") or "")
            if not evidence_path.lower().endswith((".jpg", ".jpeg")) or evidence_path not in entries:
                failures.append(f"decision {sequence} requires JPEG evidence")
            elif not is_jpeg(zip_file.read(evidence_path)):
                failures.append(f"decision {sequence} evidence is not a JPEG file")
            else:
                evidence_count += 1
    return evidence_count


def common_required_fields() -> list[str]:
    return [
        "run_id",
        "sequence",
        "mode",
        "field_id",
        "label",
        "confidence",
        "status",
        "action",
        "frames_used",
        "reason",
        "model_version",
        "scan_index",
    ]


def side_scan_required_fields() -> list[str]:
    return [
        "row_id",
        "row_index",
        "plant_column",
        "plant_number",
        "plant_key",
        "x_m",
        "y_m",
    ]


def front_overview_required_fields() -> list[str]:
    return [
        "row_side",
        "bbox_px",
        "geometry_confidence",
        "plant_position_confidence",
        "geometry_reason",
    ]


def validate_front_overview_geometry(decision: dict[str, Any], failures: list[str]) -> None:
    sequence = decision.get("sequence")
    bbox = decision.get("bbox_px")
    if not isinstance(bbox, list) or len(bbox) != 4 or not all(is_number(value) for value in bbox):
        failures.append(f"decision {sequence} front bbox_px must contain four numeric values")

    row_side = str(decision.get("row_side") or "").lower()
    if row_side not in {"left", "right", "unknown"}:
        failures.append(f"decision {sequence} front row_side must be left, right, or unknown")

    geometry_confidence = decision.get("geometry_confidence")
    if not is_probability(geometry_confidence):
        failures.append(f"decision {sequence} geometry_confidence must be between 0 and 1")

    plant_position_confidence = decision.get("plant_position_confidence")
    if not is_probability(plant_position_confidence):
        failures.append(f"decision {sequence} plant_position_confidence must be between 0 and 1")

    status = str(decision.get("status") or "").lower()
    if row_side == "unknown" and status != "uncertain":
        failures.append(f"decision {sequence} unknown front row_side must be uncertain")
    if row_side in {"left", "right"}:
        missing_mapping = [
            name
            for name in ("row_id", "plant_number", "plant_key")
            if decision.get(name) in (None, "")
        ]
        if missing_mapping:
            failures.append(f"decision {sequence} clear front geometry missing mapping: {', '.join(missing_mapping)}")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_probability(value: Any) -> bool:
    return is_number(value) and 0.0 <= float(value) <= 1.0


def requires_evidence(decision: dict[str, Any]) -> bool:
    status = str(decision.get("status") or "").lower()
    action = str(decision.get("action") or "").strip().lower()
    return bool(decision.get("manual_override")) or status in {"manual", "uncertain"} or action == "inspect or treat"


def is_jpeg(payload: bytes) -> bool:
    return len(payload) >= 4 and payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9")


def validate_csv(zip_file: zipfile.ZipFile, failures: list[str]) -> int:
    rows = list(csv.reader(zip_file.read("events.csv").decode("utf-8").splitlines()))
    if not rows:
        failures.append("events.csv must not be empty")
        return 0
    if rows[0] not in (CSV_HEADER, LEGACY_CSV_HEADER, PRE_TRACK_CSV_HEADER):
        failures.append("events.csv header is not Pi-compatible")
    return len(rows) - 1


def read_json(zip_file: zipfile.ZipFile, name: str, failures: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(zip_file.read(name).decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        failures.append(f"{name} is not valid JSON: {error}")
        return {}
    return value if isinstance(value, dict) else {}


def read_jsonl(zip_file: zipfile.ZipFile, name: str, failures: list[str]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for line_number, line in enumerate(zip_file.read(name).decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            failures.append(f"{name}:{line_number} is not valid JSON: {error}")
            continue
        if isinstance(value, dict):
            decisions.append(value)
    return decisions


def result(
    path: Path,
    failures: list[str],
    decision_count: int = 0,
    evidence_count: int = 0,
) -> dict[str, Any]:
    return {
        "ok": not failures,
        "path": str(path),
        "decision_count": decision_count,
        "evidence_count": evidence_count,
        "failures": failures,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Android field export ZIP from the app.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument(
        "--require-real-field",
        action="store_true",
        help="Reject instrumentation and synthetic android_test_* bundles for field-readiness signoff.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    report = validate_export_bundle(args.bundle, require_real_field=args.require_real_field)
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
