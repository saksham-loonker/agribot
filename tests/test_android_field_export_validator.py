import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "49_validate_android_field_export.py"


def load_module():
    spec = importlib.util.spec_from_file_location("android_field_export_validator", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AndroidFieldExportValidatorTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_accepts_completed_field_export_with_required_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "field_export.zip"
            create_bundle(bundle, include_evidence=True)

            report = self.module.validate_export_bundle(bundle)

        self.assertTrue(report["ok"])
        self.assertEqual([], report["failures"])
        self.assertEqual(2, report["decision_count"])
        self.assertEqual(1, report["evidence_count"])

    def test_accepts_skipped_decision_without_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "field_export.zip"
            create_bundle(bundle, include_evidence=True, include_skipped=True)

            report = self.module.validate_export_bundle(bundle)

        self.assertTrue(report["ok"])
        self.assertEqual([], report["failures"])
        self.assertEqual(3, report["decision_count"])
        self.assertEqual(1, report["evidence_count"])

    def test_rejects_sick_uncertain_or_manual_decision_without_jpeg_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "field_export.zip"
            create_bundle(bundle, include_evidence=False)

            report = self.module.validate_export_bundle(bundle)

        self.assertFalse(report["ok"])
        self.assertIn("decision 2 requires JPEG evidence", report["failures"])

    def test_rejects_evidence_path_with_non_jpeg_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "field_export.zip"
            create_bundle(bundle, include_evidence=True, evidence_bytes=b"not-a-jpeg")

            report = self.module.validate_export_bundle(bundle)

        self.assertFalse(report["ok"])
        self.assertIn("decision 2 evidence is not a JPEG file", report["failures"])

    def test_rejects_incomplete_or_wrong_model_bundle_export(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "field_export.zip"
            create_bundle(bundle, completed=False, model_bundle_id="old-bundle", include_evidence=True)

            report = self.module.validate_export_bundle(bundle)

        self.assertFalse(report["ok"])
        self.assertIn("latest_run completed must be true", report["failures"])
        self.assertIn("latest_run model_bundle_id must be agribot-model-bundle-v001", report["failures"])

    def test_rejects_mismatched_decision_counts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "field_export.zip"
            create_bundle(bundle, include_evidence=True, latest_decision_count=9, summary_total_decisions=8)

            report = self.module.validate_export_bundle(bundle)

        self.assertFalse(report["ok"])
        self.assertIn("latest_run decision_count must match decisions.jsonl", report["failures"])
        self.assertIn("summary total_decisions must match decisions.jsonl", report["failures"])

    def test_rejects_csv_row_count_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "field_export.zip"
            create_bundle(bundle, include_evidence=True, omit_second_csv_row=True)

            report = self.module.validate_export_bundle(bundle)

        self.assertFalse(report["ok"])
        self.assertIn("events.csv row count must match decisions.jsonl", report["failures"])

    def test_real_field_gate_rejects_instrumentation_run_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "field_export.zip"
            create_bundle(bundle, include_evidence=True, run_id="android_test_1780991039457")

            smoke_report = self.module.validate_export_bundle(bundle)
            field_report = self.module.validate_export_bundle(bundle, require_real_field=True)

        self.assertTrue(smoke_report["ok"])
        self.assertFalse(field_report["ok"])
        self.assertIn("field export must come from a real field run, not android_test_*", field_report["failures"])

    def test_rejects_missing_app_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "field_export.zip"
            create_bundle(bundle, include_evidence=True, include_app_log=False)

            report = self.module.validate_export_bundle(bundle)

        self.assertFalse(report["ok"])
        self.assertIn("missing app_log.jsonl", report["failures"])

    def test_rejects_app_log_without_started_event(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "field_export.zip"
            create_bundle(bundle, include_evidence=True, app_log_event_type="model_loaded")

            report = self.module.validate_export_bundle(bundle)

        self.assertFalse(report["ok"])
        self.assertIn("app_log.jsonl must include started event", report["failures"])

    def test_accepts_front_overview_export_with_geometry_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "front_export.zip"
            create_front_bundle(bundle)

            report = self.module.validate_export_bundle(bundle)

        self.assertTrue(report["ok"])
        self.assertEqual([], report["failures"])
        self.assertEqual(1, report["decision_count"])

    def test_rejects_front_overview_without_bbox(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "front_export.zip"
            create_front_bundle(bundle, decision_overrides={"bbox_px": None})

            report = self.module.validate_export_bundle(bundle)

        self.assertFalse(report["ok"])
        self.assertIn("decision 1 missing required fields: bbox_px", report["failures"])
        self.assertIn("decision 1 front bbox_px must contain four numeric values", report["failures"])

    def test_rejects_front_overview_unknown_side_without_uncertainty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "front_export.zip"
            create_front_bundle(bundle, decision_overrides={"row_side": "unknown", "status": "ok"})

            report = self.module.validate_export_bundle(bundle)

        self.assertFalse(report["ok"])
        self.assertIn("decision 1 unknown front row_side must be uncertain", report["failures"])

    def test_rejects_clear_front_overview_side_without_plant_mapping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "front_export.zip"
            create_front_bundle(bundle, decision_overrides={"row_side": "left", "plant_number": None, "plant_key": None})

            report = self.module.validate_export_bundle(bundle)

        self.assertFalse(report["ok"])
        self.assertIn("decision 1 clear front geometry missing mapping: plant_number, plant_key", report["failures"])

    def test_rejects_diagnostics_that_require_a_network(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "field_export.zip"
            create_bundle(
                bundle,
                include_evidence=True,
                diagnostics={"offline": True, "network_required": True, "status": "captured"},
            )

            report = self.module.validate_export_bundle(bundle)

        self.assertFalse(report["ok"])
        self.assertIn("diagnostics network_required must be false", report["failures"])

    def test_rejects_obsolete_cpu_default_diagnostics_field(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "field_export.zip"
            create_bundle(
                bundle,
                include_evidence=True,
                diagnostics={
                    "offline": True,
                    "network_required": False,
                    "status": "captured",
                    "cpu_default": True,
                },
            )

            report = self.module.validate_export_bundle(bundle)

        self.assertFalse(report["ok"])
        self.assertIn("diagnostics cpu_default is obsolete; use cpu_thread_profile", report["failures"])


def create_bundle(
    path,
    include_evidence,
    completed=True,
    model_bundle_id="agribot-model-bundle-v001",
    evidence_bytes=b"\xff\xd8\xff\xe0agribot-test\xff\xd9",
    include_skipped=False,
    include_app_log=True,
    app_log_event_type="started",
    latest_decision_count=None,
    summary_total_decisions=None,
    omit_second_csv_row=False,
    run_id="run_1",
    diagnostics=None,
):
    decisions = [
        decision(sequence=1, label="Healthy", status="ok", evidence_path=None, run_id=run_id),
        decision(
            sequence=2,
            label="Early_blight",
            status="manual",
            manual_override=True,
            evidence_path="evidence_frames/decision_2.jpg",
            run_id=run_id,
        ),
    ]
    if include_skipped:
        decisions.append(
            decision(sequence=3, label="Skipped", status="skipped", action="Rescan this plant", run_id=run_id),
        )
    latest_run = {
        "latest_run_id": run_id,
        "run_id": run_id,
        "completed": completed,
        "mode": "SIDE_SCAN",
        "state": "completed" if completed else "recording",
        "decision_count": latest_decision_count if latest_decision_count is not None else len(decisions),
        "sick_count": 1,
        "uncertain_count": 0,
        "model_bundle_id": model_bundle_id,
    }
    summary = {
        "run_id": run_id,
        "completed": completed,
        "total_decisions": summary_total_decisions if summary_total_decisions is not None else len(decisions),
        "sick_count": 1,
        "uncertain_count": 0,
    }
    metadata = {
        "run_id": run_id,
        "origin": "android",
        "mode": "SIDE_SCAN",
        "model_bundle_id": model_bundle_id,
    }
    field_layout = {
        "active_field_id": "field_2",
        "fields": [
            {
                "field_id": "Field 2",
                "active_row_id": "B",
                "rows": [{"row_id": "B", "row_index": 2, "plants_per_row": 57}],
            }
        ],
    }
    csv_lines = [
        "sequence,timestamp,epoch_time,field_id,row_id,row_index,plant_column,plant_number,"
        "plant_key,x_m,y_m,plant_display,scan_index,plant_id,label,confidence,status,action,"
        "frames_used,reason,late_frames,gate_avg_latency_ms,gate_max_latency_ms,threads,fps_target",
        "1,2026-06-08T09:25:21Z,1780910721.0,Field 2,B,2,1,1,Field 2|B|1,0.0,1.4,"
        "Field 2 / Row B / Plant 1,1,1,Healthy,0.91,ok,No action,2,primary_agreement,0,80.0,90.0,4,5.0",
    ]
    if not omit_second_csv_row:
        csv_lines.append(
            "2,2026-06-08T09:25:22Z,1780910722.0,Field 2,B,2,2,2,Field 2|B|2,0.35,1.4,"
            "Field 2 / Row B / Plant 2,2,2,Early_blight,0.91,manual,Inspect or treat,2,"
            "primary_agreement,0,80.0,90.0,4,5.0",
        )
    if include_skipped:
        csv_lines.append(
            "3,2026-06-08T09:25:23Z,1780910723.0,Field 2,B,2,3,3,Field 2|B|3,0.7,1.4,"
            "Field 2 / Row B / Plant 3,3,3,Skipped,0.91,skipped,Rescan this plant,2,"
            "primary_agreement,0,80.0,90.0,4,5.0",
        )
    csv = "\n".join(csv_lines) + "\n"

    with zipfile.ZipFile(path, "w") as zip_file:
        zip_file.writestr("metadata.json", json.dumps(metadata))
        zip_file.writestr("summary.json", json.dumps(summary))
        zip_file.writestr("latest_run.json", json.dumps(latest_run))
        zip_file.writestr("field_layout.json", json.dumps(field_layout))
        zip_file.writestr(
            "diagnostics.json",
            json.dumps(
                diagnostics
                or {
                    "offline": True,
                    "network_required": False,
                    "status": "captured",
                    "cpu_thread_profile": "test",
                    "measurement_source": "none",
                }
            )
            + "\n",
        )
        zip_file.writestr("events.csv", csv)
        if include_app_log:
            zip_file.writestr(
                "app_log.jsonl",
                json.dumps(
                    {
                        "id": f"{run_id}_{app_log_event_type}_20260608T092400Z",
                        "run_id": run_id,
                        "timestamp": "2026-06-08T09:24:00Z",
                        "source": "agribot_android",
                        "level": "info",
                        "type": app_log_event_type,
                        "payload": {"mode": "SIDE_SCAN", "target_fps": 5},
                    }
                )
                + "\n",
            )
        zip_file.writestr("decisions.jsonl", "\n".join(json.dumps(item) for item in decisions) + "\n")
        if include_evidence:
            zip_file.writestr("evidence_frames/decision_2.jpg", evidence_bytes)


def create_front_bundle(path, decision_overrides=None):
    run_id = "android_20260608_150000"
    current_decision = front_decision(run_id=run_id)
    for key, value in (decision_overrides or {}).items():
        current_decision[key] = value
    latest_run = {
        "latest_run_id": run_id,
        "run_id": run_id,
        "completed": True,
        "mode": "FRONT_ROW_OVERVIEW",
        "state": "completed",
        "decision_count": 1,
        "sick_count": 0,
        "uncertain_count": 0,
        "model_bundle_id": "agribot-model-bundle-v001",
    }
    summary = {
        "run_id": run_id,
        "completed": True,
        "total_decisions": 1,
        "sick_count": 0,
        "uncertain_count": 0,
    }
    metadata = {
        "run_id": run_id,
        "origin": "android",
        "mode": "FRONT_ROW_OVERVIEW",
        "model_bundle_id": "agribot-model-bundle-v001",
    }
    field_layout = {
        "active_field_id": "field_2",
        "fields": [
            {
                "field_id": "Field 2",
                "active_row_id": "B",
                "rows": [{"row_id": "B", "row_index": 2, "plants_per_row": 57}],
            }
        ],
    }
    csv = "\n".join(
        [
            "sequence,timestamp,epoch_time,field_id,row_id,row_index,plant_column,plant_number,"
            "plant_key,x_m,y_m,plant_display,scan_index,plant_id,label,confidence,status,action,"
            "frames_used,reason,late_frames,gate_avg_latency_ms,gate_max_latency_ms,threads,fps_target",
            "1,2026-06-08T09:25:21Z,1780910721.0,field_2,B,2,4,4,B:4,,1.4,"
            "B:4,1,4,Healthy,0.91,ok,No action,5,front_burst_review,0,0.0,0.0,4,5.0",
        ]
    ) + "\n"

    with zipfile.ZipFile(path, "w") as zip_file:
        zip_file.writestr("metadata.json", json.dumps(metadata))
        zip_file.writestr("summary.json", json.dumps(summary))
        zip_file.writestr("latest_run.json", json.dumps(latest_run))
        zip_file.writestr("field_layout.json", json.dumps(field_layout))
        zip_file.writestr(
            "diagnostics.json",
            "{\"offline\":true,\"network_required\":false,\"status\":\"captured\","
            "\"cpu_thread_profile\":\"test\",\"measurement_source\":\"none\"}\n",
        )
        zip_file.writestr("events.csv", csv)
        zip_file.writestr(
            "app_log.jsonl",
            json.dumps(
                {
                    "id": f"{run_id}_started_20260608T092400Z",
                    "run_id": run_id,
                    "timestamp": "2026-06-08T09:24:00Z",
                    "source": "agribot_android",
                    "level": "info",
                    "type": "started",
                    "payload": {"mode": "FRONT_ROW_OVERVIEW", "target_fps": 5},
                }
            )
            + "\n",
        )
        zip_file.writestr("decisions.jsonl", json.dumps(current_decision) + "\n")


def decision(
    sequence,
    label,
    status,
    action=None,
    manual_override=False,
    evidence_path=None,
    run_id="run_1",
):
    action_value = action if action is not None else ("No action" if status == "ok" else "Inspect or treat")
    return {
        "run_id": run_id,
        "sequence": sequence,
        "timestamp": "2026-06-08T09:25:21Z",
        "mode": "SIDE_SCAN",
        "field_id": "Field 2",
        "row_id": "B",
        "row_index": 2,
        "plant_column": sequence,
        "plant_number": sequence,
        "plant_key": f"Field 2|B|{sequence}",
        "x_m": 0.35 * (sequence - 1),
        "y_m": 1.4,
        "label": label,
        "confidence": 0.91,
        "raw_label": label,
        "status": status,
        "action": action_value,
        "frames_used": 2,
        "reason": "primary_agreement",
        "late_frames": 0,
        "gate_avg_latency_ms": 80.0,
        "gate_max_latency_ms": 90.0,
        "model_version": "agribot-model-bundle-v001",
        "scan_index": sequence,
        "scan_pass": 1,
        "planned_total_plants": 57,
        "plant_display": f"Field 2 / Row B / Plant {sequence}",
        "plant_id": sequence,
        "manual_override": manual_override,
        "evidence_path": evidence_path,
        "evidence_status": "captured" if evidence_path else None,
    }


def front_decision(run_id="android_20260608_150000"):
    return {
        "run_id": run_id,
        "sequence": 1,
        "timestamp": "2026-06-08T09:25:21Z",
        "mode": "FRONT_ROW_OVERVIEW",
        "field_id": "field_2",
        "row_id": "B",
        "row_side": "left",
        "row_index": 2,
        "plant_column": 4,
        "plant_number": 4,
        "plant_key": "B:4",
        "y_m": 1.4,
        "bbox_px": [120.0, 220.0, 260.0, 420.0],
        "geometry_confidence": 0.86,
        "plant_position_confidence": 0.79,
        "geometry_reason": "within_left_rail",
        "label": "Healthy",
        "confidence": 0.91,
        "raw_label": "Healthy",
        "status": "ok",
        "action": "No action",
        "frames_used": 5,
        "reason": "front_burst_review",
        "late_frames": 0,
        "gate_avg_latency_ms": 0.0,
        "gate_max_latency_ms": 0.0,
        "model_version": "agribot-model-bundle-v001",
        "scan_index": 1,
        "plant_id": 4,
        "manual_override": False,
        "evidence_path": None,
        "evidence_status": None,
    }


if __name__ == "__main__":
    unittest.main()
