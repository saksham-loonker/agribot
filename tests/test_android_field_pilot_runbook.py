import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "agribot_android_app" / "FIELD_PILOT_RUNBOOK.md"
README = ROOT / "agribot_android_app" / "README_APK.md"


class AndroidFieldPilotRunbookTest(unittest.TestCase):
    def test_runbook_exists_and_is_linked_from_apk_readme(self):
        self.assertTrue(RUNBOOK.exists())
        readme = README.read_text(encoding="utf-8")

        self.assertIn("FIELD_PILOT_RUNBOOK.md", readme)

    def test_runbook_preserves_physical_phone_and_real_export_gates(self):
        text = RUNBOOK.read_text(encoding="utf-8")

        self.assertIn("python 47_android_model_benchmark.py", text)
        self.assertIn("dist\\embedded_model_benchmark_target_phone.json", text)
        self.assertIn("python 48_android_release_readiness.py", text)
        self.assertIn("--require-field-ready", text)
        self.assertIn("dist\\agribot_real_field_export.zip", text)
        self.assertIn("Do not claim field readiness from an emulator benchmark.", text)
        self.assertIn("Do not claim field readiness from `android_test_*` exports.", text)

    def test_runbook_covers_both_required_recording_modes(self):
        text = RUNBOOK.read_text(encoding="utf-8")

        self.assertIn("## Side Scan Pilot", text)
        self.assertIn("Tap Start Side Scan", text)
        self.assertIn("Use Retake", text)
        self.assertIn("Use Snapshot", text)
        self.assertIn("## Front Row Overview Pilot", text)
        self.assertIn("Capture Burst", text)
        self.assertIn("Mark ambiguous geometry as uncertain", text)

    def test_runbook_keeps_offline_direct_replacement_constraints_visible(self):
        text = RUNBOOK.read_text(encoding="utf-8")

        self.assertIn("without a Raspberry Pi", text)
        self.assertIn("internet access", text)
        self.assertIn("cloud inference", text)
        self.assertIn("grant only the camera permission", text)
        self.assertIn("has no `INTERNET` permission", text)


if __name__ == "__main__":
    unittest.main()
