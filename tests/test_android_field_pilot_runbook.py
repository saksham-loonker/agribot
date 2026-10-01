import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "agribot_android_app" / "FIELD_PILOT_RUNBOOK.md"
README = ROOT / "agribot_android_app" / "README_APK.md"


class AndroidFieldPilotRunbookTest(unittest.TestCase):
    def test_runbook_exists_and_is_linked_from_apk_readme(self):
        self.assertTrue(RUNBOOK.exists())
        self.assertIn("FIELD_PILOT_RUNBOOK.md", README.read_text(encoding="utf-8"))

    def test_runbook_keeps_physical_phone_and_real_export_gates(self):
        text = RUNBOOK.read_text(encoding="utf-8")
        self.assertIn("AgribotModelGoldenTest", text)
        self.assertIn("AgribotBench", text)
        self.assertIn("python 49_validate_android_field_export.py", text)
        self.assertIn("--require-real-field", text)
        self.assertIn("dist\\agribot_real_field_export.zip", text)
        self.assertIn("Do not claim field readiness from an emulator benchmark.", text)
        self.assertIn("Do not claim field readiness from `android_test_*` exports", text)
        self.assertIn("named human reviewer", text)

    def test_runbook_covers_both_modes(self):
        text = RUNBOOK.read_text(encoding="utf-8")
        self.assertIn("## 2. Check a plant pilot", text)
        self.assertIn("## 3. Walk rows pilot", text)
        self.assertIn("**Next plant**", text)
        self.assertIn("Share all data and photos (ZIP)", text)

    def test_runbook_keeps_offline_constraints_visible(self):
        text = RUNBOOK.read_text(encoding="utf-8")
        self.assertIn("without a Raspberry Pi", text)
        self.assertIn("internet access", text)
        self.assertIn("cloud inference", text)
        self.assertIn("grant only the camera permission", text)
        self.assertIn("has no `INTERNET`", text)


if __name__ == "__main__":
    unittest.main()
