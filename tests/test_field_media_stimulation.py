from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("field_media_stimulation", ROOT / "50_stimulate_field_media.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

FIELD_SPEC = importlib.util.spec_from_file_location(
    "field_media_fixture_runner", ROOT / "tests" / "field_stimulation" / "stimulate_field_media.py"
)
assert FIELD_SPEC is not None and FIELD_SPEC.loader is not None
FIELD_MODULE = importlib.util.module_from_spec(FIELD_SPEC)
FIELD_SPEC.loader.exec_module(FIELD_MODULE)


class FieldMediaStimulationTests(unittest.TestCase):
    def test_boundary_boxes_remain_inside_an_aspect_ratio_padded_frame(self) -> None:
        report = MODULE.run_geometry_stimulation(960, 640)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(len(report["mapped_boundary_boxes"]), 5)
        for left, top, right, bottom in report["mapped_boundary_boxes"].values():
            self.assertGreaterEqual(left, 0.0)
            self.assertGreaterEqual(top, 0.0)
            self.assertLessEqual(right, 960.0)
            self.assertLessEqual(bottom, 640.0)
            self.assertLess(left, right)
            self.assertLess(top, bottom)

    def test_invalid_box_is_rejected(self) -> None:
        self.assertIsNone(MODULE.map_model_box_to_frame((float("nan"), 0.0, 20.0, 20.0), 960, 640))
        self.assertIsNone(MODULE.map_model_box_to_frame((20.0, 20.0, 10.0, 10.0), 960, 640))

    def test_runner_fails_when_no_local_image_fixtures_are_available(self) -> None:
        with tempfile.TemporaryDirectory() as fixture_dir:
            with patch.object(FIELD_MODULE, "FIXTURE_DIR", Path(fixture_dir)):
                report = FIELD_MODULE.run()

        self.assertEqual(report["status"], "failed")
        self.assertIn("no image fixtures", " ".join(report["failures"]))

    def test_checked_public_image_is_optional_for_general_python_suite(self) -> None:
        image_value = os.environ.get("AGRIBOT_FIELD_MEDIA_IMAGE")
        if not image_value:
            self.skipTest("set AGRIBOT_FIELD_MEDIA_IMAGE to run the optional public-media check")
        image = Path(image_value)
        if not image.is_file():
            self.skipTest("AGRIBOT_FIELD_MEDIA_IMAGE does not point to a file")
        report = MODULE.build_report(image)
        self.assertEqual(report["status"], "pass")
        self.assertFalse(report["android_model_inference"]["executed"])


if __name__ == "__main__":
    unittest.main()
