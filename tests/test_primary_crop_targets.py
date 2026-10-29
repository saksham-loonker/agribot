"""Primary crop supervision must respect annotation ambiguity and image scope."""

from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest

READY = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "timm", "sklearn", "cv2")
)
if READY:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_training/v3"))
    from phase41_siglip_genus import targets


@unittest.skipUnless(READY, "Optional training environment required")
class PrimaryCropTargetTests(unittest.TestCase):
    def labels(self, rows):
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "annotations.csv"
            csv.write_text(
                "filename,class\n"
                "healthy.jpg,Tomato leaf\n"
                "blight.jpg,Tomato Early blight leaf\n"
                "pepper.jpg,Bell_pepper leaf\n"
                "mixed.jpg,Tomato leaf\n"
                "mixed.jpg,Bell_pepper leaf\n",
                encoding="utf-8",
            )
            return targets(rows, csv).tolist()

    def test_both_healthy_and_diseased_tomato_share_taxonomy(self):
        rows = [
            dict(path=name, source="plantdoc")
            for name in ("healthy.jpg", "blight.jpg", "pepper.jpg")
        ]
        self.assertEqual(self.labels(rows), [1, 1, 0])

    def test_conflicting_or_missing_crop_annotations_are_masked(self):
        rows = [
            dict(path=name, source="plantdoc") for name in ("mixed.jpg", "missing.jpg")
        ]
        self.assertEqual(self.labels(rows), [-1, -1])

    def test_primary_image_taxonomy_is_never_copied_to_a_detected_roi(self):
        rows = [
            dict(path="healthy.jpg", source=source, box=[0, 0, 100, 100])
            for source in ("plantdoc", "ad_greenhouse")
        ]
        self.assertEqual(self.labels(rows), [-1, -1])

    def test_unknown_source_is_masked_and_condition_label_does_not_define_crop(self):
        rows = [
            dict(path="image.jpg", source=source, label=label)
            for source, label in (
                ("unknown", "Healthy"),
                ("ad_greenhouse", "Other"),
                ("tom2024", "Healthy"),
            )
        ]
        self.assertEqual(self.labels(rows), [-1, 1, 1])


if __name__ == "__main__":
    unittest.main()
