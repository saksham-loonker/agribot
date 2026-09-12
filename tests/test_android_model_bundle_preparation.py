import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "46_prepare_android_model_bundle.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prepare_android_model_bundle", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AndroidModelBundlePreparationTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_manifest_records_labels_sha256(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            classifier = root / "classifier.tflite"
            detector = root / "detector.tflite"
            labels = root / "labels.json"
            classifier.write_bytes(b"classifier")
            detector.write_bytes(b"detector")
            labels.write_text("[\"Early_blight\",\"Healthy\"]\n", encoding="utf-8")
            expected_hash = self.module.sha256(labels)

            with patch.object(self.module, "APP_LABELS", labels), \
                    patch.object(self.module, "inspect_tflite_tensors", return_value={"status": "missing"}):
                manifest = self.module.build_manifest(classifier, detector)

        self.assertEqual(expected_hash, manifest["labels_sha256"])


if __name__ == "__main__":
    unittest.main()
