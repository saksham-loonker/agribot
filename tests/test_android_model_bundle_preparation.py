import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from contextlib import ExitStack
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

    def build_app_assets_only_report(
        self,
        root: Path,
        *,
        classifier_hash_override: str | None = None,
        detector_hash_override: str | None = None,
        labels_override: list[str] | None = None,
        classifier_content: bytes = b"classifier asset",
    ):
        classifier = root / "classifier.tflite"
        detector = root / "detector.tflite"
        labels = root / "labels.json"
        manifest = root / "model_manifest.json"
        classifier.write_bytes(classifier_content)
        detector.write_bytes(b"detector asset")
        app_labels = self.module.EXPECTED_LABELS if labels_override is None else labels_override
        labels.write_text(json.dumps(app_labels) + "\n", encoding="utf-8")
        manifest.write_text(
            json.dumps(
                {
                    "labels_sha256": self.module.sha256(labels),
                    "models": {
                        "classifier": {
                            "file": "models/classifier_fastcrop_float32.tflite",
                            "sha256": classifier_hash_override or self.module.sha256(classifier),
                        },
                        "detector": {
                            "file": "models/detector_nano_256_raw_float32.tflite",
                            "sha256": detector_hash_override or self.module.sha256(detector),
                        },
                    },
                }
            ),
            encoding="utf-8",
        )

        with ExitStack() as stack:
            stack.enter_context(patch.object(self.module, "CLASSIFIER_SOURCE", root / "missing-classifier.pt"))
            stack.enter_context(patch.object(self.module, "DETECTOR_SOURCE", root / "missing-detector.pt"))
            stack.enter_context(patch.object(self.module, "CLASSIFIER_EXPORTS", {"int8": root / "missing-classifier-int8.tflite"}))
            stack.enter_context(patch.object(self.module, "DETECTOR_EXPORTS", {"int8": root / "missing-detector-int8.tflite"}))
            stack.enter_context(patch.object(self.module, "APP_CLASSIFIER", classifier))
            stack.enter_context(patch.object(self.module, "APP_DETECTOR", detector))
            stack.enter_context(patch.object(self.module, "APP_LABELS", labels))
            stack.enter_context(patch.object(self.module, "APP_MANIFEST", manifest))
            stack.enter_context(patch.object(self.module, "inspect_tflite_tensors", return_value={"status": "unverified"}))
            return self.module.build_report([], app_assets_only=True)

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

    def test_app_assets_only_readiness_does_not_require_training_weights_or_alternate_exports(self):
        with tempfile.TemporaryDirectory(dir=self.module.ROOT) as temp_dir:
            report = self.build_app_assets_only_report(Path(temp_dir))

        self.assertTrue(report["ready"], report["blockers"])
        self.assertEqual([], report["blockers"])

    def test_app_assets_only_readiness_rejects_model_hash_mismatch(self):
        with tempfile.TemporaryDirectory(dir=self.module.ROOT) as temp_dir:
            report = self.build_app_assets_only_report(Path(temp_dir), classifier_hash_override="0" * 64)

        self.assertFalse(report["ready"])
        self.assertTrue(any("classifier" in blocker and "SHA-256" in blocker for blocker in report["blockers"]))

    def test_app_assets_only_readiness_rejects_detector_hash_mismatch(self):
        with tempfile.TemporaryDirectory(dir=self.module.ROOT) as temp_dir:
            report = self.build_app_assets_only_report(Path(temp_dir), detector_hash_override="0" * 64)

        self.assertFalse(report["ready"])
        self.assertTrue(any("detector" in blocker and "SHA-256" in blocker for blocker in report["blockers"]))

    def test_app_assets_only_readiness_rejects_labels_changed_with_manifest_hash(self):
        labels = list(self.module.EXPECTED_LABELS)
        labels[0], labels[1] = labels[1], labels[0]
        with tempfile.TemporaryDirectory(dir=self.module.ROOT) as temp_dir:
            report = self.build_app_assets_only_report(Path(temp_dir), labels_override=labels)

        self.assertFalse(report["ready"])
        self.assertTrue(any("labels" in blocker and "expected" in blocker for blocker in report["blockers"]))

    def test_app_assets_only_readiness_rejects_unhydrated_lfs_pointer(self):
        pointer = (
            b"version https://git-lfs.github.com/spec/v1\n"
            b"oid sha256:8478c47e795bbf2f1beffe2f21b20d04aa5410148560d7f1912bfa8fbf2158f8\n"
            b"size 51385748\n"
        )
        with tempfile.TemporaryDirectory(dir=self.module.ROOT) as temp_dir:
            report = self.build_app_assets_only_report(
                Path(temp_dir),
                classifier_hash_override="8478c47e795bbf2f1beffe2f21b20d04aa5410148560d7f1912bfa8fbf2158f8",
                classifier_content=pointer,
            )

        self.assertFalse(report["ready"])
        self.assertTrue(any("classifier" in blocker and "SHA-256" in blocker for blocker in report["blockers"]))


if __name__ == "__main__":
    unittest.main()
