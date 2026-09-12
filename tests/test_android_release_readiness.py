import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "48_android_release_readiness.py"


def load_module():
    spec = importlib.util.spec_from_file_location("android_release_readiness", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AndroidReleaseReadinessTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_manifest_accepts_exact_offline_model_bundle(self):
        result = self.module.evaluate_model_bundle(manifest(), labels())

        self.assertTrue(result["ok"])
        self.assertEqual([], result["failures"])

    def test_manifest_rejects_label_reordering(self):
        wrong_labels = labels()
        wrong_labels[1], wrong_labels[2] = wrong_labels[2], wrong_labels[1]

        result = self.module.evaluate_model_bundle(manifest(), wrong_labels)

        self.assertFalse(result["ok"])
        self.assertIn("labels are not in Pi-compatible order", result["failures"])

    def test_manifest_rejects_labels_hash_mismatch(self):
        current = manifest()
        current["labels_sha256"] = "0" * 64

        result = self.module.evaluate_model_bundle(current, labels())

        self.assertFalse(result["ok"])
        self.assertIn(
            "labels sha256 expected 'c9dd8ac5138e41dbb3cd9a7130542c70e463277318941d6a17acc5841c03992b', got '0000000000000000000000000000000000000000000000000000000000000000'",
            result["failures"],
        )

    def test_field_ready_requires_signed_release_and_physical_benchmark(self):
        report = self.module.evaluate_readiness(
            model_bundle={"ok": True, "failures": []},
            debug_apk={"ok": True, "failures": []},
            release_apk={
                "ok": False,
                "signed": False,
                "failures": ["release APK is not signed with v2/v3 scheme"],
            },
            benchmark=benchmark(is_emulator=True),
            field_export={"ok": False, "failures": ["field export bundle not provided"]},
        )

        self.assertTrue(report["development_ready"])
        self.assertFalse(report["field_ready"])
        self.assertIn("release APK is not signed with v2/v3 scheme", report["field_blockers"])
        self.assertEqual(1, report["field_blockers"].count("release APK is not signed with v2/v3 scheme"))
        self.assertIn("benchmark is from an emulator, not a physical phone", report["field_blockers"])
        self.assertIn("field export bundle not provided", report["field_blockers"])

    def test_field_ready_passes_with_signed_release_and_latency_gates(self):
        report = self.module.evaluate_readiness(
            model_bundle={"ok": True, "failures": []},
            debug_apk={"ok": True, "failures": []},
            release_apk={"ok": True, "signed": True, "failures": []},
            benchmark=benchmark(is_emulator=False),
            field_export={"ok": True, "failures": []},
        )

        self.assertTrue(report["development_ready"])
        self.assertTrue(report["field_ready"])
        self.assertEqual([], report["field_blockers"])

    def test_apk_readiness_rejects_legacy_packaged_assets(self):
        class BuildModule:
            @staticmethod
            def use_local_toolchain():
                return None

            @staticmethod
            def verify_apk_metadata(path):
                return None

            @staticmethod
            def verify_no_legacy_apk_assets(path):
                raise SystemExit("[ERROR] APK contains legacy Pi dashboard/Capacitor assets")

            @staticmethod
            def verify_apk_signature(path):
                return True

        with tempfile.TemporaryDirectory() as temp_dir:
            apk = Path(temp_dir) / "agribot.apk"
            apk.write_bytes(b"apk")
            with patch.object(self.module, "load_build_module", return_value=BuildModule):
                result = self.module.evaluate_apk(apk, require_signature=True)

        self.assertFalse(result["ok"])
        self.assertTrue(result["signed"])
        self.assertIn("legacy Pi dashboard", result["failures"][0])

    def test_benchmark_accepts_ready_status_with_bundle_suffix(self):
        current = benchmark(is_emulator=False)
        current["readiness_status"] = "Ready: agribot-model-bundle-v001"

        result = self.module.evaluate_benchmark(current)

        self.assertTrue(result["ok"])
        self.assertEqual([], result["failures"])

    def test_benchmark_treats_emulator_timings_as_smoke_only(self):
        current = benchmark(is_emulator=True)
        current["classifier"] = {"avg_ms": 3000.0, "p95_ms": 3500.0}
        current["detector"] = {"p95_ms": 700.0}

        result = self.module.evaluate_benchmark(current)

        self.assertFalse(result["ok"])
        self.assertEqual(["benchmark is from an emulator, not a physical phone"], result["failures"])

    def test_benchmark_rejects_physical_phone_without_arm64_abi(self):
        current = benchmark(is_emulator=False)
        current["device"]["supported_abis"] = "armeabi-v7a"

        result = self.module.evaluate_benchmark(current)

        self.assertFalse(result["ok"])
        self.assertIn("benchmark physical phone does not advertise arm64-v8a ABI", result["failures"])

    def test_benchmark_rejects_wrong_package_or_model_bundle(self):
        wrong_package = benchmark(is_emulator=False)
        wrong_package["package"] = "com.example.other"
        wrong_bundle = benchmark(is_emulator=False)
        wrong_bundle["model_bundle_id"] = "old-bundle"

        package_result = self.module.evaluate_benchmark(wrong_package)
        bundle_result = self.module.evaluate_benchmark(wrong_bundle)

        self.assertFalse(package_result["ok"])
        self.assertIn("benchmark package does not match com.sakshyam.agribot", package_result["failures"])
        self.assertFalse(bundle_result["ok"])
        self.assertIn("benchmark model bundle does not match agribot-model-bundle-v001", bundle_result["failures"])

    def test_benchmark_rejects_non_ready_model_status(self):
        stale = benchmark(is_emulator=False)
        stale["readiness_status"] = "missing classifier"

        result = self.module.evaluate_benchmark(stale)

        self.assertFalse(result["ok"])
        self.assertIn("benchmark model readiness was not Ready", result["failures"])

    def test_cli_accepts_custom_physical_benchmark_report_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_report = Path(temp_dir) / "target-phone.json"
            custom_bundle = Path(temp_dir) / "field-export.zip"
            args = self.module.parse_args(
                ["--benchmark-report", str(custom_report), "--field-export-bundle", str(custom_bundle)],
            )

        self.assertEqual(custom_report, args.benchmark_report)
        self.assertEqual(custom_bundle, args.field_export_bundle)

    def test_readiness_field_export_requires_real_field_bundle(self):
        calls = []

        class Validator:
            @staticmethod
            def validate_export_bundle(path, require_real_field=False):
                calls.append((path, require_real_field))
                return {"ok": True, "failures": []}

        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "field-export.zip"
            with patch.object(self.module, "load_field_export_module", return_value=Validator):
                result = self.module.evaluate_field_export(bundle)

        self.assertTrue(result["ok"])
        self.assertEqual([(bundle, True)], calls)


def manifest():
    return {
        "labels_sha256": "c9dd8ac5138e41dbb3cd9a7130542c70e463277318941d6a17acc5841c03992b",
        "models": {
            "classifier": {
                "file": "models/classifier_fastcrop_float32.tflite",
                "sha256": "8478c47e795bbf2f1beffe2f21b20d04aa5410148560d7f1912bfa8fbf2158f8",
                "confidence_threshold": 0.62,
                "high_confidence_threshold": 0.90,
            },
            "detector": {
                "file": "models/detector_nano_256_raw_float32.tflite",
                "sha256": "83e7521447c7de4730dfea79eb830d286a4ecf44aa3f5cc8c6080947969f6910",
            },
        },
        "compatibility": {
            "min_android_sdk": 26,
            "cpu_default": True,
            "requires_network": False,
        },
    }


def labels():
    return [
        "Early_blight",
        "Healthy",
        "Late_blight",
        "Leaf Miner",
        "Magnesium Deficiency",
        "Nitrogen Deficiency",
        "Pottassium Deficiency",
        "Spotted Wilt Virus",
    ]


def benchmark(is_emulator):
    return {
        "package": "com.sakshyam.agribot",
        "model_bundle_id": "agribot-model-bundle-v001",
        "readiness_status": "Ready",
        "device": {
            "manufacturer": "Test",
            "model": "ReferencePhone",
            "supported_abis": "arm64-v8a,armeabi-v7a",
            "is_emulator": is_emulator,
        },
        "classifier": {"avg_ms": 120.0, "p95_ms": 150.0},
        "detector": {"p95_ms": 260.0},
    }


if __name__ == "__main__":
    unittest.main()
