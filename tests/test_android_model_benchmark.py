import argparse
import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "47_android_model_benchmark.py"


def load_module():
    spec = importlib.util.spec_from_file_location("android_model_benchmark", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AndroidModelBenchmarkGateTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_accepts_physical_phone_when_limits_pass(self):
        self.module.validate_report(
            report(
                is_emulator=False,
                model="ReferencePhone",
                classifier_avg_ms=120.0,
                classifier_p95_ms=150.0,
                detector_p95_ms=260.0,
            ),
            args(require_physical=True, expected_model="ReferencePhone", enforce_target_limits=True),
        )

    def test_accepts_ready_status_with_bundle_suffix(self):
        current = report(is_emulator=False, model="ReferencePhone")
        current["readiness_status"] = "Ready: agribot-model-bundle-v001"

        self.module.validate_report(current, args())

    def test_rejects_emulator_when_physical_phone_required(self):
        with self.assertRaisesRegex(SystemExit, "Physical target phone required"):
            self.module.validate_report(
                report(is_emulator=True, model="sdk_gphone64_x86_64"),
                args(require_physical=True),
            )

    def test_rejects_unexpected_device_model(self):
        with self.assertRaisesRegex(SystemExit, "Expected benchmark device model"):
            self.module.validate_report(
                report(is_emulator=False, model="DifferentPhone"),
                args(expected_model="ReferencePhone"),
            )

    def test_rejects_latency_gate_failures(self):
        with self.assertRaisesRegex(SystemExit, "classifier avg_ms"):
            self.module.validate_report(
                report(
                    is_emulator=False,
                    model="ReferencePhone",
                    classifier_avg_ms=161.0,
                    classifier_p95_ms=201.0,
                    detector_p95_ms=301.0,
                ),
                args(enforce_target_limits=True),
            )

    def test_rejects_physical_phone_without_arm64_abi(self):
        with self.assertRaisesRegex(SystemExit, "arm64-v8a"):
            self.module.validate_report(
                report(
                    is_emulator=False,
                    model="ReferencePhone",
                    supported_abis="armeabi-v7a",
                ),
                args(require_physical=True),
            )

    def test_rejects_benchmark_from_wrong_package(self):
        stale = report(is_emulator=False, model="ReferencePhone")
        stale["package"] = "com.example.other"

        with self.assertRaisesRegex(SystemExit, "Benchmark package"):
            self.module.validate_report(stale, args())

    def test_rejects_benchmark_from_wrong_model_bundle(self):
        stale = report(is_emulator=False, model="ReferencePhone")
        stale["model_bundle_id"] = "old-bundle"

        with self.assertRaisesRegex(SystemExit, "Benchmark model_bundle_id"):
            self.module.validate_report(stale, args())

    def test_rejects_benchmark_when_model_readiness_was_not_ready(self):
        stale = report(is_emulator=False, model="ReferencePhone")
        stale["readiness_status"] = "missing detector"

        with self.assertRaisesRegex(SystemExit, "Benchmark model readiness"):
            self.module.validate_report(stale, args())


def args(
    require_physical=False,
    expected_model=None,
    enforce_target_limits=False,
):
    return argparse.Namespace(
        require_physical=require_physical,
        expected_model=expected_model,
        enforce_target_limits=enforce_target_limits,
    )


def report(
    is_emulator,
    model,
    classifier_avg_ms=100.0,
    classifier_p95_ms=150.0,
    detector_p95_ms=250.0,
    supported_abis="arm64-v8a,armeabi-v7a",
):
    return {
        "package": "com.sakshyam.agribot",
        "model_bundle_id": "agribot-model-bundle-v001",
        "readiness_status": "Ready",
        "device": {
            "manufacturer": "Test",
            "model": model,
            "supported_abis": supported_abis,
            "is_emulator": is_emulator,
        },
        "classifier": {
            "avg_ms": classifier_avg_ms,
            "p95_ms": classifier_p95_ms,
        },
        "detector": {
            "p95_ms": detector_p95_ms,
        },
    }


if __name__ == "__main__":
    unittest.main()
