#!/usr/bin/env python3
"""Summarize native Agribot Android build and field-readiness evidence."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "agribot_android_app" / "android" / "app" / "src" / "main" / "assets"
MANIFEST_PATH = ASSETS_DIR / "model_manifest.json"
LABELS_PATH = ASSETS_DIR / "labels" / "labels.json"
DIST_DIR = ROOT / "dist"
DEBUG_APK = DIST_DIR / "agribot-field-app-debug.apk"
RELEASE_APK = DIST_DIR / "agribot-field-app-release.apk"
BENCHMARK_REPORT = DIST_DIR / "embedded_model_benchmark.json"
EXPECTED_PACKAGE_NAME = "com.sakshyam.agribot"
EXPECTED_MODEL_BUNDLE_ID = "agribot-model-bundle-v001"
REQUIRED_ABI = "arm64-v8a"

EXPECTED_LABELS = [
    "Early_blight",
    "Healthy",
    "Late_blight",
    "Leaf Miner",
    "Magnesium Deficiency",
    "Nitrogen Deficiency",
    "Pottassium Deficiency",
    "Spotted Wilt Virus",
]
EXPECTED_CLASSIFIER_SHA = "8478c47e795bbf2f1beffe2f21b20d04aa5410148560d7f1912bfa8fbf2158f8"
EXPECTED_DETECTOR_SHA = "83e7521447c7de4730dfea79eb830d286a4ecf44aa3f5cc8c6080947969f6910"
EXPECTED_LABELS_SHA = "c9dd8ac5138e41dbb3cd9a7130542c70e463277318941d6a17acc5841c03992b"
TARGET_CLASSIFIER_AVG_MS = 160.0
TARGET_CLASSIFIER_P95_MS = 200.0
TARGET_DETECTOR_P95_MS = 300.0


def unique_failures(failures: list[str]) -> list[str]:
    unique: list[str] = []
    for failure in failures:
        if failure not in unique:
            unique.append(failure)
    return unique


def load_build_module() -> Any:
    module_path = ROOT / "45_build_android_apk.py"
    spec = importlib.util.spec_from_file_location("build_android_apk", module_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    spec.loader.exec_module(module)
    return module


def load_field_export_module() -> Any:
    module_path = ROOT / "49_validate_android_field_export.py"
    spec = importlib.util.spec_from_file_location("android_field_export_validator", module_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_model_bundle(manifest: dict[str, Any], labels: list[str]) -> dict[str, Any]:
    classifier = manifest.get("models", {}).get("classifier", {})
    detector = manifest.get("models", {}).get("detector", {})
    compatibility = manifest.get("compatibility", {})
    failures: list[str] = []

    expected_values = {
        "classifier file": (classifier.get("file"), "models/classifier_fastcrop_float32.tflite"),
        "classifier sha256": (classifier.get("sha256"), EXPECTED_CLASSIFIER_SHA),
        "classifier confidence_threshold": (classifier.get("confidence_threshold"), 0.62),
        "classifier high_confidence_threshold": (classifier.get("high_confidence_threshold"), 0.90),
        "detector file": (detector.get("file"), "models/detector_nano_256_raw_float32.tflite"),
        "detector sha256": (detector.get("sha256"), EXPECTED_DETECTOR_SHA),
        "labels sha256": (manifest.get("labels_sha256"), EXPECTED_LABELS_SHA),
        "min Android SDK": (compatibility.get("min_android_sdk"), 26),
        "CPU default": (compatibility.get("cpu_default"), True),
        "network requirement": (compatibility.get("requires_network"), False),
    }
    for name, (actual, expected) in expected_values.items():
        if actual != expected:
            failures.append(f"{name} expected {expected!r}, got {actual!r}")
    if labels != EXPECTED_LABELS:
        failures.append("labels are not in Pi-compatible order")

    return {"ok": not failures, "failures": failures}


def evaluate_apk(path: Path, require_signature: bool) -> dict[str, Any]:
    failures: list[str] = []
    signed = False
    if not path.exists():
        return {"ok": False, "signed": False, "path": str(path), "failures": [f"APK not found: {path}"]}

    build_module = load_build_module()
    try:
        build_module.use_local_toolchain()
        build_module.verify_apk_metadata(path)
        build_module.verify_no_legacy_apk_assets(path)
    except SystemExit as error:
        failures.append(str(error))

    signed = bool(build_module.verify_apk_signature(path))
    if require_signature and not signed:
        failures.append("release APK is not signed with v2/v3 scheme")

    return {
        "ok": not failures,
        "signed": signed,
        "path": str(path),
        "bytes": path.stat().st_size,
        "failures": failures,
    }


def device_supports_required_abi(device: dict[str, Any]) -> bool:
    supported_abis = device.get("supported_abis")
    if isinstance(supported_abis, list):
        values = [str(value).strip() for value in supported_abis]
    else:
        values = [value.strip() for value in str(supported_abis or "").split(",")]
    return REQUIRED_ABI in values


def evaluate_benchmark(report: dict[str, Any] | None, report_path: Path = BENCHMARK_REPORT) -> dict[str, Any]:
    if report is None:
        return {
            "ok": False,
            "path": str(report_path),
            "failures": [f"benchmark report not found: {report_path}"],
        }

    device = report.get("device") or {}
    classifier = report.get("classifier") or {}
    detector = report.get("detector") or {}
    failures: list[str] = []
    if str(report.get("package") or "") != EXPECTED_PACKAGE_NAME:
        failures.append(f"benchmark package does not match {EXPECTED_PACKAGE_NAME}")
    if str(report.get("model_bundle_id") or "") != EXPECTED_MODEL_BUNDLE_ID:
        failures.append(f"benchmark model bundle does not match {EXPECTED_MODEL_BUNDLE_ID}")
    if not str(report.get("readiness_status") or "").lower().startswith("ready"):
        failures.append("benchmark model readiness was not Ready")
    is_emulator = bool(device.get("is_emulator"))
    if is_emulator:
        failures.append("benchmark is from an emulator, not a physical phone")
    else:
        if not device_supports_required_abi(device):
            failures.append(f"benchmark physical phone does not advertise {REQUIRED_ABI} ABI")
        if float(classifier.get("avg_ms", float("inf"))) > TARGET_CLASSIFIER_AVG_MS:
            failures.append(f"classifier avg_ms exceeds {TARGET_CLASSIFIER_AVG_MS}")
        if float(classifier.get("p95_ms", float("inf"))) > TARGET_CLASSIFIER_P95_MS:
            failures.append(f"classifier p95_ms exceeds {TARGET_CLASSIFIER_P95_MS}")
        if float(detector.get("p95_ms", float("inf"))) > TARGET_DETECTOR_P95_MS:
            failures.append(f"detector p95_ms exceeds {TARGET_DETECTOR_P95_MS}")
    return {
        "ok": not failures,
        "path": str(report_path),
        "device": device,
        "classifier": classifier,
        "detector": detector,
        "failures": failures,
    }


def evaluate_field_export(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"ok": False, "path": None, "failures": ["field export bundle not provided"]}
    validator = load_field_export_module()
    return validator.validate_export_bundle(path, require_real_field=True)


def evaluate_readiness(
    model_bundle: dict[str, Any],
    debug_apk: dict[str, Any],
    release_apk: dict[str, Any],
    benchmark: dict[str, Any],
    field_export: dict[str, Any] | None = None,
) -> dict[str, Any]:
    benchmark_check = benchmark if "failures" in benchmark else evaluate_benchmark(benchmark)
    field_export_check = field_export if field_export is not None else evaluate_field_export(None)
    release_failures = list(release_apk.get("failures", []))
    if not release_apk.get("signed", False):
        release_failures.append("release APK is not signed with v2/v3 scheme")
    development_blockers = unique_failures([
        *model_bundle.get("failures", []),
        *debug_apk.get("failures", []),
    ])
    field_blockers = unique_failures([
        *development_blockers,
        *release_failures,
        *benchmark_check.get("failures", []),
        *field_export_check.get("failures", []),
    ])
    return {
        "development_ready": not development_blockers,
        "field_ready": not field_blockers,
        "development_blockers": development_blockers,
        "field_blockers": field_blockers,
        "checks": {
            "model_bundle": model_bundle,
            "debug_apk": debug_apk,
            "release_apk": release_apk,
            "benchmark": benchmark_check,
            "field_export": field_export_check,
        },
    }


def build_report(
    benchmark_report_path: Path = BENCHMARK_REPORT,
    field_export_bundle: Path | None = None,
) -> dict[str, Any]:
    manifest = load_json(MANIFEST_PATH)
    labels = load_json(LABELS_PATH)
    benchmark_report = load_json(benchmark_report_path) if benchmark_report_path.exists() else None
    model_bundle = evaluate_model_bundle(manifest, labels)
    debug_apk = evaluate_apk(DEBUG_APK, require_signature=False)
    release_apk = evaluate_apk(RELEASE_APK, require_signature=True)
    benchmark = evaluate_benchmark(benchmark_report, benchmark_report_path)
    field_export = evaluate_field_export(field_export_bundle)
    return evaluate_readiness(model_bundle, debug_apk, release_apk, benchmark, field_export)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DIST_DIR / "android_release_readiness.json")
    parser.add_argument(
        "--benchmark-report",
        type=Path,
        default=BENCHMARK_REPORT,
        help="Benchmark JSON to use for field readiness, usually a physical-phone report.",
    )
    parser.add_argument(
        "--field-export-bundle",
        type=Path,
        help="Real Android field export ZIP to validate for field readiness.",
    )
    parser.add_argument("--require-development-ready", action="store_true")
    parser.add_argument("--require-field-ready", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    report = build_report(args.benchmark_report, args.field_export_bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    print(f"Readiness report: {args.output}")
    if args.require_development_ready and not report["development_ready"]:
        raise SystemExit("Android build is not development-ready.")
    if args.require_field_ready and not report["field_ready"]:
        raise SystemExit("Android build is not field-ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
