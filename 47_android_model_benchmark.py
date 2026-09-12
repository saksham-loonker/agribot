#!/usr/bin/env python3
"""Run the native Android embedded-model benchmark and pull its JSON report."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANDROID_DIR = ROOT / "agribot_android_app" / "android"
DIST_DIR = ROOT / "dist"
REPORT_NAME = "embedded_model_benchmark.json"
PACKAGE_NAME = "com.sakshyam.agribot"
TEST_PACKAGE_NAME = "com.sakshyam.agribot.test"
BENCHMARK_CLASS = "com.sakshyam.agribot.AgribotEmbeddedModelBenchmarkTest"
MODEL_BUNDLE_ID = "agribot-model-bundle-v001"
REQUIRED_ABI = "arm64-v8a"
TARGET_CLASSIFIER_AVG_MS = 160.0
TARGET_CLASSIFIER_P95_MS = 200.0
TARGET_DETECTOR_P95_MS = 300.0


def run(command: list[str], cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command))
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def android_home() -> Path:
    value = os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT")
    if value:
        return Path(value)
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidate = Path(local_app_data) / "Android" / "Sdk"
        if candidate.exists():
            return candidate
    raise SystemExit("ANDROID_HOME is not set and the default Windows SDK path was not found.")


def adb_path() -> Path:
    exe = "adb.exe" if os.name == "nt" else "adb"
    path = android_home() / "platform-tools" / exe
    if not path.exists():
        raise SystemExit(f"adb not found at {path}")
    return path


def adb_command(adb: Path, device: str | None) -> list[str]:
    command = [str(adb)]
    if device:
        command.extend(["-s", device])
    return command


def gradlew_path() -> Path:
    exe = "gradlew.bat" if os.name == "nt" else "gradlew"
    path = ANDROID_DIR / exe
    if not path.exists():
        raise SystemExit(f"Gradle wrapper not found at {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DIST_DIR / REPORT_NAME,
        help="Where to write the pulled benchmark JSON report.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Only pull the last report from the installed debug app.",
    )
    parser.add_argument(
        "--device",
        help="ADB device serial to target when more than one device is connected.",
    )
    parser.add_argument(
        "--require-physical",
        action="store_true",
        help="Fail if the benchmark report came from an emulator.",
    )
    parser.add_argument(
        "--expected-model",
        help="Fail unless the benchmark report device model matches this exact value.",
    )
    parser.add_argument(
        "--enforce-target-limits",
        action="store_true",
        help="Fail unless the report meets the Side Scan and Front Overview latency gates.",
    )
    return parser.parse_args()


def device_supports_required_abi(device: dict) -> bool:
    supported_abis = device.get("supported_abis")
    if isinstance(supported_abis, list):
        values = [str(value).strip() for value in supported_abis]
    else:
        values = [value.strip() for value in str(supported_abis or "").split(",")]
    return REQUIRED_ABI in values


def validate_report(parsed: dict, args: argparse.Namespace) -> None:
    package = str(parsed.get("package") or "")
    model_bundle_id = str(parsed.get("model_bundle_id") or "")
    readiness_status = str(parsed.get("readiness_status") or "")
    device = parsed.get("device") or {}
    model = str(device.get("model") or "")
    manufacturer = str(device.get("manufacturer") or "")
    is_emulator = bool(device.get("is_emulator"))

    if package != PACKAGE_NAME:
        raise SystemExit(f"Benchmark package {package!r} does not match {PACKAGE_NAME!r}")
    if model_bundle_id != MODEL_BUNDLE_ID:
        raise SystemExit(f"Benchmark model_bundle_id {model_bundle_id!r} does not match {MODEL_BUNDLE_ID!r}")
    if not readiness_status.lower().startswith("ready"):
        raise SystemExit(f"Benchmark model readiness was not Ready: {readiness_status!r}")
    if args.require_physical and is_emulator:
        raise SystemExit(
            "Physical target phone required, but benchmark report is from emulator "
            f"{manufacturer} {model}".strip()
        )
    if not is_emulator and not device_supports_required_abi(device):
        raise SystemExit(f"Benchmark physical phone does not advertise required {REQUIRED_ABI} ABI")
    if args.expected_model and model != args.expected_model:
        raise SystemExit(f"Expected benchmark device model {args.expected_model!r}, got {model!r}")
    if not args.enforce_target_limits:
        return

    classifier = parsed.get("classifier") or {}
    detector = parsed.get("detector") or {}
    failures = []
    if float(classifier.get("avg_ms", float("inf"))) > TARGET_CLASSIFIER_AVG_MS:
        failures.append(f"classifier avg_ms > {TARGET_CLASSIFIER_AVG_MS}")
    if float(classifier.get("p95_ms", float("inf"))) > TARGET_CLASSIFIER_P95_MS:
        failures.append(f"classifier p95_ms > {TARGET_CLASSIFIER_P95_MS}")
    if float(detector.get("p95_ms", float("inf"))) > TARGET_DETECTOR_P95_MS:
        failures.append(f"detector p95_ms > {TARGET_DETECTOR_P95_MS}")
    if failures:
        raise SystemExit("Benchmark latency gates failed: " + "; ".join(failures))


def main() -> int:
    args = parse_args()
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ANDROID_HOME", str(android_home()))

    if not args.skip_test:
        run(
            [
                str(gradlew_path()),
                ":app:installDebug",
                ":app:installDebugAndroidTest",
                "--stacktrace",
                "--no-daemon",
                "--max-workers=1",
            ],
            cwd=ANDROID_DIR,
        )

    adb = adb_path()
    adb_base = adb_command(adb, args.device)
    if not args.skip_test:
        instrumentation = run(
            adb_base
            + [
                "shell",
                "am",
                "instrument",
                "-w",
                "-e",
                "class",
                BENCHMARK_CLASS,
                f"{TEST_PACKAGE_NAME}/androidx.test.runner.AndroidJUnitRunner",
            ],
            capture=True,
        )
        output = instrumentation.stdout or ""
        print(output)
        if "FAILURES!!!" in output or "INSTRUMENTATION_CODE: -1" in output or "OK (" not in output:
            raise SystemExit(f"Benchmark instrumentation did not pass:\n{output}")

    remote_path = f"files/benchmarks/{REPORT_NAME}"
    result = run(
        adb_base + ["exec-out", "run-as", PACKAGE_NAME, "cat", remote_path],
        capture=True,
    )
    payload = result.stdout or ""
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as error:
        raise SystemExit(f"Benchmark report was not valid JSON: {error}\n{payload[:500]}") from error

    validate_report(parsed, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(parsed, indent=2) + "\n", encoding="utf-8")
    print(f"Benchmark report: {args.output}")
    print(json.dumps(parsed, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
