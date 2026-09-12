#!/usr/bin/env python3
"""Build the native offline Agribot Android APK."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ANDROID = ROOT / "agribot_android_app" / "android"
LOCAL_JDK = ROOT / ".toolchains" / "jdk"
LOCAL_ANDROID_SDK = ROOT / ".toolchains" / "android-sdk"
ANDROID_BUILD_TOOLS_VERSION = "35.0.0"
EXPECTED_PACKAGE_NAME = "com.sakshyam.agribot"
FORBIDDEN_APK_ASSET_PREFIXES = (
    "assets/public/",
)
FORBIDDEN_APK_ASSETS = (
    "assets/capacitor.config.json",
    "assets/capacitor.plugins.json",
)


def gradlew() -> str:
    return str(ANDROID / "gradlew.bat") if os.name == "nt" else "./gradlew"


def tool_name(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def script_name(name: str) -> str:
    return f"{name}.bat" if os.name == "nt" else name


def valid_android_sdk(path: Path) -> bool:
    return (
        (path / "platform-tools" / tool_name("adb")).exists()
        and (path / "build-tools" / ANDROID_BUILD_TOOLS_VERSION / tool_name("aapt")).exists()
    )


def android_sdk_candidates() -> list[Path]:
    candidates: list[Path] = []
    for env_name in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value))
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "Android" / "Sdk")
    candidates.append(LOCAL_ANDROID_SDK)
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def choose_android_sdk() -> Path | None:
    for candidate in android_sdk_candidates():
        if valid_android_sdk(candidate):
            return candidate
    return None


def android_build_tool(name: str) -> Path | None:
    android_sdk = choose_android_sdk()
    if not android_sdk:
        return None
    suffix = script_name(name) if name == "apksigner" else tool_name(name)
    candidate = android_sdk / "build-tools" / ANDROID_BUILD_TOOLS_VERSION / suffix
    return candidate if candidate.exists() else None


def use_local_toolchain() -> None:
    path_parts: list[str] = []
    if (LOCAL_JDK / "bin" / tool_name("java")).exists():
        os.environ.setdefault("JAVA_HOME", str(LOCAL_JDK))
        path_parts.append(str(LOCAL_JDK / "bin"))
    android_sdk = choose_android_sdk()
    if android_sdk:
        os.environ["ANDROID_HOME"] = str(android_sdk)
        os.environ["ANDROID_SDK_ROOT"] = str(android_sdk)
        cmdline_tools = android_sdk / "cmdline-tools" / "latest" / "bin"
        if (cmdline_tools / script_name("sdkmanager")).exists():
            path_parts.append(str(cmdline_tools))
        path_parts.append(str(android_sdk / "platform-tools"))
    if path_parts:
        os.environ["PATH"] = os.pathsep.join(path_parts + [os.environ.get("PATH", "")])


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def run_capture(cmd: list[str], cwd: Path | None = None) -> str:
    print("+", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout


def local_signing_properties() -> dict[str, str]:
    path = ANDROID / "signing.properties"
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def release_signing_configured() -> bool:
    required = [
        "AGRIBOT_RELEASE_STORE_FILE",
        "AGRIBOT_RELEASE_STORE_PASSWORD",
        "AGRIBOT_RELEASE_KEY_ALIAS",
        "AGRIBOT_RELEASE_KEY_PASSWORD",
    ]
    local_values = local_signing_properties()
    return all(os.environ.get(name) or local_values.get(name) for name in required)


def find_apk(variant: str, signed_release: bool = False) -> Path:
    output_dir = ANDROID / "app" / "build" / "outputs" / "apk" / variant
    if variant == "release":
        # AGP can produce app-release.apk signed with the configured release
        # key, or app-release-unsigned.apk when signing is intentionally absent.
        # Prefer the requested form but fall back to the other canonical output
        # so the verification stage reports the actual artifact instead of
        # failing with a misleading "APK not found" error.
        names = (
            ("app-release.apk", "app-release-unsigned.apk")
            if signed_release
            else ("app-release-unsigned.apk", "app-release.apk")
        )
        for name in names:
            candidate = output_dir / name
            if candidate.exists():
                return candidate
        return output_dir / names[0]

    exact = output_dir / f"app-{variant}.apk"
    if exact.exists():
        return exact
    matches = sorted(output_dir.glob("*.apk"))
    if matches:
        return matches[0]
    return exact


def verify_apk_metadata(apk: Path) -> None:
    aapt = android_build_tool("aapt")
    if not aapt:
        print("aapt was not found; skipping APK metadata verification.")
        return

    badging = run_capture([str(aapt), "dump", "badging", str(apk)])
    permissions = run_capture([str(aapt), "dump", "permissions", str(apk)])
    package_marker = f"package: name='{EXPECTED_PACKAGE_NAME}'"
    missing = [] if package_marker in badging else [package_marker]
    missing.extend(item for item in [
        "sdkVersion:'26'",
        "targetSdkVersion:'35'",
        "native-code: 'arm64-v8a'",
    ] if item not in badging)
    if missing:
        raise SystemExit(f"[ERROR] APK metadata check failed for {apk}: missing {missing}")
    if "android.permission.CAMERA" not in permissions:
        raise SystemExit(f"[ERROR] APK metadata check failed for {apk}: CAMERA permission missing")
    if "android.permission.INTERNET" in permissions:
        raise SystemExit(f"[ERROR] APK metadata check failed for {apk}: INTERNET permission present")
    print(
        "APK metadata verified: package com.sakshyam.agribot, "
        "minSdk 26, targetSdk 35, arm64-v8a, CAMERA only."
    )


def verify_no_legacy_apk_assets(apk: Path) -> None:
    with zipfile.ZipFile(apk) as archive:
        names = archive.namelist()
    forbidden = [
        name
        for name in names
        if name in FORBIDDEN_APK_ASSETS
        or any(name.startswith(prefix) for prefix in FORBIDDEN_APK_ASSET_PREFIXES)
    ]
    if forbidden:
        raise SystemExit(f"[ERROR] APK contains legacy Pi dashboard/Capacitor assets: {forbidden}")
    print("APK asset surface verified: no legacy Pi dashboard or Capacitor assets.")


def verify_apk_signature(apk: Path) -> bool:
    apksigner = android_build_tool("apksigner")
    if not apksigner:
        print("apksigner was not found; skipping APK signature verification.")
        return False
    try:
        output = run_capture([str(apksigner), "verify", "--verbose", str(apk)])
    except subprocess.CalledProcessError as error:
        print(error.stdout or "")
        return False
    print(output.strip())
    modern_scheme_verified = any(
        marker in output
        for marker in (
            "Verified using v2 scheme (APK Signature Scheme v2): true",
            "Verified using v3 scheme (APK Signature Scheme v3): true",
            "Verified using v3.1 scheme (APK Signature Scheme v3.1): true",
        )
    )
    return "Verifies" in output and "DOES NOT VERIFY" not in output and modern_scheme_verified


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["debug", "release"], default="debug")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument(
        "--require-signed",
        action="store_true",
        help="Fail release builds unless the copied APK verifies with apksigner.",
    )
    args = parser.parse_args()

    use_local_toolchain()
    task_variant = args.variant.capitalize()
    tasks = [f":app:assemble{task_variant}"]
    if not args.skip_tests:
        tasks = [
            ":domain:test",
            ":camera:testDebugUnitTest",
            ":ml:testDebugUnitTest",
            ":data:testDebugUnitTest",
            ":feature-scan:testDebugUnitTest",
            ":app:testDebugUnitTest",
            *tasks,
        ]
    run([gradlew(), *tasks, "--no-daemon", "--max-workers=1"], ANDROID)

    built_apk = find_apk(args.variant, signed_release=args.variant == "release" and release_signing_configured())
    if not built_apk.exists():
        raise SystemExit(f"[ERROR] Gradle finished but APK was not found: {built_apk}")

    destination = ROOT / "dist" / f"agribot-field-app-{args.variant}.apk"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built_apk, destination)
    print(f"APK built: {built_apk}")
    print(f"APK copied: {destination}")
    verify_apk_metadata(destination)
    verify_no_legacy_apk_assets(destination)

    should_verify_signature = args.variant == "debug" or args.require_signed or bool(os.environ.get("AGRIBOT_RELEASE_STORE_FILE"))
    signature_ok = verify_apk_signature(destination) if should_verify_signature else False
    if args.variant == "release" and args.require_signed and not signature_ok:
        raise SystemExit("[ERROR] Release APK did not verify; provide AGRIBOT_RELEASE_* signing values.")
    if args.variant == "release" and not signature_ok:
        print("Release signing env vars were not set or did not produce a signed APK; do not use this APK for field distribution.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
