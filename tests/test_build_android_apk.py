import importlib.util
import io
import subprocess
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "45_build_android_apk.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_android_apk", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BuildAndroidApkVerifierTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.apk = Path("dist/agribot-field-app-debug.apk")

    def test_metadata_accepts_expected_native_apk_surface(self):
        with patched_aapt(self.module, badging(), permissions()):
            with quiet_stdout():
                self.module.verify_apk_metadata(self.apk)

    def test_metadata_rejects_wrong_package_name(self):
        with patched_aapt(self.module, badging(package_name="com.example.oldwrapper"), permissions()):
            with self.assertRaisesRegex(SystemExit, "package: name='com.sakshyam.agribot'"):
                self.module.verify_apk_metadata(self.apk)

    def test_metadata_rejects_internet_permission(self):
        with patched_aapt(self.module, badging(), permissions(include_internet=True)):
            with self.assertRaisesRegex(SystemExit, "INTERNET permission present"):
                self.module.verify_apk_metadata(self.apk)

    def test_signature_accepts_v2_or_newer_signature(self):
        with patched_apksigner(self.module, signature_output(v2=True)):
            with quiet_stdout():
                self.assertTrue(self.module.verify_apk_signature(self.apk))

    def test_signature_rejects_v1_only_signature(self):
        with patched_apksigner(self.module, signature_output(v1=True)):
            with quiet_stdout():
                self.assertFalse(self.module.verify_apk_signature(self.apk))

    def test_signature_rejects_failed_verification(self):
        error = subprocess.CalledProcessError(
            returncode=1,
            cmd=["apksigner"],
            output="DOES NOT VERIFY\nERROR: Missing META-INF/MANIFEST.MF\n",
        )
        with patch.object(self.module, "android_build_tool", return_value=Path("apksigner")), \
                patch.object(self.module, "run_capture", side_effect=error):
            with quiet_stdout():
                self.assertFalse(self.module.verify_apk_signature(self.apk))

    def test_asset_verifier_accepts_native_model_assets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            apk = Path(temp_dir) / "agribot.apk"
            with zipfile.ZipFile(apk, "w") as archive:
                archive.writestr("assets/model_manifest.json", "{}")
                archive.writestr("assets/labels/labels.json", "[]")
                archive.writestr("assets/models/classifier_fastcrop_float32.tflite", "")

            with quiet_stdout():
                self.module.verify_no_legacy_apk_assets(apk)

    def test_asset_verifier_rejects_legacy_pi_dashboard_assets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            apk = Path(temp_dir) / "agribot.apk"
            with zipfile.ZipFile(apk, "w") as archive:
                archive.writestr("assets/public/index.html", "")
                archive.writestr("assets/capacitor.config.json", "{}")

            with self.assertRaisesRegex(SystemExit, "legacy Pi dashboard"):
                self.module.verify_no_legacy_apk_assets(apk)

    def test_unsigned_release_selection_ignores_stale_signed_apk(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            android = Path(temp_dir) / "agribot_android_app" / "android"
            release_dir = android / "app" / "build" / "outputs" / "apk" / "release"
            release_dir.mkdir(parents=True)
            stale_signed = release_dir / "app-release.apk"
            fresh_unsigned = release_dir / "app-release-unsigned.apk"
            stale_signed.write_text("stale", encoding="utf-8")
            fresh_unsigned.write_text("fresh", encoding="utf-8")

            with patch.object(self.module, "ANDROID", android):
                self.assertEqual(fresh_unsigned, self.module.find_apk("release", signed_release=False))
                self.assertEqual(stale_signed, self.module.find_apk("release", signed_release=True))

    def test_release_selection_does_not_fallback_to_unexpected_apk(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            android = Path(temp_dir) / "agribot_android_app" / "android"
            release_dir = android / "app" / "build" / "outputs" / "apk" / "release"
            release_dir.mkdir(parents=True)
            (release_dir / "old-random-release.apk").write_text("old", encoding="utf-8")

            with patch.object(self.module, "ANDROID", android):
                self.assertEqual(release_dir / "app-release-unsigned.apk", self.module.find_apk("release"))

    def test_release_signing_configured_accepts_local_signing_properties(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            android = Path(temp_dir) / "agribot_android_app" / "android"
            android.mkdir(parents=True)
            (android / "signing.properties").write_text(
                "\n".join(
                    [
                        "AGRIBOT_RELEASE_STORE_FILE=release-signing/agribot-field-pilot.jks",
                        "AGRIBOT_RELEASE_STORE_PASSWORD=local",
                        "AGRIBOT_RELEASE_KEY_ALIAS=agribot_field_pilot",
                        "AGRIBOT_RELEASE_KEY_PASSWORD=local",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.object(self.module, "ANDROID", android), \
                    patch.dict("os.environ", {}, clear=True):
                self.assertTrue(self.module.release_signing_configured())


def patched_aapt(module, badging_output, permissions_output):
    def fake_capture(command):
        if command[2] == "badging":
            return badging_output
        if command[2] == "permissions":
            return permissions_output
        raise AssertionError(f"Unexpected aapt command: {command}")

    return patch.multiple(
        module,
        android_build_tool=lambda name: Path(name),
        run_capture=fake_capture,
    )


def quiet_stdout():
    return redirect_stdout(io.StringIO())


def patched_apksigner(module, output):
    return patch.multiple(
        module,
        android_build_tool=lambda name: Path(name),
        run_capture=lambda command: output,
    )


def badging(package_name="com.sakshyam.agribot"):
    return "\n".join(
        [
            f"package: name='{package_name}' versionCode='1' versionName='1.0'",
            "sdkVersion:'26'",
            "targetSdkVersion:'35'",
            "native-code: 'arm64-v8a'",
        ]
    )


def permissions(include_internet=False):
    lines = ["uses-permission: name='android.permission.CAMERA'"]
    if include_internet:
        lines.append("uses-permission: name='android.permission.INTERNET'")
    return "\n".join(lines)


def signature_output(v1=False, v2=False, v3=False):
    return "\n".join(
        [
            "Verifies",
            f"Verified using v1 scheme (JAR signing): {str(v1).lower()}",
            f"Verified using v2 scheme (APK Signature Scheme v2): {str(v2).lower()}",
            f"Verified using v3 scheme (APK Signature Scheme v3): {str(v3).lower()}",
            "Number of signers: 1",
        ]
    )


if __name__ == "__main__":
    unittest.main()
