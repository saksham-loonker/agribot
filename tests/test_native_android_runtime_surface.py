import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "agribot_android_app"


class NativeAndroidRuntimeSurfaceTest(unittest.TestCase):
    def test_package_json_has_no_capacitor_runtime(self):
        package_json = json.loads((APP_DIR / "package.json").read_text(encoding="utf-8"))
        dependencies = {
            **package_json.get("dependencies", {}),
            **package_json.get("devDependencies", {}),
        }
        scripts = package_json.get("scripts", {})

        self.assertFalse(any(name.startswith("@capacitor/") for name in dependencies))
        self.assertFalse(any("cap " in command or command.startswith("cap") for command in scripts.values()))

    def test_generated_capacitor_android_shims_are_absent(self):
        forbidden_paths = [
            APP_DIR / "capacitor.config.json",
            APP_DIR / "package-lock.json",
            APP_DIR / "android" / "capacitor.settings.gradle",
            APP_DIR / "android" / "app" / "capacitor.build.gradle",
        ]

        for path in forbidden_paths:
            self.assertFalse(path.exists(), f"Capacitor runtime artifact should be absent: {path}")

    def test_native_apk_assets_do_not_ship_legacy_pi_dashboard_client(self):
        assets_dir = APP_DIR / "android" / "app" / "src" / "main" / "assets"
        forbidden_paths = [
            assets_dir / "capacitor.config.json",
            assets_dir / "capacitor.plugins.json",
            assets_dir / "public",
        ]

        for path in forbidden_paths:
            self.assertFalse(path.exists(), f"Legacy Pi dashboard asset should not ship in native APK: {path}")

    def test_native_gradle_settings_do_not_include_capacitor_modules(self):
        settings = (APP_DIR / "android" / "settings.gradle").read_text(encoding="utf-8")

        self.assertNotIn("capacitor", settings.lower())
        self.assertIn("include ':app'", settings)
        self.assertIn("include ':feature-scan'", settings)

    def test_release_signing_can_use_ignored_local_properties(self):
        build_gradle = (APP_DIR / "android" / "app" / "build.gradle").read_text(encoding="utf-8")
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

        self.assertIn('rootProject.file("signing.properties")', build_gradle)
        self.assertIn("localSigningProperties.load", build_gradle)
        self.assertIn("*.jks", gitignore)
        self.assertIn("signing.properties", gitignore)

    def test_android_runtime_models_are_configured_for_lfs_and_unignored(self):
        model_paths = [
            "agribot_android_app/android/app/src/main/assets/models/classifier_fastcrop_float32.tflite",
            "agribot_android_app/android/app/src/main/assets/models/detector_nano_256_raw_float32.tflite",
        ]
        attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

        for model_path in model_paths:
            self.assertIn(model_path, attributes)
        self.assertIn("filter=lfs diff=lfs merge=lfs -text", attributes)
        self.assertIn("!agribot_android_app/android/app/src/main/assets/models/", gitignore)
        self.assertIn("!agribot_android_app/android/app/src/main/assets/models/*.tflite", gitignore)


if __name__ == "__main__":
    unittest.main()
