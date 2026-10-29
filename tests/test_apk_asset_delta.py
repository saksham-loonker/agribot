import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

path = Path(__file__).resolve().parents[1] / "model_training/v3/apk_asset_delta.py"
spec = importlib.util.spec_from_file_location("apk_asset_delta", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ApkAssetDeltaTest(unittest.TestCase):
    def fixture(self, root):
        assets = root / "assets"
        assets.mkdir()
        payloads = {
            "leaf.tflite": bytes(range(256)) * 1024,
            "detector.tflite": b"detector" * 1000,
        }
        contracts = []
        for name, value in payloads.items():
            (assets / name).write_bytes(value)
            contracts.append(dict(file=name, sha256=hashlib.sha256(value).hexdigest()))
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                dict(classifier=dict(members=[contracts[0]]), detector=contracts[1])
            )
        )
        apk = root / "original.apk"
        with zipfile.ZipFile(apk, "w") as archive:
            archive.writestr(
                "classes.dex",
                b"dex\x00other content",
                compress_type=zipfile.ZIP_DEFLATED,
            )
            for name, value in payloads.items():
                archive.writestr("assets/models/" + name, value)
            archive.writestr("resources.arsc", b"last entry")
        return apk, manifest, assets

    def test_reconstructs_complete_zip_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            apk, manifest, assets = self.fixture(root)
            recipe = root / "recipe"
            result = module.prepare(apk, manifest, recipe)
            self.assertLess(result["transfer_bytes"], 1000)
            target = root / "restored.apk"
            module.reconstruct(recipe, assets, target)
            self.assertEqual(apk.read_bytes(), target.read_bytes())

    def test_wrong_local_model_cannot_replace_existing_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            apk, manifest, assets = self.fixture(root)
            recipe = root / "recipe"
            module.prepare(apk, manifest, recipe)
            target = root / "restored.apk"
            target.write_bytes(b"existing output")
            (assets / "leaf.tflite").write_bytes(b"wrong model")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                module.reconstruct(recipe, assets, target)
            self.assertEqual(b"existing output", target.read_bytes())

    def test_corrupt_literal_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            apk, manifest, assets = self.fixture(root)
            recipe = root / "recipe"
            module.prepare(apk, manifest, recipe)
            (recipe / "literal.gz").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Invalid transfer"):
                module.reconstruct(recipe, assets, root / "restored.apk")

    def test_stale_research_model_in_apk_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            apk, manifest, _ = self.fixture(root)
            with zipfile.ZipFile(apk, "a") as archive:
                archive.writestr("assets/models/rejected_context.tflite", b"stale model")
            with self.assertRaisesRegex(ValueError, "undeclared"):
                module.prepare(apk, manifest, root / "recipe")


if __name__ == "__main__":
    unittest.main()
