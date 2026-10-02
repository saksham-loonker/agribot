"""The bundled Android model contract: manifest <-> shipped .tflite files <-> golden set <-> reference maths."""
import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "agribot_android_app/android/app/src/main/assets"
GOLDEN_TEST = ROOT / "agribot_android_app/android/app/src/androidTest/assets/golden"
GOLDEN_JVM = ROOT / "agribot_android_app/android/ml/src/test/resources/golden"
LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"


class AndroidModelContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ASSETS / "model_manifest.json").read_text(encoding="utf-8"))

    def test_manifest_schema_and_labels(self):
        m = self.manifest
        self.assertEqual(m["schema"], 2)
        c = m["classifier"]
        self.assertEqual(c["labels"][c["other_index"]], "Other")
        self.assertIn("Healthy", c["labels"])
        self.assertEqual(len(c["labels"]), 9)
        self.assertGreater(c["temperature"], 0)
        d = m["detector"]
        self.assertEqual(d["output_rows"], ["x1", "y1", "x2", "y2", "score"])
        self.assertEqual(d["input"][1], d["input"][2])
        self.assertTrue(0 < d["score_threshold"] < 1 and 0 < d["nms_iou"] < 1)

    def test_model_files_match_manifest_hashes(self):
        files = [(x["file"], x["sha256"]) for x in self.manifest["classifier"]["members"]]
        files.append((self.manifest["detector"]["file"], self.manifest["detector"]["sha256"]))
        for name, sha in files:
            data = (ASSETS / "models" / name).read_bytes()
            if data.startswith(LFS_POINTER):
                self.skipTest("model files are Git LFS pointers in this checkout (run git lfs pull)")
            self.assertEqual(hashlib.sha256(data).hexdigest(), sha, name)
        self.assertTrue(self.manifest["bundle_id"].endswith(files[0][1][:8]))

    def test_golden_sets_are_identical_and_complete(self):
        expected = json.loads((GOLDEN_TEST / "golden_expected.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(expected["entries"]), 10)
        for e in expected["entries"]:
            a = (GOLDEN_TEST / e["file"]).read_bytes()
            self.assertEqual(a, (GOLDEN_JVM / e["file"]).read_bytes(), e["file"])
            self.assertNotIn(b"iCCP", a, f"{e['file']} carries a colour profile Android would apply")
            self.assertEqual(len(e["whole_image"]["member_logits"]), len(self.manifest["classifier"]["members"]))
        self.assertEqual((GOLDEN_TEST / "golden_expected.json").read_bytes(), (GOLDEN_JVM / "golden_expected.json").read_bytes())

    def test_reference_maths_is_the_shipped_version(self):
        ref = (ROOT / "model_training/v2/final/canon.py").read_text(encoding="utf-8")
        self.assertEqual(ref, (ROOT / "model_training/v2/bundle/golden/canon_reference.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
