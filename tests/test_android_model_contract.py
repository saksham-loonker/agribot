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
        cls.manifest = json.loads(
            (ASSETS / "model_manifest.json").read_text(encoding="utf-8")
        )

    def test_manifest_schema_and_labels(self):
        m = self.manifest
        self.assertIn(m["schema"], (2, 3, 4, 5))
        c = m["classifier"]
        self.assertEqual(c["labels"][c["other_index"]], "Other")
        self.assertIn("Healthy", c["labels"])
        self.assertEqual(len(c["labels"]), 9)
        self.assertEqual(
            m["schema"] == 4, bool(c.get("reference_disease_priority", False))
        )
        if m["schema"] in (3, 4):
            self.assertEqual(len(c["members"]), 1)
            self.assertEqual(c["members"][0]["output"], [1, 18])
            confirmation = c["view_confirmation"]
            self.assertIn(
                confirmation["policy"],
                ("changed_positive_full_roi", "changed_decision_full_roi"),
            )
            self.assertEqual(
                confirmation,
                dict(
                    policy=confirmation["policy"],
                    reference_logit_offset=9,
                    padding_value=114,
                ),
            )
        else:
            self.assertNotIn("view_confirmation", c)
        if m["schema"] == 5:
            self.assertEqual(c["members"][0]["output"], [1, 12])
            self.assertEqual(
                c["auxiliary_outputs"],
                ["compact_positive_H", "compact_negative_H", "adapted_dino_H"],
            )
            recovery = m["primary_recovery"]
            self.assertEqual(recovery["scope"], "primary_visible_view")
            self.assertEqual(recovery["policy"], "phase69_whole_local")
            self.assertEqual(recovery["minimum_frames"], 7)
            self.assertEqual(recovery["maximum_leaves"], 6)
            self.assertEqual(recovery["model"]["output"], [1, 4])
            self.assertEqual(
                recovery["model"]["outputs"],
                ["positive_H", "negative_H", "tomato_genus", "natural_context"],
            )
        self.assertGreater(c["temperature"], 0)
        d = m["detector"]
        self.assertEqual(d["output_rows"], ["x1", "y1", "x2", "y2", "score"])
        self.assertEqual(d["input"][1], d["input"][2])
        self.assertTrue(0 < d["score_threshold"] < 1 and 0 < d["nms_iou"] < 1)

    def test_model_files_match_manifest_hashes(self):
        files = [
            (x["file"], x["sha256"]) for x in self.manifest["classifier"]["members"]
        ]
        files.append(
            (self.manifest["detector"]["file"], self.manifest["detector"]["sha256"])
        )
        if self.manifest["schema"] == 5:
            context = self.manifest["primary_recovery"]["model"]
            files.append((context["file"], context["sha256"]))
        for name, sha in files:
            with (ASSETS / "models" / name).open("rb") as stream:
                if stream.read(len(LFS_POINTER)) == LFS_POINTER:
                    self.skipTest(
                        "model files are Git LFS pointers in this checkout (run git lfs pull)"
                    )
                stream.seek(0)
                self.assertEqual(
                    hashlib.file_digest(stream, "sha256").hexdigest(), sha, name
                )
        self.assertIn(files[0][1][:8], self.manifest["bundle_id"])

    def test_golden_sets_are_identical_and_complete(self):
        expected = json.loads(
            (GOLDEN_TEST / "golden_expected.json").read_text(encoding="utf-8")
        )
        self.assertGreaterEqual(len(expected["entries"]), 10)
        for e in expected["entries"]:
            a = (GOLDEN_TEST / e["file"]).read_bytes()
            self.assertEqual(a, (GOLDEN_JVM / e["file"]).read_bytes(), e["file"])
            self.assertNotIn(
                b"iCCP", a, f"{e['file']} carries a colour profile Android would apply"
            )
            self.assertEqual(
                len(e["whole_image"]["member_logits"]),
                len(self.manifest["classifier"]["members"]),
            )
        self.assertEqual(
            (GOLDEN_TEST / "golden_expected.json").read_bytes(),
            (GOLDEN_JVM / "golden_expected.json").read_bytes(),
        )

    def test_reference_maths_is_the_shipped_version(self):
        ref = (ROOT / "model_training/v2/final/canon.py").read_text(encoding="utf-8")
        self.assertEqual(
            ref,
            (ROOT / "model_training/v2/bundle/golden/canon_reference.py").read_text(
                encoding="utf-8"
            ),
        )


if __name__ == "__main__":
    unittest.main()
