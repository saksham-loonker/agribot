"""A cache must never cross changed weights, preprocessing or source inputs."""

from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_training/v3"))
from feature_cache_reuse import digest, make_plan, seed_features


class FeatureReuseTest(unittest.TestCase):
    def fixture(self, root):
        repo = root / "repo"
        files = {
            "model_training/v2/final/canon.py": "CANON = 1\n",
            "tests/model_evaluation/canonical_runtime.py": "RUNTIME = 1\n",
            "model_training/v3/app_eval.py": "def classifier_letterbox_input(x):\n    return x\n",
            "model_training/v3/permissive_encoder.py": "class CanonicalPermissiveEncoder:\n    def forward(self, x):\n        return x\n",
        }
        for name, content in files.items():
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        source, target = root / "source", root / "target"
        for directory in [
            source / "cnn_inputs",
            target / "cnn_inputs",
            source / "siglip2_feature_heads",
            target / "siglip2_feature_heads",
        ]:
            directory.mkdir(parents=True)
        old_rows = [
            dict(path="first", set="training"),
            dict(path="second", set="training"),
        ]
        (source / "cnn_inputs/rows.json").write_text(json.dumps(old_rows))
        (target / "cnn_inputs/rows.json").write_text(
            json.dumps(old_rows + [dict(path="third", set="training")])
        )
        pixels = source / "pixels.npy"
        np.save(pixels, np.zeros((2, 1, 1, 3), np.float32))
        added = target / "pixels.npy"
        np.save(added, np.ones((1, 1, 1, 3), np.float32))
        (source / "cnn_inputs/pixels.json").write_text(
            json.dumps(dict(center=[str(pixels)]))
        )
        (target / "cnn_inputs/pixels.json").write_text(
            json.dumps(dict(center=[str(pixels), str(added)]))
        )
        old = source / "siglip2_feature_heads"
        out = target / "siglip2_feature_heads"
        for directory in [old, out]:
            (directory / "encoder.pt").write_bytes(b"same encoder weights")
        np.save(old / "features.npy", np.array([[1, 2, 3], [4, 5, 6]], np.float32))
        identity = dict(
            encoder_sha256=digest(old / "encoder.pt"),
            rows_sha256=digest(source / "cnn_inputs/rows.json"),
            ids_sha256=hashlib.sha256(np.array([0, 1], np.int64).tobytes()).hexdigest(),
        )
        (old / "feature_identity.json").write_text(json.dumps(identity))
        for name in ["restored_encoder_parity.json", "input_contract_parity.json"]:
            (old / name).write_text(json.dumps(dict(passed=True)))
        make_plan(source, target, "siglip2", repo, repo)
        return source, target, out, repo

    def test_matching_prefix_computes_only_new_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target, out, repo = self.fixture(Path(temp))
            values, missing = seed_features(
                target, out, np.array([0, 1, 2], np.int64), 3, "siglip2", repo
            )
            np.testing.assert_array_equal(
                values[:2], np.load(source / "siglip2_feature_heads/features.npy")
            )
            np.testing.assert_array_equal(missing, [2])

    def test_changed_encoder_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            _, target, out, repo = self.fixture(Path(temp))
            (out / "encoder.pt").write_bytes(b"different weights")
            with self.assertRaisesRegex(AssertionError, "Encoder weights changed"):
                seed_features(
                    target, out, np.array([0, 1, 2], np.int64), 3, "siglip2", repo
                )

    def test_changed_preprocessing_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            _, target, out, repo = self.fixture(Path(temp))
            (repo / "model_training/v2/final/canon.py").write_text("CANON = 255\n")
            with self.assertRaisesRegex(AssertionError, "Preprocessing"):
                seed_features(
                    target, out, np.array([0, 1, 2], np.int64), 3, "siglip2", repo
                )

    def test_changed_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            _, target, out, repo = self.fixture(Path(temp))
            path = target / "cnn_inputs/rows.json"
            rows = json.loads(path.read_text())
            rows[0]["path"] = "changed"
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(AssertionError, "Input row prefix"):
                seed_features(
                    target, out, np.array([0, 1, 2], np.int64), 3, "siglip2", repo
                )

    def test_changed_runtime_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            _, target, out, repo = self.fixture(Path(temp))
            with patch(
                "feature_cache_reuse.runtime_identity",
                return_value={"torch": "changed"},
            ):
                with self.assertRaisesRegex(AssertionError, "runtime changed"):
                    seed_features(
                        target, out, np.array([0, 1, 2], np.int64), 3, "siglip2", repo
                    )

    def test_modified_canonical_pixel_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target, out, repo = self.fixture(Path(temp))
            pixels = source / "pixels.npy"
            before = pixels.stat()
            with pixels.open("r+b") as stream:
                stream.seek(-4, 2)
                stream.write(np.float32(255.0).tobytes())
            os.utime(pixels, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000000))
            with self.assertRaisesRegex(AssertionError, "pixel shard modified"):
                seed_features(
                    target, out, np.array([0, 1, 2], np.int64), 3, "siglip2", repo
                )


if __name__ == "__main__":
    unittest.main()
