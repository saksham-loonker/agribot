"""A new hybrid must reject a component whose cached values have changed."""

from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

READY = all(importlib.util.find_spec(name) for name in ("torch", "timm", "numpy"))
if READY:
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_training/v3"))
    from feature_cache_reuse import digest, record_completed_contract
    from hybrid_field_encoder import extract


@unittest.skipUnless(READY, "Optional encoder environment required")
class HybridCacheIntegrityTests(unittest.TestCase):
    def test_modified_component_fails_before_loading_or_running_an_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = np.array([0], np.int64)
            row_text = json.dumps([dict(id="leaf", path="human-leaf.jpg")])
            directories = []
            for family in ("siglip2", "dinov2_large"):
                work = root / family
                inputs = work / "cnn_inputs"
                inputs.mkdir(parents=True)
                (inputs / "rows.json").write_text(row_text)
                shard = work / "pixels.npy"
                np.save(shard, np.zeros((1, 1, 1, 3), np.float32))
                (inputs / "pixels.json").write_text(
                    json.dumps(dict(center=[str(shard)]))
                )
                directory = work / (family + "_feature_heads")
                directory.mkdir()
                (directory / "encoder.pt").write_bytes(b"unchanged encoder")
                np.save(directory / "features.npy", np.ones((1, 2), np.float32))
                identity = dict(
                    family=family,
                    rows_sha256=digest(inputs / "rows.json"),
                    ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                    encoder_sha256=digest(directory / "encoder.pt"),
                    restricted_teacher_used=False,
                )
                (directory / "feature_identity.json").write_text(json.dumps(identity))
                for name in ("restored_encoder_parity", "input_contract_parity"):
                    (directory / (name + ".json")).write_text(
                        json.dumps(dict(passed=True))
                    )
                record_completed_contract(work, directory, family)
                directories.append(directory)
            (root / "cnn_inputs").mkdir()
            (root / "cnn_inputs/rows.json").write_text(row_text)
            (root / "source_links.json").write_text(
                json.dumps(
                    dict(
                        siglip_features=str(directories[0]),
                        dino_features=str(directories[1]),
                    )
                )
            )
            np.save(directories[0] / "features.npy", np.full((1, 2), 255, np.float32))
            out = root / "hybrid_field_feature_heads"
            out.mkdir()
            with patch(
                "hybrid_field_encoder.torch.load",
                side_effect=RuntimeError("encoder must not be loaded"),
            ):
                with self.assertRaisesRegex(
                    AssertionError, "Completed cache identity changed"
                ):
                    extract(root, out, ids, "hybrid_field")


if __name__ == "__main__":
    unittest.main()
