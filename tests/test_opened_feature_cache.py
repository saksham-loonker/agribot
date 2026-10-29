"""Completed opened-image caches must reject stale pixels and encoder identities."""

from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

READY = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "timm", "sklearn", "cv2")
)
if READY:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_training/v3"))
    import opened_siglip_features as cache


@unittest.skipUnless(READY, "Optional training environment required")
class OpenedFeatureCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "phase23_siglip"
        self.encoder = self.source / "siglip2_feature_heads"
        self.encoder.mkdir(parents=True)
        (self.source / "cnn_inputs").mkdir()
        (self.source / "cnn_inputs/rows.json").write_text("[]")
        (self.encoder / "encoder.pt").write_bytes(b"frozen encoder")
        (self.encoder / "features.npy").write_bytes(b"source features")
        self.image = self.root / "leaf.jpg"
        self.image.write_bytes(b"old pixels")
        self.data = dict(photos={str(self.image): {}}, identity={"opened": "fixed"})
        self.out = self.root / "opened_siglip_feature_cache"
        self.out.mkdir()
        identity = dict(
            encoder_sha256=cache.digest(self.encoder / "encoder.pt"),
            source_feature_array_sha256=cache.digest(self.encoder / "features.npy"),
            source_rows_sha256=cache.digest(self.source / "cnn_inputs/rows.json"),
            paths_sha256=hashlib.sha256(
                json.dumps([str(self.image)]).encode()
            ).hexdigest(),
            preprocessing={"preprocessing": "fixed"},
            runtime={"runtime": "fixed"},
            data_identity=self.data["identity"],
            cold_olid_used=False,
        )
        (self.out / "protocol.json").write_text(json.dumps(identity))
        (self.out / "features.npy").write_bytes(b"completed features")
        (self.out / "inputs.json").write_text(
            json.dumps(
                [dict(path=str(self.image), file_sha256=cache.digest(self.image))]
            )
        )
        (self.out / "complete.json").write_text(
            json.dumps(
                dict(
                    features_sha256=cache.digest(self.out / "features.npy"),
                    inputs_sha256=cache.digest(self.out / "inputs.json"),
                    protocol_sha256=cache.digest(self.out / "protocol.json"),
                )
            )
        )
        # The separate source-cache validator has its own corruption tests.
        for target, value in (
            ("validate_completed_contract", None),
            ("preprocessing_contract", identity["preprocessing"]),
            ("runtime_identity", identity["runtime"]),
        ):
            mock = patch.object(cache, target, return_value=value)
            mock.start()
            self.addCleanup(mock.stop)

    def test_verified_completion_returns_without_loading_an_encoder(self):
        with patch.object(cache, "CanonicalPermissiveEncoder") as encoder:
            self.assertEqual(cache.prepare(self.root, self.data), self.out)
            encoder.assert_not_called()

    def test_changed_source_bytes_reject_cache_despite_unchanged_metadata(self):
        before = self.image.stat()
        self.image.write_bytes(b"new pixels")
        os.utime(self.image, ns=(before.st_atime_ns, before.st_mtime_ns))
        with self.assertRaisesRegex(AssertionError, "source image changed"):
            cache.prepare(self.root, self.data)

    def test_corrupt_cached_features_reject_completion(self):
        (self.out / "features.npy").write_bytes(b"corrupted features")
        with self.assertRaises(AssertionError):
            cache.prepare(self.root, self.data)

    def test_different_encoder_cannot_reuse_original_features(self):
        (self.encoder / "encoder.pt").write_bytes(b"adapted encoder")
        with self.assertRaisesRegex(AssertionError, "cache identity changed"):
            cache.prepare(self.root, self.data)


if __name__ == "__main__":
    unittest.main()
