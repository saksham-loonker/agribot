"""Resume must reject different pixels even with unchanged path/size/mtime."""

from pathlib import Path
import hashlib
import os
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_training/v3"))
from evaluation_inputs import input_identity


class EvaluationInputTests(unittest.TestCase):
    def test_identical_images_retain_order_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "leaf.jpg"
            path.write_bytes(b"source pixels")
            rows = [dict(id=i, path=str(path)) for i in ("second", "first")]
            actual = input_identity(rows)
            self.assertEqual(actual, input_identity(rows))
            self.assertEqual([r["id"] for r in actual], ["second", "first"])
            self.assertEqual(
                actual[0]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
            )

    def test_changed_bytes_cannot_reuse_predictions_despite_same_file_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "leaf.jpg"
            path.write_bytes(b"old pixels")
            before = path.stat()
            rows = [dict(id="leaf", path=str(path))]
            original = input_identity(rows)
            path.write_bytes(b"new pixels")
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            self.assertEqual(path.stat().st_size, before.st_size)
            self.assertEqual(path.stat().st_mtime_ns, before.st_mtime_ns)
            self.assertNotEqual(original, input_identity(rows))

    def test_publisher_hash_mismatch_fails_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "leaf.jpg"
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "source image changed"):
                input_identity([dict(id="leaf", path=str(path), file_sha256="0" * 64)])

    def test_duplicate_manifest_ids_fail_before_cache_admission(self):
        with self.assertRaisesRegex(ValueError, "duplicate input IDs"):
            input_identity([dict(id="same", path="a"), dict(id="same", path="b")])

    def test_deleted_source_cannot_silently_reuse_a_cached_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                input_identity([dict(id="leaf", path=str(Path(tmp) / "deleted.jpg"))])


if __name__ == "__main__":
    unittest.main()
