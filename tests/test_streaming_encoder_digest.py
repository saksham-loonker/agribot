"""Large feature caches must not require a second full-file allocation to hash."""

from pathlib import Path
import hashlib
import importlib.util
import sys
import tempfile
import unittest
from unittest.mock import patch

READY = all(importlib.util.find_spec(name) is not None for name in ("torch", "timm"))
if READY:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_training/v3"))
    from permissive_encoder import digest


@unittest.skipUnless(READY, "Optional training environment required")
class StreamingEncoderDigestTests(unittest.TestCase):
    def test_encoder_digest_retains_sha256_without_whole_file_reads(self):
        data = bytes(range(256)) * 8193
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.npy"
            path.write_bytes(data)
            with patch.object(
                Path,
                "read_bytes",
                side_effect=AssertionError("Unbounded whole-file allocation"),
            ):
                self.assertEqual(digest(path), hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    unittest.main()
