"""Canonical pixel shards must preserve values and sample/label ordering."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

READY = importlib.util.find_spec("numpy") is not None
if READY:
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_training/v3"))
    from campaign_pixels import CompositePixels


@unittest.skipUnless(READY, "NumPy is optional")
class CanonicalPixelShardTests(unittest.TestCase):
    def test_cross_shard_and_reordered_reads_preserve_exact_normalized_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            expected = np.repeat(
                np.array([-2.1, -1.0, 0.0, 1.5, 2.4], np.float32)[:, None], 12, axis=1
            ).reshape(5, 2, 2, 3)
            np.save(root / "a.npy", expected[:3])
            np.save(root / "b.npy", expected[3:])
            with CompositePixels([root / "a.npy", root / "b.npy"]) as pixels:
                np.testing.assert_array_equal(
                    pixels[[4, 0, 3, 2, -1]], expected[[4, 0, 3, 2, -1]]
                )
                np.testing.assert_array_equal(pixels[2:5], expected[2:5])
                np.testing.assert_array_equal(pixels[-1], expected[-1])
                self.assertEqual(pixels[[]].shape, (0, 2, 2, 3))
                with self.assertRaises(IndexError):
                    pixels[5]
            with self.assertRaises(ValueError):
                pixels[0]

    def test_shards_reject_silent_dtype_conversion(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "half.npy"
            np.save(path, np.zeros((1, 2, 2, 3), np.float16))
            with self.assertRaises(ValueError):
                CompositePixels([path])


if __name__ == "__main__":
    unittest.main()
