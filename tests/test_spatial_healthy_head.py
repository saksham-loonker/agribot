"""Localized unhealthy evidence must survive pooling into a leaf verdict."""

from pathlib import Path
import importlib.util
import sys
import unittest

READY = importlib.util.find_spec("torch") is not None
if READY:
    import torch
    from torch import nn

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_training/v3"))
    from spatial_healthy_head import SpatialHealthyHead


@unittest.skipUnless(READY, "Optional training environment required")
class SpatialHealthyHeadTests(unittest.TestCase):
    def fixture(self):
        model = SpatialHealthyHead(1)
        model.local = nn.Identity()
        return model

    def test_uniform_spatial_evidence_does_not_change_with_region_count(self):
        model = self.fixture()
        for side in (1, 3, 7):
            torch.testing.assert_close(
                model(torch.full((2, 1, side, side), 3.0)), torch.full((2,), 3.0)
            )

    def test_small_unhealthy_region_is_not_hidden_by_healthy_average(self):
        features = torch.full((1, 1, 7, 7), 8.0)
        features[0, 0, 3, 3] = -8.0
        self.assertGreater(float(features.mean()), 7.0)
        self.assertLess(float(self.fixture()(features)), 0.0)

    def test_aggregate_negative_gradient_focuses_on_the_local_conflict(self):
        features = torch.full((1, 1, 7, 7), 8.0)
        features[0, 0, 3, 3] = -8.0
        features.requires_grad_(True)
        result = self.fixture()(features)
        torch.nn.functional.binary_cross_entropy_with_logits(
            result, torch.zeros(1)
        ).backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(
            float(features.grad[0, 0, 3, 3]), 100 * float(features.grad[0, 0, 0, 0])
        )


if __name__ == "__main__":
    unittest.main()
