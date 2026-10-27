"""The copied DINO tail must preserve original tokens, logits and initialization."""

from pathlib import Path
import importlib.util
import sys
import unittest

READY = all(importlib.util.find_spec(name) is not None for name in ("torch", "timm"))
if READY:
    import torch
    import timm

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_training/v3"))
    from partial_dino_healthy import (
        PartialDinoHealthyTail,
        prefix_features,
        prefix_identity,
        healthy_margin,
    )


@unittest.skipUnless(READY, "Optional training environment required")
class PartialDinoHealthyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(1519)
        timm.layers.set_fused_attn(False)
        self.original = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=False,
            num_classes=9,
            img_size=224,
        ).eval()
        self.images = torch.randn(1, 3, 224, 224)

    def test_cached_prefix_reconstructs_original_logits_and_healthy_probability(self):
        model = PartialDinoHealthyTail(self.original).eval()
        with torch.no_grad():
            expected = self.original(self.images)
            actual = model(prefix_features(self.original, self.images))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        posterior = (
            (expected + expected.new_tensor([0.0] * 8 + [0.75])) / 0.8814700841903687
        ).softmax(1)
        torch.testing.assert_close(healthy_margin(actual).sigmoid(), posterior[:, 1])

    def test_adaptation_cannot_modify_the_original_recognizer_or_frozen_prefix(self):
        model = PartialDinoHealthyTail(self.original).train()
        original = {
            key: value.clone() for key, value in self.original.state_dict().items()
        }
        identity = prefix_identity(model.backbone)
        with torch.no_grad():
            tokens = prefix_features(self.original, self.images)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0.001
        )
        initial = model(tokens).detach().clone()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            healthy_margin(model(tokens)), torch.ones(1)
        )
        loss.backward()
        optimizer.step()
        self.assertEqual(identity, prefix_identity(model.backbone))
        self.assertTrue(
            all(
                torch.equal(value, self.original.state_dict()[key])
                for key, value in original.items()
            )
        )
        self.assertFalse(torch.equal(initial, model(tokens).detach()))


if __name__ == "__main__":
    unittest.main()
