"""A separate adapted tail must retain the original prefix and initialization."""

from pathlib import Path
import importlib.util
import sys
import unittest

READY = all(importlib.util.find_spec(name) is not None for name in ("torch", "timm"))
if READY:
    import torch
    import timm

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_training/v3"))
    from partial_cnn_healthy import (
        PartialCnnHealthyTail,
        prefix_features,
        prefix_identity,
    )


@unittest.skipUnless(READY, "Optional training environment required")
class PartialCnnHealthyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(1491)
        self.cnn = timm.create_model(
            "mobilenetv3_small_100", pretrained=False, num_classes=2
        ).eval()
        self.inputs = torch.randn(2, 3, 224, 224)

    def test_split_forward_reproduces_the_original_healthy_logit(self):
        model = PartialCnnHealthyTail(self.cnn).eval()
        with torch.no_grad():
            actual = model(prefix_features(self.cnn, self.inputs))
            expected = self.cnn(self.inputs)[:, 0]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_training_cannot_modify_original_weights_or_prefix_statistics(self):
        model = PartialCnnHealthyTail(self.cnn).train()
        original = {key: value.clone() for key, value in self.cnn.state_dict().items()}
        identity = prefix_identity(model.cnn)
        with torch.no_grad():
            inputs = prefix_features(self.cnn, self.inputs)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0.01
        )
        before = model(inputs).detach().clone()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(inputs), torch.ones(2)
        )
        loss.backward()
        optimizer.step()
        self.assertEqual(prefix_identity(model.cnn), identity)
        self.assertTrue(
            all(
                torch.equal(value, self.cnn.state_dict()[key])
                for key, value in original.items()
            )
        )
        self.assertFalse(torch.equal(before, model(inputs).detach()))
        self.assertTrue(
            all(
                not m.training
                for m in model.modules()
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
            )
        )


if __name__ == "__main__":
    unittest.main()
