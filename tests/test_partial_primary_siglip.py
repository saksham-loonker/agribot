from pathlib import Path
import importlib.util
import sys
import unittest

READY = all(importlib.util.find_spec(name) is not None for name in ("torch", "timm"))
if READY:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from timm.models.vision_transformer import VisionTransformer

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_training/v3"))
    from partial_primary_siglip import (
        PartialPrimarySiglip,
        prefix_features,
        prefix_identity,
    )


@unittest.skipUnless(READY, "Optional training environment required")
class PartialPrimarySiglipTests(unittest.TestCase):
    def encoder(self):
        encoder = nn.Module()
        encoder.network = VisionTransformer(
            img_size=32,
            patch_size=8,
            embed_dim=32,
            depth=4,
            num_heads=4,
            num_classes=32,
            global_pool="avg",
        )
        encoder.dimension = 32
        encoder.side = 32
        for name, values in [
            ("canonical_mean", [0.485, 0.456, 0.406]),
            ("canonical_std", [0.229, 0.224, 0.225]),
            ("native_mean", [0.5, 0.5, 0.5]),
            ("native_std", [0.5, 0.5, 0.5]),
        ]:
            encoder.register_buffer(name, torch.tensor(values).view(1, 3, 1, 1))
        return encoder.eval()

    def test_prefix_tail_preserves_original_normalization_and_pooled_features(self):
        encoder = self.encoder()
        model = PartialPrimarySiglip(encoder).eval()
        image = torch.randn(2, 32, 32, 3)
        native = (
            image.permute(0, 3, 1, 2) * encoder.canonical_std
            + encoder.canonical_mean
            - encoder.native_mean
        ) / encoder.native_std
        with torch.inference_mode():
            expected = F.normalize(encoder.network(native), dim=1) * (32**0.5)
            actual = model.features(prefix_features(model.encoder, image))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_tail_update_and_restore_preserve_original_and_frozen_prefix(self):
        encoder = self.encoder()
        model = PartialPrimarySiglip(encoder)
        before = prefix_identity(model.encoder)
        original = {k: v.clone() for k, v in encoder.state_dict().items()}
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0.001
        )
        with torch.no_grad():
            tokens = prefix_features(model.encoder, torch.randn(2, 32, 32, 3))
        model(tokens).square().mean().backward()
        optimizer.step()
        self.assertEqual(prefix_identity(model.encoder), before)
        restored = PartialPrimarySiglip(encoder).eval()
        restored.load_tail(model.tail_state())
        model.eval()
        torch.testing.assert_close(restored(tokens), model(tokens), rtol=0, atol=0)
        for key, value in encoder.state_dict().items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)
