"""Canonical-input binary verifier with frozen validation calibration."""

from pathlib import Path
import json
import timm
import torch
from torch import nn


class CanonicalVerifier(nn.Module):
    def __init__(self, state, calibration):
        super().__init__()
        self.network = timm.create_model(
            state["architecture"], pretrained=False, num_classes=1
        )
        self.network.load_state_dict(state["model"], strict=True)
        for name, values in [
            ("canonical_mean", [0.485, 0.456, 0.406]),
            ("canonical_std", [0.229, 0.224, 0.225]),
            ("mean", state["mean"]),
            ("std", state["std"]),
        ]:
            self.register_buffer(name, torch.tensor(values).view(1, 3, 1, 1))
        self.register_buffer(
            "calibration", torch.tensor(calibration, dtype=torch.float32)
        )

    def forward(self, nhwc):
        return self.logits(nhwc).sigmoid()

    def logits(self, nhwc):
        x = nhwc.permute(0, 3, 1, 2)
        z = self.network(
            (x * self.canonical_std + self.canonical_mean - self.mean) / self.std
        )
        return z * self.calibration[0] + self.calibration[1]


def restore(directory):
    directory = Path(directory)
    state = torch.load(directory / "best.pt", weights_only=True, map_location="cpu")
    result = json.loads((directory / "result.json").read_text())
    return CanonicalVerifier(state, result["calibration"]).eval()
