"""Image-scoped Healthy-tomato verifier, separate from the disease classifier."""

from pathlib import Path
import json

import torch
from torch import nn

from leaf_feature_expert import LeafFeatureHead
from mobileclip_features import CanonicalMobileClip, digest


class MobileClipScene(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.encoder = CanonicalMobileClip(spec["encoder"])
        self.head = LeafFeatureHead(spec["head_kind"], 512)
        self.register_buffer(
            "calibration", torch.tensor(spec["calibration"], dtype=torch.float32)
        )

    def forward(self, nhwc):
        return self.head(self.encoder(nhwc)) * self.calibration[0] + self.calibration[1]


def load(directory):
    directory = Path(directory)
    state = torch.load(directory / "best.pt", weights_only=True, map_location="cpu")
    if state.get("architecture") == "MobileClipSceneSft":
        result = json.loads((directory / "result.json").read_text())
        spec = dict(
            state["spec"],
            calibration=result["calibration"],
            training_configuration=state["configuration"],
        )
        model = MobileClipScene(spec).eval().requires_grad_(False)
        model.load_state_dict(state["model"], strict=True)
        model.calibration.copy_(
            torch.tensor(result["calibration"], dtype=torch.float32)
        )
        return spec, model
    assert state["protocol"]["task_scope"] == "primary_image"
    encoder_path = directory.parent / "encoder.pt"
    assert (
        digest(encoder_path) == state["protocol"]["feature_identity"]["encoder_sha256"]
    )
    encoder = torch.load(encoder_path, weights_only=True, map_location="cpu")
    result = json.loads((directory / "result.json").read_text())
    spec = dict(
        architecture="MobileClipScene",
        encoder=encoder["spec"],
        head_kind=state["kind"],
        calibration=result["calibration"],
        task_scope="primary_image",
        head_protocol=state["protocol"],
    )
    model = MobileClipScene(spec).eval().requires_grad_(False)
    model.encoder.load_state_dict(encoder["model"], strict=True)
    model.head.load_state_dict(state["model"], strict=True)
    return spec, model


def restore(path):
    state = torch.load(path, weights_only=True, map_location="cpu")
    assert (
        state["architecture"] == "MobileClipScene"
        and state["task_scope"] == "primary_image"
    )
    model = MobileClipScene(state).eval()
    model.load_state_dict(state["model"], strict=True)
    return model
