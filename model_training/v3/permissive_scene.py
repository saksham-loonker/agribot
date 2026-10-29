"""Separately calibrated image Healthy verifier built from Apache-2.0 weights."""

from pathlib import Path
import json

import torch
from torch import nn

from leaf_feature_expert import LeafFeatureHead
from permissive_encoder import CanonicalPermissiveEncoder, digest


class PermissiveScene(nn.Module):
    def __init__(self, spec):
        super().__init__()
        assert spec["encoder"]["license"] == "apache-2.0"
        if spec["encoder"].get("family") == "hybrid_field":
            from hybrid_field_encoder import HybridFieldEncoder

            self.encoder = HybridFieldEncoder(spec["encoder"])
        elif spec["encoder"].get("family") == "dinov2_large":
            from dino_large_encoder import DinoLargeEncoder

            self.encoder = DinoLargeEncoder(spec["encoder"])
        else:
            self.encoder = CanonicalPermissiveEncoder(spec["encoder"])
        self.head = LeafFeatureHead(spec["head_kind"], spec["encoder"]["dimension"])
        self.register_buffer(
            "calibration", torch.tensor(spec["calibration"], dtype=torch.float32)
        )

    def forward(self, nhwc):
        return self.head(self.encoder(nhwc)) * self.calibration[0] + self.calibration[1]


def load(directory):
    directory = Path(directory)
    state = torch.load(directory / "best.pt", weights_only=True, map_location="cpu")
    if state.get("architecture") == "PermissiveSceneSft":
        assert state["configuration"]["model_family"] == "siglip2"
        assert state["configuration"]["restricted_teacher_used"] is False
        spec = dict(state["spec"])
        result = json.loads((directory / "result.json").read_text())
        model = PermissiveScene(spec).eval().requires_grad_(False)
        model.load_state_dict(state["model"], strict=True)
        spec["calibration"] = result["calibration"]
        model.calibration.copy_(torch.tensor(spec["calibration"]))
        return spec, model
    assert state["protocol"]["task_scope"] == "primary_image"
    identity = state["protocol"]["feature_identity"]
    assert (
        identity["family"] in ("siglip2", "dinov2_large", "hybrid_field")
        and identity["restricted_teacher_used"] is False
    )
    encoder_path = directory.parent / "encoder.pt"
    assert digest(encoder_path) == identity["encoder_sha256"]
    encoder = torch.load(encoder_path, weights_only=True, map_location="cpu")
    result = json.loads((directory / "result.json").read_text())
    spec = dict(
        architecture="PermissiveScene",
        encoder=encoder["spec"],
        head_kind=state["kind"],
        calibration=result["calibration"],
        task_scope="primary_image",
        head_protocol=state["protocol"],
        restricted_teacher_used=False,
    )
    model = PermissiveScene(spec).eval().requires_grad_(False)
    model.encoder.load_state_dict(encoder["model"], strict=True)
    model.head.load_state_dict(state["model"], strict=True)
    return spec, model


def restore(path):
    state = torch.load(path, weights_only=True, map_location="cpu")
    assert (
        state["architecture"] == "PermissiveScene"
        and state["restricted_teacher_used"] is False
    )
    model = PermissiveScene(state).eval().requires_grad_(False)
    model.load_state_dict(state["model"], strict=True)
    return model
