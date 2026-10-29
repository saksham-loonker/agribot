"""Shared Apache encoder with three independently calibrated image logits.

Output order: positive Healthy, tomato genus, negative-guard Healthy. These
are sigmoid logits, never a three-class softmax or nine-class diagnosis.
"""

from pathlib import Path
import json

import torch
from torch import nn

from leaf_feature_expert import LeafFeatureHead
from permissive_encoder import digest
from phase13_context_model import PermissiveContextFusion, build as build_base


class PermissiveContextGuard(nn.Module):
    def __init__(self, spec):
        super().__init__()
        assert spec["restricted_teacher_used"] is False
        self.base = PermissiveContextFusion(spec["base"])
        self.negative_head = LeafFeatureHead(spec["negative_head_kind"], 768)
        self.register_buffer(
            "negative_calibration",
            torch.tensor(spec["negative_calibration"], dtype=torch.float32),
        )

    def forward(self, nhwc):
        features = self.base.permissive.encoder(nhwc)
        z = self.base.calibrated_logits(nhwc, permissive_features=features)
        healthy = (
            z.clamp(-self.base.input_clip, self.base.input_clip) @ self.base.weights
            + self.base.bias
        )
        negative = (
            self.negative_head(features) * self.negative_calibration[0]
            + self.negative_calibration[1]
        )
        return torch.stack([healthy, z[:, 3], negative], 1)


def build(campaign, candidate):
    identity = candidate["guard_identity"]
    root = Path(identity.get("model_root", Path(campaign) / "phase13_siglip"))
    assert identity["positive_source"] == "fusion" and identity["negative_source"] in (
        "cvar_bce",
        "partial_auc",
        "reinforce_label_reward",
        "expected_logit_reward",
        "siglip2",
    )
    directory = Path(identity["negative_directory"])
    for name, key in [
        ("best.pt", "negative_checkpoint_sha256"),
        ("result.json", "negative_calibration_sha256"),
    ]:
        assert digest(directory / name) == identity[key]
    state = torch.load(directory / "best.pt", weights_only=True, map_location="cpu")
    assert state["protocol"].get("restricted_teacher_used", False) is False
    assert state["protocol"]["feature_identity"]["restricted_teacher_used"] is False
    assert (
        state["protocol"]["feature_identity"]["encoder_sha256"]
        == candidate["input_identity"]["encoder_sha256"]
    )
    base_spec, base = build_base(campaign, candidate, root)
    spec = dict(
        architecture="PermissiveContextGuard",
        base=base_spec,
        negative_head_kind=state["kind"],
        negative_calibration=json.loads((directory / "result.json").read_text())[
            "calibration"
        ],
        restricted_teacher_used=False,
        policy=candidate["policy"],
        output_contract=[
            "healthy_positive_logit",
            "tomato_genus_logit",
            "healthy_guard_logit",
        ],
        primary_classifier_unchanged=True,
    )
    model = PermissiveContextGuard(spec).eval().requires_grad_(False)
    model.base.load_state_dict(base.state_dict(), strict=True)
    model.negative_head.load_state_dict(state["model"], strict=True)
    return spec, model


def restore(path):
    state = torch.load(path, weights_only=True, map_location="cpu")
    assert state["architecture"] == "PermissiveContextGuard"
    model = PermissiveContextGuard(state).eval().requires_grad_(False)
    model.load_state_dict(state["model"], strict=True)
    return model
