"""Self-contained licensed union; no original disease probabilities changed."""

from pathlib import Path
import json

import torch
from torch import nn

from leaf_feature_expert import LeafFeatureHead
from permissive_encoder import digest
from phase13_context_model import PermissiveContextFusion, build as build_base


class PermissiveContextUnion(nn.Module):
    def __init__(self, spec):
        super().__init__()
        assert spec["restricted_teacher_used"] is False
        self.base = PermissiveContextFusion(spec["base"])
        self.head = LeafFeatureHead(spec["head_kind"], 768)
        self.register_buffer(
            "calibration", torch.tensor(spec["calibration"], dtype=torch.float32)
        )

    def forward(self, nhwc):
        pair = self.base(nhwc)
        # Deliberate unoptimized prototype. Export may share encoder features
        # only after the equivalent graph passes the same parity checks.
        z = (
            self.head(self.base.permissive.encoder(nhwc)) * self.calibration[0]
            + self.calibration[1]
        )
        return torch.cat([pair, z[:, None]], 1)


def build(campaign, candidate, model_root):
    root = Path(model_root)
    identity = candidate["union"]
    assert (
        digest(root / "fusion_refined_selection/candidate.json")
        == identity["teacher_candidate_sha256"]
    )
    teacher = json.loads((root / "fusion_refined_selection/candidate.json").read_text())
    assert candidate["fusion"] == teacher["fusion"]
    assert (
        digest(root / "fusion_refined_selection_validation/pytorch_bundle.pt")
        == identity["teacher_bundle_sha256"]
    )
    directory = Path(identity["student_directory"])
    assert digest(directory / "best.pt") == identity["student_checkpoint_sha256"]
    assert digest(directory / "result.json") == identity["student_calibration_sha256"]
    state = torch.load(directory / "best.pt", weights_only=True, map_location="cpu")
    assert state["protocol"]["restricted_teacher_used"] is False
    assert (
        state["protocol"]["feature_identity"]["encoder_sha256"]
        == candidate["input_identity"]["encoder_sha256"]
    )
    base_spec, base = build_base(campaign, candidate, model_root)
    spec = dict(
        architecture="PermissiveContextUnion",
        base=base_spec,
        head_kind=state["kind"],
        calibration=json.loads((directory / "result.json").read_text())["calibration"],
        restricted_teacher_used=False,
        policy=candidate["policy"],
    )
    model = PermissiveContextUnion(spec).eval().requires_grad_(False)
    model.base.load_state_dict(base.state_dict(), strict=True)
    model.head.load_state_dict(state["model"], strict=True)
    return spec, model


def restore(path):
    state = torch.load(path, weights_only=True, map_location="cpu")
    assert state["architecture"] == "PermissiveContextUnion"
    model = PermissiveContextUnion(state).eval().requires_grad_(False)
    model.load_state_dict(state["model"], strict=True)
    return model
