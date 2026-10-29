"""Independent MobileCLIP Healthy verifier with the installed recognizer frozen."""

from pathlib import Path
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from campaign_cnn_export import make_model as make_reference
from leaf_feature_expert import LeafFeatureHead
from leaf_feature_repair import reference_components
from leaf_healthy_repair import fusion
from mobileclip_features import CanonicalMobileClip, digest
from train import LABELS


class MobileClipRepair(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.reference = make_reference(spec["reference"])
        self.encoder = CanonicalMobileClip(spec["encoder"])
        self.family = spec["family"]
        dimension = 896 if self.family == "mobileclip_dino" else 512
        self.head = LeafFeatureHead(spec["head_kind"], dimension)
        self.fusion = fusion(
            spec["policy"]["healthy_min"],
            spec["policy"]["disease_max"],
            spec["policy"].get("allow_leaf_disease_override", False),
        )
        self.register_buffer(
            "calibration", torch.tensor(spec["calibration"], dtype=torch.float32)
        )
        self.reference_logits = spec.get("reference_logits", True)
        assert (
            not spec["policy"].get("allow_leaf_disease_override", False)
            or self.reference_logits
        )
        self.species_min = 0.0

    def components(self, nhwc):
        reference, original = reference_components(self.reference, nhwc)
        features = self.encoder(nhwc)
        if self.family == "mobileclip_dino":
            features = torch.cat(
                [features, F.normalize(original, dim=1) * math.sqrt(384)], 1
            )
        healthy = self.head(features) * self.calibration[0] + self.calibration[1]
        return reference, healthy, torch.ones_like(healthy)

    def propose(self, reference, healthy, species_unused):
        return self.fusion(
            reference, torch.stack([healthy, torch.full_like(healthy, 20.0)], 1)
        )

    def forward(self, nhwc):
        reference, h, g = self.components(nhwc)
        candidate = self.propose(reference, h, g)
        return (
            torch.cat([candidate, reference], 1) if self.reference_logits else candidate
        )


def spec_and_weights(root, directory, policy):
    root, directory = Path(root), Path(directory)
    base = torch.load(
        root.parent / "export_cnn_safe/pytorch_bundle.pt",
        weights_only=True,
        map_location="cpu",
    )
    head = torch.load(directory / "best.pt", weights_only=True, map_location="cpu")
    encoder_path = directory.parent / "encoder.pt"
    encoder = torch.load(encoder_path, weights_only=True, map_location="cpu")
    identity = head["protocol"]["feature_identity"]
    assert digest(encoder_path) == identity["encoder_sha256"]
    result = json.loads((directory / "result.json").read_text())
    spec = dict(
        architecture="MobileClipRepair",
        labels=LABELS,
        reference={k: v for k, v in base.items() if k != "model"},
        encoder=encoder["spec"],
        family=identity["family"],
        head_kind=head["kind"],
        calibration=result["calibration"],
        policy=policy,
        reference_logits=True,
        head_protocol=head["protocol"],
    )
    model = MobileClipRepair(spec).eval()
    model.reference.load_state_dict(base["model"], strict=True)
    model.encoder.load_state_dict(encoder["model"], strict=True)
    model.head.load_state_dict(head["model"], strict=True)
    return spec, model.requires_grad_(False)


def restore(path):
    state = torch.load(path, weights_only=True, map_location="cpu")
    assert state["architecture"] == "MobileClipRepair" and state["labels"] == LABELS
    model = MobileClipRepair(state).eval()
    model.load_state_dict(state["model"], strict=True)
    return model
