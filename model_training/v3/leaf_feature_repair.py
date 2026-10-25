"""Repair Healthy with a tiny head sharing the original DINO inference pass."""

from pathlib import Path
import json

import torch
from torch import nn

from campaign_cnn_export import make_model as make_reference
from campaign_consensus import agreement
from leaf_feature_expert import LeafFeatureHead
from leaf_healthy_repair import fusion
from train import LABELS


def reference_components(base, nhwc):
    image = nhwc.permute(0, 3, 1, 2)
    features = base.backbone.forward_head(
        base.backbone.forward_features(image), pre_logits=True
    )
    raw = base.backbone.head(features)
    expert = base.cnn(
        (image * base.canonical_std + base.canonical_mean - base.cnn_mean)
        / base.cnn_std
    )
    if base.confirmation is not None:
        cnn = expert * base.cnn_calibration[:, 0] + base.cnn_calibration[:, 1]
        expert = agreement(base.confirmation(features), cnn)
    return base.fusion(raw, expert), features


class LeafFeatureRepair(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.reference = make_reference(spec["reference"])
        self.head = LeafFeatureHead(spec["head_kind"])
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
        reference, features = reference_components(self.reference, nhwc)
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


def spec_and_weights(root, head_directory, policy):
    root = Path(root)
    head_directory = Path(head_directory)
    baseline = torch.load(
        root.parent / "export_cnn_safe/pytorch_bundle.pt",
        weights_only=True,
        map_location="cpu",
    )
    head = torch.load(head_directory / "best.pt", weights_only=True, map_location="cpu")
    result = json.loads((head_directory / "result.json").read_text())
    spec = dict(
        architecture="LeafFeatureRepair",
        labels=LABELS,
        reference={k: v for k, v in baseline.items() if k != "model"},
        head_kind=head["kind"],
        calibration=result["calibration"],
        policy=policy,
        reference_logits=True,
        head_protocol=head["protocol"],
        binary_task="Healthy tomato leaf versus all other/unhealthy leaf views; no separate species gate",
    )
    model = LeafFeatureRepair(spec).eval()
    model.reference.load_state_dict(baseline["model"], strict=True)
    model.head.load_state_dict(head["model"], strict=True)
    return spec, model


def restore(path):
    state = torch.load(path, weights_only=True, map_location="cpu")
    assert state["architecture"] == "LeafFeatureRepair" and state["labels"] == LABELS
    model = LeafFeatureRepair(state).eval()
    model.load_state_dict(state["model"], strict=True)
    return model
