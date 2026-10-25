"""Leaf-only copied-tail Healthy verifier; original disease branch stays frozen."""

from pathlib import Path
import json

import torch
from torch import nn

from campaign_adapter import prefix
from campaign_cnn_export import make_model as make_reference
from campaign_consensus import agreement
from leaf_healthy_repair import fusion
from phase9_scene_sft import SceneExpert
from train import LABELS


class LeafTailRepair(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.reference = make_reference(spec["reference"])
        self.depth = spec["depth"]
        self.healthy_tail = SceneExpert(self.reference.backbone, self.depth)
        self.register_buffer(
            "calibration", torch.tensor(spec["calibration"], dtype=torch.float32)
        )
        self.fusion = fusion(
            spec["policy"]["healthy_min"],
            spec["policy"]["disease_max"],
            spec["policy"].get("allow_leaf_disease_override", False),
        )
        self.reference_logits = spec.get("reference_logits", True)
        assert (
            not spec["policy"].get("allow_leaf_disease_override", False)
            or self.reference_logits
        )
        self.species_min = 0.0

    def components(self, nhwc):
        base = self.reference
        image = nhwc.permute(0, 3, 1, 2)
        tokens = prefix(base.backbone, image, self.depth)
        healthy = self.healthy_tail(tokens) * self.calibration[0] + self.calibration[1]
        for block in base.backbone.blocks[-self.depth :]:
            tokens = block(tokens)
        features = base.backbone.forward_head(
            base.backbone.norm(tokens), pre_logits=True
        )
        raw = base.backbone.head(features)
        expert = base.cnn(
            (image * base.canonical_std + base.canonical_mean - base.cnn_mean)
            / base.cnn_std
        )
        if base.confirmation is not None:
            expert = agreement(
                base.confirmation(features),
                expert * base.cnn_calibration[:, 0] + base.cnn_calibration[:, 1],
            )
        return base.fusion(raw, expert), healthy, torch.ones_like(healthy)

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


def spec_and_weights(root, tail_directory, policy):
    root, tail_directory = Path(root), Path(tail_directory)
    base = torch.load(
        root.parent / "export_cnn_safe/pytorch_bundle.pt",
        weights_only=True,
        map_location="cpu",
    )
    tail = torch.load(tail_directory / "best.pt", weights_only=True, map_location="cpu")
    assert tail["configuration"]["task_scope"] == "annotated_leaf"
    result = json.loads((tail_directory / "result.json").read_text())
    spec = dict(
        architecture="LeafTailRepair",
        labels=LABELS,
        reference={k: v for k, v in base.items() if k != "model"},
        depth=tail["depth"],
        calibration=result["calibration"],
        policy=policy,
        reference_logits=True,
        configuration=tail["configuration"],
    )
    model = LeafTailRepair(spec).eval()
    model.reference.load_state_dict(base["model"], strict=True)
    model.healthy_tail.load_state_dict(tail["model"], strict=True)
    model.requires_grad_(False)
    return spec, model


def restore(path):
    state = torch.load(path, weights_only=True, map_location="cpu")
    assert state["architecture"] == "LeafTailRepair" and state["labels"] == LABELS
    model = LeafTailRepair(state).eval()
    model.load_state_dict(state["model"], strict=True)
    return model
