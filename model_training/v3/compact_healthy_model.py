"""Small Healthy repair head sharing the unchanged deployed DINO/CNN features."""

from pathlib import Path
import math

import torch
from torch import nn
from torch.nn import functional as F

from campaign_cnn_export import make_model
from campaign_consensus import agreement
from leaf_feature_expert import LeafFeatureHead

T = 0.8814700841903687


class FrozenLeafFeatures(nn.Module):
    def __init__(self, primary_spec):
        super().__init__()
        self.primary = make_model(primary_spec)

    def components(self, nhwc):
        x = nhwc.permute(0, 3, 1, 2)
        dino = self.primary.backbone.forward_head(
            self.primary.backbone.forward_features(x), pre_logits=True
        )
        return self.from_dino(nhwc, dino)

    def from_dino(self, nhwc, dino):
        x = nhwc.permute(0, 3, 1, 2)
        rgb = (
            x * self.primary.canonical_std
            + self.primary.canonical_mean
            - self.primary.cnn_mean
        ) / self.primary.cnn_std
        cnn = self.primary.cnn.forward_head(
            self.primary.cnn.forward_features(rgb), pre_logits=True
        )
        raw = self.primary.backbone.head(dino)
        expert = self.primary.cnn.get_classifier()(cnn)
        if self.primary.confirmation is not None:
            calibrated = (
                expert * self.primary.cnn_calibration[:, 0]
                + self.primary.cnn_calibration[:, 1]
            )
            expert = agreement(self.primary.confirmation(dino), calibrated)
        reference = self.primary.fusion(raw, expert)
        features = torch.cat(
            [
                F.normalize(dino, dim=1) * math.sqrt(384),
                F.normalize(cnn, dim=1) * math.sqrt(1024),
            ],
            1,
        )
        return features, reference


def calibrated_probabilities(logits):
    bias = logits.new_tensor([0.0] * 8 + [0.75])
    return F.softmax((logits + bias) / T, dim=-1)


def propose(reference, healthy_probability, minimum):
    """Raise only Healthy; keep every other original class ratio and verdict."""
    p = calibrated_probabilities(reference)
    top2 = p.topk(2, dim=-1)
    top, confidence, margin = (
        top2.indices[:, 0],
        top2.values[:, 0],
        top2.values[:, 0] - top2.values[:, 1],
    )
    confident_healthy = (top == 1) & (confidence >= 0.6) & (margin >= 0.2)
    # Named winners, including uncertainty, are protected. The new branch
    # addresses only rejected/uncertain Healthy or Other leaf observations.
    eligible = ((top == 1) | (top == 8)) & ~confident_healthy
    disease_mass = p[:, [0, 2, 3, 4, 5, 6, 7]].sum(1)
    eligible = eligible & ~((p[:, 8] <= 0.1) & (disease_mass >= 0.6))
    active = (
        eligible & (healthy_probability >= minimum) & (healthy_probability > p[:, 1])
    )
    updated = p * ((1 - healthy_probability) / (1 - p[:, 1]).clamp_min(1e-8))[:, None]
    # Avoid an in-place tensor update so the exported graph stays functional.
    mask = p.new_tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])[None]
    updated = updated * (1 - mask) + healthy_probability[:, None] * mask
    logits = T * updated.clamp_min(1e-12).log() - p.new_tensor([0.0] * 8 + [0.75])
    return torch.where(active[:, None], logits, reference)


class CompactHealthyRepair(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.features = FrozenLeafFeatures(spec["primary_spec"])
        self.head = LeafFeatureHead(spec["head_kind"], 1408)
        self.register_buffer(
            "calibration", torch.tensor(spec["calibration"], dtype=torch.float32)
        )
        self.minimum = spec["healthy_min"]

    def forward(self, nhwc):
        features, reference = self.features.components(nhwc)
        healthy = (
            self.head(features) * self.calibration[0] + self.calibration[1]
        ).sigmoid()
        candidate = propose(reference, healthy, self.minimum)
        return torch.cat([candidate, reference], 1)


def primary(campaign):
    path = Path(campaign) / "export_cnn_safe/pytorch_bundle.pt"
    state = torch.load(path, weights_only=True, map_location="cpu")
    spec = {k: v for k, v in state.items() if k != "model"}
    model = FrozenLeafFeatures(spec).eval().requires_grad_(False)
    state["model"].setdefault("cnn_calibration", model.primary.cnn_calibration)
    model.primary.load_state_dict(state["model"], strict=True)
    return spec, model


def restore(path):
    state = torch.load(path, weights_only=True, map_location="cpu")
    assert state["architecture"] == "CompactHealthyRepair"
    assert state["original_classifier_frozen"] and not state["restricted_teacher_used"]
    model = CompactHealthyRepair(state).eval().requires_grad_(False)
    model.load_state_dict(state["model"], strict=True)
    return model
