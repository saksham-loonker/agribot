"""Leaf-scoped Healthy repair with the complete installed recognizer frozen.

Only independently corroborated Healthy probability can be raised. The new
experts cannot supply any disease name. An optional 18-value output preserves
the exact reference logits for crop confirmation and disease-priority voting.
"""

from pathlib import Path
import json

import torch
from torch import nn

from campaign_cnn_export import make_model as make_reference
from campaign_cnn_fusion import ExpertFusion
from phase7_verifier import CanonicalVerifier
from train import LABELS


def fusion(healthy_min, disease_max):
    return ExpertFusion(
        positive=1.0,
        negative=0.0,
        known=0.0,
        calibration=[[1.0, 0.0], [1.0, 0.0]],
        positive_scope="legacy",
        protect_healthy=True,
        known_scope="healthy",
        healthy_min=healthy_min,
        known_min=0.0,
        require_known_for_healthy=False,
        negative_policy="abstain",
        require_healthy_for_known=True,
        disease_max=disease_max,
        healthy_disease_margin=-1.0,
    )


class LeafHealthyRepair(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.reference = make_reference(spec["reference"])
        self.healthy = CanonicalVerifier(spec["healthy"], spec["healthy_calibration"])
        self.species = CanonicalVerifier(spec["species"], spec["species_calibration"])
        self.fusion = fusion(
            spec["policy"]["healthy_min"], spec["policy"]["disease_max"]
        )
        self.species_min = spec["policy"]["species_min"]
        self.reference_logits = spec.get("reference_logits", True)

    def components(self, nhwc):
        return (
            self.reference(nhwc),
            self.healthy.logits(nhwc).ravel(),
            self.species(nhwc).ravel(),
        )

    def propose(self, reference, healthy_logit, species_probability):
        healthy = torch.where(
            species_probability >= self.species_min,
            healthy_logit,
            torch.full_like(healthy_logit, -100.0),
        )
        expert = torch.stack([healthy, torch.full_like(healthy, 20.0)], 1)
        return self.fusion(reference, expert)

    def forward(self, nhwc):
        reference, healthy, species = self.components(nhwc)
        candidate = self.propose(reference, healthy, species)
        return (
            torch.cat([candidate, reference], 1) if self.reference_logits else candidate
        )


def spec_and_weights(root, species_directory, policy):
    root = Path(root)
    species_directory = Path(species_directory)
    baseline = torch.load(
        root.parent / "export_cnn_safe/pytorch_bundle.pt",
        weights_only=True,
        map_location="cpu",
    )
    healthy = torch.load(
        root / "leaf_healthy/best.pt", weights_only=True, map_location="cpu"
    )
    species = torch.load(
        species_directory / "best.pt", weights_only=True, map_location="cpu"
    )
    spec = dict(
        architecture="LeafHealthyRepair",
        labels=LABELS,
        reference={k: v for k, v in baseline.items() if k != "model"},
        healthy={k: healthy[k] for k in ["architecture", "mean", "std"]},
        species={k: species[k] for k in ["architecture", "mean", "std"]},
        healthy_calibration=json.loads((root / "leaf_healthy/result.json").read_text())[
            "calibration"
        ],
        species_calibration=json.loads((species_directory / "result.json").read_text())[
            "calibration"
        ],
        policy=policy,
        reference_logits=True,
    )
    model = LeafHealthyRepair(spec).eval()
    model.reference.load_state_dict(baseline["model"], strict=True)
    model.healthy.network.load_state_dict(healthy["model"], strict=True)
    model.species.network.load_state_dict(species["model"], strict=True)
    return spec, model


def restore(path):
    state = torch.load(path, weights_only=True, map_location="cpu")
    assert state["architecture"] == "LeafHealthyRepair" and state["labels"] == LABELS
    model = LeafHealthyRepair(state).eval()
    model.load_state_dict(state["model"], strict=True)
    return model
