"""Shared-prefix deployment models for the phase-69 qualified development route.

Leaf output: original nine logits, compact positive/negative Healthy and adapted
DINO Healthy. Whole output: positive/negative Healthy, tomato genus and context.
Routing and temporal evidence remain separate from these model outputs.
"""

from pathlib import Path
import argparse
import copy
import json
import math

import torch
from torch import nn
from torch.nn import functional as F
import timm

from compact_healthy_model import FrozenLeafFeatures, primary
from leaf_feature_expert import LeafFeatureHead
from opened_context_data import read
from partial_dino_healthy import PartialDinoHealthyTail, healthy_margin
from partial_dino_healthy import (
    prefix_features as dino_prefix,
    prefix_identity as dino_identity,
)
from partial_primary_siglip import (
    prefix_features as siglip_prefix,
    prefix_identity as siglip_identity,
)
from permissive_encoder import CanonicalPermissiveEncoder, digest
from permissive_scene import load as load_scene
from training_support import atomic_save


class QualifiedLeafModel(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.frozen = FrozenLeafFeatures(spec["primary_spec"])
        backbone = self.frozen.primary.backbone
        assert backbone.global_pool == "token" and backbone.attn_pool is None
        self.adapted_blocks = copy.deepcopy(backbone.blocks[-2:])
        self.adapted_norm = copy.deepcopy(backbone.norm)
        self.adapted_fc_norm = copy.deepcopy(backbone.fc_norm)
        self.adapted_head = copy.deepcopy(backbone.head)
        self.positive = LeafFeatureHead(spec["positive_kind"], 1408)
        self.negative = LeafFeatureHead(spec["negative_kind"], 1408)
        self.register_buffer(
            "calibrations", torch.tensor(spec["calibrations"], dtype=torch.float32)
        )

    def forward(self, nhwc):
        backbone = self.frozen.primary.backbone
        prefix = dino_prefix(backbone, nhwc.permute(0, 3, 1, 2))
        original = backbone.norm(backbone.blocks[-2:](prefix))
        dino = backbone.forward_head(original, pre_logits=True)
        features, reference = self.frozen.from_dino(nhwc, dino)
        adapted = self.adapted_norm(self.adapted_blocks(prefix))
        raw = self.adapted_head(self.adapted_fc_norm(adapted[:, 0]))
        logits = torch.stack(
            [self.positive(features), self.negative(features), healthy_margin(raw)], 1
        )
        probability = (
            logits * self.calibrations[:, 0] + self.calibrations[:, 1]
        ).sigmoid()
        return torch.cat([reference, probability], 1)


class SiglipSuffix(nn.Module):
    def __init__(self, network):
        super().__init__()
        assert network.attn_pool is not None and network.num_prefix_tokens == 0
        self.blocks = copy.deepcopy(network.blocks[-2:])
        self.norm = copy.deepcopy(network.norm)
        self.pool = copy.deepcopy(network.attn_pool)
        self.fc_norm = copy.deepcopy(network.fc_norm)
        self.head = copy.deepcopy(network.head)

    def forward(self, prefix):
        tokens = self.norm(self.blocks(prefix))
        return F.normalize(
            self.head(self.fc_norm(self.pool(tokens))), dim=1
        ) * math.sqrt(768)


class QualifiedWholeModel(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.base = CanonicalPermissiveEncoder(spec["encoder"])
        self.positive_suffix = SiglipSuffix(self.base.network)
        self.positive = LeafFeatureHead(spec["positive_kind"], 768)
        self.heads = nn.ModuleList(
            [LeafFeatureHead(kind, 768) for kind in spec["other_kinds"]]
        )
        self.register_buffer(
            "calibrations", torch.tensor(spec["calibrations"], dtype=torch.float32)
        )

    def forward(self, nhwc):
        prefix = siglip_prefix(self.base, nhwc)
        network = self.base.network
        tokens = network.norm(network.blocks[-2:](prefix))
        original = F.normalize(network.forward_head(tokens), dim=1) * math.sqrt(768)
        logits = [self.positive(self.positive_suffix(prefix))]
        logits.extend(head(original) for head in self.heads)
        stacked = torch.stack(logits, 1)
        return (stacked * self.calibrations[:, 0] + self.calibrations[:, 1]).sigmoid()


def restore(path):
    timm.layers.set_fused_attn(False)
    state = torch.load(path, weights_only=True, map_location="cpu")
    model = {
        "QualifiedLeafModel": QualifiedLeafModel,
        "QualifiedWholeModel": QualifiedWholeModel,
    }[state["architecture"]](state["spec"])
    model.load_state_dict(state["model"], strict=True)
    return model.eval().requires_grad_(False)


def build(campaign):
    p = Path(campaign)
    out = p / "phase70_qualified"
    out.mkdir(exist_ok=True)
    parent = read(p / "phase69_whole_local/result.json")
    assert parent["passing"] == 1 and all(parent["candidate"]["checks"].values())
    assert (
        not parent["cold_olid_used"]
        and not parent["candidate"]["old_failures"]
        and not parent["candidate"]["internal_failures"]
    )
    components = {
        "primary": "export_cnn_safe/pytorch_bundle.pt",
        "manifest": "export_cnn_safe/model_manifest.json",
        "positive_leaf": "phase29_compact/head_bce_source_balanced/epoch_0.pt",
        "negative_leaf": "phase29_compact/head_bce_source_balanced/best.pt",
        "adapted_dino": "phase55_dino_continuation/head/candidate.pt",
        "original_siglip": "phase23_siglip/siglip2_feature_heads/encoder.pt",
        "positive_whole": "phase23_siglip/sft_ad_tail2_focal_lowfpr/best.pt",
        "positive_whole_calibration": "phase23_siglip/sft_ad_tail2_focal_lowfpr/result.json",
        "negative_whole": "phase23_siglip/risk_heads/cvar_bce/best.pt",
        "negative_whole_calibration": "phase23_siglip/risk_heads/cvar_bce/result.json",
        "genus": "phase41_siglip_genus/best.pt",
        "genus_calibration": "phase41_siglip_genus/result.json",
        "context": "phase39_context_router/best.pt",
        "context_calibration": "phase39_context_router/result.json",
    }
    identity = {
        name: dict(path=path, sha256=digest(p / path))
        for name, path in components.items()
    }
    expected = parent["protocol"]
    for name, key in [("primary", "primary_bundle_sha256")]:
        assert identity[name]["sha256"] == expected["map_contract"][key]
    for name, key in [
        ("positive_leaf", "positive_leaf_sha256"),
        ("negative_leaf", "negative_leaf_sha256"),
        ("adapted_dino", "spatial_sha256"),
        ("genus", "genus_sha256"),
        ("context", "context_sha256"),
    ]:
        assert identity[name]["sha256"] == expected[key]
    assert (
        identity["positive_whole"]["sha256"]
        == expected["data_identity"]["supervised_checkpoint_sha256"]
    )
    assert (
        identity["negative_whole"]["sha256"]
        == expected["data_identity"]["negative_checkpoint_sha256"]
    )
    construction = dict(
        parent_protocol_sha256=digest(p / "phase69_whole_local/protocol.json"),
        parent_result_sha256=digest(p / "phase69_whole_local/result.json"),
        components=identity,
        model_source_sha256=digest(__file__),
        cold_olid_used=False,
        ad_independent=False,
        policy=expected,
        scope="Qualified development candidate; restored/export/fresh/device gates still required",
    )
    frozen = out / "construction.json"
    if frozen.exists():
        assert read(frozen) == construction
    else:
        frozen.write_text(json.dumps(construction, indent=2))
    if (out / "candidate.json").exists():
        candidate = read(out / "candidate.json")
        assert candidate["construction_sha256"] == digest(frozen)
        for item in candidate["bundles"].values():
            assert digest(out / item["file"]) == item["sha256"]
        return candidate
    timm.layers.set_fused_attn(False)
    primary_spec, original = primary(p)
    states = [
        torch.load(p / components[name], weights_only=True, map_location="cpu")
        for name in ("positive_leaf", "negative_leaf", "adapted_dino")
    ]
    leaf_spec = dict(
        primary_spec=primary_spec,
        positive_kind=states[0]["kind"],
        negative_kind=states[1]["kind"],
        calibrations=[
            states[0]["calibration"],
            states[1]["calibration"],
            states[2]["calibration"],
        ],
    )
    leaf = QualifiedLeafModel(leaf_spec).eval().requires_grad_(False)
    leaf.frozen.load_state_dict(original.state_dict(), strict=True)
    leaf.positive.load_state_dict(states[0]["model"], strict=True)
    leaf.negative.load_state_dict(states[1]["model"], strict=True)
    adapted = PartialDinoHealthyTail(original.primary.backbone)
    adapted.load_state_dict(states[2]["model"], strict=True)
    assert (
        dino_identity(original.primary.backbone)
        == dino_identity(adapted.backbone)
        == states[2]["prefix_state_sha256"]
    )
    for target, source in [
        ("adapted_blocks", adapted.backbone.blocks[-2:]),
        ("adapted_norm", adapted.backbone.norm),
        ("adapted_fc_norm", adapted.backbone.fc_norm),
        ("adapted_head", adapted.backbone.head),
    ]:
        getattr(leaf, target).load_state_dict(source.state_dict(), strict=True)
    atomic_save(
        dict(
            architecture="QualifiedLeafModel", spec=leaf_spec, model=leaf.state_dict()
        ),
        out / "leaf.pt",
    )
    del states, original, adapted, leaf
    encoder = torch.load(
        p / components["original_siglip"], weights_only=True, map_location="cpu"
    )
    positive_spec, positive = load_scene((p / components["positive_whole"]).parent)
    heads = []
    calibrations = [positive_spec["calibration"]]
    for name in ("negative_whole", "genus", "context"):
        state = torch.load(p / components[name], weights_only=True, map_location="cpu")
        heads.append(state)
        calibrations.append(read(p / components[name + "_calibration"])["calibration"])
    spec = dict(
        encoder=encoder["spec"],
        positive_kind=positive_spec["head_kind"],
        other_kinds=[state["kind"] for state in heads],
        calibrations=calibrations,
    )
    whole = QualifiedWholeModel(spec).eval().requires_grad_(False)
    whole.base.load_state_dict(encoder["model"], strict=True)
    assert siglip_identity(whole.base) == siglip_identity(
        positive.encoder
    ), "Whole models do not share the same frozen prefix"
    whole.positive_suffix.load_state_dict(
        SiglipSuffix(positive.encoder.network).state_dict(), strict=True
    )
    whole.positive.load_state_dict(positive.head.state_dict(), strict=True)
    for head, state in zip(whole.heads, heads):
        head.load_state_dict(state["model"], strict=True)
    atomic_save(
        dict(architecture="QualifiedWholeModel", spec=spec, model=whole.state_dict()),
        out / "whole.pt",
    )
    candidate = dict(
        construction_sha256=digest(frozen),
        bundles={
            name: dict(
                file=name + ".pt",
                sha256=digest(out / (name + ".pt")),
                bytes=(out / (name + ".pt")).stat().st_size,
            )
            for name in ("leaf", "whole")
        },
        outputs=dict(
            leaf=[
                "original_logits_0_to_8",
                "compact_positive_H",
                "compact_negative_H",
                "adapted_dino_H",
            ],
            whole=["positive_H", "negative_H", "tomato_genus", "natural_context"],
        ),
        no_rejected_alternative_weights=True,
        cold_olid_used=False,
        ad_independent=False,
    )
    (out / "candidate.json").write_text(json.dumps(candidate, indent=2))
    return candidate


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    args = ap.parse_args()
    print(json.dumps(build(args.campaign)), flush=True)
