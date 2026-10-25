"""Self-contained five-expert image context graph; no disease logits are changed."""

from pathlib import Path
import json

import torch
from torch import nn

from campaign_adapter import prefix
from mobileclip_scene import MobileClipScene, load as load_mobile
from phase2_export import DomainClassifier
from phase7_feature_heads import BinaryHead
from phase7_verifier import CanonicalVerifier
from phase9_scene_sft import SceneExpert


class ContextFusion(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.domain = DomainClassifier(spec["domain"])
        self.scene_depth = spec["scene_depth"]
        self.scene = SceneExpert(self.domain.backbone, self.scene_depth)
        self.species = CanonicalVerifier(spec["species"], spec["species_calibration"])
        self.mobile = MobileClipScene(spec["mobile"])
        self.heads = nn.ModuleList([BinaryHead(kind) for kind in spec["head_kinds"]])
        self.register_buffer(
            "head_calibration",
            torch.tensor(spec["head_calibration"], dtype=torch.float32),
        )
        self.register_buffer(
            "scene_calibration",
            torch.tensor(spec["scene_calibration"], dtype=torch.float32),
        )
        self.register_buffer(
            "weights", torch.tensor(spec["fit"]["weights"], dtype=torch.float32)
        )
        self.register_buffer(
            "bias", torch.tensor(spec["fit"]["bias"], dtype=torch.float32)
        )
        self.input_clip = spec["fit"]["input_clip"]

    def calibrated_logits(self, nhwc):
        image = nhwc.permute(0, 3, 1, 2)
        tokens = prefix(self.domain.backbone, image, self.domain.depth)
        original = tokens
        for block in self.domain.backbone.blocks[-self.domain.depth :]:
            original = block(original)
        original = self.domain.backbone.norm(original)[:, 0]
        adapted = self.domain.expert.features(tokens)
        features = torch.cat([original, adapted], 1)
        z = [
            head(features) * self.head_calibration[i, 0] + self.head_calibration[i, 1]
            for i, head in enumerate(self.heads)
        ]
        scene_tokens = (
            tokens
            if self.scene_depth == self.domain.depth
            else prefix(self.domain.backbone, image, self.scene_depth)
        )
        z += [
            self.scene(scene_tokens) * self.scene_calibration[0]
            + self.scene_calibration[1],
            self.species.logits(nhwc).ravel(),
            self.mobile(nhwc).ravel(),
        ]
        return torch.stack(z, 1)

    def components(self, nhwc):
        z = self.calibrated_logits(nhwc)
        healthy = z.clamp(-self.input_clip, self.input_clip) @ self.weights + self.bias
        return z, torch.stack([healthy, z[:, 3]], 1)

    def forward(self, nhwc):
        return self.components(nhwc)[1]


def build(campaign, candidate):
    p = Path(campaign)
    root = p / "phase6_roi"
    domain = torch.load(
        root / "export_initial/pytorch_bundle.pt", weights_only=True, map_location="cpu"
    )
    scene = torch.load(
        root / "scene_sft_d12_focal/best.pt", weights_only=True, map_location="cpu"
    )
    species = torch.load(
        root / "species/best.pt", weights_only=True, map_location="cpu"
    )
    mobile_fit = json.loads(
        (p / "phase11_scene/mobileclip_feature_heads/ranked.json").read_text()
    )[0]
    mobile_spec, mobile = load_mobile(Path(mobile_fit["checkpoint"]).parent)
    kinds = ["linear_adapted", "mlp_both"]
    heads = [
        torch.load(
            root / "feature_heads/healthy_scene" / kind / "best.pt",
            weights_only=True,
            map_location="cpu",
        )
        for kind in kinds
    ]
    calibration = [
        json.loads(
            (root / "feature_heads/healthy_scene" / kind / "result.json").read_text()
        )["calibration"]
        for kind in kinds
    ]
    spec = dict(
        architecture="ContextFusion",
        domain={k: v for k, v in domain.items() if k != "model"},
        scene_depth=scene["depth"],
        scene_calibration=json.loads(
            (root / "scene_sft_d12_focal/result.json").read_text()
        )["calibration"],
        species={k: species[k] for k in ["architecture", "mean", "std"]},
        species_calibration=json.loads((root / "species/result.json").read_text())[
            "calibration"
        ],
        mobile=mobile_spec,
        head_kinds=kinds,
        head_calibration=calibration,
        fit=candidate["fusion"],
        policy=candidate["policy"],
        scope="Image-level primary Healthy tomato evidence; original nine-class classifier is a separate unchanged model",
    )
    model = ContextFusion(spec).eval().requires_grad_(False)
    model.domain.load_state_dict(domain["model"], strict=True)
    model.scene.load_state_dict(scene["model"], strict=True)
    model.species.network.load_state_dict(species["model"], strict=True)
    model.mobile.load_state_dict(mobile.state_dict(), strict=True)
    for module, state in zip(model.heads, heads):
        module.load_state_dict(state["model"], strict=True)
    return spec, model


def restore(path):
    state = torch.load(path, weights_only=True, map_location="cpu")
    assert state["architecture"] == "ContextFusion"
    model = ContextFusion(state).eval().requires_grad_(False)
    model.load_state_dict(state["model"], strict=True)
    return model
