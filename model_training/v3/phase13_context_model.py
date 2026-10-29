"""Apache-2.0 SigLIP2 with pre-Apple DINO/CNN experts; no restricted teacher."""

from pathlib import Path
import json

import torch
from torch import nn

from campaign_adapter import prefix
from permissive_scene import PermissiveScene, load as load_permissive
from permissive_encoder import digest
from phase2_export import DomainClassifier
from phase7_feature_heads import BinaryHead
from phase7_verifier import CanonicalVerifier
from phase9_scene_sft import SceneExpert


class PermissiveContextFusion(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.domain = DomainClassifier(spec["domain"])
        self.scene_depth = spec["scene_depth"]
        self.scene = SceneExpert(self.domain.backbone, self.scene_depth)
        self.species = CanonicalVerifier(spec["species"], spec["species_calibration"])
        self.permissive = PermissiveScene(spec["permissive"])
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

    def calibrated_logits(self, nhwc, permissive_features=None):
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
            (
                self.permissive(nhwc).ravel()
                if permissive_features is None
                else self.permissive.head(permissive_features)
                * self.permissive.calibration[0]
                + self.permissive.calibration[1]
            ),
        ]
        return torch.stack(z, 1)

    def components(self, nhwc):
        z = self.calibrated_logits(nhwc)
        healthy = z.clamp(-self.input_clip, self.input_clip) @ self.weights + self.bias
        return z, torch.stack([healthy, z[:, 3]], 1)

    def forward(self, nhwc):
        return self.components(nhwc)[1]


def build(campaign, candidate, model_root):
    p = Path(campaign)
    root = p / "phase6_roi"
    identity = candidate["fusion"]["input_identity"]
    old = identity["legacy_input_identity"]
    checks = [
        (root / "export_initial/pytorch_bundle.pt", old["feature_model_sha256"]),
        (root / "scene_sft_d12_focal/best.pt", old["scene_cnn_sha256"]),
        (root / "scene_sft_d12_focal/result.json", old["scene_cnn_calibration_sha256"]),
        (root / "species/best.pt", old["species_cnn_sha256"]),
        (root / "species/result.json", old["species_cnn_calibration_sha256"]),
    ]
    for i, kind in enumerate(["linear_adapted", "mlp_both"]):
        checks.extend(
            [
                (
                    root / "feature_heads/healthy_scene" / kind / "best.pt",
                    old["head_hashes"][i],
                ),
                (
                    root / "feature_heads/healthy_scene" / kind / "result.json",
                    old["head_calibration_hashes"][i],
                ),
            ]
        )
    for path, expected in checks:
        assert digest(path) == expected, ("Legacy source changed", str(path))
    domain = torch.load(
        root / "export_initial/pytorch_bundle.pt", weights_only=True, map_location="cpu"
    )
    scene = torch.load(
        root / "scene_sft_d12_focal/best.pt", weights_only=True, map_location="cpu"
    )
    species = torch.load(
        root / "species/best.pt", weights_only=True, map_location="cpu"
    )
    permissive_fit = json.loads(
        (Path(model_root) / "siglip2_feature_heads/ranked.json").read_text()
    )[0]
    permissive_spec, verifier = load_permissive(
        Path(permissive_fit["checkpoint"]).parent
    )
    directory = Path(permissive_fit["checkpoint"]).parent
    for path, expected in [
        (directory / "best.pt", identity["mobile_head_sha256"]),
        (directory / "result.json", identity["mobile_calibration_sha256"]),
        (directory.parent / "encoder.pt", identity["mobile_encoder_sha256"]),
    ]:
        assert digest(path) == expected, ("SigLIP2 source changed", str(path))
    assert permissive_spec["restricted_teacher_used"] is False
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
        architecture="PermissiveContextFusion",
        domain={k: v for k, v in domain.items() if k != "model"},
        scene_depth=scene["depth"],
        scene_calibration=json.loads(
            (root / "scene_sft_d12_focal/result.json").read_text()
        )["calibration"],
        species={k: species[k] for k in ["architecture", "mean", "std"]},
        species_calibration=json.loads((root / "species/result.json").read_text())[
            "calibration"
        ],
        permissive=permissive_spec,
        restricted_teacher_used=False,
        head_kinds=kinds,
        head_calibration=calibration,
        fit=candidate["fusion"],
        policy=candidate["policy"],
        scope="Image-level primary Healthy tomato evidence; original nine-class classifier is a separate unchanged model",
    )
    model = PermissiveContextFusion(spec).eval().requires_grad_(False)
    model.domain.load_state_dict(domain["model"], strict=True)
    model.scene.load_state_dict(scene["model"], strict=True)
    model.species.network.load_state_dict(species["model"], strict=True)
    model.permissive.load_state_dict(verifier.state_dict(), strict=True)
    for module, state in zip(model.heads, heads):
        module.load_state_dict(state["model"], strict=True)
    return spec, model


def restore(path):
    state = torch.load(path, weights_only=True, map_location="cpu")
    assert state["architecture"] == "PermissiveContextFusion"
    assert state["restricted_teacher_used"] is False
    model = PermissiveContextFusion(state).eval().requires_grad_(False)
    model.load_state_dict(state["model"], strict=True)
    return model
