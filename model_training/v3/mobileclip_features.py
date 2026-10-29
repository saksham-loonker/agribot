"""Pinned MobileCLIP image features over the existing canonical tensor contract.

The 224 RGB tensor is converted from ImageNet normalization to RGB [0,1], then
resized internally to the checkpoint's 256 pixels with exportable bilinear
interpolation. This explicitly differs from its official PIL bicubic pipeline;
training and deployment must both use this recorded adaptation.
"""

from pathlib import Path
import hashlib
import json
import math

import numpy as np
import timm
import torch
from torch import nn
from torch.nn import functional as F

from campaign_pixels import open_pixels


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CanonicalMobileClip(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.image_model = timm.create_model(spec["model_name"], pretrained=False)
        self.dimension = spec["dimension"]
        self.side = spec["internal_side"]
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, nhwc):
        rgb = nhwc.permute(0, 3, 1, 2) * self.std + self.mean
        rgb = F.interpolate(
            rgb, size=(self.side, self.side), mode="bilinear", align_corners=False
        )
        return F.normalize(self.image_model(rgb), dim=1) * math.sqrt(self.dimension)


def extract(root, out, ids, family):
    root, out = Path(root), Path(out)
    source = json.loads((root / "mobileclip_source.json").read_text())
    assert source["revision"] == "577a2e992da2c7ba7cadc2a490bcfadfb2d69bce"
    assert digest(source["local_file"]) == source["file_sha256"]
    assert source["pretrained_cfg"]["mean"] == [0.0, 0.0, 0.0]
    assert source["pretrained_cfg"]["std"] == [1.0, 1.0, 1.0]
    spec = dict(
        model_name=source["model_name"],
        dimension=512,
        internal_side=256,
        input_adaptation="canonical NHWC224 -> RGB01 -> bilinear256 align_corners=False -> MobileCLIP -> L2 * sqrt512",
        source={
            k: v for k, v in source.items() if k not in ("local_file", "pretrained_cfg")
        },
    )
    checkpoint = out / "encoder.pt"
    if not checkpoint.exists():
        pretrained = timm.create_model(
            source["model_name"],
            pretrained=True,
            pretrained_cfg_overlay={"file": source["local_file"]},
        ).eval()
        encoder = CanonicalMobileClip(spec).eval()
        encoder.image_model.load_state_dict(pretrained.state_dict(), strict=True)
        torch.save(dict(spec=spec, model=encoder.state_dict()), checkpoint)
        del pretrained, encoder
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    assert state["spec"] == spec
    identity = dict(
        rows_sha256=digest(root / "cnn_inputs/rows.json"),
        encoder_sha256=digest(checkpoint),
        source=spec["source"],
        feature_scope=spec["input_adaptation"],
        family=family,
        encoder_weights_loaded_strictly=True,
        ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
    )
    cache, meta = out / "features.npy", out / "feature_identity.json"
    if family == "mobileclip_dino":
        identity["original_features_sha256"] = digest(
            root / "leaf_feature_heads/features.npy"
        )
        identity["mixed_features"] = (
            "Concatenate MobileCLIP RMS1 features and original DINO L2 * sqrt384"
        )
    if cache.exists():
        assert json.loads(meta.read_text()) == identity
        assert json.loads((out / "restored_encoder_parity.json").read_text())["passed"]
        return np.load(cache), identity
    if family == "mobileclip_dino":
        mobile = np.load(root / "mobileclip_feature_heads/features.npy")
        mobile_id = json.loads(
            (root / "mobileclip_feature_heads/feature_identity.json").read_text()
        )
        assert (
            mobile_id["rows_sha256"] == identity["rows_sha256"]
            and mobile_id["ids_sha256"] == identity["ids_sha256"]
        )
        assert mobile_id["source"] == identity["source"]
        original_id = json.loads(
            (root / "leaf_feature_heads/feature_identity.json").read_text()
        )
        assert (
            original_id["rows_sha256"] == identity["rows_sha256"]
            and original_id["ids_sha256"] == identity["ids_sha256"]
        )
        original = np.load(root / "leaf_feature_heads/features.npy")
        normalized = (
            original
            / np.maximum(np.linalg.norm(original, axis=1, keepdims=True), 1e-12)
            * math.sqrt(384)
        )
        features = np.concatenate([mobile, normalized], axis=1)
    else:
        encoder = CanonicalMobileClip(spec).cuda().eval().requires_grad_(False)
        encoder.load_state_dict(state["model"], strict=True)
        pixels = open_pixels(root / "cnn_inputs")
        total = len(json.loads((root / "cnn_inputs/rows.json").read_text()))
        features = np.zeros((total, spec["dimension"]), np.float32)
        with torch.inference_mode():
            for start in range(0, len(ids), 256):
                part = ids[start : start + 256]
                x = torch.from_numpy(np.array(pixels[part], copy=True)).cuda()
                z = encoder(x)
                assert torch.isfinite(z).all()
                features[part] = z.cpu().numpy()
                if start % 8192 == 0:
                    print(
                        json.dumps(
                            dict(
                                stage="mobileclip_features",
                                complete=start + len(part),
                                total=len(ids),
                            )
                        ),
                        flush=True,
                    )
        del encoder
        restored = CanonicalMobileClip(spec).cuda().eval().requires_grad_(False)
        restored.load_state_dict(state["model"], strict=True)
        probe_ids = ids[np.linspace(0, len(ids) - 1, 54, dtype=int)]
        with torch.inference_mode():
            x = torch.from_numpy(np.array(pixels[probe_ids], copy=True)).cuda()
            error = float(
                np.max(np.abs(restored(x).cpu().numpy() - features[probe_ids]))
            )
        assert error < 0.002, (
            "Restored pretrained encoder/cache alignment failed",
            error,
        )
        del restored
    parity = (
        dict(passed=True, cases=54, max_abs_feature_error=error)
        if family == "mobileclip"
        else dict(
            passed=True,
            inherited_mobileclip_parity=json.loads(
                (
                    root / "mobileclip_feature_heads/restored_encoder_parity.json"
                ).read_text()
            ),
        )
    )
    (out / "restored_encoder_parity.json").write_text(json.dumps(parity, indent=2))
    np.save(cache, features)
    meta.write_text(json.dumps(identity, indent=2))
    return features, identity
