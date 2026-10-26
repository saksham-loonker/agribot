"""Pinned Apache-2.0 SigLIP2 image features; no restricted teacher dependency."""

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
from training_support import atomic_save

REPOSITORY = "timm/vit_base_patch16_siglip_256.v2_webli"
REVISION = "ac4cc431026674dcac29e34a00d3bd2048390eb3"
SOURCE_SHA256 = "13ab9d238865f89b14005c5e92a099add04cbfab331a64245acab813509cb2b0"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CanonicalPermissiveEncoder(nn.Module):
    def __init__(self, spec):
        super().__init__()
        assert spec["license"] == "apache-2.0"
        self.network = timm.create_model(spec["model_name"], pretrained=False)
        self.dimension = spec["dimension"]
        self.side = spec["internal_side"]
        for name, value in [
            ("canonical_mean", [0.485, 0.456, 0.406]),
            ("canonical_std", [0.229, 0.224, 0.225]),
            ("native_mean", spec["native_mean"]),
            ("native_std", spec["native_std"]),
        ]:
            self.register_buffer(
                name, torch.tensor(value, dtype=torch.float32).view(1, 3, 1, 1)
            )

    def forward(self, nhwc):
        rgb = nhwc.permute(0, 3, 1, 2) * self.canonical_std + self.canonical_mean
        rgb = F.interpolate(
            rgb, size=(self.side, self.side), mode="bilinear", align_corners=False
        )
        x = (rgb - self.native_mean) / self.native_std
        return F.normalize(self.network(x), dim=1) * math.sqrt(self.dimension)


def extract(root, out, ids, family):
    root, out = Path(root), Path(out)
    assert family == "siglip2"
    source = json.loads((root / "encoder_source.json").read_text())
    assert (
        source["repository"] == REPOSITORY
        and source["revision"] == REVISION
        and source["file_sha256"] == SOURCE_SHA256
        and source["license"] == "apache-2.0"
    )
    assert digest(source["local_file"]) == SOURCE_SHA256
    spec = dict(
        model_name=source["model_name"],
        dimension=source["feature_dimension"],
        internal_side=256,
        native_mean=source["pretrained_cfg"]["mean"],
        native_std=source["pretrained_cfg"]["std"],
        license="apache-2.0",
        source={
            k: source[k] for k in ["repository", "revision", "file_sha256", "license"]
        },
        input_adaptation="ImageNet-normalized NHWC224 -> RGB01 -> bilinear256 -> native mean/std -> SigLIP2 -> L2/RMS normalization",
        restricted_teacher_used=False,
    )
    assert (
        spec["native_mean"] == [0.5, 0.5, 0.5]
        and spec["native_std"] == [0.5, 0.5, 0.5]
        and spec["dimension"] == 768
    )
    checkpoint = out / "encoder.pt"
    if not checkpoint.exists():
        pretrained = (
            timm.create_model(
                source["model_name"],
                pretrained=True,
                pretrained_cfg_overlay={"file": source["local_file"]},
            )
            .cuda()
            .eval()
            .requires_grad_(False)
        )
        encoder = CanonicalPermissiveEncoder(spec).cuda().eval().requires_grad_(False)
        encoder.network.load_state_dict(pretrained.state_dict(), strict=True)
        with torch.inference_mode():
            colors = torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [0.5, 0.5, 0.5],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.1, 0.4, 0.8],
                ],
                device="cuda",
            )
            rgb = colors[:, None, None, :].expand(-1, 224, 224, -1)
            canonical = (
                rgb - torch.tensor([0.485, 0.456, 0.406], device="cuda")
            ) / torch.tensor([0.229, 0.224, 0.225], device="cuda")
            native = colors[:, :, None, None].expand(-1, -1, 256, 256)
            native = (native - 0.5) / 0.5
            expected = F.normalize(pretrained(native), dim=1) * math.sqrt(768)
            error = float((encoder(canonical) - expected).abs().max())
            assert error < 0.002, ("Canonical/native RGB contract mismatch", error)
        (out / "input_contract_parity.json").write_text(
            json.dumps(
                dict(
                    passed=True,
                    cases=7,
                    max_abs_feature_error=error,
                    license="apache-2.0",
                    restricted_teacher_used=False,
                ),
                indent=2,
            )
        )
        atomic_save(
            dict(
                spec=spec, model={k: v.cpu() for k, v in encoder.state_dict().items()}
            ),
            checkpoint,
        )
        del pretrained, encoder
    assert json.loads((out / "input_contract_parity.json").read_text())["passed"]
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    assert state["spec"] == spec
    identity = dict(
        rows_sha256=digest(root / "cnn_inputs/rows.json"),
        encoder_sha256=digest(checkpoint),
        source=spec["source"],
        family=family,
        feature_scope=spec["input_adaptation"],
        ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
        encoder_weights_loaded_strictly=True,
        restricted_teacher_used=False,
    )
    cache, meta = out / "features.npy", out / "feature_identity.json"
    if cache.exists():
        assert (
            json.loads(meta.read_text()) == identity
            and json.loads((out / "restored_encoder_parity.json").read_text())["passed"]
        )
        return np.load(cache), identity
    encoder = CanonicalPermissiveEncoder(spec).cuda().eval().requires_grad_(False)
    encoder.load_state_dict(state["model"], strict=True)
    pixels = open_pixels(root / "cnn_inputs")
    from feature_cache_reuse import seed_features

    features, missing_ids = seed_features(root, out, ids, 768, family)
    with torch.inference_mode():
        for start in range(0, len(missing_ids), 128):
            part = missing_ids[start : start + 128]
            x = torch.from_numpy(np.array(pixels[part], copy=True)).cuda()
            z = encoder(x)
            assert torch.isfinite(z).all()
            features[part] = z.cpu().numpy()
            if start % 4096 == 0:
                print(
                    json.dumps(
                        dict(
                            stage="siglip2_features",
                            complete=start + len(part),
                            total=len(missing_ids),
                            reused=len(ids) - len(missing_ids),
                        )
                    ),
                    flush=True,
                )
    del encoder
    restored = CanonicalPermissiveEncoder(spec).cuda().eval().requires_grad_(False)
    restored.load_state_dict(state["model"], strict=True)
    probe = ids[np.linspace(0, len(ids) - 1, 54, dtype=int)]
    with torch.inference_mode():
        error = float(
            np.max(
                np.abs(
                    restored(
                        torch.from_numpy(np.array(pixels[probe], copy=True)).cuda()
                    )
                    .cpu()
                    .numpy()
                    - features[probe]
                )
            )
        )
    assert error < 0.002, ("Restored SigLIP2/cache mismatch", error)
    (out / "restored_encoder_parity.json").write_text(
        json.dumps(dict(passed=True, cases=54, max_abs_feature_error=error), indent=2)
    )
    np.save(cache, features)
    meta.write_text(json.dumps(identity, indent=2))
    return features, identity
