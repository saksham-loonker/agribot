"""Original Meta Apache-2.0 DINOv2-L features on canonical 224 RGB tensors.

The timm port's conflicting license metadata is not used as a weight source.
The publisher's 518 positional grid is resampled by timm at checkpoint load;
this explicit 224 adaptation is shared by fitting and executable inference.
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
from permissive_encoder import digest
from training_support import atomic_save

SOURCE_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"
)


SOURCE_SHA256 = "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428"


class DinoLargeEncoder(nn.Module):
    def __init__(self, spec):
        super().__init__()
        assert spec["license"] == "apache-2.0" and spec["family"] == "dinov2_large"
        assert spec["dimension"] == 1024 and spec["internal_side"] == 224
        self.network = timm.create_model(
            spec["model_name"], pretrained=False, img_size=224, num_classes=0
        )

    def forward(self, nhwc):
        return F.normalize(self.network(nhwc.permute(0, 3, 1, 2)), dim=1) * math.sqrt(
            1024
        )


def extract(root, out, ids, family):
    root, out = Path(root), Path(out)
    assert family == "dinov2_large"
    source = json.loads((root / "encoder_source.json").read_text())
    assert (
        source["url"] == SOURCE_URL
        and source["repository"] == "facebookresearch/dinov2"
    )
    assert (
        source["license"] == "apache-2.0" and source["restricted_teacher_used"] is False
    )
    assert digest(source["local_file"]) == source["file_sha256"] == SOURCE_SHA256
    spec = dict(
        family=family,
        model_name=source["model_name"],
        dimension=1024,
        internal_side=224,
        native_mean=[0.485, 0.456, 0.406],
        native_std=[0.229, 0.224, 0.225],
        license="apache-2.0",
        source={k: source[k] for k in ["repository", "url", "file_sha256", "license"]},
        input_adaptation="ImageNet-normalized RGB NHWC224 -> NCHW224 -> native DINOv2-L; timm positional-grid resampling at source load; L2/RMS features",
        restricted_teacher_used=False,
    )
    checkpoint = out / "encoder.pt"
    if not checkpoint.exists():
        pretrained = (
            timm.create_model(
                source["model_name"],
                pretrained=True,
                img_size=224,
                num_classes=0,
                pretrained_cfg_overlay=dict(
                    file=source["local_file"], hf_hub_id=None, license="apache-2.0"
                ),
            )
            .cuda()
            .eval()
            .requires_grad_(False)
        )
        encoder = DinoLargeEncoder(spec).cuda().eval().requires_grad_(False)
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
                rgb - torch.tensor(spec["native_mean"], device="cuda")
            ) / torch.tensor(spec["native_std"], device="cuda")
            native = canonical.permute(0, 3, 1, 2)
            expected = F.normalize(pretrained(native), dim=1) * math.sqrt(1024)
            error = float((encoder(canonical) - expected).abs().max())
        assert error < 0.002, ("Native canonical input contract mismatch", error)
        (out / "input_contract_parity.json").write_text(
            json.dumps(
                dict(
                    passed=True,
                    cases=7,
                    max_abs_feature_error=error,
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
        del encoder, pretrained
        torch.cuda.empty_cache()
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    assert state["spec"] == spec
    assert json.loads((out / "input_contract_parity.json").read_text())["passed"]
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
        from feature_cache_reuse import validate_completed_contract

        validate_completed_contract(root, out, family)
        assert json.loads(meta.read_text()) == identity
        assert json.loads((out / "restored_encoder_parity.json").read_text())["passed"]
        return np.load(cache), identity
    encoder = DinoLargeEncoder(spec).cuda().eval().requires_grad_(False)
    encoder.load_state_dict(state["model"], strict=True)
    pixels = open_pixels(root / "cnn_inputs")
    from feature_cache_reuse import seed_features

    features, missing_ids = seed_features(root, out, ids, 1024, family)
    with torch.inference_mode():
        for start in range(0, len(missing_ids), 64):
            part = missing_ids[start : start + 64]
            x = torch.from_numpy(np.array(pixels[part], copy=True)).cuda()
            z = encoder(x)
            assert torch.isfinite(z).all()
            features[part] = z.cpu().numpy()
            if start % 2048 == 0:
                print(
                    json.dumps(
                        dict(
                            stage="dino_large_features",
                            complete=start + len(part),
                            total=len(missing_ids),
                            reused=len(ids) - len(missing_ids),
                        )
                    ),
                    flush=True,
                )
    del encoder
    restored = DinoLargeEncoder(spec).cuda().eval().requires_grad_(False)
    restored.load_state_dict(state["model"], strict=True)
    probe = ids[np.linspace(0, len(ids) - 1, 54, dtype=int)]
    with torch.inference_mode():
        actual = (
            restored(torch.from_numpy(np.array(pixels[probe], copy=True)).cuda())
            .cpu()
            .numpy()
        )
    error = float(np.max(np.abs(actual - features[probe])))
    assert error < 0.002, ("Restored native feature parity failed", error)
    (out / "restored_encoder_parity.json").write_text(
        json.dumps(
            dict(passed=True, cases=len(probe), max_abs_feature_error=error), indent=2
        )
    )
    np.save(cache, features)
    meta.write_text(json.dumps(identity, indent=2))
    from feature_cache_reuse import record_completed_contract

    record_completed_contract(root, out, family)
    return features, identity
