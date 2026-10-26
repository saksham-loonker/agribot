"""Explicitly audited spatial-cache registration and strict subsequent reuse."""

from pathlib import Path
import json
import math
import numpy as np
import torch
from torch.nn import functional as F

from compact_healthy_model import primary
from feature_cache_reuse import preprocessing_contract, runtime_identity
from permissive_encoder import digest


def validate(campaign, audit=False):
    p = Path(campaign)
    root = p / "phase37_spatial"
    old = p / "phase29_compact"
    complete = json.loads((root / "complete.json").read_text())
    protocol = json.loads((root / "cache_protocol.json").read_text())
    repo = Path(__file__).resolve().parents[2]
    source = p / "phase37_prepare_source"
    canonical = preprocessing_contract(repo, "original_cnn_spatial")
    assert canonical == preprocessing_contract(source, "original_cnn_spatial")
    for relative in (
        "model_training/v3/compact_healthy_model.py",
        "model_training/v3/campaign_cnn_export.py",
    ):
        assert (repo / relative).read_text() == (
            source / relative
        ).read_text(), "Frozen CNN feature implementation changed"
    assert protocol["primary_bundle_sha256"] == digest(
        p / "export_cnn_safe/pytorch_bundle.pt"
    )
    assert protocol["input_descriptors_sha256"] == digest(old / "inputs.json")
    assert protocol["fitting_rows_sha256"] == digest(old / "fitting_rows.json")
    assert complete["protocol_sha256"] == digest(root / "cache_protocol.json")
    assert complete["maps_sha256"] == digest(root / "maps.npy")
    contract = dict(
        maps_sha256=complete["maps_sha256"],
        protocol_sha256=complete["protocol_sha256"],
        preprocessing=canonical,
        runtime=runtime_identity(),
        preparation_source_manifest_sha256=digest(source / "source_manifest.json"),
        primary_bundle_sha256=protocol["primary_bundle_sha256"],
        fitting_rows_sha256=protocol["fitting_rows_sha256"],
        input_descriptors_sha256=protocol["input_descriptors_sha256"],
    )
    path = root / "cache_contract.json"
    if path.exists():
        assert (
            json.loads(path.read_text()) == contract
        ), "Spatial cache identity changed"
        assert json.loads((root / "pooled_feature_parity.json").read_text())["passed"]
        return contract
    assert (
        audit
    ), "Spatial cache requires explicit full-row pooled-feature audit before registration"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, holder = primary(p)
    cnn = holder.primary.cnn.cuda().eval().requires_grad_(False)
    maps = np.load(root / "maps.npy", mmap_mode="r")
    features = np.load(old / "features.npy", mmap_mode="r")
    assert maps.shape == (len(features), 576, 7, 7) and maps.dtype == np.float32
    error = 0.0
    with torch.inference_mode():
        for start in range(0, len(maps), 512):
            x = torch.from_numpy(np.array(maps[start : start + 512], copy=True)).cuda()
            actual = F.normalize(
                cnn.forward_head(x, pre_logits=True), dim=1
            ) * math.sqrt(1024)
            error = max(
                error,
                float(
                    np.max(
                        np.abs(
                            actual.cpu().numpy() - features[start : start + 512, 384:]
                        )
                    )
                ),
            )
    assert error < 0.002, ("Spatial pooling changed the deployed CNN features", error)
    (root / "pooled_feature_parity.json").write_text(
        json.dumps(
            dict(
                passed=True,
                rows=len(maps),
                maximum_error=error,
                no_dino_inference=True,
                original_cnn_weights_frozen=True,
            ),
            indent=2,
        )
    )
    path.write_text(json.dumps(contract, indent=2))
    return contract
