"""Licensed SigLIP2 + original Meta DINOv2-L image representation fusion."""

from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from torch import nn

from campaign_pixels import open_pixels
from dino_large_encoder import DinoLargeEncoder
from permissive_encoder import CanonicalPermissiveEncoder, digest
from training_support import atomic_save


class HybridFieldEncoder(nn.Module):
    def __init__(self, spec):
        super().__init__()
        assert spec["family"] == "hybrid_field" and spec["license"] == "apache-2.0"
        assert spec["dimension"] == 1792 and spec["restricted_teacher_used"] is False
        self.siglip = CanonicalPermissiveEncoder(spec["siglip"])
        self.dino = DinoLargeEncoder(spec["dino"])

    def forward(self, nhwc):
        return torch.cat([self.siglip(nhwc), self.dino(nhwc)], 1)


def extract(root, out, ids, family):
    root, out = Path(root), Path(out)
    assert family == "hybrid_field"
    links = json.loads((root / "source_links.json").read_text())
    directories = [Path(links[k]) for k in ["siglip_features", "dino_features"]]
    identities = [
        json.loads((d / "feature_identity.json").read_text()) for d in directories
    ]
    rows_sha = digest(root / "cnn_inputs/rows.json")
    assert all(
        i["rows_sha256"] == rows_sha and i["restricted_teacher_used"] is False
        for i in identities
    )
    assert identities[0]["ids_sha256"] == identities[1]["ids_sha256"]
    assert identities[0]["ids_sha256"] == hashlib.sha256(ids.tobytes()).hexdigest()
    assert (
        identities[0]["family"] == "siglip2"
        and identities[1]["family"] == "dinov2_large"
    )
    assert all(
        json.loads((d / "restored_encoder_parity.json").read_text())["passed"]
        for d in directories
    )
    assert all(
        json.loads((d / "input_contract_parity.json").read_text())["passed"]
        for d in directories
    )
    for d, identity in zip(directories, identities):
        assert digest(d / "encoder.pt") == identity["encoder_sha256"]
    cache, meta = out / "features.npy", out / "feature_identity.json"
    if cache.exists():
        from feature_cache_reuse import validate_completed_contract

        identity = json.loads(meta.read_text())
        assert identity["rows_sha256"] == rows_sha
        assert identity["ids_sha256"] == identities[0]["ids_sha256"]
        assert identity["source_identities"] == identities
        assert identity["encoder_sha256"] == digest(out / "encoder.pt")
        assert json.loads((out / "restored_encoder_parity.json").read_text())["passed"]
        assert json.loads((out / "input_contract_parity.json").read_text())["passed"]
        validate_completed_contract(root, out, family)
        return np.load(cache), identity
    # Feature identities alone do not pin the component arrays or input shards.
    # Validate both completed caches before constructing a new representation.
    from feature_cache_reuse import validate_completed_contract

    for directory, component in zip(directories, ("siglip2", "dinov2_large")):
        validate_completed_contract(directory.parent, directory, component)
    source_states = [
        torch.load(d / "encoder.pt", weights_only=True, map_location="cpu")
        for d in directories
    ]
    spec = dict(
        family=family,
        dimension=1792,
        internal_side=224,
        license="apache-2.0",
        siglip=source_states[0]["spec"],
        dino=source_states[1]["spec"],
        restricted_teacher_used=False,
    )
    encoder = HybridFieldEncoder(spec).cuda().eval().requires_grad_(False)
    encoder.siglip.load_state_dict(source_states[0]["model"], strict=True)
    encoder.dino.load_state_dict(source_states[1]["model"], strict=True)
    del source_states
    features = np.concatenate([np.load(d / "features.npy") for d in directories], 1)
    assert features.shape[1] == 1792 and np.isfinite(features).all()
    checkpoint = out / "encoder.pt"
    if not checkpoint.exists():
        atomic_save(
            dict(
                spec=spec, model={k: v.cpu() for k, v in encoder.state_dict().items()}
            ),
            checkpoint,
        )
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    assert state["spec"] == spec
    restored = HybridFieldEncoder(spec).cuda().eval().requires_grad_(False)
    restored.load_state_dict(state["model"], strict=True)
    del state, encoder
    pixels = open_pixels(root / "cnn_inputs")
    probe = ids[np.linspace(0, len(ids) - 1, 54, dtype=int)]
    with torch.inference_mode():
        actual = (
            restored(torch.from_numpy(np.array(pixels[probe], copy=True)).cuda())
            .cpu()
            .numpy()
        )
    error = float(np.max(np.abs(actual - features[probe])))
    assert error < 0.002, (
        "Hybrid graph differs from cached source representations",
        error,
    )
    identity = dict(
        family=family,
        rows_sha256=rows_sha,
        ids_sha256=identities[0]["ids_sha256"],
        encoder_sha256=digest(checkpoint),
        source_identities=identities,
        restricted_teacher_used=False,
    )
    meta = out / "feature_identity.json"
    if meta.exists():
        assert json.loads(meta.read_text()) == identity
    else:
        meta.write_text(json.dumps(identity, indent=2))
    np.save(out / "features.npy", features)
    (out / "restored_encoder_parity.json").write_text(
        json.dumps(
            dict(passed=True, cases=len(probe), max_abs_feature_error=error), indent=2
        )
    )
    (out / "input_contract_parity.json").write_text(
        json.dumps(
            dict(
                passed=True,
                cases=14,
                composition="Both component RGB/native contracts passed independently; combined graph compared with cached real-image features",
                restricted_teacher_used=False,
            ),
            indent=2,
        )
    )
    from feature_cache_reuse import record_completed_contract

    record_completed_contract(root, out, family)
    return features, identity
