"""Pre-Apple DINO/CNN logits for admitted human-labelled training families.

No restricted teacher outputs are used. Internal validation rows may be cached
for later evaluation but are explicitly excluded from combiner fitting.
"""

from pathlib import Path
import concurrent.futures
import json

import numpy as np
from PIL import Image
import timm
import torch

from app_eval import classifier_letterbox_input
from permissive_encoder import digest
from phase13_context_model import restore


def extract(campaign, root):
    p, root = Path(campaign), Path(root)
    records = json.loads((root / "admission_manifest.json").read_text())
    verified = root / "fusion_refined_selection_validation"
    result = json.loads((verified / "result.json").read_text())
    assert result["checks"]["development_graph_matches"]
    assert result["restricted_teacher_used"] is False
    bundle = verified / "pytorch_bundle.pt"
    assert digest(bundle) == result["bundle_sha256"]
    out = root / "admitted_legacy_features"
    out.mkdir(exist_ok=True)
    identity = dict(
        bundle_sha256=digest(bundle),
        admission_sha256=digest(root / "admission_manifest.json"),
        columns="Four pre-Apple DINO/CNN calibrated logits; SigLIP2 output discarded",
        restricted_teacher_used=False,
    )
    meta = out / "input_identity.json"
    if meta.exists():
        assert json.loads(meta.read_text()) == identity
    cache = out / "calibrated_logits.npy"
    if cache.exists():
        z = np.load(cache)
        assert z.shape == (len(records), 4) and np.isfinite(z).all()
        return z
    meta.write_text(json.dumps(identity, indent=2))
    timm.layers.set_fused_attn(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = restore(bundle).cuda().eval()

    def prepare(record):
        assert digest(record["path"]) == record["file_sha256"]
        with Image.open(record["path"]) as im:
            return classifier_letterbox_input(np.asarray(im.convert("RGB")), 224)

    z = np.empty((len(records), 4), np.float32)
    with torch.inference_mode(), concurrent.futures.ThreadPoolExecutor(8) as pool:
        for start in range(0, len(records), 32):
            part = records[start : start + 32]
            x = torch.from_numpy(
                np.concatenate(list(pool.map(prepare, part)), 0)
            ).cuda()
            z[start : start + len(part)] = (
                model.calibrated_logits(x)[:, :4].cpu().numpy()
            )
    assert np.isfinite(z).all()
    np.save(cache, z)
    return z
