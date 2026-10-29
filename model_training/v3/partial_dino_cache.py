"""FP32 fitting-token cache in bounded shared memory; no evaluation leakage."""

from pathlib import Path
import ast
import concurrent.futures
import hashlib
import json
import math
import shutil
import time

import numpy as np
from PIL import Image
import timm
import torch
from torch.nn import functional as F

from app_eval import canon, classifier_letterbox_input
from campaign_pixels import open_pixels
from opened_context_data import read
from partial_cnn_cache import prepare as validate_inputs
from partial_dino_healthy import prefix_cut, prefix_features, prefix_identity
from permissive_encoder import digest
from phase28_roi_confirmation import pixel_key


def original_dino(campaign):
    state = torch.load(
        Path(campaign) / "export_cnn_safe/pytorch_bundle.pt",
        weights_only=True,
        map_location="cpu",
    )
    timm.layers.set_fused_attn(False)
    model = (
        timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=False,
            num_classes=9,
            img_size=224,
        )
        .eval()
        .requires_grad_(False)
    )
    model.load_state_dict(
        {
            key[len("backbone.") :]: value
            for key, value in state["model"].items()
            if key.startswith("backbone.")
        },
        strict=True,
    )
    return model


def input_rows(campaign):
    p = Path(campaign)
    rows = read(p / "phase29_compact/inputs.json")
    for name in ("phase38_context_spatial", "phase46_paper_primary"):
        rows += [
            dict(row, kind="scene", index=row["source_row"])
            for row in read(p / name / "new_inputs.json")
        ]
    return rows


class CanonicalInputs:
    def __init__(self, campaign):
        self.old = open_pixels(Path(campaign) / "phase10_leaf/cnn_inputs")
        self.scene = open_pixels(Path(campaign) / "phase23_siglip/cnn_inputs")

    def __call__(self, row):
        if row["kind"] in ("old", "scene"):
            pixels = self.old if row["kind"] == "old" else self.scene
            value = np.array(pixels[row["index"]], copy=True)[None]
        else:
            with Image.open(row["path"]) as image:
                rgb = np.asarray(image.convert("RGB"))
            value = (
                classifier_letterbox_input
                if row["policy"] == "full"
                else canon.classifier_input
            )(rgb, 224, row["box"])
        assert pixel_key(value) == row["input_sha256"], "DINO canonical input changed"
        return value

    def close(self):
        for value in (self.old, self.scene):
            if hasattr(value, "close"):
                value.close()


def prepare(campaign):
    p = Path(campaign)
    root = p / "phase51_dino_adaptation/prefix"
    root.mkdir(parents=True, exist_ok=True)
    parent = validate_inputs(p)
    rows = input_rows(p)
    fitting = read(p / "phase46_paper_primary/fitting_rows.json")
    tables = np.unique(np.array([r["table"] for r in fitting], np.int64))
    source = ast.parse(Path(__file__).with_name("partial_dino_healthy.py").read_text())
    prefix_ast = [
        ast.dump(node, include_attributes=False)
        for node in source.body
        if getattr(node, "name", None) in ("prefix_cut", "prefix_features")
    ]
    protocol = dict(
        parent_input_contract_sha256=digest(parent / "protocol.json"),
        bundle_sha256=digest(p / "export_cnn_safe/pytorch_bundle.pt"),
        fitting_rows_sha256=digest(p / "phase46_paper_primary/fitting_rows.json"),
        fitting_tables_sha256=hashlib.sha256(tables.tobytes()).hexdigest(),
        prefix_source_sha256=hashlib.sha256(
            json.dumps(prefix_ast).encode()
        ).hexdigest(),
        cache_source_sha256=digest(__file__),
        representation="FP32 DINOv2-S tokens before the final two original blocks; no rounding/quantization",
        shape=[len(tables), 257, 384],
        storage="Bounded /dev/shm cache for fitting inputs only; ephemeral across PC reboot",
        cold_olid_used=False,
        ad_independent=False,
    )
    frozen = root / "protocol.json"
    if frozen.exists():
        assert read(frozen) == protocol, "DINO prefix cache identity changed"
    else:
        frozen.write_text(json.dumps(protocol, indent=2))
    if (root / "complete.json").exists():
        complete = read(root / "complete.json")
        assert complete["protocol_sha256"] == digest(frozen)
        assert complete["features_sha256"] == digest(
            root / "features.npy"
        ), "Shared-memory cache lost/changed; do not silently resume against different inputs"
        assert complete["tables_sha256"] == digest(root / "tables.npy")
        return root
    started = time.monotonic()
    required = int(np.prod(protocol["shape"])) * 4
    assert shutil.disk_usage("/dev/shm").free >= required + (
        512 << 20
    ), "Insufficient bounded shared-memory capacity"
    memory = {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    }
    assert memory["MemAvailable"] > required + (
        2 << 30
    ), "Insufficient host memory; preserve existing useful caches/jobs"
    assert torch.cuda.get_device_properties(0).total_memory > required + (4 << 30)
    directory = Path("/dev/shm") / ("agribot-dino-" + digest(frozen)[:16])
    directory.mkdir(exist_ok=True)
    target = directory / "features.npy"
    assert (
        not target.exists()
    ), "Unregistered token cache exists; audit/version it rather than overwrite"
    array = np.lib.format.open_memmap(
        directory / "features.partial.npy",
        mode="w+",
        dtype=np.float32,
        shape=tuple(protocol["shape"]),
    )
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = original_dino(p).cuda()
    old = np.load(p / "phase29_compact/features.npy", mmap_mode="r")
    pixels = CanonicalInputs(p)
    maximum_error, audited = 0.0, 0
    try:
        with concurrent.futures.ThreadPoolExecutor(8) as pool, torch.inference_mode():
            for start in range(0, len(tables), 128):
                indices = tables[start : start + 128]
                batch = (
                    torch.from_numpy(
                        np.concatenate(
                            list(pool.map(pixels, [rows[i] for i in indices])), 0
                        )
                    )
                    .cuda()
                    .permute(0, 3, 1, 2)
                )
                tokens = prefix_features(model, batch)
                assert tokens.shape[1:] == (257, 384) and torch.isfinite(tokens).all()
                array[start : start + len(indices)] = tokens.cpu().numpy()
                mask = indices < len(old)
                if mask.any():
                    restored = model.forward_head(
                        model.norm(model.blocks[prefix_cut(model) :](tokens)),
                        pre_logits=True,
                    )
                    restored = (
                        (F.normalize(restored, dim=1) * math.sqrt(384)).cpu().numpy()
                    )
                    error = float(
                        np.max(np.abs(restored[mask] - old[indices[mask], :384]))
                    )
                    maximum_error = max(maximum_error, error)
                    assert (
                        maximum_error < 0.002
                    ), "DINO prefix does not reconstruct original frozen features"
                    audited += int(mask.sum())
    finally:
        pixels.close()
    array.flush()
    array._mmap.close()
    (directory / "features.partial.npy").replace(target)
    (root / "features.npy").symlink_to(target)
    np.save(root / "tables.npy", tables)
    complete = dict(
        protocol_sha256=digest(frozen),
        features_sha256=digest(target),
        tables_sha256=digest(root / "tables.npy"),
        prefix_state_sha256=prefix_identity(model),
        inputs=len(tables),
        bytes=required,
        original_feature_rows_audited=audited,
        maximum_original_feature_error=maximum_error,
        elapsed_seconds=time.monotonic() - started,
        cold_olid_used=False,
    )
    (root / "complete.json").write_text(json.dumps(complete, indent=2))
    print(json.dumps(complete), flush=True)
    return root
