"""Cache full-scene frozen SigLIP features on opened inputs exactly once."""

from pathlib import Path
import concurrent.futures
import hashlib
import json
import time
import numpy as np
from PIL import Image
import timm
import torch

from app_eval import classifier_letterbox_input
from campaign_pixels import open_pixels
from feature_cache_reuse import (
    preprocessing_contract,
    runtime_identity,
    validate_completed_contract,
)
from opened_context_data import read
from permissive_encoder import CanonicalPermissiveEncoder, digest
from phase28_roi_confirmation import pixel_key


def prepare(campaign, data):
    p = Path(campaign)
    source = p / "phase23_siglip"
    directory = source / "siglip2_feature_heads"
    validate_completed_contract(source, directory, "siglip2")
    out = p / "opened_siglip_feature_cache"
    out.mkdir(exist_ok=True)
    paths = sorted(data["photos"])
    identity = dict(
        encoder_sha256=digest(directory / "encoder.pt"),
        source_feature_array_sha256=digest(directory / "features.npy"),
        source_rows_sha256=digest(source / "cnn_inputs/rows.json"),
        paths_sha256=hashlib.sha256(json.dumps(paths).encode()).hexdigest(),
        preprocessing=preprocessing_contract(
            Path(__file__).resolve().parents[2], "siglip2"
        ),
        runtime=runtime_identity(),
        data_identity=data["identity"],
        cold_olid_used=False,
    )
    frozen = out / "protocol.json"
    if frozen.exists():
        assert read(frozen) == identity, "Opened feature cache identity changed"
    else:
        frozen.write_text(json.dumps(identity, indent=2))
    if (out / "complete.json").exists():
        result = read(out / "complete.json")
        assert digest(out / "features.npy") == result["features_sha256"]
        assert digest(out / "inputs.json") == result["inputs_sha256"]
        assert result["protocol_sha256"] == digest(frozen)
        for record in read(out / "inputs.json"):
            assert (
                digest(record["path"]) == record["file_sha256"]
            ), "Opened source image changed"
        return out
    started = time.monotonic()
    rows = read(source / "cnn_inputs/rows.json")
    old_lookup = {r["path"]: i for i, r in enumerate(rows)}
    source_features = np.load(directory / "features.npy", mmap_mode="r")
    pixels = open_pixels(source / "cnn_inputs")
    features = np.zeros((len(paths), 768), np.float32)
    descriptors = []
    tensors = {}
    reused = {}

    def input_row(path):
        record = data["photos"][path]
        sha = digest(path)
        if record.get("file_sha256"):
            assert sha == record["file_sha256"]
        with Image.open(path) as im:
            tensor = classifier_letterbox_input(np.asarray(im.convert("RGB")), 224)
        key = pixel_key(tensor)
        old = old_lookup.get(path)
        if old is not None:
            assert (
                rows[old].get("box") is None
                and rows[old]["input_variant"] == "full_roi"
            )
            assert np.array_equal(
                tensor[0], pixels[old]
            ), "Matching source path has different canonical pixels"
        return (
            dict(path=path, file_sha256=sha, input_sha256=key, source_index=old),
            tensor,
        )

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for i, (record, tensor) in enumerate(pool.map(input_row, paths)):
            descriptors.append(record)
            if record["source_index"] is not None:
                features[i] = source_features[record["source_index"]]
                reused[record["input_sha256"]] = i
            else:
                tensors.setdefault(record["input_sha256"], tensor)
    for i, r in enumerate(descriptors):
        if r["source_index"] is None and r["input_sha256"] in reused:
            features[i] = features[reused[r["input_sha256"]]]
    pending = sorted(set(tensors) - set(reused))
    if hasattr(pixels, "close"):
        pixels.close()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timm.layers.set_fused_attn(False)
    new = {}
    inference_started = time.monotonic()
    if pending:
        state = torch.load(
            directory / "encoder.pt", map_location="cpu", weights_only=True
        )
        model = (
            CanonicalPermissiveEncoder(state["spec"])
            .cuda()
            .eval()
            .requires_grad_(False)
        )
        model.load_state_dict(state["model"], strict=True)
        with torch.inference_mode():
            for start in range(0, len(pending), 64):
                keys = pending[start : start + 64]
                batch = torch.from_numpy(
                    np.concatenate([tensors[k] for k in keys], 0)
                ).cuda()
                values = model(batch).cpu().numpy()
                new.update({key: value for key, value in zip(keys, values)})
        del model
        torch.cuda.empty_cache()
    for i, r in enumerate(descriptors):
        if r["input_sha256"] in new:
            features[i] = new[r["input_sha256"]]
    assert np.isfinite(features).all() and (np.linalg.norm(features, axis=1) > 1).all()
    np.save(out / "features.npy", features)
    (out / "inputs.json").write_text(json.dumps(descriptors, indent=2))
    report = dict(
        protocol_sha256=digest(frozen),
        features_sha256=digest(out / "features.npy"),
        inputs_sha256=digest(out / "inputs.json"),
        inputs=len(paths),
        reused_inputs=sum(r["input_sha256"] in reused for r in descriptors),
        new_unique_encoder_inputs=len(pending),
        inference_seconds=time.monotonic() - inference_started,
        total_seconds=time.monotonic() - started,
        cold_olid_used=False,
    )
    (out / "complete.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return out
