"""Copied isolated-view SigLIP suffix and exact-identity prefix reuse."""

from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
from PIL import Image
import torch
from torch import nn

from app_eval import classifier_letterbox_input
from feature_cache_reuse import preprocessing_contract, runtime_identity
from opened_context_data import read
from partial_primary_siglip import (
    PartialPrimarySiglip,
    prefix_features,
    prefix_identity,
)
from permissive_encoder import digest
from phase28_roi_confirmation import pixel_key


class PopulationSiglipTail(PartialPrimarySiglip):
    def __init__(self, encoder, mean, std):
        super().__init__(encoder)
        self.head = nn.Sequential(nn.Linear(768, 128), nn.SiLU(), nn.Linear(128, 1))
        self.register_buffer("feature_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("feature_std", torch.tensor(std, dtype=torch.float32))

    def forward(self, prefix):
        value = (self.features(prefix) - self.feature_mean) / self.feature_std
        return self.head(value).ravel()


def freeze(path, value):
    if path.exists():
        assert read(path) == value, ("Frozen identity changed", str(path))
    else:
        path.write_text(json.dumps(value, indent=2))


def prepare_prefix(campaign, model, paths, pooled_features):
    p = Path(campaign)
    out = p / "phase85_population_sft/prefix"
    out.mkdir(parents=True, exist_ok=True)
    descriptors = {
        r["path"]: r for r in read(p / "opened_siglip_feature_cache/inputs.json")
    }
    descriptors.update(
        {r["path"]: r for r in read(p / "phase81_consumed_olid_features/inputs.json")}
    )
    selected = {path: descriptors[path] for path in sorted(paths)}
    for path, record in selected.items():
        assert digest(path) == record["file_sha256"]
    donor = p / "phase67_primary_siglip/prefix"
    donor_protocol = read(donor / "protocol.json")
    donor_complete = read(donor / "complete.json")
    assert donor_complete["protocol_sha256"] == digest(donor / "protocol.json")
    assert donor_complete["inputs_sha256"] == digest(donor / "inputs.json")
    assert donor_protocol["prefix_state_sha256"] == prefix_identity(model.encoder)
    assert donor_protocol["preprocessing"] == preprocessing_contract(
        Path(__file__).parents[2], "siglip2"
    )
    assert donor_protocol["runtime"] == runtime_identity()
    donor_arrays = []
    for shard in donor_complete["shards"]:
        assert digest(donor / shard["file"]) == shard["sha256"]
        donor_arrays.append((shard, np.load(donor / shard["file"], mmap_mode="r")))
    lookup = {}
    donor_inputs = read(donor / "inputs.json")
    for index, record in enumerate(donor_inputs):
        shard, array = next(
            (s, a) for s, a in donor_arrays if s["start"] <= index < s["end"]
        )
        lookup[record["input_sha256"]] = (array, index - shard["start"])
    key_paths = {}
    for path, record in selected.items():
        key_paths.setdefault(record["input_sha256"], path)
    keys = sorted(key_paths)
    missing = sorted(set(keys) - set(lookup))
    known = len(keys) - len(missing)
    protocol = dict(
        prefix_state_sha256=donor_protocol["prefix_state_sha256"],
        donor_complete_sha256=digest(donor / "complete.json"),
        encoder_sha256=digest(p / "phase23_siglip/siglip2_feature_heads/encoder.pt"),
        inputs=selected,
        missing_keys_sha256=hashlib.sha256(json.dumps(missing).encode()).hexdigest(),
        selected_keys_sha256=hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
        shape=[len(missing), 256, 768],
        dtype="float32",
        preprocessing=donor_protocol["preprocessing"],
        runtime=runtime_identity(),
        source_sha256=digest(__file__),
        representation="Original SigLIP before final two blocks; final pooled features cannot substitute for these training inputs",
        old_prefix_inputs_reused=known,
        new_intermediate_inputs_required=len(missing),
        ad_independent=False,
        olid_independent=False,
    )
    freeze(out / "protocol.json", protocol)
    path = out / "missing_tokens.npy"
    progress = out / "progress.json"
    complete = out / "complete.json"
    if complete.exists():
        metadata = read(complete)
        assert metadata["protocol_sha256"] == digest(out / "protocol.json")
        if missing:
            assert digest(path) == metadata["missing_tokens_sha256"]
        values = np.load(path, mmap_mode="r") if missing else None
    else:
        assert shutil.disk_usage(p).free > len(missing) * 256 * 768 * 4 + 1_000_000_000
        saved = (
            read(progress)
            if progress.exists()
            else dict(completed=0, maximum_feature_error=0.0)
        )
        if progress.exists():
            assert saved["protocol_sha256"] == digest(out / "protocol.json")
        values = (
            np.lib.format.open_memmap(
                path,
                mode="r+" if progress.exists() else "w+",
                dtype=np.float32,
                shape=(len(missing), 256, 768),
            )
            if missing
            else None
        )
        maximum = saved["maximum_feature_error"]
        model.eval()
        with torch.inference_mode():
            for start in range(saved["completed"], len(missing), 32):
                part = missing[start : start + 32]
                pixels = []
                for key in part:
                    image_path = key_paths[key]
                    assert digest(image_path) == selected[image_path]["file_sha256"]
                    with Image.open(image_path) as image:
                        tensor = classifier_letterbox_input(
                            np.asarray(image.convert("RGB")), 224
                        )
                    assert pixel_key(tensor) == key
                    pixels.append(tensor)
                tokens = prefix_features(
                    model.encoder, torch.from_numpy(np.concatenate(pixels, 0)).cuda()
                )
                actual = model.features(tokens).cpu().numpy()
                expected = np.stack([pooled_features[key_paths[key]] for key in part])
                maximum = max(maximum, float(np.max(np.abs(actual - expected))))
                assert maximum <= 0.002, ("Prefix/pool normalization mismatch", maximum)
                values[start : start + len(part)] = tokens.cpu().numpy()
                values.flush()
                temporary = progress.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(
                        dict(
                            protocol_sha256=digest(out / "protocol.json"),
                            completed=start + len(part),
                            maximum_feature_error=maximum,
                        )
                    )
                )
                temporary.replace(progress)
                print(
                    json.dumps(
                        dict(
                            stage="missing_population_prefix",
                            completed=start + len(part),
                            missing=len(missing),
                            reused=known,
                            maximum_feature_error=maximum,
                        )
                    ),
                    flush=True,
                )
        metadata = dict(
            protocol_sha256=digest(out / "protocol.json"),
            missing_tokens_sha256=digest(path) if missing else None,
            new_prefix_inputs=len(missing),
            reused_prefix_inputs=known,
            maximum_feature_error=maximum,
            selected_inputs=len(keys),
            ad_independent=False,
            olid_independent=False,
        )
        complete.write_text(json.dumps(metadata, indent=2))
    for index, key in enumerate(missing):
        lookup[key] = (values, index)
    key_index = {key: i for i, key in enumerate(keys)}
    needed = len(keys) * 256 * 768 * 4
    assert (
        needed < torch.cuda.mem_get_info()[0] - 6_000_000_000
    ), "Preserve GPU headroom; do not overallocate the token cache"
    gpu = torch.empty((len(keys), 256, 768), dtype=torch.float32, device="cuda")
    for start in range(0, len(keys), 64):
        part = keys[start : start + 64]
        batch = np.stack([lookup[key][0][lookup[key][1]] for key in part])
        gpu[start : start + len(part)] = torch.from_numpy(batch).cuda()
    return (
        gpu,
        {path: key_index[record["input_sha256"]] for path, record in selected.items()},
        metadata,
    )
