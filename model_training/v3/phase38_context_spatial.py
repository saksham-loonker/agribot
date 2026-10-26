"""Add human primary-image field supervision without assigning ROI labels.

The spatial control lacked TOM whole-image fitting examples. Reuse its complete
leaf maps, append only missing canonical scene maps, and retain family splits.
"""

from pathlib import Path
import argparse
import concurrent.futures
import json
import time

import numpy as np
from PIL import Image
import torch

from app_eval import classifier_letterbox_input
from campaign_pixels import open_pixels
from compact_healthy_model import primary
from feature_cache_reuse import validate_completed_contract
from opened_context_data import read
from permissive_encoder import digest
from phase28_roi_confirmation import pixel_key
from spatial_cache import validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    args = ap.parse_args()
    p = Path(args.campaign)
    old = p / "phase29_compact"
    base = p / "phase37_spatial"
    out = p / "phase38_context_spatial"
    out.mkdir(exist_ok=True)
    contract = validate(p)
    source = p / "phase23_siglip"
    validate_completed_contract(source, source / "siglip2_feature_heads", "siglip2")
    rows = read(source / "cnn_inputs/rows.json")
    selected = [
        i
        for i, r in enumerate(rows)
        if r["source"] == "tom2024"
        and r["split"] in ("train", "val")
        and r["health_y"] in (0, 1)
    ]
    assert selected and all(rows[i].get("box") is None for i in selected)
    protocol = dict(
        old_maps_contract=contract,
        source_rows_sha256=digest(source / "cnn_inputs/rows.json"),
        old_inputs_sha256=digest(old / "inputs.json"),
        old_fitting_rows_sha256=digest(old / "fitting_rows.json"),
        added_source="TOM2024 publisher primary whole-image binary targets only; no labels copied onto detected ROIs or local patches",
        input_scope="Exact full-scene canonical letterbox224 already used by the human-only scene corpus",
        source_verification="All fresh canonical inputs must equal the immutable cache; check recorded file hashes where present and record actual hashes for all source files",
        new_dino_inference=0,
        encoder_weights_frozen=True,
        cold_olid_used=False,
        script_sha256=digest(__file__),
    )
    frozen = out / "protocol.json"
    if frozen.exists():
        assert read(frozen) == protocol
    else:
        frozen.write_text(json.dumps(protocol, indent=2))
    if (out / "complete.json").exists():
        print((out / "complete.json").read_text())
        return
    started = time.monotonic()
    pixels = open_pixels(source / "cnn_inputs")
    previous = read(old / "inputs.json")
    lookup = {r["input_sha256"]: i for i, r in enumerate(previous)}
    offset = len(previous)
    additions = []
    descriptors = []
    source_files = {}

    def prepare(i):
        r = rows[i]
        file_sha = digest(r["path"])
        if r.get("file_sha256"):
            assert file_sha == r["file_sha256"]
        with Image.open(r["path"]) as im:
            actual = classifier_letterbox_input(np.asarray(im.convert("RGB")), 224)
        stored = np.asarray(pixels[i])[None]
        assert np.array_equal(
            actual, stored
        ), "Primary scene input differs from cached pixels"
        return i, pixel_key(stored), file_sha

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for i, key, file_sha in pool.map(prepare, selected):
            r = rows[i]
            source_files[r["path"]] = file_sha
            if key not in lookup:
                lookup[key] = offset + len(descriptors)
                descriptors.append(
                    dict(
                        source_row=i,
                        input_sha256=key,
                        path=r["path"],
                        file_sha256=file_sha,
                    )
                )
            additions.append(
                dict(
                    table=lookup[key],
                    id=r["id"],
                    source=r["source"],
                    group=r["group"],
                    split=r["split"],
                    y=int(r["health_y"]),
                    role="primary_whole_scene",
                )
            )
    combined = read(old / "fitting_rows.json") + additions
    classes = {}
    for r in combined:
        classes.setdefault(r["table"], set()).add(r["y"])
    contradictory = {k for k, v in classes.items() if len(v) > 1}
    val_tables = {r["table"] for r in combined if r["split"] == "val"}
    val_groups = {r["group"] for r in combined if r["split"] == "val"}
    retained = []
    quarantines = []
    for r in combined:
        reason = None
        if r["table"] in contradictory:
            reason = "Contradictory human targets on identical canonical input"
        elif r["split"] == "train" and (
            r["table"] in val_tables or r["group"] in val_groups
        ):
            reason = "Training input/family overlaps validation"
        if reason:
            quarantines.append(dict(r, reason=reason))
        else:
            retained.append(r)
    assert not (
        {r["group"] for r in retained if r["split"] == "train"}
        & {r["group"] for r in retained if r["split"] == "val"}
    )
    assert not (
        {r["table"] for r in retained if r["split"] == "train"}
        & {r["table"] for r in retained if r["split"] == "val"}
    )
    (out / "fitting_rows.json").write_text(json.dumps(retained, indent=2))
    (out / "fitting_quarantines.json").write_text(json.dumps(quarantines, indent=2))
    (out / "new_inputs.json").write_text(json.dumps(descriptors, indent=2))
    (out / "source_files.json").write_text(json.dumps(source_files, indent=2))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, holder = primary(p)
    model = holder.primary.cuda().eval().requires_grad_(False)
    maps = np.lib.format.open_memmap(
        out / "maps.partial.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(descriptors), 576, 7, 7),
    )
    with torch.inference_mode():
        for start in range(0, len(descriptors), 256):
            ids = [r["source_row"] for r in descriptors[start : start + 256]]
            batch = np.array(pixels[ids], copy=True)
            for r, tensor in zip(descriptors[start : start + 256], batch):
                assert pixel_key(tensor) == r["input_sha256"]
            x = torch.from_numpy(batch).cuda().permute(0, 3, 1, 2)
            rgb = (
                x * model.canonical_std + model.canonical_mean - model.cnn_mean
            ) / model.cnn_std
            maps[start : start + len(ids)] = (
                model.cnn.forward_features(rgb).cpu().numpy()
            )
    maps.flush()
    maps._mmap.close()
    (out / "maps.partial.npy").replace(out / "maps.npy")
    if hasattr(pixels, "close"):
        pixels.close()
    report = dict(
        protocol_sha256=digest(frozen),
        base_maps_sha256=contract["maps_sha256"],
        new_maps_sha256=digest(out / "maps.npy"),
        fitting_rows_sha256=digest(out / "fitting_rows.json"),
        new_inputs_sha256=digest(out / "new_inputs.json"),
        source_files_sha256=digest(out / "source_files.json"),
        reused_spatial_maps=offset,
        added_primary_rows=len(additions),
        new_cnn_maps=len(descriptors),
        new_dino_inference=0,
        quarantined_rows=len(quarantines),
        fitting_rows=len(retained),
        elapsed_seconds=time.monotonic() - started,
        cold_olid_used=False,
    )
    (out / "complete.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
