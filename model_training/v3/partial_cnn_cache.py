"""Identity-checked original CNN prefixes for one bounded supervised adaptation."""

from pathlib import Path
import ast
import concurrent.futures
import hashlib
import json
import time

import numpy as np
from PIL import Image
import timm
import torch

from app_eval import canon, classifier_letterbox_input
from campaign_pixels import open_pixels
from feature_cache_reuse import preprocessing_contract, runtime_identity
from opened_context_data import read
from partial_cnn_healthy import prefix_cut, prefix_features, prefix_identity
from permissive_encoder import digest
from phase28_roi_confirmation import pixel_key


def original_cnn(campaign):
    state = torch.load(
        Path(campaign) / "export_cnn_safe/pytorch_bundle.pt",
        weights_only=True,
        map_location="cpu",
    )
    cnn = (
        timm.create_model(state["cnn_architecture"], pretrained=False, num_classes=2)
        .eval()
        .requires_grad_(False)
    )
    cnn.load_state_dict(
        {
            key[4:]: value
            for key, value in state["model"].items()
            if key.startswith("cnn.")
        },
        strict=True,
    )
    normalization = {
        name: state["model"][name]
        for name in ("canonical_mean", "canonical_std", "cnn_mean", "cnn_std")
    }
    return cnn, state["cnn_architecture"], normalization


def prepare(campaign):
    p = Path(campaign)
    out = p / "phase49_partial_cnn/prefix"
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    base = p / "phase29_compact"
    context = p / "phase38_context_spatial"
    paper = p / "phase46_paper_primary"
    rows = read(base / "inputs.json")
    for root in (context, paper):
        rows += [
            dict(row, kind="scene", index=row["source_row"])
            for row in read(root / "new_inputs.json")
        ]
    fitting = read(paper / "fitting_rows.json")
    assert max(r["table"] for r in fitting) < len(rows)
    assert not (
        {r["group"] for r in fitting if r["split"] == "train"}
        & {r["group"] for r in fitting if r["split"] == "val"}
    )
    paths = sorted({r["path"] for r in rows if r.get("kind") != "old"})
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        images = list(zip(paths, pool.map(digest, paths)))
    pixel_stores = []
    for work in (p / "phase10_leaf/cnn_inputs", p / "phase23_siglip/cnn_inputs"):
        pixel_stores.append(
            dict(
                manifest_sha256=digest(work / "pixels.json"),
                files=[
                    dict(
                        path=path,
                        bytes=Path(path).stat().st_size,
                        mtime_ns=Path(path).stat().st_mtime_ns,
                    )
                    for path in read(work / "pixels.json")["center"]
                ],
            )
        )
    tree = ast.parse(Path(__file__).with_name("partial_cnn_healthy.py").read_text())
    prefix_source = [
        ast.dump(node, include_attributes=False)
        for node in tree.body
        if getattr(node, "name", None) in ("prefix_cut", "prefix_features")
    ]
    protocol = dict(
        representation="Original MobileNetV3-Small before its final two blocks, FP32; no DINO inference",
        bundle_sha256=digest(p / "export_cnn_safe/pytorch_bundle.pt"),
        input_descriptors_sha256=digest(base / "inputs.json"),
        context_inputs_sha256=digest(context / "new_inputs.json"),
        paper_inputs_sha256=digest(paper / "new_inputs.json"),
        fitting_rows_sha256=digest(paper / "fitting_rows.json"),
        image_files_identity_sha256=hashlib.sha256(
            json.dumps(images).encode()
        ).hexdigest(),
        pixel_stores=pixel_stores,
        preprocessing=preprocessing_contract(
            Path(__file__).resolve().parents[2], "original_cnn_prefix"
        ),
        prefix_source_sha256=hashlib.sha256(
            json.dumps(prefix_source).encode()
        ).hexdigest(),
        normalization_source_sha256=hashlib.sha256(
            ast.dump(
                next(
                    node
                    for node in ast.parse(Path(__file__).read_text()).body
                    if getattr(node, "name", None) == "original_cnn"
                ),
                include_attributes=False,
            ).encode()
        ).hexdigest(),
        runtime=runtime_identity(),
        cache_source_sha256=digest(__file__),
        inputs=len(rows),
        cold_olid_used=False,
        ad_independent=False,
    )
    frozen = out / "protocol.json"
    if frozen.exists():
        assert read(frozen) == protocol, "Prefix cache identity changed"
    else:
        frozen.write_text(json.dumps(protocol, indent=2))
    if (out / "complete.json").exists():
        complete = read(out / "complete.json")
        assert complete["protocol_sha256"] == digest(frozen)
        assert complete["features_sha256"] == digest(out / "features.npy")
        assert complete["maximum_original_map_error"] < 0.002
        return out
    references = [
        np.load(root / "maps.npy", mmap_mode="r")
        for root in (p / "phase37_spatial", context, paper)
    ]
    for root, array in zip((p / "phase37_spatial", context, paper), references):
        complete = read(root / "complete.json")
        expected = complete.get("maps_sha256", complete.get("new_maps_sha256"))
        assert digest(root / "maps.npy") == expected
    assert sum(len(a) for a in references) == len(rows)
    old_pixels = open_pixels(p / "phase10_leaf/cnn_inputs")
    scene_pixels = open_pixels(p / "phase23_siglip/cnn_inputs")
    cnn, architecture, normalization = original_cnn(p)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cnn = cnn.cuda()
    normalization = {key: value.cuda() for key, value in normalization.items()}
    cut = prefix_cut(cnn)
    cached = None
    maximum_error = 0.0

    def pixels(row):
        if row["kind"] in ("old", "scene"):
            store = old_pixels if row["kind"] == "old" else scene_pixels
            value = np.array(store[row["index"]], copy=True)[None]
        else:
            with Image.open(row["path"]) as image:
                rgb = np.asarray(image.convert("RGB"))
            value = (
                classifier_letterbox_input
                if row["policy"] == "full"
                else canon.classifier_input
            )(rgb, 224, row["box"])
        assert pixel_key(value) == row["input_sha256"], "Canonical prefix input changed"
        return value

    def reference(start, count):
        values = []
        for array in references:
            if start < len(array):
                part = array[start : min(len(array), start + count)]
                values.append(part)
                count -= len(part)
                start = 0
            else:
                start -= len(array)
            if count == 0:
                break
        return np.concatenate(values, 0)

    with concurrent.futures.ThreadPoolExecutor(8) as pool, torch.inference_mode():
        for start in range(0, len(rows), 256):
            batch = (
                torch.from_numpy(
                    np.concatenate(list(pool.map(pixels, rows[start : start + 256])), 0)
                )
                .cuda()
                .permute(0, 3, 1, 2)
            )
            rgb = (
                batch * normalization["canonical_std"]
                + normalization["canonical_mean"]
                - normalization["cnn_mean"]
            ) / normalization["cnn_std"]
            prefix = prefix_features(cnn, rgb)
            if cached is None:
                cached = np.lib.format.open_memmap(
                    out / "features.partial.npy",
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(rows), *prefix.shape[1:]),
                )
            assert torch.isfinite(prefix).all()
            cached[start : start + len(batch)] = prefix.cpu().numpy()
            restored = cnn.blocks[cut:](prefix).cpu().numpy()
            maximum_error = max(
                maximum_error,
                float(np.max(np.abs(restored - reference(start, len(batch))))),
            )
            assert (
                maximum_error < 0.002
            ), "Prefix/suffix does not reconstruct original cached CNN maps"
    cached.flush()
    cached._mmap.close()
    (out / "features.partial.npy").replace(out / "features.npy")
    for store in (old_pixels, scene_pixels):
        if hasattr(store, "close"):
            store.close()
    complete = dict(
        protocol_sha256=digest(frozen),
        features_sha256=digest(out / "features.npy"),
        prefix_state_sha256=prefix_identity(cnn),
        architecture=architecture,
        cut=cut,
        inputs=len(rows),
        maximum_original_map_error=maximum_error,
        elapsed_seconds=time.monotonic() - started,
        new_dino_inference=0,
        cold_olid_used=False,
    )
    (out / "complete.json").write_text(json.dumps(complete, indent=2))
    print(json.dumps(complete), flush=True)
    return out
