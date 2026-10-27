"""Reuse original DINO/CNN ROI features for primary-image bag supervision.

No detected leaf receives a clinical ground-truth label. Paired ROI features
form an image-level bag; its publisher primary-view label supervises the bag.
"""

from pathlib import Path
import hashlib
import json
import sqlite3
import numpy as np
from PIL import Image
import timm
import torch

from app_eval import canon, classifier_letterbox_input
from compact_healthy_model import primary
from feature_cache_reuse import preprocessing_contract, runtime_identity
from opened_context_data import read
from partial_dino_cache import input_rows
from permissive_encoder import digest
from phase28_roi_confirmation import pixel_key
from phase29_compact_prepare import region_key
from phase84_population_head import population
from phase85_population_sft import evaluation_data
from population_siglip_tail import freeze


def cached_mobile(db, sha, pixels):
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    key = hashlib.sha256(
        sha.encode() + str(pixels.shape).encode() + pixels.tobytes()
    ).hexdigest()
    saved = db.execute(
        "SELECT values_blob FROM outputs WHERE identity=?", (key,)
    ).fetchone()
    assert (
        saved is not None
    ), "Exact cached mobile input missing; do not silently repeat inference"
    return np.frombuffer(saved[0], dtype=np.float32).copy()


def prepare(campaign, data):
    p = Path(campaign)
    out = p / "phase87_diagnostic_bag_features"
    out.mkdir(exist_ok=True)
    assert not read(p / "phase85_population_sft/result.json")["passed"]
    source = p / "phase29_compact"
    complete = read(source / "complete.json")
    original_protocol = read(source / "protocol.json")
    assert complete["protocol_sha256"] == digest(source / "protocol.json")
    assert digest(source / "features.npy") == complete["features_sha256"]
    assert original_protocol["primary_bundle_sha256"] == digest(
        p / "export_cnn_safe/pytorch_bundle.pt"
    )
    restored_protocol = read(p / "phase73_mobile_verified/protocol.json")
    assert read(p / "phase73_mobile_verified/result.json")["passed"]
    assert restored_protocol["preprocessing"] == preprocessing_contract(
        Path(__file__).parents[2], "whole_dino"
    )
    assert original_protocol["model_source_sha256"] == digest(
        Path(__file__).with_name("compact_healthy_model.py")
    )
    descriptors = input_rows(p)
    assert (
        hashlib.sha256(json.dumps(descriptors).encode()).hexdigest()
        == read(p / "phase73_mobile_verified/protocol.json")["leaf_descriptors_sha256"]
    )
    old_values = np.load(source / "features.npy", mmap_mode="r")
    old_count = len(old_values)
    base_descriptors = read(source / "inputs.json")
    assert len(base_descriptors) == old_count
    lookup = {record["input_sha256"]: i for i, record in enumerate(base_descriptors)}
    center, full = [
        read(source / (name + "_indices.json")) for name in ("center", "full")
    ]
    bags = {
        path: dict(
            center=[center[region_key(path, leaf["box"])] for leaf in row["leaves"]],
            full=[full[region_key(path, leaf["box"])] for leaf in row["leaves"]],
        )
        for path, row in data["photos"].items()
    }
    entries, records, packets, before, _ = evaluation_data(p, data)
    # The legacy cache pinned all of app_eval.py, including later changes to
    # unrelated replay code. Bridge that stale whole-file hash with the already
    # verified preprocessing AST plus exact input bytes for every reused ROI.
    old_files = read(source / "source_files.json")
    checked_old_inputs = 0
    for path, row in data["photos"].items():
        assert id(row) in entries
        if not population(row, entries[id(row)][0]):
            continue
        assert digest(path) == old_files[path]
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        for leaf in row["leaves"]:
            key = region_key(path, leaf["box"])
            for index, tensor in [
                (center[key], canon.classifier_input(rgb, 224, leaf["box"])),
                (full[key], classifier_letterbox_input(rgb, 224, leaf["box"])),
            ]:
                assert pixel_key(tensor) == base_descriptors[index]["input_sha256"]
                checked_old_inputs += 1
    needed = {
        r["path"]
        for r in before
        if population(r, packets[r["id"]]) and packets[r["id"]]["leaves"]
    }
    identities = [
        dict(path=r["path"], file_sha256=r["file_sha256"])
        for r in records
        if r["path"] in needed
    ]
    protocol = dict(
        source_sha256=digest(__file__),
        original_features_sha256=complete["features_sha256"],
        original_descriptors_sha256=hashlib.sha256(
            json.dumps(descriptors).encode()
        ).hexdigest(),
        original_classifier_sha256=original_protocol["primary_bundle_sha256"],
        model_source_sha256=digest(
            Path(__file__).with_name("compact_healthy_model.py")
        ),
        preprocessing_sha256=digest(Path(__file__).with_name("app_eval.py")),
        runtime=runtime_identity(),
        inputs=identities,
        cached_mobile_source_sha256=digest(
            p / "phase76_fresh_acceptance/mobile_scores.jsonl"
        ),
        cached_mobile_database_sha256=digest(
            p / "phase76_fresh_acceptance/cache/predictions.sqlite3"
        ),
        flags=dict(dtype="float32", tf32=False, fused_attention=False),
        preprocessing_bridge=dict(
            legacy_whole_module_sha256=original_protocol["input_preprocessing_sha256"],
            current_whole_module_sha256=digest(Path(__file__).with_name("app_eval.py")),
            verified_mobile_protocol_sha256=digest(
                p / "phase73_mobile_verified/protocol.json"
            ),
            exact_reused_roi_inputs_checked=checked_old_inputs,
        ),
        scope="Primary-image weak bag labels only; no per-leaf truth assignment",
        encoder_weights_changed=False,
        ad_independent=False,
        olid_independent=False,
    )
    frozen = out / "protocol.json"
    freeze(frozen, protocol)
    if (out / "complete.json").exists():
        report = read(out / "complete.json")
        assert report["protocol_sha256"] == digest(frozen)
        assert digest(out / "new_features.npy") == report["new_features_sha256"]
        assert digest(out / "bags.json") == report["bags_sha256"]
        return (
            np.concatenate([old_values, np.load(out / "new_features.npy")]),
            read(out / "bags.json"),
            report,
        )
    mobile = read(p / "phase76_fresh_acceptance/frozen_protocol.json")["models"]
    db = sqlite3.connect(
        "file:"
        + str(p / "phase76_fresh_acceptance/cache/predictions.sqlite3")
        + "?mode=ro",
        uri=True,
    )
    work = {}
    new_bags = {}
    for record in records:
        path = record["path"]
        if path not in needed:
            continue
        assert digest(path) == record["file_sha256"]
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        tensor, letterbox = canon.detector_input(rgb, 320)
        raw = cached_mobile(db, mobile["detector"], tensor).reshape(1, 5, 2100)
        contract = data["manifest"]["detector"]
        detections = canon.decode_detections(
            raw,
            letterbox,
            rgb.shape[1],
            rgb.shape[0],
            contract["score_threshold"],
            contract["nms_iou"],
            contract["max_detections"],
        )[:6]
        assert len(detections) == len(packets[record["id"]]["leaves"])
        pair = dict(center=[], full=[])
        for box in detections:
            for view, pixels in [
                ("center", canon.classifier_input(rgb, 224, box[:4])),
                ("full", classifier_letterbox_input(rgb, 224, box[:4])),
            ]:
                key = pixel_key(pixels)
                pair[view].append(key)
                if key not in lookup:
                    work.setdefault(
                        key, dict(path=path, box=box[:4], view=view, input_sha256=key)
                    )
        new_bags[path] = pair
    keys = sorted(work)
    for index, key in enumerate(keys):
        lookup[key] = old_count + index
    for path, pair in new_bags.items():
        bags[path] = {
            view: [lookup[key] for key in values] for view, values in pair.items()
        }
    (out / "new_inputs.json").write_text(json.dumps([work[key] for key in keys]))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timm.layers.set_fused_attn(False)
    _, model = primary(p)
    model = model.cuda().eval()
    progress = out / "progress.json"
    feature_path = out / "new_features.npy"
    saved = (
        read(progress) if progress.exists() else dict(completed=0, max_logit_error=0.0)
    )
    if progress.exists():
        assert saved["protocol_sha256"] == digest(frozen)
    values = np.lib.format.open_memmap(
        feature_path,
        mode="r+" if progress.exists() else "w+",
        dtype=np.float32,
        shape=(len(keys), 1408),
    )
    maximum = saved["max_logit_error"]
    with torch.inference_mode():
        for start in range(saved["completed"], len(keys), 64):
            part = keys[start : start + 64]
            pixels = []
            reference = []
            for key in part:
                entry = work[key]
                with Image.open(entry["path"]) as image:
                    rgb = np.asarray(image.convert("RGB"))
                tensor = (
                    canon.classifier_input(rgb, 224, entry["box"])
                    if entry["view"] == "center"
                    else classifier_letterbox_input(rgb, 224, entry["box"])
                )
                assert pixel_key(tensor) == key
                pixels.append(tensor)
                reference.append(cached_mobile(db, mobile["leaf"], tensor)[:9])
            features, actual = model.components(
                torch.from_numpy(np.concatenate(pixels, 0)).cuda()
            )
            maximum = max(
                maximum,
                float(np.max(np.abs(actual.cpu().numpy() - np.stack(reference)))),
            )
            assert maximum <= 0.002, (
                "Original diagnostic graph/pixel mismatch",
                maximum,
            )
            values[start : start + len(part)] = features.cpu().numpy()
            values.flush()
            temporary = progress.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    dict(
                        protocol_sha256=digest(frozen),
                        completed=start + len(part),
                        max_logit_error=maximum,
                    )
                )
            )
            temporary.replace(progress)
            print(
                json.dumps(
                    dict(
                        stage="new_diagnostic_features",
                        completed=start + len(part),
                        new_inputs=len(keys),
                        old_inputs_reused=old_count,
                        max_logit_error=maximum,
                    )
                ),
                flush=True,
            )
    db.close()
    del model
    torch.cuda.empty_cache()
    (out / "bags.json").write_text(json.dumps(bags))
    report = dict(
        protocol_sha256=digest(frozen),
        new_features_sha256=digest(feature_path),
        bags_sha256=digest(out / "bags.json"),
        new_encoder_inputs=len(keys),
        old_encoder_inputs_reused=old_count,
        new_images=len(needed),
        original_mobile_max_logit_error=maximum,
        old_disease_weights_changed=False,
        olid_independent=False,
        ad_independent=False,
    )
    (out / "complete.json").write_text(json.dumps(report, indent=2))
    return np.concatenate([old_values, np.asarray(values)]), bags, report
