"""True leaf supervision and frozen deployed-model features, no new encoder.

Detector-propagated fitting labels and weak TOM proposals are excluded. AD
supplies human primary-image targets only, never labels for its detected ROIs.
All old evaluation cases stay present. OLID remains sealed.
"""

from pathlib import Path
import argparse
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
from campaign_reference import safe_logits
from compact_healthy_model import primary
from opened_context_data import load, read
from permissive_encoder import digest
from phase28_roi_confirmation import pixel_key


def region_key(path, box):
    return json.dumps([path, box], separators=(",", ":"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    a = ap.parse_args()
    p = Path(a.campaign)
    out = p / "phase29_compact"
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timm.layers.set_fused_attn(False)
    data = load(p)
    leaf = p / "phase10_leaf"
    rows = read(leaf / "cnn_inputs/rows.json")
    ev = read(leaf / "eval_rows.json")
    feature_dir = leaf / "leaf_feature_heads"
    identity = read(feature_dir / "feature_identity.json")
    assert digest(leaf / "cnn_inputs/rows.json") == identity["rows_sha256"]
    assert (
        digest(p / "export_cnn_safe/pytorch_bundle.pt")
        == identity["reference_bundle_sha256"]
    )
    assert read(feature_dir / "reference_parity.json")["passed"]
    previously_computed = np.array(
        [
            i
            for i, r in enumerate(rows)
            if r["set"] in ev
            or (
                r["set"] == "training"
                and r["split"] in ("train", "val")
                and r["health_y"] >= 0
            )
        ],
        np.int64,
    )
    assert (
        hashlib.sha256(previously_computed.tobytes()).hexdigest()
        == identity["ids_sha256"]
    )
    strong = []
    for i, r in enumerate(rows):
        if (
            r["set"] != "training"
            or r["split"] not in ("train", "val")
            or r["health_y"] < 0
        ):
            continue
        if r["source"] in ("tom2024", "swtt5wy2pr"):
            continue
        if r.get("view") in ("detector", "detected"):
            continue
        if (
            r["source"] == "fieldplant"
            or r.get("view") == "annotation"
            or (r["source"] == "bangladesh_binary" and r.get("box") is not None)
        ):
            strong.append(i)
    old_eval = [i for i, r in enumerate(rows) if r["set"] in ev]
    selected = np.array(sorted(set(strong) | set(old_eval)), np.int64)
    assert set(selected) <= set(previously_computed)
    admitted = [
        r
        for r in read(p / "phase23_data/admission_manifest.json")
        if r["source"] == "ad_greenhouse"
    ]
    assert len(admitted) == 502
    protocol = dict(
        primary_bundle_sha256=identity["reference_bundle_sha256"],
        leaf_rows_sha256=identity["rows_sha256"],
        old_dino_features_sha256=digest(feature_dir / "features.npy"),
        human_admission_sha256=digest(p / "phase23_data/admission_manifest.json"),
        input_preprocessing_sha256=digest(Path(__file__).with_name("app_eval.py")),
        script_sha256=digest(__file__),
        model_source_sha256=digest(
            Path(__file__).with_name("compact_healthy_model.py")
        ),
        training_scope="Human leaf annotations/single-primary-leaf classification inputs; human AD primary images. No AD detector-ROI label propagation.",
        excluded_fitting_views=[
            "detector",
            "detected",
            "weak TOM proposals",
            "paper symptom-region geometry",
        ],
        all_original_evaluation_cases_preserved=True,
        original_classifier_frozen=True,
        restricted_teacher_used=False,
        cold_olid_used=False,
        ad_independent=False,
        planned_head_controls=["One 1408-dimensional MLP with weighted BCE"],
        maximum_epochs=100,
        checkpoint_epochs=[0, 4, 9, 19, 39, 59, 79, 99],
        healthy_thresholds=[0.8, 0.9, 0.95, 0.99],
        mandatory_two_view_confirmation=True,
    )
    path = out / "protocol.json"
    if path.exists():
        assert read(path) == protocol
    else:
        path.write_text(json.dumps(protocol, indent=2))
    if (out / "complete.json").exists():
        complete = read(out / "complete.json")
        assert complete["protocol_sha256"] == digest(path)
        print("Feature cache already complete", flush=True)
        return
    started = time.monotonic()
    pixels = open_pixels(leaf / "cnn_inputs")
    old_dino = np.load(feature_dir / "features.npy", mmap_mode="r")
    descriptors = []
    lookup = {}
    old_map = {}
    for start in range(0, len(selected), 256):
        part = selected[start : start + 256]
        batch = pixels[part]
        for i, tensor in zip(part, batch):
            key = pixel_key(tensor)
            if key not in lookup:
                lookup[key] = len(descriptors)
                descriptors.append(dict(kind="old", index=int(i), input_sha256=key))
            else:
                first = descriptors[lookup[key]]
                assert first["kind"] == "old"
                assert (
                    np.max(np.abs(old_dino[int(i)] - old_dino[first["index"]])) < 0.002
                )
            old_map[int(i)] = lookup[key]
    old_count = len(descriptors)
    center = {}
    for i in old_eval:
        r = rows[i]
        key = region_key(r["path"], r["box"])
        if key in center:
            assert center[key] == old_map[i]
        center[key] = old_map[i]
    all_records = [r for values in data["base"].values() for r in values] + data[
        "internal"
    ]
    regions = {}
    for row in all_records:
        regions.setdefault(row["path"], set()).update(
            tuple(x["box"]) for x in row["leaves"]
        )
    ad_by_path = {r["path"]: r for r in admitted}
    for path in ad_by_path:
        regions.setdefault(path, set())
    full = {}
    ad_map = {}
    file_hashes = {}

    def prepare(item):
        path, boxes = item
        expected = data["photos"].get(path, ad_by_path.get(path, {})).get("file_sha256")
        actual = digest(path)
        if expected:
            assert actual == expected
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        values = []
        if path in ad_by_path:
            values.append(("ad", None, canon.classifier_input(rgb, 224)))
        for box in sorted(boxes):
            values.append(
                ("center", list(box), canon.classifier_input(rgb, 224, list(box)))
            )
            values.append(
                ("full", list(box), classifier_letterbox_input(rgb, 224, list(box)))
            )
        return path, actual, values

    def register(path, box, policy, tensor):
        key = pixel_key(tensor)
        if key not in lookup:
            lookup[key] = len(descriptors)
            descriptors.append(
                dict(kind="image", path=path, box=box, policy=policy, input_sha256=key)
            )
        return lookup[key]

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for path, file_hash, values in pool.map(prepare, regions.items()):
            file_hashes[path] = file_hash
            for kind, box, tensor in values:
                index = register(
                    path, box, "full" if kind == "full" else "center", tensor
                )
                key = region_key(path, box)
                if kind == "ad":
                    ad_map[path] = index
                elif kind == "full":
                    full[key] = index
                else:
                    if key in center:
                        assert (
                            center[key] == index
                        ), "Stored primary ROI tensor differs from the actual app canonical crop"
                    center[key] = index
    fitting = []
    for i in strong:
        r = rows[i]
        fitting.append(
            dict(
                table=old_map[i],
                id=r["id"],
                source=r["source"],
                group=r["group"],
                split=r["split"],
                y=int(r["health_y"]),
                role="annotated_leaf" if r.get("box") is not None else "primary_image",
            )
        )
    for r in admitted:
        fitting.append(
            dict(
                table=ad_map[r["path"]],
                id=r["id"],
                source=r["source"],
                group=r["group"],
                split=r["split"],
                y=int(r["label"] == "Healthy"),
                role="primary_image",
            )
        )
    labels = {}
    for r in fitting:
        labels.setdefault(r["table"], set()).add(r["y"])
    contradictory = {key for key, v in labels.items() if len(v) > 1}
    val_tables = {r["table"] for r in fitting if r["split"] == "val"}
    val_groups = {r["group"] for r in fitting if r["split"] == "val"}
    excluded = []
    retained = []
    for r in fitting:
        reason = None
        if r["table"] in contradictory:
            reason = "Contradictory human labels on identical canonical tensor"
        elif r["split"] == "train" and (
            r["group"] in val_groups or r["table"] in val_tables
        ):
            reason = "Training overlap with validation input/family"
        if reason:
            excluded.append(dict(r, reason=reason))
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
    (out / "fitting_exclusions.json").write_text(json.dumps(excluded, indent=2))
    (out / "inputs.json").write_text(json.dumps(descriptors))
    (out / "center_indices.json").write_text(json.dumps(center))
    (out / "full_indices.json").write_text(json.dumps(full))
    (out / "source_files.json").write_text(json.dumps(file_hashes))
    features = np.lib.format.open_memmap(
        out / "features.partial.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(descriptors), 1408),
    )
    references = np.lib.format.open_memmap(
        out / "reference_logits.partial.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(descriptors), 9),
    )
    spec, model = primary(p)
    model = model.cuda().eval()
    maximum_error = 0.0
    reference_cache = safe_logits(leaf)
    with torch.inference_mode():
        for start in range(0, old_count, 256):
            part = descriptors[start : min(start + 256, old_count)]
            ids = np.array([r["index"] for r in part])
            image = torch.from_numpy(np.array(pixels[ids], copy=True)).cuda()
            dino = torch.from_numpy(np.array(old_dino[ids], copy=True)).cuda()
            f, z = model.from_dino(image, dino)
            error = float(np.abs(z.cpu().numpy() - reference_cache[ids]).max())
            maximum_error = max(maximum_error, error)
            assert error < 0.002, (
                "Primary DINO/CNN feature composition differs from installed comparator",
                error,
            )
            features[start : start + len(part)] = f.cpu().numpy()
            references[start : start + len(part)] = z.cpu().numpy()

    def input_tensor(r):
        with Image.open(r["path"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        tensor = (
            classifier_letterbox_input
            if r["policy"] == "full"
            else canon.classifier_input
        )(rgb, 224, r["box"])
        assert pixel_key(tensor) == r["input_sha256"]
        return tensor

    with torch.inference_mode(), concurrent.futures.ThreadPoolExecutor(8) as pool:
        for start in range(old_count, len(descriptors), 64):
            part = descriptors[start : start + 64]
            inputs = torch.from_numpy(
                np.concatenate(list(pool.map(input_tensor, part)), 0)
            ).cuda()
            f, z = model.components(inputs)
            features[start : start + len(part)] = f.cpu().numpy()
            references[start : start + len(part)] = z.cpu().numpy()
            if start == old_count or (start - old_count) % 1024 == 0:
                print(
                    json.dumps(
                        dict(
                            stage="new_inputs",
                            complete=start + len(part),
                            total=len(descriptors),
                        )
                    ),
                    flush=True,
                )
    probes = np.linspace(0, len(descriptors) - 1, 54, dtype=int)
    # Strictly reconstructed components must also match the unmodified source
    # network, including its own original calibration/fusion, on fresh tensors.
    for index in probes:
        r = descriptors[index]
        tensor = (
            np.asarray(pixels[r["index"]])[None]
            if r["kind"] == "old"
            else input_tensor(r)
        )
        with torch.inference_mode():
            actual = (
                model.primary(torch.from_numpy(np.array(tensor, copy=True)).cuda())
                .cpu()
                .numpy()[0]
            )
        maximum_error = max(
            maximum_error, float(np.max(np.abs(actual - references[index])))
        )
    assert maximum_error < 0.002
    features.flush()
    references.flush()
    features._mmap.close()
    references._mmap.close()
    (out / "features.partial.npy").replace(out / "features.npy")
    (out / "reference_logits.partial.npy").replace(out / "reference_logits.npy")
    report = dict(
        protocol_sha256=digest(out / "protocol.json"),
        inputs=len(descriptors),
        old_dino_vectors_reused=old_count,
        new_dino_inputs=len(descriptors) - old_count,
        fitting_rows=len(retained),
        fitting_quarantines=len(excluded),
        primary_max_abs_logit_error=maximum_error,
        features_sha256=digest(out / "features.npy"),
        fitting_rows_sha256=digest(out / "fitting_rows.json"),
        elapsed_seconds=time.monotonic() - started,
        original_classifier_frozen=True,
        cold_olid_used=False,
    )
    (out / "complete.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
