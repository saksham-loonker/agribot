"""Prepare a leaf-only binary control from explicit FieldPlant instance labels.

Reuse pre-teacher pixel shards; do not duplicate or alter them. Weak TOM image
and inferred-ROI training targets are masked for this leaf-specific control.
Opened app evaluation rows keep their original labels and remain in regression.
The separate AD source is checked for overlap without diagnostic inference.
"""

from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import shutil

import numpy as np
from PIL import Image
import torch

from app_eval import canon, classifier_letterbox_input
from campaign_cnn_export import restore
from campaign_reference import safe_logits


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    a = ap.parse_args()
    p = Path(a.campaign)
    base = p / "phase5"
    source = p / "phase10_fieldplant"
    root = p / "phase10_leaf"
    root.mkdir(exist_ok=True)
    work = root / "cnn_inputs"
    work.mkdir(exist_ok=True)
    audit = json.loads((source / "membership_audit.json").read_text())
    assert audit["leaf_instances"] > 500
    identity = dict(
        base_rows_sha256=digest(base / "cnn_inputs/rows.json"),
        leaf_manifest_sha256=digest(source / "leaf_manifest.json"),
        masked_sources=["tom2024"],
        views=["center", "full_roi"],
        cold_ad_manifest_sha256=digest(
            p / "phase9_ad_source/sealed_geometric_manifest.json"
        ),
    )
    protocol = root / "input_protocol.json"
    if protocol.exists():
        assert json.loads(protocol.read_text()) == identity
    else:
        protocol.write_text(json.dumps(identity, indent=2))
    complete = root / "complete.json"
    if complete.exists():
        assert json.loads(complete.read_text())["rows_sha256"] == digest(
            work / "rows.json"
        )
        return
    records = json.loads((source / "leaf_manifest.json").read_text())
    cold = json.loads(
        (p / "phase9_ad_source/sealed_geometric_manifest.json").read_text()
    )
    cold_file = {r["file_sha256"] for r in cold}
    cold_rgb = {r["rgb32_sha256"] for r in cold}
    cold_hash = np.array([int(r["phash"], 16) for r in cold], np.uint64)
    blocked = {
        r["group"]
        for r in records
        if r["file_sha256"] in cold_file
        or r["rgb32_sha256"] in cold_rgb
        or int(np.bitwise_count(cold_hash ^ np.uint64(int(r["phash"], 16))).min()) <= 2
    }
    records = [r for r in records if r["group"] not in blocked]
    rows = json.loads((base / "cnn_inputs/rows.json").read_text())
    masked = 0
    for row in rows:
        if row["set"] == "training" and row["source"] == "tom2024":
            row["masked_prior_image_label"] = row["label"]
            row["label"] = "Unverified_leaf_target"
            row["health_y"] = -1.0
            row["known_y"] = -1.0
            masked += 1
    old_count = len(rows)
    for r in records:
        for view in ["center", "full_roi"]:
            rows.append(
                dict(
                    r,
                    set="training",
                    row=-1,
                    leaf=-1,
                    view="fieldplant_" + view,
                    input_variant=view,
                    feature_index=-1,
                    annotation_provenance="Explicit publisher leaf box; Healthy versus binary non-Healthy only.",
                )
            )
    extra = rows[old_count:]
    assert len(extra) > 1000
    assert not (
        {r["group"] for r in rows if r["set"] == "training" and r["split"] == "train"}
        & {r["group"] for r in rows if r["set"] == "training" and r["split"] == "val"}
    )
    target = work / "fieldplant_pixels.npy"
    partial = work / "fieldplant_pixels.partial.npy"
    # This task-owned partial cache may restart only after its input identity matches.
    pixels = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float32, shape=(len(extra), 224, 224, 3)
    )

    def prepare(row):
        with Image.open(row["path"]) as im:
            image = np.asarray(im.convert("RGB"))
        return (
            classifier_letterbox_input(image, 224, row["box"])
            if row["input_variant"] == "full_roi"
            else canon.classifier_input(image, 224, row["box"])
        )[0]

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for start in range(0, len(extra), 128):
            pixels[start : start + 128] = np.stack(
                list(pool.map(prepare, extra[start : start + 128]))
            )
            if start % 1024 == 0:
                print(
                    json.dumps(dict(stage="pixels", rows=start, total=len(extra))),
                    flush=True,
                )
    pixels.flush()
    pixels._mmap.close()
    partial.replace(target)
    (work / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    parts = json.loads((base / "cnn_inputs/pixels.json").read_text())["center"]
    (work / "pixels.json").write_text(
        json.dumps(dict(center=[*parts, str(target)]), indent=2)
    )
    for name in ["eval_rows.json", "reserved_manifest.json"]:
        shutil.copyfile(base / name, root / name)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = restore(p / "export_cnn_safe/pytorch_bundle.pt").cuda().eval()
    warm = dict(
        architecture="mobilenetv3_small_100.lamb_in1k",
        model={k: v.cpu().clone() for k, v in model.cnn.state_dict().items()},
        mean=model.cnn_mean.flatten().cpu().tolist(),
        std=model.cnn_std.flatten().cpu().tolist(),
        labels=["Healthy", "Known"],
        policy="center",
        source_bundle_sha256=digest(p / "export_cnn_safe/pytorch_bundle.pt"),
    )
    torch.save(warm, root / "safe_cnn_warm.pt")
    pixels = np.load(target, mmap_mode="r")
    raw = []
    safe = []
    with torch.inference_mode():
        for start in range(0, len(extra), 128):
            x = (
                torch.from_numpy(np.array(pixels[start : start + 128], copy=True))
                .cuda()
                .permute(0, 3, 1, 2)
            )
            z = model.backbone(x)
            c = model.cnn(
                (x * model.canonical_std + model.canonical_mean - model.cnn_mean)
                / model.cnn_std
            )
            raw.append(z.cpu().numpy())
            safe.append(model.fusion(z, c).cpu().numpy())
    np.save(
        root / "reference_logits.npy",
        np.concatenate([np.load(base / "reference_logits.npy"), *raw]),
    )
    np.save(
        root / "safe_reference_logits.npy", np.concatenate([safe_logits(base), *safe])
    )
    pixels._mmap.close()
    report = dict(
        input_identity=identity,
        rows=len(rows),
        rows_sha256=digest(work / "rows.json"),
        new_leaf_views=len(extra),
        new_leaf_instances=len(records),
        masked_weak_training_rows=masked,
        cold_ad_overlap_families_excluded=sorted(blocked),
        cold_ad_predictions_opened=False,
        original_reference_assets=str(p / "original_reference"),
        scope="Leaf-instance binary control. Pre-teacher baseline shards reused, all TOM training targets masked, app regression labels unchanged. No new nine-class disease labels inferred.",
    )
    complete.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
