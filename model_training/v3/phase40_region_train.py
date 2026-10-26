"""Fit a new binary symptom-region veto using SLIF's actual annotation role.

This is not the quarantined whole-leaf trial and does not resume its weights.
The primary detector/classifier remain unchanged. Healthy-region versus
unhealthy-region supervision is explicit; no disease name is emitted by a veto.
"""

from pathlib import Path
import argparse
import collections
import concurrent.futures
import json
import time

import torch
from libreyolo import LibreYOLO9

from opened_context_data import read
from permissive_encoder import digest
from phase3_detector_data import fetch


def prepare(p):
    root = p / "phase40_region_veto"
    root.mkdir(exist_ok=True)
    source = p / "phase3_detector/source_rows.json"
    rows = read(source)
    metadata = read(p / "slif_all_files.json")
    catalog = {r["name"]: r for r in metadata}
    sealed = read(p / "phase24_olid_source/sealed_manifest.json")
    cold_sha = {r["file_sha256"] for r in sealed}
    cold_phash = [int(r["phash"], 16) for r in sealed]
    excluded_groups = set()
    for row in rows:
        if row["file_sha256"] in cold_sha or any(
            (int(row["phash"], 16) ^ v).bit_count() <= 4 for v in cold_phash
        ):
            excluded_groups.add(row["split_group"])
    selected = [r for r in rows if r["split_group"] not in excluded_groups]
    assert set(
        c
        for r in selected
        if r["published_file"].split("/")[1] == "Healthy"
        for c in r["raw_classes"]
    ) == {4}
    categories = collections.defaultdict(set)
    for row in selected:
        for c in row["raw_classes"]:
            categories[c].add(row["published_file"].split("/")[1])
    assert all(len(names) == 1 for names in categories.values())
    protocol = dict(
        source_rows_sha256=digest(source),
        source_catalog_sha256=digest(p / "slif_all_files.json"),
        source="SLIF-Tomato v1, 10.34740/kaggle/ds/8911633, CC BY 4.0",
        source_paper="https://doi.org/10.1007/s00521-026-12271-0",
        annotation_role="Publisher diseased regions and Healthy leaf regions, never generic whole-leaf extents",
        raw_classes={str(k): sorted(v) for k, v in categories.items()},
        binary_mapping={
            "4": "healthy_region",
            "other_publisher_ids": "unhealthy_region",
        },
        former_generic_trial_remains_quarantined=True,
        one_published_view_per_source_family=True,
        preserve_original_group_split=True,
        healthy_training_repeat=4,
        excluded_training_families_for_sealed_overlap=sorted(excluded_groups),
        sealed_membership_sha256=digest(p / "phase24_olid_source/sealed_manifest.json"),
        cold_predictions_used=False,
        script_sha256=digest(__file__),
    )
    frozen = root / "data_protocol.json"
    if frozen.exists():
        assert read(frozen) == protocol
    else:
        frozen.write_text(json.dumps(protocol, indent=2))
    if (root / "data_complete.json").exists():
        return root

    def acquire(row):
        name = Path(row["published_file"]).name
        target_name = "Detection_Dataset/images/" + name
        assert target_name in catalog, "Missing original detection image"
        path = fetch(catalog[target_name], p / "sources/slif_v1")
        assert (
            digest(path) == row["file_sha256"]
        ), "Detection/classification image bytes differ; annotation alignment must be reviewed"
        label = (
            p
            / "sources/slif_v1/Detection_Dataset/lables"
            / Path(name).with_suffix(".txt")
        )
        assert digest(label) == row["annotation_sha256"]
        return dict(row, detection_image=str(path))

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        selected = list(pool.map(acquire, selected))
    assert not (
        {r["split_group"] for r in selected if r["split"] == "train"}
        & {r["split_group"] for r in selected if r["split"] == "val"}
    )
    counts = collections.Counter()
    for row in selected:
        split = row["split"]
        folder = row["published_file"].split("/")[1]
        counts[split + ":" + folder] += 1
        image_dir = root / "dataset/images" / split
        label_dir = root / "dataset/labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        repeats = 4 if split == "train" and folder == "Healthy" else 1
        for n in range(repeats):
            stem = row["group"] + "_repeat" + str(n)
            image = image_dir / (stem + ".jpg")
            if not image.exists():
                image.symlink_to(row["detection_image"])
            text = "".join(
                str(0 if c == 4 else 1) + " " + " ".join(str(v) for v in box) + "\n"
                for c, box in zip(row["raw_classes"], row["boxes"])
            )
            label_dir.joinpath(stem + ".txt").write_text(text)
    (root / "dataset/data.yaml").write_text(
        "path: "
        + str(root / "dataset")
        + "\ntrain: images/train\nval: images/val\nnc: 2\nnames: [healthy_region, unhealthy_region]\n"
    )
    (root / "source_rows.json").write_text(json.dumps(selected, indent=2))
    (root / "data_complete.json").write_text(
        json.dumps(
            dict(
                protocol=protocol,
                counts=dict(counts),
                aligned_images=len(selected),
                all_detection_image_sha256_aligned=True,
                rows_sha256=digest(root / "source_rows.json"),
            ),
            indent=2,
        )
    )
    print(
        json.dumps(
            dict(data_ready=True, counts=dict(counts), aligned_images=len(selected))
        ),
        flush=True,
    )
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    p = Path(args.campaign)
    root = prepare(p)
    if args.prepare_only:
        return
    checkpoint = Path(
        "/home/genesis/sakshyam-agribot/v2/det_runs/y9t_416_comb/weights/best.pt"
    )
    protocol = dict(
        data_sha256=digest(root / "data_complete.json"),
        warm_original_detector_sha256=digest(checkpoint),
        task="Binary healthy/unhealthy region evidence, veto Healthy only; no generic leaf or named-disease output",
        primary_models_unchanged=True,
        input_size=640,
        epochs=20,
        seed=1409,
        lr=0.0001,
        healthy_repeat=4,
        validation="Original held-out publisher region boxes; source-familiar development only",
        cheap_gate="After training, two fixed confidence cutoffs .75/.9 against unchanged app gates; no export before those pass",
        cold_olid_predictions_used=False,
    )
    frozen = root / "training_protocol.json"
    if frozen.exists():
        assert read(frozen) == protocol
    else:
        frozen.write_text(json.dumps(protocol, indent=2))
    if (root / "training_complete.json").exists():
        return
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    last = root / "runs/region_veto_640/weights/last.pt"
    model = LibreYOLO9(
        str(last if last.exists() else checkpoint),
        size="t",
        nb_classes=2 if last.exists() else 1,
    )
    kwargs = dict(
        data=str(root / "dataset/data.yaml"),
        epochs=20,
        imgsz=640,
        batch=32,
        optimizer="AdamW",
        lr0=0.0001,
        workers=6,
        project=str(root / "runs"),
        name="region_veto_640",
        exist_ok=True,
        amp_dtype="bfloat16",
        patience=6,
        eval_interval=2,
        save_period=4,
        seed=1409,
        no_aug_epochs=4,
        mosaic_prob=0.2,
        max_labels=300,
    )
    if last.exists():
        kwargs["resume"] = True
    else:
        kwargs["pretrained"] = str(checkpoint)
    started = time.monotonic()
    result = model.train(**kwargs)
    best = root / "runs/region_veto_640/weights/best.pt"
    report = dict(
        protocol=protocol,
        checkpoint_sha256=digest(best),
        result=result,
        seconds=time.monotonic() - started,
        cold_olid_predictions_used=False,
    )
    (root / "training_complete.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    print(
        json.dumps(
            dict(
                training_complete=True,
                seconds=report["seconds"],
                checkpoint_sha256=report["checkpoint_sha256"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
