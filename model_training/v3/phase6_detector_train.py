"""Limited detector adaptation with original labels plus audited weak leaf proposals.

Pseudo boxes are training targets only. All validation boxes are the unchanged
original leaf annotations; teacher agreement is never reported as ground truth.
No SLIF/bpfd region labels or prospective reserved photographs are used.
"""

from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import time

import torch
from libreyolo import LibreYOLO9
from campaign_data import profile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(campaign, repeat):
    out = campaign / "phase6_detector"
    out.mkdir(exist_ok=True)
    complete = out / "data_complete.json"
    proposals_path = campaign / "phase6_teacher_leaf/training_predictions.jsonl"
    development = json.loads(
        (campaign / "phase5/development_manifest.json").read_text()
    )
    expected = {
        r["id"]: r
        for r in development
        if r["source"] == "tom2024"
        and r["label"] == "Healthy"
        and r["split"] == "train"
    }
    pseudo = [json.loads(line) for line in proposals_path.read_text().splitlines()]
    assert len({r["id"] for r in pseudo}) == len(pseudo) == len(expected)
    audit = json.loads(
        (campaign / "phase6_teacher_leaf/annotation_role_audit.json").read_text()
    )
    assert (
        audit["eligible_for_pseudo_training"]
        and audit["allowed_training_scope"] == "tom2024 Healthy train only"
    )
    identity = dict(
        proposals_sha256=sha(proposals_path),
        source_manifest_sha256=sha(campaign / "phase5/development_manifest.json"),
        repeat=repeat,
        audit_sha256=sha(campaign / "phase6_teacher_leaf/annotation_role_audit.json"),
    )
    if complete.exists():
        assert json.loads(complete.read_text())["identity"] == identity
        return out
    original = Path("/home/genesis/sakshyam-agribot/v2/det_data/leaf_combined")
    old_val = sorted((original / "images/val").iterdir())
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        profiles = list(
            pool.map(lambda path: profile(dict(path=str(path)))[0], old_val)
        )
    old_hashes = {r["file_sha256"] for r in profiles}
    old_pixels = {r["model_input_sha256"] for r in profiles}
    old_phash = [int(r["phash"], 16) for r in profiles]
    reserved = json.loads((campaign / "phase5/reserved_manifest.json").read_text())
    reserved_groups = {r["group"] for r in reserved}
    excluded = []
    selected = []
    for row in pseudo:
        record = expected[row["id"]]
        assert (
            row["file_sha256"] == record["file_sha256"]
            and row["group"] == record["group"]
        )
        assert row["split"] == "train" and row["group"] not in reserved_groups
        assert sha(row["path"]) == row["file_sha256"]
        if (
            row["file_sha256"] in old_hashes
            or row["model_input_sha256"] in old_pixels
            or any((int(row["phash"], 16) ^ h).bit_count() <= 2 for h in old_phash)
        ):
            excluded.append(
                dict(
                    id=row["id"],
                    reason="Overlap/near match with original detector validation",
                )
            )
            continue
        if not row["teacher_boxes"]:
            excluded.append(
                dict(
                    id=row["id"],
                    reason="No confident pseudo boxes; never treated as negative background",
                )
            )
            continue
        selected.append(row)
    assert len(selected) >= 100
    data = out / "dataset"
    for split in ["train", "val"]:
        for kind in ["images", "labels"]:
            (data / kind / split).mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ["train", "val"]:
        images = sorted((original / "images" / split).iterdir())
        counts["original_" + split] = len(images)
        for image in images:
            label = original / "labels" / split / (image.stem + ".txt")
            assert label.exists()
            target = data / "images" / split / ("old_" + image.name)
            lt = data / "labels" / split / ("old_" + image.stem + ".txt")
            if not target.exists():
                target.symlink_to(image.resolve())
            if not lt.exists():
                lt.symlink_to(label.resolve())
    for n, row in enumerate(selected):
        width, height = row["width"], row["height"]
        labels = []
        for box, score in zip(row["teacher_boxes"], row["teacher_scores"]):
            x1, y1, x2, y2 = box
            assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height and score >= 0.25
            labels.append(
                "0 "
                + " ".join(
                    f"{v:.9f}"
                    for v in [
                        ((x1 + x2) / 2) / width,
                        ((y1 + y2) / 2) / height,
                        (x2 - x1) / width,
                        (y2 - y1) / height,
                    ]
                )
                + "\n"
            )
        for index in range(repeat):
            stem = f"pseudo_{n:04d}_r{index}"
            image = data / "images/train" / (stem + Path(row["path"]).suffix.lower())
            label = data / "labels/train" / (stem + ".txt")
            if not image.exists():
                image.symlink_to(Path(row["path"]).resolve())
            label.write_text("".join(labels))
    yaml = data / "data.yaml"
    yaml.write_text(
        f"path: {data}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['leaf']\n"
    )
    (out / "source_rows.json").write_text(json.dumps(selected, indent=2))
    (out / "excluded.json").write_text(json.dumps(excluded, indent=2))
    report = dict(
        identity=identity,
        annotation_role="whole_leaf_boundary",
        pseudo_annotations=True,
        quarantined=False,
        source="Original leaf_combined labels plus TOM2024 Healthy train-only GroundingDINO proposals",
        source_doi="10.17632/3d4yg89rtr.1",
        source_license="CC BY 4.0",
        teacher=audit["teacher_identity"],
        counts=dict(
            counts,
            pseudo_photos=len(selected),
            pseudo_boxes=sum(len(r["teacher_boxes"]) for r in selected),
            excluded=len(excluded),
        ),
        limitations="Predicted boxes are not exhaustive human ground truth. Their omitted leaves and boundary errors can bias training. New field image-level app evaluation and original validation retention are mandatory.",
        reserved_used=False,
        source_rows_sha256=sha(out / "source_rows.json"),
    )
    complete.write_text(json.dumps(report, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--repeat", type=int, default=4)
    a = ap.parse_args()
    assert a.epochs > 0 and a.repeat > 0
    campaign = Path(a.campaign)
    out = prepare(campaign, a.repeat)
    checkpoint = Path(
        "/home/genesis/sakshyam-agribot/v2/det_runs/y9t_416_comb/weights/best.pt"
    )
    config = dict(
        checkpoint_sha256=sha(checkpoint),
        data_sha256=sha(out / "source_rows.json"),
        epochs=a.epochs,
        size=320,
        seed=137,
        learning_rate=0.00005,
        pseudo_repeat=a.repeat,
        checkpoint_selection="Original held-out leaf boxes only; no teacher pseudo-label validation score.",
        original_mAP50_95_max_loss=0.02,
        promotion="Requires actual-byte app regression gates and untouched photo acceptance; training completion alone never permits installation.",
    )
    protocol = out / "training_protocol.json"
    if protocol.exists():
        assert json.loads(protocol.read_text()) == config
    else:
        protocol.write_text(json.dumps(config, indent=2))
    if (out / "training_complete.json").exists():
        return
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    baseline = out / "original_validation.json"
    if not baseline.exists():
        m = LibreYOLO9(str(checkpoint), size="t", nb_classes=1)
        result = m.val(
            data=str(out / "dataset/data.yaml"),
            imgsz=320,
            batch=32,
            workers=4,
            plots=False,
            project=str(out / "validation"),
            name="original",
            exist_ok=True,
        )
        baseline.write_text(json.dumps(result, indent=2, default=str))
        del m
        torch.cuda.empty_cache()
    last = out / "runs/y9t_pseudo_leaf_320/weights/last.pt"
    m = LibreYOLO9(str(last if last.exists() else checkpoint), size="t", nb_classes=1)
    kwargs = dict(
        data=str(out / "dataset/data.yaml"),
        epochs=a.epochs,
        imgsz=320,
        batch=64,
        optimizer="AdamW",
        lr0=0.00005,
        workers=6,
        project=str(out / "runs"),
        name="y9t_pseudo_leaf_320",
        exist_ok=True,
        amp_dtype="bfloat16",
        patience=8,
        eval_interval=2,
        save_period=2,
        seed=137,
        no_aug_epochs=3,
        mosaic_prob=0.2,
        max_labels=300,
    )
    if last.exists():
        kwargs["resume"] = True
    else:
        kwargs["pretrained"] = str(checkpoint)
    started = time.time()
    result = m.train(**kwargs)
    (out / "training_complete.json").write_text(
        json.dumps(
            dict(result=result, seconds=time.time() - started, protocol=config),
            indent=2,
            default=str,
        )
    )
    print(
        json.dumps(dict(state="training_complete", seconds=time.time() - started)),
        flush=True,
    )


if __name__ == "__main__":
    main()
