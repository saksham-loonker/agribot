"""Admit the opened classification source into a new grouped fitting phase."""

from pathlib import Path
import argparse, concurrent.futures, hashlib, json, threading, time
import numpy as np
from PIL import Image
import timm
import torch
from app_eval import Models, image_record, canon
from train import LABELS
from training_support import load_initial_weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", required=True)
    a = p.parse_args()
    campaign = Path(a.campaign)
    root = campaign / "phase2"
    work = root / "cnn_inputs"
    work.mkdir(parents=True, exist_ok=True)
    if (root / "complete.json").exists():
        previous = json.loads((root / "complete.json").read_text())
        assert (
            previous["rows_sha256"]
            == hashlib.sha256((work / "rows.json").read_bytes()).hexdigest()
        )
        assert (
            previous["source_manifest_sha256"]
            == hashlib.sha256(
                (campaign / "phase2_source_manifest.json").read_bytes()
            ).hexdigest()
        )
        print("Verified completed phase-two cache; no files changed", flush=True)
        return
    source = json.loads((campaign / "phase2_source_manifest.json").read_text())
    base = json.loads((campaign / "cnn_inputs/rows.json").read_text())
    rows = [dict(r) for r in base]
    ev = json.loads((campaign / "dense_aligned/eval_rows.json").read_text())
    ev["phase2_selection"] = []
    local = threading.local()
    assets = campaign / "original_reference"
    started = time.time()

    def detect(record):
        if not hasattr(local, "model"):
            local.model = Models(assets, threads=1)
        if record["split"] == "val":
            result = image_record(local.model, record)
            return record, result["detections"], result
        with Image.open(record["path"]) as im:
            image = np.asarray(im.convert("RGB"))
        return record, local.model.detect(image), None

    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        for i, (record, detections, photo) in enumerate(pool.map(detect, source)):
            common = dict(
                record,
                set="training",
                row=-1,
                leaf=-1,
                health_y=float(record["label"] == "Healthy"),
                known_y=float(record["label"] != "Other"),
                feature_index=-1,
            )
            rows.append(dict(common, view="annotation", box=None))
            if detections:
                rows.append(dict(common, view="detected", box=detections[0][:4]))
            if photo:
                index = len(ev["phase2_selection"])
                ev["phase2_selection"].append(photo)
                for leaf, observation in enumerate(photo["leaves"]):
                    rows.append(
                        dict(
                            common,
                            set="phase2_selection",
                            split="regression",
                            row=index,
                            leaf=leaf,
                            box=observation["box"],
                            view="selected",
                        )
                    )
            if i % 100 == 0:
                print(
                    json.dumps(
                        dict(
                            stage="detector_alignment",
                            photos=i,
                            total=len(source),
                            seconds=round(time.time() - started, 1),
                        )
                    ),
                    flush=True,
                )
    extra = rows[len(base) :]
    (work / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    (root / "eval_rows.json").write_text(json.dumps(ev, indent=2) + "\n")
    path = work / "extra_center.npy"
    if path.exists():
        raise ValueError("Extra cache exists; do not overwrite an unverified identity")
    pixels = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(len(extra), 224, 224, 3)
    )

    def prepare(r):
        with Image.open(r["path"]) as im:
            image = np.asarray(im.convert("RGB"))
        return canon.classifier_input(image, 224, r["box"])[0]

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for start in range(0, len(extra), 64):
            pixels[start : start + 64] = np.stack(
                list(pool.map(prepare, extra[start : start + 64]))
            )
            if start % 512 == 0:
                print(
                    json.dumps(
                        dict(
                            stage="exact_pixels",
                            rows=start,
                            total=len(extra),
                            seconds=round(time.time() - started, 1),
                        )
                    ),
                    flush=True,
                )
    pixels.flush()
    (work / "pixels.json").write_text(
        json.dumps(
            dict(center=[str(campaign / "cnn_inputs/center.npy"), str(path)]), indent=2
        )
    )
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = (
        timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=False,
            num_classes=9,
            img_size=224,
        )
        .cuda()
        .eval()
    )
    model.requires_grad_(False)
    checkpoint = "/home/genesis/sakshyam-agribot/v2/runs/field_vits2/best.pt"
    load_initial_weights(
        model, checkpoint, LABELS, "vit_small_patch14_dinov2.lvd142m", 224
    )
    parts = []
    with torch.no_grad():
        for start in range(0, len(extra), 128):
            parts.append(
                model(
                    torch.from_numpy(np.ascontiguousarray(pixels[start : start + 128]))
                    .cuda()
                    .permute(0, 3, 1, 2)
                ).cpu()
            )
    old = torch.load(campaign / "dense_aligned/features.pt", weights_only=True)[
        "logits"
    ].numpy()
    binary = np.load(campaign / "cnn_inputs/extra_reference_logits.npy")
    reference = np.concatenate([old, binary, torch.cat(parts).numpy()])
    assert len(reference) == len(rows)
    np.save(root / "reference_logits.npy", reference)
    # Validate real selected ROIs against independent original TFLite outputs.
    errors = []
    for i, r in enumerate(rows):
        if r["set"] == "phase2_selection":
            errors.append(
                float(
                    np.abs(
                        reference[i]
                        - np.asarray(
                            ev["phase2_selection"][r["row"]]["leaves"][r["leaf"]][
                                "member_logits"
                            ][0]
                        )
                    ).max()
                )
            )
    assert max(errors, default=0) < 0.002, max(errors)
    metadata = dict(
        schema=1,
        rows=len(rows),
        base_rows=len(base),
        new_records=len(source),
        new_views=len(extra),
        new_validation_photos=len(ev["phase2_selection"]),
        rows_sha256=hashlib.sha256((work / "rows.json").read_bytes()).hexdigest(),
        source_manifest_sha256=hashlib.sha256(
            (campaign / "phase2_source_manifest.json").read_bytes()
        ).hexdigest(),
        max_reference_logit_error=max(errors, default=0),
        source_revision="The former external source is now development data; a new external source is required for acceptance.",
        original_reference_assets=str(assets),
        seconds=time.time() - started,
    )
    (root / "complete.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
