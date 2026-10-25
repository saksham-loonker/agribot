"""Prepare development views only; the prospective reservation is never inferred."""

from pathlib import Path
import argparse, concurrent.futures, hashlib, json, threading, time
import numpy as np
from PIL import Image
import torch
from app_eval import Models, image_record, canon, classifier_letterbox_input
from campaign_cnn_export import restore
from campaign_reference import safe_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    a = ap.parse_args()
    p = Path(a.campaign)
    q = p / "phase2"
    root = p / "phase5"
    work = root / "cnn_inputs"
    work.mkdir(exist_ok=True)
    if (root / "complete.json").exists():
        previous = json.loads((root / "complete.json").read_text())
        assert (
            previous["rows_sha256"]
            == hashlib.sha256((work / "rows.json").read_bytes()).hexdigest()
        )
        return
    assert (root / "sources_complete.json").exists()
    source = json.loads((root / "development_manifest.json").read_text())
    reserved = json.loads((root / "reserved_manifest.json").read_text())
    assert not ({r["group"] for r in source} & {r["group"] for r in reserved})
    base = json.loads((q / "cnn_inputs/rows.json").read_text())
    rows = [dict(r) for r in base]
    ev = json.loads((q / "eval_rows.json").read_text())
    ev["phase5_selection"] = []
    local = threading.local()
    started = time.time()

    def detect(record):
        if not hasattr(local, "model"):
            local.model = Models(p / "original_reference", 1)
        if record["split"] == "val":
            result = image_record(local.model, record)
            return record, result["detections"], result
        image = np.asarray(Image.open(record["path"]).convert("RGB"))
        return record, local.model.detect(image), None

    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        for n, (record, detections, photo) in enumerate(pool.map(detect, source)):
            common = dict(
                record,
                set="training",
                row=-1,
                leaf=-1,
                health_y=float(record["label"] == "Healthy"),
                known_y=(
                    -1.0
                    if record["label"] == "Diseased_unspecified"
                    else float(record["label"] != "Other")
                ),
                feature_index=-1,
            )
            # Scene-level disease is not assigned to each possibly healthy leaf.
            rows.append(
                dict(common, view="full_roi", input_variant="full_roi", box=None)
            )
            leaf_supervision = (
                record["label"] == "Healthy" or record["source"] == "swtt5wy2pr"
            )
            candidates = [
                d
                for d in detections
                if d[4] >= 0.5
                and (d[2] - d[0]) * (d[3] - d[1])
                >= 0.05 * record["width"] * record["height"]
            ]
            if leaf_supervision and candidates:
                d = max(candidates, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
                rows.append(
                    dict(common, view="full_roi", input_variant="full_roi", box=d[:4])
                )
                if record["label"] == "Healthy":
                    rows.append(
                        dict(common, view="center", input_variant="center", box=d[:4])
                    )
            if photo:
                index = len(ev["phase5_selection"])
                ev["phase5_selection"].append(photo)
                for leaf, observation in enumerate(photo["leaves"]):
                    rows.append(
                        dict(
                            common,
                            set="phase5_selection",
                            split="regression",
                            row=index,
                            leaf=leaf,
                            box=observation["box"],
                            view="selected",
                            input_variant="center",
                        )
                    )
            if n % 200 == 0:
                print(
                    json.dumps(
                        dict(
                            stage="detector_alignment",
                            photos=n,
                            total=len(source),
                            seconds=time.time() - started,
                        )
                    ),
                    flush=True,
                )
    extra = rows[len(base) :]
    (work / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    (root / "eval_rows.json").write_text(json.dumps(ev, indent=2) + "\n")
    path = work / "extra_pixels.npy"
    if path.exists():
        raise ValueError(
            "Partial pixel shard exists; verify its identity before reusing it"
        )
    pixels = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(len(extra), 224, 224, 3)
    )

    def prepare(row):
        image = np.asarray(Image.open(row["path"]).convert("RGB"))
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
                    json.dumps(
                        dict(
                            stage="pixels",
                            rows=start,
                            total=len(extra),
                            seconds=time.time() - started,
                        )
                    ),
                    flush=True,
                )
    pixels.flush()
    parts = json.loads((q / "cnn_inputs/pixels.json").read_text())["center"]
    (work / "pixels.json").write_text(
        json.dumps(
            dict(
                center=[*parts, str(path)],
                input_variant="Per-row explicit center or full_roi; values always float32 canonical normalized RGB",
            ),
            indent=2,
        )
    )
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = restore(p / "export_cnn_safe/pytorch_bundle.pt").cuda().eval()
    original = []
    safe = []
    with torch.inference_mode():
        for start in range(0, len(extra), 128):
            x = (
                torch.from_numpy(np.ascontiguousarray(pixels[start : start + 128]))
                .cuda()
                .permute(0, 3, 1, 2)
            )
            rz = model.backbone(x)
            cz = model.cnn(
                (x * model.canonical_std + model.canonical_mean - model.cnn_mean)
                / model.cnn_std
            )
            original.append(rz.cpu())
            safe.append(model.fusion(rz, cz).cpu())
    ref = np.concatenate(
        [np.load(q / "reference_logits.npy"), torch.cat(original).numpy()]
    )
    safe_reference = np.concatenate([safe_logits(q), torch.cat(safe).numpy()])
    assert len(rows) == len(ref) == len(safe_reference)
    np.save(root / "reference_logits.npy", ref)
    np.save(root / "safe_reference_logits.npy", safe_reference)
    errors = []
    for i, row in enumerate(rows):
        if row["set"] == "phase5_selection":
            errors.append(
                float(
                    abs(
                        ref[i]
                        - np.asarray(
                            ev["phase5_selection"][row["row"]]["leaves"][row["leaf"]][
                                "member_logits"
                            ][0]
                        )
                    ).max()
                )
            )
    assert max(errors, default=0) < 0.002, max(errors)
    report = dict(
        schema=1,
        rows=len(rows),
        new_views=len(extra),
        source_photos=len(source),
        new_validation_photos=len(ev["phase5_selection"]),
        rows_sha256=hashlib.sha256((work / "rows.json").read_bytes()).hexdigest(),
        development_manifest_sha256=hashlib.sha256(
            (root / "development_manifest.json").read_bytes()
        ).hexdigest(),
        reserved_manifest_sha256=hashlib.sha256(
            (root / "reserved_manifest.json").read_bytes()
        ).hexdigest(),
        reserved_predictions_opened=False,
        original_reference_assets=str(p / "original_reference"),
        max_original_reference_parity=max(errors, default=0),
        preprocessing="Existing centered canonical views plus explicitly marked full-ROI views; ImageNet normalization unchanged",
        coarse_label_rule="Unspecified scene disease is binary-only on the full scene; never a named diagnosis on each leaf",
        seconds=time.time() - started,
    )
    (root / "complete.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
