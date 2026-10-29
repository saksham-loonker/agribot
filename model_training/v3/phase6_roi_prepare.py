"""Add independently proposed Healthy leaf positions to the existing training cache.

The image's publisher label supplies Healthy supervision; the teacher supplies
only ROI geometry. All added rows are training rows. No weak diseased image
label is assigned to an individual leaf, and validation/reservation stay fixed.
"""

from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import shutil
import time

import numpy as np
from PIL import Image
import torch

from app_eval import canon, classifier_letterbox_input
from campaign_cnn_export import restore
from campaign_reference import safe_logits


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    a = ap.parse_args()
    p = Path(a.campaign)
    base = p / "phase5"
    root = p / "phase6_roi"
    root.mkdir(exist_ok=True)
    work = root / "cnn_inputs"
    work.mkdir(exist_ok=True)
    proposals_path = p / "phase6_teacher_leaf/training_predictions.jsonl"
    identity = dict(
        base_rows_sha256=sha(base / "cnn_inputs/rows.json"),
        proposals_sha256=sha(proposals_path),
        min_short_side_pixels=24,
        min_area_pixels=1024,
        views=["center", "full_roi"],
    )
    complete = root / "complete.json"
    if complete.exists():
        previous = json.loads(complete.read_text())
        assert previous["input_identity"] == identity and previous[
            "rows_sha256"
        ] == sha(work / "rows.json")
        return
    pixel_path = work / "teacher_healthy_pixels.npy"
    assert (
        not pixel_path.exists()
    ), "Partial cache requires an explicit identity audit before recovery"
    rows = json.loads((base / "cnn_inputs/rows.json").read_text())
    old_count = len(rows)
    proposals = [json.loads(line) for line in proposals_path.read_text().splitlines()]
    source = {
        r["id"]: r for r in json.loads((base / "development_manifest.json").read_text())
    }
    reserved = json.loads((base / "reserved_manifest.json").read_text())
    held_groups = {r["group"] for r in reserved}
    used = set()
    started = time.time()
    for proposal in proposals:
        record = source[proposal["id"]]
        assert (
            record["split"] == "train"
            and record["source"] == "tom2024"
            and record["label"] == "Healthy"
        )
        assert (
            record["group"] not in held_groups
            and record["file_sha256"] == proposal["file_sha256"]
        )
        assert sha(record["path"]) == record["file_sha256"]
        for box, score in zip(proposal["teacher_boxes"], proposal["teacher_scores"]):
            width, height = box[2] - box[0], box[3] - box[1]
            if min(width, height) < 24 or width * height < 1024:
                continue
            used.add(record["id"])
            for view in ["center", "full_roi"]:
                rows.append(
                    dict(
                        record,
                        set="training",
                        row=-1,
                        leaf=-1,
                        view="teacher_" + view,
                        input_variant=view,
                        box=box,
                        health_y=1.0,
                        known_y=1.0,
                        feature_index=-1,
                        roi_teacher_score=score,
                        roi_teacher_revision=proposal["teacher_identity"]["revision"],
                        annotation_provenance="Publisher Healthy image label; model-proposed leaf ROI geometry only",
                    )
                )
    extra = rows[old_count:]
    assert len(extra) >= 500
    assert not (
        {r["group"] for r in rows if r["split"] == "train"}
        & {r["group"] for r in rows if r["split"] == "val"}
    )
    assert not ({r["path"] for r in extra} & {r["path"] for r in reserved})
    pixels = np.lib.format.open_memmap(
        pixel_path, mode="w+", dtype=np.float32, shape=(len(extra), 224, 224, 3)
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
    pixels.flush()
    (work / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    parts = json.loads((base / "cnn_inputs/pixels.json").read_text())["center"]
    (work / "pixels.json").write_text(
        json.dumps(
            dict(
                center=[*parts, str(pixel_path)],
                input_variant="Per-row explicit center/full_roi, always canonical float32 RGB ImageNet normalization",
            ),
            indent=2,
        )
    )
    for filename in [
        "eval_rows.json",
        "development_manifest.json",
        "reserved_manifest.json",
    ]:
        shutil.copyfile(base / filename, root / filename)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = restore(p / "export_cnn_safe/pytorch_bundle.pt").cuda().eval()
    reference = []
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
            reference.append(rz.cpu().numpy())
            safe.append(model.fusion(rz, cz).cpu().numpy())
    np.save(
        root / "reference_logits.npy",
        np.concatenate([np.load(base / "reference_logits.npy"), *reference]),
    )
    np.save(
        root / "safe_reference_logits.npy", np.concatenate([safe_logits(base), *safe])
    )
    report = dict(
        schema=1,
        input_identity=identity,
        rows=len(rows),
        base_rows=old_count,
        new_views=len(extra),
        new_training_photos=len(used),
        rows_sha256=sha(work / "rows.json"),
        reserved_manifest_sha256=sha(root / "reserved_manifest.json"),
        reserved_predictions_opened=False,
        evaluation="Exact phase-five photo cohorts and original-detector ROI geometry unchanged; all added rows are training only.",
        supervision="Healthy from publisher image labels; leaf geometry is weak model output. No new per-leaf diseased/Other targets inferred.",
        original_reference_assets=str(p / "original_reference"),
        seconds=time.time() - started,
    )
    complete.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
