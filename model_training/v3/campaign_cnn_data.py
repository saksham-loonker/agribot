"""Exact pixel caches for independent CNN specialists and full-leaf ablations.

The clean author training partition of the binary source now becomes binary
fitting/development data. Its author validation partition stays sealed. Coarse
disease labels never become nine-class Other targets.
"""

from pathlib import Path
import collections, concurrent.futures, hashlib, json, re, time
import numpy as np
from PIL import Image
from app_eval import canon

ROOT = Path("/home/genesis/agribot-data/recognition_campaign_20261024")


def letterbox_classifier(image, box, size=224):
    if box is not None:
        l, t, r, b = canon.crop_box(image.shape[1], image.shape[0], box, 0.1)
        image = image[t:b, l:r]
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    px, py = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), 114.0, np.float32)
    out[py : py + nh, px : px + nw] = canon.resize(image, nw, nh)
    return (out / 255.0 - canon.MEAN) / canon.STD


def main():
    out = ROOT / "cnn_inputs"
    out.mkdir(exist_ok=True)
    original = json.loads((ROOT / "dense_aligned/feature_rows.json").read_text())
    rows = []
    labels = [
        "Early_blight",
        "Healthy",
        "Late_blight",
        "Leaf Miner",
        "Magnesium Deficiency",
        "Nitrogen Deficiency",
        "Pottassium Deficiency",
        "Spotted Wilt Virus",
        "Other",
    ]
    for i, r in enumerate(original):
        row = dict(r)
        row["feature_index"] = i
        row["split"] = r["split"] if r["set"] == "training" else "regression"
        row["health_y"] = (
            float(r["label"] == "Healthy") if r["label"] in labels else -1.0
        )
        row["known_y"] = float(r["label"] != "Other") if r["label"] in labels else -1.0
        rows.append(row)
    binary = json.loads((ROOT / "binary_diagnostic_manifest.json").read_text())
    acceptance = json.loads((ROOT / "binary_acceptance_manifest.json").read_text())
    reserved = json.loads((ROOT / "locked_binary_manifest.json").read_text())
    reserved = [r for r in reserved if "/Images/val/" in r["path"]]

    def group(r):
        stem = Path(r["path"]).stem
        number = int(re.search(r"\d+", stem).group())
        return "binary:" + stem[0] + ":capture_block_" + str(number // 16)

    reserved_groups = {group(r) for r in reserved}
    reserved_files = {r["file_sha256"] for r in reserved}
    excluded = []
    for r in binary:
        key = group(r)
        if (
            key in reserved_groups
            or r["file_sha256"] in reserved_files
            or any(
                (int(r["phash"], 16) ^ int(t["phash"], 16)).bit_count() <= 2
                for t in reserved
            )
        ):
            excluded.append(r["id"])
            continue
        file = Path(r["path"])
        annotation = (
            file.parents[2] / "labels" / file.parent.name / (file.stem + ".txt")
        )
        with Image.open(file) as image:
            w, h = image.size
        split = (
            "val"
            if int(hashlib.sha256(("cnn17:" + key).encode()).hexdigest()[:8], 16) % 100
            >= 80
            else "train"
        )
        for n, line in enumerate(annotation.read_text().splitlines()):
            class_id, cx, cy, bw, bh = map(float, line.split())
            assert int(class_id) == (0 if r["label"] == "Healthy" else 1)
            box = [
                (cx - bw / 2) * w,
                (cy - bh / 2) * h,
                (cx + bw / 2) * w,
                (cy + bh / 2) * h,
            ]
            row = dict(r)
            row.update(
                id=r["id"] + f":annotated:{n}",
                group=key,
                split=split,
                set="training",
                box=box,
                feature_index=-1,
                health_y=float(class_id == 0),
                known_y=1.0 if class_id == 0 else -1.0,
            )
            rows.append(row)
    assert not (
        {r["group"] for r in rows if r["split"] == "train"}
        & {r["group"] for r in rows if r["split"] == "val"}
    )
    (out / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    protocol = dict(
        binary_fitting_revision="Before any binary-source predictions; author training only, clean agreement labels only",
        author_validation_sealed=len(acceptance),
        binary_training_excluded_overlap=excluded,
        counts={
            str(k): v
            for k, v in collections.Counter(
                (r["source"], r["split"], r["health_y"], r["known_y"]) for r in rows
            ).items()
        },
        original_feature_rows_sha256=hashlib.sha256(
            (ROOT / "dense_aligned/feature_rows.json").read_bytes()
        ).hexdigest(),
        rows_sha256=hashlib.sha256((out / "rows.json").read_bytes()).hexdigest(),
        acceptance_sha256=hashlib.sha256(
            (ROOT / "binary_acceptance_manifest.json").read_bytes()
        ).hexdigest(),
        preprocessing=[
            "exact center crop",
            "entire padded ROI letterboxed with RGB 114, same ImageNet normalization",
        ],
    )
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    arrays = {}
    for policy in ["center", "letterbox"]:
        path = out / (policy + ".npy")
        if path.exists():
            raise ValueError("Pixel cache exists; verify identity before replacing it")
        arrays[policy] = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(len(rows), 224, 224, 3)
        )

    def prepare(r):
        with Image.open(r["path"]) as im:
            image = np.asarray(im.convert("RGB"))
        return canon.classifier_input(image, 224, r["box"])[0], letterbox_classifier(
            image, r["box"]
        )

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for i in range(0, len(rows), 64):
            prepared = list(pool.map(prepare, rows[i : i + 64]))
            for j, (center, letterbox) in enumerate(prepared):
                arrays["center"][i + j] = center
                arrays["letterbox"][i + j] = letterbox
            if i % 1024 == 0:
                print(
                    json.dumps(
                        dict(
                            rows=i,
                            total=len(rows),
                            seconds=round(time.time() - start, 1),
                        )
                    ),
                    flush=True,
                )
    for data in arrays.values():
        data.flush()
    (out / "complete.json").write_text(
        json.dumps(
            dict(
                rows=len(rows),
                seconds=time.time() - start,
                rows_sha256=protocol["rows_sha256"],
            ),
            indent=2,
        )
    )
    print("Both pixel policies cached", len(rows), flush=True)


if __name__ == "__main__":
    main()
