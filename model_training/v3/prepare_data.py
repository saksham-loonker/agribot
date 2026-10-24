"""Build a leak-controlled training manifest and the locked diagnostic manifest.

Sources are pinned public revisions (see docs/training/RETRAINING_HANDOFF.md):
  * mamta-joshi-gehlot/Tomato-Village   Variant-A (8 supported classes, field)
  * pratikkayal/PlantDoc-Object-Detection-Dataset (field photos + VOC boxes)
  * spMohanty/PlantVillage-Dataset       deterministic diagnostic sample only

Design choices:
  * Entire source groups are split before any crop is generated. Tomato-Village
    groups are camera filenames without the augmentation suffix; PlantDoc groups
    are source photos. This removes the official TV train/test group overlap from
    the new run's train/selection/calibration/test partitions.
  * PlantDoc object boxes become training crops. Non-tomato and unsupported
    tomato conditions become Other; they are not mixed into a healthy/disease
    label for mixed-label photographs.
  * PlantVillage is evaluation-only. The 255 diagnostic files are the exact
    seeded sample used by the 2026-10-15 baseline and are never trained on.
  * The official source TEST splits remain a separate diagnostic replay. They
    are NOT locked-test evidence for the new model unless every related group
    was excluded from training; that is checked and reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model_training/v2/final"))
import canon  # noqa: E402

LABELS = [
    "Early_blight", "Healthy", "Late_blight", "Leaf Miner", "Magnesium Deficiency",
    "Nitrogen Deficiency", "Pottassium Deficiency", "Spotted Wilt Virus", "Other",
]
OTHER = LABELS.index("Other")
PINNED = {
    "tomatovillage": ("mamta-joshi-gehlot/Tomato-Village", "dfc246014558293987b381c1edb49398d9f570ac"),
    "plantdoc": ("pratikkayal/PlantDoc-Object-Detection-Dataset", "4730a233a555b30ee98e0879c63ad25d82407455"),
    "plantvillage": ("spMohanty/PlantVillage-Dataset", "7f7ecc7e1eaca78107e3affe7cb5abd9427e139a"),
}
PD_MAP = {
    "Tomato Early blight leaf": "Early_blight",
    "Tomato leaf late blight": "Late_blight",
    "Tomato leaf": "Healthy",
}
PV_MAP = {
    "Tomato___Early_blight": "Early_blight",
    "Tomato___Late_blight": "Late_blight",
    "Tomato___healthy": "Healthy",
}
PV_SEED = 20261015
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
CACHE_SIZE = 256


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pixel_hash(image: Image.Image) -> str:
    return sha256_bytes(image.convert("RGB").tobytes())


def source_url(source: str, path: str) -> str:
    repo, rev = PINNED[source]
    return f"https://raw.githubusercontent.com/{repo}/{rev}/{path}"


def canonical_crop(image: np.ndarray, box=None, size: int = CACHE_SIZE) -> np.ndarray:
    """Raw uint8 [size,size,3] crop using the app's canonical algorithm."""
    h, w = image.shape[:2]
    if box is not None:
        l, t, r, b = canon.crop_box(w, h, box, 0.10)
        if r - l < 2 or b - t < 2:
            raise ValueError(f"empty crop {box} for image {w}x{h}")
        image = image[t:b, l:r]
        h, w = image.shape[:2]
    side = min(w, h)
    if side < 2:
        raise ValueError(f"degenerate image {w}x{h}")
    ox, oy = (w - side) // 2, (h - side) // 2
    x = canon.resize(image[oy:oy + side, ox:ox + side], size, size)
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def allocate_splits(groups: list[dict], seed: int, fractions=(0.7, 0.1, 0.1, 0.1)) -> dict[str, str]:
    """Grouped, class-stratified, deterministic 70/10/10/10 partition."""
    by_label = defaultdict(list)
    for group in groups:
        by_label[group["primary_label"]].append(group)
    result = {}
    for label, items in sorted(by_label.items()):
        items = sorted(items, key=lambda g: g["group"])
        random.Random(f"{seed}:tv-pd:{label}").shuffle(items)
        n = len(items)
        if n == 1:
            counts = [1, 0, 0, 0]
        elif n < 4:
            counts = [max(1, n - 1), 0, 0, n - max(1, n - 1)]
        else:
            counts = [max(1, int(round(n * fractions[0]))), max(1, int(round(n * fractions[1]))),
                      max(1, int(round(n * fractions[2])))]
            counts.append(max(1, n - sum(counts)))
            while sum(counts) > n:
                i = max(range(4), key=lambda k: counts[k])
                counts[i] -= 1
        cursor = 0
        for name, count in zip(("train", "val", "calib", "test"), counts):
            for group in items[cursor:cursor + count]:
                result[group["group"]] = name
            cursor += count
    assert len(result) == len(groups)
    return result


def scan_tomato_village(root: Path) -> list[dict]:
    records = []
    base = root / "Variant-a(Multiclass Classification)"
    if not base.is_dir():
        raise SystemExit(f"Missing Tomato-Village Variant-A: {base}")
    for path in sorted(base.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        parts = path.relative_to(base).parts
        if len(parts) < 3 or parts[0] not in ("train", "val", "test"):
            continue
        raw_label, filename = parts[1], path.name
        if raw_label not in LABELS:
            continue
        group = "tomatovillage:" + path.stem.split("_")[0]
        source_path = "Variant-a(Multiclass Classification)/" + "/".join(parts)
        records.append(dict(
            source="tomatovillage", source_path=source_path, path=str(path),
            source_revision=PINNED["tomatovillage"][1], source_url=source_url("tomatovillage", source_path),
            raw_label=raw_label, label=raw_label, group=group, source_split=parts[0],
            box=None, objects=[], licence="See Tomato-Village repository terms; not redistributed in this repo",
            review_status="source_label_not_reviewed",
        ))
    return records


def scan_plantdoc(root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    import xml.etree.ElementTree as ET
    train, test, all_images = [], [], []
    for split in ("TRAIN", "TEST"):
        folder = root / split
        if not folder.is_dir():
            raise SystemExit(f"Missing PlantDoc split: {folder}")
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            xml = path.with_suffix(".xml")
            if not xml.exists():
                continue
            tree = ET.parse(xml).getroot()
            objects = []
            for obj in tree.findall("object"):
                box = obj.find("bndbox")
                if box is None:
                    continue
                try:
                    values = [float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
                except (TypeError, ValueError):
                    continue
                name = (obj.findtext("name") or "").strip()
                objects.append(dict(source_label=name, label=PD_MAP.get(name, "Other"),
                                    box=values, supported=name in PD_MAP))
            if not objects:
                continue
            unique = {o["label"] for o in objects}
            image_label = next(iter(unique)) if len(unique) == 1 else "Mixed"
            size = tree.find("size")
            annotation_size = None
            if size is not None:
                try:
                    annotation_size = [int(float(size.findtext(k))) for k in ("width", "height")]
                except (TypeError, ValueError):
                    pass
            source_path = f"{split}/{path.name}"
            with Image.open(path) as im:
                width, height = im.size
            record = dict(
                source="plantdoc", source_path=source_path, path=str(path),
                source_revision=PINNED["plantdoc"][1], source_url=source_url("plantdoc", source_path),
                raw_label=image_label, label=image_label, group="plantdoc:" + path.stem,
                source_split=split, box=None, objects=objects, annotation_size=annotation_size,
                width=width, height=height,
                licence="PlantDoc dataset terms; images not redistributed in this repo",
                review_status="source_label_not_reviewed",
            )
            (train if split == "TRAIN" else test).append(record)
            all_images.append(record)
    return train, test, all_images


def scan_plantvillage(root: Path) -> list[dict]:
    meta = root / "diagnostic255.json"
    if not meta.exists():
        raise SystemExit(f"Missing PlantVillage diagnostic list: {meta}")
    entries = json.loads(meta.read_text())
    records = []
    for entry in entries:
        local = Path(entry["path"])
        if not local.is_absolute():
            local = root / local
        if not local.exists():
            raise SystemExit(f"Missing PlantVillage diagnostic file: {local}")
        category = entry["category"]
        identifier = Path(entry["source_path"]).stem
        group = "plantvillage:" + category + ":" + identifier
        records.append(dict(
            source="plantvillage", source_path=entry["source_path"], path=str(local),
            source_revision=PINNED["plantvillage"][1], source_url=source_url("plantvillage", entry["source_path"]),
            raw_label=category, label=entry["label"], group=group,
            source_split="seeded_public_sample", box=None, objects=[],
            licence="PlantVillage dataset terms; evaluation-only sample",
            review_status="source_label_not_reviewed",
        ))
    return records


def build_training_manifest(tv: list[dict], pd_train: list[dict], pd_test: list[dict], seed: int):
    """Split TV/PD source groups, then expand PlantDoc objects into crops."""
    # Collapse exact duplicate source images across groups before splitting.
    seen = {}
    clean = []
    exclusions = []
    for record in tv + pd_train:
        data = Path(record["path"]).read_bytes()
        digest = sha256_bytes(data)
        if digest in seen:
            exclusions.append(dict(source_path=record["source_path"], reason="exact_duplicate",
                                   kept=seen[digest]))
            continue
        seen[digest] = record["source_path"]
        record["source_file_sha256"] = digest
        clean.append(record)

    groups = {}
    for record in clean:
        group = groups.setdefault(record["group"], dict(
            group=record["group"], source=record["source"], primary_label=record["label"]))
    # PlantDoc images with mixed labels get a stable primary label for stratification.
    for record in clean:
        if record["source"] == "plantdoc":
            labels = [o["label"] for o in record["objects"]]
            supported = [label for label in labels if label != "Other"]
            groups[record["group"]]["primary_label"] = (supported[0] if supported else "Other")
    split_of = allocate_splits(list(groups.values()), seed)

    crops = []
    # Tomato-Village: whole image is already a close-up leaf photo.
    for record in clean:
        if record["source"] != "tomatovillage":
            continue
        split = split_of[record["group"]]
        item = dict(record)
        item.update(id=f"tv:{record['source_split']}:{record['source_path'].split('/')[-1]}",
                    split=split, cache_index=None)
        crops.append(item)
    # PlantDoc: one sample per annotated leaf crop. Crops inherit the photo group split.
    for record in clean:
        if record["source"] != "plantdoc":
            continue
        split = split_of[record["group"]]
        for index, obj in enumerate(record["objects"]):
            box = obj["box"]
            if box[2] - box[0] < 16 or box[3] - box[1] < 16:
                continue
            # Author boxes were drawn against annotation_size. Skip boxes that
            # do not intersect the actual decoded image (dimension mismatches).
            w, h = record["width"], record["height"]
            if box[2] <= 0 or box[3] <= 0 or box[0] >= w or box[1] >= h:
                continue
            if record.get("annotation_size") not in (None, [w, h]):
                continue
            item = dict(record)
            item.update(id=f"pd:{record['source_path']}:{index}", split=split, box=box,
                        label=obj["label"], raw_label=obj["source_label"], objects=[])
            crops.append(item)

    # Reject duplicate pixel content inside the training cache as well.
    hashes = {}
    deduped = []
    for item in crops:
        digest = sha256_bytes(Path(item["path"]).read_bytes())
        key = (tuple(item["box"]) if item.get("box") is not None else None, digest)
        if key in hashes:
            exclusions.append(dict(id=item["id"], reason="duplicate_crop", kept=hashes[key]))
            continue
        hashes[key] = item["id"]
        deduped.append(item)

    # Locked test requires no source group overlap with train/val/calib.
    train_groups = {i["group"] for i in deduped if i["split"] != "test"}
    test_groups = {i["group"] for i in deduped if i["split"] == "test"}
    assert not (train_groups & test_groups), "group leakage between development and locked test"

    # PlantDoc source TEST images are diagnostic only. Compute exact TRAIN/TEST overlaps.
    train_hashes = {sha256_bytes(Path(r["path"]).read_bytes()) for r in pd_train}
    for record in pd_test:
        record["analysis_eligible"] = sha256_bytes(Path(record["path"]).read_bytes()) not in train_hashes
    return deduped, split_of, exclusions, pd_test


def cache_images(items: list[dict], out: Path):
    """Write a uint8 [n,256,256,3] memmap and assign cache_index."""
    usable = [item for item in items if item["split"] in ("train", "val", "calib")]
    mmap = np.lib.format.open_memmap(out / "cache_256.npy", mode="w+",
                                     dtype=np.uint8, shape=(len(usable), CACHE_SIZE, CACHE_SIZE, 3))
    for index, item in enumerate(usable):
        with Image.open(item["path"]) as im:
            image = np.asarray(im.convert("RGB"))
        mmap[index] = canonical_crop(image, item.get("box"))
        item["cache_index"] = index
    mmap.flush()
    del mmap
    # Locked test crops are cached separately so PyTorch candidate evaluation is cheap.
    test_items = [item for item in items if item["split"] == "test"]
    tmap = np.lib.format.open_memmap(out / "test_cache_256.npy", mode="w+",
                                     dtype=np.uint8, shape=(len(test_items), CACHE_SIZE, CACHE_SIZE, 3))
    for index, item in enumerate(test_items):
        with Image.open(item["path"]) as im:
            image = np.asarray(im.convert("RGB"))
        tmap[index] = canonical_crop(image, item.get("box"))
        item["test_cache_index"] = index
    tmap.flush()
    del tmap
    return usable, test_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(Path.home() / "agribot-data"))
    parser.add_argument("--out", default=str(Path.home() / "agribot-data/work"))
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    data_root, out = Path(args.data_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tv = scan_tomato_village(data_root / "Tomato-Village")
    pd_train, pd_test, _ = scan_plantdoc(data_root / "PlantDoc")
    pv = scan_plantvillage(data_root / "PlantVillage")
    training, split_of, exclusions, pd_test = build_training_manifest(tv, pd_train, pd_test, args.seed)

    # Diagnostics: exact source TEST splits + seeded PlantVillage sample.
    diagnostic = []
    for record in tv:
        if record["source_split"] != "test":
            continue
        item = dict(record)
        item.update(id=f"tv-test:{record['source_path'].split('/')[-1]}", split="source_test",
                    source_eligible=True, analysis_eligible=True)
        diagnostic.append(item)
    for record in pd_test:
        item = dict(record)
        item.update(id="pd-test:" + Path(record["source_path"]).stem, split="source_test",
                    source_eligible=record["analysis_eligible"], analysis_eligible=record["analysis_eligible"])
        diagnostic.append(item)
    for record in pv:
        item = dict(record)
        item.update(id="pv-diag:" + Path(record["source_path"]).stem + ":" + Path(record["path"]).stem,
                    split="seeded_public_sample", source_eligible=True, analysis_eligible=True)
        diagnostic.append(item)

    # A source-test photo is only clean evidence for the new model if its
    # related group is absent from all development partitions. Baseline replay
    # still uses analysis_eligible (its own source-overlap rule).
    dev_groups = {item["group"] for item in training if item["split"] != "test"}
    for item in diagnostic:
        item["new_model_group_overlap"] = item["group"] in dev_groups
        item["new_model_clean"] = item["analysis_eligible"] and not item["new_model_group_overlap"]

    # Locked final test for the new model: grouped TV test photos, PlantDoc
    # source TEST photos (minus exact TRAIN duplicates), and evaluation-only
    # PlantVillage diagnostics. These never enter training or calibration.
    locked = []
    for item in training:
        if item["source"] == "tomatovillage" and item["split"] == "test":
            whole = dict(item)
            whole.update(id="tv-locked:" + item["source_path"].split("/")[-1], split="locked_test",
                         source_eligible=True, analysis_eligible=True, new_model_clean=True,
                         new_model_group_overlap=False)
            locked.append(whole)
    for record in pd_test:
        item = dict(record)
        item.update(id="pd-locked:" + Path(record["source_path"]).stem, split="locked_test",
                    source_eligible=record["analysis_eligible"], analysis_eligible=record["analysis_eligible"],
                    new_model_clean=record["analysis_eligible"], new_model_group_overlap=False)
        locked.append(item)
    for record in pv:
        item = dict(record)
        item.update(id="pv-locked:" + Path(record["source_path"]).stem + ":" + Path(record["path"]).stem,
                    split="locked_test", source_eligible=True, analysis_eligible=True,
                    new_model_clean=True, new_model_group_overlap=False)
        locked.append(item)

    if not args.no_cache:
        cache_images(training, out)
    (out / "data_manifest.json").write_text(json.dumps(training, indent=1))
    (out / "diagnostic_manifest.json").write_text(json.dumps(diagnostic, indent=1))
    (out / "locked_test_manifest.json").write_text(json.dumps(locked, indent=1))
    (out / "data_exclusions.json").write_text(json.dumps(exclusions, indent=1))

    summary = dict(
        seed=args.seed,
        pinned_revisions={k: {"repo": v[0], "commit": v[1]} for k, v in PINNED.items()},
        classes=LABELS,
        training=dict(
            n=len(training),
            by_split={s: sum(i["split"] == s for i in training) for s in ("train", "val", "calib", "test")},
            by_source=dict(Counter(i["source"] for i in training)),
            by_split_source={s: dict(Counter(i["source"] for i in training if i["split"] == s))
                             for s in ("train", "val", "calib", "test")},
            by_split_label={s: dict(Counter(i["label"] for i in training if i["split"] == s))
                            for s in ("train", "val", "calib", "test")},
            groups={s: len({i["group"] for i in training if i["split"] == s})
                    for s in ("train", "val", "calib", "test")},
        ),
        diagnostic=dict(
            n=len(diagnostic),
            by_source=dict(Counter(i["source"] for i in diagnostic)),
            by_label=dict(Counter(i["label"] for i in diagnostic)),
            baseline_eligible=sum(i["analysis_eligible"] for i in diagnostic),
            new_model_clean=sum(i["new_model_clean"] for i in diagnostic),
            excluded_new_model_group_overlap=sum(i["new_model_group_overlap"] for i in diagnostic),
            locked_test=dict(
                n=len(locked),
                by_source=dict(Counter(i["source"] for i in locked)),
                by_label=dict(Counter(i["label"] for i in locked)),
                eligible=sum(i["analysis_eligible"] for i in locked),
            ),
        ),
        exclusions=dict(n=len(exclusions), by_reason=dict(Counter(e["reason"] for e in exclusions))),
    )
    (out / "split_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
