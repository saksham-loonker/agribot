"""
Build a clean single-class crop detector dataset from every usable detector
dataset in this workspace.

The archive YOLO/VOC annotation folders in this project may contain HTTP error
text instead of labels. This script validates every label line and skips invalid
annotation files instead of letting bad labels poison training.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Pair:
    image: Path
    label: Path
    split: str
    source: str
    label_kind: str


def parse_args():
    p = argparse.ArgumentParser(description="Prepare combined single-class crop detector YOLO dataset")
    p.add_argument("--out", default="combined_crop_detector_yolo")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument(
        "--include-tomato-test",
        action="store_true",
        help="Include tomato_crop_disease_yolo/test in training data. Leave off for honest held-out benchmarking.",
    )
    return p.parse_args()


def safe_name(path: Path) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in path.as_posix()).strip("_")


def image_for_stem(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
        candidate = images_dir / f"{stem}{ext.upper()}"
        if candidate.exists():
            return candidate
    return None


def valid_yolo_lines(label_path: Path) -> list[str]:
    lines = []
    try:
        raw_lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []
    for raw in raw_lines:
        parts = raw.strip().split()
        if len(parts) < 5:
            return []
        try:
            [float(x) for x in parts[:5]]
        except ValueError:
            return []
        cx, cy, bw, bh = [float(x) for x in parts[1:5]]
        if bw <= 0 or bh <= 0:
            continue
        if max(abs(cx), abs(cy), abs(bw), abs(bh)) > 1.5:
            return []
        lines.append(f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
    return lines


def voc_to_yolo_lines(xml_path: Path) -> list[str]:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []
    if root.tag != "annotation":
        return []

    try:
        width = float(root.findtext("size/width") or 0)
        height = float(root.findtext("size/height") or 0)
    except ValueError:
        return []
    if width <= 0 or height <= 0:
        return []

    lines = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            xmin = float(box.findtext("xmin") or 0)
            ymin = float(box.findtext("ymin") or 0)
            xmax = float(box.findtext("xmax") or 0)
            ymax = float(box.findtext("ymax") or 0)
        except ValueError:
            continue
        xmin = max(0.0, min(width, xmin))
        xmax = max(0.0, min(width, xmax))
        ymin = max(0.0, min(height, ymin))
        ymax = max(0.0, min(height, ymax))
        bw = xmax - xmin
        bh = ymax - ymin
        if bw <= 1 or bh <= 1:
            continue
        cx = xmin + bw / 2.0
        cy = ymin + bh / 2.0
        lines.append(f"0 {cx / width:.8f} {cy / height:.8f} {bw / width:.8f} {bh / height:.8f}")
    return lines


def collect_yolo_split(root: Path, split: str, label_dir_name: str, source: str) -> list[Pair]:
    images_dir = root / split / "images"
    labels_dir = root / split / label_dir_name
    if not images_dir.is_dir() or not labels_dir.is_dir():
        return []
    pairs = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = image_for_stem(images_dir, label_path.stem)
        if image_path is not None:
            pairs.append(Pair(image=image_path, label=label_path, split=split, source=source, label_kind="yolo"))
    return pairs


def collect_voc_split(root: Path, split: str, source: str) -> list[Pair]:
    images_dir = root / split / "images"
    labels_dir = root / split / "pascal_voc"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        return []
    pairs = []
    for label_path in sorted(labels_dir.glob("*.xml")):
        image_path = image_for_stem(images_dir, label_path.stem)
        if image_path is not None:
            pairs.append(Pair(image=image_path, label=label_path, split=split, source=source, label_kind="voc"))
    return pairs


def collect_candidates(include_tomato_test: bool) -> list[Pair]:
    candidates = []
    for split in ("train", "val", "test"):
        candidates.extend(collect_yolo_split(Path("dataset"), split, "labels", "dataset"))

    for split in ("train", "val", "test"):
        if split == "test" and not include_tomato_test:
            continue
        candidates.extend(collect_yolo_split(Path("tomato_crop_disease_yolo"), split, "labels", "tomato_crop_disease_yolo"))

    for archive_root in (Path("archive/Variant-c(Object Detection)"), Path("archive (2)/Variant-c(Object Detection)")):
        source = safe_name(archive_root)
        for split in ("train", "val"):
            candidates.extend(collect_yolo_split(archive_root, split, "yolo", source))
            candidates.extend(collect_voc_split(archive_root, split, source))
    return candidates


def label_lines(pair: Pair) -> list[str]:
    if pair.label_kind == "voc":
        return voc_to_yolo_lines(pair.label)
    return valid_yolo_lines(pair.label)


def file_digest(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_pairs(pairs: list[tuple[Pair, list[str]]], val_frac: float, seed: int):
    train = []
    val = []
    remainder = []
    for pair, lines in pairs:
        if pair.split == "train":
            train.append((pair, lines))
        elif pair.split == "val":
            val.append((pair, lines))
        else:
            remainder.append((pair, lines))

    random.Random(seed).shuffle(remainder)
    val_count = int(round(len(remainder) * val_frac))
    val.extend(remainder[:val_count])
    train.extend(remainder[val_count:])
    return {"train": train, "val": val}


def write_pair(out: Path, split: str, index: int, pair: Pair, lines: list[str]):
    stem = f"{index:06d}_{safe_name(Path(pair.source) / pair.image.stem)}"
    image_out = out / split / "images" / f"{stem}{pair.image.suffix.lower()}"
    label_out = out / split / "labels" / f"{stem}.txt"
    image_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pair.image, image_out)
    label_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_yaml(out: Path):
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "nc: 1\n"
        "names: ['crop']\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    out = Path(args.out)
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"[ERROR] Output exists; use --overwrite: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    candidates = collect_candidates(args.include_tomato_test)
    accepted = []
    skipped = 0
    duplicate_images = 0
    seen_images = set()
    source_counts: dict[str, int] = {}
    for pair in candidates:
        lines = label_lines(pair)
        if not lines:
            skipped += 1
            continue
        digest = file_digest(pair.image)
        if digest in seen_images:
            duplicate_images += 1
            continue
        seen_images.add(digest)
        accepted.append((pair, lines))
        source_counts[pair.source] = source_counts.get(pair.source, 0) + 1

    if not accepted:
        raise SystemExit("[ERROR] No valid image/label pairs found")

    splits = split_pairs(accepted, args.val_frac, args.seed)
    for split, rows in splits.items():
        for index, (pair, lines) in enumerate(rows):
            write_pair(out, split, index, pair, lines)

    write_yaml(out)
    print(f"Candidates       : {len(candidates)}")
    print(f"Accepted         : {len(accepted)}")
    print(f"Skipped invalid  : {skipped}")
    print(f"Skipped duplicate: {duplicate_images}")
    print(f"Train pairs      : {len(splits['train'])}")
    print(f"Val pairs        : {len(splits['val'])}")
    print("Sources:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")
    print(f"Dataset YAML     : {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
