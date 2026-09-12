"""
2_prepare.py
============
Builds a single-class YOLO detector dataset from the local
Crop_detection_tomato_chilli download.

Expected source layout:
  Crop_detection_tomato_chilli/
    Chilli/
      images/*.jpg
      labels/*.txt
    Tomato/
      images/*.jpg
      labels/*.txt

This source is not internally consistent:
  - Tomato labels are already YOLO-normalized in most files.
  - Chilli labels use YOLO center/width/height semantics, but in pixel units.

This script reads real image sizes and normalizes both formats automatically.
All source class IDs are collapsed to class 0 ("crop") because the detector's
job is only to crop the relevant plant object cleanly and consistently.

Output:
  dataset/
    data.yaml
    train/images
    train/labels
    val/images
    val/labels
    test/images
    test/labels
    split_manifest.json

Run:
  python 2_prepare.py
  python 2_prepare.py --src "./Crop_detection_tomato_chilli"
"""

import argparse
import json
import random
import shutil
import struct
import sys
from collections import defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


DEFAULT_SRC = Path(__file__).resolve().parent / "Crop_detection_tomato_chilli"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        default=str(DEFAULT_SRC),
        help="Path to Crop_detection_tomato_chilli root",
    )
    p.add_argument(
        "--out",
        default="./dataset",
        help="Output YOLO dataset directory",
    )
    p.add_argument("--train-frac", type=float, default=0.80)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def validate_split_fracs(train_frac: float, val_frac: float, test_frac: float):
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-9:
        raise SystemExit(
            f"[ERROR] Split fractions must sum to 1.0, got {total:.6f}."
        )


def read_image_shape(img_path: Path) -> tuple[int, int]:
    try:
        with open(img_path, "rb") as handle:
            header = handle.read(32)

            if header.startswith(b"\x89PNG\r\n\x1a\n"):
                w, h = struct.unpack(">II", header[16:24])
                return int(w), int(h)

            if header[:2] == b"BM":
                w, h = struct.unpack("<II", header[18:26])
                return int(w), int(h)

            if header[:2] == b"\xff\xd8":
                handle.seek(2)
                while True:
                    marker_prefix = handle.read(1)
                    if not marker_prefix:
                        break
                    if marker_prefix != b"\xff":
                        continue
                    marker = handle.read(1)
                    while marker == b"\xff":
                        marker = handle.read(1)
                    if not marker or marker in {b"\xd8", b"\xd9"}:
                        continue

                    size_bytes = handle.read(2)
                    if len(size_bytes) != 2:
                        break
                    segment_size = struct.unpack(">H", size_bytes)[0]
                    if segment_size < 2:
                        break

                    if marker in {
                        b"\xc0", b"\xc1", b"\xc2", b"\xc3",
                        b"\xc5", b"\xc6", b"\xc7",
                        b"\xc9", b"\xca", b"\xcb",
                        b"\xcd", b"\xce", b"\xcf",
                    }:
                        segment = handle.read(segment_size - 2)
                        if len(segment) < 5:
                            break
                        h, w = struct.unpack(">HH", segment[1:5])
                        return int(w), int(h)

                    handle.seek(segment_size - 2, 1)
    except OSError:
        pass

    try:
        from PIL import Image

        with Image.open(img_path) as image:
            w, h = image.size
            return int(w), int(h)
    except Exception:
        raise ValueError(f"cannot read image size: {img_path}")


def normalize_box(
    cx: float,
    cy: float,
    bw: float,
    bh: float,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    # Auto-detect whether the source uses normalized YOLO values or pixel units.
    if max(abs(cx), abs(cy), abs(bw), abs(bh)) > 1.5:
        cx /= img_w
        bw /= img_w
        cy /= img_h
        bh /= img_h

    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    bw = max(1e-6, min(1.0, bw))
    bh = max(1e-6, min(1.0, bh))
    return cx, cy, bw, bh


def remap_label_file(src: Path, img_w: int, img_h: int) -> list[str]:
    lines_out = []
    with open(src, "r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            parts = raw_line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cx, cy, bw, bh = [float(value) for value in parts[1:5]]
            except ValueError:
                continue

            cx, cy, bw, bh = normalize_box(cx, cy, bw, bh, img_w, img_h)
            lines_out.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines_out


def collect_samples(src_root: Path) -> list[dict]:
    samples = []
    for species_dir in sorted(d for d in src_root.iterdir() if d.is_dir()):
        img_dir = species_dir / "images"
        lbl_dir = species_dir / "labels"
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            continue

        for img_path in sorted(img_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            samples.append(
                {
                    "species": species_dir.name,
                    "image": img_path,
                    "label": lbl_path,
                }
            )

    if not samples:
        raise SystemExit(f"[ERROR] No image/label pairs found in: {src_root}")
    return samples


def stratified_split(
    samples: list[dict],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    by_species = defaultdict(list)
    for sample in samples:
        by_species[sample["species"]].append(sample)

    splits = {split: [] for split in SPLITS}
    for species, items in sorted(by_species.items()):
        shuffled = items[:]
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_train = int(n_total * train_frac)
        n_val = int(n_total * val_frac)

        splits["train"].extend(shuffled[:n_train])
        splits["val"].extend(shuffled[n_train:n_train + n_val])
        splits["test"].extend(shuffled[n_train + n_val:])

        if n_total > 0 and n_train == 0:
            splits["train"].append(splits["test"].pop())

    for split in SPLITS:
        rng.shuffle(splits[split])
    return splits


def prepare_output_root(out_root: Path):
    for split in SPLITS:
        split_dir = out_root / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)


def write_dataset(
    split_map: dict[str, list[dict]],
    out_root: Path,
) -> tuple[dict, list[dict]]:
    stats = {
        split: {"images": 0, "boxes": 0, "species": defaultdict(int)}
        for split in SPLITS
    }
    manifest = []
    used_stems = set()

    for split in SPLITS:
        img_dst = out_root / split / "images"
        lbl_dst = out_root / split / "labels"

        for index, sample in enumerate(
            tqdm(split_map[split], desc=f"  Writing {split}", leave=False),
            start=1,
        ):
            img_path = sample["image"]
            lbl_path = sample["label"]
            species = sample["species"]

            try:
                img_w, img_h = read_image_shape(img_path)
            except ValueError as exc:
                print(f"[WARN] {exc}")
                continue

            label_lines = remap_label_file(lbl_path, img_w, img_h)
            if not label_lines:
                continue

            stem = f"{species.lower()}_{img_path.stem}"
            while stem in used_stems:
                stem = f"{species.lower()}_{img_path.stem}_{index}"
            used_stems.add(stem)

            dst_img = img_dst / f"{stem}{img_path.suffix.lower()}"
            dst_lbl = lbl_dst / f"{stem}.txt"

            shutil.copy2(img_path, dst_img)
            dst_lbl.write_text("\n".join(label_lines), encoding="utf-8")

            stats[split]["images"] += 1
            stats[split]["boxes"] += len(label_lines)
            stats[split]["species"][species] += 1
            manifest.append(
                {
                    "split": split,
                    "species": species,
                    "image": str(dst_img.resolve()),
                    "label": str(dst_lbl.resolve()),
                    "source_image": str(img_path.resolve()),
                    "source_label": str(lbl_path.resolve()),
                }
            )

    return stats, manifest


def write_yaml(out_root: Path) -> Path:
    yaml_path = out_root / "data.yaml"
    yaml_text = "\n".join(
        [
            f"path: '{out_root.resolve()}'",
            "train: train/images",
            "val: val/images",
            "test: test/images",
            "nc: 1",
            "names: ['crop']",
            "",
        ]
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def write_manifest(out_root: Path, manifest: list[dict]):
    manifest_path = out_root / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path


def main():
    args = parse_args()
    validate_split_fracs(args.train_frac, args.val_frac, args.test_frac)

    src_root = Path(args.src)
    out_root = Path(args.out)

    if not src_root.exists():
        raise SystemExit(f"[ERROR] Source dataset not found: {src_root}")

    print(f"\n{'=' * 64}")
    print("  Preparing chilli + tomato crop detector dataset")
    print(f"{'=' * 64}")
    print(f"  Source : {src_root.resolve()}")
    print(f"  Output : {out_root.resolve()}")
    print(
        "  Split  : "
        f"{int(args.train_frac * 100)}/{int(args.val_frac * 100)}/{int(args.test_frac * 100)}"
    )
    print(f"  Seed   : {args.seed}\n")

    samples = collect_samples(src_root)
    split_map = stratified_split(samples, args.train_frac, args.val_frac, args.seed)

    out_root.mkdir(parents=True, exist_ok=True)
    prepare_output_root(out_root)
    stats, manifest = write_dataset(split_map, out_root)
    yaml_path = write_yaml(out_root)
    manifest_path = write_manifest(out_root, manifest)

    total_images = 0
    total_boxes = 0
    print("  Results:")
    for split in SPLITS:
        split_stats = stats[split]
        total_images += split_stats["images"]
        total_boxes += split_stats["boxes"]
        species_bits = ", ".join(
            f"{species}:{count}"
            for species, count in sorted(split_stats["species"].items())
        ) or "none"
        print(
            f"    {split:5s} -> {split_stats['images']:3d} images, "
            f"{split_stats['boxes']:4d} boxes  [{species_bits}]"
        )
    print(f"    total -> {total_images:3d} images, {total_boxes:4d} boxes")
    print(f"\n  Wrote : {yaml_path}")
    print(f"  Wrote : {manifest_path}")
    print("\n  Next step: python 3_train.py")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
