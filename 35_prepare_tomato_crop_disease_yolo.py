"""
Prepare tomato_crop_disease for YOLO detector training/evaluation.

The source dataset is laid out as:

  tomato_crop_disease/
    images/*.jpg
    labels/*.txt

Its labels use class id 1. For a single-class YOLO detector we remap that id to
0 and create a train/val/test split with the normal Ultralytics directory
layout.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(description="Prepare tomato_crop_disease YOLO dataset")
    p.add_argument("--src", default="tomato_crop_disease")
    p.add_argument("--out", default="tomato_crop_disease_yolo")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--all-test", action="store_true", help="Put all images in test split only")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def collect_pairs(src: Path) -> list[tuple[Path, Path]]:
    images_dir = src / "images"
    labels_dir = src / "labels"
    if not images_dir.is_dir():
        raise SystemExit(f"[ERROR] Images dir not found: {images_dir}")
    if not labels_dir.is_dir():
        raise SystemExit(f"[ERROR] Labels dir not found: {labels_dir}")

    pairs = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                pairs.append((image_path, label_path))
    if not pairs:
        raise SystemExit(f"[ERROR] No image/label pairs found under: {src}")
    return pairs


def split_pairs(pairs: list[tuple[Path, Path]], train_frac: float, val_frac: float, seed: int):
    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)
    if train_frac < 0 or val_frac < 0 or train_frac + val_frac >= 1.0:
        raise SystemExit("[ERROR] Fractions must satisfy train >= 0, val >= 0, train + val < 1")
    n = len(shuffled)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return {"train": train, "val": val, "test": test}


def remap_label(src: Path, dest: Path):
    lines = []
    for line in src.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        parts[0] = "0"
        lines.append(" ".join(parts[:5]))
    dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def copy_split(split_name: str, pairs: list[tuple[Path, Path]], out: Path):
    image_out = out / split_name / "images"
    label_out = out / split_name / "labels"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    for image_path, label_path in pairs:
        shutil.copy2(image_path, image_out / image_path.name)
        remap_label(label_path, label_out / f"{image_path.stem}.txt")


def write_yaml(out: Path):
    yaml_text = (
        f"path: {out.resolve()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "nc: 1\n"
        "names: ['tomato_crop_disease']\n"
    )
    (out / "data.yaml").write_text(yaml_text, encoding="utf-8")


def main():
    args = parse_args()
    src = Path(args.src)
    out = Path(args.out)
    if not src.is_dir():
        raise SystemExit(f"[ERROR] Source dataset not found: {src}")
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"[ERROR] Output exists; use --overwrite: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(src)
    if args.all_test:
        splits = {"train": [], "val": [], "test": pairs}
    else:
        splits = split_pairs(pairs, args.train_frac, args.val_frac, args.seed)

    for split_name, split_pairs_ in splits.items():
        copy_split(split_name, split_pairs_, out)
        print(f"{split_name}: {len(split_pairs_)} image/label pairs")

    write_yaml(out)
    print(f"Wrote YOLO dataset -> {out}")
    print(f"Data YAML          -> {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
