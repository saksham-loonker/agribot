"""
5_download_plantvillage.py
==========================
Verifies the Tomato-Village dataset is present locally.

DATASET FACTS (Tomato-Village):
  4,525 images · 8 classes · 1 crop species (Tomato)
  Three variants:
    a) Multiclass Classification  ← WE USE THIS
    b) MultiLabel Classification
    c) Object Detection
  Pre-split into train / val / test.

  8 classes:
    Early_blight, Healthy, Late_blight, Leaf Miner,
    Magnesium Deficiency, Nitrogen Deficiency,
    Pottassium Deficiency, Spotted Wilt Virus

  Split sizes:
    train: 3,162 images
    val  :   902 images
    test :   461 images

  This is a real-world field dataset collected in Jodhpur and
  Jaipur districts of Rajasthan, India. Unlike PlantVillage
  (lab/controlled environment), Tomato-Village contains images
  captured in natural field conditions.

  Reference:
    Gehlot, M., Saxena, R.K. & Gandhi, G.C.
    "Tomato-Village": a dataset for end-to-end tomato disease
    detection in a real-world environment.
    Multimedia Systems (2023).

RUN:
  python 5_download_plantvillage.py
  python 5_download_plantvillage.py --src ./tomatovillage
"""

import argparse
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


DATASET_DIR  = "./tomatovillage"
VARIANT_NAME = "Variant-a(Multiclass Classification)"
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=DATASET_DIR,
                   help="Root of the Tomato-Village dataset")
    return p.parse_args()


def verify_dataset(src_dir: str):
    src = Path(src_dir)
    variant = src / VARIANT_NAME

    if not src.exists():
        raise SystemExit(
            f"[ERROR] Tomato-Village dataset not found at: {src}\n"
            f"        Place the dataset folder at: {Path(DATASET_DIR).resolve()}"
        )

    if not variant.exists():
        raise SystemExit(
            f"[ERROR] Variant-a folder not found at: {variant}\n"
            f"        Expected: {variant.resolve()}"
        )

    splits = ["train", "val", "test"]
    for split in splits:
        if not (variant / split).is_dir():
            raise SystemExit(
                f"[ERROR] Missing split folder: {variant / split}"
            )

    print(f"\n{'='*60}")
    print(f"  Tomato-Village Dataset — Verification")
    print(f"{'='*60}")
    print(f"  Path   : {variant.resolve()}")
    print(f"  Variant: Multiclass Classification\n")

    grand_total = 0
    all_classes = set()
    for split in splits:
        split_dir = variant / split
        class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        split_total = 0
        print(f"  {split}/")
        for d in class_dirs:
            all_classes.add(d.name)
            imgs = [f for f in d.iterdir() if f.suffix in IMAGE_EXTS]
            count = len(imgs)
            split_total += count
            print(f"    {d.name:<30} {count:>5} images")
        print(f"    {'─'*40}")
        print(f"    {'Total':<30} {split_total:>5}\n")
        grand_total += split_total

    print(f"  Classes : {len(all_classes)}")
    print(f"  Total   : {grand_total} images")
    print(f"\n  ✓ Dataset verified.")
    print(f"\n  Next step: python 6_prepare_clf.py")
    print(f"{'='*60}\n")


def main():
    args = parse_args()
    verify_dataset(args.src)


if __name__ == "__main__":
    main()
