"""
6_prepare_clf.py
================
Copies Tomato-Village Variant-a (Multiclass) into clf_dataset/,
writes a class_map.json, and prints imbalance statistics.

WHY THIS STEP IS NEEDED:
  Tomato-Village already has train/val/test splits, but we
  copy them into a standard clf_dataset/ layout and generate
  class_map.json + split_stats.json for later scripts.

  The dataset is class-imbalanced:
    - Leaf Miner              : ~1024 images
    - Pottassium Deficiency   : ~72   images
  We record per-class counts so the training script can
  apply a weighted sampler to compensate for imbalance.

  8 classes:
    Early_blight, Healthy, Late_blight, Leaf Miner,
    Magnesium Deficiency, Nitrogen Deficiency,
    Pottassium Deficiency, Spotted Wilt Virus

OUTPUT:
  clf_dataset/
    train/<ClassName>/image.jpg ...
    val/<ClassName>/image.jpg ...
    test/<ClassName>/image.jpg ...
    class_map.json     {0: "Early_blight", 1: ..., 7: ...}
    split_stats.json   {class_name: {train: N, val: N, test: N}}

RUN:
  python 6_prepare_clf.py
  python 6_prepare_clf.py --src "./tomatovillage/Variant-a(Multiclass Classification)" --out ./clf_dataset
"""

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15
SEED       = 42

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=None,
                   help="Path to either an unsplit class-folder dataset or a dataset already split into train/val/test")
    p.add_argument("--out", default="./clf_dataset",
                   help="Output folder for split dataset")
    return p.parse_args()


def resolve_src(user_src: str | None) -> Path:
    if user_src:
        return Path(user_src)

    candidates = [
        Path("./tomatovillage/Variant-a(Multiclass Classification)"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def collect_images(src: Path) -> dict[str, list[Path]]:
    """Returns {class_name: [image_paths]}."""
    data = {}
    class_dirs = sorted([d for d in src.iterdir() if d.is_dir()])
    if not class_dirs:
        raise SystemExit(f"[ERROR] No class subdirectories found in: {src}")
    for d in class_dirs:
        imgs = sorted([f for f in d.iterdir() if f.suffix in IMAGE_EXTS])
        if imgs:
            data[d.name] = imgs
    return data


def is_pre_split_dataset(src: Path) -> bool:
    return all((src / split).is_dir() for split in ("train", "val", "test"))


def collect_split_dataset(src: Path) -> tuple[dict, dict, dict]:
    train = collect_images(src / "train")
    val   = collect_images(src / "val")
    test  = collect_images(src / "test")
    return train, val, test


def stratified_split(
    data: dict[str, list[Path]],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[dict, dict, dict]:
    rng = random.Random(seed)
    train, val, test = defaultdict(list), defaultdict(list), defaultdict(list)

    for cls, imgs in data.items():
        shuffled = imgs[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)
        # remainder goes to test
        train[cls] = shuffled[:n_train]
        val[cls]   = shuffled[n_train : n_train + n_val]
        test[cls]  = shuffled[n_train + n_val :]

    return dict(train), dict(val), dict(test)


def copy_split(
    split_data: dict[str, list[Path]],
    split_name: str,
    out_root: Path,
):
    for cls, imgs in tqdm(split_data.items(), desc=f"  Copying {split_name}", leave=False):
        dst_dir = out_root / split_name / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy2(img, dst_dir / img.name)


def main():
    args = parse_args()
    src  = resolve_src(args.src)
    out  = Path(args.out)

    if not src.exists():
        raise SystemExit(
            f"[ERROR] Source not found: {src}\n"
            f"        Run python 5_download_plantvillage.py first to verify dataset."
        )

    print(f"\n{'='*60}")
    print(f"  Tomato-Village → clf_dataset")
    print(f"{'='*60}")
    print(f"  Source : {src.resolve()}")
    print(f"  Output : {out.resolve()}")
    if is_pre_split_dataset(src):
        print("  Split  : using existing train / val / test folders")
    else:
        print(f"  Split  : {int(TRAIN_FRAC*100)} / {int(VAL_FRAC*100)} / {int(TEST_FRAC*100)}")
    print(f"  Seed   : {SEED}\n")

    out.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        split_dir = out / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)

    if is_pre_split_dataset(src):
        train, val, test = collect_split_dataset(src)
        class_names = sorted(set(train) | set(val) | set(test))
        data = {
            cls: train.get(cls, []) + val.get(cls, []) + test.get(cls, [])
            for cls in class_names
        }
    else:
        data = collect_images(src)
        train, val, test = stratified_split(data, TRAIN_FRAC, VAL_FRAC, SEED)
        class_names = sorted(data.keys())

    print(f"  Found {len(data)} classes, "
          f"{sum(len(v) for v in data.values())} total images.")

    copy_split(train, "train", out)
    copy_split(val,   "val",   out)
    copy_split(test,  "test",  out)

    # ── class_map.json ────────────────────────────────────────────────────────
    class_map   = {i: name for i, name in enumerate(class_names)}
    with open(out / "class_map.json", "w") as f:
        json.dump(class_map, f, indent=2)

    # ── split_stats.json ──────────────────────────────────────────────────────
    stats = {}
    total_train = total_val = total_test = 0
    for cls in class_names:
        stats[cls] = {
            "train": len(train.get(cls, [])),
            "val":   len(val.get(cls,   [])),
            "test":  len(test.get(cls,  [])),
            "total": len(data[cls]),
        }
        total_train += stats[cls]["train"]
        total_val   += stats[cls]["val"]
        total_test  += stats[cls]["test"]

    with open(out / "split_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ── Class imbalance report ────────────────────────────────────────────────
    counts = [(cls, len(imgs)) for cls, imgs in data.items()]
    counts.sort(key=lambda x: x[1])

    max_count = counts[-1][1]
    min_count = counts[0][1]
    imbalance_ratio = max_count / min_count

    print(f"\n  Class imbalance report:")
    print(f"  {'Class':<55} {'Total':>6}")
    print(f"  {'-'*63}")
    for cls, n in counts:
        bar = "█" * int(30 * n / max_count)
        print(f"  {cls:<55} {n:>6}  {bar}")

    print(f"\n  Largest class : {counts[-1][0]} ({max_count})")
    print(f"  Smallest class: {counts[0][0]} ({min_count})")
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}×")
    print(f"  → Training will use WeightedRandomSampler to compensate.\n")

    print(f"  Split totals:")
    print(f"    train: {total_train}")
    print(f"    val  : {total_val}")
    print(f"    test : {total_test}")
    print(f"    total: {total_train + total_val + total_test}")
    print(f"\n  Wrote: {out}/class_map.json")
    print(f"  Wrote: {out}/split_stats.json")
    print(f"\n  Next step: python 7_train_clf.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
