"""
Prepare a classifier dataset that matches the fast Pi deployment crop.

The detector-first benchmark is accurate but too slow on Pi CPU. Fast center or
mask crops are quick enough, but the current classifier was not trained on those
crop distributions. This script creates a cropped copy of clf_dataset/ so the
classifier can be fine-tuned for the actual deployment path.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2

from inference_rpi import center_crop, downscale_if_needed, mask_crop


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(description="Create fast-crop classifier dataset")
    p.add_argument("--src", default="clf_dataset")
    p.add_argument("--out", default="clf_dataset_fastcrop")
    p.add_argument("--max-edge", type=int, default=256)
    p.add_argument("--crop-mode", choices=["center", "mask"], default="mask")
    p.add_argument("--crop-scale", type=float, default=0.55)
    p.add_argument("--crop-pad", type=float, default=0.05)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def iter_images(split_dir: Path):
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                yield class_dir.name, image_path


def crop_image(img, args):
    img = downscale_if_needed(img, args.max_edge)
    if args.crop_mode == "mask":
        return mask_crop(img, crop_pad=args.crop_pad, fallback_scale=args.crop_scale)
    return center_crop(img, crop_scale=args.crop_scale)


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

    manifest = {
        "source": str(src.resolve()),
        "max_edge": args.max_edge,
        "crop_mode": args.crop_mode,
        "crop_scale": args.crop_scale,
        "crop_pad": args.crop_pad,
        "splits": {},
    }

    total = 0
    for split in ("train", "val", "test"):
        split_dir = src / split
        if not split_dir.is_dir():
            continue

        split_count = 0
        for class_name, image_path in iter_images(split_dir):
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            crop = crop_image(img, args)
            dest_dir = out / split / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / image_path.name
            if dest.suffix.lower() not in {".jpg", ".jpeg"}:
                dest = dest.with_suffix(".jpg")
            cv2.imwrite(str(dest), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            split_count += 1
            total += 1

        manifest["splits"][split] = split_count
        print(f"{split}: {split_count} images")

    for metadata_name in ("class_map.json", "split_stats.json"):
        source_metadata = src / metadata_name
        if source_metadata.exists():
            shutil.copy2(source_metadata, out / metadata_name)

    manifest["total"] = total
    with open(out / "fast_crop_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Wrote {total} cropped images -> {out}")


if __name__ == "__main__":
    main()
