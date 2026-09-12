"""
5b_crop_tomatovillage.py
========================
Runs the leaf-detection cropping system on every image in the
Tomato-Village dataset IN-PLACE, overwriting originals with crops.

For each image:
  1. Detect all leaves using the trained YOLO model
  2. Crop each leaf exactly to the detected bounding box (no padding)
  3. Resize to 256×256
  4. Overwrite the original file with the crop

If an image has 1 leaf  → overwrites original
If an image has N leaves → overwrites original + adds _leaf2, _leaf3, ...
If no leaf detected      → resizes original to 256×256

WARNING: This modifies the dataset in-place. Originals are overwritten.

Usage:
  python 5b_crop_tomatovillage.py
  python 5b_crop_tomatovillage.py --src ./tomatovillage
  python 5b_crop_tomatovillage.py --conf 0.57 --weights /path/to/best.pt
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = str(Path(__file__).resolve().parent / "runs" / "detect" / "runs" / "detect" / "leaf_detector_v1" / "weights" / "best.pt")
CONF_THRESHOLD  = 0.57
IOU_THRESHOLD   = 0.45
IMG_SIZE        = 640

PAD_RATIO       = 0.0
PAD_MIN_PX      = 0
PAD_MAX_PX      = 0
OUTPUT_SIZE     = 256

SRC_DIR         = "./tomatovillage"
VARIANT         = "Variant-a(Multiclass Classification)"
IMAGE_EXTS      = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Crop all leaves in Tomato-Village dataset in-place")
    p.add_argument("--src", default=SRC_DIR, help="Root of Tomato-Village dataset")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="YOLO weights (.pt)")
    p.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=IOU_THRESHOLD, help="NMS IoU threshold")
    return p.parse_args()


def load_model(weights_path: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("[ERROR] Run: pip install ultralytics")
    w = Path(weights_path)
    if not w.exists():
        raise SystemExit(f"[ERROR] Weights not found: {w}")
    return YOLO(str(w))


def crop_leaves(img: np.ndarray, boxes_xyxy: np.ndarray, confs: np.ndarray) -> list[np.ndarray]:
    """Crop each detected leaf exactly to its bounding box, resize to OUTPUT_SIZE."""
    H, W = img.shape[:2]
    order = sorted(range(len(boxes_xyxy)),
                   key=lambda i: (boxes_xyxy[i][1], boxes_xyxy[i][0]))

    crops = []
    for i in order:
        x1, y1, x2, y2 = boxes_xyxy[i]

        x1p = max(0, int(x1))
        y1p = max(0, int(y1))
        x2p = min(W, int(x2))
        y2p = min(H, int(y2))

        crop = img[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            continue

        crop_resized = cv2.resize(crop, (OUTPUT_SIZE, OUTPUT_SIZE),
                                  interpolation=cv2.INTER_LANCZOS4)
        crops.append(crop_resized)
    return crops


def process_image(model, img_path: Path, conf: float, iou: float) -> int:
    """Detect leaves, crop, and overwrite in-place. Returns number of crops saved."""
    img = cv2.imread(str(img_path))
    if img is None:
        return 0

    results = model.predict(
        source=str(img_path),
        conf=conf,
        iou=iou,
        imgsz=IMG_SIZE,
        augment=True,
        verbose=False,
    )[0]

    boxes = results.boxes
    stem = img_path.stem
    suffix = img_path.suffix
    parent = img_path.parent

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs_arr = boxes.conf.cpu().numpy()
        crops = crop_leaves(img, xyxy, confs_arr)
    else:
        crops = []

    if len(crops) == 0:
        # Fallback: resize the whole image
        resized = cv2.resize(img, (OUTPUT_SIZE, OUTPUT_SIZE),
                             interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(img_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return 1

    # Overwrite original with first crop
    cv2.imwrite(str(img_path), crops[0], [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Additional leaves → new files in same directory
    for idx, crop in enumerate(crops[1:], start=2):
        out_path = parent / f"{stem}_leaf{idx}{suffix}"
        cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return len(crops)


def main():
    args = parse_args()
    src_root = Path(args.src) / VARIANT

    if not src_root.exists():
        raise SystemExit(
            f"[ERROR] Source not found: {src_root}\n"
            f"        Run python 5_download_plantvillage.py first."
        )

    # Collect every image
    all_images = []
    for img_path in sorted(src_root.rglob("*")):
        if img_path.is_file() and img_path.suffix in IMAGE_EXTS:
            all_images.append(img_path)

    if not all_images:
        raise SystemExit(f"[ERROR] No images found under: {src_root}")

    print(f"Tomato-Village Leaf Cropper (IN-PLACE)")
    print(f"{'='*50}")
    print(f"Dataset: {src_root}")
    print(f"Images : {len(all_images)}")
    print(f"Conf   : {args.conf}")
    print(f"Padding: none (exact bounding box)")
    print(f"Crop   : {OUTPUT_SIZE}×{OUTPUT_SIZE}px")
    print(f"WARNING: Originals will be overwritten!\n")

    model = load_model(args.weights)
    print(f"Model loaded: {args.weights}\n")

    total_crops = 0
    no_detect = 0
    multi_leaf = 0

    for idx, img_path in enumerate(all_images, start=1):
        n = process_image(model, img_path, args.conf, args.iou)
        total_crops += n

        if n == 0:
            no_detect += 1
        elif n > 1:
            multi_leaf += 1

        if idx % 200 == 0 or idx == len(all_images):
            print(f"  [{idx:>5}/{len(all_images)}] {total_crops} crops saved so far...")

    print(f"\n{'='*50}")
    print(f"Done!")
    print(f"  Total input images : {len(all_images)}")
    print(f"  Total output crops : {total_crops}")
    print(f"  Single-leaf images : {len(all_images) - no_detect - multi_leaf}")
    print(f"  Multi-leaf images  : {multi_leaf}")
    print(f"  No detection (fallback): {no_detect}")
    print(f"\nDataset modified in-place at: {src_root.resolve()}")


if __name__ == "__main__":
    main()
