"""
4_infer.py
==========
Detects all individual leaves in an image (or folder of images),
then saves each leaf as a separate cropped image.

CROP SPECIFICATION:
    • Crop is a tight rectangle around each individual leaf — exact
      bounding box, no extra padding
    • Crop window is clamped to image boundaries (no out-of-bounds access)
    • Each crop is resized to 256×256 pixels — matches PlantVillage format,
      ensuring the output images are consistently sized and compatible
      with downstream classification pipelines trained on PlantVillage

OUTPUT per input image:
  output/<image_stem>/
    leaf_001_conf0.87.jpg
    leaf_002_conf0.91.jpg
    ...

NOTE:
    • This script saves ONLY leaf crops.
    • It does NOT save any overview or original input image.

SINGLE IMAGE:
  python 4_infer.py --image path/to/photo.jpg

FOLDER (batch):
  python 4_infer.py --folder path/to/photos/

CUSTOM WEIGHTS:
  python 4_infer.py --image photo.jpg --weights ./runs/leaf_detector_v1/weights/best.pt
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS  = str(Path(__file__).parent / "models" / "detector.pt")
CONF_THRESHOLD   = 0.30     # detection confidence cutoff
IOU_THRESHOLD    = 0.45     # NMS IoU threshold
IMG_SIZE         = 640      # must match training imgsz

OUTPUT_SIZE      = 256      # resize each crop to this (matches PlantVillage)
OUTPUT_ROOT      = Path("./output")
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image",  help="Single image path")
    g.add_argument("--folder", help="Folder of images (jpg/png)")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS,
                   help="Path to trained weights (.pt)")
    p.add_argument("--conf",   type=float, default=CONF_THRESHOLD)
    p.add_argument("--iou",    type=float, default=IOU_THRESHOLD)
    p.add_argument("--out",    default=str(OUTPUT_ROOT))
    return p.parse_args()


def load_model(weights_path: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("[ERROR] Run: pip install ultralytics")

    w = Path(weights_path)
    if not w.exists():
        raise SystemExit(
            f"[ERROR] Weights not found: {w}\n"
            f"        Run python 3_train.py first."
        )
    print(f"Loading model: {w}")
    return YOLO(str(w))


def crop_and_save(
    img: np.ndarray,
    boxes_xyxy: np.ndarray,
    confs: np.ndarray,
    out_dir: Path,
):
    """
        For each detected bounding box:
            1. Crop exactly to the detected bounding box
            2. Clamp crop to image boundary
            3. Resize to OUTPUT_SIZE × OUTPUT_SIZE
            4. Save as JPEG

    Boxes are sorted top-left → bottom-right for consistent numbering.
    Returns number of crops saved.
    """
    H, W = img.shape[:2]

    # Sort: top-to-bottom, then left-to-right
    order = sorted(range(len(boxes_xyxy)),
                   key=lambda i: (boxes_xyxy[i][1], boxes_xyxy[i][0]))

    saved = 0
    for rank, i in enumerate(order):
        x1, y1, x2, y2 = boxes_xyxy[i]
        conf = confs[i]

        # ── Exact bounding box crop, clamped to image bounds ─────
        x1p = max(0, int(x1))
        y1p = max(0, int(y1))
        x2p = min(W, int(x2))
        y2p = min(H, int(y2))

        crop = img[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            continue

        # ── Resize to OUTPUT_SIZE × OUTPUT_SIZE (PlantVillage compatible) ────
        crop_resized = cv2.resize(
            crop,
            (OUTPUT_SIZE, OUTPUT_SIZE),
            interpolation=cv2.INTER_LANCZOS4,  # high-quality downscale
        )

        fname = f"crop_{rank+1:03d}_conf{conf:.2f}.jpg"
        cv2.imwrite(
            str(out_dir / fname),
            crop_resized,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        saved += 1

    return saved


def run_on_image(model, img_path: Path, out_root: Path, conf: float, iou: float):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [SKIP] Cannot read: {img_path}")
        return 0

    # ── Run detection ─────────────────────────────────────────────────────────
    results = model.predict(
        source  = str(img_path),
        conf    = conf,
        iou     = iou,
        imgsz   = IMG_SIZE,
        augment = True,     # test-time augmentation: better recall on small leaves
        verbose = False,
    )[0]

    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        print(f"  {img_path.name}: no leaves detected "
              f"(try lowering --conf, currently {conf})")
        return 0

    xyxy  = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()

    # ── Output directory for this image ──────────────────────────────────────
    out_dir = out_root / img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Crop and save ─────────────────────────────────────────────────────────
    n = crop_and_save(img, xyxy, confs, out_dir)

    print(f"  {img_path.name}: {len(xyxy)} detected → {n} crops saved  [{out_dir}]")
    return n


def collect_images(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"[ERROR] Folder not found: {folder}")
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def main():
    args  = parse_args()
    out   = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists() or not image_path.is_file():
            raise SystemExit(f"[ERROR] Image not found: {image_path}")
    else:
        folder_path = Path(args.folder)
        if not folder_path.exists() or not folder_path.is_dir():
            raise SystemExit(f"[ERROR] Folder not found: {folder_path}")

    model = load_model(args.weights)

    print(f"\nPadding  : none (exact bounding box)")
    print(f"Crop out : {OUTPUT_SIZE}×{OUTPUT_SIZE}px  (PlantVillage compatible)")
    print(f"Output   : {out.resolve()}\n")

    if args.image:
        total = run_on_image(model, image_path, out, args.conf, args.iou)
        print(f"\nTotal crops: {total}")
    else:
        images = collect_images(folder_path)
        if not images:
            raise SystemExit(f"[ERROR] No images found in: {args.folder}")
        print(f"Found {len(images)} images in {args.folder}\n")
        total = 0
        crops_per_image = []
        for img_path in images:
            n = run_on_image(model, img_path, out, args.conf, args.iou)
            crops_per_image.append(n)
            total += n

        # ── Distribution summary ──────────────────────────────────────────
        print(f"\nBatch complete. Total crops: {total}")
        from collections import Counter
        dist = Counter(crops_per_image)
        print("Crops-per-image distribution:")
        for k in sorted(dist):
            bar = "█" * dist[k]
            print(f"  {k:>2} crops: {dist[k]:>4} images  {bar}")


if __name__ == "__main__":
    main()
