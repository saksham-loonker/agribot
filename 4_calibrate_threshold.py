"""
4_calibrate_threshold.py
========================
Confidence-threshold calibration for leaf detection with real accuracy.

Uses YOLO-format ground-truth labels to compute precision, recall, and F1
at every threshold from 0.05 to 1.00 (step 0.02).  Matches predicted boxes
to ground-truth boxes via IoU so the recommendation is based on actual
detection quality, not just box counts.

Runs on ALL images (no sampling) for unbiased results.

Usage:
  python 4_calibrate_threshold.py --images /path/to/images --labels /path/to/labels
  python 4_calibrate_threshold.py \\
      --images dataset/test/images --labels dataset/test/labels \\
      --weights runs/detect/.../best.pt --iou-match 0.5
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

DEFAULT_WEIGHTS = str(Path(__file__).parent / "models" / "detector.pt")
IMG_SIZE = 640


def parse_args():
    p = argparse.ArgumentParser(description="Threshold calibration with accuracy")
    p.add_argument("--images", required=True, help="Folder of test images")
    p.add_argument("--labels", required=True, help="Folder of YOLO-format .txt labels")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="YOLO .pt weights")
    p.add_argument("--iou-nms", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--iou-match", type=float, default=0.50,
                   help="IoU threshold to count a prediction as a true positive")
    p.add_argument("--imgsz", type=int, default=IMG_SIZE, help="Inference image size")
    return p.parse_args()


def load_model(weights_path: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("[ERROR] Run: pip install ultralytics")
    w = Path(weights_path)
    if not w.exists():
        raise SystemExit(f"[ERROR] Weights not found: {w}")
    print(f"Loading model: {w}")
    return YOLO(str(w))


def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """Read YOLO label file → Nx4 array of [x1, y1, x2, y2] in pixel coords."""
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32)
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            if max(abs(cx), abs(cy), abs(bw), abs(bh)) > 1.5:
                cx /= img_w
                bw /= img_w
                cy /= img_h
                bh /= img_h
            x1 = (cx - bw / 2) * img_w
            y1 = (cy - bh / 2) * img_h
            x2 = (cx + bw / 2) * img_w
            y2 = (cy + bh / 2) * img_h
            boxes.append([x1, y1, x2, y2])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def compute_iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between pred (Mx4) and gt (Nx4)."""
    x1 = np.maximum(pred[:, None, 0], gt[None, :, 0])
    y1 = np.maximum(pred[:, None, 1], gt[None, :, 1])
    x2 = np.minimum(pred[:, None, 2], gt[None, :, 2])
    y2 = np.minimum(pred[:, None, 3], gt[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_p = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_g = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    union = area_p[:, None] + area_g[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def collect_pairs(img_dir: Path, lbl_dir: Path):
    """Return list of (image_path, label_path) for every image with a label."""
    exts = {".jpg", ".jpeg", ".png"}
    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in exts:
            continue
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        pairs.append((img_path, lbl_path))
    if not pairs:
        raise SystemExit(f"[ERROR] No images found in: {img_dir}")
    return pairs


def run_all_detections(model, pairs, iou_nms: float, imgsz: int):
    """Run model once at lowest conf and cache raw results.
    Returns list of (pred_xyxy, pred_confs, gt_xyxy) per image."""
    cache = []
    for img_path, lbl_path in pairs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_boxes = load_gt_boxes(lbl_path, w, h)

        result = model.predict(
            source=str(img_path),
            conf=0.01,          # very low — we threshold later in software
            iou=iou_nms,
            imgsz=imgsz,
            augment=True,
            verbose=False,
        )[0]

        if result.boxes is not None and len(result.boxes):
            pred_xyxy = result.boxes.xyxy.cpu().numpy()
            pred_conf = result.boxes.conf.cpu().numpy()
        else:
            pred_xyxy = np.zeros((0, 4), dtype=np.float32)
            pred_conf = np.zeros(0, dtype=np.float32)

        cache.append((pred_xyxy, pred_conf, gt_boxes))
    return cache


def evaluate_at_conf(cache, conf_thresh: float, iou_match: float):
    """Compute precision, recall, F1 at a given confidence threshold."""
    tp_total = 0
    fp_total = 0
    fn_total = 0

    for pred_xyxy, pred_conf, gt_boxes in cache:
        mask = pred_conf >= conf_thresh
        preds = pred_xyxy[mask]
        n_pred = len(preds)
        n_gt = len(gt_boxes)

        if n_pred == 0 and n_gt == 0:
            continue
        if n_pred == 0:
            fn_total += n_gt
            continue
        if n_gt == 0:
            fp_total += n_pred
            continue

        iou_mat = compute_iou_matrix(preds, gt_boxes)
        matched_gt = set()
        matched_pred = set()

        # Greedy matching: highest IoU first
        flat = []
        for pi in range(n_pred):
            for gi in range(n_gt):
                if iou_mat[pi, gi] >= iou_match:
                    flat.append((iou_mat[pi, gi], pi, gi))
        flat.sort(key=lambda x: -x[0])

        for _, pi, gi in flat:
            if pi in matched_pred or gi in matched_gt:
                continue
            matched_pred.add(pi)
            matched_gt.add(gi)

        tp = len(matched_pred)
        tp_total += tp
        fp_total += (n_pred - tp)
        fn_total += (n_gt - tp)

    precision = tp_total / max(tp_total + fp_total, 1)
    recall = tp_total / max(tp_total + fn_total, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
    }


def main():
    args = parse_args()
    img_dir = Path(args.images)
    lbl_dir = Path(args.labels)

    pairs = collect_pairs(img_dir, lbl_dir)
    model = load_model(args.weights)

    print(f"\nRunning detection on ALL {len(pairs)} images (conf=0.01, cached)...")
    cache = run_all_detections(model, pairs, iou_nms=args.iou_nms, imgsz=args.imgsz)
    print(f"Cached {len(cache)} results. Sweeping thresholds...\n")

    # Thresholds: 0.05 to 0.99 in steps of 0.02
    thresholds = [round(0.05 + i * 0.02, 2) for i in range(48)]  # 0.05, 0.07, ..., 0.99

    header = (f"{'conf':>5} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | "
              f"{'TP':>5} | {'FP':>5} | {'FN':>5}")
    print(header)
    print("-" * len(header))

    best_f1 = -1.0
    best_row = None
    results = []

    for conf in thresholds:
        row = evaluate_at_conf(cache, conf, args.iou_match)
        row["conf"] = conf
        results.append(row)

        marker = ""
        if row["f1"] > best_f1:
            best_f1 = row["f1"]
            best_row = row
            marker = " ◀ best"

        print(f"{conf:5.2f} | {row['precision']:6.3f} | {row['recall']:6.3f} | "
              f"{row['f1']:6.3f} | {row['tp']:5d} | {row['fp']:5d} | "
              f"{row['fn']:5d}{marker}")

    print(f"\n{'='*60}")
    print(f"Best F1: {best_row['f1']:.4f}  at --conf {best_row['conf']:.2f}")
    print(f"  Precision: {best_row['precision']:.4f}  Recall: {best_row['recall']:.4f}")
    print(f"  TP: {best_row['tp']}  FP: {best_row['fp']}  FN: {best_row['fn']}")
    print(f"\nRecommended command:")
    print(f"  python 4_infer.py --folder {img_dir} --conf {best_row['conf']:.2f} --iou {args.iou_nms}")


if __name__ == "__main__":
    main()
