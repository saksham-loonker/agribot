"""
Benchmark detector quality on tomato_crop_disease.

The tomato_crop_disease dataset is an images/labels YOLO-format benchmark, not
a class-folder classifier benchmark. This script evaluates detector boxes
class-agnostically against those labels and reports precision, recall, F1,
latency, and FPS. The default strict benchmark target is F1 >= 90%, but this
dataset has many disease-region boxes per image. For deployment triage, the
image-hit metric is also reported because it answers whether the model found at
least one valid disease/crop region in the image.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from inference_rpi import DET_CONF_THRESHOLD, DET_IMGSZ, DET_IOU_THRESHOLD, load_model, setup_runtime
from runtime_suite_common import suite_csv_path, write_json


SCRIPT_DIR = Path(__file__).parent
DEFAULT_DATASET = SCRIPT_DIR / "tomato_crop_disease"
DEFAULT_DETECTOR = SCRIPT_DIR / "runtime_exports" / "detector_openvino_model"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
TARGET_F1 = 0.90
TARGET_IMAGE_HIT_RATE = 0.90


def parse_args():
    p = argparse.ArgumentParser(description="tomato_crop_disease detector benchmark")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--split", default="test", help="Split to use when dataset has train/val/test folders")
    p.add_argument("--all-splits", action="store_true", help="Use train/val/test images together")
    p.add_argument("--images", default="")
    p.add_argument("--labels", default="")
    p.add_argument("--det-model", default=str(DEFAULT_DETECTOR))
    p.add_argument("--imgsz", type=int, default=DET_IMGSZ)
    p.add_argument("--iou-nms", type=float, default=DET_IOU_THRESHOLD)
    p.add_argument("--iou-match", type=float, default=0.50)
    p.add_argument("--min-conf", type=float, default=0.01)
    p.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-det", type=int, default=20, help="Maximum boxes kept per image before threshold sweep")
    p.add_argument(
        "--target-metric",
        choices=["f1", "image_hit_rate"],
        default="f1",
        help="Pass/fail metric for this run. F1 is strict box matching; image_hit_rate is image-level detection.",
    )
    p.add_argument(
        "--target-value",
        type=float,
        default=None,
        help="Override pass threshold. Defaults: 0.90 for both F1 and image_hit_rate.",
    )
    return p.parse_args()


def parse_thresholds(value: str) -> list[float]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if part:
            number = float(part)
            if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                raise SystemExit("[ERROR] --thresholds values must be finite and between 0 and 1")
            out.append(number)
    if not out:
        raise SystemExit("[ERROR] --thresholds needs at least one value")
    return out


def resolve_dirs(args) -> tuple[Path, Path]:
    dataset = Path(args.dataset)
    if args.images:
        images = Path(args.images)
    elif (dataset / "images").is_dir():
        images = dataset / "images"
    else:
        images = dataset / args.split / "images"

    if args.labels:
        labels = Path(args.labels)
    elif (dataset / "labels").is_dir():
        labels = dataset / "labels"
    else:
        labels = dataset / args.split / "labels"

    if not images.is_dir():
        raise SystemExit(f"[ERROR] Images dir not found: {images}")
    if not labels.is_dir():
        raise SystemExit(f"[ERROR] Labels dir not found: {labels}")
    return images, labels


def collect_pairs(images_dir: Path, labels_dir: Path, limit: int) -> list[tuple[Path, Path]]:
    pairs = []
    image_stems = set()
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        image_stems.add(image_path.stem)
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise SystemExit(f"[ERROR] Missing label for image: {image_path.name}")
        pairs.append((image_path, label_path))
    extra_labels = sorted(path.stem for path in labels_dir.glob("*.txt") if path.stem not in image_stems)
    if extra_labels:
        raise SystemExit(f"[ERROR] Labels without images: {', '.join(extra_labels[:5])}")
    if limit > 0:
        pairs = pairs[:limit]
    if not pairs:
        raise SystemExit(f"[ERROR] No image/label pairs found under: {images_dir}")
    return pairs


def collect_dataset_pairs(args) -> tuple[list[tuple[Path, Path]], str, str]:
    dataset = Path(args.dataset)
    if args.all_splits:
        if args.images or args.labels:
            raise SystemExit("[ERROR] --all-splits cannot be combined with --images/--labels")
        pairs = []
        image_sources = []
        label_sources = []
        for split in ("train", "val", "test"):
            images_dir = dataset / split / "images"
            labels_dir = dataset / split / "labels"
            if not images_dir.is_dir() or not labels_dir.is_dir():
                continue
            split_pairs = collect_pairs(images_dir, labels_dir, limit=0)
            pairs.extend(split_pairs)
            image_sources.append(str(images_dir))
            label_sources.append(str(labels_dir))
        if args.limit > 0:
            pairs = pairs[:args.limit]
        if not pairs:
            raise SystemExit(f"[ERROR] No all-split image/label pairs found under: {dataset}")
        return pairs, ",".join(image_sources), ",".join(label_sources)

    images_dir, labels_dir = resolve_dirs(args)
    pairs = collect_pairs(images_dir, labels_dir, args.limit)
    return pairs, str(images_dir), str(labels_dir)


def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            raise ValueError(f"Malformed YOLO label {label_path}: {line!r}")
        try:
            class_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid YOLO label {label_path}: {line!r}") from exc
        if class_id < 0 or not all(math.isfinite(value) for value in (cx, cy, bw, bh)):
            raise ValueError(f"Non-finite/negative YOLO label {label_path}: {line!r}")
        if max(abs(cx), abs(cy), abs(bw), abs(bh)) > 1.5:
            cx /= img_w
            bw /= img_w
            cy /= img_h
            bh /= img_h
        if bw <= 0 or bh <= 0 or not all(0.0 <= value <= 1.0 for value in (cx, cy, bw, bh)):
            raise ValueError(f"YOLO box outside normalized bounds in {label_path}: {line!r}")
        x1 = (cx - bw / 2.0) * img_w
        y1 = (cy - bh / 2.0) * img_h
        x2 = (cx + bw / 2.0) * img_w
        y2 = (cy + bh / 2.0) * img_h
        boxes.append([x1, y1, x2, y2])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)), dtype=np.float32)
    x1 = np.maximum(pred[:, None, 0], gt[None, :, 0])
    y1 = np.maximum(pred[:, None, 1], gt[None, :, 1])
    x2 = np.minimum(pred[:, None, 2], gt[None, :, 2])
    y2 = np.minimum(pred[:, None, 3], gt[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_p = np.maximum(0.0, pred[:, 2] - pred[:, 0]) * np.maximum(0.0, pred[:, 3] - pred[:, 1])
    area_g = np.maximum(0.0, gt[:, 2] - gt[:, 0]) * np.maximum(0.0, gt[:, 3] - gt[:, 1])
    union = area_p[:, None] + area_g[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def evaluate_at_threshold(cache: list[dict], threshold: float, iou_match: float) -> dict:
    tp_total = fp_total = fn_total = 0
    image_hits = 0
    images_with_gt = 0

    for row in cache:
        gt = row["gt"]
        pred = row["pred"][row["conf"] >= threshold]
        if len(gt) > 0:
            images_with_gt += 1

        if len(pred) == 0:
            fn_total += len(gt)
            continue
        if len(gt) == 0:
            fp_total += len(pred)
            continue

        mat = iou_matrix(pred, gt)
        candidates = []
        for pred_i in range(mat.shape[0]):
            for gt_i in range(mat.shape[1]):
                if mat[pred_i, gt_i] >= iou_match:
                    candidates.append((float(mat[pred_i, gt_i]), pred_i, gt_i))
        candidates.sort(reverse=True)

        matched_pred = set()
        matched_gt = set()
        for _, pred_i, gt_i in candidates:
            if pred_i in matched_pred or gt_i in matched_gt:
                continue
            matched_pred.add(pred_i)
            matched_gt.add(gt_i)

        tp = len(matched_pred)
        tp_total += tp
        fp_total += len(pred) - tp
        fn_total += len(gt) - tp
        if tp > 0:
            image_hits += 1

    precision = tp_total / max(tp_total + fp_total, 1)
    recall = tp_total / max(tp_total + fn_total, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    image_hit_rate = image_hits / max(images_with_gt, 1)
    return {
        "conf": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "image_hit_rate": image_hit_rate,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
    }


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    if args.limit < 0 or args.threads < 1 or args.imgsz <= 0 or args.max_det < 1:
        raise SystemExit("[ERROR] limit must be >= 0, threads/imgsz/max-det must be positive")
    for name in ("iou_nms", "iou_match", "min_conf"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise SystemExit(f"[ERROR] {name} must be finite and between 0 and 1")
    setup_runtime(args.threads)
    pairs, images_ref, labels_ref = collect_dataset_pairs(args)
    thresholds = parse_thresholds(args.thresholds)
    target_value = args.target_value
    if target_value is None:
        target_value = TARGET_F1 if args.target_metric == "f1" else TARGET_IMAGE_HIT_RATE
    if not math.isfinite(float(target_value)) or not 0.0 <= float(target_value) <= 1.0:
        raise SystemExit("[ERROR] --target-value must be finite and between 0 and 1")

    model_path = Path(args.det_model)
    if not model_path.exists():
        raise SystemExit(f"[ERROR] Detector not found: {model_path}")
    model = load_model(str(model_path), task="detect")

    print("tomato_crop_disease detector benchmark")
    print(f"  detector : {model_path}")
    print(f"  images   : {images_ref}")
    print(f"  labels   : {labels_ref}")
    print(f"  samples  : {len(pairs)}")
    print(f"  imgsz    : {args.imgsz}")
    print()

    cache = []
    total_time = 0.0
    for image_path, label_path in pairs:
        img = cv2.imread(str(image_path))
        if img is None:
            raise SystemExit(f"[ERROR] Cannot decode image: {image_path}")
        h, w = img.shape[:2]
        gt = load_gt_boxes(label_path, w, h)

        started = time.perf_counter()
        result = model.predict(
            source=img,
            conf=args.min_conf,
            iou=args.iou_nms,
            imgsz=args.imgsz,
            max_det=args.max_det,
            augment=False,
            verbose=False,
        )[0]
        elapsed = time.perf_counter() - started
        total_time += elapsed

        if result.boxes is not None and len(result.boxes):
            pred = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            conf = result.boxes.conf.cpu().numpy().astype(np.float32)
        else:
            pred = np.zeros((0, 4), dtype=np.float32)
            conf = np.zeros((0,), dtype=np.float32)
        cache.append({"pred": pred, "conf": conf, "gt": gt})

    if not cache:
        raise SystemExit("[ERROR] No images were processed")
    if len(cache) != len(pairs):
        raise SystemExit(
            f"[ERROR] Benchmark incomplete: processed {len(cache)} of {len(pairs)} images"
        )

    rows = [evaluate_at_threshold(cache, threshold, args.iou_match) for threshold in thresholds]
    for row in rows:
        row["target_metric"] = args.target_metric
        row["target_value"] = target_value
        row["metric_pass"] = row[args.target_metric] >= target_value
        # A limited smoke run is useful diagnostics, but it is not a complete
        # benchmark gate and must never be recorded as production evidence.
        row["target_pass"] = row["metric_pass"] and args.limit <= 0

    best = max(
        rows,
        key=lambda row: (
            row[args.target_metric],
            row["f1"],
            row["precision"],
            row["recall"],
        ),
    )
    avg_fps = len(cache) / total_time if total_time > 0 else 0.0
    avg_latency_ms = 1000.0 * total_time / len(cache)

    tracking_dir = suite_csv_path("tomato_crop_disease_detector_results.csv").parent
    csv_path = tracking_dir / "tomato_crop_disease_detector_results.csv"
    json_path = tracking_dir / "tomato_crop_disease_detector_best.json"
    write_csv(csv_path, rows)
    payload = {
        "target_metric": args.target_metric,
        "target_value": target_value,
        "target_f1": TARGET_F1,
        "target_image_hit_rate": TARGET_IMAGE_HIT_RATE,
        "target_pass": best["target_pass"],
        "metric_pass": best["metric_pass"],
        "complete": args.limit <= 0,
        "best": best,
        "detector": str(model_path.resolve()),
        "dataset": str(Path(args.dataset).resolve()),
        "split": "all" if args.all_splits else args.split,
        "images": images_ref,
        "labels": labels_ref,
        "samples": len(cache),
        "imgsz": args.imgsz,
        "max_det": args.max_det,
        "iou_match": args.iou_match,
        "avg_fps": avg_fps,
        "avg_latency_ms": avg_latency_ms,
        "csv": str(csv_path.resolve()),
    }
    write_json(json_path, payload)

    print(f"{'conf':>6} {'prec':>7} {'recall':>7} {'f1':>7} {'hit/img':>8} {'tp':>5} {'fp':>5} {'fn':>5}")
    for row in rows:
        marker = "  < best" if row is best else ""
        print(
            f"{row['conf']:6.2f} {row['precision']:7.3f} {row['recall']:7.3f} "
            f"{row['f1']:7.3f} {row['image_hit_rate']:8.3f} "
            f"{row['tp']:5d} {row['fp']:5d} {row['fn']:5d}{marker}"
        )
    print()
    print(
        f"Best {args.target_metric}: {best[args.target_metric] * 100:.2f}% at conf={best['conf']:.2f} "
        f"({'PASS' if best['target_pass'] else 'FAIL'} vs {target_value * 100:.0f}% target)"
    )
    print(f"Strict F1     : {best['f1'] * 100:.2f}%")
    print(f"Image hit     : {best['image_hit_rate'] * 100:.2f}%")
    print(f"Speed  : {avg_fps:.2f} FPS, {avg_latency_ms:.0f} ms/image")
    print(f"CSV    : {csv_path}")
    print(f"JSON   : {json_path}")
    if not best["target_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
