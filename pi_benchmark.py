#!/usr/bin/env python3
"""
pi_benchmark.py — Self-contained, max-optimized benchmark for Raspberry Pi.

NO external script imports. Graceful Ctrl+C. Supports .pt and NCNN models.
Pre-cropped leaf images → classifier-only by default (skips slow detector).

Usage:
  python3 pi_benchmark.py --test ./test_data
  python3 pi_benchmark.py --test ./test_data --with-detector
  python3 pi_benchmark.py --test ./test_data --limit 50

NCNN mode (much faster on Pi):
  python3 pi_benchmark.py --test ./test_data --clf-model ./models_ncnn/classifier_ncnn
"""

import argparse
import signal
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

# ── Graceful exit ─────────────────────────────────────────────────────────────
_interrupted = False


def _signal_handler(sig, frame):
    global _interrupted
    _interrupted = True
    print("\n\n  [!] Interrupted — printing results so far...\n")


signal.signal(signal.SIGINT, _signal_handler)

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR         = Path(__file__).parent
DEFAULT_DET_MODEL  = SCRIPT_DIR / "models" / "detector.pt"
DEFAULT_CLF_MODEL  = SCRIPT_DIR / "models" / "classifier.pt"
DET_CONF           = 0.35
DET_IOU            = 0.45
CLF_CONF_THRESHOLD = 0.57
IMG_EXTS           = {".jpg", ".jpeg", ".png", ".bmp"}
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Self-contained Pi benchmark — FPS + Accuracy")
    p.add_argument("--test", required=True,
                   help="Test folder with class subfolders")
    p.add_argument("--det-model", default=str(DEFAULT_DET_MODEL),
                   help="Detector model path (.pt or NCNN folder)")
    p.add_argument("--clf-model", default=str(DEFAULT_CLF_MODEL),
                   help="Classifier model path (.pt or NCNN folder)")
    p.add_argument("--det-imgsz", type=int, default=320,
                   help="Detector inference size")
    p.add_argument("--max-edge", type=int, default=320,
                   help="Resize long edge before inference")
    p.add_argument("--max-leaves", type=int, default=1,
                   help="Classify only top-N detected leaves")
    p.add_argument("--conf", type=float, default=CLF_CONF_THRESHOLD,
                   help="Classifier confidence threshold")
    p.add_argument("--with-detector", action="store_true",
                   help="Run detector+classifier (default: classifier-only)")
    p.add_argument("--threads", type=int, default=4,
                   help="CPU threads")
    p.add_argument("--limit", type=int, default=0,
                   help="Max images to evaluate (0 = all)")
    return p.parse_args()


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup_threads(n: int):
    cv2.setNumThreads(n)
    try:
        import torch
        torch.set_num_threads(n)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def load_model(path_str: str, task: str = None):
    from ultralytics import YOLO
    p = Path(path_str)
    if not p.exists():
        sys.exit(f"[ERROR] Model not found: {p}")
    kwargs = {}
    if task:
        kwargs["task"] = task
    elif p.is_dir():
        # NCNN folders can't auto-detect task; read metadata
        meta = p / "metadata.yaml"
        if meta.exists():
            import yaml
            with open(meta) as f:
                kwargs["task"] = yaml.safe_load(f).get("task", "detect")
    return YOLO(str(p), **kwargs)


# ── Image helpers ─────────────────────────────────────────────────────────────

def resize_img(img, max_edge: int):
    h, w = img.shape[:2]
    if max(h, w) <= max_edge:
        return img
    scale = max_edge / max(h, w)
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_and_crop(det_model, img, imgsz: int, max_leaves: int):
    result = det_model.predict(
        source=img, conf=DET_CONF, iou=DET_IOU, imgsz=imgsz,
        augment=False, verbose=False)[0]

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    order = np.argsort(-confs)
    if max_leaves > 0:
        order = order[:max_leaves]

    H, W = img.shape[:2]
    crops = []
    for i in order:
        x1, y1, x2, y2 = xyxy[i]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(W, int(x2)), min(H, int(y2))
        c = img[y1:y2, x1:x2]
        if c.size > 0:
            crops.append(c)
    return crops


# ── Classification ────────────────────────────────────────────────────────────

def classify(clf_model, crop, clf_conf: float, healthy_idx):
    result = clf_model.predict(source=crop, verbose=False)[0]
    probs = result.probs
    best_class = clf_model.names[int(probs.top5[0])]
    best_conf = float(probs.top5conf[0])

    if "healthy" not in best_class.lower() and best_conf >= clf_conf:
        return best_class, True

    return "Healthy", False


def find_healthy_idx(names):
    for idx, name in names.items():
        if "healthy" in name.lower():
            return int(idx)
    return None


def vote(labels):
    if not labels:
        return "Healthy"
    counts = Counter(labels)
    return counts.most_common(1)[0][0]


# ── Data collection ───────────────────────────────────────────────────────────

def collect_samples(test_dir: Path):
    samples = []
    for cls_dir in sorted(test_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() in IMG_EXTS:
                samples.append((img_path, cls_dir.name))
    return samples


# ── Main benchmark ────────────────────────────────────────────────────────────

def run(args):
    global _interrupted

    setup_threads(args.threads)

    test_dir = Path(args.test)
    if not test_dir.is_dir():
        sys.exit(f"[ERROR] Not a directory: {test_dir}")

    samples = collect_samples(test_dir)
    if args.limit > 0:
        samples = samples[:args.limit]
    if not samples:
        sys.exit(f"[ERROR] No images in: {test_dir}")

    n_total = len(samples)

    # Load models
    clf_model = load_model(args.clf_model)
    healthy_idx = find_healthy_idx(clf_model.names)
    det_model = None
    if args.with_detector:
        det_model = load_model(args.det_model)

    mode_str = "detect+classify" if args.with_detector else "classify-only"
    print(f"\n  Mode      : {mode_str}")
    print(f"  Classifier: {args.clf_model}")
    if args.with_detector:
        print(f"  Detector  : {args.det_model}")
    print(f"  Images    : {n_total}")
    print(f"  Threads   : {args.threads}")
    print(f"  Max edge  : {args.max_edge}px")
    print()

    n_correct = 0
    n_done = 0
    total_time = 0.0
    class_stats = defaultdict(lambda: [0, 0])

    # Progress bar (soft dependency)
    pbar = None
    try:
        from tqdm import tqdm
        pbar = tqdm(total=n_total, desc="Benchmarking", unit="img")
    except Exception:
        pass

    for img_path, true_label in samples:
        if _interrupted:
            break

        img = cv2.imread(str(img_path))
        if img is None:
            if pbar:
                pbar.update(1)
            continue

        t0 = time.perf_counter()

        img_small = resize_img(img, args.max_edge)

        if args.with_detector and det_model is not None:
            crops = detect_and_crop(det_model, img_small, args.det_imgsz, args.max_leaves)
            if not crops:
                crops = [img_small]
        else:
            crops = [img_small]

        diseases = []
        for crop in crops:
            label, is_disease = classify(clf_model, crop, args.conf, healthy_idx)
            if is_disease:
                diseases.append(label)

        pred = vote(diseases) if diseases else "Healthy"

        elapsed = time.perf_counter() - t0
        total_time += elapsed
        n_done += 1

        ok = pred.strip().lower() == true_label.strip().lower()
        if ok:
            n_correct += 1
        class_stats[true_label][1] += 1
        if ok:
            class_stats[true_label][0] += 1

        if pbar:
            acc = (n_correct / n_done) * 100
            fps = n_done / total_time if total_time > 0 else 0
            pbar.set_postfix(acc=f"{acc:.1f}%", fps=f"{fps:.2f}")
            pbar.update(1)

    if pbar:
        pbar.close()

    # ── Print results (always, even after Ctrl+C) ────────────────────────────
    if n_done == 0:
        print("  No images processed.")
        return

    overall_acc = (n_correct / n_done) * 100
    avg_fps = n_done / total_time if total_time > 0 else 0

    sep = "=" * 56
    status = " (INTERRUPTED)" if _interrupted else ""
    print(f"\n{sep}")
    print(f"  Benchmark Results{status}")
    print(f"{sep}")
    print(f"  Mode          : {mode_str}")
    print(f"  Images tested : {n_done}")
    print(f"  Overall acc   : {overall_acc:.2f}%  ({n_correct}/{n_done})")
    print(f"  Avg FPS       : {avg_fps:.2f}")
    print(f"  Total time    : {total_time:.1f}s")
    print(f"  Avg latency   : {(total_time / n_done) * 1000:.0f} ms/img")
    print()
    print("  Per-class accuracy:")
    for label in sorted(class_stats):
        c_ok, c_total = class_stats[label]
        c_acc = (c_ok / c_total) * 100 if c_total else 0
        print(f"    {label:<28s}  {c_acc:5.1f}%  ({c_ok}/{c_total})")
    print(f"{sep}\n")


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
