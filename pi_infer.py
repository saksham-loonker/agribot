#!/usr/bin/env python3
"""
pi_infer.py — Self-contained inference for Raspberry Pi.

Single image → single word verdict. Supports .pt and NCNN models.
Graceful Ctrl+C. No imports from any other scripts.

Usage:
  python3 pi_infer.py image.jpg
  python3 pi_infer.py image.jpg --with-detector
  python3 pi_infer.py image.jpg --clf-model ./models_ncnn/classifier_ncnn
"""

import argparse
import signal
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Graceful exit ─────────────────────────────────────────────────────────────
def _signal_handler(sig, frame):
    sys.exit(0)

signal.signal(signal.SIGINT, _signal_handler)

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR         = Path(__file__).parent
DEFAULT_DET_MODEL  = SCRIPT_DIR / "models" / "detector.pt"
DEFAULT_CLF_MODEL  = SCRIPT_DIR / "models" / "classifier.pt"
DET_CONF           = 0.35
DET_IOU            = 0.45
CLF_CONF_THRESHOLD = 0.57


def parse_args():
    p = argparse.ArgumentParser(description="Single-image inference")
    p.add_argument("image", help="Path to image")
    p.add_argument("--det-model", default=str(DEFAULT_DET_MODEL))
    p.add_argument("--clf-model", default=str(DEFAULT_CLF_MODEL))
    p.add_argument("--det-imgsz", type=int, default=320)
    p.add_argument("--det-conf", type=float, default=DET_CONF)
    p.add_argument("--det-iou", type=float, default=DET_IOU)
    p.add_argument("--max-edge", type=int, default=320)
    p.add_argument("--max-leaves", type=int, default=1)
    p.add_argument("--conf", type=float, default=CLF_CONF_THRESHOLD)
    p.add_argument("--with-detector", action="store_true")
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args()


def setup_threads(n):
    cv2.setNumThreads(n)
    try:
        import torch
        torch.set_num_threads(n)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def load_model(path_str, task=None):
    from ultralytics import YOLO
    p = Path(path_str)
    if not p.exists():
        print(f"Error: model not found: {p}", file=sys.stderr)
        sys.exit(1)
    kwargs = {}
    if task:
        kwargs["task"] = task
    elif p.is_dir():
        meta = p / "metadata.yaml"
        if meta.exists():
            import yaml
            with open(meta) as f:
                kwargs["task"] = yaml.safe_load(f).get("task", "detect")
    return YOLO(str(p), **kwargs)


def resize_img(img, max_edge):
    h, w = img.shape[:2]
    if max(h, w) <= max_edge:
        return img
    scale = max_edge / max(h, w)
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


def detect_and_crop(det_model, img, imgsz, max_leaves, det_conf, det_iou):
    result = det_model.predict(
        source=img, conf=det_conf, iou=det_iou, imgsz=imgsz,
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


def classify(clf_model, crop, clf_conf):
    result = clf_model.predict(source=crop, verbose=False)[0]
    probs = result.probs
    best_class = clf_model.names[int(probs.top5[0])]
    best_conf = float(probs.top5conf[0])
    if best_conf < clf_conf:
        return "Uncertain"
    if "healthy" not in best_class.lower() and best_conf >= clf_conf:
        return best_class
    return "Healthy"


def main():
    args = parse_args()
    setup_threads(args.threads)

    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: cannot read image: {args.image}", file=sys.stderr)
        sys.exit(1)

    clf_model = load_model(args.clf_model)

    img_small = resize_img(img, args.max_edge)

    if args.with_detector:
        det_model = load_model(args.det_model)
        crops = detect_and_crop(
            det_model,
            img_small,
            args.det_imgsz,
            args.max_leaves,
            args.det_conf,
            args.det_iou,
        )
        if not crops:
            crops = [img_small]
    else:
        crops = [img_small]

    from collections import Counter
    diseases = []
    labels = []
    for crop in crops:
        label = classify(clf_model, crop, args.conf)
        labels.append(label)
        if label != "Healthy":
            diseases.append(label)

    diseases = [label for label in labels if label not in {"Healthy", "Uncertain"}]
    if diseases:
        verdict = Counter(diseases).most_common(1)[0][0]
    elif any(label == "Uncertain" for label in labels):
        verdict = "Uncertain"
    else:
        verdict = "Healthy"

    print(verdict)


if __name__ == "__main__":
    main()
