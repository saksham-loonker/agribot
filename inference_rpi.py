#!/usr/bin/env python3
"""
inference_rpi.py
================
Low-overhead Raspberry Pi inference path for crop detection + plant problem
classification. Supports Ultralytics `.pt` models and exported NCNN folders.

The main speed wins are:
  - smaller default detector input,
  - one-time warmup,
  - no test-time augmentation,
  - optional detector skip,
  - batched classification across detected crops.
"""

import argparse
import math
import signal
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).parent
DEFAULT_DET_MODEL = SCRIPT_DIR / "models" / "detector.pt"
DEFAULT_CLF_MODEL = SCRIPT_DIR / "models" / "classifier.pt"

DET_CONF_THRESHOLD = 0.30
DET_IOU_THRESHOLD = 0.45
DET_IMGSZ = 256
MAX_EDGE = 256
MAX_LEAVES = 1
CLF_CONF_THRESHOLD = 0.765
HIGH_CONF_THRESHOLD = 0.765
MIN_FRAME_FPS = 5.0
MAX_FRAME_LATENCY_MS = 200.0
PREFERRED_FRAME_LATENCY_MS = 180.0
PRIMARY_FRAMES = 2
BACKUP_FRAMES = 1
THREADS = 4


@dataclass(frozen=True)
class PlantDecision:
    label: str
    confidence: float
    status: str
    frames_used: int
    reason: str


def _signal_handler(sig, frame):
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)


def parse_args():
    p = argparse.ArgumentParser(description="Fast Raspberry Pi inference")
    p.add_argument("image", help="Path to input image")
    p.add_argument("--det-model", default=str(DEFAULT_DET_MODEL))
    p.add_argument("--clf-model", default=str(DEFAULT_CLF_MODEL))
    p.add_argument("--det-imgsz", type=int, default=DET_IMGSZ)
    p.add_argument("--det-conf", type=float, default=DET_CONF_THRESHOLD)
    p.add_argument("--det-iou", type=float, default=DET_IOU_THRESHOLD)
    p.add_argument("--max-edge", type=int, default=MAX_EDGE)
    p.add_argument("--max-leaves", type=int, default=MAX_LEAVES)
    p.add_argument("--crop-mode", choices=["none", "center", "mask"], default="none")
    p.add_argument("--crop-scale", type=float, default=0.70)
    p.add_argument("--crop-pad", type=float, default=0.12)
    p.add_argument("--conf", type=float, default=CLF_CONF_THRESHOLD)
    p.add_argument("--high-conf", type=float, default=HIGH_CONF_THRESHOLD)
    p.add_argument("--threads", type=int, default=THREADS)
    p.add_argument("--with-detector", action="store_true")
    return p.parse_args()


def setup_runtime(threads: int):
    if not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    cv2.setNumThreads(threads)
    try:
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    except ImportError:
        pass
    except RuntimeError:
        # PyTorch refuses to change inter-op threads after work has started;
        # the already configured runtime is still valid in that case.
        pass


def _infer_task_from_export_dir(model_dir: Path) -> str | None:
    meta = model_dir / "metadata.yaml"
    if not meta.exists():
        return None
    try:
        import yaml

        with open(meta, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        task = payload.get("task")
        return str(task) if task in {"classify", "detect", "segment", "pose"} else None
    except (OSError, TypeError, ValueError):
        return None


def load_model(path_str: str, task: str | None = None):
    from ultralytics import YOLO

    model_path = Path(path_str)
    if not model_path.exists() or not (model_path.is_file() or model_path.is_dir()):
        raise SystemExit(f"[ERROR] Model not found: {model_path}")

    kwargs = {}
    if task:
        kwargs["task"] = task
    elif model_path.is_dir():
        inferred = _infer_task_from_export_dir(model_path)
        if inferred:
            kwargs["task"] = inferred

    return YOLO(str(model_path), **kwargs)


def warmup_model(model, task: str, imgsz: int = DET_IMGSZ):
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    try:
        model.predict(source=dummy, verbose=False, imgsz=imgsz)
    except TypeError:
        model.predict(source=dummy, verbose=False)


def load_models(
    det_model_path: str = str(DEFAULT_DET_MODEL),
    clf_model_path: str = str(DEFAULT_CLF_MODEL),
    threads: int = THREADS,
    with_detector: bool = True,
    det_imgsz: int = DET_IMGSZ,
):
    if det_imgsz <= 0:
        raise ValueError("det_imgsz must be positive")
    setup_runtime(threads)
    clf_model = load_model(clf_model_path, task="classify")
    warmup_model(clf_model, task="classify", imgsz=224)

    det_model = None
    if with_detector:
        det_model = load_model(det_model_path, task="detect")
        warmup_model(det_model, task="detect", imgsz=det_imgsz)

    return det_model, clf_model


def downscale_if_needed(img: np.ndarray, max_edge: int = MAX_EDGE) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_edge:
        return img
    scale = max_edge / max(h, w)
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)


def center_crop(img: np.ndarray, crop_scale: float = 0.70) -> np.ndarray:
    scale = max(0.05, min(1.0, float(crop_scale)))
    h, w = img.shape[:2]
    crop_w = max(1, int(round(w * scale)))
    crop_h = max(1, int(round(h * scale)))
    x1 = max(0, (w - crop_w) // 2)
    y1 = max(0, (h - crop_h) // 2)
    return img[y1:y1 + crop_h, x1:x1 + crop_w]


def mask_crop(
    img: np.ndarray,
    crop_pad: float = 0.12,
    fallback_scale: float = 0.70,
) -> np.ndarray:
    h, w = img.shape[:2]
    if h < 4 or w < 4:
        return img

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    green_yellow = (hue >= 18) & (hue <= 95) & (sat >= 25) & (val >= 30)
    red_brown = (((hue <= 18) | (hue >= 165)) & (sat >= 45) & (val >= 30))
    saturated_leaf = (sat >= 45) & (val >= 35) & (val <= 245)
    mask = (green_yellow | red_brown | saturated_leaf).astype(np.uint8) * 255

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(24.0, 0.004 * h * w)
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        boxes.append((x, y, x + bw, y + bh))

    if not boxes:
        return center_crop(img, fallback_scale)

    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)

    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        return center_crop(img, fallback_scale)

    if (box_w * box_h) > 0.92 * h * w:
        return center_crop(img, fallback_scale)

    pad = int(round(max(box_w, box_h) * max(0.0, float(crop_pad))))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return img[y1:y2, x1:x2]


def detect_leaves(
    det_model,
    img: np.ndarray,
    det_conf: float = DET_CONF_THRESHOLD,
    det_iou: float = DET_IOU_THRESHOLD,
    det_imgsz: int = DET_IMGSZ,
    max_leaves: int = MAX_LEAVES,
) -> list[np.ndarray]:
    if max_leaves < 0:
        raise ValueError("max_leaves must be >= 0")
    if det_model is None:
        return []

    result = det_model.predict(
        source=img,
        conf=det_conf,
        iou=det_iou,
        imgsz=det_imgsz,
        augment=False,
        verbose=False,
    )[0]

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    valid = np.isfinite(xyxy).all(axis=1) & np.isfinite(confs)
    xyxy = xyxy[valid]
    confs = confs[valid]
    if len(xyxy) == 0:
        return []
    order = np.argsort(-confs)
    if max_leaves > 0:
        order = order[:max_leaves]

    h, w = img.shape[:2]
    crops = []
    for index in order:
        x1, y1, x2, y2 = xyxy[index]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append(crop)
    return crops


def _find_healthy_idx(names) -> int | None:
    items = names.items() if isinstance(names, dict) else enumerate(names)
    for idx, name in items:
        lowered = str(name).lower()
        if "healthy" in lowered or "normal" in lowered:
            return int(idx)
    return None


def _result_to_record(result, model_names, clf_conf: float, healthy_idx: int | None) -> dict:
    probs = result.probs
    if probs is None:
        raise RuntimeError("Classifier result did not contain probabilities")
    top1 = int(probs.top1)
    confidence = float(probs.top1conf)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise RuntimeError(f"Classifier returned invalid confidence: {confidence!r}")
    if not math.isfinite(float(clf_conf)) or not 0.0 <= float(clf_conf) <= 1.0:
        raise ValueError("clf_conf must be between 0 and 1")
    if isinstance(model_names, dict):
        valid_index = top1 in model_names
    else:
        valid_index = 0 <= top1 < len(model_names)
    if not valid_index:
        raise RuntimeError(f"Classifier returned unknown class index: {top1}")
    label = str(model_names[top1])

    is_healthy = "healthy" in label.lower() or "normal" in label.lower()
    if healthy_idx is not None and top1 == healthy_idx:
        is_healthy = True

    if confidence < clf_conf:
        return {
            "label": "Uncertain",
            "confidence": confidence,
            "raw_label": label,
            "is_disease": False,
            "is_uncertain": True,
        }

    if is_healthy:
        return {
            "label": "Healthy",
            "confidence": confidence,
            "raw_label": label,
            "is_disease": False,
            "is_uncertain": False,
        }

    return {
        "label": label,
        "confidence": confidence,
        "raw_label": label,
        "is_disease": True,
        "is_uncertain": False,
    }


def classify_crops(
    clf_model,
    crops: list[np.ndarray],
    clf_conf: float = CLF_CONF_THRESHOLD,
    healthy_idx: int | None = None,
) -> list[dict]:
    if not crops:
        return []
    results = clf_model.predict(source=crops, verbose=False)
    if len(results) != len(crops):
        raise RuntimeError(
            f"Classifier returned {len(results)} results for {len(crops)} crops"
        )
    return [
        _result_to_record(result, clf_model.names, clf_conf, healthy_idx)
        for result in results
    ]


def classify_crop(
    clf_model,
    crop: np.ndarray,
    clf_conf: float = CLF_CONF_THRESHOLD,
    healthy_idx: int | None = None,
) -> dict:
    records = classify_crops(clf_model, [crop], clf_conf, healthy_idx)
    if records:
        return records[0]
    return {
        "label": "Uncertain",
        "confidence": 0.0,
        "raw_label": "Uncertain",
        "is_disease": False,
        "is_uncertain": True,
    }


def compute_verdict(records: list[dict]) -> str:
    if not records:
        return "Uncertain"
    diseased = [record["label"] for record in records if record.get("is_disease")]
    if diseased:
        return Counter(diseased).most_common(1)[0][0]
    if any(record.get("is_uncertain") for record in records):
        return "Uncertain"
    if records:
        return "Healthy"
    return "Uncertain"


def infer_frame(
    det_model,
    clf_model,
    img: np.ndarray,
    *,
    with_detector: bool = False,
    det_conf: float = DET_CONF_THRESHOLD,
    det_iou: float = DET_IOU_THRESHOLD,
    det_imgsz: int = DET_IMGSZ,
    max_edge: int = MAX_EDGE,
    max_leaves: int = MAX_LEAVES,
    crop_mode: str = "none",
    crop_scale: float = 0.70,
    crop_pad: float = 0.12,
    clf_conf: float = CLF_CONF_THRESHOLD,
    healthy_idx: int | None = None,
) -> dict:
    for name, value in (
        ("det_conf", det_conf),
        ("det_iou", det_iou),
        ("crop_scale", crop_scale),
        ("crop_pad", crop_pad),
        ("clf_conf", clf_conf),
    ):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= float(det_conf) <= 1.0 or not 0.0 <= float(det_iou) <= 1.0:
        raise ValueError("det_conf and det_iou must be between 0 and 1")
    if not 0.0 <= float(clf_conf) <= 1.0:
        raise ValueError("clf_conf must be between 0 and 1")
    if max_edge <= 0 or det_imgsz <= 0 or max_leaves < 0:
        raise ValueError("max_edge/det_imgsz must be positive and max_leaves cannot be negative")
    if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3 or img.size == 0:
        raise ValueError("img must be a non-empty BGR image array")
    img_small = downscale_if_needed(img, max_edge)
    crops = (
        detect_leaves(
            det_model,
            img_small,
            det_conf=det_conf,
            det_iou=det_iou,
            det_imgsz=det_imgsz,
            max_leaves=max_leaves,
        )
        if with_detector
        else []
    )
    if not crops:
        if crop_mode == "mask":
            crops = [mask_crop(img_small, crop_pad, crop_scale)]
        elif crop_mode == "center":
            crops = [center_crop(img_small, crop_scale)]
        else:
            crops = [img_small]

    records = classify_crops(clf_model, crops, clf_conf, healthy_idx)
    verdict = compute_verdict(records)
    confidence = max((record["confidence"] for record in records), default=0.0)
    return {
        "label": verdict,
        "confidence": confidence,
        "records": records,
        "is_uncertain": verdict == "Uncertain",
    }


class PlantDecisionGate:
    """Two primary frames plus one backup frame confidence gate."""

    def __init__(
        self,
        *,
        high_conf: float = HIGH_CONF_THRESHOLD,
        max_frames: int = PRIMARY_FRAMES + BACKUP_FRAMES,
    ):
        if not math.isfinite(float(high_conf)) or not 0.0 <= float(high_conf) <= 1.0:
            raise ValueError("high_conf must be between 0 and 1")
        if max_frames < PRIMARY_FRAMES:
            raise ValueError("max_frames must allow the two primary frames")
        self.high_conf = high_conf
        self.max_frames = max_frames
        self.frames: list[dict] = []

    def reset(self):
        self.frames.clear()

    def add(self, prediction: dict) -> PlantDecision | None:
        if not isinstance(prediction, dict):
            raise TypeError("prediction must be a mapping")
        confidence = float(prediction.get("confidence", 0.0))
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("prediction confidence must be between 0 and 1")
        self.frames.append(prediction)
        if len(self.frames) >= PRIMARY_FRAMES:
            first, second = self.frames[0], self.frames[1]
            if self._agree_high_conf(first, second):
                return self._decision(first["label"], [first, second], "primary_agreement")

        if len(self.frames) >= self.max_frames:
            return self._fallback_decision()

        return None

    def _agree_high_conf(self, left: dict, right: dict) -> bool:
        return (
            left.get("label") == right.get("label")
            and left.get("label") != "Uncertain"
            and float(left.get("confidence", 0.0)) >= self.high_conf
            and float(right.get("confidence", 0.0)) >= self.high_conf
        )

    def _decision(self, label: str, frames: list[dict], reason: str) -> PlantDecision:
        confidence = min(float(frame.get("confidence", 0.0)) for frame in frames)
        return PlantDecision(
            label=label,
            confidence=confidence,
            status="ok" if label != "Uncertain" else "uncertain",
            frames_used=len(self.frames),
            reason=reason,
        )

    def _fallback_decision(self) -> PlantDecision:
        usable = [
            frame
            for frame in self.frames
            if frame.get("label") != "Uncertain"
            and float(frame.get("confidence", 0.0)) >= self.high_conf
        ]
        if not usable:
            return self._decision("Uncertain", self.frames, "low_confidence")

        labels = Counter(frame["label"] for frame in usable)
        label, votes = labels.most_common(1)[0]
        tied = sum(1 for _, count in labels.items() if count == votes) > 1
        if votes >= 2 and not tied:
            voted = [frame for frame in usable if frame["label"] == label]
            return self._decision(label, voted, "backup_majority")

        return self._decision("Uncertain", self.frames, "disagreement")


def main():
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"[ERROR] Image not found: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"[ERROR] Cannot read image: {image_path}")

    det_model, clf_model = load_models(
        det_model_path=args.det_model,
        clf_model_path=args.clf_model,
        threads=args.threads,
        with_detector=args.with_detector,
        det_imgsz=args.det_imgsz,
    )
    healthy_idx = _find_healthy_idx(clf_model.names)

    prediction = infer_frame(
        det_model,
        clf_model,
        img,
        with_detector=args.with_detector,
        det_conf=args.det_conf,
        det_iou=args.det_iou,
        det_imgsz=args.det_imgsz,
        max_edge=args.max_edge,
        max_leaves=args.max_leaves,
        crop_mode=args.crop_mode,
        crop_scale=args.crop_scale,
        crop_pad=args.crop_pad,
        clf_conf=args.conf,
        healthy_idx=healthy_idx,
    )
    print(prediction["label"])


if __name__ == "__main__":
    main()
