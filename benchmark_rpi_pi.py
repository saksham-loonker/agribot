#!/usr/bin/env python3
"""
benchmark_rpi_pi.py
===================
Fast Raspberry Pi benchmark for the crop-detect + classify pipeline.

This is intended as the low-latency benchmark entrypoint for Pi 5-class
devices. It reuses the optimized helpers from inference_rpi.py so benchmark
numbers and single-image inference stay consistent.
"""

import argparse
import math
import signal
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2

from inference_rpi import (
    CLF_CONF_THRESHOLD,
    DET_CONF_THRESHOLD,
    DET_IMGSZ,
    DET_IOU_THRESHOLD,
    MAX_FRAME_LATENCY_MS,
    MAX_EDGE,
    MAX_LEAVES,
    MIN_FRAME_FPS,
    PREFERRED_FRAME_LATENCY_MS,
    _find_healthy_idx,
    infer_frame,
    load_models,
)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_INTERRUPTED = False


def _signal_handler(sig, frame):
    global _INTERRUPTED
    _INTERRUPTED = True
    print("\n\n[!] Interrupted - printing results so far...\n")


signal.signal(signal.SIGINT, _signal_handler)


def parse_args():
    p = argparse.ArgumentParser(description="Raspberry Pi benchmark")
    p.add_argument("--test", required=True, help="Folder with class subfolders")
    p.add_argument("--det-model", default="./models/detector.pt")
    p.add_argument("--clf-model", default="./models/classifier.pt")
    p.add_argument("--det-imgsz", type=int, default=DET_IMGSZ)
    p.add_argument("--det-conf", type=float, default=DET_CONF_THRESHOLD)
    p.add_argument("--det-iou", type=float, default=DET_IOU_THRESHOLD)
    p.add_argument("--max-edge", type=int, default=MAX_EDGE)
    p.add_argument("--max-leaves", type=int, default=MAX_LEAVES)
    p.add_argument("--crop-mode", choices=["none", "center", "mask"], default="none")
    p.add_argument("--crop-scale", type=float, default=0.70)
    p.add_argument("--crop-pad", type=float, default=0.12)
    p.add_argument("--conf", type=float, default=CLF_CONF_THRESHOLD)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--sequential-limit", action="store_true")
    p.add_argument("--with-detector", action="store_true")
    return p.parse_args()


def collect_samples(test_dir: Path) -> list[tuple[Path, str]]:
    samples = []
    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in IMG_EXTS:
                samples.append((img_path, class_dir.name))
    return samples


def limit_samples_balanced(samples: list[tuple[Path, str]], limit: int) -> list[tuple[Path, str]]:
    if limit <= 0 or len(samples) <= limit:
        return samples

    by_label: dict[str, deque[tuple[Path, str]]] = defaultdict(deque)
    for sample in samples:
        by_label[sample[1]].append(sample)

    labels = sorted(by_label)
    limited = []
    while len(limited) < limit:
        added = False
        for label in labels:
            if by_label[label]:
                limited.append(by_label[label].popleft())
                added = True
                if len(limited) >= limit:
                    break
        if not added:
            break
    return limited


def normalize_label(label: str) -> str:
    return label.strip().lower()


def run_benchmark(args):
    test_dir = Path(args.test)
    if not test_dir.is_dir():
        raise SystemExit(f"[ERROR] Test folder not found: {test_dir}")

    samples = collect_samples(test_dir)
    if args.limit > 0:
        if getattr(args, "sequential_limit", False):
            samples = samples[:args.limit]
        else:
            samples = limit_samples_balanced(samples, args.limit)
    if not samples:
        raise SystemExit(f"[ERROR] No images found in: {test_dir}")
    if args.threads < 1:
        raise ValueError("threads must be a positive integer")
    for name in ("det_conf", "det_iou", "conf"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if args.max_edge <= 0 or args.det_imgsz <= 0 or args.max_leaves < 0:
        raise ValueError("max_edge/det_imgsz must be positive and max_leaves cannot be negative")

    det_model, clf_model = load_models(
        det_model_path=args.det_model,
        clf_model_path=args.clf_model,
        threads=args.threads,
        with_detector=args.with_detector,
        det_imgsz=args.det_imgsz,
    )
    healthy_idx = _find_healthy_idx(clf_model.names)

    total = len(samples)
    processed = 0
    correct = 0
    total_time = 0.0
    max_latency_ms = 0.0
    over_preferred_latency = 0
    over_hard_latency = 0
    class_stats = defaultdict(lambda: [0, 0])

    print(f"\nMode      : {'detect+classify' if args.with_detector else 'classify-only'}")
    print(f"Classifier: {args.clf_model}")
    if args.with_detector:
        print(f"Detector  : {args.det_model}")
    print(f"Images    : {total}")
    print(f"Threads   : {args.threads}")
    print(f"Max edge  : {args.max_edge}px")
    print()

    pbar = None
    try:
        from tqdm import tqdm

        pbar = tqdm(total=total, desc="Benchmarking", unit="img")
    except Exception:
        pass

    for img_path, true_label in samples:
        if _INTERRUPTED:
            break

        img = cv2.imread(str(img_path))
        if img is None:
            if pbar:
                pbar.update(1)
            continue

        started = time.perf_counter()
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
        verdict = prediction["label"]

        elapsed = time.perf_counter() - started
        latency_ms = elapsed * 1000.0
        max_latency_ms = max(max_latency_ms, latency_ms)
        if latency_ms > PREFERRED_FRAME_LATENCY_MS:
            over_preferred_latency += 1
        if latency_ms > MAX_FRAME_LATENCY_MS:
            over_hard_latency += 1
        total_time += elapsed
        processed += 1

        is_correct = normalize_label(verdict) == normalize_label(true_label)
        if is_correct:
            correct += 1
        class_stats[true_label][1] += 1
        if is_correct:
            class_stats[true_label][0] += 1

        if pbar:
            acc = 100.0 * correct / processed
            fps = processed / total_time if total_time > 0 else 0.0
            pbar.set_postfix(acc=f"{acc:.1f}%", fps=f"{fps:.2f}")
            pbar.update(1)

    if pbar:
        pbar.close()

    if processed == 0:
        raise RuntimeError("No images were processed; refusing to emit a passing benchmark")

    overall_acc = 100.0 * correct / processed
    avg_fps = processed / total_time if total_time > 0 else 0.0
    avg_latency_ms = 1000.0 * total_time / processed
    meets_fps = avg_fps >= MIN_FRAME_FPS
    meets_latency = avg_latency_ms <= MAX_FRAME_LATENCY_MS and over_hard_latency == 0

    sep = "=" * 56
    status = " (INTERRUPTED)" if _INTERRUPTED else ""
    print(f"\n{sep}")
    print(f"Benchmark Results{status}")
    print(sep)
    print(f"Images tested : {processed}")
    print(f"Overall acc   : {overall_acc:.2f}%  ({correct}/{processed})")
    print(f"Avg FPS       : {avg_fps:.2f}")
    print(f"Total time    : {total_time:.1f}s")
    print(f"Avg latency   : {avg_latency_ms:.0f} ms/img")
    print(f"Max latency   : {max_latency_ms:.0f} ms/img")
    print(f">180 ms       : {over_preferred_latency}/{processed}")
    print(f">200 ms       : {over_hard_latency}/{processed}")
    print(f"Pi target     : {'PASS' if meets_fps and meets_latency else 'FAIL'}")
    print("\nPer-class accuracy:")
    for label in sorted(class_stats):
        class_correct, class_total = class_stats[label]
        class_acc = 100.0 * class_correct / class_total if class_total else 0.0
        print(f"  {label:<28s}  {class_acc:5.1f}%  ({class_correct}/{class_total})")
    print(f"{sep}\n")

    result = {
        "images_tested": processed,
        "images_requested": total,
        "complete": not _INTERRUPTED and processed == total,
        "overall_accuracy": overall_acc / 100.0,
        "avg_fps": avg_fps,
        "total_time_sec": total_time,
        "avg_latency_ms": avg_latency_ms,
        "max_latency_ms": max_latency_ms,
        "over_preferred_latency_frames": over_preferred_latency,
        "over_hard_latency_frames": over_hard_latency,
        "meets_min_fps": meets_fps,
        "meets_latency_ceiling": meets_latency,
        "per_class": {
            label: {
                "accuracy": (100.0 * class_correct / class_total if class_total else 0.0) / 100.0,
                "correct": class_correct,
                "total": class_total,
            }
            for label, (class_correct, class_total) in class_stats.items()
        },
    }
    if not result["complete"]:
        raise RuntimeError(
            f"Benchmark incomplete: processed {processed} of {total} images"
        )
    return result


def main():
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
