#!/usr/bin/env python3
"""
benchmark_rpi.py — FPS + Accuracy benchmark for the leaf disease pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Runs the full detect+classify pipeline on every image in a test
folder structured as:

  test_dir/
    Early_blight/
      img1.jpg  img2.jpg  ...
    Healthy/
      ...
    Spotted Wilt Virus/
      ...
    ...  (8 classes)

Measures:
  • Per-class accuracy
  • Overall accuracy
  • FPS (frames per second, full pipeline including detection)

USAGE:
  python3 benchmark_rpi.py --test /path/to/test
  python3 benchmark_rpi.py --test /path/to/test --conf 0.50
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

# Re-use pipeline functions from inference_rpi in same directory
sys.path.insert(0, str(Path(__file__).parent))
from inference_rpi import (
    load_models,
    downscale_if_needed,
    detect_leaves,
    classify_crops,
    compute_verdict,
    _find_healthy_idx,
    CLF_CONF_THRESHOLD,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark FPS and accuracy")
    p.add_argument("--test", required=True,
                   help="Path to test folder (class subfolders = ground truth)")
    p.add_argument("--conf", type=float, default=CLF_CONF_THRESHOLD,
                   help=f"Classifier confidence threshold (default {CLF_CONF_THRESHOLD})")
    return p.parse_args()


def collect_samples(test_dir: Path):
    """
    Returns list of (image_path, true_label) from class subfolders.
    Folder name = ground-truth class label.
    """
    samples = []
    for cls_dir in sorted(test_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        label = cls_dir.name
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() in IMG_EXTS:
                samples.append((img_path, label))
    return samples


def normalise_label(label: str) -> str:
    """Lower-case + strip for robust comparison."""
    return label.strip().lower()


def run_benchmark(test_dir: Path, clf_conf: float):
    det_model, clf_model = load_models()
    healthy_idx = _find_healthy_idx(clf_model.names)

    samples = collect_samples(test_dir)
    if not samples:
        sys.exit(f"[ERROR] No images found in: {test_dir}")

    n_total   = len(samples)
    n_correct = 0

    # per-class tracking: {label: [correct, total]}
    class_stats = defaultdict(lambda: [0, 0])

    total_time = 0.0

    print(f"Running benchmark on {n_total} images ...")
    print(f"Confidence threshold: {clf_conf}\n")

    try:
        from tqdm import tqdm
        pbar = tqdm(total=n_total, desc="Processing", unit="img")
    except ImportError:
        pbar = None

    for i, (img_path, true_label) in enumerate(samples, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            if pbar:
                pbar.update(1)
            continue

        t_start = time.perf_counter()

        img_scaled  = downscale_if_needed(img)
        crops       = detect_leaves(det_model, img_scaled)
        if not crops:
            crops = [img_scaled]

        leaf_results = classify_crops(clf_model, crops, clf_conf, healthy_idx)
        verdict = compute_verdict(leaf_results)

        elapsed = time.perf_counter() - t_start
        total_time += elapsed

        correct = normalise_label(verdict) == normalise_label(true_label)
        if correct:
            n_correct += 1

        class_stats[true_label][1] += 1
        if correct:
            class_stats[true_label][0] += 1

        running_acc = n_correct / i * 100
        fps = i / total_time if total_time > 0 else 0.0
        if pbar:
            pbar.set_postfix(acc=f"{running_acc:.1f}%", fps=f"{fps:.2f}")
            pbar.update(1)

    if pbar:
        pbar.close()
    print()

    # ── Results ───────────────────────────────────────────────────────────────
    overall_acc = n_correct / n_total * 100
    avg_fps     = n_total / total_time

    sep = "=" * 56
    print(f"\n{sep}")
    print(f"  Benchmark Results")
    print(f"{sep}")
    print(f"  Images tested : {n_total}")
    print(f"  Overall acc   : {overall_acc:.2f}%  ({n_correct}/{n_total})")
    print(f"  Avg FPS       : {avg_fps:.2f}")
    print(f"  Total time    : {total_time:.1f}s")
    print()
    print(f"  Per-class accuracy:")
    for label in sorted(class_stats):
        correct_c, total_c = class_stats[label]
        acc_c = correct_c / total_c * 100 if total_c else 0
        bar = "#" * int(acc_c / 5)
        print(f"    {label:<28s}  {acc_c:5.1f}%  ({correct_c}/{total_c})  {bar}")
    print(f"{sep}\n")


def main():
    args = parse_args()
    test_dir = Path(args.test)
    if not test_dir.is_dir():
        sys.exit(f"[ERROR] Test folder not found: {test_dir}")
    run_benchmark(test_dir, args.conf)


if __name__ == "__main__":
    main()
