#!/usr/bin/env python3
"""
Sweep classifier confidence gates for the fast Raspberry Pi deployment path.

The Pi benchmark with --conf 0.00 measures raw frame accuracy. Deployment should
not force low-confidence labels, so this script runs inference once, records each
prediction confidence, then evaluates many confidence thresholds without rerunning
the model. A row's "emitted accuracy" is the accuracy among non-uncertain frames.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2

from inference_rpi import (
    DET_CONF_THRESHOLD,
    DET_IMGSZ,
    DET_IOU_THRESHOLD,
    MAX_EDGE,
    MAX_LEAVES,
    THREADS,
    _find_healthy_idx,
    infer_frame,
    load_models,
)
from runtime_suite_common import (
    DATASET_TRACKING_DIR,
    MAX_SELECTED_FRAME_LATENCY_MS,
    MIN_ACCEPTED_FRAME_ACCURACY,
    MIN_PIPELINE_FPS,
    PREFERRED_SELECTED_FRAME_LATENCY_MS,
    STRICT_FRAME_ACCURACY,
    write_json,
)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_float_list(value: str) -> list[float]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    if not out:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Sweep confidence gates for fast classifier deployment")
    p.add_argument("--test", default="clf_dataset/test", help="Class-folder test split")
    p.add_argument("--clf-model", default="runtime_exports/classifier_openvino_model")
    p.add_argument("--det-model", default="runtime_exports/detector_openvino_model")
    p.add_argument("--with-detector", action="store_true", help="Optional: score detector+classifier instead")
    p.add_argument("--det-imgsz", type=int, default=DET_IMGSZ)
    p.add_argument("--det-conf", type=float, default=DET_CONF_THRESHOLD)
    p.add_argument("--det-iou", type=float, default=DET_IOU_THRESHOLD)
    p.add_argument("--max-leaves", type=int, default=MAX_LEAVES)
    p.add_argument("--max-edge", type=int, default=MAX_EDGE)
    p.add_argument("--crop-mode", choices=["none", "center", "mask"], default="mask")
    p.add_argument("--crop-scale", type=float, default=0.55)
    p.add_argument("--crop-pad", type=float, default=0.05)
    p.add_argument("--threads", type=int, default=THREADS)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--sequential-limit", action="store_true")
    p.add_argument("--thresholds", type=parse_float_list, default=None)
    p.add_argument("--threshold-min", type=float, default=0.0)
    p.add_argument("--threshold-max", type=float, default=0.99)
    p.add_argument("--threshold-step", type=float, default=0.01)
    p.add_argument("--min-coverage", type=float, default=0.70)
    p.add_argument("--out-dir", default=str(DATASET_TRACKING_DIR))
    return p.parse_args()


def normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().replace("_", " ").replace("-", " ").split())


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

    limited = []
    labels = sorted(by_label)
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


def build_thresholds(args) -> list[float]:
    if args.thresholds is not None:
        values = args.thresholds
    else:
        if args.threshold_step <= 0:
            raise SystemExit("[ERROR] --threshold-step must be > 0")
        count = int(round((args.threshold_max - args.threshold_min) / args.threshold_step)) + 1
        values = [args.threshold_min + i * args.threshold_step for i in range(max(0, count))]

    cleaned = []
    values = [0.0, *values]
    for value in values:
        value = float(value)
        if not math.isfinite(value):
            raise SystemExit("[ERROR] confidence thresholds must be finite")
        value = round(max(0.0, min(1.0, value)), 6)
        if value not in cleaned:
            cleaned.append(value)
    return sorted(cleaned)


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def score_images(args, samples: list[tuple[Path, str]]) -> tuple[list[dict], dict]:
    det_model, clf_model = load_models(
        det_model_path=args.det_model,
        clf_model_path=args.clf_model,
        threads=args.threads,
        with_detector=args.with_detector,
        det_imgsz=args.det_imgsz,
    )
    healthy_idx = _find_healthy_idx(clf_model.names)

    total_time = 0.0
    max_latency_ms = 0.0
    over_preferred = 0
    rows = []

    pbar = None
    try:
        from tqdm import tqdm

        pbar = tqdm(total=len(samples), desc="Scoring", unit="img")
    except Exception:
        pass

    for img_path, true_label in samples:
        img = cv2.imread(str(img_path))
        if img is None:
            if pbar:
                pbar.update(1)
            continue

        started = time.perf_counter()
        pred = infer_frame(
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
            clf_conf=0.0,
            healthy_idx=healthy_idx,
        )
        elapsed = time.perf_counter() - started

        latency_ms = elapsed * 1000.0
        total_time += elapsed
        max_latency_ms = max(max_latency_ms, latency_ms)
        if latency_ms > PREFERRED_SELECTED_FRAME_LATENCY_MS:
            over_preferred += 1

        label = str(pred["label"])
        confidence = float(pred["confidence"])
        correct = normalize_label(label) == normalize_label(true_label)
        rows.append(
            {
                "image": str(img_path),
                "true": true_label,
                "pred": label,
                "confidence": confidence,
                "correct": correct,
                "latency_ms": latency_ms,
            }
        )

        if pbar:
            raw_acc = sum(1 for row in rows if row["correct"]) / len(rows)
            fps = len(rows) / total_time if total_time > 0 else 0.0
            pbar.set_postfix(acc=f"{raw_acc * 100:.1f}%", fps=f"{fps:.2f}")
            pbar.update(1)

    if pbar:
        pbar.close()

    processed = len(rows)
    if processed != len(samples):
        raise RuntimeError(
            f"Confidence sweep incomplete: processed {processed} of {len(samples)} images"
        )
    timing = {
        "processed": processed,
        "total_time_sec": total_time,
        "avg_fps": processed / total_time if total_time > 0 else 0.0,
        "avg_latency_ms": (total_time * 1000.0 / processed) if processed else 0.0,
        "max_latency_ms": max_latency_ms,
        "over_preferred_latency_frames": over_preferred,
    }
    return rows, timing


def sweep_thresholds(predictions: list[dict], thresholds: list[float]) -> list[dict]:
    total = len(predictions)
    rows = []
    for threshold in thresholds:
        emitted = [row for row in predictions if float(row["confidence"]) >= threshold]
        uncertain = total - len(emitted)
        correct = sum(1 for row in emitted if row["correct"])
        wrong = len(emitted) - correct
        emitted_accuracy = correct / len(emitted) if emitted else 0.0
        coverage = len(emitted) / total if total else 0.0
        rows.append(
            {
                "threshold": threshold,
                "emitted_accuracy": emitted_accuracy,
                "coverage": coverage,
                "wrong_rate": wrong / total if total else 0.0,
                "uncertain_rate": uncertain / total if total else 0.0,
                "emitted": len(emitted),
                "correct": correct,
                "wrong": wrong,
                "uncertain": uncertain,
                "total": total,
                "passes_accepted": emitted_accuracy >= MIN_ACCEPTED_FRAME_ACCURACY,
                "passes_strict": emitted_accuracy >= STRICT_FRAME_ACCURACY,
            }
        )
    return rows


def choose_row(
    rows: list[dict],
    target: float,
    min_coverage: float,
    timing: dict | None = None,
) -> dict | None:
    speed_ok = timing is None or (
        timing.get("avg_fps", 0.0) >= MIN_PIPELINE_FPS
        and timing.get("max_latency_ms", float("inf")) <= MAX_SELECTED_FRAME_LATENCY_MS
    )
    eligible = [
        row for row in rows
        if row["emitted"] > 0
        and row["emitted_accuracy"] >= target
        and row["coverage"] >= min_coverage
        and speed_ok
    ]
    if not eligible:
        eligible = [
            row for row in rows
            if row["emitted"] > 0 and row["emitted_accuracy"] >= target and speed_ok
        ]
    if not eligible:
        return None
    return max(eligible, key=lambda row: (row["coverage"], row["emitted_accuracy"], -row["threshold"]))


def per_class_at_threshold(predictions: list[dict], threshold: float) -> list[dict]:
    stats = defaultdict(lambda: {"total": 0, "emitted": 0, "correct": 0, "wrong": 0, "uncertain": 0})
    for row in predictions:
        label = row["true"]
        stats[label]["total"] += 1
        if float(row["confidence"]) < threshold:
            stats[label]["uncertain"] += 1
            continue
        stats[label]["emitted"] += 1
        if row["correct"]:
            stats[label]["correct"] += 1
        else:
            stats[label]["wrong"] += 1

    out = []
    for label in sorted(stats):
        item = stats[label]
        emitted = item["emitted"]
        total = item["total"]
        out.append(
            {
                "class": label,
                "emitted_accuracy": item["correct"] / emitted if emitted else 0.0,
                "coverage": emitted / total if total else 0.0,
                **item,
            }
        )
    return out


def print_summary(rows: list[dict], timing: dict, strict_row: dict | None, accepted_row: dict | None, best_accuracy_row: dict):
    raw = rows[0] if rows and rows[0]["threshold"] == 0 else min(rows, key=lambda row: row["threshold"])

    print("\n" + "=" * 74)
    print("Classifier Confidence Gate Sweep")
    print("=" * 74)
    print(f"Images        : {raw['total']}")
    print(f"Raw accuracy  : {raw['emitted_accuracy'] * 100:.2f}% ({raw['correct']}/{raw['total']})")
    print(f"Avg FPS       : {timing['avg_fps']:.2f}")
    print(f"Avg latency   : {timing['avg_latency_ms']:.0f} ms/img")
    print(f"Max latency   : {timing['max_latency_ms']:.0f} ms/img")
    print(f"Pi speed      : {'PASS' if timing['avg_fps'] >= MIN_PIPELINE_FPS else 'FAIL'}")
    print()

    def line(title: str, row: dict | None):
        if row is None:
            print(f"{title:<15}: none")
            return
        print(
            f"{title:<15}: conf>={row['threshold']:.2f}  "
            f"emit_acc={row['emitted_accuracy'] * 100:6.2f}%  "
            f"coverage={row['coverage'] * 100:6.2f}%  "
            f"wrong={row['wrong']:3d}  uncertain={row['uncertain']:3d}"
        )

    line("Strict 97.2", strict_row)
    line("Accepted 96", accepted_row)
    line("Best accuracy", best_accuracy_row)

    interesting = {0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95}
    for row in (strict_row, accepted_row, best_accuracy_row):
        if row is not None:
            interesting.add(float(row["threshold"]))

    print("\n  conf  emit_acc  coverage  wrong  uncertain  emitted")
    print("  ----  --------  --------  -----  ---------  -------")
    for row in rows:
        if round(float(row["threshold"]), 6) not in {round(v, 6) for v in interesting}:
            continue
        print(
            f"  {row['threshold']:>4.2f}  "
            f"{row['emitted_accuracy'] * 100:>7.2f}%  "
            f"{row['coverage'] * 100:>7.2f}%  "
            f"{row['wrong']:>5d}  "
            f"{row['uncertain']:>9d}  "
            f"{row['emitted']:>7d}"
        )


def main():
    args = parse_args()
    if args.limit < 0 or args.threads < 1 or args.max_edge <= 0 or args.det_imgsz <= 0:
        raise SystemExit("[ERROR] limit must be >= 0; threads, max-edge, and det-imgsz must be positive")
    for name in ("det_conf", "det_iou", "crop_scale", "crop_pad", "threshold_min", "threshold_max", "threshold_step", "min_coverage"):
        value = float(getattr(args, name))
        if not math.isfinite(value):
            raise SystemExit(f"[ERROR] {name} must be finite")
    for name in ("det_conf", "det_iou", "min_coverage", "threshold_min", "threshold_max"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"[ERROR] {name} must be between 0 and 1")
    if args.threshold_max < args.threshold_min or args.threshold_step <= 0:
        raise SystemExit("[ERROR] threshold range must be ordered and step must be positive")
    test_dir = Path(args.test)
    if not test_dir.is_dir():
        raise SystemExit(f"[ERROR] Test folder not found: {test_dir}")

    samples = collect_samples(test_dir)
    if args.limit > 0:
        samples = samples[:args.limit] if args.sequential_limit else limit_samples_balanced(samples, args.limit)
    if not samples:
        raise SystemExit(f"[ERROR] No images found under: {test_dir}")

    thresholds = build_thresholds(args)
    predictions, timing = score_images(args, samples)
    sweep_rows = sweep_thresholds(predictions, thresholds)
    strict_row = choose_row(sweep_rows, STRICT_FRAME_ACCURACY, args.min_coverage, timing)
    accepted_row = choose_row(sweep_rows, MIN_ACCEPTED_FRAME_ACCURACY, args.min_coverage, timing)
    best_accuracy_row = max(
        [row for row in sweep_rows if row["emitted"] > 0],
        key=lambda row: (row["emitted_accuracy"], row["coverage"], -row["threshold"]),
    )
    recommended = strict_row or accepted_row
    diagnostic_row = recommended or best_accuracy_row

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = out_dir / "classifier_confidence_gate_predictions.csv"
    sweep_csv = out_dir / "classifier_confidence_gate_sweep.csv"
    best_json = out_dir / "classifier_confidence_gate_best.json"

    write_csv(predictions_csv, predictions)
    write_csv(sweep_csv, sweep_rows)
    payload = {
        "status": "pass" if recommended is not None else "fail",
        "complete": len(predictions) == len(samples),
        "targets": {
            "accepted_frame_accuracy": MIN_ACCEPTED_FRAME_ACCURACY,
            "strict_frame_accuracy": STRICT_FRAME_ACCURACY,
            "min_fps": MIN_PIPELINE_FPS,
            "min_coverage_for_recommendation": args.min_coverage,
        },
        "settings": {
            "test": str(test_dir.resolve()),
            "classifier": str(Path(args.clf_model).resolve()),
            "with_detector": args.with_detector,
            "max_edge": args.max_edge,
            "crop_mode": args.crop_mode,
            "crop_scale": args.crop_scale,
            "crop_pad": args.crop_pad,
            "threads": args.threads,
        },
        "timing": timing,
        "strict": strict_row,
        "accepted": accepted_row,
        "best_accuracy": best_accuracy_row,
        "recommended": recommended,
        "per_class_recommended": per_class_at_threshold(predictions, diagnostic_row["threshold"]),
        "predictions_csv": str(predictions_csv.resolve()),
        "sweep_csv": str(sweep_csv.resolve()),
    }
    write_json(best_json, payload)

    print_summary(sweep_rows, timing, strict_row, accepted_row, best_accuracy_row)
    print(f"\nCSV  -> {sweep_csv}")
    print(f"JSON -> {best_json}")
    if recommended is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
