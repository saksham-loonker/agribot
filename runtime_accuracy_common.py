"""
runtime_accuracy_common.py
==========================
Shared classifier accuracy evaluation for exported runtimes.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from runtime_suite_common import (
    CLASSIFIER_IMGSZ,
    DEFAULT_CLASSIFIER_PT,
    DEFAULT_TEST_DIR,
    accuracy_report_dir,
    classifier_model_path,
    ensure_parent,
    normalize_runtime_name,
    suite_csv_path,
    upsert_csv,
    write_json,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def normalize_label(value: str) -> str:
    parts = value.strip().lower().replace("_", " ").replace("-", " ").split()
    return " ".join(parts)


def collect_samples(test_dir: Path) -> list[tuple[Path, str]]:
    if not test_dir.is_dir():
        return []
    samples = []
    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                samples.append((image_path, class_dir.name))
    return samples


def load_model(model_path: Path):
    from ultralytics import YOLO

    if not model_path.exists() or not (model_path.is_file() or model_path.is_dir()):
        raise SystemExit(f"[ERROR] Model not found: {model_path}")
    return YOLO(str(model_path), task="classify")


def predict_top1(model, paths: list[Path], imgsz: int) -> list[dict]:
    if not paths:
        return []
    if imgsz <= 0:
        raise ValueError("imgsz must be positive")
    results = model.predict(source=[str(path) for path in paths], imgsz=imgsz, verbose=False)
    if len(results) != len(paths):
        raise RuntimeError(
            f"Classifier returned {len(results)} results for {len(paths)} inputs; refusing incomplete accuracy"
        )
    rows = []
    for path, result in zip(paths, results):
        if result.probs is None:
            raise RuntimeError(f"Classifier returned no probabilities for {path}")
        top1 = int(result.probs.top1)
        confidence = float(result.probs.top1conf)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise RuntimeError(f"Classifier returned invalid confidence for {path}: {confidence!r}")
        names = model.names
        if isinstance(names, dict):
            valid_index = top1 in names
        else:
            valid_index = 0 <= top1 < len(names)
        if not valid_index:
            raise RuntimeError(f"Classifier returned unknown class index for {path}: {top1}")
        label = str(names[top1])
        rows.append(
            {
                "image": str(path),
                "pred": label,
                "pred_norm": normalize_label(label),
                "confidence": confidence,
            }
        )
    return rows


def predict_all(
    model,
    samples: list[tuple[Path, str]],
    imgsz: int,
    batch_size: int,
    progress_label: str | None = None,
) -> list[dict]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rows = []
    total = len(samples)
    if progress_label:
        print(f"  {progress_label}: 0/{total} (0.0%)", flush=True)

    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        batch_paths = [path for path, _ in batch]
        batch_rows = predict_top1(model, batch_paths, imgsz)
        for row, (_, true_label) in zip(batch_rows, batch):
            row["true"] = true_label
            row["true_norm"] = normalize_label(true_label)
            row["correct"] = row["pred_norm"] == row["true_norm"]
        rows.extend(batch_rows)

        if progress_label:
            done = min(start + len(batch), total)
            pct = (done / total * 100.0) if total else 100.0
            print(f"\r  {progress_label}: {done}/{total} ({pct:.1f}%)", end="", flush=True)

    if progress_label:
        print("", flush=True)
    return rows


def summarize_rows(rows: list[dict]) -> tuple[dict, list[dict], list[dict]]:
    class_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    disagreements = []

    correct = 0
    for row in rows:
        if row["correct"]:
            correct += 1
        else:
            disagreements.append(row)
        class_stats[row["true"]]["total"] += 1
        if row["correct"]:
            class_stats[row["true"]]["correct"] += 1

    per_class = []
    for label in sorted(class_stats):
        stats = class_stats[label]
        acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
        per_class.append(
            {
                "class": label,
                "accuracy": acc,
                "correct": stats["correct"],
                "total": stats["total"],
            }
        )

    overall = {
        "images": len(rows),
        "correct": correct,
        "accuracy": (correct / len(rows)) if rows else 0.0,
    }
    return overall, per_class, disagreements


def compare_with_baseline(rows: list[dict], baseline_rows: list[dict]) -> tuple[dict, list[dict]]:
    by_image = {row["image"]: row for row in baseline_rows}
    matched = 0
    same = 0
    drift = []

    for row in rows:
        baseline = by_image.get(row["image"])
        if not baseline:
            continue
        matched += 1
        if baseline["pred_norm"] == row["pred_norm"]:
            same += 1
        else:
            drift.append(
                {
                    "image": row["image"],
                    "true": row["true"],
                    "runtime_pred": row["pred"],
                    "runtime_conf": row["confidence"],
                    "pt_pred": baseline["pred"],
                    "pt_conf": baseline["confidence"],
                }
            )

    summary = {
        "matched_images": matched,
        "agreement": (same / matched) if matched else 0.0,
        "disagreements": len(drift),
    }
    if rows and matched != len(rows):
        raise RuntimeError(
            f"Baseline coverage is incomplete: matched {matched} of {len(rows)} images"
        )
    return summary, drift


def write_text_report(
    path: Path,
    runtime: str,
    model_path: Path,
    overall: dict,
    per_class: list[dict],
    elapsed: float,
    baseline_summary: dict | None,
    disagreements: list[dict],
    drift: list[dict] | None,
):
    lines = []
    lines.append("=" * 70)
    lines.append(f"Runtime Accuracy Report - {runtime}")
    lines.append("=" * 70)
    lines.append(f"Model   : {model_path}")
    lines.append(f"Images   : {overall['images']}")
    lines.append(f"Correct  : {overall['correct']}")
    lines.append(f"Accuracy : {overall['accuracy'] * 100:.2f}%")
    lines.append(f"Elapsed  : {elapsed:.2f}s")
    lines.append(
        f"FPS      : {(overall['images'] / elapsed):.2f}" if elapsed > 0 else "FPS      : n/a"
    )

    if baseline_summary is not None:
        lines.append(
            f"PT agreement : {baseline_summary['agreement'] * 100:.2f}% "
            f"({baseline_summary['matched_images'] - baseline_summary['disagreements']}/"
            f"{baseline_summary['matched_images']})"
        )

    lines.append("")
    lines.append(f"{'Class':<40} {'Acc':>8} {'Correct':>10} {'Total':>8}")
    lines.append("-" * 70)
    for row in per_class:
        lines.append(
            f"{row['class']:<40} {row['accuracy'] * 100:>7.2f}% "
            f"{row['correct']:>10} {row['total']:>8}"
        )

    if disagreements:
        lines.append("")
        lines.append("Top prediction errors:")
        for row in disagreements[:20]:
            lines.append(
                f"  {Path(row['image']).name}: true={row['true']} pred={row['pred']} "
                f"conf={row['confidence']:.4f}"
            )

    if drift:
        lines.append("")
        lines.append("Top PT/runtime drift cases:")
        for row in drift[:20]:
            lines.append(
                f"  {Path(row['image']).name}: true={row['true']} "
                f"pt={row['pt_pred']} ({row['pt_conf']:.4f}) "
                f"runtime={row['runtime_pred']} ({row['runtime_conf']:.4f})"
            )

    ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_accuracy(
    runtime: str,
    test_dir: Path,
    imgsz: int,
    batch_size: int,
    compare_to_pt: bool,
    *,
    min_images: int = 1,
):
    runtime = normalize_runtime_name(runtime)
    if min_images < 1:
        raise ValueError("min_images must be at least 1")
    model_path = classifier_model_path(runtime)
    samples = collect_samples(test_dir)
    if not samples:
        raise SystemExit(f"[ERROR] No images found in: {test_dir}")
    if len(samples) < min_images:
        raise SystemExit(
            f"[ERROR] Accuracy gate requires at least {min_images} images; found {len(samples)}"
        )

    model = load_model(model_path)
    started = time.perf_counter()
    rows = predict_all(
        model,
        samples,
        imgsz=imgsz,
        batch_size=batch_size,
        progress_label=f"{runtime} inference",
    )
    elapsed = time.perf_counter() - started

    overall, per_class, disagreements = summarize_rows(rows)
    if overall["images"] != len(samples):
        raise RuntimeError(
            f"Accuracy run is incomplete: evaluated {overall['images']} of {len(samples)} images"
        )
    baseline_summary = None
    drift = None

    if compare_to_pt and runtime != "pt":
        baseline_model = load_model(DEFAULT_CLASSIFIER_PT)
        baseline_rows = predict_all(
            baseline_model,
            samples,
            imgsz=imgsz,
            batch_size=batch_size,
            progress_label="pt baseline",
        )
        baseline_summary, drift = compare_with_baseline(rows, baseline_rows)

    report_dir = accuracy_report_dir(runtime)
    json_path = report_dir / "accuracy_report.json"
    txt_path = report_dir / "accuracy_report.txt"
    predictions_path = report_dir / "predictions.json"

    payload = {
        "status": "pass",
        "complete": True,
        "runtime": runtime,
        "model": str(model_path.resolve()),
        "test_dir": str(test_dir.resolve()),
        "elapsed_sec": elapsed,
        "overall": overall,
        "per_class": per_class,
        "baseline_agreement": baseline_summary,
        "errors": disagreements[:200],
        "drift": (drift or [])[:200],
    }
    write_json(json_path, payload)
    write_json(predictions_path, {"predictions": rows})
    write_text_report(
        txt_path,
        runtime=runtime,
        model_path=model_path,
        overall=overall,
        per_class=per_class,
        elapsed=elapsed,
        baseline_summary=baseline_summary,
        disagreements=disagreements,
        drift=drift,
    )
    upsert_csv(
        suite_csv_path("accuracy_results.csv"),
        key_fields=["runtime"],
        row={
            "runtime": runtime,
            "model": str(model_path.resolve()),
            "test_dir": str(test_dir.resolve()),
            "images": overall["images"],
            "correct": overall["correct"],
            "accuracy": f"{overall['accuracy']:.6f}",
            "elapsed_sec": f"{elapsed:.6f}",
            "img_per_sec": f"{(overall['images'] / elapsed) if elapsed > 0 else 0.0:.6f}",
            "pt_agreement": (
                f"{baseline_summary['agreement']:.6f}" if baseline_summary is not None else ""
            ),
            "pt_disagreements": (
                baseline_summary["disagreements"] if baseline_summary is not None else ""
            ),
            "report_json": str(json_path.resolve()),
            "report_txt": str(txt_path.resolve()),
            "predictions_json": str(predictions_path.resolve()),
        },
    )

    print(f"\n[{runtime}]")
    print(f"  Model      : {model_path}")
    print(f"  Test set   : {test_dir}")
    print(f"  Accuracy   : {overall['accuracy'] * 100:.2f}%")
    print(f"  Elapsed    : {elapsed:.2f}s")
    if elapsed > 0:
        print(f"  Throughput : {overall['images'] / elapsed:.2f} img/s")
    if baseline_summary is not None:
        print(f"  PT agree   : {baseline_summary['agreement'] * 100:.2f}%")
    print(f"  Report     : {txt_path}")
    return payload


def build_parser(runtime: str):
    parser = argparse.ArgumentParser(description=f"{runtime} classifier accuracy test")
    parser.add_argument("--test", default=str(DEFAULT_TEST_DIR))
    parser.add_argument("--imgsz", type=int, default=CLASSIFIER_IMGSZ)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--min-images", type=int, default=1)
    parser.add_argument("--no-compare", action="store_true")
    return parser


def run_runtime_cli(runtime: str):
    args = build_parser(runtime).parse_args()
    if args.min_images < 1:
        raise SystemExit("[ERROR] --min-images must be at least 1")
    result = run_accuracy(
        runtime=runtime,
        test_dir=Path(args.test),
        imgsz=args.imgsz,
        batch_size=max(1, args.batch),
        compare_to_pt=not args.no_compare,
        min_images=args.min_images,
    )
    return result
