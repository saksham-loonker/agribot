"""
runtime_benchmark_common.py
===========================
Shared pipeline benchmark wrapper for ordered runtime tests.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

from benchmark_rpi_pi import run_benchmark
from runtime_suite_common import (
    DEFAULT_TEST_DIR,
    PIPELINE_DET_IMGSZ,
    PIPELINE_MAX_EDGE,
    PIPELINE_MAX_LEAVES,
    PIPELINE_THREADS,
    benchmark_report_dir,
    classifier_model_path,
    detector_model_path,
    normalize_runtime_name,
    require_finite_float,
    suite_csv_path,
    upsert_csv,
    write_json,
)


def build_parser(runtime: str):
    parser = argparse.ArgumentParser(description=f"{runtime} pipeline benchmark")
    parser.add_argument("--test", default=str(DEFAULT_TEST_DIR))
    parser.add_argument("--with-detector", action="store_true")
    parser.add_argument("--det-conf", type=float, default=0.30)
    parser.add_argument("--det-iou", type=float, default=0.45)
    parser.add_argument("--det-imgsz", type=int, default=PIPELINE_DET_IMGSZ)
    parser.add_argument("--max-edge", type=int, default=PIPELINE_MAX_EDGE)
    parser.add_argument("--max-leaves", type=int, default=PIPELINE_MAX_LEAVES)
    parser.add_argument("--crop-mode", choices=["none", "center", "mask"], default="none")
    parser.add_argument("--crop-scale", type=float, default=0.70)
    parser.add_argument("--crop-pad", type=float, default=0.12)
    parser.add_argument("--threads", type=int, default=PIPELINE_THREADS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.57)
    return parser


def run_runtime_benchmark(runtime: str):
    runtime = normalize_runtime_name(runtime)
    args = build_parser(runtime).parse_args()
    if args.limit < 0 or args.max_leaves < 0 or args.threads < 1:
        raise SystemExit("[ERROR] limit/max-leaves must be non-negative and threads must be positive")
    for name in ("det_conf", "det_iou", "crop_scale", "crop_pad", "conf"):
        try:
            require_finite_float(getattr(args, name), name)
        except ValueError as exc:
            raise SystemExit(f"[ERROR] {exc}") from exc
    for name in ("det_conf", "det_iou", "conf"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"[ERROR] {name} must be between 0 and 1")

    report_dir = benchmark_report_dir(runtime)
    namespace = Namespace(
        test=args.test,
        det_model=str(detector_model_path(runtime)),
        clf_model=str(classifier_model_path(runtime)),
        det_imgsz=args.det_imgsz,
        det_conf=args.det_conf,
        det_iou=args.det_iou,
        max_edge=args.max_edge,
        max_leaves=args.max_leaves,
        crop_mode=args.crop_mode,
        crop_scale=args.crop_scale,
        crop_pad=args.crop_pad,
        conf=args.conf,
        threads=args.threads,
        limit=args.limit,
        with_detector=args.with_detector,
    )

    metrics = run_benchmark(namespace)
    if not metrics.get("complete", False):
        raise RuntimeError("Benchmark did not cover the complete requested sample set")
    payload = {
        "runtime": runtime,
        "detector_model": namespace.det_model,
        "classifier_model": namespace.clf_model,
        "args": vars(namespace),
        "metrics": metrics,
    }
    out_path = report_dir / "benchmark_report.json"
    write_json(out_path, payload)
    upsert_csv(
        suite_csv_path("benchmark_results.csv"),
        key_fields=["runtime", "with_detector"],
        row={
            "runtime": runtime,
            "with_detector": namespace.with_detector,
            "test_dir": namespace.test,
            "classifier_model": namespace.clf_model,
            "detector_model": namespace.det_model,
            "det_imgsz": namespace.det_imgsz,
            "det_conf": namespace.det_conf,
            "det_iou": namespace.det_iou,
            "max_edge": namespace.max_edge,
            "max_leaves": namespace.max_leaves,
            "crop_mode": namespace.crop_mode,
            "crop_scale": namespace.crop_scale,
            "crop_pad": namespace.crop_pad,
            "threads": namespace.threads,
            "limit": namespace.limit,
            "conf": namespace.conf,
            "images_tested": metrics["images_tested"],
            "images_requested": metrics.get("images_requested", metrics["images_tested"]),
            "complete": metrics["complete"],
            "overall_accuracy": f"{metrics['overall_accuracy']:.6f}",
            "avg_fps": f"{metrics['avg_fps']:.6f}",
            "avg_latency_ms": f"{metrics['avg_latency_ms']:.6f}",
            "max_latency_ms": f"{metrics['max_latency_ms']:.6f}",
            "over_preferred_latency_frames": metrics["over_preferred_latency_frames"],
            "over_hard_latency_frames": metrics["over_hard_latency_frames"],
            "meets_min_fps": metrics["meets_min_fps"],
            "meets_latency_ceiling": metrics["meets_latency_ceiling"],
            "total_time_sec": f"{metrics['total_time_sec']:.6f}",
            "report_json": str(out_path.resolve()),
        },
    )
    print(f"Saved benchmark report -> {out_path}")
