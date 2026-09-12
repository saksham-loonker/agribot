"""
OpenVINO detector-first deployment sweep.

This script maximizes the deployed pipeline, not classifier-only accuracy:
detector crop -> classifier -> confidence gate. It tests a small grid of
settings and records both the most accurate result and the best result that
still satisfies the Pi timing target.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

from benchmark_rpi_pi import run_benchmark
from runtime_suite_common import (
    DEFAULT_TEST_DIR,
    MAX_SELECTED_FRAME_LATENCY_MS,
    MIN_PIPELINE_FPS,
    PREFERRED_SELECTED_FRAME_LATENCY_MS,
    classifier_model_path,
    detector_model_path,
    suite_csv_path,
    upsert_csv,
    write_json,
)


def parse_csv_numbers(value: str, cast):
    items = []
    for part in value.split(","):
        part = part.strip()
        if part:
            items.append(cast(part))
    if not items:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return items


def parse_args():
    p = argparse.ArgumentParser(description="Optimize OpenVINO detector-first deployment settings")
    p.add_argument("--test", default=str(DEFAULT_TEST_DIR))
    p.add_argument("--clf-model", default=str(classifier_model_path("openvino")))
    p.add_argument("--det-model", default=str(detector_model_path("openvino")))
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-leaves", type=int, default=1)
    p.add_argument("--max-edges", default="256,320,384")
    p.add_argument(
        "--det-imgszs",
        default="256",
        help="Must match the fixed detector OpenVINO export size unless the detector was re-exported.",
    )
    p.add_argument("--clf-confs", default="0.00,0.35,0.57")
    p.add_argument("--det-confs", default="0.15,0.30")
    p.add_argument("--det-ious", default="0.45")
    p.add_argument("--same-size-only", action="store_true", default=False)
    p.add_argument("--full-grid", action="store_false", dest="same_size_only")
    return p.parse_args()


def candidate_pairs(max_edges: list[int], det_imgszs: list[int], same_size_only: bool):
    if not same_size_only:
        for max_edge in max_edges:
            for det_imgsz in det_imgszs:
                yield max_edge, det_imgsz
        return

    det_sizes = set(det_imgszs)
    for max_edge in max_edges:
        if max_edge in det_sizes:
            yield max_edge, max_edge


def deployment_pass(metrics: dict) -> bool:
    return (
        metrics["avg_fps"] >= MIN_PIPELINE_FPS
        and metrics["avg_latency_ms"] <= MAX_SELECTED_FRAME_LATENCY_MS
        and metrics["max_latency_ms"] <= MAX_SELECTED_FRAME_LATENCY_MS
        and metrics["over_hard_latency_frames"] == 0
    )


def score_for_accuracy(row: dict):
    return (
        row["accuracy"],
        row["avg_fps"],
        -row["avg_latency_ms"],
    )


def main():
    args = parse_args()
    max_edges = parse_csv_numbers(args.max_edges, int)
    det_imgszs = parse_csv_numbers(args.det_imgszs, int)
    clf_confs = parse_csv_numbers(args.clf_confs, float)
    det_confs = parse_csv_numbers(args.det_confs, float)
    det_ious = parse_csv_numbers(args.det_ious, float)

    csv_path = suite_csv_path("openvino_accuracy_sweep.csv")
    json_path = suite_csv_path("openvino_accuracy_sweep_best.json")
    rows = []

    print("OpenVINO detector-first sweep")
    print(f"  test       : {args.test}")
    print(f"  limit      : {args.limit or 'all'}")
    print(f"  classifier : {args.clf_model}")
    print(f"  detector   : {args.det_model}")
    print()

    run_index = 0
    for max_edge, det_imgsz in candidate_pairs(max_edges, det_imgszs, args.same_size_only):
        for clf_conf in clf_confs:
            for det_conf in det_confs:
                for det_iou in det_ious:
                    run_index += 1
                    print(
                        f"[{run_index}] max_edge={max_edge} det_imgsz={det_imgsz} "
                        f"clf_conf={clf_conf:.2f} det_conf={det_conf:.2f} det_iou={det_iou:.2f}"
                    )
                    namespace = Namespace(
                        test=args.test,
                        det_model=args.det_model,
                        clf_model=args.clf_model,
                        det_imgsz=det_imgsz,
                        det_conf=det_conf,
                        det_iou=det_iou,
                        max_edge=max_edge,
                        max_leaves=args.max_leaves,
                        conf=clf_conf,
                        threads=args.threads,
                        limit=args.limit,
                        sequential_limit=False,
                        with_detector=True,
                    )
                    try:
                        metrics = run_benchmark(namespace)
                    except RuntimeError as exc:
                        row = {
                            "runtime": "openvino",
                            "max_edge": max_edge,
                            "det_imgsz": det_imgsz,
                            "clf_conf": clf_conf,
                            "det_conf": det_conf,
                            "det_iou": det_iou,
                            "max_leaves": args.max_leaves,
                            "limit": args.limit,
                            "images": 0,
                            "accuracy": "",
                            "avg_fps": "",
                            "avg_latency_ms": "",
                            "max_latency_ms": "",
                            "over_180ms_frames": "",
                            "over_200ms_frames": "",
                            "deployable": False,
                            "error": str(exc).splitlines()[0],
                        }
                        upsert_csv(
                            csv_path,
                            key_fields=["runtime", "max_edge", "det_imgsz", "clf_conf", "det_conf", "det_iou", "limit"],
                            row=row,
                        )
                        print(f"  skipped: {row['error']}")
                        continue
                    row = {
                        "runtime": "openvino",
                        "max_edge": max_edge,
                        "det_imgsz": det_imgsz,
                        "clf_conf": clf_conf,
                        "det_conf": det_conf,
                        "det_iou": det_iou,
                        "max_leaves": args.max_leaves,
                        "limit": args.limit,
                        "images": metrics["images_tested"],
                        "accuracy": metrics["overall_accuracy"],
                        "avg_fps": metrics["avg_fps"],
                        "avg_latency_ms": metrics["avg_latency_ms"],
                        "max_latency_ms": metrics["max_latency_ms"],
                        "over_180ms_frames": metrics["over_preferred_latency_frames"],
                        "over_200ms_frames": metrics["over_hard_latency_frames"],
                        "deployable": deployment_pass(metrics),
                        "error": "",
                    }
                    rows.append(row)
                    upsert_csv(
                        csv_path,
                        key_fields=["runtime", "max_edge", "det_imgsz", "clf_conf", "det_conf", "det_iou", "limit"],
                        row=row,
                    )

    best_accuracy = max(rows, key=score_for_accuracy) if rows else None
    deployable_rows = [row for row in rows if row["deployable"]]
    best_deployable = max(deployable_rows, key=score_for_accuracy) if deployable_rows else None
    best_speed = max(rows, key=lambda row: row["avg_fps"]) if rows else None

    payload = {
        "targets": {
            "min_fps": MIN_PIPELINE_FPS,
            "preferred_latency_ms": PREFERRED_SELECTED_FRAME_LATENCY_MS,
            "max_latency_ms": MAX_SELECTED_FRAME_LATENCY_MS,
        },
        "best_accuracy": best_accuracy,
        "best_deployable": best_deployable,
        "best_speed": best_speed,
        "csv": str(csv_path.resolve()),
    }
    write_json(json_path, payload)

    print(f"\nSweep CSV  -> {csv_path}")
    print(f"Best JSON  -> {json_path}")
    if best_accuracy:
        print(
            "Best accuracy -> "
            f"{best_accuracy['accuracy'] * 100:.2f}% at "
            f"max_edge={best_accuracy['max_edge']} det_imgsz={best_accuracy['det_imgsz']} "
            f"clf_conf={best_accuracy['clf_conf']:.2f} det_conf={best_accuracy['det_conf']:.2f}; "
            f"{best_accuracy['avg_fps']:.2f} FPS"
        )
    if best_deployable:
        print(
            "Best deployable -> "
            f"{best_deployable['accuracy'] * 100:.2f}% at "
            f"max_edge={best_deployable['max_edge']} det_imgsz={best_deployable['det_imgsz']} "
            f"clf_conf={best_deployable['clf_conf']:.2f} det_conf={best_deployable['det_conf']:.2f}; "
            f"{best_deployable['avg_fps']:.2f} FPS"
        )
    else:
        print("Best deployable -> none in this sweep met the 5 FPS / 200 ms ceiling")


if __name__ == "__main__":
    main()
