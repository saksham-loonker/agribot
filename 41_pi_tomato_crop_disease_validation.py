#!/usr/bin/env python3
"""
Power-safe end-to-end validation for the tomato_crop_disease benchmark.

This validates the YOLO detection benchmark dataset directly. It does not run
the 8-class classifier benchmark. Normal deployment threading is recorded as 4,
while today's undervoltage-limited validation defaults to 1 thread.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR / "runtime_reports" / "tomato_crop_disease_end_to_end"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(description="Validate tomato_crop_disease end-to-end")
    p.add_argument("--dataset", default="tomato_crop_disease")
    p.add_argument("--det-model", default="models/detector_nano_256.pt")
    p.add_argument("--threads", type=int, default=1, help="Today's safe validation thread count")
    p.add_argument("--normal-threads", type=int, default=4, help="Normal target after power fix")
    p.add_argument("--imgsz", type=int, default=256)
    p.add_argument("--max-det", type=int, default=20)
    p.add_argument("--limit", type=int, default=0, help="0 means full tomato_crop_disease dataset")
    p.add_argument("--target-metric", choices=["image_hit_rate", "f1"], default="image_hit_rate")
    p.add_argument("--target-value", type=float, default=0.90)
    p.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50")
    p.add_argument("--log", default="")
    return p.parse_args()


def command_string(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_capture(cmd: list[str]) -> dict:
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = proc.stdout or ""
        return {
            "cmd": command_string(cmd),
            "returncode": proc.returncode,
            "elapsed_sec": time.perf_counter() - started,
            "output": output.strip(),
        }
    except FileNotFoundError as exc:
        return {
            "cmd": command_string(cmd),
            "returncode": 127,
            "elapsed_sec": time.perf_counter() - started,
            "output": str(exc),
        }


def run_streaming(name: str, cmd: list[str], log_handle, *, required: bool = True) -> dict:
    print(f"\n[{name}] {command_string(cmd)}", flush=True)
    print(f"\n[{name}] {command_string(cmd)}", file=log_handle, flush=True)

    started = time.perf_counter()
    tail = deque(maxlen=240)
    proc = subprocess.Popen(
        cmd,
        cwd=SCRIPT_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        print(line, end="", file=log_handle, flush=True)
        tail.append(line.rstrip("\n"))
    returncode = proc.wait()
    elapsed = time.perf_counter() - started

    result = {
        "name": name,
        "cmd": command_string(cmd),
        "returncode": returncode,
        "elapsed_sec": elapsed,
        "tail": list(tail),
    }
    if required and returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {returncode}")
    return result


def collect_pairs(dataset: Path) -> list[tuple[Path, Path]]:
    images = dataset / "images"
    labels = dataset / "labels"
    if not images.is_dir():
        raise SystemExit(f"[ERROR] Missing images dir: {images}")
    if not labels.is_dir():
        raise SystemExit(f"[ERROR] Missing labels dir: {labels}")

    pairs = []
    for image_path in sorted(images.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = labels / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))
    if not pairs:
        raise SystemExit(f"[ERROR] No image/label pairs found in: {dataset}")
    return pairs


def main():
    args = parse_args()
    dataset = SCRIPT_DIR / args.dataset
    det_model = SCRIPT_DIR / args.det_model
    if not det_model.exists():
        raise SystemExit(f"[ERROR] Detector not found: {args.det_model}")

    pairs = collect_pairs(dataset)
    expected = len(pairs) if args.limit <= 0 else min(args.limit, len(pairs))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log) if args.log else REPORT_DIR / f"tomato_end_to_end_{stamp}.log"
    report_path = REPORT_DIR / f"tomato_end_to_end_{stamp}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start_throttle = run_capture(["vcgencmd", "get_throttled"])
    start_temp = run_capture(["vcgencmd", "measure_temp"])
    steps = []
    failures = []

    print("tomato_crop_disease end-to-end validation")
    print(f"  dataset        : {dataset}")
    print(f"  image/label set: {len(pairs)} pairs")
    print(f"  detector       : {args.det_model}")
    print(f"  threads today  : {args.threads}")
    print(f"  normal threads : {args.normal_threads}")
    print(f"  expected run   : {expected} images")
    print(f"  start power    : {start_throttle['output']}")

    with log_path.open("w", encoding="utf-8") as log_handle:
        print(f"tomato_crop_disease validation started {stamp}", file=log_handle)
        print(f"dataset pairs: {len(pairs)}", file=log_handle)
        try:
            steps.append(
                run_streaming(
                    "py_compile",
                    [
                        sys.executable,
                        "-m",
                        "py_compile",
                        "inference_rpi.py",
                        "34_benchmark_tomato_crop_disease.py",
                    ],
                    log_handle,
                )
            )

            steps.append(
                run_streaming(
                    "tomato_crop_disease_benchmark",
                    [
                        sys.executable,
                        "34_benchmark_tomato_crop_disease.py",
                        "--dataset",
                        args.dataset,
                        "--det-model",
                        args.det_model,
                        "--imgsz",
                        str(args.imgsz),
                        "--max-det",
                        str(args.max_det),
                        "--threads",
                        str(args.threads),
                        "--limit",
                        str(args.limit),
                        "--thresholds",
                        args.thresholds,
                        "--target-metric",
                        args.target_metric,
                        "--target-value",
                        str(args.target_value),
                    ],
                    log_handle,
                )
            )
        except Exception as exc:
            failures.append(str(exc))
            print(f"\n[ERROR] {exc}", file=log_handle, flush=True)
            print(f"\n[ERROR] {exc}", flush=True)

    end_throttle = run_capture(["vcgencmd", "get_throttled"])
    end_temp = run_capture(["vcgencmd", "measure_temp"])

    best_path = SCRIPT_DIR / "dataset" / "runtime_tracking" / "tomato_crop_disease_detector_best.json"
    best_payload = None
    if best_path.exists():
        try:
            best_payload = json.loads(best_path.read_text(encoding="utf-8"))
        except Exception:
            best_payload = None

    report = {
        "started_at": stamp,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "status": "pass" if not failures else "fail",
        "dataset": str(dataset.resolve()),
        "image_label_pairs": len(pairs),
        "expected_images_this_run": expected,
        "detector": str(det_model.resolve()),
        "threads_used_today": args.threads,
        "normal_threads_after_power_fix": args.normal_threads,
        "settings": vars(args),
        "power": {
            "start_throttled": start_throttle["output"],
            "start_temp": start_temp["output"],
            "end_throttled": end_throttle["output"],
            "end_temp": end_temp["output"],
            "note": "0x50000 means undervoltage history.",
        },
        "steps": steps,
        "failures": failures,
        "benchmark_best_json": str(best_path.resolve()),
        "benchmark_best": best_payload,
        "log": str(log_path.resolve()),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"Tomato validation : {report['status'].upper()}")
    print(f"Images checked    : {expected}")
    print(f"Report            : {report_path}")
    print(f"Log               : {log_path}")
    if best_payload:
        best = best_payload.get("best", {})
        metric = best_payload.get("target_metric")
        print(f"Best {metric:<14}: {best.get(metric, 0.0) * 100:.2f}%")
        print(f"Strict F1         : {best.get('f1', 0.0) * 100:.2f}%")
        print(f"Image hit         : {best.get('image_hit_rate', 0.0) * 100:.2f}%")
    print(f"Start throttled   : {start_throttle['output']}")
    print(f"End throttled     : {end_throttle['output']}")
    print("=" * 72)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
