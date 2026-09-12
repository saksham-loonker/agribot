#!/usr/bin/env python3
"""
Run the full power-safe Raspberry Pi software validation.

This is intentionally conservative for today's undervoltage-limited Pi:
classifier deployment is validated end-to-end with threads=1, while the normal
deployment default remains threads=4 for after the power supply is fixed.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR / "runtime_reports" / "end_to_end"


def parse_args():
    p = argparse.ArgumentParser(description="Power-safe Pi end-to-end validation")
    p.add_argument("--threads", type=int, default=1, help="Threads for today's validation")
    p.add_argument("--normal-threads", type=int, default=4, help="Normal deployment thread target after power fix")
    p.add_argument("--classifier-limit", type=int, default=0, help="0 means full classifier test split")
    p.add_argument("--test", default="clf_dataset/test")
    p.add_argument("--clf-model", default="runtime_exports/classifier_openvino_model")
    p.add_argument("--classifier-pt", default="models/classifier_deploy_fastcrop.pt")
    p.add_argument("--conf", type=float, default=0.765)
    p.add_argument("--high-conf", type=float, default=0.765)
    p.add_argument("--max-edge", type=int, default=256)
    p.add_argument("--crop-mode", choices=["mask", "center", "none"], default="mask")
    p.add_argument("--crop-scale", type=float, default=0.55)
    p.add_argument("--crop-pad", type=float, default=0.05)
    p.add_argument("--gate-threshold-step", type=float, default=0.005)
    p.add_argument("--gate-min-coverage", type=float, default=0.70)
    p.add_argument("--skip-tomato-smoke", action="store_true")
    p.add_argument("--tomato-det-model", default="models/detector_nano_256.pt")
    p.add_argument("--tomato-limit", type=int, default=10)
    p.add_argument("--log", default="")
    return p.parse_args()


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def command_string(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_capture(cmd: list[str]) -> dict:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        output = proc.stdout or ""
        return {
            "cmd": command_string(cmd),
            "returncode": proc.returncode,
            "elapsed_sec": time.perf_counter() - started,
            "output": output[-32768:].strip(),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": command_string(cmd),
            "returncode": 124,
            "elapsed_sec": time.perf_counter() - started,
            "output": f"command timed out: {(exc.stdout or '')[-4096:]}",
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


def require_path(path: str, *, directory: bool | None = None):
    p = SCRIPT_DIR / path
    if directory is True and not p.is_dir():
        raise SystemExit(f"[ERROR] Required directory missing: {path}")
    if directory is False and not p.is_file():
        raise SystemExit(f"[ERROR] Required file missing: {path}")
    if directory is None and not p.exists():
        raise SystemExit(f"[ERROR] Required path missing: {path}")


def ensure_openvino_classifier(args, log_handle) -> dict | None:
    if (SCRIPT_DIR / args.clf_model).is_dir():
        print(f"OpenVINO classifier already exists: {args.clf_model}")
        return None
    require_path(args.classifier_pt, directory=False)
    return run_streaming(
        "export_openvino_classifier",
        [
            sys.executable,
            "37_train_deployment_fastcrop_classifier.py",
            "--skip-prepare",
            "--skip-train",
            "--skip-eval",
            "--base-model",
            args.classifier_pt,
            "--copy-best-to",
            args.classifier_pt,
            "--export-openvino",
            "--openvino-out",
            args.clf_model,
            "--imgsz",
            "224",
        ],
        log_handle,
    )


def main():
    args = parse_args()
    if args.threads < 1 or args.normal_threads < 1 or args.max_edge <= 0:
        raise SystemExit("[ERROR] thread counts and max-edge must be positive")
    for name in ("conf", "high_conf", "gate_min_coverage", "gate_threshold_step", "crop_scale", "crop_pad"):
        value = float(getattr(args, name))
        if not math.isfinite(value):
            raise SystemExit(f"[ERROR] {name} must be finite")
    for name in ("conf", "high_conf", "gate_min_coverage"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"[ERROR] {name} must be between 0 and 1")
    if args.gate_threshold_step <= 0:
        raise SystemExit("[ERROR] gate-threshold-step must be positive")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log) if args.log else REPORT_DIR / f"end_to_end_{stamp}.log"
    report_path = REPORT_DIR / f"end_to_end_{stamp}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    require_path(args.test, directory=True)
    require_path(args.classifier_pt, directory=False)

    steps = []
    failures = []
    start_throttle = run_capture(["vcgencmd", "get_throttled"])
    start_temp = run_capture(["vcgencmd", "measure_temp"])

    with log_path.open("w", encoding="utf-8") as log_handle:
        print(f"End-to-end validation started: {now()}", file=log_handle)
        print(f"Working dir: {SCRIPT_DIR}", file=log_handle)
        print(f"Validation threads: {args.threads}", file=log_handle)
        print(f"Normal deployment threads after power fix: {args.normal_threads}", file=log_handle)

        print("Power at start:")
        print(start_throttle["output"])
        print(start_temp["output"])

        try:
            steps.append(
                run_streaming(
                    "py_compile",
                    [
                        sys.executable,
                        "-m",
                        "py_compile",
                        "inference_rpi.py",
                        "pi_plant_loop.py",
                        "benchmark_rpi_pi.py",
                        "runtime_benchmark_common.py",
                        "runtime_suite_common.py",
                        "24_benchmark_runtime_openvino.py",
                        "34_benchmark_tomato_crop_disease.py",
                        "38_sweep_classifier_confidence_gate.py",
                    ],
                    log_handle,
                )
            )

            export_step = ensure_openvino_classifier(args, log_handle)
            if export_step is not None:
                steps.append(export_step)

            steps.append(
                run_streaming(
                    "full_classifier_benchmark",
                    [
                        sys.executable,
                        "24_benchmark_runtime_openvino.py",
                        "--test",
                        args.test,
                        "--limit",
                        str(args.classifier_limit),
                        "--threads",
                        str(args.threads),
                        "--max-edge",
                        str(args.max_edge),
                        "--conf",
                        str(args.conf),
                        "--crop-mode",
                        args.crop_mode,
                        "--crop-scale",
                        str(args.crop_scale),
                        "--crop-pad",
                        str(args.crop_pad),
                    ],
                    log_handle,
                )
            )

            steps.append(
                run_streaming(
                    "confidence_gate_sweep",
                    [
                        sys.executable,
                        "38_sweep_classifier_confidence_gate.py",
                        "--test",
                        args.test,
                        "--clf-model",
                        args.clf_model,
                        "--threads",
                        str(args.threads),
                        "--max-edge",
                        str(args.max_edge),
                        "--crop-mode",
                        args.crop_mode,
                        "--crop-scale",
                        str(args.crop_scale),
                        "--crop-pad",
                        str(args.crop_pad),
                        "--threshold-step",
                        str(args.gate_threshold_step),
                        "--min-coverage",
                        str(args.gate_min_coverage),
                    ],
                    log_handle,
                )
            )

            if not args.skip_tomato_smoke:
                if (SCRIPT_DIR / args.tomato_det_model).exists() and (SCRIPT_DIR / "tomato_crop_disease").is_dir():
                    steps.append(
                        run_streaming(
                            "tomato_image_hit_smoke",
                            [
                                sys.executable,
                                "34_benchmark_tomato_crop_disease.py",
                                "--det-model",
                                args.tomato_det_model,
                                "--imgsz",
                                "256",
                                "--max-det",
                                "20",
                                "--threads",
                                str(args.threads),
                                "--limit",
                                str(args.tomato_limit),
                                "--target-metric",
                                "image_hit_rate",
                                "--target-value",
                                "0.90",
                            ],
                            log_handle,
                            required=False,
                        )
                    )
                else:
                    print("[tomato_image_hit_smoke] skipped: detector or dataset missing")

        except Exception as exc:
            failures.append(str(exc))
            print(f"\n[ERROR] {exc}", file=log_handle, flush=True)
            print(f"\n[ERROR] {exc}", flush=True)

    end_throttle = run_capture(["vcgencmd", "get_throttled"])
    end_temp = run_capture(["vcgencmd", "measure_temp"])

    report = {
        "started_at": stamp,
        "finished_at": now(),
        "status": "pass" if not failures else "fail",
        "threads_used": args.threads,
        "normal_threads_after_power_fix": args.normal_threads,
        "power": {
            "start_throttled": start_throttle["output"],
            "start_temp": start_temp["output"],
            "end_throttled": end_throttle["output"],
            "end_temp": end_temp["output"],
            "note": "Any 0x50000 value indicates undervoltage history.",
        },
        "settings": vars(args),
        "steps": steps,
        "failures": failures,
        "log": str(log_path.resolve()),
        "key_outputs": {
            "benchmark_report": str((SCRIPT_DIR / "runtime_reports" / "benchmark" / "openvino" / "benchmark_report.json").resolve()),
            "confidence_gate_best": str((SCRIPT_DIR / "dataset" / "runtime_tracking" / "classifier_confidence_gate_best.json").resolve()),
            "tomato_best": str((SCRIPT_DIR / "dataset" / "runtime_tracking" / "tomato_crop_disease_detector_best.json").resolve()),
        },
    }
    benchmark_path = SCRIPT_DIR / "runtime_reports" / "benchmark" / "openvino" / "benchmark_report.json"
    gate_path = SCRIPT_DIR / "dataset" / "runtime_tracking" / "classifier_confidence_gate_best.json"
    for label, path in (("benchmark", benchmark_path), ("confidence_gate", gate_path)):
        if not path.exists():
            failures.append(f"Missing {label} output: {path}")
            continue
        try:
            output = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"Invalid {label} output: {exc}")
            continue
        if label == "benchmark":
            metrics = output.get("metrics", {})
            if not metrics.get("complete", False) or not metrics.get("meets_min_fps", False) or not metrics.get("meets_latency_ceiling", False):
                failures.append("Benchmark output did not satisfy completeness, FPS, and latency gates")
        elif output.get("status") != "pass" or not output.get("complete", False):
            failures.append("Confidence gate output did not pass a complete deployment gate")
    report["status"] = "pass" if not failures else "fail"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"End-to-end status : {report['status'].upper()}")
    print(f"Report            : {report_path}")
    print(f"Log               : {log_path}")
    print(f"Start throttled   : {start_throttle['output']}")
    print(f"End throttled     : {end_throttle['output']}")
    print("=" * 72)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
