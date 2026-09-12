#!/usr/bin/env python3
"""
No-inference Raspberry Pi preflight.

Use this when the Pi has undervoltage and powers off under model load. It does
not load YOLO/OpenVINO/PyTorch models and does not run inference. It validates
the software package, datasets, labels, scripts, and import availability, then
writes a report that is safe to generate on weak power.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "runtime_reports" / "preflight"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

SCRIPT_FILES = [
    "inference_rpi.py",
    "pi_plant_loop.py",
    "benchmark_rpi_pi.py",
    "runtime_benchmark_common.py",
    "runtime_suite_common.py",
    "runtime_export_common.py",
    "24_benchmark_runtime_openvino.py",
    "34_benchmark_tomato_crop_disease.py",
    "38_sweep_classifier_confidence_gate.py",
    "40_pi_end_to_end_validation.py",
    "41_pi_tomato_crop_disease_validation.py",
]


def parse_args():
    p = argparse.ArgumentParser(description="No-inference Pi package preflight")
    p.add_argument("--tomato-dataset", default="tomato_crop_disease")
    p.add_argument("--classifier-test", default="clf_dataset/test")
    p.add_argument("--classifier-model", default="models/classifier_deploy_fastcrop.pt")
    p.add_argument("--classifier-openvino", default="runtime_exports/classifier_openvino_model")
    p.add_argument("--detector-model", default="models/detector_nano_256.pt")
    p.add_argument("--strict", action="store_true", help="Exit nonzero if optional items are missing")
    return p.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(cmd: list[str], *, timeout_sec: float = 30.0) -> dict:
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "output": (proc.stdout or "")[-32768:].strip(),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "returncode": 124,
            "output": f"command timed out after {timeout_sec:g}s: {(exc.stdout or '')[-4096:]}",
        }
    except FileNotFoundError as exc:
        return {"cmd": cmd, "returncode": 127, "output": str(exc)}


def file_info(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    payload = {
        "exists": True,
        "path": str(path),
        "is_dir": path.is_dir(),
        "is_file": path.is_file(),
    }
    if path.is_file():
        payload["bytes"] = path.stat().st_size
        payload["sha256"] = sha256(path)
    elif path.is_dir():
        files = []
        for item in path.rglob("*"):
            if item.is_symlink() or not item.is_file():
                continue
            try:
                files.append(item)
            except OSError:
                continue
        payload["files"] = len(files)
        payload["bytes"] = sum(item.stat().st_size for item in files)
    return payload


def count_class_folder(root: Path) -> dict:
    if not root.is_dir():
        return {"exists": False, "path": str(root)}
    per_class = {}
    total = 0
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        count = sum(1 for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        per_class[class_dir.name] = count
        total += count
    return {
        "exists": True,
        "path": str(root),
        "classes": per_class,
        "images": total,
    }


def validate_yolo_dataset(root: Path) -> dict:
    images_dir = root / "images"
    labels_dir = root / "labels"
    payload = {
        "exists": root.is_dir(),
        "path": str(root),
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "image_count": 0,
        "label_count": 0,
        "paired_count": 0,
        "box_count": 0,
        "class_ids": {},
        "bad_labels": [],
        "missing_labels": [],
        "extra_labels": [],
    }
    if not images_dir.is_dir() or not labels_dir.is_dir():
        return payload

    image_paths = [p for p in sorted(images_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    label_paths = [p for p in sorted(labels_dir.iterdir()) if p.is_file() and p.suffix.lower() == ".txt"]
    payload["image_count"] = len(image_paths)
    payload["label_count"] = len(label_paths)

    class_ids = Counter()
    image_stems = {image_path.stem for image_path in image_paths}
    payload["extra_labels"] = sorted(
        path.stem for path in label_paths if path.stem not in image_stems
    )[:50]
    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            payload["missing_labels"].append(image_path.name)
            continue
        payload["paired_count"] += 1
        for line_no, line in enumerate(label_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                payload["bad_labels"].append({"file": label_path.name, "line": line_no, "text": line})
                continue
            try:
                class_id = int(parts[0])
                values = [float(value) for value in parts[1:5]]
                if class_id < 0 or any(not math.isfinite(value) for value in values):
                    raise ValueError("non-finite/negative label")
                if any(value < 0 or value > 1 for value in values):
                    raise ValueError("coordinate outside normalized range")
                if values[2] <= 0 or values[3] <= 0:
                    raise ValueError("box width/height must be positive")
                class_ids[class_id] += 1
                payload["box_count"] += 1
            except Exception as exc:
                payload["bad_labels"].append({"file": label_path.name, "line": line_no, "text": line, "error": str(exc)})

    payload["class_ids"] = dict(sorted(class_ids.items()))
    payload["bad_labels"] = payload["bad_labels"][:50]
    payload["missing_labels"] = payload["missing_labels"][:50]
    return payload


def main():
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / (
        f"preflight_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}.json"
    )

    power = {
        "throttled": run(["vcgencmd", "get_throttled"]),
        "temp": run(["vcgencmd", "measure_temp"]),
    }

    compile_cmd = [sys.executable, "-m", "py_compile", *SCRIPT_FILES]
    compile_result = run(compile_cmd)

    import_result = run(
        [
            sys.executable,
            "-c",
            "import cv2, numpy, yaml, tqdm, openvino, ultralytics; print('imports ok')",
        ]
    )

    tomato = validate_yolo_dataset(ROOT / args.tomato_dataset)
    classifier_test = count_class_folder(ROOT / args.classifier_test)

    files = {
        "classifier_model": file_info(ROOT / args.classifier_model),
        "classifier_openvino": file_info(ROOT / args.classifier_openvino),
        "detector_model": file_info(ROOT / args.detector_model),
        "requirements": file_info(ROOT / "requirements-runtime-pi.txt"),
    }

    checks = {
        "scripts_compile": compile_result["returncode"] == 0,
        "imports_available": import_result["returncode"] == 0,
        "tomato_dataset_100_pairs": tomato.get("paired_count") == 100,
        "tomato_labels_valid": (
            not tomato.get("bad_labels")
            and not tomato.get("missing_labels")
            and not tomato.get("extra_labels")
            and tomato.get("image_count") == tomato.get("label_count")
        ),
        "classifier_model_present": files["classifier_model"]["exists"],
        "classifier_openvino_present": files["classifier_openvino"]["exists"],
        "detector_model_present": files["detector_model"]["exists"],
    }

    required_ok = (
        checks["scripts_compile"]
        and checks["imports_available"]
        and checks["tomato_dataset_100_pairs"]
        and checks["tomato_labels_valid"]
        and checks["classifier_model_present"]
        and checks["classifier_openvino_present"]
        and checks["detector_model_present"]
    )

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "pass" if required_ok else "fail",
        "note": "No model loading or inference was performed.",
        "power": power,
        "checks": checks,
        "compile": compile_result,
        "imports": import_result,
        "tomato_crop_disease": tomato,
        "classifier_test": classifier_test,
        "files": files,
    }
    report_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    print("=" * 72)
    print("No-Inference Pi Preflight")
    print("=" * 72)
    print(f"Status              : {payload['status'].upper()}")
    print(f"Report              : {report_path}")
    print(f"Throttled           : {power['throttled']['output']}")
    print(f"Tomato pairs        : {tomato.get('paired_count')}/100")
    print(f"Tomato boxes        : {tomato.get('box_count')}")
    print(f"Tomato class ids    : {tomato.get('class_ids')}")
    print(f"Scripts compile     : {checks['scripts_compile']}")
    print(f"Imports available   : {checks['imports_available']}")
    print(f"Detector present    : {checks['detector_model_present']}")
    print(f"Classifier OV exists: {checks['classifier_openvino_present']}")
    print("=" * 72)

    if not required_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
