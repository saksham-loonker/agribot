"""
46_prepare_android_model_bundle.py
==================================
Checks and optionally prepares the Android TFLite model bundle expected by
plan.md. This script never invents model results: it reports missing inputs,
exports only when explicitly requested, and writes hashes only from files that
exist on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from runtime_export_common import export_variant


ROOT = Path(__file__).parent.resolve()
ANDROID_ASSETS = ROOT / "agribot_android_app" / "android" / "app" / "src" / "main" / "assets"
ANDROID_EXPORTS = ROOT / "android_model_exports"

EXPECTED_LABELS = [
    "Early_blight",
    "Healthy",
    "Late_blight",
    "Leaf Miner",
    "Magnesium Deficiency",
    "Nitrogen Deficiency",
    "Pottassium Deficiency",
    "Spotted Wilt Virus",
]

CLASSIFIER_SOURCE = ROOT / "models" / "classifier_deploy_fastcrop.pt"
DETECTOR_SOURCE = ROOT / "models" / "detector_nano_256.pt"
CLASSIFIER_CALIBRATION_DATA = ROOT / "clf_dataset_deploy_fastcrop"
DETECTOR_DATASET_ROOT = ROOT / "dataset"
DETECTOR_CALIBRATION_DATA = ANDROID_EXPORTS / "detector_calibration.yaml"

CLASSIFIER_EXPORTS = {
    "float32": ANDROID_EXPORTS / "classifier" / "classifier_fastcrop_float32.tflite",
    "float16": ANDROID_EXPORTS / "classifier" / "classifier_fastcrop_float16.tflite",
    "int8": ANDROID_EXPORTS / "classifier" / "classifier_fastcrop_int8.tflite",
}
DETECTOR_EXPORTS = {
    "float32": ANDROID_EXPORTS / "detector" / "detector_nano_256_float32.tflite",
    "float16": ANDROID_EXPORTS / "detector" / "detector_nano_256_float16.tflite",
    "int8": ANDROID_EXPORTS / "detector" / "detector_nano_256_int8.tflite",
    "nms_float32": ANDROID_EXPORTS / "detector" / "detector_nano_256_nms_float32.tflite",
    "raw_float32": ANDROID_EXPORTS / "detector" / "detector_nano_256_raw_float32.tflite",
}

APP_CLASSIFIER = ANDROID_ASSETS / "models" / "classifier_fastcrop_float32.tflite"
APP_DETECTOR = ANDROID_ASSETS / "models" / "detector_nano_256_raw_float32.tflite"
APP_LABELS = ANDROID_ASSETS / "labels" / "labels.json"
APP_MANIFEST = ANDROID_ASSETS / "model_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare/check Android model bundle readiness")
    parser.add_argument("--export", action="store_true", help="Run Ultralytics TFLite export for missing artifacts")
    parser.add_argument("--copy-to-app-assets", action="store_true", help="Copy selected runnable TFLite exports into Android assets")
    parser.add_argument("--write-manifest", action="store_true", help="Write app model_manifest.json with current hashes")
    parser.add_argument("--write-bundle-manifest", action="store_true", help="Write android_model_exports/bundle/manifest.json")
    parser.add_argument("--json", action="store_true", help="Print the readiness report as JSON only")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def inspect_pt_labels(path: Path, task: str) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path.relative_to(ROOT)), "status": "missing", "labels": []}
    try:
        from ultralytics import YOLO

        model = YOLO(str(path), task=task)
        names = model.names
        if isinstance(names, dict):
            labels = [names[index] for index in sorted(names)]
        else:
            labels = list(names)
        return {
            "path": str(path.relative_to(ROOT)),
            "status": "ok" if labels == EXPECTED_LABELS else "label_mismatch",
            "labels": labels,
        }
    except Exception as exc:
        return {
            "path": str(path.relative_to(ROOT)),
            "status": "unverified",
            "error": str(exc),
            "labels": [],
        }


def maybe_export_models(enabled: bool, failures: list[str]) -> None:
    if not enabled:
        return
    if not CLASSIFIER_SOURCE.exists():
        failures.append(f"Cannot export: missing {CLASSIFIER_SOURCE.relative_to(ROOT)}")
        return
    if not DETECTOR_SOURCE.exists():
        failures.append(f"Cannot export: missing {DETECTOR_SOURCE.relative_to(ROOT)}")
        return
    write_detector_calibration_yaml()

    export_tflite_variant(CLASSIFIER_SOURCE, "classify", 224, CLASSIFIER_EXPORTS["float32"])
    export_tflite_variant(CLASSIFIER_SOURCE, "classify", 224, CLASSIFIER_EXPORTS["float16"], half=True)
    export_tflite_int8_variant(
        source=CLASSIFIER_SOURCE,
        task="classify",
        imgsz=224,
        destination=CLASSIFIER_EXPORTS["int8"],
        data=str(CLASSIFIER_CALIBRATION_DATA),
        generated_name="classifier_deploy_fastcrop_integer_quant.tflite",
        failures=failures,
    )
    export_tflite_variant(DETECTOR_SOURCE, "detect", 256, DETECTOR_EXPORTS["float32"])
    export_tflite_variant(DETECTOR_SOURCE, "detect", 256, DETECTOR_EXPORTS["float16"], half=True)
    export_tflite_variant(DETECTOR_SOURCE, "detect", 256, DETECTOR_EXPORTS["nms_float32"], nms=True)
    export_tflite_int8_variant(
        source=DETECTOR_SOURCE,
        task="detect",
        imgsz=256,
        destination=DETECTOR_EXPORTS["int8"],
        data=str(DETECTOR_CALIBRATION_DATA),
        generated_name="detector_nano_256_integer_quant.tflite",
        failures=failures,
    )


def export_tflite_int8_variant(
    source: Path,
    task: str,
    imgsz: int,
    destination: Path,
    data: str,
    generated_name: str,
    failures: list[str],
) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    generated = source.with_name(f"{source.stem}_saved_model") / generated_name
    if generated.exists() and generated.stat().st_size > 0:
        ensure_parent(destination)
        shutil.copy2(generated, destination)
        return
    export_error: Exception | None = None
    try:
        export_variant(source, task, "tflite", imgsz, destination, int8=True, data=data)
    except Exception as exc:
        export_error = exc
    recovered = recover_quantized_export(
        source,
        generated_name,
        destination,
        failures,
    )
    if export_error is not None and not recovered:
        failures.append(str(export_error))


def export_tflite_variant(
    source: Path,
    task: str,
    imgsz: int,
    destination: Path,
    *,
    half: bool = False,
    nms: bool = False,
) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    export_variant(source, task, "tflite", imgsz, destination, half=half, nms=nms)


def write_detector_calibration_yaml() -> None:
    ensure_parent(DETECTOR_CALIBRATION_DATA)
    DETECTOR_CALIBRATION_DATA.write_text(
        "\n".join(
            [
                f"path: {DETECTOR_DATASET_ROOT.as_posix()}",
                "train: train/images",
                "val: val/images",
                "test: test/images",
                "nc: 1",
                "names: ['crop']",
                "",
            ]
        ),
        encoding="utf-8",
    )


def recover_quantized_export(source: Path, generated_name: str, destination: Path, failures: list[str]) -> bool:
    if destination.exists() and destination.stat().st_size > 0:
        return True
    generated = source.with_name(f"{source.stem}_saved_model") / generated_name
    if generated.exists() and generated.stat().st_size > 0:
        ensure_parent(destination)
        shutil.copy2(generated, destination)
        return True
    failures.append(f"Quantized export was not produced: {generated.relative_to(ROOT)}")
    return False


def copy_app_assets(enabled: bool, failures: list[str]) -> None:
    if not enabled:
        return
    copy_pairs = [
        (CLASSIFIER_EXPORTS["float32"], APP_CLASSIFIER),
        (DETECTOR_EXPORTS["raw_float32"], APP_DETECTOR),
    ]
    for source, destination in copy_pairs:
        if not source.exists():
            failures.append(f"Cannot copy missing export: {source.relative_to(ROOT)}")
            continue
        ensure_parent(destination)
        shutil.copy2(source, destination)
    ensure_parent(APP_LABELS)
    APP_LABELS.write_text(json.dumps(EXPECTED_LABELS, indent=2) + "\n", encoding="utf-8")


def build_manifest(classifier_path: Path, detector_path: Path) -> dict[str, Any]:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    detector_tensors = inspect_tflite_tensors(detector_path)
    detector_output_candidates = output_candidate_count(detector_tensors)
    return {
        "bundle_id": "agribot-model-bundle-v001",
        "created_at": now,
        "labels_sha256": sha256(APP_LABELS) if APP_LABELS.exists() else "",
        "models": {
            "classifier": {
                "file": "models/classifier_fastcrop_float32.tflite",
                "source": "models/classifier_deploy_fastcrop.pt",
                "input_width": 224,
                "input_height": 224,
                "color_space": "RGB",
                "normalization": "ultralytics_classify_default",
                "labels_file": "labels/labels.json",
                "sha256": sha256(classifier_path) if classifier_path.exists() else "",
                "confidence_threshold": 0.62,
                "high_confidence_threshold": 0.90,
            },
            "detector": {
                "file": "models/detector_nano_256_raw_float32.tflite",
                "source": "models/detector_nano_256.pt",
                "input_width": 256,
                "input_height": 256,
                "postprocess": "kotlin_yolo_xywh_conf_nms",
                "sha256": sha256(detector_path) if detector_path.exists() else "",
                "output_candidates": detector_output_candidates,
                "output_layout": "raw_yolo_features_first",
                "class_count": 1,
                "has_objectness": False,
                "coordinate_space": "normalized",
                "color_space": "RGB",
                "normalization": "zero_to_one",
                "labels_file": "labels/detector_labels.json",
            },
        },
        "compatibility": {
            "min_android_sdk": 26,
            "cpu_default": True,
            "requires_network": False,
        },
    }


def maybe_write_manifests(args: argparse.Namespace, failures: list[str]) -> None:
    if args.write_manifest:
        if not APP_CLASSIFIER.exists() or not APP_DETECTOR.exists():
            failures.append("Cannot write app manifest: app model assets are missing")
        else:
            APP_MANIFEST.write_text(json.dumps(build_manifest(APP_CLASSIFIER, APP_DETECTOR), indent=2) + "\n", encoding="utf-8")
    if args.write_bundle_manifest:
        bundle_manifest = ANDROID_EXPORTS / "bundle" / "manifest.json"
        if not CLASSIFIER_EXPORTS["int8"].exists() or not DETECTOR_EXPORTS["int8"].exists():
            failures.append("Cannot write bundle manifest: int8 TFLite exports are missing")
        else:
            ensure_parent(bundle_manifest)
            payload = build_manifest(CLASSIFIER_EXPORTS["int8"], DETECTOR_EXPORTS["int8"])
            bundle_manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def path_status(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
    }
    if path.exists():
        payload["bytes"] = path.stat().st_size
        payload["sha256"] = sha256(path)
        if path.suffix == ".tflite":
            payload["tensors"] = inspect_tflite_tensors(path)
    return payload


def inspect_tflite_tensors(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing"}
    try:
        import numpy as np
        import tensorflow as tf

        interpreter = tf.lite.Interpreter(model_path=str(path))
        interpreter.allocate_tensors()
        payload = {
            "status": "ok",
            "inputs": tensor_details(interpreter.get_input_details()),
            "outputs": tensor_details(interpreter.get_output_details()),
        }
        try:
            for detail in interpreter.get_input_details():
                tensor = np.zeros(detail["shape"], dtype=detail["dtype"])
                interpreter.set_tensor(detail["index"], tensor)
            interpreter.invoke()
            payload["invoke_status"] = "ok"
        except Exception as exc:
            payload["invoke_status"] = "failed"
            payload["invoke_error"] = str(exc)
        return payload
    except Exception as exc:
        return {"status": "unverified", "error": str(exc)}


def tensor_details(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": str(detail.get("name", "")),
            "shape": detail.get("shape", []).tolist(),
            "dtype": str(detail.get("dtype", "")),
        }
        for detail in details
    ]


def output_candidate_count(tensors: dict[str, Any]) -> int:
    outputs = tensors.get("outputs") if tensors.get("status") == "ok" else None
    if not outputs:
        return 0
    shape = outputs[0].get("shape", [])
    if len(shape) < 3:
        return 0
    fields_first = int(shape[-2])
    fields_last = int(shape[-1])
    if fields_last == 1:
        return fields_last
    if 4 <= fields_last <= 10 and fields_first > fields_last:
        return fields_first
    if 4 <= fields_first <= 10 and fields_last > fields_first:
        return fields_last
    return max(fields_first, fields_last)


def build_report(action_failures: list[str]) -> dict[str, Any]:
    source_models = {
        "classifier": path_status(CLASSIFIER_SOURCE),
        "detector": path_status(DETECTOR_SOURCE),
    }
    pt_labels = {
        "classifier": inspect_pt_labels(CLASSIFIER_SOURCE, "classify"),
        "detector": inspect_pt_labels(DETECTOR_SOURCE, "detect"),
    }
    exports = {
        "classifier": {name: path_status(path) for name, path in CLASSIFIER_EXPORTS.items()},
        "detector": {name: path_status(path) for name, path in DETECTOR_EXPORTS.items()},
    }
    app_assets = {
        "classifier": path_status(APP_CLASSIFIER),
        "detector": path_status(APP_DETECTOR),
        "labels": path_status(APP_LABELS),
        "manifest": path_status(APP_MANIFEST),
    }

    blockers = list(action_failures)
    for name, status in source_models.items():
        if not status["exists"]:
            blockers.append(f"Missing source {name}: {status['path']}")
    for name, variants in exports.items():
        int8 = variants["int8"]
        if not int8["exists"]:
            blockers.append(f"Missing Android int8 {name} export: {int8['path']}")
    for name, status in app_assets.items():
        if not status["exists"]:
            blockers.append(f"Missing app asset {name}: {status['path']}")
    if pt_labels["classifier"]["status"] != "ok":
        blockers.append(f"Classifier .pt label order not verified: {pt_labels['classifier']['status']}")
    detector_candidates = output_candidate_count(app_assets["detector"].get("tensors", {}))
    if app_assets["detector"]["exists"] and detector_candidates < 2:
        blockers.append(f"Detector TFLite output candidate count is too small for Front Overview: {detector_candidates}")
    for name in ("classifier", "detector"):
        tensors = app_assets[name].get("tensors", {})
        if tensors.get("status") == "ok" and tensors.get("invoke_status") != "ok":
            blockers.append(f"App {name} TFLite runtime invocation failed: {tensors.get('invoke_error', 'unknown error')}")

    return {
        "ready": not blockers,
        "blockers": blockers,
        "expected_labels": EXPECTED_LABELS,
        "source_models": source_models,
        "pt_labels": pt_labels,
        "exports": exports,
        "app_assets": app_assets,
    }


def main() -> int:
    args = parse_args()
    action_failures: list[str] = []
    try:
        maybe_export_models(args.export, action_failures)
        copy_app_assets(args.copy_to_app_assets, action_failures)
        maybe_write_manifests(args, action_failures)
    except Exception as exc:
        action_failures.append(str(exc))

    report = build_report(action_failures)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(json.dumps(report, indent=2))
        if report["ready"]:
            print("[OK] Android model bundle is ready.")
        else:
            print("[NOT READY] Android model bundle is incomplete.")
            for blocker in report["blockers"]:
                print(f"  - {blocker}")
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
