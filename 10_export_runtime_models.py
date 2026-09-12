"""
10_export_runtime_models.py
===========================
Exports detector and classifier models into clearly named runtime variants.

This keeps all exported artifacts under:
  runtime_exports/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from runtime_export_common import export_variant
from runtime_suite_common import (
    CLASSIFIER_IMGSZ,
    DEFAULT_CLASSIFIER_PT,
    DEFAULT_DETECTOR_PT,
    DETECTOR_EXPORT_IMGSZ,
    EXPORTS_DIR,
    RUNTIME_ORDER,
    classifier_model_path,
    detector_model_path,
    ensure_dir,
    move_exported_path,
    suite_csv_path,
    upsert_csv,
    write_json,
)


def parse_args():
    p = argparse.ArgumentParser(description="Export detector/classifier runtime variants")
    p.add_argument("--classifier", default=str(DEFAULT_CLASSIFIER_PT))
    p.add_argument("--detector", default=str(DEFAULT_DETECTOR_PT))
    p.add_argument(
        "--runtimes",
        nargs="+",
        default=[name for name in RUNTIME_ORDER if name != "pt"],
        help="Subset of runtimes to export: onnx ncnn openvino tflite",
    )
    p.add_argument("--clf-imgsz", type=int, default=CLASSIFIER_IMGSZ)
    p.add_argument("--det-imgsz", type=int, default=DETECTOR_EXPORT_IMGSZ)
    p.add_argument("--openvino-int8-classifier", action="store_true")
    p.add_argument("--openvino-int8-detector", action="store_true")
    p.add_argument("--openvino-clf-data", default=str(Path(__file__).parent / "clf_dataset"))
    p.add_argument("--openvino-det-data", default=str(Path(__file__).parent / "dataset" / "data.yaml"))
    p.add_argument("--openvino-fraction", type=float, default=1.0)
    p.add_argument(
        "--overwrite-exports",
        action="store_true",
        help="Replace existing generated runtime exports after explicit review",
    )
    return p.parse_args()


def main():
    args = parse_args()
    classifier_pt = Path(args.classifier)
    detector_pt = Path(args.detector)

    if not classifier_pt.exists():
        raise SystemExit(f"[ERROR] Classifier not found: {classifier_pt}")
    if not detector_pt.exists():
        raise SystemExit(f"[ERROR] Detector not found: {detector_pt}")

    runtimes = []
    for runtime in args.runtimes:
        if runtime not in RUNTIME_ORDER or runtime == "pt":
            raise SystemExit(f"[ERROR] Unsupported export runtime: {runtime}")
        runtimes.append(runtime)

    ensure_dir(EXPORTS_DIR)
    report = {
        "classifier": str(classifier_pt.resolve()),
        "detector": str(detector_pt.resolve()),
        "exports": [],
        "failures": [],
    }

    print(f"\nExport root: {EXPORTS_DIR.resolve()}\n")

    for runtime in runtimes:
        print(f"[{runtime}]")

        clf_out = classifier_model_path(runtime)
        det_out = detector_model_path(runtime)

        try:
            exported = export_variant(
                classifier_pt,
                task="classify",
                runtime=runtime,
                imgsz=args.clf_imgsz,
                final_path=clf_out,
                int8=runtime == "openvino" and args.openvino_int8_classifier,
                data=args.openvino_clf_data if runtime == "openvino" and args.openvino_int8_classifier else None,
                fraction=args.openvino_fraction if runtime == "openvino" and args.openvino_int8_classifier else None,
                overwrite=args.overwrite_exports,
            )
            print(f"  classifier -> {exported}")
            report["exports"].append(
                {
                    "runtime": runtime,
                    "task": "classify",
                    "path": str(exported.resolve()),
                }
            )
            upsert_csv(
                suite_csv_path("export_results.csv"),
                key_fields=["runtime", "task"],
                row={
                    "runtime": runtime,
                    "task": "classify",
                    "status": "ok",
                    "path": str(exported.resolve()),
                    "error": "",
                },
            )
        except Exception as exc:
            print(f"  classifier export failed: {exc}")
            report["failures"].append(
                {"runtime": runtime, "task": "classify", "error": str(exc)}
            )
            upsert_csv(
                suite_csv_path("export_results.csv"),
                key_fields=["runtime", "task"],
                row={
                    "runtime": runtime,
                    "task": "classify",
                    "status": "failed",
                    "path": "",
                    "error": str(exc),
                },
            )

        try:
            exported = export_variant(
                detector_pt,
                task="detect",
                runtime=runtime,
                imgsz=args.det_imgsz,
                final_path=det_out,
                int8=runtime == "openvino" and args.openvino_int8_detector,
                data=args.openvino_det_data if runtime == "openvino" and args.openvino_int8_detector else None,
                fraction=args.openvino_fraction if runtime == "openvino" and args.openvino_int8_detector else None,
                overwrite=args.overwrite_exports,
            )
            print(f"  detector   -> {exported}")
            report["exports"].append(
                {
                    "runtime": runtime,
                    "task": "detect",
                    "path": str(exported.resolve()),
                }
            )
            upsert_csv(
                suite_csv_path("export_results.csv"),
                key_fields=["runtime", "task"],
                row={
                    "runtime": runtime,
                    "task": "detect",
                    "status": "ok",
                    "path": str(exported.resolve()),
                    "error": "",
                },
            )
        except Exception as exc:
            print(f"  detector export failed: {exc}")
            report["failures"].append(
                {"runtime": runtime, "task": "detect", "error": str(exc)}
            )
            upsert_csv(
                suite_csv_path("export_results.csv"),
                key_fields=["runtime", "task"],
                row={
                    "runtime": runtime,
                    "task": "detect",
                    "status": "failed",
                    "path": "",
                    "error": str(exc),
                },
            )

        print()

    report_path = EXPORTS_DIR / "export_report.json"
    write_json(report_path, report)
    print(f"Report -> {report_path}")


if __name__ == "__main__":
    main()
