"""
16_openvino_retry.py
====================
Runs an isolated OpenVINO retry experiment with a smaller input size so the
baseline OpenVINO rows remain untouched.

The safe default uses FP16 exports and a smaller benchmark input. INT8 remains
available as an explicit opt-in because calibration/export on Raspberry Pi can
be expensive enough to destabilize the device.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from contextlib import contextmanager
from argparse import Namespace
from pathlib import Path

from benchmark_rpi_pi import run_benchmark
from runtime_accuracy_common import (
    DEFAULT_CLASSIFIER_PT,
    collect_samples,
    compare_with_baseline,
    load_model,
    predict_all,
    summarize_rows,
    write_text_report,
)
from runtime_export_common import export_variant
from runtime_suite_common import (
    DEFAULT_CLASSIFIER_PT as CLASSIFIER_PT,
    DEFAULT_DETECTOR_PT,
    DEFAULT_TEST_DIR,
    EXPORTS_DIR,
    REPORTS_DIR,
    ensure_dir,
    suite_csv_path,
    upsert_csv,
    write_json,
)


EXPERIMENT_KEY = "openvino_retry"
EXPERIMENT_LABEL = "OpenVino Retry"
DEFAULT_RETRY_CLF_IMGSZ = 192
DEFAULT_RETRY_DET_IMGSZ = 192
DEFAULT_RETRY_MAX_EDGE = 192
DEFAULT_RETRY_FRACTION = 0.05
DEFAULT_EXPORT_THREADS = 2
DEFAULT_CALIBRATION_MAX_IMAGES = 32
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description=EXPERIMENT_LABEL)
    parser.add_argument("--classifier", default=str(CLASSIFIER_PT))
    parser.add_argument("--detector", default=str(DEFAULT_DETECTOR_PT))
    parser.add_argument("--test", default=str(DEFAULT_TEST_DIR))
    parser.add_argument("--clf-imgsz", type=int, default=DEFAULT_RETRY_CLF_IMGSZ)
    parser.add_argument("--det-imgsz", type=int, default=DEFAULT_RETRY_DET_IMGSZ)
    parser.add_argument("--max-edge", type=int, default=DEFAULT_RETRY_MAX_EDGE)
    parser.add_argument("--max-leaves", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.57)
    parser.add_argument("--calibration-fraction", type=float, default=DEFAULT_RETRY_FRACTION)
    parser.add_argument("--calibration-max-images", type=int, default=DEFAULT_CALIBRATION_MAX_IMAGES)
    parser.add_argument("--export-threads", type=int, default=DEFAULT_EXPORT_THREADS)
    parser.add_argument("--classifier-int8", action="store_true")
    parser.add_argument("--no-half", action="store_true")
    parser.add_argument("--no-compare", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--skip-benchmarks", action="store_true")
    parser.add_argument(
        "--classifier-only",
        action="store_true",
        help="Run only classifier accuracy and classifier-only benchmark; skip detector benchmark/export.",
    )
    parser.add_argument(
        "--full-pipeline-only",
        action="store_true",
        help="Run only the detect+classify benchmark.",
    )
    args = parser.parse_args()
    if args.classifier_only and args.full_pipeline_only:
        parser.error("--classifier-only and --full-pipeline-only cannot be used together")
    return args


def should_run_classifier_only(args) -> bool:
    return not args.full_pipeline_only


def should_run_full_pipeline(args) -> bool:
    return not args.classifier_only


def experiment_classifier_path() -> Path:
    return EXPORTS_DIR / EXPERIMENT_KEY / "classifier_openvino_model"


def experiment_detector_path() -> Path:
    return EXPORTS_DIR / EXPERIMENT_KEY / "detector_openvino_model"


def accuracy_dir() -> Path:
    path = REPORTS_DIR / "accuracy" / EXPERIMENT_KEY
    ensure_dir(path)
    return path


def benchmark_dir() -> Path:
    path = REPORTS_DIR / "benchmark" / EXPERIMENT_KEY
    ensure_dir(path)
    return path


def calibration_dir() -> Path:
    path = REPORTS_DIR / "calibration" / EXPERIMENT_KEY
    ensure_dir(path)
    return path


def summary_csv_path() -> Path:
    return suite_csv_path("openvino_retry_results.csv")


def update_summary(**fields):
    row = {"experiment": EXPERIMENT_KEY, "label": EXPERIMENT_LABEL}
    row.update(fields)
    upsert_csv(summary_csv_path(), key_fields=["experiment"], row=row)


@contextmanager
def limited_export_threads(thread_count: int):
    thread_count = max(1, int(thread_count))
    env_keys = (
        "OMP_NUM_THREADS",
        "OPENVINO_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    previous = {key: os.environ.get(key) for key in env_keys}
    for key in env_keys:
        os.environ[key] = str(thread_count)

    torch_state = None
    try:
        import torch

        try:
            torch_state = torch.get_num_threads()
            torch.set_num_threads(thread_count)
            torch.set_num_interop_threads(1)
        except Exception:
            torch_state = None
    except Exception:
        torch_state = None

    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        if torch_state is not None:
            try:
                import torch

                torch.set_num_threads(torch_state)
            except Exception:
                pass


def _looks_like_classification_split(path: Path) -> bool:
    if not path.is_dir():
        return False
    for class_dir in path.iterdir():
        if not class_dir.is_dir():
            continue
        for image_path in class_dir.iterdir():
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                return True
    return False


def _safe_link_or_copy(source: Path, dest: Path):
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        else:
            shutil.rmtree(dest)
    try:
        dest.symlink_to(source, target_is_directory=True)
    except OSError:
        shutil.copytree(source, dest)


def _reset_dir(path: Path):
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _iter_class_images(split_root: Path) -> list[tuple[str, list[Path]]]:
    classes = []
    for class_dir in sorted(split_root.iterdir()):
        if not class_dir.is_dir():
            continue
        images = [
            image_path
            for image_path in sorted(class_dir.iterdir())
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS
        ]
        if images:
            classes.append((class_dir.name, images))
    return classes


def prepare_classifier_calibration_root(source_root: Path, max_images: int) -> Path:
    train_dir = source_root / "train"
    val_dir = source_root / "val"
    if _looks_like_classification_split(train_dir) and _looks_like_classification_split(val_dir):
        return source_root

    for candidate in (source_root / "val", source_root / "test", source_root / "train", source_root):
        if _looks_like_classification_split(candidate):
            fallback_split = candidate
            break
    else:
        raise SystemExit(
            f"[ERROR] Could not find a usable classification calibration split under: {source_root}"
        )

    temp_root = calibration_dir() / "clf_dataset"
    _reset_dir(temp_root)
    train_root = temp_root / "train"
    val_root = temp_root / "val"
    _reset_dir(train_root)
    _reset_dir(val_root)

    classes = _iter_class_images(fallback_split)
    if not classes:
        raise SystemExit(f"[ERROR] No class images found under: {fallback_split}")

    budget = max(1, int(max_images))
    total_classes = len(classes)
    per_class = max(1, budget // total_classes)
    remainder = max(0, budget - (per_class * total_classes))

    for index, (class_name, images) in enumerate(classes):
        class_budget = per_class + (1 if index < remainder else 0)
        selected = images[:class_budget]
        train_class_dir = train_root / class_name
        val_class_dir = val_root / class_name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)
        for image_path in selected:
            _safe_link_or_copy(image_path, train_class_dir / image_path.name)
            _safe_link_or_copy(image_path, val_class_dir / image_path.name)
    return temp_root


def export_models(args) -> tuple[Path, Path]:
    classifier_pt = Path(args.classifier)
    detector_pt = Path(args.detector)
    classifier_out = experiment_classifier_path()
    detector_out = experiment_detector_path()
    classifier_calibration_root = None
    if args.classifier_int8:
        classifier_calibration_root = prepare_classifier_calibration_root(
            Path(__file__).parent / "clf_dataset",
            max_images=args.calibration_max_images,
        )

    if not classifier_pt.exists():
        raise SystemExit(f"[ERROR] Classifier not found: {classifier_pt}")
    if not detector_pt.exists():
        raise SystemExit(f"[ERROR] Detector not found: {detector_pt}")

    classifier_half = not args.no_half and not args.classifier_int8
    detector_half = not args.no_half

    with limited_export_threads(args.export_threads):
        classifier_exported = export_variant(
            classifier_pt,
            task="classify",
            runtime="openvino",
            imgsz=args.clf_imgsz,
            final_path=classifier_out,
            half=classifier_half,
            int8=args.classifier_int8,
            data=(str(classifier_calibration_root) if classifier_calibration_root else None),
            fraction=args.calibration_fraction if args.classifier_int8 else None,
        )
        if should_run_full_pipeline(args):
            detector_exported = export_variant(
                detector_pt,
                task="detect",
                runtime="openvino",
                imgsz=args.det_imgsz,
                final_path=detector_out,
                half=detector_half,
                int8=False,
            )
        else:
            detector_exported = detector_out

    update_summary(
        classifier_export_status="ok",
        classifier_model=str(classifier_exported.resolve()),
        detector_export_status=("ok" if should_run_full_pipeline(args) else "skipped"),
        detector_model=(str(detector_exported.resolve()) if should_run_full_pipeline(args) else ""),
        classifier_imgsz=args.clf_imgsz,
        detector_imgsz=args.det_imgsz,
        calibration_fraction=args.calibration_fraction,
        classifier_precision=("int8" if args.classifier_int8 else ("fp16" if classifier_half else "fp32")),
        detector_precision=(("fp16" if detector_half else "fp32") if should_run_full_pipeline(args) else ""),
        classifier_calibration_data=(
            str(classifier_calibration_root.resolve()) if classifier_calibration_root is not None else ""
        ),
        calibration_max_images=(args.calibration_max_images if args.classifier_int8 else ""),
        export_threads=args.export_threads,
    )
    return classifier_exported, detector_exported


def run_accuracy_experiment(args, classifier_model_path: Path):
    test_dir = Path(args.test)
    samples = collect_samples(test_dir)
    if not samples:
        raise SystemExit(f"[ERROR] No images found in: {test_dir}")

    model = load_model(classifier_model_path)
    started = time.perf_counter()
    rows = predict_all(
        model,
        samples,
        imgsz=args.clf_imgsz,
        batch_size=1,
        progress_label=f"{EXPERIMENT_KEY} inference",
    )
    elapsed = time.perf_counter() - started
    overall, per_class, disagreements = summarize_rows(rows)

    baseline_summary = None
    drift = None
    if not args.no_compare:
        baseline_model = load_model(DEFAULT_CLASSIFIER_PT)
        baseline_rows = predict_all(
            baseline_model,
            samples,
            imgsz=args.clf_imgsz,
            batch_size=1,
            progress_label="pt baseline",
        )
        baseline_summary, drift = compare_with_baseline(rows, baseline_rows)

    report_root = accuracy_dir()
    json_path = report_root / "accuracy_report.json"
    txt_path = report_root / "accuracy_report.txt"
    predictions_path = report_root / "predictions.json"

    payload = {
        "experiment": EXPERIMENT_KEY,
        "model": str(classifier_model_path.resolve()),
        "test_dir": str(test_dir.resolve()),
        "imgsz": args.clf_imgsz,
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
        runtime=EXPERIMENT_LABEL,
        model_path=classifier_model_path,
        overall=overall,
        per_class=per_class,
        elapsed=elapsed,
        baseline_summary=baseline_summary,
        disagreements=disagreements,
        drift=drift,
    )

    update_summary(
        accuracy=f"{overall['accuracy']:.6f}",
        classifier_img_per_sec=f"{(overall['images'] / elapsed) if elapsed > 0 else 0.0:.6f}",
        pt_agreement=(f"{baseline_summary['agreement']:.6f}" if baseline_summary is not None else ""),
        accuracy_report=str(txt_path.resolve()),
        accuracy_report_json=str(json_path.resolve()),
        predictions_json=str(predictions_path.resolve()),
        test_dir=str(test_dir.resolve()),
    )

    print(f"\n[{EXPERIMENT_LABEL}]")
    print(f"  Model      : {classifier_model_path}")
    print(f"  Test set   : {test_dir}")
    print(f"  Accuracy   : {overall['accuracy'] * 100:.2f}%")
    print(f"  Elapsed    : {elapsed:.2f}s")
    if elapsed > 0:
        print(f"  Throughput : {overall['images'] / elapsed:.2f} img/s")
    if baseline_summary is not None:
        print(f"  PT agree   : {baseline_summary['agreement'] * 100:.2f}%")
    print(f"  Report     : {txt_path}")


def run_benchmark_experiment(args, classifier_model_path: Path, detector_model_path: Path, *, with_detector: bool):
    namespace = Namespace(
        test=args.test,
        det_model=str(detector_model_path),
        clf_model=str(classifier_model_path),
        det_imgsz=args.det_imgsz,
        max_edge=args.max_edge,
        max_leaves=args.max_leaves,
        conf=args.conf,
        threads=args.threads,
        limit=args.limit,
        with_detector=with_detector,
    )
    metrics = run_benchmark(namespace)
    suffix = "full_pipeline" if with_detector else "classifier_only"
    out_path = benchmark_dir() / f"{suffix}_benchmark_report.json"
    payload = {
        "experiment": EXPERIMENT_KEY,
        "mode": suffix,
        "classifier_model": namespace.clf_model,
        "detector_model": namespace.det_model,
        "args": vars(namespace),
        "metrics": metrics,
    }
    write_json(out_path, payload)

    if with_detector:
        update_summary(
            benchmark_full_pipeline_fps=f"{metrics['avg_fps']:.6f}",
            benchmark_full_pipeline_accuracy=f"{metrics['overall_accuracy']:.6f}",
            full_pipeline_report=str(out_path.resolve()),
            max_edge=args.max_edge,
            threads=args.threads,
        )
    else:
        update_summary(
            benchmark_classifier_only_fps=f"{metrics['avg_fps']:.6f}",
            benchmark_classifier_only_accuracy=f"{metrics['overall_accuracy']:.6f}",
            classifier_only_report=str(out_path.resolve()),
            max_edge=args.max_edge,
            threads=args.threads,
        )
    return metrics


def main():
    args = parse_args()
    classifier_model_path = experiment_classifier_path()
    detector_model_path = experiment_detector_path()

    update_summary(
        classifier_export_status="pending",
        detector_export_status=("pending" if should_run_full_pipeline(args) else "skipped"),
        classifier_imgsz=args.clf_imgsz,
        detector_imgsz=args.det_imgsz,
        max_edge=args.max_edge,
        calibration_fraction=args.calibration_fraction,
        calibration_max_images=(args.calibration_max_images if args.classifier_int8 else ""),
        classifier_precision=("int8" if args.classifier_int8 else ("fp16" if not args.no_half else "fp32")),
        detector_precision=(("fp16" if not args.no_half else "fp32") if should_run_full_pipeline(args) else ""),
        export_threads=args.export_threads,
    )

    if not args.skip_export:
        classifier_model_path, detector_model_path = export_models(args)

    if not args.skip_accuracy:
        run_accuracy_experiment(args, classifier_model_path)

    if not args.skip_benchmarks:
        clf_metrics = None
        pipeline_metrics = None
        if should_run_classifier_only(args):
            clf_metrics = run_benchmark_experiment(
                args,
                classifier_model_path,
                detector_model_path,
                with_detector=False,
            )
        if should_run_full_pipeline(args):
            pipeline_metrics = run_benchmark_experiment(
                args,
                classifier_model_path,
                detector_model_path,
                with_detector=True,
            )
        print(f"\n{EXPERIMENT_LABEL} summary")
        if clf_metrics is not None:
            print(f"  classifier-only FPS : {clf_metrics['avg_fps']:.2f}")
        if pipeline_metrics is not None:
            print(f"  full-pipeline FPS   : {pipeline_metrics['avg_fps']:.2f}")
        print(f"  summary CSV         : {summary_csv_path()}")


if __name__ == "__main__":
    main()