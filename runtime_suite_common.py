"""
runtime_suite_common.py
=======================
Shared paths and helpers for runtime export/evaluation scripts.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import csv
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"
EXPORTS_DIR = SCRIPT_DIR / "runtime_exports"
REPORTS_DIR = SCRIPT_DIR / "runtime_reports"
DATASET_TRACKING_DIR = SCRIPT_DIR / "dataset" / "runtime_tracking"

DEFAULT_CLASSIFIER_PT = MODELS_DIR / "classifier.pt"
DEFAULT_DETECTOR_PT = MODELS_DIR / "detector.pt"
DEFAULT_TEST_DIR = SCRIPT_DIR / "clf_dataset" / "test"

CLASSIFIER_IMGSZ = 384
DETECTOR_EXPORT_IMGSZ = 256
PIPELINE_DET_IMGSZ = 256
PIPELINE_MAX_EDGE = 256
PIPELINE_MAX_LEAVES = 1
PIPELINE_THREADS = 4
MIN_ACCEPTED_FRAME_ACCURACY = 0.96
STRICT_FRAME_ACCURACY = 0.972
MAX_RUNTIME_ACCURACY_DROP_VS_PT = 0.005
MIN_PIPELINE_FPS = 5.0
MAX_SELECTED_FRAME_LATENCY_MS = 200.0
PREFERRED_SELECTED_FRAME_LATENCY_MS = 180.0

RUNTIME_ORDER = ["pt", "onnx", "ncnn", "openvino", "tflite"]
PI_DEPLOYMENT_RUNTIMES = ["pt", "ncnn", "openvino"]

# Only generated artifact roots may be replaced by helpers in this module.
# Keeping this allow-list local to the repository prevents a malformed
# command-line path from turning an export/report operation into a broad
# deletion primitive.
_REPLACEABLE_ROOTS = (
    EXPORTS_DIR,
    REPORTS_DIR,
    DATASET_TRACKING_DIR,
    SCRIPT_DIR / "runs",
    SCRIPT_DIR / "clf_runs",
    SCRIPT_DIR / "clf_dataset_deploy_fastcrop",
    SCRIPT_DIR / "field_archives",
)


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _resolved_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _assert_replaceable(path: str | Path) -> Path:
    resolved = _resolved_path(path)
    if resolved == SCRIPT_DIR.resolve():
        raise ValueError("Refusing to replace the project root")
    if not any(_is_within(resolved, root) for root in _REPLACEABLE_ROOTS):
        raise ValueError(f"Refusing to replace path outside generated artifact roots: {path}")
    return resolved


def reset_path(path: Path):
    """Remove one generated artifact path after validating its scope.

    This function is intentionally strict because it is used by export code.
    Callers needing to replace a stable model must create a versioned artifact
    and promote it explicitly; an arbitrary path is never a valid target.
    """
    resolved = _assert_replaceable(path)
    if resolved.is_dir():
        shutil.rmtree(resolved)
    elif resolved.exists():
        resolved.unlink()


def normalize_runtime_name(runtime: str) -> str:
    value = runtime.strip().lower()
    if value not in RUNTIME_ORDER:
        raise ValueError(f"Unsupported runtime: {runtime}")
    return value


def classifier_model_path(runtime: str) -> Path:
    runtime = normalize_runtime_name(runtime)
    if runtime == "pt":
        return DEFAULT_CLASSIFIER_PT
    if runtime == "onnx":
        return EXPORTS_DIR / "classifier_onnx.onnx"
    if runtime == "ncnn":
        return EXPORTS_DIR / "classifier_ncnn_model"
    if runtime == "openvino":
        return EXPORTS_DIR / "classifier_openvino_model"
    if runtime == "tflite":
        return EXPORTS_DIR / "classifier_tflite.tflite"
    raise ValueError(runtime)


def detector_model_path(runtime: str) -> Path:
    runtime = normalize_runtime_name(runtime)
    if runtime == "pt":
        return DEFAULT_DETECTOR_PT
    if runtime == "onnx":
        return EXPORTS_DIR / "detector_onnx.onnx"
    if runtime == "ncnn":
        return EXPORTS_DIR / "detector_ncnn_model"
    if runtime == "openvino":
        return EXPORTS_DIR / "detector_openvino_model"
    if runtime == "tflite":
        return EXPORTS_DIR / "detector_tflite.tflite"
    raise ValueError(runtime)


def accuracy_report_dir(runtime: str) -> Path:
    runtime = normalize_runtime_name(runtime)
    path = REPORTS_DIR / "accuracy" / runtime
    ensure_dir(path)
    return path


def benchmark_report_dir(runtime: str) -> Path:
    runtime = normalize_runtime_name(runtime)
    path = REPORTS_DIR / "benchmark" / runtime
    ensure_dir(path)
    return path


def move_exported_path(
    exported_path: str | Path,
    final_path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Promote an exporter result without silently destroying a prior artifact.

    Exporters may return either a file or a directory.  The old implementation
    deleted ``final_path`` and moved the result into place, which made a
    failed/partial export capable of destroying the last known-good model.
    Promotion is now copy-first and refuses an existing destination unless the
    caller explicitly opts into replacement of a generated path.
    """
    source = _resolved_path(exported_path)
    final = _assert_replaceable(final_path)
    if not source.exists():
        raise FileNotFoundError(f"Exporter result does not exist: {exported_path}")
    if source == final:
        return final
    if final.exists():
        if not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing export: {final}. "
                "Choose a new versioned destination or pass overwrite=True explicitly."
            )
        reset_path(final)
    ensure_parent(final)
    if source.is_dir():
        shutil.copytree(source, final)
    else:
        shutil.copy2(source, final)
    return final


def write_json(path: Path, payload: dict):
    """Write JSON atomically and reject non-finite numbers.

    A report is used as a gate input; a truncated or NaN-containing report is
    not a valid success artifact.  ``os.replace`` makes readers see either the
    previous complete report or the new complete report.
    """
    ensure_parent(path)
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(destination.parent),
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except (TypeError, ValueError) as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise ValueError(f"JSON payload for {destination} is not finite/serializable") from exc
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _stringify(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def upsert_csv(path: Path, key_fields: list[str], row: dict):
    ensure_parent(path)
    normalized_row = {key: _stringify(value) for key, value in row.items()}

    def safe_cell(value: str) -> str:
        first = value[:1]
        numeric_sign = first in {"+", "-"} and len(value) > 1 and value[1] in "0123456789."
        return f"'{value}" if first in {"=", "@"} or (first in {"+", "-"} and not numeric_sign) else value

    normalized_row = {key: safe_cell(value) for key, value in normalized_row.items()}

    existing_rows = []
    fieldnames = list(normalized_row.keys())
    if path.exists():
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_rows = list(reader)
            for name in reader.fieldnames or []:
                if name not in fieldnames:
                    fieldnames.append(name)

    for name in fieldnames:
        normalized_row.setdefault(name, "")

    replaced = False
    for existing in existing_rows:
        if all(existing.get(field, "") == normalized_row.get(field, "") for field in key_fields):
            for name in fieldnames:
                existing[name] = normalized_row.get(name, existing.get(name, ""))
            replaced = True
            break

    if not replaced:
        existing_rows.append({name: normalized_row.get(name, "") for name in fieldnames})

    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=str(destination.parent),
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for existing in existing_rows:
                writer.writerow({name: safe_cell(existing.get(name, "")) for name in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def require_finite_float(value: float, name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    """Validate a metric/threshold before it can participate in a gate."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return number


def suite_csv_path(name: str) -> Path:
    ensure_dir(DATASET_TRACKING_DIR)
    return DATASET_TRACKING_DIR / name
