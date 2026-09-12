"""
runtime_export_common.py
========================
Shared helpers for exporting runtime variants.
"""

from __future__ import annotations

import os
from importlib import metadata
from pathlib import Path

from runtime_suite_common import move_exported_path


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _major_minor(version: str) -> tuple[int, int] | None:
    parts = version.split(".")
    if len(parts) < 2:
        return None

    values = []
    for token in parts[:2]:
        digits = ""
        for char in token:
            if char.isdigit():
                digits += char
            else:
                break
        if not digits:
            return None
        values.append(int(digits))
    return values[0], values[1]


def _check_tflite_deps():
    try:
        import tensorflow as tf
    except ImportError as exc:
        message = (
            f"TFLite export requires tensorflow but import failed: {exc}\n"
            "On Raspberry Pi 5 aarch64 install TensorFlow from PyPI, not piwheels.\n"
            "Run:\n"
            "  env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL -u PIP_FIND_LINKS \\\n"
            "    PIP_CONFIG_FILE=/dev/null python -m pip install --no-cache-dir \\\n"
            "    --index-url https://pypi.org/simple tensorflow\n"
        )
        if isinstance(exc, ModuleNotFoundError) and exc.name == "imp":
            message += (
                "Detected an old flatbuffers package that still imports the removed Python 3.13 "
                "'imp' module. This usually comes from a global piwheels pip config.\n"
                "Reinstall flatbuffers from PyPI only with:\n"
                "  env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL -u PIP_FIND_LINKS \\\n"
                "    PIP_CONFIG_FILE=/dev/null python -m pip install --no-cache-dir \\\n"
                "    --index-url https://pypi.org/simple --force-reinstall flatbuffers\n"
            )
        raise RuntimeError(message) from exc

    tf_version = str(getattr(tf, "__version__", "unknown"))
    tf_major_minor = _major_minor(tf_version)
    tf_keras_hint = (
        f"{tf_major_minor[0]}.{tf_major_minor[1]}.*" if tf_major_minor is not None else "<matching-version>"
    )
    tf_keras_version = _distribution_version("tf-keras")
    if not tf_keras_version:
        raise RuntimeError(
            "TFLite export requires tf-keras to be installed explicitly for the ONNX->TFLite step.\n"
            f"Detected tensorflow=={tf_version}. Install a matching tf-keras build from PyPI only with:\n"
            "  env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL -u PIP_FIND_LINKS \\\n"
            "    PIP_CONFIG_FILE=/dev/null python -m pip install --no-cache-dir --no-deps \\\n"
            f"    --index-url https://pypi.org/simple tf-keras=={tf_keras_hint}\n"
        )

    if _major_minor(tf_keras_version) != tf_major_minor:
        raise RuntimeError(
            "TFLite export found incompatible TensorFlow packages.\n"
            f"Detected tensorflow=={tf_version} and tf-keras=={tf_keras_version}.\n"
            "Install a matching tf-keras version from PyPI only, for example:\n"
            "  env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL -u PIP_FIND_LINKS \\\n"
            "    PIP_CONFIG_FILE=/dev/null python -m pip install --no-cache-dir --no-deps \\\n"
            f"    --index-url https://pypi.org/simple --force-reinstall tf-keras=={tf_keras_hint}\n"
            "Then rerun the export."
        )

    missing = [
        name
        for name in ("onnx2tf", "sng4onnx", "onnx-graphsurgeon", "ai-edge-litert")
        if _distribution_version(name) is None
    ]
    if missing:
        package_list = " ".join(missing)
        raise RuntimeError(
            "TFLite export dependencies are incomplete. Missing: "
            f"{', '.join(missing)}\n"
            "Install them from PyPI only with:\n"
            "  env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL -u PIP_FIND_LINKS \\\n"
            f"    PIP_CONFIG_FILE=/dev/null python -m pip install --no-cache-dir --index-url https://pypi.org/simple {package_list}\n"
        )


def export_variant(
    model_path: Path,
    task: str,
    runtime: str,
    imgsz: int,
    final_path: Path,
    *,
    half: bool = False,
    int8: bool = False,
    data: str | None = None,
    fraction: float | None = None,
    dynamic: bool | None = None,
    nms: bool | None = None,
    overwrite: bool = False,
):
    from ultralytics import YOLO

    if runtime == "tflite":
        _check_tflite_deps()

    model = YOLO(str(model_path), task=task)
    kwargs = {
        "format": runtime,
        "imgsz": imgsz,
        "batch": 1,
        "device": "cpu",
    }
    if nms is not None:
        kwargs["nms"] = nms
    if runtime == "onnx":
        kwargs["simplify"] = True
        kwargs["dynamic"] = False if dynamic is None else dynamic
        kwargs["half"] = half
    elif runtime == "ncnn":
        kwargs["half"] = half
    elif runtime == "openvino":
        kwargs["half"] = half
        kwargs["int8"] = int8
        kwargs["dynamic"] = False if dynamic is None else dynamic
        if int8:
            if not data:
                raise ValueError("OpenVINO INT8 export requires a calibration dataset path via 'data'.")
            kwargs["data"] = data
            if fraction is not None:
                kwargs["fraction"] = fraction
    elif runtime == "tflite":
        kwargs["half"] = half
        kwargs["int8"] = int8
        if int8 and data:
            kwargs["data"] = data
        if int8 and fraction is not None:
            kwargs["fraction"] = fraction

    previous_skip = os.environ.get("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS")
    if runtime == "tflite":
        os.environ["ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS"] = "1"
    try:
        exported_path = model.export(**kwargs)
    finally:
        if runtime == "tflite":
            if previous_skip is None:
                os.environ.pop("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", None)
            else:
                os.environ["ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS"] = previous_skip
    return move_exported_path(exported_path, final_path, overwrite=overwrite)
