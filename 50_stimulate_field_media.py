#!/usr/bin/env python3
"""Run a bounded public-field-media and detector-geometry stimulation.

This is deliberately not a model-accuracy benchmark. It validates that a real
decoded field image can pass basic capture-quality checks and that detector
boxes at the frame boundary survive the same letterbox/unletterbox geometry
used by the Android pipeline. The Android TFLite model is exercised by the
native test suite; an actual phone inference run remains a separate gate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable


PUBLIC_SOURCES = [
    {
        "title": "Healthy tomato leaves",
        "page": "https://commons.wikimedia.org/wiki/File:Healthy_tomato_leaves_(7871755330).jpg",
        "media": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Healthy_tomato_leaves_%287871755330%29.jpg/960px-Healthy_tomato_leaves_%287871755330%29.jpg",
        "license": "CC BY 2.0",
    },
    {
        "title": "Tomatoes in the field",
        "page": "https://commons.wikimedia.org/wiki/File:Tomatoes_in_the_field_(4648307711).jpg",
        "media": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/42/Tomatoes_in_the_field_%284648307711%29.jpg/960px-Tomatoes_in_the_field_%284648307711%29.jpg",
        "license": "CC BY 2.0",
    },
    {
        "title": "Lufa Farms tomato rows",
        "page": "https://commons.wikimedia.org/wiki/File:Lufa_Farms_Tomato_Rows.jpg",
        "media": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Lufa_Farms_Tomato_Rows.jpg/960px-Lufa_Farms_Tomato_Rows.jpg",
        "license": "CC BY-SA 2.0",
    },
    {
        "title": "Powdery mildew on tomato leaves",
        "page": "https://commons.wikimedia.org/wiki/File:O%C3%AFdium_sur_Feuilles_de_Plants_de_Tomate.JPG",
        "media": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/O%C3%AFdium_sur_Feuilles_de_Plants_de_Tomate.JPG/960px-O%C3%AFdium_sur_Feuilles_de_Plants_de_Tomate.JPG",
        "license": "CC BY-SA/GFDL",
    },
]


def letterbox_parameters(frame_width: int, frame_height: int, model_width: int, model_height: int) -> tuple[float, float, float]:
    if min(frame_width, frame_height, model_width, model_height) <= 0:
        raise ValueError("all frame and model dimensions must be positive")
    scale = min(model_width / frame_width, model_height / frame_height)
    resized_width = frame_width * scale
    resized_height = frame_height * scale
    return scale, (model_width - resized_width) / 2.0, (model_height - resized_height) / 2.0


def map_model_box_to_frame(
    model_box: Iterable[float],
    frame_width: int,
    frame_height: int,
    model_width: int = 256,
    model_height: int = 256,
) -> tuple[float, float, float, float] | None:
    """Map an xyxy model-space box into the bounded camera frame."""

    values = tuple(float(value) for value in model_box)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        return None
    scale, pad_x, pad_y = letterbox_parameters(frame_width, frame_height, model_width, model_height)
    left, top, right, bottom = values
    left = max(0.0, min(float(frame_width), (left - pad_x) / scale))
    top = max(0.0, min(float(frame_height), (top - pad_y) / scale))
    right = max(0.0, min(float(frame_width), (right - pad_x) / scale))
    bottom = max(0.0, min(float(frame_height), (bottom - pad_y) / scale))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def frame_box_to_model(
    frame_box: Iterable[float],
    frame_width: int,
    frame_height: int,
    model_width: int = 256,
    model_height: int = 256,
) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in frame_box)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("frame_box must contain four finite values")
    scale, pad_x, pad_y = letterbox_parameters(frame_width, frame_height, model_width, model_height)
    left, top, right, bottom = values
    return (
        left * scale + pad_x,
        top * scale + pad_y,
        right * scale + pad_x,
        bottom * scale + pad_y,
    )


def _image_metrics(image_path: Path) -> dict[str, Any]:
    try:
        from PIL import Image, ImageStat
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("Pillow is required for public-media stimulation") from error

    with Image.open(image_path) as source:
        image = source.convert("RGB")
        width, height = image.size
        stat = ImageStat.Stat(image)
        channels = [float(value) for value in stat.mean]
        brightness = sum(channels) / (3.0 * 255.0)
        contrast = sum(float(value) for value in stat.stddev) / (3.0 * 255.0)
        sample = image.resize((min(width, 160), min(height, 160)))
        if hasattr(sample, "get_flattened_data"):
            pixels = sample.get_flattened_data()
        else:  # pragma: no cover - compatibility with older Pillow releases
            pixels = sample.getdata()
        finite_pixels = all(
            math.isfinite(float(channel))
            for pixel in pixels
            for channel in (pixel if isinstance(pixel, tuple) else (pixel,))
        )
        return {
            "path": str(image_path),
            "format": source.format,
            "width": width,
            "height": height,
            "mode": source.mode,
            "brightness_0_to_1": round(brightness, 6),
            "contrast_0_to_1": round(contrast, 6),
            "finite_sample_pixels": finite_pixels,
            "quality_stimulation": "accept" if finite_pixels and brightness > 0.04 and brightness < 0.96 and contrast > 0.025 else "retry",
        }


def run_geometry_stimulation(frame_width: int, frame_height: int) -> dict[str, Any]:
    frame_boundary_boxes = {
        "full_frame": (0.0, 0.0, float(frame_width), float(frame_height)),
        "top_left": (0.0, 0.0, float(frame_width) * 0.2, float(frame_height) * 0.2),
        "top_right": (float(frame_width) * 0.8, 0.0, float(frame_width), float(frame_height) * 0.2),
        "bottom_left": (0.0, float(frame_height) * 0.8, float(frame_width) * 0.2, float(frame_height)),
        "bottom_right": (
            float(frame_width) * 0.8,
            float(frame_height) * 0.8,
            float(frame_width),
            float(frame_height),
        ),
    }
    boundary_boxes = {
        name: frame_box_to_model(box, frame_width, frame_height)
        for name, box in frame_boundary_boxes.items()
    }
    mapped: dict[str, list[float]] = {}
    failures: list[str] = []
    for name, model_box in boundary_boxes.items():
        frame_box = map_model_box_to_frame(model_box, frame_width, frame_height)
        if frame_box is None:
            failures.append(f"{name} was rejected")
            continue
        mapped[name] = [round(value, 4) for value in frame_box]
        left, top, right, bottom = frame_box
        if not (0.0 <= left < right <= frame_width and 0.0 <= top < bottom <= frame_height):
            failures.append(f"{name} escaped frame bounds")

    roundtrip_failures: list[str] = []
    frame_boxes = [
        (0.0, 0.0, float(frame_width), float(frame_height)),
        (0.0, 0.0, float(frame_width) * 0.2, float(frame_height) * 0.2),
        (float(frame_width) * 0.8, float(frame_height) * 0.8, float(frame_width), float(frame_height)),
    ]
    for index, frame_box in enumerate(frame_boxes):
        model_box = frame_box_to_model(frame_box, frame_width, frame_height)
        recovered = map_model_box_to_frame(model_box, frame_width, frame_height)
        if recovered is None or any(abs(left - right) > 0.01 for left, right in zip(frame_box, recovered)):
            roundtrip_failures.append(f"roundtrip {index} drifted")

    failures.extend(roundtrip_failures)
    _, pad_x, pad_y = letterbox_parameters(frame_width, frame_height, 256, 256)
    if pad_y > 0 and map_model_box_to_frame((0.0, 0.0, 32.0, pad_y * 0.5), frame_width, frame_height) is not None:
        failures.append("letterbox padding was accepted as a camera box")
    if pad_x > 0 and map_model_box_to_frame((0.0, 0.0, pad_x * 0.5, 32.0), frame_width, frame_height) is not None:
        failures.append("horizontal letterbox padding was accepted as a camera box")
    return {
        "model_input": [256, 256],
        "frame": [frame_width, frame_height],
        "mapped_boundary_boxes": mapped,
        "roundtrip_checks": len(frame_boxes),
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }


def build_report(image_path: Path) -> dict[str, Any]:
    image = _image_metrics(image_path)
    geometry = run_geometry_stimulation(int(image["width"]), int(image["height"]))
    return {
        "schema": "agribot.public_field_media_stimulation.v1",
        "status": "pass" if image["quality_stimulation"] == "accept" and geometry["status"] == "pass" else "fail",
        "image": image,
        "geometry": geometry,
        "public_sources": PUBLIC_SOURCES,
        "android_model_inference": {
            "executed": False,
            "reason": "This host has no Android TFLite runtime/device camera session; native Android parser, crop, tracker, gate, and frame-quality tests are the executable local gate.",
        },
        "interpretation": [
            "The decoded public image passed bounded image-quality and coordinate-geometry stimulation.",
            "This does not establish disease classification accuracy or detector recall on unseen public media.",
            "A physical ARM64 phone run with the embedded model and a reviewed field export is still required before agronomic use.",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Downloaded public field image to stimulate")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(args.image)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"public-media stimulation failed: {error}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
