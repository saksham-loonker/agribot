#!/usr/bin/env python3
"""Exercise Android-equivalent field preprocessing and box geometry on media."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
MODEL_SIZE = 256
MIN_BOX_SIDE_PX = 12.0
MIN_BOX_AREA_RATIO = 0.002


def resize_for_mask(image: Image.Image, max_edge: int = 256) -> np.ndarray:
    rgb = image.convert("RGB")
    scale = min(1.0, max_edge / max(rgb.width, rgb.height))
    if scale < 1.0:
        rgb = rgb.resize(
            (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))),
            Image.Resampling.BILINEAR,
        )
    return np.asarray(rgb, dtype=np.float32)


def rgb_mask_bounds(image: Image.Image) -> tuple[int, int, int, int] | None:
    values = resize_for_mask(image)
    values /= 255.0
    red, green, blue = values[..., 0], values[..., 1], values[..., 2]
    maximum = np.max(values, axis=-1)
    minimum = np.min(values, axis=-1)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    red_max = (maximum == red) & (delta > 1e-6)
    green_max = (maximum == green) & (delta > 1e-6)
    blue_max = (maximum == blue) & (delta > 1e-6)
    hue[red_max] = ((green[red_max] - blue[red_max]) / delta[red_max]) % 6.0
    hue[green_max] = (blue[green_max] - red[green_max]) / delta[green_max] + 2.0
    hue[blue_max] = (red[blue_max] - green[blue_max]) / delta[blue_max] + 4.0
    hue = hue * 30.0  # OpenCV's 0..180 hue scale.
    saturation = np.where(maximum <= 1e-6, 0.0, delta / maximum * 255.0)
    value = maximum * 255.0
    mask = (
        ((hue >= 18.0) & (hue <= 95.0) & (saturation >= 25.0) & (value >= 30.0))
        | (((hue <= 18.0) | (hue >= 165.0)) & (saturation >= 45.0) & (value >= 30.0))
        | ((saturation >= 45.0) & (value >= 35.0) & (value <= 245.0))
    )
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def letterbox_shape(width: int, height: int) -> tuple[int, int, int, int, float, float]:
    scale = min(MODEL_SIZE / width, MODEL_SIZE / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    pad_x = math.floor((MODEL_SIZE - resized_width) / 2)
    pad_y = math.floor((MODEL_SIZE - resized_height) / 2)
    return resized_width, resized_height, pad_x, pad_y, resized_width / width, resized_height / height


def round_trip_box(width: int, height: int, box: tuple[float, float, float, float]) -> float:
    resized_width, resized_height, pad_x, pad_y, scale_x, scale_y = letterbox_shape(width, height)
    left, top, right, bottom = box
    model_box = (
        left * scale_x + pad_x,
        top * scale_y + pad_y,
        right * scale_x + pad_x,
        bottom * scale_y + pad_y,
    )
    restored = (
        (model_box[0] - pad_x) / scale_x,
        (model_box[1] - pad_y) / scale_y,
        (model_box[2] - pad_x) / scale_x,
        (model_box[3] - pad_y) / scale_y,
    )
    return max(abs(a - b) for a, b in zip(box, restored))


def dataset_boxes() -> list[dict]:
    image_root = ROOT / "dataset" / "test" / "images"
    label_root = ROOT / "dataset" / "test" / "labels"
    records = []
    for label_path in sorted(label_root.glob("*.txt")):
        image_path = next((candidate for candidate in image_root.glob(f"{label_path.stem}.*")), None)
        if image_path is None:
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, box_width, box_height = map(float, parts[:5])
            absolute = (
                (cx - box_width / 2) * width,
                (cy - box_height / 2) * height,
                (cx + box_width / 2) * width,
                (cy + box_height / 2) * height,
            )
            records.append({"image": image_path.name, "width": width, "height": height, "box": absolute})
    return records


def run() -> dict:
    report: dict = {"fixtures": [], "annotated_boxes": {"count": 0, "max_round_trip_error_px": 0.0, "boundary_errors": []}}
    failures: list[str] = []
    fixture_paths = sorted(FIXTURE_DIR.glob("*.jpg"))
    if not fixture_paths:
        failures.append("no image fixtures found; add an appropriately licensed JPEG to the fixtures directory")

    for path in fixture_paths:
        with Image.open(path) as image:
            mask_values = resize_for_mask(image)
            bounds = rgb_mask_bounds(image)
            if image.width < 32 or image.height < 32:
                failures.append(f"fixture too small: {path.name}")
            if bounds is None:
                failures.append(f"RGB vegetation mask empty: {path.name}")
            mask_width, mask_height = mask_values.shape[1], mask_values.shape[0]
            report["fixtures"].append({
                "name": path.name,
                "width": image.width,
                "height": image.height,
                "mask_width": mask_width,
                "mask_height": mask_height,
                "mask_bounds": bounds,
                "mask_area_fraction": None if bounds is None else ((bounds[2] - bounds[0]) * (bounds[3] - bounds[1])) / (mask_width * mask_height),
            })

    boxes = dataset_boxes()
    report["annotated_boxes"]["count"] = len(boxes)
    for record in boxes:
        width, height = record["width"], record["height"]
        error = round_trip_box(width, height, record["box"])
        report["annotated_boxes"]["max_round_trip_error_px"] = max(report["annotated_boxes"]["max_round_trip_error_px"], error)
        if error > 1.5:
            failures.append(f"letterbox round trip drift {error:.2f}px: {record['image']}")

        left, top, right, bottom = record["box"]
        for edge_box in ((0.0, 0.0, right, bottom), (left, top, float(width), float(height))):
            edge_error = round_trip_box(width, height, edge_box)
            if edge_error > 1.5:
                report["annotated_boxes"]["boundary_errors"].append({"image": record["image"], "error_px": edge_error})

    report["status"] = "passed" if not failures else "failed"
    report["failures"] = failures
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "field_stimulation_report.json")
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
