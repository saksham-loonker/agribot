"""Compare shared canonical pixel caches with fresh image-file preprocessing."""

from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from PIL import Image

from app_eval import classifier_letterbox_input
from campaign_pixels import open_pixels
from permissive_encoder import digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    rows = json.loads((root / "cnn_inputs/rows.json").read_text())
    admitted = json.loads((root / "admission_manifest.json").read_text())
    offset = len(rows) - len(admitted)
    ids = sorted(
        set(
            np.linspace(0, offset - 1, 54, dtype=int).tolist()
            + [offset + i for i, r in enumerate(admitted) if r["split"] == "val"]
        )
    )
    pixels = open_pixels(root / "cnn_inputs")
    results = []
    for i in ids:
        with Image.open(rows[i]["path"]) as im:
            fresh = classifier_letterbox_input(np.asarray(im.convert("RGB")), 224)[0]
        cached = np.asarray(pixels[i])
        assert (
            fresh.shape == cached.shape == (224, 224, 3) and cached.dtype == np.float32
        )
        error = np.abs(fresh - cached)
        results.append(
            dict(
                index=i,
                source=rows[i]["source"],
                admitted=i >= offset,
                max_abs_error=float(error.max()),
                mean_abs_error=float(error.mean()),
                cached_min=float(cached.min()),
                cached_max=float(cached.max()),
            )
        )
    maximum = max(r["max_abs_error"] for r in results)
    result = dict(
        passed=maximum < 1e-6,
        cases=len(results),
        max_abs_pixel_error=maximum,
        rows_sha256=digest(root / "cnn_inputs/rows.json"),
        script_sha256=hashlib.sha256(Path(__file__).read_text().encode()).hexdigest(),
        records=results,
    )
    (root / "canonical_pixel_cache_audit_verified.json").write_text(
        json.dumps(result, indent=2)
    )
    assert result["passed"], "Source-file/canonical-cache mismatch; stop training"
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
