"""Regenerate the 13 golden expectations against a candidate bundle's exported bytes."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from app_eval import Models, posterior, canon

ROOT = Path(__file__).resolve().parents[2]
TARGETS = [
    "model_training/v2/bundle/golden/golden_expected.json",
    "agribot_android_app/android/ml/src/test/resources/golden/golden_expected.json",
    "agribot_android_app/android/app/src/androidTest/assets/golden/golden_expected.json",
]


def energy(probabilities, logits, manifest):
    c = manifest["classifier"]
    combined = np.log(np.mean([canon.softmax(z) for z in logits], axis=0) + 1e-12)
    combined[c["other_index"]] += c["other_logit_bias"]
    temperature = c["temperature"]
    z = combined / temperature
    known = [k for k in range(len(combined)) if k != c["other_index"]]
    return float(temperature * (np.log(np.exp(z[known] - z[known].max()).sum()) + z[known].max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True)
    parser.add_argument("--golden-dir", default="model_training/v2/bundle/golden")
    parser.add_argument("--out", required=True)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    assets = Path(args.assets)
    golden_dir = Path(args.golden_dir)
    template = json.loads((golden_dir / "golden_expected.json").read_text())
    models = Models(assets, threads=4)
    manifest = models.manifest
    entries = []
    for entry in template["entries"]:
        image = np.asarray(Image.open(golden_dir / entry["file"]).convert("RGB"))
        logits = models.classify(image)
        probs = posterior(logits, manifest)
        detections = models.detect(image)
        updated = dict(entry)
        updated["whole_image"] = {
            "member_logits": logits,
            "probs": probs.tolist(),
            "energy": energy(probs, logits, manifest),
            "label": manifest["classifier"]["labels"][int(probs.argmax())],
        }
        updated["detections"] = [list(map(float, d)) for d in detections]
        if detections:
            box = detections[0][:4]
            crop_logits = models.classify(image, box)
            crop_probs = posterior(crop_logits, manifest)
            updated["top_crop"] = {
                "box": [float(v) for v in box],
                "member_logits": crop_logits,
                "probs": crop_probs.tolist(),
                "label": manifest["classifier"]["labels"][int(crop_probs.argmax())],
            }
        else:
            updated.pop("top_crop", None)
        entries.append(updated)
        print(entry["file"], updated["whole_image"]["label"], len(detections), flush=True)
    template["entries"] = entries
    template["note"] = ("Expected outputs from the candidate .tflite files with canon.py preprocessing; "
                        "regenerated from exported bytes, source labels are unchanged")
    template["bundle_id"] = manifest["bundle_id"]
    template["classifier_sha256"] = manifest["classifier"]["members"][0]["sha256"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(template, indent=1))
    if args.install:
        for target in TARGETS:
            shutil.copy2(out, ROOT / target)
            print("installed", target)
    print("wrote", out)


if __name__ == "__main__":
    main()
