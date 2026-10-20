"""Create a same-contract Android bundle and mirror the manifest metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, help="candidate disease_classifier_0.tflite")
    parser.add_argument("--production-assets", default="agribot_android_app/android/app/src/main/assets")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bundle-id", default=None)
    parser.add_argument("--heldout", default=None, help="optional JSON with genuine locked-test metrics")
    args = parser.parse_args()
    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    production = Path(args.production_assets)
    manifest = json.loads((production / "model_manifest.json").read_text())
    calibration = json.loads(Path(args.calibration).read_text())
    run_meta = json.loads((Path(args.run_dir) / "meta.json").read_text())
    candidate = Path(args.candidate)
    digest = sha(candidate)
    bundle_id = args.bundle_id or f"agribot-v3-{digest[:8]}"
    target = out / "models" / "disease_classifier_0.tflite"
    shutil.copy2(candidate, target)
    shutil.copy2(production / "models" / "leaf_detector.tflite", out / "models" / "leaf_detector.tflite")
    manifest["bundle_id"] = bundle_id
    manifest["status"] = ("DRAFT: candidate trained autonomously; requires named human reviewer sign-off, "
                          "agronomist label review and licence review before release")
    manifest["classifier"]["members"] = [{
        "file": "disease_classifier_0.tflite",
        "sha256": digest,
        "arch": run_meta["model"],
        "licence": "Apache-2.0 (DINOv2 timm backbone); training data terms apply",
        "input": [1, run_meta["img"], run_meta["img"], 3],
    }]
    manifest["classifier"]["temperature"] = calibration["temperature"]
    manifest["classifier"]["other_logit_bias"] = calibration["other_bias"]
    manifest["classifier"]["energy_gate"] = False
    manifest["classifier"]["energy_reject_below"] = None
    manifest["classifier"]["combine"] = ("log(mean_i softmax(logits_i)); then add other_logit_bias to index 8; "
                                         "label = argmax; confidence = softmax(z / temperature)")
    manifest["training"] = {
        "kind": "pretrained-backbone retraining (shipped TFLite is not trainable)",
        "backbone": run_meta["model"],
        "initialization": "timm DINOv2 pretrained weights; fresh nine-class head",
        "run": Path(args.run_dir).name,
        "run_args": run_meta.get("args"),
        "data_manifest_sha256": run_meta.get("data_manifest_sha256"),
        "calibration_split_only": True,
        "revisions": calibration.get("provenance", {}),
        "selected_epoch": run_meta.get("best_epoch"),
    }
    manifest["calibration"] = {
        "temperature": calibration["temperature"],
        "other_logit_bias": calibration["other_bias"],
        "objective": calibration["selection"]["objective"],
        "chosen": calibration["selection"]["chosen"],
        "calibration_metrics": calibration["calibration_metrics"],
        "provenance": calibration["provenance"],
    }
    manifest["heldout_results"] = json.loads(Path(args.heldout).read_text()) if args.heldout else {}
    manifest["artifact"] = {"sha256": digest, "size_bytes": candidate.stat().st_size,
                            "parity": "see parity.json in the training run"}
    (out / "model_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(dict(bundle_id=bundle_id, classifier_sha256=digest, size_bytes=candidate.stat().st_size,
                          temperature=calibration["temperature"], other_bias=calibration["other_bias"]), indent=1))


if __name__ == "__main__":
    main()
