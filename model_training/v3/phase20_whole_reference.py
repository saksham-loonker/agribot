"""Actual installed-classifier whole-image evidence for no-leaf decisions.

Older crop replay records retain historical whole-image predictions. Refresh
the reference used by a new router from one immutable installed LiteRT model;
the original baseline verdicts, detector boxes and source records stay intact.
"""

from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import threading

import numpy as np
from PIL import Image

from app_eval import Models, posterior
from campaign_heads import app_evaluate
from campaign_reference import safe_logits
from permissive_encoder import digest


def read(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    a = ap.parse_args()
    p = Path(a.campaign)
    out = p / "phase20_whole_reference"
    out.mkdir(exist_ok=True)
    leaf = p / "phase10_leaf"
    assets = p / "export_cnn_safe"
    manifest = read(assets / "model_manifest.json")
    base = app_evaluate(
        safe_logits(leaf),
        read(leaf / "cnn_inputs/rows.json"),
        read(leaf / "eval_rows.json"),
        manifest,
        True,
    )
    records = {
        r["path"]: r
        for values in base.values()
        for r in values["rows"]
        if not r["leaves"]
    }
    internal = p / "phase12_scene/union_app_validation/baseline_predictions.jsonl"
    for line in internal.read_text().splitlines():
        r = json.loads(line)
        if not r["leaves"]:
            records[r["path"]] = r
    paths = sorted(records)
    protocol = dict(
        classifier_manifest_sha256=digest(assets / "model_manifest.json"),
        classifier_sha256=manifest["classifier"]["members"][0]["sha256"],
        paths_sha256=hashlib.sha256(json.dumps(paths).encode()).hexdigest(),
        internal_baseline_sha256=digest(internal),
        code_sha256=hashlib.sha256(Path(__file__).read_text().encode()).hexdigest(),
        app_eval_sha256=hashlib.sha256(
            Path(__file__).with_name("app_eval.py").read_text().encode()
        ).hexdigest(),
        purpose="One consistent actual-byte no-leaf whole-frame reference, not fictional leaf observations or a new diagnosis",
        cold_ad_used=False,
        records=len(paths),
    )
    assert (
        protocol["classifier_sha256"]
        == "d70135804cf750341218e89a055370bef46a9b337951897b9f55f2570844ed39"
    )
    meta = out / "protocol.json"
    if meta.exists():
        assert read(meta) == protocol
    else:
        meta.write_text(json.dumps(protocol, indent=2))
    predictions = out / "predictions.jsonl"
    done = (
        {
            r["path"]: r
            for r in [json.loads(l) for l in predictions.read_text().splitlines()]
        }
        if predictions.exists()
        else {}
    )
    local = threading.local()

    def evaluate(path):
        if not hasattr(local, "model"):
            local.model = Models(assets, threads=2)
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        logits = local.model.classify(rgb)
        probabilities = posterior(logits, manifest)
        assert probabilities.shape == (9,) and np.isfinite(probabilities).all()
        return dict(
            path=path,
            file_sha256=digest(path),
            whole_member_logits=logits,
            whole_probabilities=probabilities.tolist(),
            whole_label=manifest["classifier"]["labels"][int(probabilities.argmax())],
        )

    remaining = [path for path in paths if path not in done]
    with predictions.open("a") as stream, concurrent.futures.ThreadPoolExecutor(
        2
    ) as pool:
        for result in pool.map(evaluate, remaining):
            done[result["path"]] = result
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            if len(done) % 20 == 0:
                print(
                    json.dumps(dict(complete=len(done), total=len(paths))), flush=True
                )
    assert set(done) == set(paths)
    result = dict(
        complete=True,
        records=len(done),
        predictions_sha256=digest(predictions),
        protocol=protocol,
        cold_ad_used=False,
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
