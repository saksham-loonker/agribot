"""Evaluate a later prototype on the consumed phase-eight regression only.

This command never opens a new acceptance source and cannot promote a model.
The baseline is the immutable one-time phase-eight replay. Changed model or
implementation identities cannot silently resume an existing output directory.
"""

from pathlib import Path
import argparse
import json

import timm
import torch

from app_eval import Models
from phase2_select import retain
from phase5_scene_ablation import photo_metrics
from phase8_accept import Prototype, digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--selection-run", required=True)
    ap.add_argument("--run-name", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    out = root / a.run_name
    out.mkdir(exist_ok=True)
    selected = root / a.selection_run / "candidate.json"
    candidate = json.loads(selected.read_text())
    assert candidate["retained"] and candidate["field_healthy_passed"]
    previous = root / "phase8_acceptance"
    assert (previous / "result.json").exists()
    records = json.loads((root / "reserved_manifest.json").read_text())
    baseline = {
        r["id"]: r["baseline"]
        for r in [
            json.loads(line)
            for line in (previous / "predictions.jsonl").read_text().splitlines()
        ]
    }
    assert len(baseline) == len(records) == 246
    fit = candidate["fusion"]
    identity = fit["input_identity"]
    for key, path in [
        ("feature_model_sha256", root / "export_initial/pytorch_bundle.pt"),
        ("scene_cnn_sha256", root / fit["scene_run"] / "best.pt"),
        ("scene_cnn_calibration_sha256", root / fit["scene_run"] / "result.json"),
        ("species_cnn_sha256", root / "species/best.pt"),
        ("species_cnn_calibration_sha256", root / "species/result.json"),
    ]:
        assert identity[key] == digest(path)
    protocol = dict(
        scope="Opened regression only, not fresh acceptance; no promotion is possible from this command.",
        candidate_sha256=digest(selected),
        manifest_sha256=digest(root / "reserved_manifest.json"),
        baseline_predictions_sha256=digest(previous / "predictions.jsonl"),
        input_identity=identity,
        script_sha256=digest(__file__),
    )
    frozen = out / "protocol.json"
    if frozen.exists():
        assert json.loads(frozen.read_text()) == protocol
    else:
        frozen.write_text(json.dumps(protocol, indent=2))
    if (out / "result.json").exists():
        print((out / "result.json").read_text())
        return
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timm.layers.set_fused_attn(False)
    model = Prototype(root, candidate)
    path = out / "predictions.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done[r["id"]] = r
    with path.open("a") as stream, torch.inference_mode():
        for i, r in enumerate(records):
            if r["id"] in done:
                continue
            assert digest(r["path"]) == r["file_sha256"]
            observed = model.record(r)
            stream.write(json.dumps(observed) + "\n")
            stream.flush()
            done[r["id"]] = observed
            if i % 30 == 0:
                print(json.dumps(dict(evaluated=i + 1, total=len(records))), flush=True)
    sources = sorted({r["source"] for r in records})
    b = {
        s: photo_metrics([baseline[r["id"]] for r in records if r["source"] == s])
        for s in sources
    }
    c = {
        s: photo_metrics([done[r["id"]] for r in records if r["source"] == s])
        for s in sources
    }
    result = dict(
        protocol=protocol,
        baseline=b,
        candidate=c,
        retained=retain(c, b),
        fresh_acceptance=False,
        promotable=False,
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            dict(
                retained=result["retained"],
                healthy={
                    s: (v["healthy"]["correct"], v["healthy"]["n"])
                    for s, v in c.items()
                },
                false_healthy={s: v["all"]["false_healthy"] for s, v in c.items()},
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
