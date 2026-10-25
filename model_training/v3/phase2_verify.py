"""Actual exported-byte regression replay for composite phase-two pixel shards."""

from pathlib import Path
import argparse, concurrent.futures, hashlib, json, threading, time
import numpy as np
import torch
from ai_edge_litert.interpreter import Interpreter
from campaign_pixels import open_pixels
from phase2_select import evaluate, retain
from campaign_verify import regression_retained


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--assets", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    q = Path(a.root)
    assets = Path(a.assets)
    out = Path(a.out)
    out.mkdir(exist_ok=True, parents=True)
    rows = json.loads((q / "cnn_inputs/rows.json").read_text())
    ev = json.loads((q / "eval_rows.json").read_text())
    manifest = json.loads((assets / "model_manifest.json").read_text())
    member = manifest["classifier"]["members"][0]
    path = assets / "models" / member["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == member["sha256"]
    original = torch.load(q / "healthy_verifier/features.pt", weights_only=True)[
        "safe_logits"
    ].numpy()
    final = original.copy()
    pixels = open_pixels(q / "cnn_inputs")
    indices = [i for i, r in enumerate(rows) if r["set"] in ev]
    local = threading.local()
    started = time.time()

    def classify(i):
        if not hasattr(local, "runtime"):
            local.runtime = Interpreter(model_path=str(path), num_threads=2)
            local.runtime.allocate_tensors()
        runtime = local.runtime
        runtime.set_tensor(
            runtime.get_input_details()[0]["index"],
            np.ascontiguousarray(pixels[i : i + 1]),
        )
        runtime.invoke()
        return i, runtime.get_tensor(runtime.get_output_details()[0]["index"])[0]

    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        for n, (i, z) in enumerate(pool.map(classify, indices)):
            assert np.isfinite(z).all()
            final[i] = z
            if n % 200 == 0:
                print(
                    json.dumps(
                        dict(crops=n, total=len(indices), seconds=time.time() - started)
                    ),
                    flush=True,
                )
    baseline = evaluate(original, rows, ev, manifest)
    metrics = evaluate(final, rows, ev, manifest)
    np.save(out / "actual_logits.npy", final)
    report = dict(
        model_sha256=member["sha256"],
        crops=len(indices),
        seconds=time.time() - started,
        baseline=baseline,
        candidate=metrics,
        passed=retain(metrics, baseline),
        strict_passed=regression_retained(metrics, baseline),
        note="Opened regression and grouped validation photos; unchanged detector outputs reused. Not independent acceptance.",
    )
    (out / "regression_verification.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            dict(
                passed=report["passed"],
                strict=report["strict_passed"],
                seconds=report["seconds"],
            )
        ),
        flush=True,
    )
    assert report[
        "passed"
    ], "Actual exported bytes failed predeclared noninferiority gates"


if __name__ == "__main__":
    main()
