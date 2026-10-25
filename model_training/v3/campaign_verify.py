"""Verify exported bytes against opened photo regressions and cached predictions.

The detector is unchanged. Its prior exact detections are reused while every
selected crop is classified by the exported TFLite model, then replayed through
the complete plant decision policy. No labels enter inference.
"""

from pathlib import Path
import argparse, concurrent.futures, hashlib, json, threading, time
import numpy as np
import torch
from ai_edge_litert.interpreter import Interpreter
from campaign_heads import app_evaluate


def regression_retained(candidate, baseline):
    return all(
        v["healthy"]["correct"] >= baseline[k]["healthy"]["correct"]
        and v["supported_disease"]["correct"]
        >= baseline[k]["supported_disease"]["correct"]
        and v["supported_disease"]["false_healthy"]
        <= baseline[k]["supported_disease"]["false_healthy"]
        and v["all"]["false_healthy"] <= baseline[k]["all"]["false_healthy"]
        and v["supported"]["false_other"] <= baseline[k]["supported"]["false_other"]
        and all(
            (v[s]["false_known"] or 0) <= (baseline[k][s]["false_known"] or 0)
            for s in ["unsupported_tomato", "non_tomato"]
        )
        for k, v in candidate.items()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    a = parser.parse_args()
    root = Path(a.root)
    assets = Path(a.assets)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((assets / "model_manifest.json").read_text())
    member = manifest["classifier"]["members"][0]
    assert len(manifest["classifier"]["members"]) == 1 and member["input"] == [
        1,
        224,
        224,
        3,
    ]
    path = assets / "models" / member["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == member["sha256"]
    work = root / "dense_aligned"
    cache = torch.load(work / "features.pt", weights_only=True)
    jobs = json.loads((work / "feature_rows.json").read_text())
    eval_rows = json.loads((work / "eval_rows.json").read_text())
    eval_rows.pop("selection_app")
    cnn_rows = json.loads((root / "cnn_inputs/rows.json").read_text())
    assert all(cnn_rows[i]["feature_index"] == i for i in range(len(jobs)))
    pixels = np.load(root / "cnn_inputs/center.npy", mmap_mode="r")
    indices = [i for i, r in enumerate(jobs) if r["set"] in eval_rows]
    local = threading.local()

    def classify(i):
        if not hasattr(local, "runtime"):
            local.runtime = Interpreter(model_path=str(path), num_threads=2)
            local.runtime.allocate_tensors()
        runtime = local.runtime
        tensor = np.ascontiguousarray(pixels[i : i + 1])
        runtime.set_tensor(runtime.get_input_details()[0]["index"], tensor)
        runtime.invoke()
        return i, runtime.get_tensor(runtime.get_output_details()[0]["index"]).ravel()

    final = cache["logits"].numpy().copy()
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
        for n, (i, z) in enumerate(pool.map(classify, indices)):
            assert np.isfinite(z).all()
            final[i] = z
            if n % 100 == 0:
                print(
                    json.dumps(
                        dict(
                            crops=n,
                            total=len(indices),
                            seconds=round(time.time() - started, 1),
                        )
                    ),
                    flush=True,
                )
    baseline = app_evaluate(cache["logits"].numpy(), jobs, eval_rows, manifest)
    candidate = app_evaluate(final, jobs, eval_rows, manifest, True)
    metrics = {k: v["metrics"] for k, v in candidate.items()}
    for name, result in candidate.items():
        with (out / (name + "_predictions.jsonl")).open("w") as stream:
            for r in result["rows"]:
                stream.write(json.dumps(r) + "\n")
    report = dict(
        model_sha256=member["sha256"],
        crops=len(indices),
        seconds=time.time() - started,
        baseline=baseline,
        candidate=metrics,
        passed=regression_retained(metrics, baseline),
        note="Opened regressions, not independent acceptance. Original detector outputs reused without modification.",
    )
    (out / "regression_verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            dict(passed=report["passed"], seconds=report["seconds"], metrics=metrics)
        ),
        flush=True,
    )
    assert report["passed"], "Exported model failed regression gates"


if __name__ == "__main__":
    main()
