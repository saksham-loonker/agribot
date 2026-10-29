"""Replay the deployed schema-3 crop-confirmation policy on actual model bytes."""

from pathlib import Path
import argparse, concurrent.futures, json, threading, time
import numpy as np
from PIL import Image
from app_eval import Models, classifier_letterbox_input
from campaign_pixels import open_pixels
from phase2_select import evaluate, retain
from campaign_reference import safe_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prototype", help="Optional frozen crop-policy logits for parity")
    a = ap.parse_args()
    q = Path(a.root)
    assets = Path(a.assets)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((q / "cnn_inputs/rows.json").read_text())
    ev = json.loads((q / "eval_rows.json").read_text())
    original = safe_logits(q)
    final = original.copy()
    indices = [i for i, r in enumerate(rows) if r["set"] in ev]
    pixels = open_pixels(q / "cnn_inputs")
    local = threading.local()
    started = time.time()
    errors = []

    def classify(i):
        if not hasattr(local, "model"):
            local.model = Models(assets, 2)
        model = local.model
        runtime, _ = model.members[0]
        raw = model.invoke(runtime, np.ascontiguousarray(pixels[i : i + 1])).ravel()
        assert raw.shape == (18,)

        def full(size):
            image = np.asarray(Image.open(rows[i]["path"]).convert("RGB"))
            return classifier_letterbox_input(image, size, rows[i]["box"])

        selected = model.select_view_logits([raw], full)[0]
        return (
            i,
            np.asarray(selected, np.float32),
            float(abs(raw[9:] - original[i]).max()),
        )

    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        for n, (i, z, error) in enumerate(pool.map(classify, indices)):
            final[i] = z
            errors.append(error)
            if n % 300 == 0:
                print(
                    json.dumps(
                        dict(crops=n, total=len(indices), seconds=time.time() - started)
                    ),
                    flush=True,
                )
    manifest = json.loads((assets / "model_manifest.json").read_text())
    baseline = evaluate(original, rows, ev, manifest)
    observed = evaluate(final, rows, ev, manifest)
    np.save(out / "actual_guarded_logits.npy", final)
    prototype_error = None
    if a.prototype:
        prototype = np.load(a.prototype)
        assert prototype.shape == final.shape
        prototype_error = float(abs(final[indices] - prototype[indices]).max())
    report = dict(
        model_sha256=manifest["classifier"]["members"][0]["sha256"],
        candidate=observed,
        baseline=baseline,
        baseline_output_parity_max=max(errors),
        prototype_parity_max=prototype_error,
        passed=retain(observed, baseline)
        and max(errors) < 0.002
        and (prototype_error is None or prototype_error < 0.002),
        crops=len(indices),
        seconds=time.time() - started,
        note="Opened development and regressions; schema-3 reference fallback and full-ROI confirmation run from actual TFLite bytes.",
    )
    (out / "regression_verification.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            dict(
                passed=report["passed"],
                baseline_parity=max(errors),
                prototype_parity=prototype_error,
                seconds=report["seconds"],
            )
        ),
        flush=True,
    )
    assert report["passed"], report


if __name__ == "__main__":
    main()
