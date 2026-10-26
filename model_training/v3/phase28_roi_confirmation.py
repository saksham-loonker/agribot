"""Finite actual-leaf confirmation experiment on the completed checkpoint.

Hypothesis: whole-image Healthy evidence overlooks an unhealthy detected leaf.
Reuse identical canonical tensors by SHA and infer only new ROI tensors. No
new training, label propagation or cold-source prediction occurs here.
"""

from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import itertools
import json
import time

import numpy as np
from PIL import Image
import timm
import torch

from app_eval import classifier_letterbox_input
from campaign_pixels import open_pixels
from context_guard import route_guard
from opened_context_data import load, read
from permissive_encoder import digest
from permissive_scene import load as load_model
from phase2_select import retain
from phase5_scene_ablation import grouped, photo_metrics
from phase17_context_select import coarse_counts


def pixel_key(tensor):
    assert tensor.dtype == np.float32 and tensor.shape in (
        (224, 224, 3),
        (1, 224, 224, 3),
    )
    return hashlib.sha256(tensor.tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    a = ap.parse_args()
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timm.layers.set_fused_attn(False)
    p = Path(a.campaign)
    data = load(p)
    root = p / "phase28_roi_confirmation"
    root.mkdir(exist_ok=True)
    protocol = dict(
        data["identity"],
        hypothesis="Whole-image specialist misses disease on the actual detected leaf",
        script_sha256=digest(__file__),
        helper_sha256=digest(Path(__file__).with_name("opened_context_data.py")),
        routing_sha256=digest(Path(__file__).with_name("context_guard.py")),
        preprocessing_sha256=digest(Path(__file__).with_name("app_eval.py")),
        variants=["full_roi_with_existing_10_percent_pad"],
        all_selected_actual_leaves_required=True,
        existing_healthy_and_disease_verdicts_preserved=True,
        whole_thresholds=[0.85, 0.9, 0.95],
        leaf_thresholds=[0.5, 0.7, 0.9],
        negative_cutoffs=[0.001, 0.01],
        maximum_candidate_policies=18,
        no_leaf_policy="Existing whole-image reference guard; no fabricated leaves or observations",
        stopping_rule="No promotion/full restored verification if all 18 fixed candidates fail original and per-source internal gates",
    )
    meta = root / "protocol.json"
    if meta.exists():
        assert read(meta) == protocol
    else:
        meta.write_text(json.dumps(protocol, indent=2))
    # Warm the tensor cache from exact-checkpoint validation logits. The
    # checkpoint changed, so other encoders' feature caches are never used.
    run = data["supervised_run"]
    fit = read(run / "result.json")
    values = np.load(run / "validation_logits.npz")
    pixels = open_pixels(p / "phase23_siglip/cnn_inputs")
    tensor_scores = {}
    for i, z in zip(values["indices"], values["logits"]):
        key = pixel_key(np.asarray(pixels[int(i)]))
        score = float(z) * fit["calibration"][0] + fit["calibration"][1]
        if key in tensor_scores:
            assert abs(tensor_scores[key] - score) < 0.002
        tensor_scores[key] = score
    if hasattr(pixels, "close"):
        pixels.close()
    cache_file = root / "tensor_logits.json"
    previous = read(cache_file) if cache_file.exists() else {}
    for key, z in previous.items():
        if key in tensor_scores:
            assert abs(tensor_scores[key] - z) < 0.002
        tensor_scores[key] = z
    regions = {}
    all_records = [r for v in data["base"].values() for r in v] + data["internal"]
    for row in all_records:
        regions.setdefault(row["path"], set()).update(
            tuple(leaf["box"]) for leaf in row["leaves"]
        )
    assert set(regions) == set(data["photos"])
    index, missing, reused, total = {}, {}, 0, 0

    def prepare(item):
        path, boxes = item
        with Image.open(path) as im:
            image = np.asarray(im.convert("RGB"))
        full = classifier_letterbox_input(image, 224)
        # A previously evaluated full frame is reusable only at the same
        # canonical tensor hash and exact checkpoint/preprocessing identity.
        full_key = pixel_key(full)
        if path in data["calibrated_logits"]:
            prior = data["calibrated_logits"][path]
            if full_key in tensor_scores:
                assert abs(tensor_scores[full_key] - prior) < 0.002
        return (
            path,
            [
                (box, classifier_letterbox_input(image, 224, list(box)))
                for box in sorted(boxes)
            ],
            full_key,
        )

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for path, boxes, full_key in pool.map(prepare, regions.items()):
            if path in data["calibrated_logits"]:
                tensor_scores[full_key] = data["calibrated_logits"][path]
            for box, tensor in boxes:
                total += 1
                key = pixel_key(tensor)
                index[(path, box)] = key
                if key in tensor_scores:
                    reused += 1
                else:
                    missing.setdefault(key, tensor)
    inference_started = time.monotonic()
    if missing:
        _, model = load_model(run)
        model = model.cuda().eval()
        keys = list(missing)
        with torch.inference_mode():
            for start in range(0, len(keys), 64):
                part = keys[start : start + 64]
                inputs = torch.from_numpy(
                    np.concatenate([missing[k] for k in part], 0)
                ).cuda()
                z = model(inputs).cpu().numpy()
                assert np.isfinite(z).all()
                tensor_scores.update({k: float(v) for k, v in zip(part, z)})
        del model
        torch.cuda.empty_cache()
    tmp = cache_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(tensor_scores))
    tmp.replace(cache_file)
    inference_seconds = time.monotonic() - inference_started
    roi_scores = {
        key: float(1 / (1 + np.exp(-np.clip(tensor_scores[h], -80, 80))))
        for key, h in index.items()
    }
    roi_rows = [
        dict(
            path=path, box=list(box), probability=prob, input_sha256=index[(path, box)]
        )
        for (path, box), prob in roi_scores.items()
    ]
    (root / "roi_probabilities.json").write_text(json.dumps(roi_rows))
    baseline = grouped(data["base"])
    sources = sorted({r["source"] for r in data["internal"]})
    before = {
        s: photo_metrics([r for r in data["internal"] if r["source"] == s])
        for s in sources
    }
    coarse_before = coarse_counts(data["base"])
    results = []
    for high, leaf_min, low in itertools.product(
        protocol["whole_thresholds"],
        protocol["leaf_thresholds"],
        protocol["negative_cutoffs"],
    ):
        policy = dict(
            healthy_min=high,
            negative_healthy_max=low,
            genus_min=0.0,
            protect_suspected_disease=True,
            preserve_original_healthy=True,
            in_domain_other_max=0.1,
            in_domain_disease_mass_min=0.6,
            leaf_confirmation_min=leaf_min,
        )

        def route(row):
            agreement = all(
                roi_scores[(row["path"], tuple(leaf["box"]))] >= leaf_min
                for leaf in row["leaves"]
            )
            return route_guard(
                row,
                data["manifest"],
                data["positive"][row["path"]] if agreement else 0.0,
                data["negative"][row["path"]],
                data["genus"][row["path"]],
                policy,
            )

        revised = {k: [route(r) for r in v] for k, v in data["base"].items()}
        metrics = grouped(revised)
        after = [route(r) for r in data["internal"]]
        internal = {
            s: photo_metrics([r for r in after if r["source"] == s]) for s in sources
        }
        coarse = coarse_counts(revised)
        unchanged = sum(
            r["app_replay"] != s["app_replay"]
            for k in data["base"]
            for r, s in zip(data["base"][k], revised[k])
            if r["app_replay"]["kind"] in ("DISEASE", "HEALTHY")
        )
        checks = dict(
            old_retention=retain(metrics, baseline),
            old_healthy=all(
                metrics["phase5_selection"]["by_source"][s]["healthy_recall"] >= 0.7
                for s in ["tom2024", "swtt5wy2pr"]
            ),
            coarse_retention=all(
                v[1] <= coarse_before[k][1] + int(0.005 * v[0])
                for k, v in coarse.items()
            ),
            internal_retention=retain(internal, before),
            internal_healthy_each_source=all(
                m["healthy"]["correct_rate"] >= 0.7
                for m in internal.values()
                if m["healthy"]["n"]
            ),
            original_disease_and_healthy_verdicts_unchanged=unchanged == 0,
        )
        results.append(
            dict(
                id=len(results),
                policy=policy,
                checks=checks,
                passed=all(checks.values()),
                metrics=metrics,
                internal_metrics=internal,
            )
        )
    results.sort(
        key=lambda x: (
            x["passed"],
            x["checks"]["old_retention"]
            and x["checks"]["internal_retention"]
            and x["checks"]["coarse_retention"],
            min(
                m["healthy"]["correct_rate"]
                for m in x["internal_metrics"].values()
                if m["healthy"]["n"]
            ),
        ),
        reverse=True,
    )
    report = dict(
        protocol=protocol,
        completed_policies=len(results),
        passing=sum(r["passed"] for r in results),
        best=results[0],
        roi_inputs=total,
        unique_new_roi_tensors=len(missing),
        reused_roi_tensors=reused,
        timings=dict(
            inference_seconds=inference_seconds,
            total_seconds=time.monotonic() - started,
        ),
        tensor_cache_sha256=digest(cache_file),
        cold_olid_used=False,
        ad_independent=False,
        next_step="Full restored whole-image/real-ROI verification only if passed; otherwise stop this hypothesis and inspect explicit leaf-supervision coverage",
    )
    (root / "ranked.json").write_text(json.dumps(results, indent=2))
    (root / "result.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            dict(
                passing=report["passing"],
                checks=report["best"]["checks"],
                policy=report["best"]["policy"],
                healthy={
                    s: [m["healthy"]["correct"], m["healthy"]["n"]]
                    for s, m in report["best"]["internal_metrics"].items()
                },
                roi_inputs=total,
                new_roi_tensors=len(missing),
                timings=report["timings"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
