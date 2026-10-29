"""One family-prototype veto control over unchanged deployed-model features.

Hypothesis: a global binary MLP misses sparse unhealthy appearances that a
local class-conditional neighborhood can retain. One train-only prototype per
binary class/source/family; validation fits only a monotone scalar calibration.
Two fixed veto cutoffs, no encoder training or sealed-source access.
"""

from pathlib import Path
import argparse
import collections
import json
import time
import numpy as np
import torch
from torch.nn import functional as F

from cache_first_screen import retention_failures
from campaign_cnn_fusion import calibrate
from context_guard import route_guard
from opened_context_data import load, read
from permissive_encoder import digest
from phase17_context_select import coarse_counts
from phase29_compact_prepare import region_key
from phase2_select import retain
from phase5_scene_ablation import grouped, photo_metrics
from source_balance import source_class_group_weights
from training_support import atomic_save


def normalize(features):
    return F.normalize(
        torch.cat(
            [
                F.normalize(features[:, :384], dim=1),
                F.normalize(features[:, 384:], dim=1),
            ],
            1,
        ),
        dim=1,
    )


def distances(features, bank):
    distances = (1.0 - features @ bank.T).clamp_min(1e-6)
    return distances.topk(min(3, bank.shape[0]), dim=1, largest=False).values.mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    args = ap.parse_args()
    started = time.monotonic()
    p = Path(args.campaign)
    root = p / "phase29_compact"
    out = p / "phase36_family_prototypes"
    out.mkdir(exist_ok=True)
    complete = read(root / "complete.json")
    assert digest(root / "features.npy") == complete["features_sha256"]
    assert digest(root / "fitting_rows.json") == complete["fitting_rows_sha256"]
    control = read(p / "phase32_cached_epoch_0/result.json")
    data = load(p)
    protocol = dict(
        hypothesis="Local family-prototype neighborhoods retain unhealthy appearances missed by a global binary decision boundary",
        features_sha256=complete["features_sha256"],
        fitting_rows_sha256=complete["fitting_rows_sha256"],
        data_identity=data["identity"],
        existing_control_sha256=digest(p / "phase32_cached_epoch_0/result.json"),
        base_policy=control["best"]["policy"],
        bank="Mean train-only class/source/family embedding, L2-normalized; no validation family in bank",
        representation="Equal cosine weight for unchanged original DINO and CNN embeddings",
        neighbors=3,
        score="log(mean nearest unhealthy-family cosine distance / mean nearest Healthy-family cosine distance)",
        calibration="Monotone Platt, grouped validation with class/source/family weights",
        veto_cutoffs=[0.2, 0.5],
        maximum_policies=2,
        original_disease_and_healthy_exact=True,
        cold_olid_used=False,
        ad_independent=False,
        script_sha256=digest(__file__),
        stopping_rule="Stop if neither fixed veto cutoff passes all unchanged app gates; no feature-family, neighbor-count or cutoff sweep",
    )
    frozen = out / "protocol.json"
    if frozen.exists():
        assert read(frozen) == protocol
    else:
        frozen.write_text(json.dumps(protocol, indent=2))
    if (out / "result.json").exists():
        print((out / "result.json").read_text())
        return
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    values = normalize(torch.from_numpy(np.load(root / "features.npy")).cuda())
    rows = read(root / "fitting_rows.json")
    train = [i for i, r in enumerate(rows) if r["split"] == "train"]
    val = np.array([i for i, r in enumerate(rows) if r["split"] == "val"])
    assert not ({rows[i]["group"] for i in train} & {rows[i]["group"] for i in val})
    groups = collections.defaultdict(set)
    for i in train:
        r = rows[i]
        groups[r["y"], r["source"], r["group"]].add(r["table"])
    banks = []
    for label in (0, 1):
        prototypes = [
            values[sorted(ids)].mean(0)
            for (y, source, group), ids in sorted(groups.items())
            if y == label
        ]
        banks.append(F.normalize(torch.stack(prototypes), dim=1))
    with torch.inference_mode():
        raw = (
            torch.cat(
                [
                    (distances(batch, banks[0]) / distances(batch, banks[1])).log()
                    for batch in values.split(512)
                ]
            )
            .cpu()
            .numpy()
        )
    table = np.array([r["table"] for r in rows])
    truth = np.array([r["y"] for r in rows])
    calibration = calibrate(
        raw[table[val]], truth[val], source_class_group_weights(rows, val)
    )
    scores = 1 / (1 + np.exp(-np.clip(raw * calibration[0] + calibration[1], -80, 80)))
    assert np.isfinite(scores).all()
    atomic_save(
        dict(
            architecture="TrainFamilyPrototypeHealthy",
            banks=[x.cpu() for x in banks],
            calibration=calibration,
            protocol=protocol,
        ),
        out / "prototype_head.pt",
    )
    np.save(out / "probabilities.npy", scores)
    tables = [read(root / (name + "_indices.json")) for name in ("center", "full")]
    minimum = {}
    for path, row in data["photos"].items():
        keys = [region_key(path, x["box"]) for x in row["leaves"]]
        minimum[path] = min(
            [float(scores[t[k]]) for t in tables for k in keys], default=1.0
        )
    cases = read(p / "phase34_localization_audit/cases.json")
    assert {x["path"] for x in cases} <= set(data["photos"])
    proposals = {x["path"] for x in cases}
    baseline = grouped(data["base"])
    sources = sorted({r["source"] for r in data["internal"]})
    internal_before = {
        s: photo_metrics([r for r in data["internal"] if r["source"] == s])
        for s in sources
    }
    coarse_before = coarse_counts(data["base"])
    results = []
    for cutoff in protocol["veto_cutoffs"]:
        policy = protocol["base_policy"]

        def route(row):
            path = row["path"]
            h = (
                data["positive"][path]
                if path in proposals and minimum[path] >= cutoff
                else 0.0
            )
            return route_guard(
                row,
                data["manifest"],
                h,
                data["negative"][path],
                data["genus"][path],
                policy,
            )

        revised = {k: [route(r) for r in v] for k, v in data["base"].items()}
        metrics = grouped(revised)
        after = [route(r) for r in data["internal"]]
        internal = {
            s: photo_metrics([r for r in after if r["source"] == s]) for s in sources
        }
        coarse = coarse_counts(revised)
        checks = dict(
            old_retention=retain(metrics, baseline),
            old_field_healthy=all(
                metrics["phase5_selection"]["by_source"][s]["healthy_recall"] >= 0.7
                for s in ("tom2024", "swtt5wy2pr")
            ),
            coarse_retention=all(
                v[1] <= coarse_before[k][1] + int(0.005 * v[0])
                for k, v in coarse.items()
            ),
            internal_retention=retain(internal, internal_before),
            internal_healthy_each_source=all(
                m["healthy"]["correct_rate"] >= 0.7
                for m in internal.values()
                if m["healthy"]["n"]
            ),
            original_disease_and_healthy_exact=all(
                a["app_replay"] == b["app_replay"]
                for k in revised
                for a, b in zip(data["base"][k], revised[k])
                if a["app_replay"]["kind"] in ("HEALTHY", "DISEASE")
            ),
        )
        results.append(
            dict(
                cutoff=cutoff,
                checks=checks,
                passed=all(checks.values()),
                old_failures=retention_failures(metrics, baseline),
                internal_failures=retention_failures(internal, internal_before),
                metrics=metrics,
                internal_metrics=internal,
            )
        )
    report = dict(
        protocol=protocol,
        passing=sum(r["passed"] for r in results),
        prototype_counts=[b.shape[0] for b in banks],
        checkpoint_sha256=digest(out / "prototype_head.pt"),
        results=results,
        elapsed_seconds=time.monotonic() - started,
        new_encoder_inference=0,
        encoder_weights_trained=False,
        cold_olid_used=False,
    )
    (out / "result.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            dict(
                passing=report["passing"],
                prototype_counts=report["prototype_counts"],
                elapsed_seconds=report["elapsed_seconds"],
                results=[
                    dict(
                        cutoff=r["cutoff"],
                        checks=r["checks"],
                        healthy={
                            s: [m["healthy"]["correct"], m["healthy"]["n"]]
                            for s, m in r["internal_metrics"].items()
                        },
                        failures=r["old_failures"],
                    )
                    for r in results
                ],
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
