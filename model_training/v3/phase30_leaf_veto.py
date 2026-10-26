"""Bounded cached test: field specialist plus an independently fitted leaf veto.

The specialist and leaf head were supervised on different input scopes. A
low leaf Healthy score vetoes only a newly recovered Healthy proposal, never
the original recognizer. No additional encoder inference or cold opening.
"""

from pathlib import Path
import argparse
import itertools
import json
import time

import numpy as np
import torch

from cache_first_screen import retention_failures
from context_guard import route_guard
from leaf_feature_expert import LeafFeatureHead
from opened_context_data import load, read
from permissive_encoder import digest
from phase17_context_select import coarse_counts
from phase29_compact_prepare import region_key
from phase2_select import retain
from phase5_scene_ablation import grouped, photo_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument(
        "--leaf-run",
        default="head_bce",
        choices=("head_bce", "head_bce_source_balanced"),
    )
    ap.add_argument(
        "--checkpoint-epoch",
        type=int,
        choices=(0, 4, 9, 19, 39, 59, 79, 99),
        help="Bounded reuse of an already-trained source-balanced checkpoint, four fixed policies",
    )
    args = ap.parse_args()
    p = Path(args.campaign)
    compact = p / "phase29_compact"
    out = p / (
        "phase30_leaf_veto"
        if args.leaf_run == "head_bce"
        else "phase31_source_balanced_veto"
    )
    if args.checkpoint_epoch is not None:
        assert args.leaf_run == "head_bce_source_balanced"
        out = p / ("phase32_cached_epoch_" + str(args.checkpoint_epoch))
    out.mkdir(exist_ok=True)
    complete = read(compact / "complete.json")
    assert digest(compact / "features.npy") == complete["features_sha256"]
    result = read(compact / args.leaf_run / "result.json")
    checkpoint = compact / args.leaf_run / "best.pt"
    assert digest(checkpoint) == result["checkpoint_sha256"]
    if args.checkpoint_epoch is not None:
        checkpoint = (
            compact / args.leaf_run / ("epoch_" + str(args.checkpoint_epoch) + ".pt")
        )
    protocol = dict(
        hypothesis="Independent explicit-leaf supervision can veto unhealthy leaves overlooked by the field image specialist",
        leaf_checkpoint_sha256=digest(checkpoint),
        features_sha256=complete["features_sha256"],
        fitting_protocol_sha256=digest(compact / args.leaf_run / "protocol.json"),
        whole_thresholds=[0.85, 0.95],
        leaf_veto_below=[0.05, 0.1, 0.2],
        negative_cutoffs=[0.001, 0.01],
        maximum_policies=12,
        original_disease_and_healthy_unchanged=True,
        new_encoder_inference=0,
        cold_olid_used=False,
        ad_independent=False,
        script_sha256=digest(__file__),
        stopping_rule="Stop this combination if none of the 12 fixed policies passes all unchanged app gates",
    )
    if args.checkpoint_epoch is not None:
        protocol.update(
            leaf_veto_below=[0.1, 0.2],
            negative_cutoffs=[0.001],
            maximum_policies=4,
            checkpoint_epoch=args.checkpoint_epoch,
            stopping_rule="Four fixed policies per eight already-trained checkpoints, at most 32 checks; no new fitting or encoder inference",
        )
    identity = out / "protocol.json"
    if identity.exists():
        assert read(identity) == protocol
    else:
        identity.write_text(json.dumps(protocol, indent=2))
    if (out / "result.json").exists():
        print((out / "result.json").read_text())
        return
    started = time.monotonic()
    torch.set_num_threads(4)
    data = load(p)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = LeafFeatureHead(state["kind"], 1408).eval()
    model.load_state_dict(state["model"], strict=True)
    values = torch.from_numpy(np.load(compact / "features.npy"))
    with torch.inference_mode():
        scores = (
            (model(values) * state["calibration"][0] + state["calibration"][1])
            .sigmoid()
            .numpy()
        )
    center = read(compact / "center_indices.json")
    full = read(compact / "full_indices.json")
    baseline = grouped(data["base"])
    sources = sorted({r["source"] for r in data["internal"]})
    before = {
        s: photo_metrics([r for r in data["internal"] if r["source"] == s])
        for s in sources
    }
    coarse_before = coarse_counts(data["base"])
    leaf_minimum = {}
    for path, row in data["photos"].items():
        keys = [region_key(path, leaf["box"]) for leaf in row["leaves"]]
        leaf_minimum[path] = min(
            [float(scores[table[k]]) for k in keys for table in (center, full)],
            default=1.0,
        )
    ranked = []
    for high, veto, low in itertools.product(
        protocol["whole_thresholds"],
        protocol["leaf_veto_below"],
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
            leaf_veto_below=veto,
        )

        def route(row):
            path = row["path"]
            return route_guard(
                row,
                data["manifest"],
                data["positive"][path] if leaf_minimum[path] >= veto else 0.0,
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
        unchanged = all(
            r["app_replay"] == s["app_replay"]
            for k in data["base"]
            for r, s in zip(data["base"][k], revised[k])
            if r["app_replay"]["kind"] in ("DISEASE", "HEALTHY")
        )
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
            internal_retention=retain(internal, before),
            internal_healthy_each_source=all(
                m["healthy"]["correct_rate"] >= 0.7
                for m in internal.values()
                if m["healthy"]["n"]
            ),
            original_verdicts_preserved=unchanged,
        )
        recalls = [
            m["healthy"]["correct_rate"] for m in internal.values() if m["healthy"]["n"]
        ]
        ranked.append(
            dict(
                policy=policy,
                checks=checks,
                passed=all(checks.values()),
                metrics=metrics,
                internal_metrics=internal,
                old_failures=retention_failures(metrics, baseline),
                internal_failures=retention_failures(internal, before),
                worst_internal_healthy=min(recalls),
            )
        )
    ranked.sort(
        key=lambda r: (
            r["passed"],
            r["checks"]["old_retention"]
            and r["checks"]["coarse_retention"]
            and r["checks"]["internal_retention"],
            r["worst_internal_healthy"],
        ),
        reverse=True,
    )
    report = dict(
        protocol=protocol,
        data_identity=data["identity"],
        evaluated_policies=len(ranked),
        passing=sum(r["passed"] for r in ranked),
        best=ranked[0],
        elapsed_seconds=time.monotonic() - started,
        new_encoder_inference=0,
        cold_olid_used=False,
    )
    (out / "ranked.json").write_text(json.dumps(ranked, indent=2))
    (out / "result.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            dict(
                passing=report["passing"],
                checks=ranked[0]["checks"],
                old_failures=ranked[0]["old_failures"],
                internal_failures=ranked[0]["internal_failures"],
                healthy={
                    s: [m["healthy"]["correct"], m["healthy"]["n"]]
                    for s, m in ranked[0]["internal_metrics"].items()
                },
                elapsed_seconds=report["elapsed_seconds"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
