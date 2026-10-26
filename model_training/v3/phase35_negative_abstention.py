"""Bounded abstention control for existing false Healthy, no new training.

A fixed completed recovery control is kept. A separately selected completed
leaf head can veto an existing Healthy only when BOTH canonical views of the
same actual leaf strongly agree. Original named diagnoses remain exact.
"""

from pathlib import Path
import argparse
import collections
import json
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


def probabilities(checkpoint, features):
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    model = LeafFeatureHead(state["kind"], 1408).eval()
    model.load_state_dict(state["model"], strict=True)
    with torch.inference_mode():
        return (
            (model(features) * state["calibration"][0] + state["calibration"][1])
            .sigmoid()
            .numpy()
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    args = ap.parse_args()
    p = Path(args.campaign)
    compact = p / "phase29_compact"
    out = p / "phase35_negative_abstention"
    out.mkdir(exist_ok=True)
    selected = read(p / "phase32_cached_epoch_0/result.json")
    positive = compact / "head_bce_source_balanced/epoch_0.pt"
    negative = compact / "head_bce_source_balanced/best.pt"
    assert selected["protocol"]["leaf_checkpoint_sha256"] == digest(positive)
    fit = read(compact / "head_bce_source_balanced/result.json")
    assert digest(negative) == fit["checkpoint_sha256"]
    assert digest(compact / "features.npy") == selected["protocol"]["features_sha256"]
    torch.set_num_threads(4)
    data = load(p)
    protocol = dict(
        hypothesis="Correct strongly contradicted baseline Healthy through abstention, retaining all correct Healthy counts and original disease verdicts",
        positive_leaf_checkpoint_sha256=digest(positive),
        negative_leaf_checkpoint_sha256=digest(negative),
        data_identity=data["identity"],
        base_policy=selected["best"]["policy"],
        negative_leaf_cutoffs=[0.001, 0.01],
        maximum_policies=2,
        negative_confirmation="Both center and full ROI of at least one same real leaf below cutoff",
        healthy_correction="UNSURE only; preserve raw classifier probabilities and actual frame/leaf counts",
        original_integer_floor_gates_unchanged=True,
        cold_olid_used=False,
        ad_independent=False,
        script_sha256=digest(__file__),
        stopping_rule="Stop if neither of two fixed abstention rules passes every unchanged gate; no threshold expansion",
    )
    frozen = out / "protocol.json"
    if frozen.exists():
        assert read(frozen) == protocol
    else:
        frozen.write_text(json.dumps(protocol, indent=2))
    features = torch.from_numpy(np.load(compact / "features.npy"))
    scores = probabilities(positive, features)
    negatives = probabilities(negative, features)
    tables = [read(compact / (name + "_indices.json")) for name in ("center", "full")]
    minimum = {}
    negative_agreement = {}
    for path, row in data["photos"].items():
        keys = [region_key(path, x["box"]) for x in row["leaves"]]
        minimum[path] = min(
            [float(scores[t[k]]) for t in tables for k in keys], default=1.0
        )
        negative_agreement[path] = min(
            [max(float(negatives[t[k]]) for t in tables) for k in keys], default=1.0
        )
    before = grouped(data["base"])
    sources = sorted({r["source"] for r in data["internal"]})
    internal_before = {
        s: photo_metrics([r for r in data["internal"] if r["source"] == s])
        for s in sources
    }
    coarse_before = coarse_counts(data["base"])
    results = []
    for cutoff in protocol["negative_leaf_cutoffs"]:
        policy = dict(protocol["base_policy"], preserve_original_healthy=False)

        def route(row):
            path = row["path"]
            # Recoveries retain the original completed control's veto. A
            # stronger negative signal may additionally abstain on baseline H.
            veto = negative_agreement[path] <= cutoff
            n = 0.0 if veto else data["negative"][path]
            h = (
                data["positive"][path]
                if minimum[path] >= policy["leaf_veto_below"] and not veto
                else 0.0
            )
            return route_guard(row, data["manifest"], h, n, data["genus"][path], policy)

        revised = {k: [route(r) for r in v] for k, v in data["base"].items()}
        after = grouped(revised)
        internal_rows = [route(r) for r in data["internal"]]
        internal = {
            s: photo_metrics([r for r in internal_rows if r["source"] == s])
            for s in sources
        }
        coarse = coarse_counts(revised)
        checks = dict(
            old_retention=retain(after, before),
            old_field_healthy=all(
                after["phase5_selection"]["by_source"][s]["healthy_recall"] >= 0.7
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
            original_disease_exact=all(
                a["app_replay"] == b["app_replay"]
                for k in revised
                for a, b in zip(data["base"][k], revised[k])
                if a["app_replay"]["kind"] == "DISEASE"
            ),
        )
        transitions = collections.Counter()
        for k in revised:
            for a, b in zip(data["base"][k], revised[k]):
                if (
                    a["app_replay"]["kind"] == "HEALTHY"
                    and b["app_replay"]["kind"] != "HEALTHY"
                ):
                    transitions[
                        (
                            "correct_healthy_abstained"
                            if a["label"] == "Healthy"
                            else "false_healthy_abstained"
                        )
                    ] += 1
                    assert b["app_replay"]["kind"] == "UNSURE"
                    assert (
                        b["app_replay"]["probabilities"]
                        == a["app_replay"]["probabilities"]
                    )
                    assert b["app_replay"]["frames"] == a["app_replay"]["frames"]
                    assert b["app_replay"]["leaves"] == a["app_replay"]["leaves"]
        results.append(
            dict(
                cutoff=cutoff,
                checks=checks,
                passed=all(checks.values()),
                old_failures=retention_failures(after, before),
                internal_failures=retention_failures(internal, internal_before),
                metrics=after,
                internal_metrics=internal,
                old_cohort_transitions=dict(transitions),
            )
        )
    report = dict(
        protocol=protocol,
        passing=sum(r["passed"] for r in results),
        results=results,
        new_encoder_inference=0,
        new_training=0,
        cold_olid_used=False,
    )
    (out / "result.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            [
                dict(
                    cutoff=r["cutoff"],
                    passing=r["passed"],
                    checks=r["checks"],
                    transitions=r["old_cohort_transitions"],
                    failures=r["old_failures"],
                    internal_failures=r["internal_failures"],
                )
                for r in results
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
