"""Inspect residual Healthy conflicts before choosing more fitting/inference.

Only opened development images and identity-checked existing features are used.
Publisher targets are retained, including coarse and unsupported conditions.
"""

from pathlib import Path
import argparse
import collections
import json
import numpy as np
import torch

from app_eval import scope_of
from context_guard import route_guard
from leaf_feature_expert import LeafFeatureHead
from opened_context_data import load, read
from permissive_encoder import digest
from phase29_compact_prepare import region_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    args = ap.parse_args()
    p = Path(args.campaign)
    # Version the corrected protocol; never rewrite the historical audit.
    out = p / "phase34_localization_audit_v2"
    out.mkdir(exist_ok=True)
    compact = p / "phase29_compact"
    checkpoint = compact / "head_bce_source_balanced/epoch_0.pt"
    control = read(p / "phase32_cached_epoch_0/result.json")
    assert control["protocol"]["leaf_checkpoint_sha256"] == digest(checkpoint)
    assert control["protocol"]["features_sha256"] == digest(compact / "features.npy")
    data = load(p)
    protocol = dict(
        purpose="Diagnose residual localization/label-scope limitations; no new trained control or acceptance",
        leaf_checkpoint_sha256=digest(checkpoint),
        prior_control_sha256=digest(p / "phase32_cached_epoch_0/result.json"),
        policy=control["best"]["policy"],
        script_sha256=digest(__file__),
        cold_olid_used=False,
        opened_data_identity=data["identity"],
    )
    frozen = out / "protocol.json"
    if frozen.exists():
        assert read(frozen) == protocol
    else:
        frozen.write_text(json.dumps(protocol, indent=2))
    torch.set_num_threads(4)
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    model = LeafFeatureHead(state["kind"], 1408).eval()
    model.load_state_dict(state["model"], strict=True)
    with torch.inference_mode():
        score = (
            (
                model(torch.from_numpy(np.load(compact / "features.npy")))
                * state["calibration"][0]
                + state["calibration"][1]
            )
            .sigmoid()
            .numpy()
        )
    tables = [read(compact / (name + "_indices.json")) for name in ("center", "full")]
    cases = []
    for path, row in data["photos"].items():
        keys = [region_key(path, leaf["box"]) for leaf in row["leaves"]]
        minimum = min([float(score[t[k]]) for t in tables for k in keys], default=1.0)
        policy = control["best"]["policy"]
        after = route_guard(
            row,
            data["manifest"],
            data["positive"][path] if minimum >= policy["leaf_veto_below"] else 0.0,
            data["negative"][path],
            data["genus"][path],
            policy,
        )
        if (
            row["app_replay"]["kind"] in ("HEALTHY", "DISEASE")
            or after["app_replay"]["kind"] != "HEALTHY"
            or not row.get("analysis_eligible", True)
        ):
            continue
        cases.append(
            dict(
                id=row["id"],
                path=path,
                file_sha256=row.get("file_sha256"),
                source=row["source"],
                label=row["label"],
                raw_label=row.get("raw_label"),
                scope=scope_of(row),
                baseline=row["app_replay"],
                original_leaves=len(row["leaves"]),
                boxes=[leaf["box"] for leaf in row["leaves"]],
                leaf_healthy_minimum=minimum,
                whole_healthy=data["positive"][path],
                negative_healthy=data["negative"][path],
                genus=data["genus"][path],
                false_healthy=row["label"] != "Healthy",
            )
        )
    failures = [c for c in cases if c["false_healthy"]]
    summary = dict(
        protocol=protocol,
        new_healthy_photos=len(cases),
        new_false_healthy_photos=len(failures),
        by_scope=dict(collections.Counter(c["scope"] for c in failures)),
        false_healthy_without_detected_leaf=sum(
            c["original_leaves"] == 0 for c in failures
        ),
        true_healthy_without_detected_leaf=sum(
            not c["false_healthy"] and c["original_leaves"] == 0 for c in cases
        ),
        counts_by_source=dict(collections.Counter(c["source"] for c in failures)),
        new_encoder_inference=0,
        new_fitting=0,
        cold_olid_used=False,
    )
    (out / "cases.json").write_text(json.dumps(cases, indent=2))
    (out / "false_healthy.json").write_text(json.dumps(failures, indent=2))
    (out / "result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
