"""Scope recovery by capture context and primary crop, using cached heads.

Optionally corroborate multi-leaf scenes with the completed scene/leaf spatial
head. All policies and guard roles are frozen before app-metric evaluation.
"""

from pathlib import Path
import argparse
import json
import numpy as np
import torch

from context_guard import route_guard
from leaf_feature_expert import LeafFeatureHead
from opened_context_data import load, read
from opened_guard_screen import OpenedGuardScreen
from opened_siglip_features import prepare
from permissive_encoder import digest
from phase29_compact_prepare import region_key
from spatial_cache import validate
from spatial_healthy_head import SpatialHealthyHead


def calibrated_head(directory, features):
    result = read(directory / "result.json")
    assert digest(directory / "best.pt") == result["checkpoint_sha256"]
    state = torch.load(directory / "best.pt", weights_only=True, map_location="cpu")
    model = LeafFeatureHead(state["kind"], 768).eval()
    model.load_state_dict(state["model"], strict=True)
    with torch.inference_mode():
        return (
            (model(features) * result["calibration"][0] + result["calibration"][1])
            .sigmoid()
            .numpy()
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--spatial-multileaf", action="store_true")
    args = ap.parse_args()
    p = Path(args.campaign)
    out = p / (
        "phase43_multileaf_scoped"
        if args.spatial_multileaf
        else "phase42_scoped_primary"
    )
    out.mkdir(exist_ok=True)
    data = load(p)
    cache = prepare(p, data)
    torch.set_num_threads(4)
    opened = torch.from_numpy(np.load(cache / "features.npy"))
    paths = [r["path"] for r in read(cache / "inputs.json")]
    context = dict(
        zip(paths, map(float, calibrated_head(p / "phase39_context_router", opened)))
    )
    genus = dict(
        zip(paths, map(float, calibrated_head(p / "phase41_siglip_genus", opened)))
    )
    compact = p / "phase29_compact"
    cp = compact / "head_bce_source_balanced"
    assert digest(cp / "best.pt") == read(cp / "result.json")["checkpoint_sha256"]
    assert (
        digest(cp / "epoch_0.pt")
        == read(p / "phase32_cached_epoch_0/result.json")["protocol"][
            "leaf_checkpoint_sha256"
        ]
    )
    features = torch.from_numpy(np.load(compact / "features.npy"))
    probabilities = []
    for checkpoint in (cp / "epoch_0.pt", cp / "best.pt"):
        state = torch.load(checkpoint, weights_only=True, map_location="cpu")
        head = LeafFeatureHead(state["kind"], 1408).eval()
        head.load_state_dict(state["model"], strict=True)
        with torch.inference_mode():
            probabilities.append(
                (head(features) * state["calibration"][0] + state["calibration"][1])
                .sigmoid()
                .numpy()
            )
    tables = [read(compact / (name + "_indices.json")) for name in ("center", "full")]
    keys = {
        path: [region_key(path, x["box"]) for x in r["leaves"]]
        for path, r in data["photos"].items()
    }
    positive = {
        path: min(
            [float(probabilities[0][t[k]]) for t in tables for k in values], default=1.0
        )
        for path, values in keys.items()
    }
    negative = {
        path: min(
            [max(float(probabilities[1][t[k]]) for t in tables) for k in values],
            default=1.0,
        )
        for path, values in keys.items()
    }
    protocol = dict(
        hypothesis="Explicit primary crop verification repairs the Healthy recall lost to the older crop verifier, while a capture-context router preserves isolated-leaf rejection",
        data_identity=data["identity"],
        opened_features_sha256=digest(cache / "features.npy"),
        context_checkpoint_sha256=digest(p / "phase39_context_router/best.pt"),
        context_calibration_sha256=digest(p / "phase39_context_router/result.json"),
        genus_checkpoint_sha256=digest(p / "phase41_siglip_genus/best.pt"),
        genus_calibration_sha256=digest(p / "phase41_siglip_genus/result.json"),
        positive_leaf_sha256=digest(cp / "epoch_0.pt"),
        negative_leaf_sha256=digest(cp / "best.pt"),
        whole_healthy_min=0.85,
        whole_negative_max=0.001,
        genus_min=0.95,
        positive_leaf_veto_min=0.1,
        negative_leaf_abstention_max=0.001,
        context_minima=[0.5, 0.9],
        original_disease_exact=True,
        correct_healthy_count_not_lower=True,
        cold_olid_used=False,
        ad_independent=False,
        script_sha256=digest(__file__),
        maximum_policies=2,
        stopping_rule="Stop if neither fixed policy passes all unchanged app gates; no threshold expansion",
    )
    spatial_min = {path: 1.0 for path in paths}
    if args.spatial_multileaf:
        contract = validate(p)
        directory = p / "phase38_context_spatial/head"
        fit = read(directory / "result.json")
        assert fit["protocol"]["map_contract"] == contract
        assert digest(directory / "best.pt") == fit["checkpoint_sha256"]
        state = torch.load(directory / "best.pt", weights_only=True, map_location="cpu")
        model = SpatialHealthyHead().eval()
        model.load_state_dict(state["model"], strict=True)
        maps = np.load(p / "phase37_spatial/maps.npy", mmap_mode="r")
        eligible = [
            path
            for path, row in data["photos"].items()
            if len(row["leaves"]) >= 2
            and row["app_replay"]["kind"] not in ("HEALTHY", "DISEASE")
            and data["positive"][path] >= 0.85
            and context[path] >= 0.5
            and genus[path] >= 0.95
            and positive[path] >= 0.1
        ]
        ids = sorted({t[k] for path in eligible for k in keys[path] for t in tables})
        with torch.inference_mode():
            raw = (
                torch.cat(
                    [
                        model(torch.from_numpy(np.array(maps[part], copy=True)))
                        for part in (
                            ids[start : start + 256]
                            for start in range(0, len(ids), 256)
                        )
                    ]
                ).numpy()
                if ids
                else np.array([], np.float32)
            )
        lookup = {
            i: float(
                1
                / (
                    1
                    + np.exp(
                        -np.clip(
                            z * state["calibration"][0] + state["calibration"][1],
                            -80,
                            80,
                        )
                    )
                )
            )
            for i, z in zip(ids, raw)
        }
        for path in eligible:
            spatial_min[path] = min(lookup[t[k]] for k in keys[path] for t in tables)
        protocol.update(
            hypothesis="Corroborate multi-leaf scene recovery with an existing spatial symptom head, keeping the single-leaf expert separate",
            spatial_checkpoint_sha256=fit["checkpoint_sha256"],
            spatial_training_protocol_sha256=digest(directory / "protocol.json"),
            spatial_scope="Only multi-leaf scenes (at least two actual detector ROIs); no invented leaves",
            spatial_minima=[0.2, 0.5],
            context_minima=[0.5],
            cached_spatial_inputs=len(ids),
        )
    frozen = out / "protocol.json"
    if frozen.exists():
        assert read(frozen) == protocol
    else:
        frozen.write_text(json.dumps(protocol, indent=2))
    screen = OpenedGuardScreen(data)
    results = []
    policies = (
        [(0.5, c) for c in protocol["spatial_minima"]]
        if args.spatial_multileaf
        else [(c, 0.0) for c in protocol["context_minima"]]
    )
    for context_min, local_min in policies:
        policy = dict(
            healthy_min=0.85,
            negative_healthy_max=0.001,
            genus_min=0.95,
            protect_suspected_disease=True,
            in_domain_other_max=0.1,
            in_domain_disease_mass_min=0.6,
        )

        def route(row):
            path = row["path"]
            veto = negative[path] <= 0.001
            allowed = (
                context[path] >= context_min
                and positive[path] >= 0.1
                and spatial_min[path] >= local_min
                and not veto
            )
            return route_guard(
                row,
                data["manifest"],
                data["positive"][path] if allowed else 0.0,
                0.0 if veto else data["negative"][path],
                genus[path],
                dict(policy, preserve_original_healthy=not veto),
            )

        candidate = screen.evaluate(route)
        checks = candidate["checks"]
        checks.pop("original_disease_and_healthy_exact")
        checks["original_disease_exact"] = all(
            route(row)["app_replay"] == row["app_replay"]
            for values in list(data["base"].values()) + [data["internal"]]
            for row in values
            if row["app_replay"]["kind"] == "DISEASE"
        )
        candidate.update(
            passed=all(checks.values()), context_min=context_min, spatial_min=local_min
        )
        results.append(candidate)
    result = dict(
        protocol=protocol,
        passing=sum(r["passed"] for r in results),
        results=results,
        new_encoder_inference=0,
        new_training=0,
        cold_olid_used=False,
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            dict(
                passing=result["passing"],
                results=[
                    dict(
                        context_min=r["context_min"],
                        spatial_min=r["spatial_min"],
                        checks=r["checks"],
                        healthy={
                            s: [m["healthy"]["correct"], m["healthy"]["n"]]
                            for s, m in r["internal_metrics"].items()
                        },
                        old_failures=r["old_failures"],
                        internal_failures=r["internal_failures"],
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
