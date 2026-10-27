"""One fixed scope-consistent positive/negative field-expert routing control.

Use the crop verifier's binary label decision, scope both directions of image
evidence to natural scenes, and corroborate local conflicts on real leaf ROIs.
Every original recognition gate remains unchanged. No new head fitting.
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
from phase42_scoped_screen import calibrated_head
from spatial_cache import validate
from spatial_healthy_head import SpatialHealthyHead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--paper-extension", action="store_true")
    ap.add_argument("--adapted-tail", action="store_true")
    ap.add_argument("--dino-tail", action="store_true")
    ap.add_argument("--whole-confirmation", action="store_true")
    args = ap.parse_args()
    assert not (args.adapted_tail or args.dino_tail) or args.paper_extension
    assert not (args.adapted_tail and args.dino_tail)
    assert not args.whole_confirmation or args.dino_tail
    p = Path(args.campaign)
    out = p / (
        "phase53_whole_dino"
        if args.whole_confirmation
        else (
            "phase52_adapted_dino"
            if args.dino_tail
            else (
                "phase50_adapted_cnn"
                if args.adapted_tail
                else (
                    "phase47_scoped_paper"
                    if args.paper_extension
                    else "phase44_consistent_scope"
                )
            )
        )
    )
    out.mkdir(exist_ok=True)
    data = load(p)
    cache = prepare(p, data)
    torch.set_num_threads(4)
    inputs = read(cache / "inputs.json")
    paths = [r["path"] for r in inputs]
    features = torch.from_numpy(np.load(cache / "features.npy"))
    domain = dict(
        zip(paths, map(float, calibrated_head(p / "phase39_context_router", features)))
    )
    genus = dict(
        zip(paths, map(float, calibrated_head(p / "phase41_siglip_genus", features)))
    )
    compact = p / "phase29_compact"
    cp = compact / "head_bce_source_balanced"
    checkpoints = [cp / "epoch_0.pt", cp / "best.pt"]
    values = torch.from_numpy(np.load(compact / "features.npy"))
    scores = []
    for checkpoint in checkpoints:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        head = LeafFeatureHead(state["kind"], 1408).eval()
        head.load_state_dict(state["model"], strict=True)
        with torch.inference_mode():
            scores.append(
                (head(values) * state["calibration"][0] + state["calibration"][1])
                .sigmoid()
                .numpy()
            )
    tables = [read(compact / (name + "_indices.json")) for name in ("center", "full")]
    keys = {
        path: [region_key(path, leaf["box"]) for leaf in row["leaves"]]
        for path, row in data["photos"].items()
    }
    positive = {
        path: min([float(scores[0][t[k]]) for t in tables for k in ks], default=1.0)
        for path, ks in keys.items()
    }
    negative = {
        path: min(
            [max(float(scores[1][t[k]]) for t in tables) for k in ks], default=1.0
        )
        for path, ks in keys.items()
    }
    spatial = p / (
        "phase46_paper_primary/head"
        if args.paper_extension
        else "phase38_context_spatial/head"
    )
    fit = read(spatial / "result.json")
    assert digest(spatial / "best.pt") == fit["checkpoint_sha256"]
    contract = validate(p)
    assert fit["protocol"]["map_contract"] == contract
    adapted_probabilities = None
    if args.adapted_tail or args.dino_tail:
        if args.dino_tail:
            from partial_dino_cache import prepare as prepare_prefix
        else:
            from partial_cnn_cache import prepare as prepare_prefix

        prefix_cache = prepare_prefix(p)
        spatial = p / (
            "phase51_dino_adaptation/head"
            if args.dino_tail
            else "phase49_partial_cnn/head"
        )
        fit = read(spatial / "result.json")
        assert digest(spatial / "candidate.pt") == fit["checkpoint_sha256"]
        assert fit["protocol"]["prefix_complete_sha256"] == digest(
            prefix_cache / "complete.json"
        )
        descriptor_contract = (
            p / "phase49_partial_cnn/prefix" if args.dino_tail else prefix_cache
        )
        assert read(descriptor_contract / "protocol.json")[
            "input_descriptors_sha256"
        ] == digest(compact / "inputs.json")
        assert fit["probabilities_sha256"] == digest(spatial / "probabilities.npy")
        adapted_probabilities = np.load(spatial / "probabilities.npy")
        assert np.isfinite(adapted_probabilities).all()
        assert ((adapted_probabilities >= 0) & (adapted_probabilities <= 1)).all()
    protocol = dict(
        hypothesis="Use matching task scopes for both positive and negative evidence, with corroborated multi-leaf conflicts",
        data_identity=data["identity"],
        opened_feature_cache_sha256=digest(cache / "complete.json"),
        context_sha256=digest(p / "phase39_context_router/best.pt"),
        context_calibration_sha256=digest(p / "phase39_context_router/result.json"),
        genus_sha256=digest(p / "phase41_siglip_genus/best.pt"),
        genus_calibration_sha256=digest(p / "phase41_siglip_genus/result.json"),
        positive_leaf_sha256=digest(checkpoints[0]),
        negative_leaf_sha256=digest(checkpoints[1]),
        spatial_sha256=fit["checkpoint_sha256"],
        map_contract=contract,
        context_min=0.5,
        genus_rule="Binary primary-crop classification at .5, not an unvalidated .95 confidence requirement",
        healthy_min=0.85,
        negative_max=0.001,
        positive_leaf_min=0.1,
        leaf_negative_max=0.001,
        spatial_scope="Only multi-leaf scenes, both actual views of the same leaf must agree",
        spatial_strong_unhealthy_probability=0.8,
        spatial_partial_unhealthy_probability=0.7,
        spatial_partial_leaves=2,
        original_correct_healthy_count_must_not_drop=True,
        original_disease_exact=True,
        maximum_policies=1,
        cold_olid_used=False,
        ad_independent=False,
        script_sha256=digest(__file__),
        stopping_rule="Stop this fixed route if any unchanged original or internal gate fails; no threshold expansion",
    )
    if args.paper_extension:
        protocol.update(
            hypothesis="Combine scoped field recovery with the newly supervised primary-paper spatial expert; no source-ID routing",
            spatial_training_result_sha256=digest(spatial / "result.json"),
            isolated_scene_rule="Context below .5, explicit primary crop, at least one real leaf, all center/full leaf-view probabilities at least .95, plus existing whole positive and compact confirmation",
            isolated_spatial_healthy_min=0.95,
            spatial_scope="Natural multi-leaf conflicts require corroboration; isolated recovery requires strong agreement across all actual leaf views",
            negative_whole_scope="Field CVaR only for matching natural scenes; compact negative leaf evidence remains available in both scopes",
            maximum_policies=1,
        )
    if args.adapted_tail or args.dino_tail:
        protocol.update(
            hypothesis="Replace the frozen-map spatial head with one properly scoped supervised adaptation of the original CNN final stages; routing thresholds unchanged",
            leaf_expert=(
                "PartialDinoHealthyTail" if args.dino_tail else "PartialCnnHealthyTail"
            )
            + ", original prefix frozen and deployed recognizer unchanged",
            adapted_training_result_sha256=digest(spatial / "result.json"),
            adapted_probabilities_sha256=fit["probabilities_sha256"],
            prefix_complete_sha256=digest(prefix_cache / "complete.json"),
        )
    if args.whole_confirmation:
        protocol.update(
            hypothesis="Require the same adapted recognizer to confirm the whole plant as well as its leaf evidence, preserving every earlier threshold and app gate",
            whole_confirmation_min=0.5,
            whole_confirmation_scope="Only new Healthy proposals, binary whole-primary Healthy decision; original verdict retained on conflict",
            whole_input_source_sha256=digest(
                Path(__file__).with_name("opened_dino_whole.py")
            ),
            maximum_policies=1,
        )
    frozen = out / "protocol.json"
    if frozen.exists():
        assert read(frozen) == protocol
    else:
        frozen.write_text(json.dumps(protocol, indent=2))
    eligible = [
        path
        for path, row in data["photos"].items()
        if (
            len(row["leaves"]) >= 2
            if domain[path] >= 0.5
            else args.paper_extension and len(row["leaves"]) >= 1
        )
        and row["app_replay"]["kind"] not in ("DISEASE", "HEALTHY")
        and genus[path] >= 0.5
        and data["positive"][path] >= 0.85
        and positive[path] >= 0.1
    ]
    ids = sorted({t[k] for path in eligible for k in keys[path] for t in tables})
    if adapted_probabilities is None:
        state = torch.load(spatial / "best.pt", weights_only=True, map_location="cpu")
        model = SpatialHealthyHead().eval()
        model.load_state_dict(state["model"], strict=True)
        maps = np.load(p / "phase37_spatial/maps.npy", mmap_mode="r")
        with torch.inference_mode():
            raw = (
                torch.cat(
                    [
                        model(torch.from_numpy(np.array(maps[part], copy=True)))
                        for part in (ids[i : i + 256] for i in range(0, len(ids), 256))
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
    else:
        assert not ids or max(ids) < len(adapted_probabilities)
        lookup = {i: float(adapted_probabilities[i]) for i in ids}
    conflict = {path: False for path in paths}
    isolated_positive = {path: False for path in paths}
    for path in eligible:
        # Max over the two views requires both to support the local conflict.
        leaf = [max(lookup[t[k]] for t in tables) for k in keys[path]]
        conflict[path] = any(v <= 0.2 for v in leaf) or sum(v <= 0.3 for v in leaf) >= 2
        if args.paper_extension and domain[path] < 0.5:
            isolated_positive[path] = (
                min(lookup[t[k]] for t in tables for k in keys[path]) >= 0.95
            )
    policy = dict(
        healthy_min=0.85,
        negative_healthy_max=0.001,
        genus_min=0.5,
        protect_suspected_disease=True,
        in_domain_other_max=0.1,
        in_domain_disease_mass_min=0.6,
    )

    def route_base(row):
        path = row["path"]
        scoped = domain[path] >= 0.5 and genus[path] >= 0.5
        leaf_veto = negative[path] <= 0.001
        positive_allowed = (
            (scoped or isolated_positive[path])
            and positive[path] >= 0.1
            and not conflict[path]
            and not leaf_veto
        )
        # Scope the negative whole-image expert as well as the positive expert.
        return route_guard(
            row,
            data["manifest"],
            data["positive"][path] if positive_allowed else 0.0,
            (
                0.0
                if leaf_veto
                else (
                    data["negative"][path]
                    if scoped or not args.paper_extension
                    else 1.0
                )
            ),
            genus[path],
            dict(policy, preserve_original_healthy=not (scoped or leaf_veto)),
        )

    whole_scores = {}
    whole_report = None
    if args.whole_confirmation:
        from opened_dino_whole import prepare as prepare_whole

        all_rows = [
            row
            for rows in list(data["base"].values()) + [data["internal"]]
            for row in rows
        ]
        proposed = sorted(
            {
                row["path"]
                for row in all_rows
                if row["app_replay"]["kind"] != "HEALTHY"
                and route_base(row)["app_replay"]["kind"] == "HEALTHY"
            }
        )
        whole_scores, whole_report = prepare_whole(p, data, proposed)

    def route(row):
        candidate = route_base(row)
        if (
            args.whole_confirmation
            and row["app_replay"]["kind"] != "HEALTHY"
            and candidate["app_replay"]["kind"] == "HEALTHY"
            and whole_scores[row["path"]] < 0.5
        ):
            return row
        return candidate

    candidate = OpenedGuardScreen(data).evaluate(route)
    candidate["checks"].pop("original_disease_and_healthy_exact")
    candidate["checks"]["original_disease_exact"] = all(
        route(r)["app_replay"] == r["app_replay"]
        for values in list(data["base"].values()) + [data["internal"]]
        for r in values
        if r["app_replay"]["kind"] == "DISEASE"
    )
    candidate["passed"] = all(candidate["checks"].values())
    result = dict(
        protocol=protocol,
        candidate=candidate,
        passing=int(candidate["passed"]),
        cached_spatial_inputs=len(ids),
        new_encoder_inference=(
            whole_report["new_encoder_inference_this_call"] if whole_report else 0
        ),
        whole_confirmation_cache=whole_report,
        new_training=0,
        cold_olid_used=False,
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            dict(
                passing=result["passing"],
                checks=candidate["checks"],
                healthy={
                    s: [m["healthy"]["correct"], m["healthy"]["n"]]
                    for s, m in candidate["internal_metrics"].items()
                },
                old_failures=candidate["old_failures"],
                internal_failures=candidate["internal_failures"],
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
