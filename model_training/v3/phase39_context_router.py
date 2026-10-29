"""Scope the field specialist to appropriate capture contexts, not source IDs.

Fit a linear context router using source capture-style metadata, never health
labels. Current nine-class decisions remain the default. Images from ambiguous
source contexts are masked for router fitting; no acceptance source is read.
"""

from pathlib import Path
import argparse
import copy
import json
import numpy as np
import torch
from torch.nn import functional as F

from campaign_cnn_fusion import calibrate
from context_guard import route_guard
from leaf_feature_expert import LeafFeatureHead
from opened_context_data import load, read
from opened_guard_screen import OpenedGuardScreen, ranking
from opened_siglip_features import prepare
from permissive_encoder import digest
from phase29_compact_prepare import region_key
from source_balance import source_class_group_weights
from training_support import atomic_save

DOMAIN = {
    "tom2024": 1,
    "ad_greenhouse": 1,
    "tomatovillage": 0,
    "tlid": 0,
    "swtt5wy2pr": 0,
    "bangladesh_classification": 0,
    "bangladesh_binary": 0,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    args = ap.parse_args()
    p = Path(args.campaign)
    root = p / "phase23_siglip"
    out = p / "phase39_context_router"
    out.mkdir(exist_ok=True)
    data = load(p)
    cache = prepare(p, data)
    rows = read(root / "cnn_inputs/rows.json")
    selected = [i for i, r in enumerate(rows) if r["source"] in DOMAIN]
    fitting = [dict(rows[i], y=DOMAIN[rows[i]["source"]]) for i in selected]
    assert not (
        {r["group"] for r in fitting if r["split"] == "train"}
        & {r["group"] for r in fitting if r["split"] == "val"}
    )
    compact = p / "phase29_compact"
    positive_leaf = compact / "head_bce_source_balanced/epoch_0.pt"
    negative_leaf = compact / "head_bce_source_balanced/best.pt"
    protocol = dict(
        hypothesis="A field-trained Healthy specialist must not override useful isolated-leaf rejection",
        feature_cache_sha256=digest(cache / "complete.json"),
        fitting_feature_sha256=digest(root / "siglip2_feature_heads/features.npy"),
        source_rows_sha256=digest(root / "cnn_inputs/rows.json"),
        domain_metadata=DOMAIN,
        metadata_scope="Weak source capture-style supervision, not manually adjudicated scene labels; PlantDoc/Pakistan masked for fitting",
        head="Linear normalized frozen SigLIP768",
        epochs=60,
        seed=1201,
        loss="source/class/family balanced BCE",
        domain_thresholds=[0.5, 0.9],
        tomato_genus_min=0.95,
        whole_healthy_min=0.85,
        whole_negative_max=0.001,
        leaf_positive_veto_min=0.1,
        leaf_negative_abstention_max=0.001,
        positive_leaf_sha256=digest(positive_leaf),
        negative_leaf_sha256=digest(negative_leaf),
        selection="Grouped router-validation weighted BCE; actual app gates are subsequent mandatory checks",
        data_identity=data["identity"],
        cold_olid_used=False,
        ad_independent=False,
        script_sha256=digest(__file__),
        maximum_policies=2,
        stopping_rule="Stop this router if neither fixed context threshold passes unchanged app gates; no threshold/architecture expansion",
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
    torch.manual_seed(1201)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    features = torch.from_numpy(
        np.load(root / "siglip2_feature_heads/features.npy")[selected]
    ).cuda()
    train = np.array([i for i, r in enumerate(fitting) if r["split"] == "train"])
    val = np.array([i for i, r in enumerate(fitting) if r["split"] == "val"])
    targets = np.array([r["y"] for r in fitting])
    truth = torch.tensor(targets, dtype=torch.float32, device="cuda")
    sampling = torch.tensor(
        source_class_group_weights(fitting, train), dtype=torch.float64
    )
    vw = source_class_group_weights(fitting, val)
    weights = torch.tensor(vw, dtype=torch.float32, device="cuda")
    model = LeafFeatureHead("linear", 768).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    best = float("inf")
    for epoch in range(60):
        model.train()
        order = torch.multinomial(
            sampling,
            len(train),
            replacement=True,
            generator=torch.Generator().manual_seed(120100 + epoch),
        )
        for positions in order.split(1024):
            ids = train[positions.numpy()]
            loss = F.binary_cross_entropy_with_logits(model(features[ids]), truth[ids])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            z = model(features[val])
            score = float(
                (
                    F.binary_cross_entropy_with_logits(z, truth[val], reduction="none")
                    * weights
                ).sum()
            )
        if score < best:
            best = score
            atomic_save(
                dict(
                    kind="linear",
                    model={
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    },
                    epoch=epoch,
                    protocol=protocol,
                ),
                out / "best.pt",
            )
    state = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    with torch.inference_mode():
        z = model(features[val]).cpu().numpy()
    calibration = calibrate(z, targets[val], vw)
    opened = torch.from_numpy(np.load(cache / "features.npy")).cuda()
    with torch.inference_mode():
        probability = (
            (model(opened) * calibration[0] + calibration[1]).sigmoid().cpu().numpy()
        )
    paths = [r["path"] for r in read(cache / "inputs.json")]
    domain = dict(zip(paths, map(float, probability)))
    values = torch.from_numpy(np.load(compact / "features.npy")).cuda()
    leaf_scores = []
    for checkpoint in (positive_leaf, negative_leaf):
        s = torch.load(checkpoint, weights_only=True, map_location="cpu")
        head = LeafFeatureHead(s["kind"], 1408).cuda().eval()
        head.load_state_dict(s["model"], strict=True)
        with torch.inference_mode():
            leaf_scores.append(
                (head(values) * s["calibration"][0] + s["calibration"][1])
                .sigmoid()
                .cpu()
                .numpy()
            )
    tables = [read(compact / (name + "_indices.json")) for name in ("center", "full")]
    positive_min = {}
    negative_agreement = {}
    for path, row in data["photos"].items():
        keys = [region_key(path, x["box"]) for x in row["leaves"]]
        positive_min[path] = min(
            [float(leaf_scores[0][t[k]]) for t in tables for k in keys], default=1.0
        )
        negative_agreement[path] = min(
            [max(float(leaf_scores[1][t[k]]) for t in tables) for k in keys],
            default=1.0,
        )
    screen = OpenedGuardScreen(data)
    results = []
    for threshold in protocol["domain_thresholds"]:
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
            veto = negative_agreement[path] <= 0.001
            h = (
                data["positive"][path]
                if not veto and domain[path] >= threshold and positive_min[path] >= 0.1
                else 0.0
            )
            # Preserve baseline H unless BOTH actual views strongly contradict it.
            local = dict(policy, preserve_original_healthy=not veto)
            return route_guard(
                row,
                data["manifest"],
                h,
                0.0 if veto else data["negative"][path],
                data["genus"][path],
                local,
            )

        candidate = screen.evaluate(route)
        # This control explicitly permits only H->UNSURE abstention. Correct-H
        # count and disease gates remain unchanged; exact H identity is not a gate.
        checks = candidate["checks"]
        checks.pop("original_disease_and_healthy_exact")
        checks["original_disease_exact"] = all(
            route(r)["app_replay"] == r["app_replay"]
            for values in data["base"].values()
            for r in values
            if r["app_replay"]["kind"] == "DISEASE"
        )
        candidate.update(
            passed=all(checks.values()), domain_min=threshold, policy=policy
        )
        results.append(candidate)
    result = dict(
        protocol=protocol,
        checkpoint_sha256=digest(out / "best.pt"),
        calibration=calibration,
        selected_epoch=state["epoch"],
        passing=sum(r["passed"] for r in results),
        results=results,
        encoder_weights_trained=False,
        cold_olid_used=False,
    )
    (out / "probabilities.json").write_text(json.dumps(domain))
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            dict(
                passing=result["passing"],
                selected_epoch=state["epoch"],
                results=[
                    dict(
                        threshold=r["domain_min"],
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
