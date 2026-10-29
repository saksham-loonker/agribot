"""Two bounded supervised image-bag heads on original diagnostic features.

Leaf views are representations, not independently labelled clinical samples.
One publisher primary-image label supervises pooled logits across its real ROIs.
"""

from pathlib import Path
import argparse
import copy
import json
import time
import numpy as np
import torch
from torch.nn import functional as F

from campaign_cnn_fusion import calibrate
from context_guard import route_guard
from leaf_feature_expert import LeafFeatureHead
from opened_context_data import load, read
from opened_guard_screen import OpenedGuardScreen
from permissive_encoder import digest
from phase2_select import retain
from phase5_scene_ablation import photo_metrics
from phase83_eligible_operating_points import safety as cohort_safety
from phase84_population_head import population
from phase85_population_sft import evaluation_data
from phase87_diagnostic_bag_features import prepare
from population_siglip_tail import freeze
from source_balance import source_class_group_weights
from training_support import atomic_save


def route_paired(row, retained, packet, positive, negative, minimum, manifest):
    if not population(row, packet) or not packet["leaves"]:
        return retained
    hp, _, genus, _ = packet["whole"]
    compact_positive = min(
        (v[0] for leaf in packet["leaves"] for v in leaf), default=1.0
    )
    compact_negative = min(
        (max(v[1] for v in leaf) for leaf in packet["leaves"]), default=1.0
    )
    proposed = (
        positive >= minimum
        and hp >= 0.85
        and genus >= 0.5
        and compact_positive >= 0.1
        and compact_negative > 0.001
    )
    return route_guard(
        retained,
        manifest,
        positive if proposed else 0.0,
        negative,
        genus,
        dict(
            healthy_min=minimum,
            negative_healthy_max=0.001,
            genus_min=0.5,
            protect_suspected_disease=True,
            in_domain_other_max=0.1,
            in_domain_disease_mass_min=0.6,
        ),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    args = ap.parse_args()
    p = Path(args.campaign)
    out = p / "phase88_diagnostic_bag_head"
    out.mkdir(exist_ok=True)
    if (out / "result.json").exists():
        result = read(out / "result.json")
        assert result["source_sha256"] == digest(__file__)
        print(json.dumps(dict(passing=result["passing"], completed=True)), flush=True)
        return
    data = load(p)
    values, bags, feature_metadata = prepare(p, data)
    entries, records, packets, before, previous = evaluation_data(p, data)
    rows = read(p / "phase84_population_head/fitting_rows.json")
    active = [
        i
        for i, r in enumerate(rows)
        if r["active"] and r["path"] in bags and bags[r["path"]]["center"]
    ]
    validation = np.array([r["validation"] for r in rows])
    val = np.array([i for i in active if validation[i]])
    validation_inputs = {
        key for i in val for view in bags[rows[i]["path"]].values() for key in view
    }
    overlapping_groups = {
        rows[i]["group"]
        for i in active
        if not validation[i]
        and any(
            key in validation_inputs
            for view in bags[rows[i]["path"]].values()
            for key in view
        )
    }
    train = np.array(
        [
            i
            for i in active
            if not validation[i] and rows[i]["group"] not in overlapping_groups
        ]
    )
    assert not ({rows[i]["group"] for i in train} & {rows[i]["group"] for i in val})
    assert not (
        {key for i in train for view in bags[rows[i]["path"]].values() for key in view}
        & validation_inputs
    )
    assert set(rows[i]["y"] for i in train) == set(rows[i]["y"] for i in val) == {0, 1}
    protocol = dict(
        hypothesis="The existing disease-trained DINO/CNN features provide fine-grained damage information that generic whole-image SigLIP failed to recover",
        feature_cache=feature_metadata,
        fitting_rows_sha256=digest(p / "phase84_population_head/fitting_rows.json"),
        source_sha256=digest(__file__),
        head_source_sha256=digest(Path(__file__).with_name("leaf_feature_expert.py")),
        heads=["linear", "mlp"],
        maximum_epochs=60,
        minimum_epochs=8,
        patience=6,
        learning_rate=0.001,
        negative_loss_weight=8.0,
        weight_decay=0.01,
        batch_images=64,
        seed=1889,
        aggregation="Mean raw head logits over actual detected ROIs separately for center/full views",
        supervision="One publisher primary-image label per bag; no per-detection label propagation",
        robust_loss="Healthy uses minimum view logit; non-Healthy uses maximum view logit",
        positive="Both image views must exceed max(.95, maximum eligible validation-negative positive score +1e-6)",
        negative="Both image views must be <=.001 before abstaining from original Healthy",
        calibration="Source/class/family balanced validation BCE over two views with image weights divided by two",
        train_images=len(train),
        validation_images=len(val),
        overlap_training_groups_quarantined=sorted(overlapping_groups),
        source_class_family_balanced=True,
        encoder_weights_changed=False,
        original_disease_output_unchanged=True,
        old_all_dataset_gates_unchanged=True,
        olid_independent=False,
        ad_independent=False,
        stopping="Two heads maximum; if neither passes every old gate and consumed family-validation Healthy/safety, stop before export and consider one copied diagnostic-tail adaptation",
    )
    freeze(out / "protocol.json", protocol)
    torch.set_num_threads(4)
    torch.manual_seed(1889)
    features = torch.from_numpy(values).cuda()
    truth = torch.tensor([r["y"] for r in rows], dtype=torch.float32, device="cuda")
    tables = np.zeros((len(rows), 2, 6), np.int64)
    mask = np.zeros((len(rows), 2, 6), np.float32)
    for i, row in enumerate(rows):
        bag = bags.get(row["path"])
        if bag is None:
            continue
        for view, name in enumerate(("center", "full")):
            assert len(bag[name]) <= 6
            tables[i, view, : len(bag[name])] = bag[name]
            mask[i, view, : len(bag[name])] = 1.0
    table = torch.from_numpy(tables).cuda()
    weights_mask = torch.from_numpy(mask).cuda()
    sampling = torch.from_numpy(source_class_group_weights(rows, train))
    vw = source_class_group_weights(rows, val)
    val_weight = torch.tensor(vw, dtype=torch.float32, device="cuda")

    def forward(model, ids):
        inputs = features[table[ids]]
        raw = model(inputs.reshape(-1, 1408)).reshape(len(ids), 2, 6)
        return (raw * weights_mask[ids]).sum(2) / weights_mask[ids].sum(2).clamp_min(
            1.0
        )

    base_kind = {
        r["path"]: r["app_replay"]["kind"]
        for values0 in list(data["base"].values()) + [data["internal"]]
        for r in values0
    }
    base_kind.update({r["path"]: r["app_replay"]["kind"] for r in before})
    results = []
    started = time.monotonic()
    for kind in ("linear", "mlp"):
        directory = out / kind
        directory.mkdir(exist_ok=True)
        if (directory / "result.json").exists():
            saved = read(directory / "result.json")
            assert saved["protocol_sha256"] == digest(out / "protocol.json")
            assert saved["checkpoint_sha256"] == digest(directory / "best.pt")
            results.append(saved)
            continue
        torch.manual_seed(1889)
        model = LeafFeatureHead(kind, 1408).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        best, stale, history = float("inf"), 0, []
        for epoch in range(60):
            if epoch >= 8 and stale >= 6:
                break
            model.train()
            order = torch.multinomial(
                sampling,
                len(train),
                replacement=True,
                generator=torch.Generator().manual_seed(188900 + epoch),
            )
            for batch in order.split(64):
                ids = train[batch.numpy()]
                z = forward(model, ids)
                raw = torch.where(truth[ids] == 1, z.min(1).values, z.max(1).values)
                loss = (
                    F.binary_cross_entropy_with_logits(
                        raw, truth[ids], reduction="none"
                    )
                    * torch.where(truth[ids] == 0, 8.0, 1.0)
                ).mean()
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.inference_mode():
                z = torch.cat(
                    [
                        forward(model, part)
                        for part in np.array_split(val, max(1, (len(val) + 63) // 64))
                    ]
                )
                raw = torch.where(truth[val] == 1, z.min(1).values, z.max(1).values)
                score = float(
                    (
                        F.binary_cross_entropy_with_logits(
                            raw, truth[val], reduction="none"
                        )
                        * torch.where(truth[val] == 0, 8.0, 1.0)
                        * val_weight
                    ).sum()
                )
            if score < best - 1e-5:
                best, stale = score, 0
                atomic_save(
                    dict(
                        kind=kind,
                        model={
                            name: value.detach().cpu().clone()
                            for name, value in model.state_dict().items()
                        },
                        epoch=epoch,
                        protocol_sha256=digest(out / "protocol.json"),
                    ),
                    directory / "best.pt",
                )
            else:
                stale += 1
            history.append(dict(epoch=epoch, validation_loss=score))
        state = torch.load(directory / "best.pt", weights_only=True, map_location="cpu")
        assert state["protocol_sha256"] == digest(out / "protocol.json")
        model.load_state_dict(state["model"])
        model.eval()
        with torch.inference_mode():
            z_val = (
                torch.cat(
                    [
                        forward(model, part)
                        for part in np.array_split(val, max(1, (len(val) + 63) // 64))
                    ]
                )
                .cpu()
                .numpy()
            )
            leaf_logits = (
                torch.cat([model(part) for part in features.split(1024)]).cpu().numpy()
            )
        calibration = calibrate(
            z_val.ravel(),
            np.repeat(np.array([rows[i]["y"] for i in val]), 2),
            np.repeat(vw, 2) / 2.0,
        )
        probabilities = {}
        for path, pair in bags.items():
            if not pair["center"]:
                continue
            mean = np.array(
                [np.mean(leaf_logits[pair[name]]) for name in ("center", "full")]
            )
            probability = 1 / (
                1 + np.exp(-np.clip(mean * calibration[0] + calibration[1], -80, 80))
            )
            probabilities[path] = dict(
                positive=float(probability.min()), negative=float(probability.max())
            )
        negatives = [
            probabilities[rows[i]["path"]]["positive"]
            for i in val
            if rows[i]["y"] == 0 and base_kind[rows[i]["path"]] != "HEALTHY"
        ]
        minimum = max(0.95, max(negatives) + 1e-6) if negatives else 0.95

        def revise(row):
            packet, retained = entries[id(row)]
            if row["path"] not in probabilities:
                return retained
            z = probabilities[row["path"]]
            return route_paired(
                row,
                retained,
                packet,
                z["positive"],
                z["negative"],
                minimum,
                data["manifest"],
            )

        screen = OpenedGuardScreen(data).evaluate(revise)
        screen["checks"].pop("original_disease_and_healthy_exact")
        screen["checks"]["original_disease_exact"] = all(
            revise(r)["app_replay"] == r["app_replay"]
            for values0 in list(data["base"].values()) + [data["internal"]]
            for r in values0
            if r["app_replay"]["kind"] == "DISEASE"
        )
        screen["passed"] = all(screen["checks"].values())
        after = []
        for b, r in zip(before, previous):
            if b["path"] not in probabilities:
                after.append(r)
                continue
            z = probabilities[b["path"]]
            after.append(
                route_paired(
                    b,
                    r,
                    packets[b["id"]],
                    z["positive"],
                    z["negative"],
                    minimum,
                    data["manifest"],
                )
            )
        vi = [i for i, r in enumerate(rows[-len(records) :]) if r["validation"]]
        vr = [records[i] for i in vi]
        vb = [before[i] for i in vi]
        va = [after[i] for i in vi]
        vm = photo_metrics(va)
        all_metrics = photo_metrics(after)
        bm = photo_metrics(before)
        for records0, metrics0 in [(after, all_metrics), (before, bm)]:
            metrics0["unsupported_tomato"] = photo_metrics(
                [r for r in records0 if r["label"] == "Other"]
            )["all"]
        val_safe, val_safety = cohort_safety(vr, vb, va)
        all_safe, all_safety = cohort_safety(records, before, after)
        checks = dict(
            screen["checks"],
            consumed_validation_healthy=vm["healthy"]["correct_rate"] >= 0.7,
            consumed_validation_safety=val_safe,
            consumed_all_safety=all_safe,
            consumed_all_retention=retain({"olid": all_metrics}, {"olid": bm}),
        )
        result = dict(
            kind=kind,
            passed=all(checks.values()),
            checks=checks,
            selected_epoch=state["epoch"],
            threshold=minimum,
            calibration=list(map(float, calibration)),
            old_screen=screen,
            consumed_validation=vm,
            consumed_all=all_metrics,
            validation_safety=val_safety,
            all_safety=all_safety,
            checkpoint_sha256=digest(directory / "best.pt"),
            protocol_sha256=digest(out / "protocol.json"),
            history=history,
            olid_independent=False,
            ad_independent=False,
        )
        np.save(directory / "leaf_logits.npy", leaf_logits)
        (directory / "probabilities.json").write_text(json.dumps(probabilities))
        (directory / "result.json").write_text(json.dumps(result, indent=2))
        results.append(result)
        print(
            json.dumps(
                dict(
                    kind=kind,
                    passed=result["passed"],
                    checks=checks,
                    selected_epoch=state["epoch"],
                    threshold=minimum,
                    validation_healthy=vm["healthy"],
                    all_healthy=all_metrics["healthy"],
                )
            ),
            flush=True,
        )
    result = dict(
        passing=sum(r["passed"] for r in results),
        controls=results,
        source_sha256=digest(__file__),
        feature_cache=feature_metadata,
        elapsed_seconds=time.monotonic() - started,
        olid_independent=False,
        ad_independent=False,
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
