"""Bounded supervised isolated-view expert using frozen visual features.

OLID is consumed development after phase76 failed. No acceptance claim is made.
Two heads maximum, unchanged family split and recognition gates. Only new images receive encoder inference.
"""

from pathlib import Path
import argparse
import copy
import hashlib
import json
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from campaign_cnn_fusion import calibrate
from context_guard import route_guard
from opened_context_data import load, read
from phase81_consumed_olid_features import prepare
from opened_guard_screen import OpenedGuardScreen
from permissive_encoder import digest
from phase22_accept import paired_interval
from phase5_scene_ablation import photo_metrics
from source_balance import source_class_group_weights
from training_support import atomic_save


def route_isolated(row, retained, packet, probability, minimum, manifest):
    # Field recovery and all original named disease decisions remain untouched.
    if packet["whole"][3] >= 0.5 or row["app_replay"]["kind"] == "DISEASE":
        return retained
    positive_leaf = min((v[0] for leaf in packet["leaves"] for v in leaf), default=1.0)
    negative_leaf = min(
        (max(v[1] for v in leaf) for leaf in packet["leaves"]), default=1.0
    )
    hp, _, genus, _ = packet["whole"]
    positive = (
        probability >= minimum
        and hp >= 0.85
        and genus >= 0.5
        and positive_leaf >= 0.1
        and negative_leaf > 0.001
    )
    # The isolated expert is an additional supervised view, not a lower context
    # or DINO threshold. It can abstain from Healthy at a fixed .001 cutoff.
    revised = route_guard(
        retained,
        manifest,
        probability if positive else 0.0,
        probability if genus >= 0.5 else 1.0,
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
    return revised


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    args = ap.parse_args()
    p = Path(args.campaign)
    out = p / "phase82_frozen_isolated_head"
    out.mkdir(exist_ok=True)
    assert not read(p / "phase76_fresh_acceptance/result.json")["passed"]
    data = load(p)
    features, feature_result = prepare(p, data)
    old_cases = read(p / "phase77_native_temporal/temporal_cases.json")["cases"]
    old_reference = read(p / "phase73_mobile_verified/reference.json")
    by_path, by_object, fitting = {}, {}, []
    for case in old_cases:
        name, index = case["id"].rsplit(":", 1)
        index = int(index)
        row = (
            data["internal"][index] if name == "internal" else data["base"][name][index]
        )
        packet = dict(case, fallback=case.get("fallback"))
        posterior = case.get("posterior") or case.get("fallback")
        if posterior is None:
            continue
        by_object[id(row)] = (
            packet,
            row["path"],
            dict(row, app_replay=old_reference[name][index]),
        )
        if row["label"] == "Mixed":
            continue
        if row["path"] in by_path:
            continue
        record = dict(
            row,
            y=int(row["label"] == "Healthy"),
            x=np.array(features[row["path"]], copy=True),
            packet=packet,
        )
        by_path[row["path"]] = len(fitting)
        fitting.append(record)
    olid = read(p / "phase24_olid_source/sealed_manifest.json")
    score_rows = {
        r["id"]: r
        for r in map(
            json.loads,
            (p / "phase76_fresh_acceptance/mobile_scores.jsonl")
            .read_text()
            .splitlines(),
        )
    }
    native = {
        r["id"]: r
        for r in read(p / "phase76_fresh_acceptance/native_predictions.json")["cases"]
    }
    olid_start = len(fitting)
    labels = data["manifest"]["classifier"]["labels"]
    before, retained = [], []
    for row in olid:
        packet = score_rows[row["id"]]["candidate"]
        original = native[row["id"]]["baseline"]
        frames = original["frames"]
        observations = packet["observations"]
        if observations:
            weights = np.minimum(
                np.array([x[0] * packet["quality_weight"] for x in observations]).clip(
                    0.05, 1.0
                )
                * frames,
                3.0,
            )
            posterior = np.average(
                np.array([x[1] for x in observations]), axis=0, weights=weights
            ).tolist()
        else:
            posterior = packet["fallback"]
        versions = []
        for key in ("baseline", "candidate"):
            v = dict(native[row["id"]][key])
            v["label"] = (
                labels[v["label_index"]] if v["label_index"] is not None else None
            )
            v["probabilities"] = posterior
            versions.append(
                dict(
                    row,
                    app_replay=v,
                    contrast=1.0 if packet["usable"] else 0.0,
                    sharpness=1.0 if packet["usable"] else 0.0,
                    context_whole_reference_probabilities=packet["fallback"],
                )
            )
        before.append(versions[0])
        retained.append(versions[1])
        record = dict(
            row,
            y=int(row["label"] == "Healthy"),
            x=np.array(features[row["path"]], copy=True),
            packet=packet,
        )
        by_path[row["path"]] = len(fitting)
        fitting.append(record)
    # Family membership is determined before fitting. Source names and identities
    # are used only for splitting/balancing, never supplied to either model.
    family = [str(r.get("group") or r.get("file_sha256") or r["path"]) for r in fitting]
    validation = np.array(
        [
            int(hashlib.sha256(("phase80:1807:" + group).encode()).hexdigest()[:8], 16)
            % 4
            == 0
            for group in family
        ]
    )
    train, val = np.flatnonzero(~validation), np.flatnonzero(validation)
    assert not ({family[i] for i in train} & {family[i] for i in val})
    assert not ({fitting[i]["path"] for i in train} & {fitting[i]["path"] for i in val})
    assert all("x" in r and "source" in r for r in fitting)
    for row, group in zip(fitting, family):
        row["group"] = group
    protocol = dict(
        hypothesis="A supervised original-SigLIP visual expert repairs isolated Healthy rejection after both score-only heads failed",
        data_role="AD and OLID consumed development; validation families are development only",
        source_mobile_scores_sha256=digest(
            p / "phase76_fresh_acceptance/mobile_scores.jsonl"
        ),
        original_cases_sha256=digest(p / "phase77_native_temporal/temporal_cases.json"),
        original_candidate_sha256=digest(p / "phase73_mobile_verified/result.json"),
        script_sha256=digest(__file__),
        heads=["linear", "mlp128"],
        seed=1807,
        split_sha256=hashlib.sha256(
            json.dumps(
                list(zip([r["path"] for r in fitting], family, validation.tolist()))
            ).encode()
        ).hexdigest(),
        features="Original frozen SigLIP768 on the exact canonical whole letterbox",
        feature_cache=feature_result,
        maximum_epochs=80,
        minimum_epochs=12,
        patience=8,
        learning_rate=0.001,
        negative_loss_weight=8.0,
        calibration="Source/class/family weighted validation BCE",
        positive_cutoff="max(.95, maximum validation diseased probability + 1e-6)",
        negative_cutoff=0.001,
        encoder_weights_changed=False,
        new_encoder_inputs=feature_result["newly_computed"],
        original_thresholds_unchanged=True,
        ad_independent=False,
        olid_independent=False,
        stopping_rule="Stop both frozen-visual heads if neither passes every old gate and consumed OLID family-validation Healthy>=.70 with paired safety; only then consider a copied final-block supervised adaptation",
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
    torch.manual_seed(1807)
    x = np.stack([r["x"] for r in fitting])
    y = np.array([r["y"] for r in fitting], np.float32)
    mean = x[train].mean(0)
    std = np.maximum(x[train].std(0), 0.1)
    values = torch.from_numpy((x - mean) / std)
    truth = torch.from_numpy(y)
    sampling = torch.from_numpy(source_class_group_weights(fitting, train))
    vw = source_class_group_weights(fitting, val)
    weights = torch.tensor(vw, dtype=torch.float32)
    results = []
    started = time.monotonic()
    for kind in protocol["heads"]:
        directory = out / kind
        directory.mkdir(exist_ok=True)
        torch.manual_seed(1807)
        model = (
            nn.Linear(768, 1)
            if kind == "linear"
            else nn.Sequential(nn.Linear(768, 128), nn.SiLU(), nn.Linear(128, 1))
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        best, stale, history = float("inf"), 0, []
        for epoch in range(80):
            if epoch >= 12 and stale >= 8:
                break
            model.train()
            order = torch.multinomial(
                sampling,
                len(train),
                replacement=True,
                generator=torch.Generator().manual_seed(180700 + epoch),
            )
            for batch in order.split(256):
                ids = train[batch.numpy()]
                loss = (
                    F.binary_cross_entropy_with_logits(
                        model(values[ids]).ravel(), truth[ids], reduction="none"
                    )
                    * torch.where(truth[ids] == 0, 8.0, 1.0)
                ).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.inference_mode():
                z = model(values[val]).ravel()
                score = float(
                    (
                        F.binary_cross_entropy_with_logits(
                            z, truth[val], reduction="none"
                        )
                        * torch.where(truth[val] == 0, 8.0, 1.0)
                        * weights
                    ).sum()
                )
            if score < best - 1e-5:
                best, stale = score, 0
                atomic_save(
                    dict(
                        kind=kind,
                        model=copy.deepcopy(model.state_dict()),
                        epoch=epoch,
                        mean=mean.tolist(),
                        std=std.tolist(),
                        protocol_sha256=digest(frozen),
                    ),
                    directory / "best.pt",
                )
            else:
                stale += 1
            history.append(dict(epoch=epoch, validation_bce=score))
        state = torch.load(directory / "best.pt", weights_only=True, map_location="cpu")
        model.load_state_dict(state["model"])
        model.eval()
        with torch.inference_mode():
            raw = model(values).ravel().numpy()
        calibration = calibrate(raw[val], y[val], vw)
        probabilities = 1 / (
            1 + np.exp(-np.clip(raw * calibration[0] + calibration[1], -80, 80))
        )
        eval_ids = list(by_object)
        eval_vectors = np.stack([features[by_object[key][1]] for key in eval_ids])
        with torch.inference_mode():
            eval_logits = (
                model(torch.from_numpy((eval_vectors - mean) / std)).ravel().numpy()
            )
        eval_probabilities = 1 / (
            1 + np.exp(-np.clip(eval_logits * calibration[0] + calibration[1], -80, 80))
        )
        probability_by_object = dict(zip(eval_ids, map(float, eval_probabilities)))
        minimum = max(0.95, float(probabilities[val][y[val] == 0].max()) + 1e-6)

        def revise(row):
            item = by_object.get(id(row))
            if item is None:
                return row
            packet, path, previous = item
            return route_isolated(
                row,
                previous,
                packet,
                probability_by_object[id(row)],
                minimum,
                data["manifest"],
            )

        screen = OpenedGuardScreen(data).evaluate(revise)
        screen["checks"].pop("original_disease_and_healthy_exact")
        screen["checks"]["original_disease_exact"] = all(
            revise(r)["app_replay"] == r["app_replay"]
            for rows in list(data["base"].values()) + [data["internal"]]
            for r in rows
            if r["app_replay"]["kind"] == "DISEASE"
        )
        after = [
            route_isolated(
                b,
                r,
                score_rows[b["id"]]["candidate"],
                float(probabilities[olid_start + i]),
                minimum,
                data["manifest"],
            )
            for i, (b, r) in enumerate(zip(before, retained))
        ]
        val_ids = [i for i in range(len(olid)) if validation[olid_start + i]]
        vr = [olid[i] for i in val_ids]
        vb = [before[i] for i in val_ids]
        va = [after[i] for i in val_ids]
        metrics = photo_metrics(va)
        safety = paired_interval(
            vr,
            vb,
            va,
            lambda r: r["label"] != "Healthy",
            lambda r: int(r["app_replay"]["kind"] == "HEALTHY"),
        )
        checks = dict(
            screen["checks"],
            olid_development_val_healthy=metrics["healthy"]["correct_rate"] >= 0.7,
            olid_development_val_safety=safety["point"] <= 0.0
            and safety["bootstrap_one_sided_upper_95"] <= 0.02,
        )
        result = dict(
            kind=kind,
            passed=all(checks.values()),
            checks=checks,
            threshold=minimum,
            calibration=list(map(float, calibration)),
            best_epoch=state["epoch"],
            original_screen=screen,
            consumed_olid_validation=metrics,
            validation_safety=safety,
            consumed_olid_all=photo_metrics(after),
            checkpoint_sha256=digest(directory / "best.pt"),
            history=history,
            olid_validation_n=len(val_ids),
            olid_independent=False,
            ad_independent=False,
        )
        (directory / "result.json").write_text(json.dumps(result, indent=2))
        np.save(directory / "probabilities.npy", probabilities)
        results.append(result)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "kind",
                        "passed",
                        "checks",
                        "threshold",
                        "best_epoch",
                        "olid_validation_n",
                    )
                }
            ),
            flush=True,
        )
    result = dict(
        protocol=protocol,
        controls=results,
        passing=sum(r["passed"] for r in results),
        elapsed_seconds=time.monotonic() - started,
        new_encoder_inputs=feature_result["newly_computed"],
        olid_independent=False,
        ad_independent=False,
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
