"""Check actual isolated-route eligibility before rejecting cached heads.

No weights or encoder outputs are recomputed. OLID is consumed development.
One validation-fitted operating point per already completed head, maximum four.
"""

from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import torch
from torch import nn

from opened_context_data import load, read
from opened_guard_screen import OpenedGuardScreen
from permissive_encoder import digest
from phase22_accept import paired_interval
from phase2_select import retain
from phase5_scene_ablation import photo_metrics
from phase80_cached_isolated_head import route_isolated, vector
from phase81_consumed_olid_features import prepare


def eligible(row, packet):
    v = row["app_replay"]
    if v["kind"] in ("DISEASE", "HEALTHY") or packet["whole"][3] >= 0.5:
        return False
    if v["kind"] == "UNSURE" and v.get("label") not in (None, "", "Healthy", "Other"):
        return False
    p = v.get("probabilities") or packet.get("fallback")
    if p is None or (p[8] <= 0.1 and sum(p[i] for i in (0, 2, 3, 4, 5, 6, 7)) >= 0.6):
        return False
    positive = min((view[0] for leaf in packet["leaves"] for view in leaf), default=1.0)
    negative = min(
        (max(view[1] for view in leaf) for leaf in packet["leaves"]), default=1.0
    )
    return (
        packet["usable"]
        and packet["whole"][0] >= 0.85
        and packet["whole"][2] >= 0.5
        and positive >= 0.1
        and negative > 0.001
    )


def safety(records, before, after):
    audits = {}
    for raw in [None] + sorted(
        {r["raw_label"] for r in records if r["label"] != "Healthy"}
    ):
        audit = paired_interval(
            records,
            before,
            after,
            lambda r: r["label"] != "Healthy"
            and (raw is None or r["raw_label"] == raw),
            lambda r: int(r["app_replay"]["kind"] == "HEALTHY"),
        )
        if audit is not None:
            audits[raw or "all_diseased"] = audit
    passed = all(
        v["point"] <= 0.0 and v["bootstrap_one_sided_upper_95"] <= 0.02
        for v in audits.values()
        if v["n"] >= 50
    )
    return passed, audits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    args = ap.parse_args()
    p = Path(args.campaign)
    out = p / "phase83_eligible_operating_points"
    out.mkdir(exist_ok=True)
    assert read(p / "phase80_cached_isolated_head/result.json")["passing"] == 0
    assert read(p / "phase82_frozen_isolated_head/result.json")["passing"] == 0
    data = load(p)
    features, feature_result = prepare(p, data)
    cases = read(p / "phase77_native_temporal/temporal_cases.json")["cases"]
    original = read(p / "phase73_mobile_verified/reference.json")
    entries = {}
    for case in cases:
        name, index = case["id"].rsplit(":", 1)
        index = int(index)
        row = (
            data["internal"][index] if name == "internal" else data["base"][name][index]
        )
        posterior = case.get("posterior") or case.get("fallback")
        if posterior is not None:
            entries[id(row)] = dict(
                row=row,
                packet=case,
                posterior=posterior,
                retained=dict(row, app_replay=original[name][index]),
            )
    records = read(p / "phase24_olid_source/sealed_manifest.json")
    packets = {
        r["id"]: r["candidate"]
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
    before = []
    previous = []
    labels = data["manifest"]["classifier"]["labels"]
    for row in records:
        packet = packets[row["id"]]
        observations = packet["observations"]
        frames = native[row["id"]]["baseline"]["frames"]
        if observations:
            weights = np.minimum(
                np.array([o[0] * packet["quality_weight"] for o in observations]).clip(
                    0.05, 1.0
                )
                * frames,
                3.0,
            )
            posterior = np.average(
                np.array([o[1] for o in observations]), axis=0, weights=weights
            ).tolist()
        else:
            posterior = packet["fallback"]
        for key, target in [("baseline", before), ("candidate", previous)]:
            v = dict(native[row["id"]][key])
            v["label"] = (
                labels[v["label_index"]] if v["label_index"] is not None else None
            )
            v["probabilities"] = posterior
            target.append(
                dict(
                    row,
                    app_replay=v,
                    contrast=1.0 if packet["usable"] else 0.0,
                    sharpness=1.0 if packet["usable"] else 0.0,
                    context_whole_reference_probabilities=packet["fallback"],
                )
            )
    validation = np.array(
        [
            int(
                hashlib.sha256(("phase80:1807:" + r["group"]).encode()).hexdigest()[:8],
                16,
            )
            % 4
            == 0
            for r in records
        ]
    )
    val_ids = np.flatnonzero(validation)
    controls = [
        ("scores", "phase80_cached_isolated_head", k) for k in ("linear", "mlp32")
    ] + [("visual", "phase82_frozen_isolated_head", k) for k in ("linear", "mlp128")]
    protocol = dict(
        script_sha256=digest(__file__),
        route_sha256=digest(
            Path(__file__).with_name("phase80_cached_isolated_head.py")
        ),
        hypothesis="Validation negatives that cannot enter isolated recovery must not determine its positive cutoff",
        controls=[
            dict(
                family=f,
                kind=k,
                checkpoint_sha256=digest(p / folder / k / "best.pt"),
                result_sha256=digest(p / folder / k / "result.json"),
            )
            for f, folder, k in controls
        ],
        cutoff="max(.95, maximum eligible consumed-OLID validation-negative probability + 1e-6)",
        eligibility="Exact input-based proposed isolated route, including original disease/suspected disease and closed-set guards",
        maximum_policies=4,
        new_encoder_inputs=0,
        weights_changed=False,
        olid_independent=False,
        ad_independent=False,
        all_original_gates_unchanged=True,
        stopping="No additional thresholds if these four operating points fail",
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
    results = []
    old_ids = list(entries)
    for family, folder, kind in controls:
        directory = p / folder / kind
        fitted = read(directory / "result.json")
        state = torch.load(directory / "best.pt", weights_only=True, map_location="cpu")
        dimension = 23 if family == "scores" else 768
        hidden = 32 if family == "scores" else 128
        model = (
            nn.Linear(dimension, 1)
            if kind == "linear"
            else nn.Sequential(
                nn.Linear(dimension, hidden), nn.SiLU(), nn.Linear(hidden, 1)
            )
        )
        model.load_state_dict(state["model"])
        model.eval()
        old_x = np.stack(
            [
                (
                    vector(entries[key]["packet"], entries[key]["posterior"])
                    if family == "scores"
                    else features[entries[key]["row"]["path"]]
                )
                for key in old_ids
            ]
        )
        with torch.inference_mode():
            logits = (
                model(
                    torch.from_numpy(
                        (old_x - np.array(state["mean"], np.float32))
                        / np.array(state["std"], np.float32)
                    )
                )
                .ravel()
                .numpy()
            )
        calibration = fitted["calibration"]
        old_probability = 1 / (
            1 + np.exp(-np.clip(logits * calibration[0] + calibration[1], -80, 80))
        )
        probability_by_object = dict(zip(old_ids, map(float, old_probability)))
        probability = np.load(directory / "probabilities.npy")[-len(records) :]
        allowed = np.array([eligible(b, packets[b["id"]]) for b in before])
        validation_negative = (
            validation & allowed & np.array([r["label"] != "Healthy" for r in records])
        )
        minimum = (
            max(0.95, float(probability[validation_negative].max()) + 1e-6)
            if validation_negative.any()
            else 0.95
        )

        def revise(row):
            item = entries.get(id(row))
            if item is None:
                return row
            return route_isolated(
                row,
                item["retained"],
                item["packet"],
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
        screen["passed"] = all(screen["checks"].values())
        after = [
            route_isolated(
                b, r, packets[b["id"]], float(probability[i]), minimum, data["manifest"]
            )
            for i, (b, r) in enumerate(zip(before, previous))
        ]
        vr = [records[i] for i in val_ids]
        vb = [before[i] for i in val_ids]
        va = [after[i] for i in val_ids]
        vm = photo_metrics(va)
        all_metrics = photo_metrics(after)
        bm = photo_metrics(before)
        for rows, metrics in [(after, all_metrics), (before, bm)]:
            metrics["unsupported_tomato"] = photo_metrics(
                [r for r in rows if r["label"] == "Other"]
            )["all"]
        val_safe, val_safety = safety(vr, vb, va)
        all_safe, all_safety = safety(records, before, after)
        checks = dict(
            screen["checks"],
            consumed_val_healthy=vm["healthy"]["correct_rate"] >= 0.7,
            consumed_val_safety=val_safe,
            consumed_all_safety=all_safe,
            consumed_all_retention=retain({"olid": all_metrics}, {"olid": bm}),
        )
        result = dict(
            family=family,
            kind=kind,
            passed=all(checks.values()),
            checks=checks,
            threshold=minimum,
            calibration=fitted["calibration"],
            checkpoint_sha256=digest(directory / "best.pt"),
            eligible_validation_negatives=int(validation_negative.sum()),
            consumed_validation=vm,
            consumed_all=all_metrics,
            validation_safety=val_safety,
            all_safety=all_safety,
            old_screen=screen,
            olid_independent=False,
            ad_independent=False,
        )
        results.append(result)
        print(
            json.dumps(
                dict(
                    family=family,
                    kind=kind,
                    passed=result["passed"],
                    checks=checks,
                    threshold=minimum,
                    validation_healthy=vm["healthy"],
                    all_healthy=all_metrics["healthy"],
                    eligible_validation_negatives=int(validation_negative.sum()),
                )
            ),
            flush=True,
        )
    (out / "result.json").write_text(
        json.dumps(
            dict(
                protocol=protocol,
                controls=results,
                passing=sum(r["passed"] for r in results),
                new_encoder_inputs=0,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
