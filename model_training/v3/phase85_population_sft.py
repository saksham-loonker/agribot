"""One bounded copied-block supervised control after cached heads failed.

The original disease classifier and existing field path stay frozen. Only a
copied original SigLIP suffix/pool and its isolated-view head are fitted.
"""

from pathlib import Path
import argparse
import copy
import hashlib
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
import timm

from campaign_cnn_fusion import calibrate
from opened_context_data import load, read
from opened_guard_screen import OpenedGuardScreen
from partial_primary_siglip import prefix_identity
from permissive_encoder import CanonicalPermissiveEncoder, digest
from phase2_select import retain
from phase5_scene_ablation import photo_metrics
from phase81_consumed_olid_features import prepare as prepare_features
from phase83_eligible_operating_points import safety as cohort_safety
from phase84_population_head import population, route_isolated
from population_siglip_tail import PopulationSiglipTail, freeze, prepare_prefix
from source_balance import source_class_group_weights
from training_support import atomic_save


def evaluation_data(p, data):
    reference = read(p / "phase73_mobile_verified/reference.json")
    entries = {}
    for case in read(p / "phase77_native_temporal/temporal_cases.json")["cases"]:
        name, index = case["id"].rsplit(":", 1)
        index = int(index)
        row = (
            data["internal"][index] if name == "internal" else data["base"][name][index]
        )
        entries[id(row)] = (case, dict(row, app_replay=reference[name][index]))
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
    labels = data["manifest"]["classifier"]["labels"]
    before = []
    previous = []
    for row in records:
        packet = packets[row["id"]]
        obs = packet["observations"]
        frames = native[row["id"]]["baseline"]["frames"]
        posterior = packet["fallback"]
        if obs:
            weights = np.minimum(
                np.array([o[0] * packet["quality_weight"] for o in obs]).clip(0.05, 1.0)
                * frames,
                3.0,
            )
            posterior = np.average(
                np.array([o[1] for o in obs]), axis=0, weights=weights
            ).tolist()
        for key, target in [("baseline", before), ("candidate", previous)]:
            verdict = dict(native[row["id"]][key])
            index = verdict["label_index"]
            verdict.update(
                label=labels[index] if index is not None else None,
                probabilities=posterior,
            )
            target.append(
                dict(
                    row,
                    app_replay=verdict,
                    contrast=1.0 if packet["usable"] else 0.0,
                    sharpness=1.0 if packet["usable"] else 0.0,
                    context_whole_reference_probabilities=packet["fallback"],
                )
            )
    return entries, records, packets, before, previous


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    args = ap.parse_args()
    p = Path(args.campaign)
    out = p / "phase85_population_sft"
    out.mkdir(exist_ok=True)
    assert read(p / "phase84_population_head/result.json")["passing"] == 0
    assert read(p / "phase83_eligible_operating_points/result.json")["passing"] == 0
    if (out / "result.json").exists():
        result = read(out / "result.json")
        assert result["source_sha256"] == digest(__file__)
        print(
            json.dumps(
                {key: result[key] for key in ("passed", "checks", "selected_epoch")}
            ),
            flush=True,
        )
        return
    data = load(p)
    features, feature_metadata = prepare_features(p, data)
    entries, records, packets, before, previous = evaluation_data(p, data)
    paths = {
        row["path"]
        for rows in list(data["base"].values()) + [data["internal"]]
        for row in rows
        if population(row, entries[id(row)][0])
    }
    paths.update(b["path"] for b in before if population(b, packets[b["id"]]))
    fitting = read(p / "phase84_population_head/fitting_rows.json")
    active = [i for i, r in enumerate(fitting) if r["active"]]
    assert all(fitting[i]["path"] in paths for i in active)
    train = np.array([i for i in active if not fitting[i]["validation"]])
    val = np.array([i for i in active if fitting[i]["validation"]])
    assert not (
        {fitting[i]["group"] for i in train} & {fitting[i]["group"] for i in val}
    )
    inputs = {r["path"]: r for r in read(p / "opened_siglip_feature_cache/inputs.json")}
    inputs.update(
        {r["path"]: r for r in read(p / "phase81_consumed_olid_features/inputs.json")}
    )
    assert not (
        {inputs[fitting[i]["path"]]["input_sha256"] for i in train}
        & {inputs[fitting[i]["path"]]["input_sha256"] for i in val}
    ), "Exact-input overlap crosses training/validation; repair grouping before training"
    torch.set_num_threads(4)
    torch.manual_seed(1853)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timm.layers.set_fused_attn(False)
    encoder_path = p / "phase23_siglip/siglip2_feature_heads/encoder.pt"
    state = torch.load(encoder_path, weights_only=True, map_location="cpu")
    encoder = CanonicalPermissiveEncoder(state["spec"])
    encoder.load_state_dict(state["model"], strict=True)
    warm_path = p / "phase84_population_head/mlp128/best.pt"
    warm = torch.load(warm_path, weights_only=True, map_location="cpu")
    model = PopulationSiglipTail(encoder, warm["mean"], warm["std"]).cuda().eval()
    model.head.load_state_dict(warm["model"], strict=True)
    del encoder, state
    original_prefix = prefix_identity(model.encoder)
    tokens, lookup, prefix_metadata = prepare_prefix(p, model, paths, features)
    token_ids = np.array([lookup[r["path"]] if r["active"] else -1 for r in fitting])
    assert (token_ids[train] >= 0).all() and (token_ids[val] >= 0).all()
    assert not (set(token_ids[train]) & set(token_ids[val]))
    protocol = dict(
        hypothesis="Adapt copied final two blocks and pooling to the actual isolated Healthy-versus-damage population after score and frozen visual heads failed",
        encoder_sha256=digest(encoder_path),
        warm_head_sha256=digest(warm_path),
        fitting_rows_sha256=digest(p / "phase84_population_head/fitting_rows.json"),
        prefix_cache=prefix_metadata,
        prefix_state_sha256=original_prefix,
        source_sha256=digest(__file__),
        model_source_sha256=digest(
            Path(__file__).with_name("population_siglip_tail.py")
        ),
        routing_source_sha256=digest(
            Path(__file__).with_name("phase84_population_head.py")
        ),
        epochs=8,
        minimum_epochs=4,
        patience=2,
        batch=32,
        tail_lr=1e-5,
        head_lr=1e-4,
        negative_loss_weight=8.0,
        ema_decay=0.99,
        weight_decay=0.01,
        gradient_clip=1.0,
        seed=1853,
        selection="One EMA checkpoint minimizing source/class/family balanced validation BCE with negative cost eight",
        positive_cutoff="max(.95, maximum eligible validation negative for a new Healthy proposal + 1e-6)",
        negative_cutoff=0.001,
        original_disease_classifier_unchanged=True,
        existing_field_path_unchanged=True,
        scope="Publisher primary isolated view, not whole-plant clinical health",
        ad_independent=False,
        olid_independent=False,
        stopping="At most eight epochs, stop after two stale validation epochs after epoch four. Stop this control if any unchanged recognition gate fails; no export or Android integration before passing cheap checks.",
    )
    freeze(out / "protocol.json", protocol)
    allowed = set(model.tail_state())
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [
            dict(
                params=[v for v in model.encoder.parameters() if v.requires_grad],
                lr=1e-5,
            ),
            dict(params=model.head.parameters(), lr=1e-4),
        ],
        weight_decay=0.01,
    )
    truth = torch.tensor([r["y"] for r in fitting], dtype=torch.float32, device="cuda")
    sampling = torch.from_numpy(source_class_group_weights(fitting, train))
    val_weights = source_class_group_weights(fitting, val)
    weights = torch.tensor(val_weights, dtype=torch.float32, device="cuda")
    start, best, stale, history = 0, float("inf"), 0, []
    if (out / "last.pt").exists():
        saved = torch.load(out / "last.pt", weights_only=True, map_location="cuda")
        assert saved["protocol_sha256"] == digest(out / "protocol.json")
        model.load_tail(saved["tail"])
        ema.load_state_dict(saved["ema_tail"], strict=False)
        optimizer.load_state_dict(saved["optimizer"])
        start, best, stale, history = (
            saved["epoch"] + 1,
            saved["best"],
            saved["stale"],
            saved["history"],
        )
        torch.set_rng_state(saved["torch_rng"].cpu())
        torch.cuda.set_rng_state(saved["cuda_rng"].cpu())

    def tail(module):
        return {
            name: value.detach().cpu().clone()
            for name, value in module.named_parameters()
            if name in allowed
        }

    def logits(module, ids):
        with torch.inference_mode():
            return torch.cat(
                [
                    module(tokens[part])
                    for part in np.array_split(ids, max(1, (len(ids) + 31) // 32))
                ]
            )

    started = time.monotonic()
    for epoch in range(start, 8):
        if epoch >= 4 and stale >= 2:
            break
        model.train()
        order = torch.multinomial(
            sampling,
            len(train),
            replacement=True,
            generator=torch.Generator().manual_seed(185300 + epoch),
        )
        moving = dict(ema.named_parameters())
        for positions in order.split(32):
            indices = train[positions.numpy()]
            z = model(tokens[token_ids[indices]])
            loss = (
                F.binary_cross_entropy_with_logits(z, truth[indices], reduction="none")
                * torch.where(truth[indices] == 0, 8.0, 1.0)
            ).mean()
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [v for v in model.parameters() if v.requires_grad],
                1.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            with torch.no_grad():
                for name, value in model.named_parameters():
                    if name in allowed:
                        moving[name].mul_(0.99).add_(value, alpha=0.01)
        ema.eval()
        z = logits(ema, token_ids[val])
        score = float(
            (
                F.binary_cross_entropy_with_logits(z, truth[val], reduction="none")
                * torch.where(truth[val] == 0, 8.0, 1.0)
                * weights
            ).sum()
        )
        if score < best - 1e-5:
            best, stale = score, 0
            atomic_save(
                dict(
                    tail=tail(ema),
                    epoch=epoch,
                    protocol_sha256=digest(out / "protocol.json"),
                    mean=warm["mean"],
                    std=warm["std"],
                ),
                out / "best.pt",
            )
        else:
            stale += 1
        history.append(
            dict(
                epoch=epoch,
                validation_loss=score,
                elapsed_seconds=time.monotonic() - started,
            )
        )
        atomic_save(
            dict(
                tail=tail(model),
                ema_tail=tail(ema),
                optimizer=optimizer.state_dict(),
                epoch=epoch,
                best=best,
                stale=stale,
                history=history,
                protocol_sha256=digest(out / "protocol.json"),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(),
            ),
            out / "last.pt",
        )
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(history[-1]), flush=True)
    assert (
        prefix_identity(model.encoder)
        == prefix_identity(ema.encoder)
        == original_prefix
    )
    selected = torch.load(out / "best.pt", weights_only=True, map_location="cpu")
    model.load_tail(selected["tail"])
    model.eval()
    raw = logits(model, np.arange(len(tokens))).cpu().numpy()
    raw_val = raw[token_ids[val]]
    calibration = calibrate(
        raw_val, np.array([r["y"] for r in fitting])[val], val_weights
    )
    probabilities = 1 / (
        1 + np.exp(-np.clip(raw * calibration[0] + calibration[1], -80, 80))
    )
    path_probability = {
        path: float(probabilities[index]) for path, index in lookup.items()
    }
    base_kind = {
        row["path"]: row["app_replay"]["kind"]
        for rows in list(data["base"].values()) + [data["internal"]]
        for row in rows
    }
    base_kind.update({b["path"]: b["app_replay"]["kind"] for b in before})
    cut_values = [
        path_probability[fitting[i]["path"]]
        for i in val
        if fitting[i]["y"] == 0 and base_kind[fitting[i]["path"]] != "HEALTHY"
    ]
    minimum = max(0.95, max(cut_values) + 1e-6) if cut_values else 0.95

    def revise(row):
        packet, retained = entries[id(row)]
        if not population(row, packet):
            return retained
        return route_isolated(
            row,
            retained,
            packet,
            path_probability[row["path"]],
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
        (
            route_isolated(
                b,
                r,
                packets[b["id"]],
                path_probability[b["path"]],
                minimum,
                data["manifest"],
            )
            if population(b, packets[b["id"]])
            else r
        )
        for b, r in zip(before, previous)
    ]
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
    vi = np.flatnonzero(validation)
    vr = [records[i] for i in vi]
    vb = [before[i] for i in vi]
    va = [after[i] for i in vi]
    val_metrics = photo_metrics(va)
    all_metrics = photo_metrics(after)
    baseline_metrics = photo_metrics(before)
    for rows, metrics in [(after, all_metrics), (before, baseline_metrics)]:
        metrics["unsupported_tomato"] = photo_metrics(
            [r for r in rows if r["label"] == "Other"]
        )["all"]
    val_safe, val_safety = cohort_safety(vr, vb, va)
    all_safe, all_safety = cohort_safety(records, before, after)
    checks = dict(
        screen["checks"],
        consumed_validation_healthy=val_metrics["healthy"]["correct_rate"] >= 0.7,
        consumed_validation_safety=val_safe,
        consumed_all_safety=all_safe,
        consumed_all_retention=retain(
            {"olid": all_metrics}, {"olid": baseline_metrics}
        ),
    )
    result = dict(
        passed=all(checks.values()),
        checks=checks,
        selected_epoch=selected["epoch"],
        threshold=minimum,
        calibration=list(map(float, calibration)),
        checkpoint_sha256=digest(out / "best.pt"),
        protocol_sha256=digest(out / "protocol.json"),
        source_sha256=digest(__file__),
        original_screen=screen,
        consumed_validation=val_metrics,
        consumed_all=all_metrics,
        validation_safety=val_safety,
        all_safety=all_safety,
        prefix_cache=prefix_metadata,
        elapsed_seconds=time.monotonic() - started,
        olid_independent=False,
        ad_independent=False,
    )
    (out / "probabilities.json").write_text(json.dumps(path_probability))
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "passed",
                    "checks",
                    "selected_epoch",
                    "threshold",
                    "elapsed_seconds",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
