"""One disease-aware supervised adaptation of the original DINO's last blocks."""

from pathlib import Path
import argparse
import collections
import concurrent.futures
import copy
import json
import shutil
import time
import numpy as np
import torch
from torch.nn import functional as F

from app_eval import LABELS
from campaign_cnn_fusion import calibrate
from opened_context_data import read
from partial_dino_cache import CanonicalInputs, input_rows, original_dino, prepare
from partial_dino_healthy import (
    PartialDinoHealthyTail,
    healthy_margin,
    prefix_features,
    prefix_identity,
)
from permissive_encoder import digest
from phase49_partial_cnn_train import cpu_state
from source_balance import source_class_group_weights
from training_support import atomic_save
from healthy_tail_risk import weighted_upper_tail


def condition_target(source_row, healthy, include_explicit_other=False):
    if healthy:
        return 1
    if source_row is None:
        return -1
    label = source_row["label"]
    if label in LABELS[:8] and label != "Healthy":
        return LABELS.index(label)
    raw = source_row.get("raw_label")
    if (
        include_explicit_other
        and label == "Other"
        and isinstance(raw, str)
        and raw.strip()
        and raw.strip().casefold()
        not in ("diseased_unspecified", "diseased", "unhealthy", "unknown")
    ):
        return 8
    return -1


def named_targets(campaign, fitting, descriptors, include_explicit_other=False):
    source = read(Path(campaign) / "phase10_leaf/cnn_inputs/rows.json")
    labels = []
    for row in fitting:
        descriptor = descriptors[row["table"]]
        source_row = (
            source[descriptor["index"]] if descriptor["kind"] == "old" else None
        )
        labels.append(
            condition_target(source_row, row["y"] == 1, include_explicit_other)
        )
    grouped = collections.defaultdict(set)
    for row, target in zip(fitting, labels):
        if target >= 0:
            grouped[row["table"]].add(target)
    return np.array(
        [
            target if len(grouped[row["table"]]) <= 1 else -1
            for row, target in zip(fitting, labels)
        ],
        np.int64,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--continue-from-phase51", action="store_true")
    parser.add_argument("--risk-control", action="store_true")
    parser.add_argument("--explicit-other-control", action="store_true")
    args = parser.parse_args()
    assert not (args.risk_control and args.continue_from_phase51)
    assert not args.explicit_other_control or not (
        args.risk_control or args.continue_from_phase51
    )
    sparse_evaluation = args.risk_control or args.explicit_other_control
    campaign = Path(args.campaign)
    cache = prepare(campaign)
    if args.prepare_only:
        return
    root = campaign / (
        "phase55_dino_continuation/head"
        if args.continue_from_phase51
        else "phase51_dino_adaptation/head"
    )
    if args.risk_control:
        root = campaign / "phase57_dino_risk/head"
    if args.explicit_other_control:
        root = campaign / "phase59_explicit_other/head"
    root.mkdir(parents=True, exist_ok=True)
    rows = read(campaign / "phase46_paper_primary/fitting_rows.json")
    descriptors = input_rows(campaign)
    named = named_targets(campaign, rows, descriptors, args.explicit_other_control)
    protocol = dict(
        hypothesis="Original DINO visual final-block adaptation with scoped Healthy labels and masked original named-condition supervision, preserving the deployed recognizer",
        prefix_complete_sha256=digest(cache / "complete.json"),
        fitting_rows_sha256=digest(
            campaign / "phase46_paper_primary/fitting_rows.json"
        ),
        original_named_rows_sha256=digest(
            campaign / "phase10_leaf/cnn_inputs/rows.json"
        ),
        script_sha256=digest(__file__),
        model_source_sha256=digest(Path(__file__).with_name("partial_dino_healthy.py")),
        epochs=16 if args.continue_from_phase51 else 8,
        minimum_epochs=12 if args.continue_from_phase51 else 4,
        early_stop_patience=3,
        seed=1519,
        loss="Healthy-versus-rest BCE negative cost 4 plus 0.25 original named-class CE where explicit; coarse/unknown named targets masked",
        named_counts=dict(collections.Counter(str(int(v)) for v in named)),
        sampling="binary class/source/family",
        tail_learning_rate=0.000001,
        head_learning_rate=0.00001,
        batch=64,
        weight_decay=0.01,
        ema_decay=0.995,
        selection="Single EMA checkpoint by source/class/family weighted Healthy validation BCE with negative cost 4; no AUC ranking",
        original_prefix_fp32=True,
        deployed_recognizer_unchanged=True,
        later_app_screen="One unchanged scoped-paper route; no threshold expansion, export or full restored-model replay unless the cheap gates pass",
        cold_olid_used=False,
        ad_independent=False,
        restricted_teacher_used=False,
    )
    parent = campaign / "phase51_dino_adaptation/head"
    required = None
    if sparse_evaluation:
        parent = campaign / "phase55_dino_continuation/head"
        plan = campaign / "phase57_input_plan"
        required = np.array(read(plan / "inputs.json"), dtype=np.int64)
        assert (
            required.size and required.min() >= 0 and required.max() < len(descriptors)
        )
        protocol.update(
            initialization_checkpoint_sha256=digest(parent / "best.pt"),
            initialization_protocol_sha256=digest(parent / "protocol.json"),
            required_input_plan_sha256=digest(plan / "protocol.json"),
            required_input_indices_sha256=digest(plan / "inputs.json"),
            evaluation_scope="Only all fitting-prefix inputs plus the label-blind fixed route's required indices; unscored probabilities remain NaN and must never be used",
            maximum_controls=1,
        )
    if args.risk_control:
        protocol.update(
            hypothesis="Reduce high-confidence disease-to-Healthy tail errors by supervised family-weighted negative CVaR on the copied DINO final blocks; no RL or acceptance-data fitting",
            additional_negative_risk_coefficient=0.25,
            negative_tail_mass=0.1,
            risk_loss_source_sha256=digest(
                Path(__file__).with_name("healthy_tail_risk.py")
            ),
            loss=protocol["loss"]
            + " plus 0.25 conditional worst-10%-mass negative BCE",
            selection="Single EMA checkpoint minimizing source/class/family weighted negative-cost BCE plus 0.25 weighted worst-10%-mass negative BCE",
        )
    if args.explicit_other_control:
        protocol.update(
            hypothesis="Add the missing categorical Other supervision only for explicitly named publisher unsupported conditions; retain coarse/missing-condition masking and all app gates",
            loss=protocol["loss"]
            + "; additional explicit unsupported-condition CE weight 1, masked for coarse/unknown labels",
            explicit_other_target_scope="Original old input descriptor, label Other and a nonempty named raw publisher condition; generic Diseased/Unhealthy/Unknown targets remain masked",
            explicit_other_cross_entropy_weight=1.0,
        )
        assert (named == 8).any()
    if args.continue_from_phase51:
        initial = read(parent / "protocol.json")
        for key in (
            "prefix_complete_sha256",
            "fitting_rows_sha256",
            "original_named_rows_sha256",
            "model_source_sha256",
            "seed",
            "loss",
            "sampling",
            "tail_learning_rate",
            "head_learning_rate",
            "batch",
            "weight_decay",
            "ema_decay",
            "selection",
        ):
            assert initial[key] == protocol[key], key
        history = read(parent / "history.json")
        assert len(history) == 8 and history[-1]["best_epoch"] == 7
        assert history[-1]["validation_loss"] < history[-2]["validation_loss"]
        protocol.update(
            hypothesis="Continue the capped but still-improving original DINO adaptation without restarting training or altering loss, data, learning rates or app gates",
            continuation_parent_protocol_sha256=digest(parent / "protocol.json"),
            continuation_parent_last_sha256=digest(parent / "last.pt"),
            continuation_parent_best_sha256=digest(parent / "best.pt"),
            additional_epoch_limit=8,
            stopping_rule="At most eight additional epochs; stop on three stale validation epochs after four additional epochs, then screen exactly one unchanged route",
        )
    frozen = root / "protocol.json"
    if frozen.exists():
        assert read(frozen) == protocol
    else:
        frozen.write_text(json.dumps(protocol, indent=2))
    if (root / "result.json").exists():
        result = read(root / "result.json")
        assert result["protocol"] == protocol
        assert result["checkpoint_sha256"] == digest(root / "candidate.pt")
        assert result["probabilities_sha256"] == digest(root / "probabilities.npy")
        print(json.dumps(result), flush=True)
        return
    torch.set_num_threads(4)
    torch.manual_seed(1519)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.empty_cache()
    started = time.monotonic()
    tables = np.load(cache / "tables.npy")
    lookup = np.full(len(descriptors), -1, np.int64)
    lookup[tables] = np.arange(len(tables))
    ids = np.array([lookup[r["table"]] for r in rows])
    assert (ids >= 0).all()
    target = np.array([r["y"] for r in rows])
    train = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    val = np.array([i for i, r in enumerate(rows) if r["split"] == "val"])
    assert not (set(ids[train]) & set(ids[val]))
    assert not ({rows[i]["group"] for i in train} & {rows[i]["group"] for i in val})
    tokens = torch.from_numpy(np.load(cache / "features.npy", mmap_mode="r")).cuda()
    truth = torch.tensor(target, dtype=torch.float32, device="cuda")
    classes = torch.tensor(named, dtype=torch.long, device="cuda")
    sampling = torch.tensor(
        source_class_group_weights(rows, train), dtype=torch.float64
    )
    vw = source_class_group_weights(rows, val)
    weights = torch.tensor(vw, dtype=torch.float32, device="cuda")
    original = original_dino(campaign)
    model = PartialDinoHealthyTail(original).cuda()
    expected = read(cache / "complete.json")["prefix_state_sha256"]
    assert prefix_identity(model.backbone) == expected
    if sparse_evaluation:
        initialization = torch.load(
            parent / "best.pt", weights_only=True, map_location="cpu"
        )
        assert initialization["protocol"] == read(parent / "protocol.json")
        assert (
            initialization["protocol"]["prefix_complete_sha256"]
            == protocol["prefix_complete_sha256"]
        )
        model.load_state_dict(initialization["model"], strict=True)
        assert prefix_identity(model.backbone) == expected
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [
            dict(
                params=list(model.backbone.blocks[model.cut :].parameters()),
                lr=0.000001,
            ),
            dict(
                params=list(model.backbone.norm.parameters())
                + list(model.backbone.fc_norm.parameters())
                + list(model.backbone.head.parameters()),
                lr=0.00001,
            ),
        ],
        weight_decay=0.01,
    )
    start, best, best_epoch, stale, history = 0, float("inf"), -1, 0, []
    resume = root / "last.pt"
    from_parent = args.continue_from_phase51 and not resume.exists()
    if from_parent:
        resume = parent / "last.pt"
        shutil.copy2(parent / "best.pt", root / "best.pt")
        shutil.copy2(parent / "best_val_logits.npy", root / "best_val_logits.npy")
    if resume.exists():
        state = torch.load(resume, weights_only=True, map_location="cuda")
        assert state["protocol"] == (initial if from_parent else protocol)
        model.load_state_dict(state["model"], strict=True)
        ema.load_state_dict(state["ema"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start, best, best_epoch, stale, history = (
            state["epoch"] + 1,
            state["best"],
            state["best_epoch"],
            state["stale"],
            state["history"],
        )
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng"].cpu())

    def evaluate(module, indices):
        with torch.inference_mode():
            return torch.cat(
                [
                    healthy_margin(module(tokens[part]))
                    for part in np.array_split(
                        indices, max(1, (len(indices) + 63) // 64)
                    )
                ]
            )

    for epoch in range(start, protocol["epochs"]):
        if (
            epoch >= protocol["minimum_epochs"]
            and stale >= protocol["early_stop_patience"]
        ):
            break
        model.train()
        order = torch.multinomial(
            sampling,
            len(train),
            replacement=True,
            generator=torch.Generator().manual_seed(151900 + epoch),
        )
        for positions in order.split(64):
            index = train[positions.numpy()]
            logits = model(tokens[ids[index]])
            loss = (
                F.binary_cross_entropy_with_logits(
                    healthy_margin(logits), truth[index], reduction="none"
                )
                * torch.where(truth[index] > 0, 1.0, 4.0)
            ).mean()
            if args.risk_control:
                negatives = truth[index] == 0
                loss = loss + 0.25 * weighted_upper_tail(
                    F.softplus(healthy_margin(logits)[negatives]), fraction=0.1
                )
            valid = (classes[index] >= 0) & (classes[index] < 8)
            if valid.any():
                loss = loss + 0.25 * F.cross_entropy(
                    logits[valid], classes[index][valid]
                )
            explicit_other = classes[index] == 8
            if args.explicit_other_control and explicit_other.any():
                loss = loss + F.cross_entropy(
                    logits[explicit_other], classes[index][explicit_other]
                )
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                1.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            with torch.no_grad():
                moving = dict(ema.named_parameters())
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        moving[name].mul_(0.995).add_(parameter, alpha=0.005)
        ema.eval()
        z = evaluate(ema, ids[val])
        score = float(
            (
                F.binary_cross_entropy_with_logits(z, truth[val], reduction="none")
                * torch.where(truth[val] > 0, 1.0, 4.0)
                * weights
            ).sum()
        )
        if args.risk_control:
            negatives = truth[val] == 0
            score += float(
                0.25
                * weighted_upper_tail(F.softplus(z[negatives]), weights[negatives], 0.1)
            )
        assert np.isfinite(score)
        if score < best - 0.00001:
            best, best_epoch, stale = score, epoch, 0
            atomic_save(
                dict(model=cpu_state(ema), epoch=epoch, protocol=protocol),
                root / "best.pt",
            )
            np.save(root / "best_val_logits.npy", z.cpu().numpy())
        else:
            stale += 1
        assert (
            prefix_identity(model.backbone) == prefix_identity(ema.backbone) == expected
        )
        record = dict(
            epoch=epoch,
            validation_loss=score,
            best_epoch=best_epoch,
            elapsed_seconds=time.monotonic() - started,
        )
        history.append(record)
        atomic_save(
            dict(
                model=cpu_state(model),
                ema=cpu_state(ema),
                optimizer=optimizer.state_dict(),
                epoch=epoch,
                best=best,
                best_epoch=best_epoch,
                stale=stale,
                history=history,
                protocol=protocol,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(),
            ),
            root / "last.pt",
        )
        (root / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(record), flush=True)
    selected = torch.load(root / "best.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(selected["model"], strict=True)
    model.eval()
    raw = np.full(len(descriptors), np.nan, np.float32)
    raw[tables] = evaluate(model, np.arange(len(tables))).cpu().numpy()
    error = float(
        np.max(
            np.abs(
                raw[np.array([r["table"] for r in rows])[val]]
                - np.load(root / "best_val_logits.npy")
            )
        )
    )
    assert error < 0.002
    calibration = calibrate(
        raw[np.array([r["table"] for r in rows])[val]], target[val], vw
    )
    missing = np.flatnonzero(lookup < 0)
    if sparse_evaluation:
        missing = np.intersect1d(missing, required)
    pixels = CanonicalInputs(campaign)
    try:
        with concurrent.futures.ThreadPoolExecutor(8) as pool, torch.inference_mode():
            for start in range(0, len(missing), 64):
                part = missing[start : start + 64]
                image = (
                    torch.from_numpy(
                        np.concatenate(
                            list(pool.map(pixels, [descriptors[i] for i in part])), 0
                        )
                    )
                    .cuda()
                    .permute(0, 3, 1, 2)
                )
                raw[part] = (
                    healthy_margin(model(prefix_features(model.backbone, image)))
                    .cpu()
                    .numpy()
                )
    finally:
        pixels.close()
    probabilities = 1 / (
        1 + np.exp(-np.clip(raw * calibration[0] + calibration[1], -80, 80))
    )
    np.save(root / "logits.npy", raw)
    np.save(root / "probabilities.npy", probabilities)
    scored_indices = np.flatnonzero(np.isfinite(probabilities))
    np.save(root / "scored_indices.npy", scored_indices)
    assert not sparse_evaluation or set(map(int, required)) <= set(
        map(int, scored_indices)
    )
    atomic_save(
        dict(
            architecture="PartialDinoHealthyTail",
            model=cpu_state(model),
            epoch=selected["epoch"],
            prefix_state_sha256=expected,
            calibration=calibration,
            protocol=protocol,
        ),
        root / "candidate.pt",
    )
    result = dict(
        protocol=protocol,
        checkpoint_sha256=digest(root / "candidate.pt"),
        probabilities_sha256=digest(root / "probabilities.npy"),
        scored_indices_sha256=digest(root / "scored_indices.npy"),
        scored_inputs=len(scored_indices),
        logits_sha256=digest(root / "logits.npy"),
        selected_epoch=selected["epoch"],
        calibration=calibration,
        cached_restore_maximum_error=error,
        reused_fitting_prefix_inputs=len(tables),
        new_evaluation_prefix_inputs=len(missing),
        inputs=len(descriptors),
        trainable_parameters=sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        elapsed_seconds=time.monotonic() - started,
        cold_olid_used=False,
        app_acceptance_claim=False,
    )
    (root / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
