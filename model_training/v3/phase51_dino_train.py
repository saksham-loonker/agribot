"""One disease-aware supervised adaptation of the original DINO's last blocks."""

from pathlib import Path
import argparse
import collections
import concurrent.futures
import copy
import json
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


def named_targets(campaign, fitting, descriptors):
    source = read(Path(campaign) / "phase10_leaf/cnn_inputs/rows.json")
    labels = []
    for row in fitting:
        descriptor = descriptors[row["table"]]
        label = (
            source[descriptor["index"]]["label"]
            if descriptor["kind"] == "old"
            else None
        )
        labels.append(
            1
            if row["y"] == 1
            else (
                LABELS.index(label)
                if label in LABELS[:8] and label != "Healthy"
                else -1
            )
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
    args = parser.parse_args()
    campaign = Path(args.campaign)
    cache = prepare(campaign)
    if args.prepare_only:
        return
    root = campaign / "phase51_dino_adaptation/head"
    root.mkdir(exist_ok=True)
    rows = read(campaign / "phase46_paper_primary/fitting_rows.json")
    descriptors = input_rows(campaign)
    named = named_targets(campaign, rows, descriptors)
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
        epochs=8,
        minimum_epochs=4,
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
    if (root / "last.pt").exists():
        state = torch.load(root / "last.pt", weights_only=True, map_location="cuda")
        assert state["protocol"] == protocol
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

    for epoch in range(start, 8):
        if epoch >= 4 and stale >= 3:
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
            valid = classes[index] >= 0
            if valid.any():
                loss = loss + 0.25 * F.cross_entropy(
                    logits[valid], classes[index][valid]
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
    raw = np.zeros(len(descriptors), np.float32)
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
