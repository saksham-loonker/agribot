"""One bounded supervised final-stage CNN adaptation using immutable prefixes."""

from pathlib import Path
import argparse
import copy
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from campaign_cnn_fusion import calibrate
from opened_context_data import read
from partial_cnn_cache import original_cnn, prepare
from partial_cnn_healthy import PartialCnnHealthyTail, prefix_identity
from permissive_encoder import digest
from source_balance import source_class_group_weights
from training_support import atomic_save


def cpu_state(model):
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    campaign = Path(args.campaign)
    cache = prepare(campaign)
    if args.prepare_only:
        return
    root = campaign / "phase49_partial_cnn/head"
    root.mkdir(exist_ok=True)
    protocol = dict(
        hypothesis="Adapt original CNN final visual stages after the properly scoped paper/scene coverage control, retaining its original frozen prefix",
        prefix_complete_sha256=digest(cache / "complete.json"),
        fitting_rows_sha256=digest(
            campaign / "phase46_paper_primary/fitting_rows.json"
        ),
        script_sha256=digest(__file__),
        model_source_sha256=digest(Path(__file__).with_name("partial_cnn_healthy.py")),
        supervision="Human leaf and primary-image targets within their scopes; no patch targets or AD ROI propagation",
        epochs=16,
        minimum_epochs=5,
        early_stop_patience=4,
        seed=1491,
        loss="BCE with negative cost 4",
        sampling="class/source/family",
        tail_learning_rate=0.00001,
        projection_learning_rate=0.0001,
        weight_decay=0.01,
        batch=512,
        ema_decay=0.995,
        batchnorm_running_statistics_frozen=True,
        selection="Single EMA checkpoint by source/class/family weighted validation BCE with negative cost 4; no AUC selection",
        later_app_screen="One fixed scoped-paper protocol with unchanged original/internal app gates; stop if it fails, no threshold or architecture expansion",
        deployed_recognizer_unchanged=True,
        original_prefix_unchanged=True,
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
        done = read(root / "result.json")
        assert done["protocol"] == protocol
        assert done["checkpoint_sha256"] == digest(root / "candidate.pt")
        assert done["probabilities_sha256"] == digest(root / "probabilities.npy")
        print(json.dumps(done), flush=True)
        return
    torch.set_num_threads(4)
    torch.manual_seed(1491)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    rows = read(campaign / "phase46_paper_primary/fitting_rows.json")
    table = np.array([r["table"] for r in rows])
    target = np.array([r["y"] for r in rows])
    train = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    val = np.array([i for i, r in enumerate(rows) if r["split"] == "val"])
    assert set(target[train]) == set(target[val]) == {0, 1}
    assert not (set(table[train]) & set(table[val]))
    prefix = torch.from_numpy(np.load(cache / "features.npy")).cuda()
    truth = torch.tensor(target, dtype=torch.float32, device="cuda")
    sampling = torch.tensor(
        source_class_group_weights(rows, train), dtype=torch.float64
    )
    validation_weights = source_class_group_weights(rows, val)
    val_weights = torch.tensor(validation_weights, dtype=torch.float32, device="cuda")
    cnn, architecture, _ = original_cnn(campaign)
    model = PartialCnnHealthyTail(cnn).cuda()
    expected_prefix = read(cache / "complete.json")["prefix_state_sha256"]
    assert prefix_identity(model.cnn) == expected_prefix
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [
            dict(params=list(model.cnn.blocks[model.cut :].parameters()), lr=0.00001),
            dict(
                params=list(model.cnn.conv_head.parameters())
                + list(model.cnn.classifier.parameters()),
                lr=0.0001,
            ),
        ],
        weight_decay=0.01,
    )
    start, best_score, best_epoch, stale, history = 0, float("inf"), -1, 0, []
    if (root / "last.pt").exists():
        last = torch.load(root / "last.pt", weights_only=True, map_location="cuda")
        assert last["protocol"] == protocol
        model.load_state_dict(last["model"], strict=True)
        ema.load_state_dict(last["ema"], strict=True)
        optimizer.load_state_dict(last["optimizer"])
        start = last["epoch"] + 1
        best_score, best_epoch, stale, history = (
            last["best_score"],
            last["best_epoch"],
            last["stale"],
            last["history"],
        )
        torch.set_rng_state(last["torch_rng"].cpu())
        torch.cuda.set_rng_state(last["cuda_rng"].cpu())

    def evaluate(module, ids):
        with torch.inference_mode():
            return torch.cat(
                [
                    module(prefix[part])
                    for part in np.array_split(ids, max(1, (len(ids) + 511) // 512))
                ]
            )

    for epoch in range(start, 16):
        if epoch >= 5 and stale >= 4:
            break
        model.train()
        order = torch.multinomial(
            sampling,
            len(train),
            replacement=True,
            generator=torch.Generator().manual_seed(149100 + epoch),
        )
        for positions in order.split(512):
            ids = train[positions.numpy()]
            logits = model(prefix[table[ids]])
            loss = (
                F.binary_cross_entropy_with_logits(logits, truth[ids], reduction="none")
                * torch.where(truth[ids] > 0, 1.0, 4.0)
            ).mean()
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
        z = evaluate(ema, table[val])
        score = float(
            (
                F.binary_cross_entropy_with_logits(z, truth[val], reduction="none")
                * torch.where(truth[val] > 0, 1.0, 4.0)
                * val_weights
            ).sum()
        )
        assert np.isfinite(score)
        if score < best_score - 0.00001:
            best_score, best_epoch, stale = score, epoch, 0
            atomic_save(
                dict(model=cpu_state(ema), epoch=epoch, protocol=protocol),
                root / "best.pt",
            )
            np.save(root / "best_val_logits.npy", z.cpu().numpy())
        else:
            stale += 1
        assert prefix_identity(model.cnn) == prefix_identity(ema.cnn) == expected_prefix
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
                best_score=best_score,
                best_epoch=best_epoch,
                stale=stale,
                history=history,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(),
                protocol=protocol,
            ),
            root / "last.pt",
        )
        (root / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(record), flush=True)
    chosen = torch.load(root / "best.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(chosen["model"], strict=True)
    model.eval()
    logits = evaluate(model, np.arange(len(prefix))).cpu().numpy()
    error = float(
        np.max(np.abs(logits[table[val]] - np.load(root / "best_val_logits.npy")))
    )
    assert error < 0.002
    calibration = calibrate(logits[table[val]], target[val], validation_weights)
    probabilities = 1 / (
        1 + np.exp(-np.clip(logits * calibration[0] + calibration[1], -80, 80))
    )
    np.save(root / "logits.npy", logits)
    np.save(root / "probabilities.npy", probabilities)
    atomic_save(
        dict(
            architecture="PartialCnnHealthyTail",
            cnn_architecture=architecture,
            model=cpu_state(model),
            cut=model.cut,
            prefix_state_sha256=expected_prefix,
            calibration=calibration,
            epoch=chosen["epoch"],
            protocol=protocol,
        ),
        root / "candidate.pt",
    )
    done = dict(
        protocol=protocol,
        checkpoint_sha256=digest(root / "candidate.pt"),
        probabilities_sha256=digest(root / "probabilities.npy"),
        logits_sha256=digest(root / "logits.npy"),
        selected_epoch=chosen["epoch"],
        calibration=calibration,
        cached_restore_maximum_error=error,
        inputs=len(prefix),
        trainable_parameters=sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        elapsed_seconds=time.monotonic() - started,
        cold_olid_used=False,
        app_acceptance_claim=False,
    )
    (root / "result.json").write_text(json.dumps(done, indent=2))
    print(json.dumps(done), flush=True)


if __name__ == "__main__":
    main()
