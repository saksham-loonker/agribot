"""Bounded supervised SigLIP tail adaptation for explicit unsupported conditions.

Uses the admitted primary-image task, preserves every deployed model, caches
the original FP32 prefix and evaluates only label-blind new-Healthy proposals.
"""

from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import shutil
import time

import numpy as np
from PIL import Image
import torch
import timm
from torch.nn import functional as F

from app_eval import classifier_letterbox_input
from campaign_cnn_fusion import calibrate
from campaign_pixels import open_pixels
from evaluation_inputs import input_identity
from feature_cache_reuse import (
    runtime_identity,
    preprocessing_contract,
    validate_completed_contract,
)
from opened_context_data import read
from partial_primary_siglip import (
    PartialPrimarySiglip,
    prefix_features,
    prefix_identity,
)
from permissive_encoder import CanonicalPermissiveEncoder, digest
from phase28_roi_confirmation import pixel_key
from phase62_primary_unsupported import targets
from source_balance import source_class_group_weights
from training_support import atomic_save


def freeze(path, value):
    if path.exists():
        assert read(path) == value, ("Frozen identity changed", str(path))
    else:
        path.write_text(json.dumps(value, indent=2))


def prepare(campaign, model, selected, source_features):
    p = Path(campaign)
    source = p / "phase23_siglip"
    out = p / "phase67_primary_siglip/prefix"
    out.mkdir(parents=True, exist_ok=True)
    shape = (len(selected), 256, 768)
    protocol = dict(
        encoder_sha256=digest(source / "siglip2_feature_heads/encoder.pt"),
        source_rows_sha256=digest(source / "cnn_inputs/rows.json"),
        source_pixel_manifest_sha256=digest(source / "cnn_inputs/pixels.json"),
        source_features_sha256=digest(source / "siglip2_feature_heads/features.npy"),
        prefix_state_sha256=prefix_identity(model.encoder),
        selected_indices_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
        model_source_sha256=digest(
            Path(__file__).with_name("partial_primary_siglip.py")
        ),
        preprocessing=preprocessing_contract(
            Path(__file__).resolve().parents[2], "siglip2"
        ),
        runtime=runtime_identity(),
        shape=list(shape),
        dtype="float32",
        representation="Original SigLIP tokens before its final two blocks; no rounding or quantization",
        cold_olid_used=False,
        ad_independent=False,
    )
    freeze(out / "protocol.json", protocol)
    if (out / "complete.json").exists():
        complete = read(out / "complete.json")
        assert complete["protocol_sha256"] == digest(out / "protocol.json")
        assert complete["inputs_sha256"] == digest(out / "inputs.json")
        for shard in complete["shards"]:
            assert digest(out / shard["file"]) == shard["sha256"]
        return out, complete
    started = time.monotonic()
    row_bytes = 256 * 768 * 4
    shared = Path("/dev/shm")
    shared_rows = (
        min(len(selected), (1 << 30) // row_bytes)
        if shutil.disk_usage(shared).free > (1 << 30) + (256 << 20)
        else 0
    )
    disk_bytes = (len(selected) - shared_rows) * row_bytes
    assert shutil.disk_usage(p).free > disk_bytes + (
        800 << 20
    ), "Insufficient disk headroom for exact tokens and resumable tail state"
    parts = []
    if shared_rows:
        destination = shared / (
            "agribot-primary-siglip-" + protocol["prefix_state_sha256"][:12] + ".npy"
        )
        link = out / "tokens-shared.npy"
        if link.is_symlink():
            assert link.resolve() == destination
        else:
            assert not link.exists()
            link.symlink_to(destination)
        parts.append((0, shared_rows, link))
    if shared_rows < len(selected):
        parts.append((shared_rows, len(selected), out / "tokens-disk.npy"))
    arrays = [
        np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(end - start, 256, 768)
        )
        for start, end, path in parts
    ]
    pixels = open_pixels(source / "cnn_inputs")
    inputs = []
    maximum = 0.0
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(selected), 64):
                indices = selected[start : start + 64]
                canonical = np.array(pixels[indices], copy=True)
                token = prefix_features(
                    model.encoder, torch.from_numpy(canonical).cuda()
                )
                actual = model.features(token).cpu().numpy()
                maximum = max(
                    maximum, float(np.max(np.abs(actual - source_features[indices])))
                )
                assert maximum < 0.002, (
                    "Original prefix/suffix does not match cached encoder",
                    maximum,
                )
                value = token.cpu().numpy()
                for offset, (begin, end, _) in enumerate(parts):
                    lo, hi = max(start, begin), min(start + len(indices), end)
                    if lo < hi:
                        arrays[offset][lo - begin : hi - begin] = value[
                            lo - start : hi - start
                        ]
                inputs.extend(
                    dict(index=int(i), input_sha256=pixel_key(canonical[j : j + 1]))
                    for j, i in enumerate(indices)
                )
    finally:
        if hasattr(pixels, "close"):
            pixels.close()
    for array in arrays:
        array.flush()
    (out / "inputs.json").write_text(json.dumps(inputs))
    complete = dict(
        protocol_sha256=digest(out / "protocol.json"),
        inputs_sha256=digest(out / "inputs.json"),
        shards=[
            dict(file=path.name, start=start, end=end, sha256=digest(path))
            for start, end, path in parts
        ],
        maximum_original_feature_error=maximum,
        inputs=len(selected),
        elapsed_seconds=time.monotonic() - started,
        cold_olid_used=False,
    )
    (out / "complete.json").write_text(json.dumps(complete, indent=2))
    return out, complete


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    args = ap.parse_args()
    p = Path(args.campaign)
    root = p / "phase23_siglip"
    out = p / "phase67_primary_siglip/head"
    out.mkdir(parents=True, exist_ok=True)
    validate_completed_contract(root, root / "siglip2_feature_heads", "siglip2")
    rows = read(root / "cnn_inputs/rows.json")
    csv_paths = {
        name: Path("/home/genesis/agribot-data/PlantDoc") / file
        for name, file in [("TRAIN", "train_labels.csv"), ("TEST", "test_labels.csv")]
    }
    y_all = targets(rows, read(p / "phase10_leaf/cnn_inputs/rows.json"), csv_paths)
    selected = np.flatnonzero(y_all >= 0)
    fitting = [dict(rows[i], y=int(y_all[i])) for i in selected]
    y = y_all[selected]
    train = np.array([i for i, r in enumerate(fitting) if r["split"] == "train"])
    val = np.array([i for i, r in enumerate(fitting) if r["split"] == "val"])
    assert not (
        {fitting[i]["group"] for i in train} & {fitting[i]["group"] for i in val}
    )
    assert set(y[train]) == set(y[val]) == {0, 1}
    head_directory = p / "phase65_primary_unsupported_mlp"
    head_result = read(head_directory / "result.json")
    assert digest(head_directory / "best.pt") == head_result["checkpoint_sha256"]
    assert head_result["protocol"]["source_rows_sha256"] == digest(
        root / "cnn_inputs/rows.json"
    )
    for name, path in csv_paths.items():
        assert head_result["protocol"]["publisher_taxonomy_sha256"][name] == digest(
            path
        )
    proposals_path = p / "phase64_failure_audit/new_healthy_proposals.json"
    proposals = sorted(read(proposals_path))
    protocol = dict(
        hypothesis="Adapt original SigLIP last two blocks and pooling to the explicit unsupported-versus-Healthy primary task after both frozen heads failed; no broad easy known-disease negatives or ROI label propagation",
        encoder_sha256=digest(root / "siglip2_feature_heads/encoder.pt"),
        source_rows_sha256=digest(root / "cnn_inputs/rows.json"),
        target_source_sha256=digest(
            Path(__file__).with_name("phase62_primary_unsupported.py")
        ),
        target_protocol_sha256=digest(head_directory / "protocol.json"),
        initial_head_sha256=head_result["checkpoint_sha256"],
        model_source_sha256=digest(
            Path(__file__).with_name("partial_primary_siglip.py")
        ),
        script_sha256=digest(__file__),
        selected_indices_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
        proposal_paths_sha256=hashlib.sha256(
            json.dumps(proposals).encode()
        ).hexdigest(),
        proposal_input_identity=input_identity(
            [dict(id=path, path=path) for path in proposals]
        ),
        proposal_rule="All new Healthy proposals from the fixed phase56 route, without filtering by label, condition or source",
        parent_route_protocol_sha256=digest(p / "phase56_continued_dino/protocol.json"),
        epochs=8,
        minimum_epochs=4,
        early_stop_patience=3,
        seed=1673,
        tail_lr=0.000001,
        head_lr=0.00001,
        weight_decay=0.01,
        ema_decay=0.995,
        batch=32,
        loss="Source/class/family balanced explicit unsupported BCE; Healthy false-veto cost 4",
        selection="Single EMA checkpoint minimizing source/class/family weighted validation BCE with Healthy cost 4",
        calibration="Same weighted validation objective, no app-output fitting",
        later_policy="One unchanged binary .5 veto on new Healthy proposals; no threshold expansion",
        maximum_controls=1,
        cold_olid_used=False,
        ad_independent=False,
        deployed_weights_unchanged=True,
    )
    freeze(out / "protocol.json", protocol)
    if (out / "result.json").exists():
        result = read(out / "result.json")
        assert result["protocol"] == protocol
        assert result["checkpoint_sha256"] == digest(out / "candidate.pt")
        assert result["probabilities_sha256"] == digest(out / "probabilities.json")
        print(json.dumps(result), flush=True)
        return
    torch.set_num_threads(4)
    torch.manual_seed(1673)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timm.layers.set_fused_attn(False)
    state = torch.load(
        root / "siglip2_feature_heads/encoder.pt", weights_only=True, map_location="cpu"
    )
    encoder = CanonicalPermissiveEncoder(state["spec"]).eval().requires_grad_(False)
    encoder.load_state_dict(state["model"], strict=True)
    model = PartialPrimarySiglip(encoder).cuda().eval()
    warm = torch.load(head_directory / "best.pt", weights_only=True, map_location="cpu")
    model.head.load_state_dict(warm["model"], strict=True)
    original_prefix = prefix_identity(model.encoder)
    spec = state["spec"]
    del encoder, state, warm
    source_features = np.load(
        root / "siglip2_feature_heads/features.npy", mmap_mode="r"
    )
    cache, complete = prepare(p, model, selected, source_features)
    tokens = torch.empty((len(selected), 256, 768), dtype=torch.float32, device="cuda")
    for shard in complete["shards"]:
        array = np.load(cache / shard["file"], mmap_mode="r")
        for start in range(0, len(array), 64):
            tokens[
                shard["start"] + start : shard["start"] + min(start + 64, len(array))
            ] = torch.from_numpy(np.array(array[start : start + 64], copy=True)).cuda()
    started = time.monotonic()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [
            dict(
                params=[v for v in model.encoder.parameters() if v.requires_grad],
                lr=1e-6,
            ),
            dict(params=model.head.parameters(), lr=1e-5),
        ],
        weight_decay=0.01,
    )
    truth = torch.tensor(y, dtype=torch.float32, device="cuda")
    sampling = torch.tensor(
        source_class_group_weights(fitting, train), dtype=torch.float64
    )
    vw = source_class_group_weights(fitting, val)
    weights = torch.tensor(vw, dtype=torch.float32, device="cuda")
    start, best, stale, history = 0, float("inf"), 0, []
    if (out / "last.pt").exists():
        saved = torch.load(out / "last.pt", weights_only=True, map_location="cuda")
        assert saved["protocol"] == protocol
        model.load_tail(saved["tail"])
        # EMA parameters are frozen; its state keys are checked against the trainable model.
        assert set(saved["ema_tail"]) == set(model.tail_state())
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
    trainable_names = set(model.tail_state())

    def tail(module):
        return {
            k: v.detach().cpu().clone()
            for k, v in module.named_parameters()
            if k in trainable_names
        }

    def evaluate(module, indices):
        with torch.inference_mode():
            return torch.cat(
                [
                    module(tokens[part])
                    for part in np.array_split(
                        indices, max(1, (len(indices) + 31) // 32)
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
            generator=torch.Generator().manual_seed(167300 + epoch),
        )
        for positions in order.split(32):
            ids = train[positions.numpy()]
            logits = model(tokens[ids])
            loss = (
                F.binary_cross_entropy_with_logits(logits, truth[ids], reduction="none")
                * torch.where(truth[ids] == 0, 4.0, 1.0)
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
                moving = dict(ema.named_parameters())
                for name, value in model.named_parameters():
                    if value.requires_grad:
                        moving[name].mul_(0.995).add_(value, alpha=0.005)
        ema.eval()
        z = evaluate(ema, val)
        score = float(
            (
                F.binary_cross_entropy_with_logits(z, truth[val], reduction="none")
                * torch.where(truth[val] == 0, 4.0, 1.0)
                * weights
            ).sum()
        )
        if score < best - 1e-5:
            best, stale = score, 0
            atomic_save(
                dict(tail=tail(ema), epoch=epoch, protocol=protocol), out / "best.pt"
            )
            np.save(out / "best_val_logits.npy", z.cpu().numpy())
        else:
            stale += 1
        assert (
            prefix_identity(model.encoder)
            == prefix_identity(ema.encoder)
            == original_prefix
        )
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
                protocol=protocol,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(),
            ),
            out / "last.pt",
        )
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(history[-1]), flush=True)
    selected_state = torch.load(out / "best.pt", weights_only=True, map_location="cpu")
    model.load_tail(selected_state["tail"])
    model.eval()
    raw = evaluate(model, val).cpu().numpy()
    error = float(np.max(np.abs(raw - np.load(out / "best_val_logits.npy"))))
    assert error < 0.002
    cw = vw * np.where(y[val] == 0, 4.0, 1.0)
    calibration = calibrate(raw, y[val], cw / cw.sum())
    prefix_lookup = {
        r["input_sha256"]: i for i, r in enumerate(read(cache / "inputs.json"))
    }
    scores = {}
    new_prefixes = 0
    with torch.inference_mode():
        for start in range(0, len(proposals), 32):
            paths = proposals[start : start + 32]
            values = []
            for path in paths:
                with Image.open(path) as image:
                    value = classifier_letterbox_input(
                        np.asarray(image.convert("RGB")), 224
                    )
                key = pixel_key(value)
                if key in prefix_lookup:
                    prefix = tokens[prefix_lookup[key] : prefix_lookup[key] + 1]
                else:
                    prefix = prefix_features(
                        model.encoder, torch.from_numpy(value).cuda()
                    )
                    new_prefixes += 1
                values.append(prefix)
            z = model(torch.cat(values)).cpu().numpy()
            prob = 1 / (
                1 + np.exp(-np.clip(z * calibration[0] + calibration[1], -80, 80))
            )
            scores.update(dict(zip(paths, map(float, prob))))
    (out / "probabilities.json").write_text(json.dumps(scores))
    atomic_save(
        dict(
            architecture="PartialPrimarySiglip",
            spec=spec,
            kind="mlp",
            tail=model.tail_state(),
            calibration=calibration,
            protocol=protocol,
            prefix_state_sha256=original_prefix,
        ),
        out / "candidate.pt",
    )
    result = dict(
        protocol=protocol,
        checkpoint_sha256=digest(out / "candidate.pt"),
        probabilities_sha256=digest(out / "probabilities.json"),
        prefix_complete_sha256=digest(cache / "complete.json"),
        selected_epoch=selected_state["epoch"],
        epochs=len(history),
        calibration=calibration,
        cached_restore_maximum_error=error,
        elapsed_seconds=time.monotonic() - started,
        proposal_inputs=len(proposals),
        reused_proposal_prefixes=len(proposals) - new_prefixes,
        new_encoder_prefix_inputs=new_prefixes,
        original_prefix_sha256=original_prefix,
        cold_olid_used=False,
        ad_independent=False,
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
