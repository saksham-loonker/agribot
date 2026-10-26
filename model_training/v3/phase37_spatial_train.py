"""One supervised local-feature control; no encoder updates or input relabeling."""

from pathlib import Path
import argparse
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from campaign_cnn_fusion import calibrate
from context_guard import route_guard
from opened_context_data import load, read
from opened_guard_screen import OpenedGuardScreen, ranking
from permissive_encoder import digest
from phase29_compact_prepare import region_key
from source_balance import source_class_group_weights
from spatial_cache import validate
from spatial_healthy_head import SpatialHealthyHead
from training_support import atomic_save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--audit-map-cache", action="store_true")
    ap.add_argument(
        "--context-primary",
        action="store_true",
        help="Isolated data-coverage control adding TOM primary-scene bags; no per-ROI label propagation",
    )
    args = ap.parse_args()
    p = Path(args.campaign)
    root = p / (
        "phase38_context_spatial" if args.context_primary else "phase37_spatial"
    )
    out = root / "head"
    out.mkdir(exist_ok=True)
    contract = validate(p, args.audit_map_cache)
    extra = None
    if args.context_primary:
        extra = read(root / "complete.json")
        assert extra["base_maps_sha256"] == contract["maps_sha256"]
        assert extra["new_maps_sha256"] == digest(root / "maps.npy")
        assert extra["fitting_rows_sha256"] == digest(root / "fitting_rows.json")
    data = load(p)
    protocol = dict(
        hypothesis="Spatial evidence before global average pooling retains small unhealthy regions",
        map_contract=contract,
        data_identity=data["identity"],
        head="Shared channel LayerNorm/Linear576->64/GELU/Linear1, normalized smooth minimum over 49 spatial locations",
        supervision="Only aggregate leaf/primary-image targets, never targets copied to each patch or detected AD ROI",
        loss="weighted BCE, negative cost 4",
        sampling="class/source/family",
        seed=991,
        epochs=60,
        lr=0.001,
        checkpoint_epochs=[0, 4, 9, 19, 39, 59],
        leaf_veto_minima=[0.2, 0.5],
        whole_positive_min=0.85,
        whole_negative_max=0.001,
        maximum_app_checks=12,
        original_encoders_frozen=True,
        reference_healthy_and_disease_exact=True,
        cold_olid_used=False,
        ad_independent=False,
        script_sha256=digest(__file__),
        head_source_sha256=digest(Path(__file__).with_name("spatial_healthy_head.py")),
        stopping_rule="Stop if none of 12 fixed checkpoint/policy checks passes unchanged app gates; no architecture or threshold expansion",
    )
    if extra is not None:
        protocol["primary_scene_extension"] = extra
    frozen = out / "protocol.json"
    if frozen.exists():
        assert read(frozen) == protocol
    else:
        frozen.write_text(json.dumps(protocol, indent=2))
    if (out / "result.json").exists():
        print((out / "result.json").read_text())
        return
    torch.set_num_threads(4)
    torch.manual_seed(991)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    maps = torch.from_numpy(
        np.load(p / "phase37_spatial/maps.npy", mmap_mode="r")
    ).cuda()
    if args.context_primary:
        maps = torch.cat(
            [maps, torch.from_numpy(np.load(root / "maps.npy", mmap_mode="r")).cuda()]
        )
    rows = read(
        root / "fitting_rows.json"
        if args.context_primary
        else p / "phase29_compact/fitting_rows.json"
    )
    train = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    val = np.array([i for i, r in enumerate(rows) if r["split"] == "val"])
    table = np.array([r["table"] for r in rows])
    target = np.array([r["y"] for r in rows])
    truth = torch.tensor(target, dtype=torch.float32, device="cuda")
    sampling = torch.tensor(
        source_class_group_weights(rows, train), dtype=torch.float64
    )
    weights = source_class_group_weights(rows, val)
    model = SpatialHealthyHead().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    screen = OpenedGuardScreen(data)
    indices = [
        read(p / "phase29_compact" / (name + "_indices.json"))
        for name in ("center", "full")
    ]
    keys = {
        path: [region_key(path, leaf["box"]) for leaf in r["leaves"]]
        for path, r in data["photos"].items()
    }
    started = time.monotonic()
    history = []
    ranked = []
    best = None
    passing = 0
    for epoch in range(60):
        model.train()
        order = torch.multinomial(
            sampling,
            len(train),
            replacement=True,
            generator=torch.Generator().manual_seed(99100 + epoch),
        )
        total = 0.0
        batches = 0
        for positions in order.split(1024):
            ids = train[positions.numpy()]
            z = model(maps[table[ids]])
            y = truth[ids]
            loss = (
                F.binary_cross_entropy_with_logits(z, y, reduction="none")
                * torch.where(y > 0, 1.0, 4.0)
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        if epoch not in protocol["checkpoint_epochs"]:
            continue
        model.eval()
        with torch.inference_mode():
            logits = (
                torch.cat([model(batch) for batch in maps.split(512)]).cpu().numpy()
            )
        calibration = calibrate(logits[table[val]], target[val], weights)
        scores = 1 / (
            1 + np.exp(-np.clip(logits * calibration[0] + calibration[1], -80, 80))
        )
        minimum = {
            path: min(
                [float(scores[t[k]]) for t in indices for k in values], default=1.0
            )
            for path, values in keys.items()
        }
        evaluated = []
        for cutoff in protocol["leaf_veto_minima"]:
            policy = dict(
                healthy_min=0.85,
                negative_healthy_max=0.001,
                genus_min=0.0,
                protect_suspected_disease=True,
                preserve_original_healthy=True,
                in_domain_other_max=0.1,
                in_domain_disease_mass_min=0.6,
            )

            def route(row):
                path = row["path"]
                return route_guard(
                    row,
                    data["manifest"],
                    data["positive"][path] if minimum[path] >= cutoff else 0.0,
                    data["negative"][path],
                    data["genus"][path],
                    policy,
                )

            candidate = screen.evaluate(route)
            candidate.update(
                epoch=epoch, calibration=calibration, leaf_cutoff=cutoff, policy=policy
            )
            ranked.append(candidate)
            evaluated.append(candidate)
        selected = max(evaluated, key=ranking)
        state = dict(
            architecture="SpatialHealthyHead",
            model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            calibration=calibration,
            epoch=epoch,
            protocol=protocol,
        )
        atomic_save(state, out / ("epoch_" + str(epoch) + ".pt"))
        np.save(out / ("epoch_" + str(epoch) + "_logits.npy"), logits)
        if best is None or ranking(selected) > ranking(best):
            best = selected
            atomic_save(state, out / "best.pt")
            (out / "candidate.json").write_text(json.dumps(best, indent=2))
        record = dict(
            epoch=epoch,
            loss=total / batches,
            elapsed_seconds=time.monotonic() - started,
            checks=selected["checks"],
            healthy={
                s: [m["healthy"]["correct"], m["healthy"]["n"]]
                for s, m in selected["internal_metrics"].items()
            },
            cutoff=selected["leaf_cutoff"],
        )
        history.append(record)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(record), flush=True)
        passing = passing + 1 if selected["passed"] else 0
        if passing >= 2:
            break
    state = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    restored = SpatialHealthyHead().cuda().eval()
    restored.load_state_dict(state["model"], strict=True)
    with torch.inference_mode():
        actual = torch.cat([restored(x) for x in maps.split(512)]).cpu().numpy()
    saved = np.load(out / ("epoch_" + str(state["epoch"]) + "_logits.npy"))
    error = float(np.max(np.abs(actual - saved)))
    assert error == 0.0
    result = dict(
        protocol=protocol,
        best=best,
        passing=sum(r["passed"] for r in ranked),
        evaluated_policies=len(ranked),
        checkpoint_sha256=digest(out / "best.pt"),
        selected_epoch=state["epoch"],
        restored_head_error=error,
        elapsed_seconds=time.monotonic() - started,
        encoder_weights_trained=False,
        cold_olid_used=False,
    )
    (out / "ranked.json").write_text(json.dumps(ranked, indent=2))
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            dict(
                passing=result["passing"],
                selected_epoch=state["epoch"],
                elapsed_seconds=result["elapsed_seconds"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
