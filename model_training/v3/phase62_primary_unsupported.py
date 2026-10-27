"""Explicit unsupported tomato condition versus Healthy at primary-image scope.

Known diseases, coarse disease labels, non-tomato and ambiguous annotations are
excluded. This does not assign primary-image labels to detected leaf ROIs.
"""

from pathlib import Path
import argparse
import collections
import csv
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from campaign_cnn_fusion import calibrate
from leaf_feature_expert import LeafFeatureHead
from opened_context_data import load, read
from opened_siglip_features import prepare
from permissive_encoder import digest
from source_balance import source_class_group_weights
from training_support import atomic_save

UNSUPPORTED = {
    "Tomato mold leaf",
    "Tomato leaf yellow virus",
    "Tomato leaf mosaic virus",
    "Tomato leaf bacterial spot",
    "Tomato Septoria leaf spot",
    "Tomato two spotted spider mites leaf",
}


def publisher_target(annotations):
    if not annotations:
        return -1
    decisions = {
        0 if label == "Tomato leaf" else 1 if label in UNSUPPORTED else -1
        for label in annotations
    }
    return next(iter(decisions)) if len(decisions) == 1 else -1


def targets(rows, old_rows, csv_paths):
    taxonomy = {}
    for folder, path in csv_paths.items():
        by_name = collections.defaultdict(set)
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                by_name[Path(row["filename"]).name].add(row["class"].strip())
        taxonomy[folder] = by_name
    explicit = collections.defaultdict(set)
    for row in old_rows:
        raw = row.get("raw_label")
        if (
            row["label"] == "Other"
            and isinstance(raw, str)
            and raw.strip()
            and raw.strip().casefold()
            not in {"diseased_unspecified", "diseased", "unknown", "unhealthy"}
        ):
            explicit[row["path"]].add(raw.strip())
    result = []
    for row in rows:
        if row.get("box") is not None:
            target = -1
        elif row["source"] == "plantdoc":
            path = Path(row["path"])
            target = publisher_target(
                taxonomy.get(path.parent.name.upper(), {}).get(path.name, set())
            )
        elif row["label"] == "Healthy":
            target = 0
        else:
            target = 1 if explicit[row["path"]] else -1
        result.append(target)
    return np.array(result, np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--head", choices=("linear", "mlp"), default="linear")
    args = ap.parse_args()
    p = Path(args.campaign)
    root = p / "phase23_siglip"
    out = p / "phase62_primary_unsupported"
    if args.head == "mlp":
        out = p / "phase65_primary_unsupported_mlp"
    out.mkdir(exist_ok=True)
    data = load(p)
    cache = prepare(p, data)
    rows = read(root / "cnn_inputs/rows.json")
    old_path = p / "phase10_leaf/cnn_inputs/rows.json"
    csv_paths = {
        name: Path("/home/genesis/agribot-data/PlantDoc") / file
        for name, file in [("TRAIN", "train_labels.csv"), ("TEST", "test_labels.csv")]
    }
    target = targets(rows, read(old_path), csv_paths)
    selected = np.flatnonzero(target >= 0)
    fitting = [dict(rows[i], y=int(target[i])) for i in selected]
    assert all(
        any(
            r["source"] == "plantdoc" and r["y"] == 1 and r["split"] == split
            for r in fitting
        )
        for split in ("train", "val")
    ), "Publisher unsupported targets missing: inspect path/taxonomy join before fitting"
    protocol = dict(
        hypothesis="An explicit unsupported-condition versus Healthy primary-image task excludes easy known-disease negatives and avoids ROI label-scope mismatch",
        source_rows_sha256=digest(root / "cnn_inputs/rows.json"),
        original_label_rows_sha256=digest(old_path),
        publisher_taxonomy_sha256={k: digest(v) for k, v in csv_paths.items()},
        unsupported_publisher_classes=sorted(UNSUPPORTED),
        features_sha256=digest(root / "siglip2_feature_heads/features.npy"),
        opened_cache_sha256=digest(cache / "complete.json"),
        script_sha256=digest(__file__),
        labels=dict(
            collections.Counter(str((r["source"], r["y"], r["split"])) for r in fitting)
        ),
        mask_scope="Known diseases, non-tomato, missing and mixed taxonomy, generic coarse disease and every ROI",
        head=args.head + " normalized original frozen SigLIP768",
        seed=1621,
        epochs=60,
        minimum_epochs=10,
        early_stop_patience=5,
        learning_rate=0.001,
        weight_decay=0.01,
        loss="Binary unsupported probability; Healthy false-veto cost 4",
        selection="Source/class/family weighted validation BCE with Healthy cost 4; no app-output checkpoint selection",
        later_policy="One binary .5 unsupported veto on new Healthy proposals; original verdict retained on conflict",
        maximum_policies=1,
        encoder_weights_trained=False,
        new_encoder_inference=0,
        cold_olid_used=False,
        ad_independent=False,
    )
    frozen = out / "protocol.json"
    if frozen.exists():
        assert read(frozen) == protocol
    else:
        frozen.write_text(json.dumps(protocol, indent=2))
    if (out / "result.json").exists():
        result = read(out / "result.json")
        assert result["protocol"] == protocol
        assert result["checkpoint_sha256"] == digest(out / "best.pt")
        assert result["probabilities_sha256"] == digest(out / "probabilities.json")
        print(json.dumps(result), flush=True)
        return
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(1621)
    values = torch.from_numpy(
        np.load(root / "siglip2_feature_heads/features.npy")[selected]
    )
    y = np.array([r["y"] for r in fitting])
    truth = torch.from_numpy(y.astype(np.float32))
    train = np.array([i for i, r in enumerate(fitting) if r["split"] == "train"])
    val = np.array([i for i, r in enumerate(fitting) if r["split"] == "val"])
    assert not (
        {fitting[i]["group"] for i in train} & {fitting[i]["group"] for i in val}
    )
    assert not ({fitting[i]["path"] for i in train} & {fitting[i]["path"] for i in val})
    assert set(y[train]) == set(y[val]) == {0, 1}
    sampling = torch.tensor(
        source_class_group_weights(fitting, train), dtype=torch.float64
    )
    vw = source_class_group_weights(fitting, val)
    weights = torch.tensor(vw, dtype=torch.float32)
    model = LeafFeatureHead(args.head, 768)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    best, stale, history = float("inf"), 0, []
    for epoch in range(60):
        if epoch >= 10 and stale >= 5:
            break
        model.train()
        order = torch.multinomial(
            sampling,
            len(train),
            replacement=True,
            generator=torch.Generator().manual_seed(162100 + epoch),
        )
        for positions in order.split(1024):
            ids = train[positions.numpy()]
            loss = (
                F.binary_cross_entropy_with_logits(
                    model(values[ids]), truth[ids], reduction="none"
                )
                * torch.where(truth[ids] == 0, 4.0, 1.0)
            ).mean()
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            score = float(
                (
                    F.binary_cross_entropy_with_logits(
                        model(values[val]), truth[val], reduction="none"
                    )
                    * torch.where(truth[val] == 0, 4.0, 1.0)
                    * weights
                ).sum()
            )
        if score < best - 1e-5:
            best, stale = score, 0
            atomic_save(
                dict(
                    kind=args.head,
                    model={
                        k: v.detach().clone() for k, v in model.state_dict().items()
                    },
                    epoch=epoch,
                    protocol=protocol,
                ),
                out / "best.pt",
            )
        else:
            stale += 1
        history.append(dict(epoch=epoch, validation_loss=score))
    state = torch.load(out / "best.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    with torch.inference_mode():
        z = model(values[val]).numpy()
    calibration_weights = vw * np.where(y[val] == 0, 4.0, 1.0)
    calibration_weights /= calibration_weights.sum()
    calibration = calibrate(z, y[val], calibration_weights)
    opened = torch.from_numpy(np.load(cache / "features.npy"))
    with torch.inference_mode():
        probabilities = (
            (model(opened) * calibration[0] + calibration[1]).sigmoid().numpy()
        )
    paths = [r["path"] for r in read(cache / "inputs.json")]
    (out / "probabilities.json").write_text(
        json.dumps(dict(zip(paths, map(float, probabilities))))
    )
    (out / "history.json").write_text(json.dumps(history, indent=2))
    result = dict(
        protocol=protocol,
        checkpoint_sha256=digest(out / "best.pt"),
        probabilities_sha256=digest(out / "probabilities.json"),
        calibration=calibration,
        selected_epoch=state["epoch"],
        epochs=len(history),
        elapsed_seconds=time.monotonic() - started,
        new_encoder_inference=0,
        cold_olid_used=False,
        ad_independent=False,
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
