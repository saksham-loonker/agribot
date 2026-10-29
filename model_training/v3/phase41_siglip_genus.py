"""Fit a whole-primary-image crop verifier from explicit publisher taxonomy.

PlantDoc images with conflicting annotated crop genera are masked. All other
admitted sources explicitly concern tomato. This does not reuse disease labels
as genus targets, change nine-class weights, or label any detector ROI.
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


def targets(rows, csv_path):
    taxonomy = collections.defaultdict(set)
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        for annotation in csv.DictReader(stream):
            label = annotation["class"].strip()
            assert label
            taxonomy[annotation["filename"]].add(
                int(label.lower().startswith("tomato"))
            )
    admitted = {
        "tomatovillage",
        "bangladesh_binary",
        "bangladesh_classification",
        "pakistan",
        "swtt5wy2pr",
        "tlid",
        "tom2024",
        "ad_greenhouse",
    }
    result = []
    for row in rows:
        if row.get("box") is not None:
            result.append(-1)
            continue
        if row["source"] == "plantdoc":
            values = taxonomy.get(Path(row["path"]).name, set())
            result.append(next(iter(values)) if len(values) == 1 else -1)
        else:
            result.append(1 if row["source"] in admitted else -1)
    return np.array(result, np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    args = ap.parse_args()
    p = Path(args.campaign)
    root = p / "phase23_siglip"
    out = p / "phase41_siglip_genus"
    out.mkdir(exist_ok=True)
    data = load(p)
    cache = prepare(p, data)
    rows = read(root / "cnn_inputs/rows.json")
    csv_path = Path("/home/genesis/agribot-data/PlantDoc/train_labels.csv")
    y = targets(rows, csv_path)
    selected = np.flatnonzero(y >= 0)
    fitting = [dict(rows[i], y=int(y[i])) for i in selected]
    protocol = dict(
        task="Primary-view tomato versus explicit non-tomato publisher crop taxonomy",
        source_rows_sha256=digest(root / "cnn_inputs/rows.json"),
        taxonomy_csv_sha256=digest(csv_path),
        features_sha256=digest(root / "siglip2_feature_heads/features.npy"),
        opened_feature_cache_sha256=digest(cache / "complete.json"),
        mixed_or_missing_taxonomy_masked=int((y < 0).sum()),
        head="Linear normalized original frozen SigLIP768",
        epochs=60,
        seed=1511,
        learning_rate=0.001,
        selection="Grouped primary-image genus validation weighted BCE, no disease-model targets",
        sampling="class/source/family",
        restricted_teacher_used=False,
        cold_olid_used=False,
        ad_independent=False,
        script_sha256=digest(__file__),
    )
    frozen = out / "protocol.json"
    if frozen.exists():
        assert read(frozen) == protocol
    else:
        frozen.write_text(json.dumps(protocol, indent=2))
    if (out / "result.json").exists():
        print((out / "result.json").read_text())
        return
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(1511)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    values = torch.from_numpy(
        np.load(root / "siglip2_feature_heads/features.npy")[selected]
    ).cuda()
    target = np.array([r["y"] for r in fitting])
    truth = torch.tensor(target, dtype=torch.float32, device="cuda")
    train = np.array([i for i, r in enumerate(fitting) if r["split"] == "train"])
    val = np.array([i for i, r in enumerate(fitting) if r["split"] == "val"])
    assert not (
        {fitting[i]["group"] for i in train} & {fitting[i]["group"] for i in val}
    )
    assert set(target[train]) == set(target[val]) == {0, 1}
    sampling = torch.tensor(
        source_class_group_weights(fitting, train), dtype=torch.float64
    )
    vw = source_class_group_weights(fitting, val)
    weights = torch.tensor(vw, dtype=torch.float32, device="cuda")
    model = LeafFeatureHead("linear", 768).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    best = float("inf")
    for epoch in range(60):
        model.train()
        order = torch.multinomial(
            sampling,
            len(train),
            replacement=True,
            generator=torch.Generator().manual_seed(151100 + epoch),
        )
        for positions in order.split(1024):
            ids = train[positions.numpy()]
            loss = F.binary_cross_entropy_with_logits(model(values[ids]), truth[ids])
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
                    * weights
                ).sum()
            )
        if score < best:
            best = score
            atomic_save(
                dict(
                    kind="linear",
                    model={
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    },
                    epoch=epoch,
                    protocol=protocol,
                ),
                out / "best.pt",
            )
    state = torch.load(out / "best.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    with torch.inference_mode():
        z = model(values[val]).cpu().numpy()
    calibration = calibrate(z, target[val], vw)
    opened = torch.from_numpy(np.load(cache / "features.npy")).cuda()
    with torch.inference_mode():
        scores = (
            (model(opened) * calibration[0] + calibration[1]).sigmoid().cpu().numpy()
        )
    paths = [r["path"] for r in read(cache / "inputs.json")]
    probability = dict(zip(paths, map(float, scores)))
    calibrated = 1 / (
        1 + np.exp(-np.clip(z * calibration[0] + calibration[1], -80, 80))
    )
    validation = {}
    for source in sorted({r["source"] for r in fitting}):
        indices = np.array(
            [i for i, index in enumerate(val) if fitting[index]["source"] == source]
        )
        validation[source] = dict(
            n=len(indices),
            correct_at_half=int(
                ((calibrated[indices] >= 0.5) == target[val][indices]).sum()
            ),
            tomato_above_095=int(
                ((target[val][indices] == 1) & (calibrated[indices] >= 0.95)).sum()
            ),
            tomato_n=int((target[val][indices] == 1).sum()),
        )
    result = dict(
        protocol=protocol,
        checkpoint_sha256=digest(out / "best.pt"),
        calibration=calibration,
        selected_epoch=state["epoch"],
        training_seconds=time.monotonic() - started,
        validation_by_source=validation,
        encoder_weights_trained=False,
        cold_olid_used=False,
    )
    (out / "probabilities.json").write_text(json.dumps(probability))
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
