"""Identity-checked context scores on opened development data only."""

from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from campaign_heads import app_evaluate
from campaign_reference import safe_logits
from cache_first_screen import cached_score_identity
from feature_cache_reuse import validate_completed_contract
from leaf_feature_expert import LeafFeatureHead
from permissive_encoder import digest


def read(path):
    return json.loads(Path(path).read_text())


def load(campaign, negative_mode="cvar_bce"):
    p = Path(campaign)
    root = p / "phase23_siglip"
    run = root / "sft_ad_tail2_focal_lowfpr"
    fit = read(run / "result.json")
    assert fit["checkpoint_sha256"] == digest(run / "best.pt")
    assert read(run / "restored_model_parity.json")["passed"]
    rows = read(root / "cnn_inputs/rows.json")
    assert digest(root / "cnn_inputs/rows.json") == fit["configuration"]["rows_sha256"]
    leaf = p / "phase10_leaf"
    manifest = read(p / "export_cnn_safe/model_manifest.json")
    base = {
        k: v["rows"]
        for k, v in app_evaluate(
            safe_logits(leaf),
            read(leaf / "cnn_inputs/rows.json"),
            read(leaf / "eval_rows.json"),
            manifest,
            True,
        ).items()
    }
    paths = sorted({r["path"] for values in base.values() for r in values})
    internal_path = p / "phase23_data/internal_baseline_predictions.jsonl"
    internal = [json.loads(line) for line in internal_path.read_text().splitlines()]
    whole_path = p / "phase23_data/whole_reference.jsonl"
    whole = {r["path"]: r for r in map(json.loads, whole_path.read_text().splitlines())}
    for values in list(base.values()) + [internal]:
        for row in values:
            if row["app_replay"].get("probabilities") is None:
                row["context_whole_reference_probabilities"] = whole[row["path"]][
                    "whole_probabilities"
                ]
    protocol = read(p / "phase27_cache_screen/protocol.json")
    for actual, expected in [
        (digest(run / "best.pt"), protocol["checkpoint_sha256"]),
        (digest(run / "validation_logits.npz"), protocol["validation_logits_sha256"]),
        (digest(internal_path), protocol["internal_baseline_sha256"]),
    ]:
        assert actual == expected
    values = np.load(run / "validation_logits.npz")
    calibrated = {
        rows[int(i)]["path"]: float(z) * fit["calibration"][0] + fit["calibration"][1]
        for i, z in zip(values["indices"], values["logits"])
    }
    calibrated.update(read(p / "phase27_cache_screen/supervised_missing_logits.json"))
    positive = {
        path: float(1 / (1 + np.exp(-np.clip(z, -80, 80))))
        for path, z in calibrated.items()
    }
    directory = root / "risk_heads" / negative_mode
    selection = root / ("risk_" + negative_mode + "_selection")
    cached_score_identity(directory, selection, paths)
    old_negative = np.load(selection / "probabilities.npz")["healthy"][:, 0]
    feature_directory = root / "siglip2_feature_heads"
    validate_completed_contract(root, feature_directory, "siglip2")
    feature_identity = read(feature_directory / "feature_identity.json")
    state = torch.load(directory / "best.pt", weights_only=True, map_location="cpu")
    assert state["protocol"]["feature_identity"] == feature_identity
    features = np.load(feature_directory / "features.npy", mmap_mode="r")
    row_index = {r["path"]: i for i, r in enumerate(rows)}
    internal_indices = [row_index[r["path"]] for r in internal]
    head = LeafFeatureHead(state["kind"], 768).eval()
    head.load_state_dict(state["model"], strict=True)
    calibration = read(directory / "result.json")["calibration"]
    with torch.inference_mode():
        probabilities = (
            (
                head(torch.from_numpy(np.array(features[internal_indices], copy=True)))
                * calibration[0]
                + calibration[1]
            )
            .sigmoid()
            .numpy()
        )
    negative = dict(zip(paths, map(float, old_negative)))
    negative.update({r["path"]: float(z) for r, z in zip(internal, probabilities)})
    genus_identity = read(root / "selection/input_identity.json")
    assert genus_identity["genus_checkpoint_sha256"] == digest(
        p / "phase6_roi/species/best.pt"
    )
    assert genus_identity["genus_calibration_sha256"] == digest(
        p / "phase6_roi/species/result.json"
    )
    assert read(root / "selection_validation/result.json")[
        "internal_baseline_sha256"
    ] == digest(internal_path)
    genus = dict(
        zip(paths, map(float, np.load(root / "selection/probabilities.npz")["genus"]))
    )
    genus.update(
        {
            r["path"]: float(z)
            for r, z in zip(
                internal,
                np.load(root / "selection_validation/internal_probabilities.npz")[
                    "genus"
                ],
            )
        }
    )
    all_rows = {
        r["path"]: r for values in list(base.values()) + [internal] for r in values
    }
    assert not any("phase24_olid_source" in path for path in all_rows)
    assert set(all_rows) <= set(positive) and set(all_rows) <= set(negative)
    return dict(
        base=base,
        internal=internal,
        photos=all_rows,
        rows=rows,
        paths=paths,
        manifest=manifest,
        positive=positive,
        negative=negative,
        genus=genus,
        calibrated_logits=calibrated,
        supervised_run=run,
        identity=dict(
            supervised_checkpoint_sha256=fit["checkpoint_sha256"],
            rows_sha256=fit["configuration"]["rows_sha256"],
            internal_baseline_sha256=digest(internal_path),
            baseline_manifest_sha256=digest(p / "export_cnn_safe/model_manifest.json"),
            negative_checkpoint_sha256=digest(directory / "best.pt"),
            negative_calibration_sha256=digest(directory / "result.json"),
            negative_mode=negative_mode,
            whole_reference_sha256=digest(whole_path),
            old_paths_sha256=hashlib.sha256(json.dumps(paths).encode()).hexdigest(),
            cold_olid_used=False,
            ad_independent=False,
        ),
    )
