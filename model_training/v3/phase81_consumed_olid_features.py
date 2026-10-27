"""Extend the exact original SigLIP cache with consumed OLID images only."""

from pathlib import Path
import concurrent.futures
import hashlib
import json
import numpy as np
from PIL import Image
import timm
import torch

from app_eval import classifier_letterbox_input
from feature_cache_reuse import preprocessing_contract, runtime_identity
from leaf_feature_expert import LeafFeatureHead
from opened_context_data import read
from opened_siglip_features import prepare as prepare_opened
from permissive_encoder import CanonicalPermissiveEncoder, digest
from phase28_roi_confirmation import pixel_key


def prepare(campaign, data):
    p = Path(campaign)
    assert read(p / "phase80_cached_isolated_head/result.json")["passing"] == 0
    assert not read(p / "phase76_fresh_acceptance/result.json")["passed"]
    old = prepare_opened(p, data)
    source = p / "phase23_siglip/siglip2_feature_heads/encoder.pt"
    root = p / "phase81_consumed_olid_features"
    root.mkdir(exist_ok=True)
    rows = read(p / "phase24_olid_source/sealed_manifest.json")
    old_input = read(old / "inputs.json")
    old_values = np.load(old / "features.npy", mmap_mode="r")
    all_features = {r["path"]: old_values[i] for i, r in enumerate(old_input)}
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timm.layers.set_fused_attn(False)
    protocol = dict(
        encoder_sha256=digest(source),
        old_cache_sha256=digest(old / "complete.json"),
        source_manifest_sha256=digest(p / "phase24_olid_source/sealed_manifest.json"),
        source_predictions_sha256=digest(
            p / "phase76_fresh_acceptance/mobile_scores.jsonl"
        ),
        preprocessing=preprocessing_contract(Path(__file__).parents[2], "siglip2"),
        runtime=runtime_identity(),
        source_sha256=digest(__file__),
        role="AD and OLID consumed development; not independent acceptance",
        expected_images=730,
        encoder_weights_changed=False,
    )
    frozen = root / "protocol.json"
    if frozen.exists():
        assert read(frozen) == protocol
    else:
        frozen.write_text(json.dumps(protocol, indent=2))
    completed = root / "complete.json"
    feature_path = root / "features.npy"
    if completed.exists():
        result = read(completed)
        assert result["protocol_sha256"] == digest(frozen)
        assert digest(feature_path) == result["features_sha256"]
        assert digest(root / "inputs.json") == result["inputs_sha256"]
    else:
        state = torch.load(source, weights_only=True, map_location="cpu")
        model = (
            CanonicalPermissiveEncoder(state["spec"])
            .cuda()
            .eval()
            .requires_grad_(False)
        )
        model.load_state_dict(state["model"], strict=True)
        progress = root / "progress.json"
        descriptors = read(progress)["inputs"] if progress.exists() else []
        if progress.exists():
            assert read(progress)["protocol_sha256"] == digest(frozen)
            values = np.lib.format.open_memmap(feature_path, mode="r+")
        else:
            values = np.lib.format.open_memmap(
                feature_path, mode="w+", dtype=np.float32, shape=(len(rows), 768)
            )
        reusable = {r["input_sha256"]: old_values[i] for i, r in enumerate(old_input)}
        newly_computed = sum(r["computed"] for r in descriptors)

        def prepare_image(row):
            assert digest(row["path"]) == row["file_sha256"]
            with Image.open(row["path"]) as image:
                pixels = classifier_letterbox_input(
                    np.asarray(image.convert("RGB")), 224
                )
            return (
                dict(
                    path=row["path"],
                    file_sha256=row["file_sha256"],
                    input_sha256=pixel_key(pixels),
                ),
                pixels,
            )

        with torch.inference_mode(), concurrent.futures.ThreadPoolExecutor(8) as pool:
            for start in range(len(descriptors), len(rows), 32):
                batch = list(pool.map(prepare_image, rows[start : start + 32]))
                missing = [
                    i
                    for i, (record, _) in enumerate(batch)
                    if record["input_sha256"] not in reusable
                ]
                if missing:
                    pixels = torch.from_numpy(
                        np.concatenate([batch[i][1] for i in missing], 0)
                    ).cuda()
                    outputs = model(pixels).cpu().numpy()
                    assert np.isfinite(outputs).all()
                    for i, value in zip(missing, outputs):
                        reusable[batch[i][0]["input_sha256"]] = value
                    newly_computed += len(missing)
                for index, (record, _) in enumerate(batch):
                    record["computed"] = index in missing
                    values[start + index] = reusable[record["input_sha256"]]
                    descriptors.append(record)
                values.flush()
                temporary = progress.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(dict(protocol_sha256=digest(frozen), inputs=descriptors))
                )
                temporary.replace(progress)
                print(
                    json.dumps(
                        dict(
                            stage="new_consumed_olid_features",
                            completed=len(descriptors),
                            total=len(rows),
                            new_encoder_inputs=newly_computed,
                        )
                    ),
                    flush=True,
                )
        del model
        torch.cuda.empty_cache()
        (root / "inputs.json").write_text(json.dumps(descriptors))
        scores = {
            r["id"]: r
            for r in map(
                json.loads,
                (p / "phase76_fresh_acceptance/mobile_scores.jsonl")
                .read_text()
                .splitlines(),
            )
        }
        all_values = torch.from_numpy(np.asarray(values))
        parity = {}
        for name, folder, column in [
            ("negative", "phase23_siglip/risk_heads/cvar_bce", 1),
            ("genus", "phase41_siglip_genus", 2),
            ("natural", "phase39_context_router", 3),
        ]:
            state = torch.load(
                p / folder / "best.pt", weights_only=True, map_location="cpu"
            )
            head = LeafFeatureHead(state["kind"], 768).eval()
            head.load_state_dict(state["model"], strict=True)
            calibration = read(p / folder / "result.json")["calibration"]
            with torch.inference_mode():
                actual = (
                    (head(all_values) * calibration[0] + calibration[1])
                    .sigmoid()
                    .numpy()
                )
            expected = np.array(
                [scores[r["id"]]["candidate"]["whole"][column] for r in rows]
            )
            parity[name] = float(np.max(np.abs(actual - expected)))
        assert max(parity.values()) <= 0.0002, (
            "Frozen feature/native score identity mismatch",
            parity,
        )
        result = dict(
            protocol_sha256=digest(frozen),
            features_sha256=digest(feature_path),
            inputs_sha256=digest(root / "inputs.json"),
            newly_computed=newly_computed,
            reused_old_images=len(old_input),
            new_images=len(rows),
            original_head_mobile_parity=parity,
            olid_independent=False,
            ad_independent=False,
        )
        completed.write_text(json.dumps(result, indent=2))
    values = np.load(feature_path, mmap_mode="r")
    for index, row in enumerate(rows):
        assert digest(row["path"]) == row["file_sha256"]
        all_features[row["path"]] = values[index]
    return all_features, read(completed)
