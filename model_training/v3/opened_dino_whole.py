"""Whole-primary confirmation from the adapted DINO, reusing exact input scores."""

from pathlib import Path
import concurrent.futures
import json
import time
import numpy as np
from PIL import Image
import torch

from app_eval import classifier_letterbox_input
from evaluation_inputs import input_identity
from feature_cache_reuse import digest, preprocessing_contract, runtime_identity
from opened_context_data import read
from partial_dino_cache import input_rows, original_dino
from partial_dino_healthy import (
    PartialDinoHealthyTail,
    healthy_margin,
    prefix_features,
    prefix_identity,
)
from phase28_roi_confirmation import pixel_key


def prepare(
    campaign,
    data,
    paths,
    namespace="phase53_whole_dino",
    head_namespace="phase51_dino_adaptation",
):
    started = time.monotonic()
    p = Path(campaign)
    assert namespace in (
        "phase53_whole_dino",
        "phase54_dino_conflict",
        "phase69_whole_local",
    )
    assert head_namespace in ("phase51_dino_adaptation", "phase55_dino_continuation")
    root = p / namespace / "cache"
    root.mkdir(parents=True, exist_ok=True)
    head = p / head_namespace / "head"
    result = read(head / "result.json")
    assert result["checkpoint_sha256"] == digest(head / "candidate.pt")
    assert result["probabilities_sha256"] == digest(head / "probabilities.npy")
    paths = sorted(set(paths))
    identity = input_identity(
        [dict(data["photos"][path], id=path, path=path) for path in paths]
    )
    protocol = dict(
        checkpoint_sha256=result["checkpoint_sha256"],
        input_identity=identity,
        data_identity=data["identity"],
        input_scope="Whole primary frame, exact canonical letterbox224, never a fabricated leaf",
        preprocessing=preprocessing_contract(
            Path(__file__).resolve().parents[2], "whole_dino"
        ),
        runtime=runtime_identity(),
        source_sha256=digest(__file__),
        model_source_sha256=digest(Path(__file__).with_name("partial_dino_healthy.py")),
        cached_scores_sha256=result["probabilities_sha256"],
        cold_olid_used=False,
        ad_independent=False,
    )
    frozen = root / "protocol.json"
    if frozen.exists():
        assert read(frozen) == protocol
    else:
        frozen.write_text(json.dumps(protocol, indent=2))
    if (root / "complete.json").exists():
        complete = read(root / "complete.json")
        assert complete["protocol_sha256"] == digest(frozen)
        assert complete["probabilities_sha256"] == digest(root / "probabilities.json")
        assert complete["inputs_sha256"] == digest(root / "inputs.json")
        return read(root / "probabilities.json"), dict(
            complete, new_encoder_inference_this_call=0
        )
    existing = input_rows(p)
    scores = np.load(head / "probabilities.npy")
    assert len(existing) == len(scores)
    known = {r["input_sha256"]: float(score) for r, score in zip(existing, scores)}
    imported = 0
    previous = p / "phase53_whole_dino/cache"
    if (
        namespace != "phase53_whole_dino"
        and head_namespace == "phase51_dino_adaptation"
        and (previous / "complete.json").exists()
    ):
        complete = read(previous / "complete.json")
        assert complete["protocol_sha256"] == digest(previous / "protocol.json")
        assert complete["probabilities_sha256"] == digest(
            previous / "probabilities.json"
        )
        assert complete["inputs_sha256"] == digest(previous / "inputs.json")
        prior = read(previous / "protocol.json")
        for field in (
            "checkpoint_sha256",
            "preprocessing",
            "runtime",
            "model_source_sha256",
        ):
            assert prior[field] == protocol[field]
        prior_scores = read(previous / "probabilities.json")
        for row in read(previous / "inputs.json"):
            value = float(prior_scores[row["path"]])
            assert np.isfinite(value) and 0 <= value <= 1
            if row["input_sha256"] in known:
                assert abs(known[row["input_sha256"]] - value) < 1e-5
            else:
                known[row["input_sha256"]] = value
                imported += 1
    descriptors = []
    tensors = {}

    def image(path):
        with Image.open(path) as im:
            value = classifier_letterbox_input(np.asarray(im.convert("RGB")), 224)
        return dict(path=path, input_sha256=pixel_key(value)), value

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for row, value in pool.map(image, paths):
            descriptors.append(row)
            if row["input_sha256"] not in known:
                tensors.setdefault(row["input_sha256"], value)
    keys = sorted(tensors)
    if keys:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = (
            PartialDinoHealthyTail(original_dino(p)).cuda().eval().requires_grad_(False)
        )
        state = torch.load(head / "candidate.pt", weights_only=True, map_location="cpu")
        model.load_state_dict(state["model"], strict=True)
        assert prefix_identity(model.backbone) == state["prefix_state_sha256"]
        assert state["calibration"] == result["calibration"]
        with torch.inference_mode():
            for start in range(0, len(keys), 64):
                part = keys[start : start + 64]
                batch = (
                    torch.from_numpy(np.concatenate([tensors[key] for key in part], 0))
                    .cuda()
                    .permute(0, 3, 1, 2)
                )
                z = (
                    healthy_margin(model(prefix_features(model.backbone, batch)))
                    .cpu()
                    .numpy()
                )
                probability = 1 / (
                    1
                    + np.exp(
                        -np.clip(
                            z * result["calibration"][0] + result["calibration"][1],
                            -80,
                            80,
                        )
                    )
                )
                known.update(
                    {key: float(score) for key, score in zip(part, probability)}
                )
    output = {row["path"]: known[row["input_sha256"]] for row in descriptors}
    (root / "probabilities.json").write_text(json.dumps(output, indent=2))
    (root / "inputs.json").write_text(json.dumps(descriptors, indent=2))
    complete = dict(
        protocol_sha256=digest(frozen),
        probabilities_sha256=digest(root / "probabilities.json"),
        inputs_sha256=digest(root / "inputs.json"),
        inputs=len(paths),
        reused_inputs=sum(row["input_sha256"] not in tensors for row in descriptors),
        new_unique_encoder_inputs=len(keys),
        imported_prior_unique_scores=imported,
        elapsed_seconds=time.monotonic() - started,
        cold_olid_used=False,
    )
    (root / "complete.json").write_text(json.dumps(complete, indent=2))
    return output, dict(complete, new_encoder_inference_this_call=len(keys))
