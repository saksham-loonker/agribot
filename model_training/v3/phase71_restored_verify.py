"""Verify restored shared models and exact frozen-route verdicts before export."""

from pathlib import Path
import argparse
import concurrent.futures
import copy
import hashlib
import io
import json
import runpy
import shutil
import sys
import time

import numpy as np
from PIL import Image
import torch
import timm

from app_eval import classifier_letterbox_input, replay
from evaluation_inputs import input_identity
from feature_cache_reuse import preprocessing_contract, runtime_identity
from leaf_feature_expert import LeafFeatureHead
from opened_context_data import load, read
from opened_guard_screen import OpenedGuardScreen
from partial_dino_cache import CanonicalInputs, input_rows
from permissive_encoder import digest
from phase29_compact_prepare import region_key
from phase42_scoped_screen import calibrated_head
from qualified_recognition_models import restore
from qualified_recognition_policy import route


class ReferenceCaptured(Exception):
    pass


def freeze(path, value):
    if path.exists():
        assert read(path) == value, ("Frozen verification changed", str(path))
    else:
        path.write_text(json.dumps(value, indent=2))


def reference(campaign, out):
    p = Path(campaign)
    if (out / "reference.json").exists():
        return load(p), read(out / "reference.json")
    captured = {}
    evaluate = OpenedGuardScreen.evaluate

    def capture(screen, policy):
        captured["data"] = screen.data
        captured["verdicts"] = {
            name: [policy(row)["app_replay"] for row in rows]
            for name, rows in list(screen.data["base"].items())
            + [("internal", screen.data["internal"])]
        }
        raise ReferenceCaptured

    OpenedGuardScreen.evaluate = capture
    arguments = sys.argv
    sys.argv = [
        "phase44_consistent_scope.py",
        "--campaign",
        str(p),
        "--paper-extension",
        "--dino-tail",
        "--continued-dino",
        "--whole-local-corroboration",
    ]
    try:
        runpy.run_path(
            str(
                p
                / "phase69_corroboration_source/model_training/v3/phase44_consistent_scope.py"
            ),
            run_name="__main__",
        )
    except ReferenceCaptured:
        pass
    finally:
        OpenedGuardScreen.evaluate = evaluate
        sys.argv = arguments
    assert captured
    (out / "reference.json").write_text(json.dumps(captured["verdicts"]))
    return captured["data"], captured["verdicts"]


def infer(model, identifiers, loader, out, name, width, contract_sha, device="cuda"):
    path = out / (name + ".npy")
    progress = out / (name + "_progress.json")
    start = 0
    if progress.exists():
        previous = read(progress)
        assert previous["protocol_sha256"] == contract_sha and path.exists()
        start = previous["complete"]
    values = np.lib.format.open_memmap(
        path,
        mode="r+" if path.exists() else "w+",
        dtype=np.float32,
        shape=(len(identifiers), width),
    )
    assert 0 <= start <= len(identifiers)
    with concurrent.futures.ThreadPoolExecutor(8) as pool, torch.inference_mode():
        for index in range(start, len(identifiers), 64):
            ids = identifiers[index : index + 64]
            tensor = torch.from_numpy(
                np.concatenate(list(pool.map(loader, ids)), 0)
            ).to(device)
            result = model(tensor).cpu().numpy()
            assert result.shape == (len(ids), width) and np.isfinite(result).all()
            values[index : index + len(ids)] = result
            values.flush()
            temporary = progress.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    dict(protocol_sha256=contract_sha, complete=index + len(ids))
                )
            )
            temporary.replace(progress)
    return values


class MobileModel:
    def __init__(self, path, threads):
        from ai_edge_litert.interpreter import Interpreter

        self.runtime = Interpreter(model_path=str(path), num_threads=threads)
        self.runtime.allocate_tensors()
        self.input = self.runtime.get_input_details()[0]["index"]
        self.output = self.runtime.get_output_details()[0]["index"]

    def __call__(self, tensor):
        result = []
        for value in tensor.numpy():
            self.runtime.set_tensor(self.input, value[None])
            self.runtime.invoke()
            result.append(self.runtime.get_tensor(self.output))
        return torch.from_numpy(np.concatenate(result, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--mobile", action="store_true")
    args = ap.parse_args()
    p = Path(args.campaign)
    out = p / ("phase73_mobile_verified" if args.mobile else "phase71_restored")
    out.mkdir(exist_ok=True)
    candidate = read(p / "phase70_qualified/candidate.json")
    parent = read(p / "phase69_whole_local/result.json")
    assert parent["passing"] == 1 and all(parent["candidate"]["checks"].values())
    for item in candidate["bundles"].values():
        assert digest(p / "phase70_qualified" / item["file"]) == item["sha256"]
    backend = {}
    if args.mobile:
        assert read(p / "phase71_restored/result.json")["passed"]
        exported = p / "phase72_mobile_export"
        benchmark = read(exported / "thread_benchmark.json")
        for name in ("leaf", "whole"):
            proof = read(exported / (name + "_result.json"))
            assert proof["passed"] and proof["tflite_sha256"] == digest(
                exported / (name + ".tflite")
            )
            assert benchmark[name]["selected"]["exact_to_two_threads"]
            backend[name] = dict(
                file=str(exported / (name + ".tflite")),
                sha256=proof["tflite_sha256"],
                threads=benchmark[name]["selected"]["threads"],
            )
        source = p / "phase71_restored/reference.json"
        assert (
            digest(source)
            == read(p / "phase71_restored/protocol.json")["reference_sha256"]
        )
        if not (out / "reference.json").exists():
            shutil.copy2(source, out / "reference.json")
    data, expected = reference(p, out)
    tables = [
        read(p / f"phase29_compact/{name}_indices.json") for name in ("center", "full")
    ]
    keys = {
        path: [region_key(path, leaf["box"]) for leaf in row["leaves"]]
        for path, row in data["photos"].items()
    }
    indices = sorted(
        {table[key] for row in keys.values() for key in row for table in tables}
    )
    descriptors = input_rows(p)
    whole_inputs = read(p / "opened_siglip_feature_cache/inputs.json")
    paths = [r["path"] for r in whole_inputs]
    identities = input_identity([dict(id=path, path=path) for path in paths])
    actual_hashes = {r["path"]: r["sha256"] for r in identities}
    for item in whole_inputs:
        assert actual_hashes[item["path"]] == item["file_sha256"]
    whole_dino = read(p / "phase69_whole_local/cache/probabilities.json")
    protocol = dict(
        candidate_sha256=digest(p / "phase70_qualified/candidate.json"),
        parent_result_sha256=digest(p / "phase69_whole_local/result.json"),
        parent_protocol_sha256=digest(p / "phase69_whole_local/protocol.json"),
        reference_sha256=digest(out / "reference.json"),
        leaf_indices_sha256=hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
        leaf_descriptors_sha256=hashlib.sha256(
            json.dumps(descriptors).encode()
        ).hexdigest(),
        whole_input_identity=identities,
        whole_dino_inputs=sorted(whole_dino),
        model_source_sha256=digest(
            Path(__file__).with_name("qualified_recognition_models.py")
        ),
        routing_source_sha256=digest(
            Path(__file__).with_name("qualified_recognition_policy.py")
        ),
        script_sha256=digest(__file__),
        runtime=runtime_identity(),
        preprocessing=preprocessing_contract(
            Path(__file__).resolve().parents[2], "whole_dino"
        ),
        maximum_logit_error=0.002,
        maximum_probability_error=0.0002,
        verdict_rule="Every original cached candidate kind/label/count/scope must match; every unchanged gate must still pass",
        cold_olid_used=False,
        ad_independent=False,
    )
    if args.mobile:
        from importlib import metadata

        protocol.update(
            backend="Actual exported LiteRT FP32 models",
            models=backend,
            litert_runtime=metadata.version("ai-edge-litert"),
            restored_verification_sha256=digest(p / "phase71_restored/result.json"),
            thread_benchmark_sha256=digest(
                p / "phase72_mobile_export/thread_benchmark.json"
            ),
        )
    freeze(out / "protocol.json", protocol)
    contract = digest(out / "protocol.json")
    if (out / "result.json").exists():
        result = read(out / "result.json")
        assert result["protocol_sha256"] == contract
        for name, sha in result["arrays"].items():
            assert digest(out / name) == sha
        print(json.dumps(result), flush=True)
        return
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timm.layers.set_fused_attn(False)
    leaf = (
        MobileModel(backend["leaf"]["file"], backend["leaf"]["threads"])
        if args.mobile
        else restore(p / "phase70_qualified/leaf.pt").cuda()
    )
    device = "cpu" if args.mobile else "cuda"
    pixels = CanonicalInputs(p)
    try:
        leaf_values = infer(
            leaf,
            indices,
            lambda i: pixels(descriptors[i]),
            out,
            "leaf_outputs",
            12,
            contract,
            device,
        )
    finally:
        pixels.close()

    def whole_image(path):
        raw = Path(path).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == actual_hashes[path]
        with Image.open(io.BytesIO(raw)) as image:
            return classifier_letterbox_input(np.asarray(image.convert("RGB")), 224)

    dino_paths = sorted(whole_dino)
    dino_values = infer(
        leaf, dino_paths, whole_image, out, "whole_dino_outputs", 12, contract, device
    )
    dino_error = (
        float(
            np.max(
                np.abs(
                    dino_values[:, 11]
                    - np.array([whole_dino[path] for path in dino_paths])
                )
            )
        )
        if dino_paths
        else 0.0
    )
    assert dino_error <= 0.0002
    del leaf
    torch.cuda.empty_cache()
    model = (
        MobileModel(backend["whole"]["file"], backend["whole"]["threads"])
        if args.mobile
        else restore(p / "phase70_qualified/whole.pt").cuda()
    )
    whole_values = infer(
        model, paths, whole_image, out, "whole_outputs", 4, contract, device
    )
    del model
    torch.cuda.empty_cache()
    reference_logits = np.load(
        p / "phase29_compact/reference_logits.npy", mmap_mode="r"
    )[indices]
    primary_error = float(np.max(np.abs(leaf_values[:, :9] - reference_logits)))
    assert primary_error <= 0.002
    assert np.array_equal(leaf_values[:, :9].argmax(1), reference_logits.argmax(1))
    features = torch.from_numpy(
        np.load(p / "phase29_compact/features.npy", mmap_mode="r")[indices]
    )
    errors = {}
    for offset, name in [(9, "epoch_0.pt"), (10, "best.pt")]:
        state = torch.load(
            p / "phase29_compact/head_bce_source_balanced" / name,
            weights_only=True,
            map_location="cpu",
        )
        head = LeafFeatureHead(state["kind"], 1408).eval()
        head.load_state_dict(state["model"], strict=True)
        with torch.inference_mode():
            value = (
                (head(features) * state["calibration"][0] + state["calibration"][1])
                .sigmoid()
                .numpy()
            )
        errors[name] = float(np.max(np.abs(value - leaf_values[:, offset])))
        assert errors[name] <= 0.0002
    adapted = np.load(p / "phase55_dino_continuation/head/probabilities.npy")[indices]
    errors["adapted_dino"] = float(np.max(np.abs(adapted - leaf_values[:, 11])))
    assert errors["adapted_dino"] <= 0.0002
    frozen_features = torch.from_numpy(
        np.load(p / "opened_siglip_feature_cache/features.npy")
    )
    expected_whole = np.column_stack(
        [
            [data["positive"][path] for path in paths],
            [data["negative"][path] for path in paths],
            calibrated_head(p / "phase41_siglip_genus", frozen_features),
            calibrated_head(p / "phase39_context_router", frozen_features),
        ]
    )
    whole_errors = np.max(np.abs(whole_values - expected_whole), axis=0).tolist()
    assert max(whole_errors) <= 0.0002
    leaf_lookup = {key: leaf_values[i, 9:12] for i, key in enumerate(indices)}
    original_lookup = {key: leaf_values[i, :9] for i, key in enumerate(indices)}
    whole_lookup = {path: whole_values[i] for i, path in enumerate(paths)}
    dino_lookup = {path: float(dino_values[i, 11]) for i, path in enumerate(dino_paths)}

    def revised(row):
        path = row["path"]
        views = [
            [leaf_lookup[table[key]].tolist() for table in tables] for key in keys[path]
        ]
        return route(
            row,
            data["manifest"],
            whole_lookup[path],
            views,
            dino_lookup.get(path),
            data["photos"][path]["app_replay"]["kind"],
        )

    mismatches = []
    baseline_replay_mismatches = []
    checked = 0
    for name, rows in list(data["base"].items()) + [("internal", data["internal"])]:
        assert len(rows) == len(expected[name])
        for index, row in enumerate(rows):
            old = expected[name][index]
            actual = revised(row)["app_replay"]
            checked += 1
            if any(
                actual.get(field) != old.get(field)
                for field in ("kind", "label", "frames", "leaves", "partial", "scope")
            ):
                mismatches.append(
                    dict(
                        cohort=name,
                        index=index,
                        path=row["path"],
                        before=old,
                        after=actual,
                    )
                )
            original = row["leaves"]
            leaves = copy.deepcopy(original)
            for old_leaf, new_leaf in zip(original, leaves):
                key = region_key(row["path"], old_leaf["box"])
                new_leaf["member_logits"] = [original_lookup[tables[0][key]].tolist()]
            before = replay(original, data["manifest"], row["quality_weight"])
            after = replay(leaves, data["manifest"], row["quality_weight"])
            if (before.get("kind"), before.get("label")) != (
                after.get("kind"),
                after.get("label"),
            ):
                baseline_replay_mismatches.append(
                    dict(cohort=name, index=index, before=before, after=after)
                )
    screen = OpenedGuardScreen(data).evaluate(revised)
    screen["checks"].pop("original_disease_and_healthy_exact")
    screen["checks"]["original_disease_exact"] = all(
        revised(row)["app_replay"] == row["app_replay"]
        for rows in list(data["base"].values()) + [data["internal"]]
        for row in rows
        if row["app_replay"]["kind"] == "DISEASE"
    )
    screen["passed"] = all(screen["checks"].values())
    result = dict(
        protocol_sha256=contract,
        passed=screen["passed"] and not mismatches and not baseline_replay_mismatches,
        checks=screen["checks"],
        candidate=screen,
        verdict_rows_checked=checked,
        verdict_mismatches=mismatches,
        baseline_replay_mismatches=baseline_replay_mismatches,
        primary_maximum_logit_error=primary_error,
        leaf_probability_errors=errors,
        whole_probability_errors=whole_errors,
        whole_dino_probability_error=dino_error,
        leaf_inputs=len(indices),
        whole_inputs=len(paths),
        whole_dino_inputs=len(dino_paths),
        arrays={
            name: digest(out / name)
            for name in (
                "leaf_outputs.npy",
                "whole_outputs.npy",
                "whole_dino_outputs.npy",
            )
        },
        elapsed_seconds=time.monotonic() - started,
        cold_olid_used=False,
        ad_independent=False,
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in ("candidate", "verdict_mismatches", "baseline_replay_mismatches")
            }
        ),
        flush=True,
    )
    assert result[
        "passed"
    ], "Restored candidate failed exact routing or unchanged recognition gates"


if __name__ == "__main__":
    main()
