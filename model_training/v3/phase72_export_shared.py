"""Export only the restored-verified candidate and test mobile decision boundaries."""

from pathlib import Path
import argparse
import gc
import hashlib
from importlib import metadata
import json
import time

import numpy as np
from PIL import Image
import torch
import litert_torch
from ai_edge_litert.interpreter import Interpreter

from app_eval import classifier_letterbox_input
from opened_context_data import load, read
from partial_dino_cache import CanonicalInputs, input_rows
from permissive_encoder import digest
from phase29_compact_prepare import region_key
from qualified_recognition_models import restore


def freeze(path, value):
    if path.exists():
        assert read(path) == value, ("Frozen export identity changed", str(path))
    else:
        path.write_text(json.dumps(value, indent=2))


def nearest_both(values, target, count=2):
    return sorted(
        set(
            np.flatnonzero(values < target)[
                np.argsort(target - values[values < target])[:count]
            ].tolist()
            + np.flatnonzero(values >= target)[
                np.argsort(values[values >= target] - target)[:count]
            ].tolist()
        )
    )


def make_cases(p, component, model):
    verified = p / "phase71_restored"
    proof = read(verified / "result.json")
    assert proof["passed"] and all(proof["checks"].values())
    data = load(p)
    cases = []
    if component == "leaf":
        tables = [
            read(p / f"phase29_compact/{name}_indices.json")
            for name in ("center", "full")
        ]
        indices = sorted(
            {
                table[region_key(path, leaf["box"])]
                for path, row in data["photos"].items()
                for leaf in row["leaves"]
                for table in tables
            }
        )
        assert (
            hashlib.sha256(json.dumps(indices).encode()).hexdigest()
            == read(verified / "protocol.json")["leaf_indices_sha256"]
        )
        values = np.load(verified / "leaf_outputs.npy")
        chosen = set()
        for column, targets in [(9, [0.1]), (10, [0.001]), (11, [0.2, 0.3, 0.5, 0.95])]:
            for target in targets:
                chosen.update(nearest_both(values[:, column], target))
        logits = values[:, :9].astype(np.float64)
        logits[:, 8] += 0.75
        logits /= 0.8814700841903687
        probabilities = np.exp(logits - logits.max(1, keepdims=True))
        probabilities /= probabilities.sum(1, keepdims=True)
        top = np.sort(probabilities, axis=1)[:, -2:]
        for target in (0.5, 0.6, 0.7):
            chosen.update(nearest_both(top[:, 1], target))
        for target in (0.15, 0.2):
            chosen.update(nearest_both(top[:, 1] - top[:, 0], target))
        descriptors = input_rows(p)
        pixels = CanonicalInputs(p)
        try:
            for index in sorted(chosen):
                original_index = indices[index]
                cases.append(
                    (
                        f"leaf:{original_index}",
                        pixels(descriptors[original_index]),
                        values[index : index + 1],
                    )
                )
        finally:
            pixels.close()
        whole_paths = sorted(read(p / "phase69_whole_local/cache/probabilities.json"))
        whole_values = np.load(verified / "whole_dino_outputs.npy")
        for index, path in enumerate(whole_paths):
            with Image.open(path) as image:
                tensor = classifier_letterbox_input(
                    np.asarray(image.convert("RGB")), 224
                )
            cases.append(
                (f"whole_dino:{path}", tensor, whole_values[index : index + 1])
            )
    else:
        paths = [
            row["path"] for row in read(p / "opened_siglip_feature_cache/inputs.json")
        ]
        values = np.load(verified / "whole_outputs.npy")
        chosen = set()
        for column, target in [(0, 0.85), (1, 0.001), (2, 0.5), (3, 0.5)]:
            chosen.update(nearest_both(values[:, column], target))
        for index in sorted(chosen):
            path = paths[index]
            with Image.open(path) as image:
                tensor = classifier_letterbox_input(
                    np.asarray(image.convert("RGB")), 224
                )
            cases.append((f"whole:{path}", tensor, values[index : index + 1]))
    torch.manual_seed(1729)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    with torch.inference_mode():
        for index, rgb in enumerate(
            (
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.5, 0.5, 0.5],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.1, 0.4, 0.8],
            )
        ):
            tensor = (
                ((torch.tensor(rgb) - mean) / std)
                .view(1, 1, 1, 3)
                .expand(1, 224, 224, 3)
                .contiguous()
            )
            cases.append((f"color:{index}", tensor.numpy(), model(tensor).numpy()))
        for index in range(3):
            tensor = (torch.rand(1, 224, 224, 3) - mean) / std
            cases.append((f"random:{index}", tensor.numpy(), model(tensor).numpy()))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--component", choices=("leaf", "whole"), required=True)
    args = ap.parse_args()
    p = Path(args.campaign)
    out = p / "phase72_mobile_export"
    out.mkdir(exist_ok=True)
    proof = read(p / "phase71_restored/result.json")
    assert proof["passed"] and all(proof["checks"].values())
    candidate = read(p / "phase70_qualified/candidate.json")
    component = args.component
    bundle = p / "phase70_qualified" / candidate["bundles"][component]["file"]
    assert digest(bundle) == candidate["bundles"][component]["sha256"]
    torch.set_num_threads(4)
    model = restore(bundle)
    cases = make_cases(p, component, model)
    protocol = dict(
        candidate_sha256=digest(p / "phase70_qualified/candidate.json"),
        verified_result_sha256=digest(p / "phase71_restored/result.json"),
        bundle_sha256=digest(bundle),
        component=component,
        precision="FP32 weights, inputs and outputs; no quantization",
        cases=[
            dict(
                id=name,
                input_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                expected_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
            )
            for name, value, expected in cases
        ],
        script_sha256=digest(__file__),
        model_source_sha256=digest(
            Path(__file__).with_name("qualified_recognition_models.py")
        ),
        converter=metadata.version("litert-torch"),
        runtime=metadata.version("ai-edge-litert"),
        torch=metadata.version("torch"),
        maximum_primary_logit_error=0.002,
        maximum_probability_error=0.0002,
        builtin_only=True,
        cold_olid_used=False,
        ad_independent=False,
    )
    freeze(out / (component + "_protocol.json"), protocol)
    contract = digest(out / (component + "_protocol.json"))
    result_path = out / (component + "_result.json")
    target = out / (component + ".tflite")
    if result_path.exists():
        result = read(result_path)
        assert result["protocol_sha256"] == contract and result[
            "tflite_sha256"
        ] == digest(target)
        assert result["passed"]
        print(json.dumps(result), flush=True)
        return
    started = time.monotonic()
    converted = out / (component + "_converted.json")
    if converted.exists():
        record = read(converted)
        assert record["protocol_sha256"] == contract and record[
            "tflite_sha256"
        ] == digest(target)
    else:
        temporary = target.with_suffix(".partial.tflite")
        edge = litert_torch.convert(model, (torch.from_numpy(cases[0][1]),))
        edge.export(str(temporary))
        temporary.replace(target)
        converted.write_text(
            json.dumps(dict(protocol_sha256=contract, tflite_sha256=digest(target)))
        )
        del edge
        gc.collect()
    runtime = Interpreter(model_path=str(target), num_threads=2)
    runtime.allocate_tensors()
    inputs = runtime.get_input_details()[0]
    outputs = runtime.get_output_details()[0]
    width = 12 if component == "leaf" else 4
    assert (
        inputs["shape"].tolist() == [1, 224, 224, 3] and inputs["dtype"] == np.float32
    )
    assert outputs["shape"].tolist() == [1, width] and outputs["dtype"] == np.float32
    ops = sorted({item["op_name"] for item in runtime._get_ops_details()})
    assert not any(
        op == "CUSTOM" or op.startswith("Flex") or "STABLEHLO" in op for op in ops
    ), ops
    results = []
    for name, tensor, expected in cases:
        runtime.set_tensor(inputs["index"], np.asarray(tensor, np.float32))
        runtime.invoke()
        actual = runtime.get_tensor(outputs["index"])
        assert np.isfinite(actual).all()
        raw_error = (
            float(np.max(np.abs(actual[:, :9] - expected[:, :9])))
            if component == "leaf"
            else 0.0
        )
        columns = slice(9, None) if component == "leaf" else slice(None)
        probability_error = float(
            np.max(np.abs(actual[:, columns] - expected[:, columns]))
        )
        assert raw_error <= 0.002 and probability_error <= 0.0002, (
            name,
            raw_error,
            probability_error,
        )
        if component == "leaf":
            assert int(actual[:, :9].argmax()) == int(expected[:, :9].argmax()), name
            for column, thresholds in [
                (9, [0.1]),
                (10, [0.001]),
                (11, [0.2, 0.3, 0.5, 0.95]),
            ]:
                for threshold in thresholds:
                    assert bool(float(actual[0, column]) >= threshold) == bool(
                        float(expected[0, column]) >= threshold
                    ), (name, column, threshold)
        else:
            for column, threshold in [(0, 0.85), (1, 0.001), (2, 0.5), (3, 0.5)]:
                assert bool(float(actual[0, column]) >= threshold) == bool(
                    float(expected[0, column]) >= threshold
                ), (name, column, threshold)
        results.append(
            dict(
                id=name,
                primary_logit_error=raw_error,
                probability_error=probability_error,
            )
        )
    timings = []
    for index in range(7):
        runtime.set_tensor(inputs["index"], np.asarray(cases[0][1], np.float32))
        before = time.perf_counter()
        runtime.invoke()
        if index >= 2:
            timings.append((time.perf_counter() - before) * 1000)
    np.savez(
        out / (component + "_parity_cases.npz"),
        inputs=np.concatenate([value for _, value, _ in cases], 0),
        expected=np.concatenate([expected for _, _, expected in cases], 0),
        ids=np.array([name for name, _, _ in cases]),
    )
    result = dict(
        protocol_sha256=contract,
        tflite_sha256=digest(target),
        bytes=target.stat().st_size,
        component=component,
        passed=True,
        cases=len(cases),
        rows=results,
        ops=ops,
        input_shape=inputs["shape"].tolist(),
        output_shape=outputs["shape"].tolist(),
        cpu_two_thread_median_ms=float(np.median(timings)),
        cpu_two_thread_samples_ms=timings,
        elapsed_seconds=time.monotonic() - started,
        cold_olid_used=False,
        ad_independent=False,
    )
    result_path.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in ("rows", "ops")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
