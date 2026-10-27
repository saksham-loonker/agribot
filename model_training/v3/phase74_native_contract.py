"""Freeze Android schema, scalar-policy fixtures and no-leaf baseline guard parity."""

from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

from app_eval import canon, posterior
from evaluation_inputs import input_identity
from opened_context_data import load, read
from permissive_encoder import digest
from phase29_compact_prepare import region_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    args = ap.parse_args()
    p = Path(args.campaign)
    out = p / "phase74_native_contract"
    out.mkdir(exist_ok=True)
    checked = read(p / "phase73_mobile_verified/result.json")
    assert checked["passed"] and all(checked["checks"].values())
    data = load(p)
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
    assert (
        hashlib.sha256(json.dumps(indices).encode()).hexdigest()
        == read(p / "phase73_mobile_verified/protocol.json")["leaf_indices_sha256"]
    )
    values = np.load(p / "phase73_mobile_verified/leaf_outputs.npy")
    lookup = {index: values[i, 9:12].tolist() for i, index in enumerate(indices)}
    paths = [row["path"] for row in read(p / "opened_siglip_feature_cache/inputs.json")]
    scores = np.load(p / "phase73_mobile_verified/whole_outputs.npy")
    whole = {path: scores[i].tolist() for i, path in enumerate(paths)}
    dino_paths = sorted(read(p / "phase69_whole_local/cache/probabilities.json"))
    dino_values = np.load(p / "phase73_mobile_verified/whole_dino_outputs.npy")
    dino = {path: float(dino_values[i, 11]) for i, path in enumerate(dino_paths)}
    source_refs = read(p / "phase73_mobile_verified/reference.json")
    cases = []
    aliases = 0
    for name, rows in list(data["base"].items()) + [("internal", data["internal"])]:
        for index, row in enumerate(rows):
            path = row["path"]
            before = row["app_replay"]
            after = source_refs[name][index]
            if before["kind"] != data["photos"][path]["app_replay"]["kind"]:
                aliases += 1
            action = (
                "RECOVER_HEALTHY"
                if before["kind"] != "HEALTHY" and after["kind"] == "HEALTHY"
                else (
                    "ABSTAIN_HEALTHY"
                    if before["kind"] == "HEALTHY" and after["kind"] == "UNSURE"
                    else "NONE"
                )
            )
            label = before.get("label")
            cases.append(
                dict(
                    id=f"{name}:{index}",
                    baseline_kind=before["kind"],
                    baseline_label=(
                        data["manifest"]["classifier"]["labels"].index(label)
                        if label in data["manifest"]["classifier"]["labels"]
                        else None
                    ),
                    posterior=before.get("probabilities"),
                    fallback=row.get("context_whole_reference_probabilities"),
                    whole=whole[path],
                    leaves=[
                        [lookup[table[key]] for table in tables] for key in keys[path]
                    ],
                    whole_adapted=dino.get(path),
                    usable=not (row["contrast"] < 0.015 and row["sharpness"] < 0.012),
                    expected_action=action,
                )
            )
    fixture = dict(
        schema=1,
        source_mobile_verification_sha256=digest(
            p / "phase73_mobile_verified/result.json"
        ),
        labels=data["manifest"]["classifier"]["labels"],
        cases=cases,
        observation_alias_kind_differences=aliases,
    )
    (out / "policy_cases.json").write_text(json.dumps(fixture, separators=(",", ":")))
    leaf = read(p / "phase72_mobile_export/leaf_result.json")
    context = read(p / "phase72_mobile_export/whole_result.json")
    manifest = dict(data["manifest"])
    manifest["schema"] = 5
    manifest["bundle_id"] = (
        "agribot-v3-field-"
        + leaf["tflite_sha256"][:8]
        + "-"
        + context["tflite_sha256"][:8]
    )
    manifest["status"] = (
        "FIELD RECOVERY CANDIDATE: qualification and release status tracked in release evidence"
    )
    manifest["classifier"] = dict(manifest["classifier"])
    manifest["classifier"]["members"] = [
        dict(
            file="disease_classifier_0.tflite",
            sha256=leaf["tflite_sha256"],
            arch="Original DINOv2/CNN classifier with shared frozen-prefix Healthy auxiliaries",
            licence="Apache-2.0",
            input=[1, 224, 224, 3],
            output=[1, 12],
        )
    ]
    manifest["classifier"]["auxiliary_outputs"] = [
        "compact_positive_H",
        "compact_negative_H",
        "adapted_dino_H",
    ]
    manifest["primary_recovery"] = dict(
        policy="phase69_whole_local",
        scope="primary_visible_view",
        healthy_min=0.85,
        negative_max=0.001,
        genus_min=0.5,
        natural_min=0.5,
        compact_min=0.1,
        closed_other_max=0.1,
        closed_disease_mass_min=0.6,
        strong_local_max=0.2,
        partial_local_max=0.3,
        partial_local_leaves=2,
        isolated_min=0.95,
        local_center_min=0.5,
        whole_healthy_min=0.5,
        whole_local_leaves=2,
        minimum_frames=7,
        maximum_leaves=6,
        model=dict(
            file="plant_context.tflite",
            sha256=context["tflite_sha256"],
            input=[1, 224, 224, 3],
            output=[1, 4],
            outputs=["positive_H", "negative_H", "tomato_genus", "natural_context"],
            preprocess="canonical_whole_letterbox_gray114",
            layout="NHWC",
            licence="Apache-2.0",
        ),
    )
    manifest["training"] = dict(
        architecture="Shared-prefix phase69 recognition",
        restricted_teacher_used=False,
        original_classifier_bundle_sha256="43e4bb99d6cb1164f2d63ca370316f1cb877271982ffb53fa1690a95d2fa0a47",
        original_classifier_tflite_sha256="d70135804cf750341218e89a055370bef46a9b337951897b9f55f2570844ed39",
        candidate_sha256=digest(p / "phase70_qualified/candidate.json"),
        mobile_development_verification_sha256=digest(
            p / "phase73_mobile_verified/result.json"
        ),
        ad_role="Consumed grouped training/development; permanently not independent",
        olid_used_for_training=False,
        primary_label_scope="Publisher primary visible view; not every-leaf clinical health",
    )
    target = out / "model_manifest.json"
    if target.exists():
        assert read(target) == manifest
    else:
        target.write_text(json.dumps(manifest, indent=2) + "\n")
    missing = sorted(
        path
        for path, row in data["photos"].items()
        if not row["leaves"]
        and whole[path][0] >= 0.85
        and whole[path][2] >= 0.5
        and whole[path][3] >= 0.5
    )
    protocol = dict(
        mobile_verification_sha256=digest(p / "phase73_mobile_verified/result.json"),
        model_sha256=leaf["tflite_sha256"],
        input_identity=input_identity([dict(id=path, path=path) for path in missing]),
        preprocessing="Original classifier whole-frame canonical center crop, not context letterbox",
        script_sha256=digest(__file__),
        cold_olid_used=False,
        ad_independent=False,
    )
    frozen = out / "no_leaf_protocol.json"
    if frozen.exists():
        assert read(frozen) == protocol
    else:
        frozen.write_text(json.dumps(protocol, indent=2))
    refs = {
        row["path"]: row
        for row in map(
            json.loads,
            (p / "phase23_data/whole_reference.jsonl").read_text().splitlines(),
        )
    }
    runtime = Interpreter(
        model_path=str(p / "phase72_mobile_export/leaf.tflite"), num_threads=8
    )
    runtime.allocate_tensors()
    i = runtime.get_input_details()[0]["index"]
    o = runtime.get_output_details()[0]["index"]
    parity = []
    for path in missing:
        with Image.open(path) as image:
            value = canon.classifier_input(np.asarray(image.convert("RGB")), 224)
        runtime.set_tensor(i, value)
        runtime.invoke()
        result = runtime.get_tensor(o)[0, :9]
        actual = posterior([result], data["manifest"])
        expected = np.array(refs[path]["whole_probabilities"])
        error = float(np.max(np.abs(actual - expected)))
        assert error <= 0.0002, (path, error)
        parity.append(dict(path=path, maximum_probability_error=error))
    result = dict(
        passed=True,
        policy_cases=len(cases),
        observation_alias_kind_differences=aliases,
        policy_fixture_sha256=digest(out / "policy_cases.json"),
        manifest_sha256=digest(target),
        no_leaf_inputs=len(missing),
        maximum_no_leaf_probability_error=max(
            [r["maximum_probability_error"] for r in parity], default=0
        ),
        no_leaf_parity=parity,
        cold_olid_used=False,
        ad_independent=False,
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps({k: v for k, v in result.items() if k != "no_leaf_parity"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
