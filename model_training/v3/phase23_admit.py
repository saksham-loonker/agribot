"""Admit the failed AD source as human-labelled grouped development only."""

from pathlib import Path
import argparse
import collections
import concurrent.futures
import json
import os
import shutil

import numpy as np
from PIL import Image

from app_eval import classifier_letterbox_input
from permissive_encoder import digest


def read(path):
    return json.loads(Path(path).read_text())


def split(group):
    import hashlib

    return (
        "val"
        if int(hashlib.sha256(("ad-admit677:" + group).encode()).hexdigest()[:8], 16)
        % 100
        < 20
        else "train"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    a = ap.parse_args()
    p = Path(a.campaign)
    result_path = p / "phase22_external_acceptance/result.json"
    result = read(result_path)
    assert result["source_independence_consumed"] and not result["passed"]
    assert (p / "phase9_ad_source/RETIRED_BENCHMARK.json").exists()
    root = p / "phase23_data"
    work = root / "cnn_inputs"
    work.mkdir(parents=True, exist_ok=True)
    base = p / "phase13_siglip"
    source = p / "phase9_ad_source/sealed_geometric_manifest.json"
    assert (
        digest(source)
        == "5cf37861dfe835c918643b2753a05c229e9327cd750524435006384bdcfae6b6"
    )
    admitted = [
        dict(r, split=split(r["group"]), retired_from_benchmark=True)
        for r in read(source)
    ]
    # No candidate predictions or confidence scores become training targets.
    added = [
        dict(
            r,
            id="phase23:" + r["id"],
            set="training",
            box=None,
            view="full_scene",
            input_variant="full_roi",
            health_y=float(r["label"] == "Healthy"),
            source_label=r["label"],
            label="Healthy" if r["label"] == "Healthy" else "Not_healthy_tomato",
        )
        for r in admitted
    ]
    rows = read(base / "cnn_inputs/rows.json") + added
    assert not (
        {r["group"] for r in rows if r["split"] == "train"}
        & {r["group"] for r in rows if r["split"] == "val"}
    )
    for name in ["train", "val"]:
        assert {r["health_y"] for r in added if r["split"] == name} == {0.0, 1.0}
    protocol = dict(
        base_rows_sha256=digest(base / "cnn_inputs/rows.json"),
        admission_source_sha256=digest(source),
        failed_external_result_sha256=digest(result_path),
        split_rule="SHA256 ad-admit677:group modulo100 <20 is internal val; no rerolled seed",
        source_photos=len(admitted),
        human_labels_only=True,
        original_classifier_frozen=True,
        restricted_teacher_used=False,
        consumed_ad_now_used_for_training=True,
        independent=False,
        counts=dict(
            collections.Counter(r["split"] + ":" + r["source_label"] for r in added)
        ),
    )
    complete = root / "complete.json"
    if complete.exists():
        assert read(complete)["protocol"] == protocol
        assert read(complete)["rows_sha256"] == digest(work / "rows.json")
        print("Consumed AD data already prepared")
        return
    pixels_path = root / "admitted_ad_full_scene.npy"
    pixels = np.lib.format.open_memmap(
        pixels_path, mode="w+", dtype=np.float32, shape=(len(added), 224, 224, 3)
    )

    def prepare(r):
        assert digest(r["path"]) == r["file_sha256"]
        with Image.open(r["path"]) as im:
            return classifier_letterbox_input(np.asarray(im.convert("RGB")), 224)[0]

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for i, x in enumerate(pool.map(prepare, added)):
            pixels[i] = x
    pixels.flush()
    (work / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    (work / "pixels.json").write_text(
        json.dumps(
            dict(
                center=read(base / "cnn_inputs/pixels.json")["center"]
                + [str(pixels_path)]
            ),
            indent=2,
        )
    )
    (root / "admission_manifest.json").write_text(
        json.dumps(read(base / "admission_manifest.json") + admitted, indent=2)
    )
    internal = [
        json.loads(line)
        for line in (
            p / "phase12_scene/union_app_validation/baseline_predictions.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    before = {
        r["id"]: r["baseline"]
        for r in map(
            json.loads,
            (p / "phase22_external_acceptance/predictions.jsonl")
            .read_text()
            .splitlines(),
        )
    }
    internal += [before[r["id"]] for r in admitted if r["split"] == "val"]
    assert {r["id"] for r in internal} == {
        r["id"] for r in read(root / "admission_manifest.json") if r["split"] == "val"
    }
    (root / "internal_baseline_predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in internal)
    )
    (root / "AD_ADMISSION.json").write_text(json.dumps(protocol, indent=2))
    (root / "RETIRED_BENCHMARK.json").write_text(
        json.dumps(
            dict(
                reason="All old internal families and AD are consumed development, never independent evaluation for these runs"
            ),
            indent=2,
        )
    )
    for name, family, old in [
        ("phase23_siglip", "siglip2", "phase13_siglip"),
        ("phase23_dino", "dinov2_large", "phase15_dino_large"),
        ("phase23_hybrid", "hybrid_field", "phase16_hybrid"),
    ]:
        target = p / name
        target.mkdir(exist_ok=True)
        if not (target / "cnn_inputs").exists():
            (target / "cnn_inputs").symlink_to(work, target_is_directory=True)
        for file in [
            "AD_ADMISSION.json",
            "RETIRED_BENCHMARK.json",
            "admission_manifest.json",
        ]:
            shutil.copy2(root / file, target / file)
        (target / "eval_rows.json").write_text("{}")
        if family == "hybrid_field":
            (target / "source_links.json").write_text(
                json.dumps(
                    dict(
                        siglip_features=str(p / "phase23_siglip/siglip2_feature_heads"),
                        dino_features=str(
                            p / "phase23_dino/dinov2_large_feature_heads"
                        ),
                    ),
                    indent=2,
                )
            )
        else:
            shutil.copy2(
                p / old / "encoder_source.json", target / "encoder_source.json"
            )
        features = target / (family + "_feature_heads")
        features.mkdir(exist_ok=True)
        if not (features / "encoder.pt").exists():
            os.link(
                p / old / (family + "_feature_heads/encoder.pt"),
                features / "encoder.pt",
            )
        shutil.copy2(
            p / old / (family + "_feature_heads/input_contract_parity.json"),
            features / "input_contract_parity.json",
        )
    report = dict(
        protocol=protocol,
        rows=len(rows),
        rows_sha256=digest(work / "rows.json"),
        internal_photos=len(internal),
        independent=False,
    )
    complete.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
