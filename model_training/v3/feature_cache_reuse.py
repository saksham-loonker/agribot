"""Reuse a verified frozen encoder's unchanged canonical input prefix.

An explicit plan pins encoder bytes, feature/row identities, preprocessing
source and immutable shared pixel shards. A changed encoder, input or contract
fails closed. Existing complete caches are never rebuilt by this helper.
"""

from pathlib import Path
import argparse
import ast
import hashlib
import json
from importlib import metadata

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def runtime_identity():
    result = {}
    for package in ["torch", "timm", "numpy", "Pillow", "opencv-python"]:
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result


def preprocessing_contract(repo, family):
    repo = Path(repo)
    sources = [
        ("model_training/v2/final/canon.py", None),
        ("tests/model_evaluation/canonical_runtime.py", None),
        ("model_training/v3/app_eval.py", "classifier_letterbox_input"),
    ]
    if family in ("siglip2", "hybrid_field"):
        sources.append(
            ("model_training/v3/permissive_encoder.py", "CanonicalPermissiveEncoder")
        )
    if family in ("dinov2_large", "hybrid_field"):
        sources.append(("model_training/v3/dino_large_encoder.py", "DinoLargeEncoder"))
    if family == "hybrid_field":
        sources.append(
            ("model_training/v3/hybrid_field_encoder.py", "HybridFieldEncoder")
        )
    result = {}
    for relative, name in sources:
        tree = ast.parse((repo / relative).read_text())
        node = (
            next(n for n in tree.body if getattr(n, "name", None) == name)
            if name
            else tree
        )
        result[relative + (":" + name if name else "")] = hashlib.sha256(
            ast.dump(node, include_attributes=False).encode()
        ).hexdigest()
    return result


def make_plan(source_root, target_root, family, source_snapshot, current_repo):
    source_root, target_root = Path(source_root), Path(target_root)
    directory = source_root / (family + "_feature_heads")
    identity = read(directory / "feature_identity.json")
    assert identity["encoder_sha256"] == digest(directory / "encoder.pt")
    assert identity["rows_sha256"] == digest(source_root / "cnn_inputs/rows.json")
    assert read(directory / "restored_encoder_parity.json")["passed"]
    assert read(directory / "input_contract_parity.json")["passed"]
    expected = preprocessing_contract(source_snapshot, family)
    assert expected == preprocessing_contract(
        current_repo, family
    ), "Preprocessing or encoder implementation changed"
    shards = read(source_root / "cnn_inputs/pixels.json")["center"]
    plan = dict(
        source_root=str(source_root),
        family=family,
        encoder_sha256=identity["encoder_sha256"],
        feature_identity_sha256=digest(directory / "feature_identity.json"),
        feature_array_sha256=digest(directory / "features.npy"),
        preprocessing_contract=expected,
        source_snapshot=str(source_snapshot),
        runtime_identity=runtime_identity(),
        pixel_shards=[
            dict(
                path=str(Path(f).resolve()),
                bytes=Path(f).stat().st_size,
                mtime_ns=Path(f).stat().st_mtime_ns,
            )
            for f in shards
        ],
        protection="Same shared canonical pixel files and byte/mtime identities, exact row prefix, encoder SHA and normalized source AST; final candidate still needs full fresh verification",
    )
    path = target_root / "feature_reuse.json"
    if path.exists():
        assert read(path) == plan
    else:
        path.write_text(json.dumps(plan, indent=2))
    return plan


def seed_features(root, out, ids, dimension, family, current_repo=None):
    root, out = Path(root), Path(out)
    rows = read(root / "cnn_inputs/rows.json")
    features = np.zeros((len(rows), dimension), np.float32)
    plan_file = root / "feature_reuse.json"
    if not plan_file.exists():
        return features, ids
    plan = read(plan_file)
    assert plan["family"] == family, "Feature family changed"
    assert (
        plan["runtime_identity"] == runtime_identity()
    ), "Encoder or preprocessing runtime changed"
    current_repo = (
        Path(current_repo) if current_repo else Path(__file__).resolve().parents[2]
    )
    assert plan["preprocessing_contract"] == preprocessing_contract(
        current_repo, family
    ), "Preprocessing or encoder implementation changed"
    source = Path(plan["source_root"])
    old = source / (family + "_feature_heads")
    assert (
        digest(out / "encoder.pt") == plan["encoder_sha256"]
    ), "Encoder weights changed"
    assert digest(old / "feature_identity.json") == plan["feature_identity_sha256"]
    assert digest(old / "features.npy") == plan["feature_array_sha256"]
    old_identity = read(old / "feature_identity.json")
    old_rows = read(source / "cnn_inputs/rows.json")
    assert digest(source / "cnn_inputs/rows.json") == old_identity["rows_sha256"]
    assert rows[: len(old_rows)] == old_rows, "Input row prefix changed"
    target_shards = read(root / "cnn_inputs/pixels.json")["center"]
    assert [
        str(Path(f).resolve()) for f in target_shards[: len(plan["pixel_shards"])]
    ] == [f["path"] for f in plan["pixel_shards"]], "Canonical input shards changed"
    for shard in plan["pixel_shards"]:
        stat = Path(shard["path"]).stat()
        assert (
            stat.st_size == shard["bytes"] and stat.st_mtime_ns == shard["mtime_ns"]
        ), "Canonical pixel shard modified"
    reused_ids = ids[ids < len(old_rows)]
    assert (
        hashlib.sha256(reused_ids.tobytes()).hexdigest() == old_identity["ids_sha256"]
    ), "Cached extraction membership changed"
    old_values = np.load(old / "features.npy", mmap_mode="r")
    assert (
        old_values.shape == (len(old_rows), dimension)
        and old_values.dtype == np.float32
    )
    features[reused_ids] = old_values[reused_ids]
    remaining = ids[ids >= len(old_rows)]
    assert np.isfinite(features[reused_ids]).all()
    (out / "feature_reuse_audit.json").write_text(
        json.dumps(
            dict(
                plan_sha256=digest(plan_file),
                reused_rows=len(reused_ids),
                new_rows=len(remaining),
                encoder_sha256=plan["encoder_sha256"],
                existing_complete_caches_restarted=False,
            ),
            indent=2,
        )
    )
    return features, remaining


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--target-root", required=True)
    ap.add_argument(
        "--family", choices=["siglip2", "dinov2_large", "hybrid_field"], required=True
    )
    ap.add_argument("--source-snapshot", required=True)
    ap.add_argument("--repo", required=True)
    a = ap.parse_args()
    print(
        json.dumps(
            make_plan(
                a.source_root, a.target_root, a.family, a.source_snapshot, a.repo
            ),
            indent=2,
        )
    )
