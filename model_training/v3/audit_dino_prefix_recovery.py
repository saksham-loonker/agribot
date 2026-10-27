"""Explicitly audit an unregistered completed prefix cache after interrupted hashing."""

from pathlib import Path
import argparse
import ast
import concurrent.futures
import hashlib
import json
import math
import time
import numpy as np
import torch
from torch.nn import functional as F

from feature_cache_reuse import digest
from opened_context_data import read
from partial_cnn_cache import prepare as validate_inputs
from partial_dino_cache import CanonicalInputs, input_rows, original_dino
from partial_dino_healthy import prefix_cut, prefix_features, prefix_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    args = parser.parse_args()
    p = Path(args.campaign)
    root = p / "phase51_dino_adaptation/prefix"
    assert not (
        root / "complete.json"
    ).exists(), "Already registered; use the normal validator"
    protocol = read(root / "protocol.json")
    parent = validate_inputs(p)
    assert protocol["parent_input_contract_sha256"] == digest(parent / "protocol.json")
    assert protocol["bundle_sha256"] == digest(p / "export_cnn_safe/pytorch_bundle.pt")
    assert protocol["fitting_rows_sha256"] == digest(
        p / "phase46_paper_primary/fitting_rows.json"
    )
    assert protocol["cache_source_sha256"] == digest(
        Path(__file__).with_name("partial_dino_cache.py")
    )
    source = ast.parse(Path(__file__).with_name("partial_dino_healthy.py").read_text())
    nodes = [
        ast.dump(node, include_attributes=False)
        for node in source.body
        if getattr(node, "name", None) in ("prefix_cut", "prefix_features")
    ]
    assert (
        protocol["prefix_source_sha256"]
        == hashlib.sha256(json.dumps(nodes).encode()).hexdigest()
    )
    tables = np.load(root / "tables.npy")
    expected = np.unique(
        np.array(
            [r["table"] for r in read(p / "phase46_paper_primary/fitting_rows.json")],
            np.int64,
        )
    )
    assert np.array_equal(tables, expected)
    assert (
        protocol["fitting_tables_sha256"]
        == hashlib.sha256(tables.tobytes()).hexdigest()
    )
    values = np.load(root / "features.npy", mmap_mode="r")
    assert list(values.shape) == protocol["shape"] and values.dtype == np.float32
    rows = input_rows(p)
    old = np.load(p / "phase29_compact/features.npy", mmap_mode="r")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = original_dino(p).cuda()
    pixels = CanonicalInputs(p)
    started = time.monotonic()
    maximum, extra_error, old_count, extra_count = 0.0, 0.0, 0, 0
    try:
        with concurrent.futures.ThreadPoolExecutor(8) as pool, torch.inference_mode():
            for start in range(0, len(tables), 128):
                ids = tables[start : start + 128]
                tokens = torch.from_numpy(
                    np.array(values[start : start + len(ids)], copy=True)
                ).cuda()
                assert torch.isfinite(tokens).all()
                mask = ids < len(old)
                if mask.any():
                    reconstructed = model.forward_head(
                        model.norm(model.blocks[prefix_cut(model) :](tokens)),
                        pre_logits=True,
                    )
                    reconstructed = (
                        (F.normalize(reconstructed, dim=1) * math.sqrt(384))
                        .cpu()
                        .numpy()
                    )
                    maximum = max(
                        maximum,
                        float(
                            np.max(np.abs(reconstructed[mask] - old[ids[mask], :384]))
                        ),
                    )
                    assert maximum < 0.002
                    old_count += int(mask.sum())
                if (~mask).any():
                    image = (
                        torch.from_numpy(
                            np.concatenate(
                                list(pool.map(pixels, [rows[i] for i in ids[~mask]])), 0
                            )
                        )
                        .cuda()
                        .permute(0, 3, 1, 2)
                    )
                    fresh = prefix_features(model, image)
                    extra_error = max(
                        extra_error,
                        float(
                            (fresh - tokens[torch.from_numpy(~mask).cuda()]).abs().max()
                        ),
                    )
                    assert extra_error < 0.002
                    extra_count += int((~mask).sum())
    finally:
        pixels.close()
    audit = dict(
        passed=True,
        purpose="Recover useful complete tokens; no retraining or acceptance claim",
        protocol_sha256=digest(root / "protocol.json"),
        auditor_sha256=digest(__file__),
        old_rows_verified_by_original_features=old_count,
        new_rows_verified_by_fresh_prefix=extra_count,
        maximum_original_feature_error=maximum,
        maximum_fresh_prefix_error=extra_error,
        features_sha256=digest(root / "features.npy"),
        elapsed_seconds=time.monotonic() - started,
        cold_olid_used=False,
    )
    (root / "recovery_audit.json").write_text(json.dumps(audit, indent=2))
    complete = dict(
        protocol_sha256=audit["protocol_sha256"],
        features_sha256=audit["features_sha256"],
        tables_sha256=digest(root / "tables.npy"),
        prefix_state_sha256=prefix_identity(model),
        inputs=len(tables),
        bytes=int(values.size) * 4,
        original_feature_rows_audited=old_count,
        maximum_original_feature_error=maximum,
        elapsed_seconds=audit["elapsed_seconds"],
        timing_scope="Recovery audit only; initial preparation timing unavailable",
        recovery_audit_sha256=digest(root / "recovery_audit.json"),
        cold_olid_used=False,
    )
    (root / "complete.json").write_text(json.dumps(complete, indent=2))
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
