"""Select a leaf-only Healthy repair on opened development/regression, no AD.

The reference aggregate disease verdict takes priority. This is an explicit
future runtime requirement, not yet a property of the installed Android app.
Human-annotated field validation counts distinct leaf instances, not views.
"""

from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import itertools
import json

import numpy as np
from PIL import Image
import timm
import torch

from app_eval import classifier_letterbox_input, posterior
from campaign_heads import app_evaluate
from campaign_pixels import open_pixels
from campaign_reference import safe_logits
from leaf_healthy_repair import spec_and_weights
from phase2_select import retain
from phase5_scene_ablation import grouped
from view_guard_policy import decision_key


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--species-directory")
    ap.add_argument("--head-directory", help="Use a shared original-DINO binary head")
    ap.add_argument(
        "--tail-directory", help="Use a copied DINO tail fitted on annotated leaves"
    )
    ap.add_argument("--out-name", default="leaf_selection")
    ap.add_argument(
        "--allow-leaf-disease-override",
        action="store_true",
        help="Research override requiring aggregate reference disease priority",
    )
    ap.add_argument(
        "--components-from", help="Reuse identity-checked center components"
    )
    ap.add_argument(
        "--full-roi-recovery",
        action="store_true",
        help="Research full-ROI Healthy recovery; requires a separately implemented runtime policy",
    )
    a = ap.parse_args()
    assert not (a.head_directory and a.tail_directory)
    shared_verifier = bool(a.head_directory or a.tail_directory)
    root = Path(a.root)
    out = root / a.out_name
    out.mkdir(exist_ok=True)
    species = (
        Path(a.species_directory)
        if a.species_directory
        else root.parent / "phase6_roi/species"
    )
    rows = json.loads((root / "cnn_inputs/rows.json").read_text())
    ev = json.loads((root / "eval_rows.json").read_text())
    manifest = json.loads(
        (root.parent / "original_reference/model_manifest.json").read_text()
    )
    reference = safe_logits(root)
    ids = np.array(
        [
            i
            for i, r in enumerate(rows)
            if r["set"] in ev
            or (
                r["source"] == "fieldplant"
                and r["set"] == "training"
                and r["split"] == "val"
            )
        ]
    )
    field = np.array(
        [
            i
            for i in ids
            if rows[i]["source"] == "fieldplant"
            and rows[i]["view"] == "fieldplant_center"
        ]
    )
    identity = dict(
        rows_sha256=digest(root / "cnn_inputs/rows.json"),
        reference_logits_sha256=digest(root / "safe_reference_logits.npy"),
        reference_bundle_sha256=digest(
            root.parent / "export_cnn_safe/pytorch_bundle.pt"
        ),
        healthy_sha256=digest(root / "leaf_healthy/best.pt"),
        healthy_calibration_sha256=digest(root / "leaf_healthy/result.json"),
        species_sha256=digest(species / "best.pt"),
        species_calibration_sha256=digest(species / "result.json"),
        ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
    )
    if shared_verifier:
        head_directory = Path(a.head_directory or a.tail_directory)
        for key in list(identity):
            if key.startswith(("healthy_", "species_")):
                del identity[key]
        identity.update(
            head_sha256=digest(head_directory / "best.pt"),
            head_calibration_sha256=digest(head_directory / "result.json"),
        )
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timm.layers.set_fused_attn(False)
    if shared_verifier:
        if a.tail_directory:
            from leaf_tail_repair import spec_and_weights as feature_spec
        else:
            family = json.loads((head_directory / "result.json").read_text())[
                "protocol"
            ]["feature_identity"].get("family", "dino")
            if family.startswith("mobileclip"):
                from mobileclip_repair import spec_and_weights as feature_spec
            else:
                from leaf_feature_repair import spec_and_weights as feature_spec

        spec, model = feature_spec(
            root,
            head_directory,
            dict(healthy_min=0.8, disease_max=0.4, species_min=0.0),
        )
    else:
        spec, model = spec_and_weights(
            root, species, dict(healthy_min=0.8, disease_max=0.4, species_min=0.5)
        )
    model = model.cuda().eval()
    pixels = open_pixels(root / "cnn_inputs")
    center_cache = Path(a.components_from) if a.components_from else out
    cache = center_cache / "components.npz"
    meta = center_cache / "input_identity.json"
    if cache.exists():
        assert json.loads(meta.read_text()) == identity
        saved = np.load(cache)
        healthy = saved["healthy"]
        genus = saved["species"]
    else:
        healthy = np.full(len(rows), np.nan, np.float32)
        genus = healthy.copy()
        trained_features = (
            np.load(head_directory.parent / "features.npy", mmap_mode="r")
            if a.head_directory
            else None
        )
        binary_error = 0.0
        reference_error = 0.0
        with torch.inference_mode():
            for part in np.array_split(ids, max(1, int(np.ceil(len(ids) / 256)))):
                x = torch.from_numpy(np.array(pixels[part], copy=True)).cuda()
                if shared_verifier:
                    ref, h, g = model.components(x)
                    error = float(np.max(np.abs(ref.cpu().numpy() - reference[part])))
                    assert error < 0.002, ("Shared reference parity failed", error)
                    reference_error = max(reference_error, error)
                    if trained_features is not None:
                        expected = (
                            model.head(
                                torch.from_numpy(
                                    np.array(trained_features[part], copy=True)
                                ).cuda()
                            )
                            * model.calibration[0]
                            + model.calibration[1]
                        )
                        error = float((expected.ravel() - h.ravel()).abs().max())
                        assert error < 0.002, (
                            "Head cache/runtime alignment failed",
                            error,
                        )
                        binary_error = max(binary_error, error)
                    healthy[part] = h.ravel().cpu().numpy()
                    genus[part] = g.ravel().cpu().numpy()
                else:
                    healthy[part] = model.healthy.logits(x).ravel().cpu().numpy()
                    genus[part] = model.species(x).ravel().cpu().numpy()
        np.savez(cache, healthy=healthy, species=genus)
        meta.write_text(json.dumps(identity, indent=2))
        if shared_verifier:
            (out / "component_parity.json").write_text(
                json.dumps(
                    dict(
                        passed=True,
                        cases=len(ids),
                        max_reference_error=reference_error,
                        max_binary_head_error=(
                            binary_error if a.head_directory else None
                        ),
                    ),
                    indent=2,
                )
            )
    baseline_photos = {
        k: v["rows"]
        for k, v in app_evaluate(reference, rows, ev, manifest, True).items()
    }
    baseline = grouped(baseline_photos)
    # Full-ROI components are needed only for potentially new Healthy verdicts.
    rp = np.array([posterior([reference[i]], manifest) for i in ids])
    potential = (1 / (1 + np.exp(-np.clip(healthy[ids], -80, 80))) >= 0.5) & (
        np.max(rp[:, [0, 2, 3, 4, 5, 6, 7]], axis=1) <= 0.4
    )
    if a.allow_leaf_disease_override:
        potential = 1 / (1 + np.exp(-np.clip(healthy[ids], -80, 80))) >= 0.5
    full_ids = ids if a.full_roi_recovery else ids[potential]
    full_path = out / "full_components.npz"
    full_meta = out / "full_identity.json"
    full_identity = dict(
        identity, ids_sha256=hashlib.sha256(full_ids.tobytes()).hexdigest()
    )
    if full_path.exists():
        assert json.loads(full_meta.read_text()) == full_identity
        saved = np.load(full_path)
        full_reference = saved["reference"]
        full_healthy = saved["healthy"]
        full_species = saved["species"]
    else:
        full_reference = np.zeros_like(reference)
        full_healthy = np.zeros(len(rows), np.float32)
        full_species = full_healthy.copy()

        def prepare(i):
            with Image.open(rows[i]["path"]) as im:
                image = np.asarray(im.convert("RGB"))
            return classifier_letterbox_input(image, 224, rows[i]["box"])

        with torch.inference_mode(), concurrent.futures.ThreadPoolExecutor(8) as pool:
            for start in range(0, len(full_ids), 128):
                part = full_ids[start : start + 128]
                x = torch.from_numpy(
                    np.concatenate(list(pool.map(prepare, part)), 0)
                ).cuda()
                r, h, g = model.components(x)
                full_reference[part] = r.cpu().numpy()
                full_healthy[part] = h.cpu().numpy()
                full_species[part] = g.cpu().numpy()
        np.savez(
            full_path,
            reference=full_reference,
            healthy=full_healthy,
            species=full_species,
        )
        full_meta.write_text(json.dumps(full_identity, indent=2))
    rank = []
    with torch.inference_mode():
        for hmin, dmax, gmin, guard in itertools.product(
            [0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995],
            [0.4, 0.7, 1.0] if a.allow_leaf_disease_override else [0.2, 0.3, 0.4],
            [0.0] if shared_verifier else [0.0, 0.5, 0.8, 0.9],
            [False, True],
        ):
            from leaf_healthy_repair import fusion

            model.fusion = fusion(hmin, dmax, a.allow_leaf_disease_override).cuda()
            model.species_min = gmin
            final = reference.copy()
            final[ids] = (
                model.propose(
                    torch.from_numpy(reference[ids]).cuda(),
                    torch.from_numpy(healthy[ids]).cuda(),
                    torch.from_numpy(genus[ids]).cuda(),
                )
                .cpu()
                .numpy()
            )
            if guard or a.full_roi_recovery:
                fp = (
                    model.propose(
                        torch.from_numpy(full_reference[full_ids]).cuda(),
                        torch.from_numpy(full_healthy[full_ids]).cuda(),
                        torch.from_numpy(full_species[full_ids]).cuda(),
                    )
                    .cpu()
                    .numpy()
                )
                for k, i in enumerate(full_ids):
                    candidate = decision_key(posterior([final[i]], manifest))
                    old = decision_key(posterior([reference[i]], manifest))
                    full_candidate = decision_key(posterior([fp[k]], manifest))
                    if a.full_roi_recovery:
                        if (
                            old[0] not in ("HEALTHY", "DISEASE")
                            and full_candidate[0] == "HEALTHY"
                            and (not guard or candidate[0] == "HEALTHY")
                        ):
                            final[i] = fp[k]
                        else:
                            final[i] = reference[i]
                        continue
                    if (
                        candidate[0] == "HEALTHY"
                        and candidate != old
                        and decision_key(posterior([fp[k]], manifest)) != candidate
                    ):
                        final[i] = reference[i]
            proposed = {
                k: v["rows"]
                for k, v in app_evaluate(final, rows, ev, manifest, True).items()
            }
            for name, values in proposed.items():
                for i, row in enumerate(values):
                    before = baseline_photos[name][i]
                    if before["app_replay"]["kind"] == "DISEASE":
                        row["app_replay"] = before["app_replay"]
                        row["reference_disease_priority"] = True
            metrics = grouped(proposed)
            m = metrics["phase5_selection"]
            oracle = {
                "healthy_n": 0,
                "healthy_correct": 0,
                "negative_n": 0,
                "false_healthy": 0,
            }
            for i in field:
                prediction = (
                    decision_key(posterior([final[i]], manifest))[0] == "HEALTHY"
                )
                if rows[i]["health_y"] == 1:
                    oracle["healthy_n"] += 1
                    oracle["healthy_correct"] += prediction
                else:
                    oracle["negative_n"] += 1
                    oracle["false_healthy"] += prediction
            field_target = all(
                m["by_source"][source]["healthy_recall"] >= 0.7
                for source in ["tom2024", "swtt5wy2pr"]
            )
            score = float(
                np.mean(
                    [
                        (v["healthy"]["correct_rate"] or 0)
                        + (v["supported_disease"]["correct_rate"] or 0)
                        - 3 * (v["all"]["false_healthy_rate"] or 0)
                        - 2 * (v["unsupported_tomato"]["false_known_rate"] or 0)
                        for v in metrics.values()
                    ]
                )
            )
            c = dict(
                id=len(rank),
                policy=dict(healthy_min=hmin, disease_max=dmax, species_min=gmin),
                full_roi_confirmation=guard,
                reference_disease_priority=True,
                input_policy=(
                    "full_roi_recovery"
                    if a.full_roi_recovery
                    else "center_with_optional_full_roi_confirmation"
                ),
                metrics=metrics,
                field_leaf_oracle=oracle,
                retained=retain(metrics, baseline),
                field_healthy_passed=field_target,
                score=score,
                input_identity=identity,
                fresh_ad_used=False,
            )
            if a.allow_leaf_disease_override:
                c["policy"]["allow_leaf_disease_override"] = True
            rank.append(c)
            print(
                json.dumps(
                    dict(
                        id=c["id"],
                        retained=c["retained"],
                        field_passed=field_target,
                        Healthy=m["healthy"]["correct"],
                        false_Healthy=m["all"]["false_healthy"],
                        oracle=oracle,
                    )
                ),
                flush=True,
            )
    rank.sort(
        key=lambda c: (
            c["retained"] and c["field_healthy_passed"],
            c["retained"],
            c["score"],
        ),
        reverse=True,
    )
    for name, value in [
        ("ranked", rank),
        ("candidate", rank[0]),
        ("baseline", baseline),
    ]:
        (out / (name + ".json")).write_text(json.dumps(value, indent=2))
    print(
        json.dumps(
            dict(
                best=rank[0]["id"],
                retained=rank[0]["retained"],
                field_passed=rank[0]["field_healthy_passed"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
