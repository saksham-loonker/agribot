"""Verify independent graph components and the selected union before cold data.

Uses the original eight opened cohorts and grouped internal validation only.
The retired cohort is never rescored as a benchmark. No AD predictions occur.
"""

from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import shutil

import numpy as np
from PIL import Image
import timm
import torch

from app_eval import classifier_letterbox_input
from campaign_heads import app_evaluate
from campaign_reference import safe_logits
from context_abstention import route_context
from mobileclip_scene import load, restore as restore_student
from phase2_select import retain
from phase5_scene_ablation import grouped
from phase11_context_model import restore as restore_teacher
from phase11_verify import infer
from phase12_scene_sft import teacher_logits


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(np.asarray(z, dtype=np.float64), -80, 80)))


def immutable(path, value):
    if path.exists():
        assert read(path) == value, ("Frozen identity changed", str(path))
    else:
        path.write_text(json.dumps(value, indent=2))


def prepare(path):
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    return classifier_letterbox_input(rgb, 224)


def student_infer(model, paths, path):
    if path.exists():
        return np.load(path)
    z = np.zeros(len(paths), np.float32)
    with torch.inference_mode(), concurrent.futures.ThreadPoolExecutor(8) as pool:
        for start in range(0, len(paths), 128):
            part = paths[start : start + 128]
            x = torch.from_numpy(
                np.concatenate(list(pool.map(prepare, part)), 0)
            ).cuda()
            z[start : start + len(part)] = model(x).cpu().numpy()
            if start % 1024 == 0:
                print(
                    json.dumps(
                        dict(
                            stage="student_exact",
                            complete=start + len(part),
                            total=len(paths),
                        )
                    ),
                    flush=True,
                )
    assert np.isfinite(z).all()
    np.save(path, z)
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    a = ap.parse_args()
    p = Path(a.campaign)
    root = p / "phase12_scene"
    out = root / "union_verification"
    out.mkdir(exist_ok=True)
    candidate_path = root / "union_selection/candidate.json"
    c = read(candidate_path)
    assert c["retained"] and c["field_healthy_passed"]
    policy = c["policy"]
    identity = c["input_identity"]
    teacher_bundle = p / "phase11_scene/prototype_verification/pytorch_bundle.pt"
    assert digest(teacher_bundle) == identity["teacher_bundle"]["sha256"]
    student_dir = root / ("sft_" + policy["student_loss"])
    assert (
        digest(student_dir / "best.pt")
        == identity["students"][policy["student_loss"]]["checkpoint_sha256"]
    )
    assert (
        digest(student_dir / "result.json")
        == identity["students"][policy["student_loss"]]["calibration_sha256"]
    )
    source_dir = out / "source_snapshot"
    source_dir.mkdir(exist_ok=True)
    source_hashes = {}
    for name in [
        "phase12_verify_union.py",
        "phase11_context_model.py",
        "mobileclip_scene.py",
        "mobileclip_features.py",
        "context_abstention.py",
    ]:
        src = Path(__file__).with_name(name)
        normalized = src.read_text().replace("\r\n", "\n")
        source_hashes[name] = hashlib.sha256(normalized.encode()).hexdigest()
        dst = source_dir / name
        if dst.exists():
            assert (
                hashlib.sha256(
                    dst.read_text().replace("\r\n", "\n").encode()
                ).hexdigest()
                == source_hashes[name]
            )
        else:
            shutil.copy2(src, dst)
    immutable(
        out / "protocol.json",
        dict(
            candidate_sha256=digest(candidate_path),
            identity=identity,
            source_hashes=source_hashes,
            retired_cohort_is_benchmark=False,
            cold_ad_used=False,
            scope="Original-eight regression and new grouped internal development only; no independent acceptance",
        ),
    )
    torch.set_num_threads(4)
    torch.manual_seed(593)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timm.layers.set_fused_attn(False)
    spec, student = load(student_dir)
    bundle = out / "student_bundle.pt"
    if not bundle.exists():
        torch.save(dict(spec, model=student.state_dict()), bundle)
    immutable(out / "student_bundle_identity.json", dict(sha256=digest(bundle)))
    student = student.cuda().eval()
    restored = restore_student(bundle).cuda().eval()
    with torch.inference_mode():
        x = torch.zeros(1, 224, 224, 3, device="cuda")
        torch.testing.assert_close(student(x), restored(x), rtol=0, atol=0)
    del student
    student = restored
    leaf = p / "phase10_leaf"
    rows = read(leaf / "cnn_inputs/rows.json")
    ev = read(leaf / "eval_rows.json")
    manifest = read(p / "original_reference/model_manifest.json")
    base = {
        k: v["rows"]
        for k, v in app_evaluate(safe_logits(leaf), rows, ev, manifest, True).items()
    }
    baseline = grouped(base)
    paths = sorted({r["path"] for values in base.values() for r in values})
    lookup = {path: i for i, path in enumerate(paths)}
    assert (
        hashlib.sha256(json.dumps(paths).encode()).hexdigest()
        == identity["paths_sha256"]
    )
    pair = np.load(p / "phase11_scene/prototype_verification/development/outputs.npy")
    teacher_h = sigmoid(pair[:, 0])
    genus = sigmoid(pair[:, 1])
    z = student_infer(student, paths, out / "student_development_logits.npy")
    student_h = sigmoid(z)
    expected_student = np.load(
        root / ("sft_" + policy["student_loss"] + "_selection") / "probabilities.npz"
    )["healthy"][:, 0]
    cache_error = float(np.max(np.abs(student_h - expected_student)))
    teacher = restore_teacher(teacher_bundle).cuda().eval()
    ids = set(np.linspace(0, len(paths) - 1, 48, dtype=int).tolist())
    ids.update(
        np.argsort(np.abs(teacher_h - policy["teacher_healthy_min"]))[:24].tolist()
    )
    ids.update(
        np.argsort(np.abs(student_h - policy["student_healthy_min"]))[:24].tolist()
    )
    probe = np.array(sorted(ids))
    errors = []
    with torch.inference_mode(), concurrent.futures.ThreadPoolExecutor(8) as pool:
        for start in range(0, len(probe), 16):
            part = probe[start : start + 16]
            x = torch.from_numpy(
                np.concatenate(list(pool.map(prepare, [paths[i] for i in part])), 0)
            ).cuda()
            errors.append(float(np.max(np.abs(teacher(x).cpu().numpy() - pair[part]))))
    teacher_error = max(errors)

    def recover(r, th, sh, g):
        teacher_positive = th >= policy["teacher_healthy_min"]
        student_positive = (
            sh >= policy["student_healthy_min"] and g >= policy["student_genus_min"]
        )
        return route_context(
            r,
            r,
            manifest,
            bool(teacher_positive or student_positive),
            False,
            float(th if teacher_positive else sh),
            False,
            False,
        )

    revised = {}
    matches = []
    for name, values in base.items():
        revised[name] = []
        for r in values:
            i = lookup[r["path"]]
            actual = recover(r, teacher_h[i], student_h[i], genus[i])
            expected = recover(r, teacher_h[i], expected_student[i], genus[i])
            revised[name].append(actual)

            def key(x):
                v = x["app_replay"]
                return v["kind"], v["label"], v.get("scope")

            matches.append(key(actual) == key(expected))
    metrics = grouped(revised)
    field = all(
        metrics["phase5_selection"]["by_source"][s]["healthy_recall"] >= 0.7
        for s in ["tom2024", "swtt5wy2pr"]
    )
    regression = dict(
        passed=all(matches)
        and cache_error < 0.002
        and teacher_error < 0.002
        and retain(metrics, baseline)
        and field,
        verdicts=len(matches),
        verdict_matches=sum(matches),
        student_probability_max_error=cache_error,
        teacher_probe_logit_max_error=teacher_error,
        teacher_probe_cases=len(probe),
        retained=retain(metrics, baseline),
        metrics=metrics,
    )
    (out / "regression_parity.json").write_text(json.dumps(regression, indent=2))
    assert regression[
        "passed"
    ], "Exact union differs or fails regression; do not open cold data"
    data = read(root / "cnn_inputs/rows.json")
    admitted = read(root / "admission_manifest.json")
    offset = len(data) - len(admitted)
    teacher_all = teacher_logits(root, data)
    _, new_pair = infer(
        teacher,
        [r["path"] for r in admitted],
        out / "admitted_training_development_teacher",
    )
    teacher_all[offset:] = new_pair[:, 0]
    np.save(out / "teacher_training_development_logits.npy", teacher_all)
    genus_old = np.load(
        p / "phase6_roi/scene_sft_d12_focal_fusion/calibrated_logits.npy"
    )[:, 3]
    genus_all = sigmoid(np.concatenate([genus_old, new_pair[:, 1]]))
    validation = np.load(student_dir / "validation_logits.npz")
    ids = validation["indices"]
    student_raw = validation["logits"]
    cal = read(student_dir / "result.json")["calibration"]
    sh = sigmoid(student_raw * cal[0] + cal[1])
    th = sigmoid(teacher_all[ids])
    truth = np.array([r["health_y"] for r in data])[ids]
    positive = (th >= policy["teacher_healthy_min"]) | (
        (sh >= policy["student_healthy_min"])
        & (genus_all[ids] >= policy["student_genus_min"])
    )
    primary = np.array([data[i]["source"] == "tom2024" for i in ids])
    new = ids >= offset

    def counts(mask):
        h = mask & (truth == 1)
        d = mask & (truth == 0)
        return dict(
            healthy_n=int(h.sum()),
            healthy_correct=int(positive[h].sum()),
            healthy_recall=float(positive[h].mean()) if h.any() else None,
            negative_n=int(d.sum()),
            false_healthy=int(positive[d].sum()),
            negative_false_positive=float(positive[d].mean()) if d.any() else None,
        )

    internal = dict(
        all=counts(np.ones(len(ids), bool)),
        primary=counts(primary),
        admitted_families=counts(new),
        scope="Grouped internal validation, not an independent benchmark; base checkpoint selection and calibration used this development corpus",
        cold_ad_used=False,
    )
    internal["passed_healthy_target"] = all(
        m["healthy_recall"] is None or m["healthy_recall"] >= 0.7
        for m in [internal["all"], internal["primary"], internal["admitted_families"]]
    )
    (out / "internal_validation.json").write_text(json.dumps(internal, indent=2))
    result = dict(
        passed_preacceptance=regression["passed"] and internal["passed_healthy_target"],
        regression_passed=regression["passed"],
        internal_validation=internal,
        independent_acceptance_opened=False,
        retired_cohort_scored_as_benchmark=False,
        teacher_bundle_sha256=digest(teacher_bundle),
        student_bundle_sha256=digest(bundle),
        policy=policy,
        promotable=False,
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
