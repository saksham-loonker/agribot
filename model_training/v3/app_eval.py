"""Exact TFLite + app-policy replay for baseline and candidate bundles.

This is a portable generalisation of tests/model_evaluation/evaluate_public.py.
It preserves the historical behaviour: camera frames are simulated by repeating a
static image; repeated frames are consensus probes, never independent photos.
Model SHA-256 and input shapes are checked before inference.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from PIL import Image
from evaluation_inputs import input_identity

ROOT = Path(__file__).resolve().parents[2]
CANON_PATH = ROOT / "model_training/v2/final/canon.py"
spec = importlib.util.spec_from_file_location("agribot_canonical", CANON_PATH)
canon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canon)
sys.path.insert(0, str(ROOT / "tests/model_evaluation"))
from canonical_runtime import install as install_sparse_resize

install_sparse_resize(canon)
LABELS = [
    "Early_blight",
    "Healthy",
    "Late_blight",
    "Leaf Miner",
    "Magnesium Deficiency",
    "Nitrogen Deficiency",
    "Pottassium Deficiency",
    "Spotted Wilt Virus",
    "Other",
]
OTHER = LABELS.index("Other")
HEALTHY = LABELS.index("Healthy")


def evaluation_identity(assets, manifest, limit=None):
    """Resume is valid only for identical models, inputs, policy and implementation."""
    assets = Path(assets)
    model_manifest = assets / "model_manifest.json"
    bundle = json.loads(model_manifest.read_text())
    files = [
        model_manifest,
        Path(manifest),
        Path(__file__),
        CANON_PATH,
        ROOT / "tests/model_evaluation/canonical_runtime.py",
    ]
    files += [assets / "models" / m["file"] for m in bundle["classifier"]["members"]]
    files += [assets / "models" / bundle["detector"]["file"]]
    if bundle.get("schema") in (3, 4):
        files.append(Path(__file__).with_name("view_guard_policy.py"))
    files.append(Path(__file__).with_name("evaluation_inputs.py"))
    records = json.loads(Path(manifest).read_text())
    if limit:
        records = records[:limit]
    return dict(
        schema=2,
        limit=limit,
        files=[hashlib.sha256(p.read_bytes()).hexdigest() for p in files],
        inputs=input_identity(records),
    )


def posterior(member_logits, manifest, bias=None, policy=None):
    c = manifest["classifier"]
    mean = np.mean([canon.softmax(z) for z in member_logits], axis=0)
    combined = np.log(mean + 1e-12)
    other_bias = c.get("other_logit_bias", 0.0) if bias is None else bias
    combined[c.get("other_index", OTHER)] += other_bias
    probs = canon.softmax(combined, c.get("temperature", 1.0)).astype(np.float32)
    threshold = manifest.get("_energy_reject_below")
    if threshold is not None:
        temperature = c.get("temperature", 1.0)
        z = combined / temperature
        known = [k for k in range(len(LABELS)) if k != c.get("other_index", OTHER)]
        energy = temperature * (
            np.log(np.exp(z[known] - z[known].max()).sum()) + z[known].max()
        )
        if energy < threshold:
            probs = probs.copy()
            probs[:] = 0.0
            probs[c.get("other_index", OTHER)] = 1.0
    return probs


def classifier_letterbox_input(image, size, box=None):
    if box is not None:
        l, t, r, b = canon.crop_box(image.shape[1], image.shape[0], box, 0.10)
        image = image[t:b, l:r]
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    nw = max(1, min(size, round(w * scale)))
    nh = max(1, min(size, round(h * scale)))
    px, py = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), 114.0, np.float32)
    out[py : py + nh, px : px + nw] = canon.resize(image, nw, nh)
    return ((out / 255.0 - canon.MEAN) / canon.STD)[None]


def quality(image):
    h, w = image.shape[:2]
    stride = max(1, int(math.sqrt((h * w) // 24000)))
    rgb = image[::stride, ::stride].astype(np.float64)
    sample = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    brightness = float(sample.mean() / 255)
    contrast = float(sample.std() / 255)
    grad = np.abs(np.diff(sample, axis=0)).sum() + np.abs(np.diff(sample, axis=1)).sum()
    pairs = sample.shape[0] * max(0, sample.shape[1] - 1) + sample.shape[1] * max(
        0, sample.shape[0] - 1
    )
    sharpness = float(grad / max(1, pairs) / 255)
    accept = not (
        brightness < 0.08
        or brightness > 0.95
        or (contrast < 0.015 and sharpness < 0.012)
        or (sharpness < 0.012 and contrast < 0.08)
    )
    return dict(
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        quality_accept=accept,
        quality_weight=min(1.0, max(0.4, 0.4 + 12 * sharpness)) if accept else 0.3,
    )


def replay(leaves, manifest, weight=1.0, bias=None, other_min_prob=0.5):
    if leaves and manifest["classifier"].get("reference_disease_priority", False):
        reference = []
        for leaf in leaves:
            assert (
                leaf.get("reference_member_logits") is not None
            ), "Disease-priority replay needs reference logits"
            reference.append(dict(leaf, member_logits=leaf["reference_member_logits"]))
        reference_manifest = dict(
            manifest,
            classifier=dict(manifest["classifier"], reference_disease_priority=False),
        )
        retained = replay(reference, reference_manifest, weight, bias, other_min_prob)
        if retained["kind"] == "DISEASE":
            retained["reference_disease_priority"] = True
            return retained
    if not leaves:
        return dict(
            kind="NO_LEAVES",
            label=None,
            confidence=None,
            frames=0,
            leaves=0,
            partial=False,
        )
    ps = np.asarray(
        [posterior(leaf["member_logits"], manifest, bias) for leaf in leaves],
        np.float64,
    )
    obs_weights = np.clip(
        np.asarray([leaf["score"] for leaf in leaves], np.float32) * np.float32(weight),
        0.05,
        1,
    ).astype(np.float64)
    last = None
    stable = 0
    for frame in range(1, 15):
        weights = np.minimum(frame * obs_weights, 3.0)
        plant = np.sum(ps * weights[:, None], axis=0) / weights.sum()
        order = np.argsort(-plant, kind="stable")
        top, second = map(int, order[:2])
        p1, p2 = np.float32(plant[top]), np.float32(plant[second])
        partial = False
        kind, label, confidence = "NEED_MORE_VIEWS", None, float(p1)
        if frame >= 3 and frame * len(leaves) >= 5:
            candidates = []
            for disease in range(len(LABELS)):
                if disease in (HEALTHY, OTHER):
                    continue
                strong = ps[:, disease] >= np.float32(0.7)
                if strong.sum() >= 2:
                    candidates.append(
                        (int(strong.sum()), float(ps[strong, disease].mean()), disease)
                    )
            best = (
                max(candidates, key=lambda candidate: candidate[:2])
                if candidates
                else None
            )
            if (
                best
                and top != best[2]
                and (top in (HEALTHY, OTHER) or p1 - p2 < np.float32(0.15))
            ):
                kind, label, confidence, partial = "DISEASE", best[2], best[1], True
            else:
                label = top
                if top == OTHER:
                    kind = (
                        "UNKNOWN_CONDITION"
                        if p1 >= np.float32(other_min_prob)
                        else "UNSURE"
                    )
                elif top == HEALTHY:
                    kind = (
                        "HEALTHY"
                        if p1 >= np.float32(0.60) and p1 - p2 >= np.float32(0.20)
                        else "UNSURE"
                    )
                else:
                    kind = (
                        "DISEASE"
                        if p1 >= np.float32(0.50) and p1 - p2 >= np.float32(0.15)
                        else "UNSURE"
                    )
        key = (kind, label)
        stable = stable + 1 if key == last else 0
        last = key
        if kind != "NEED_MORE_VIEWS" and (
            (kind != "UNSURE" and stable >= 2) or frame >= 14
        ):
            return dict(
                kind=kind,
                label=LABELS[label] if label is not None else None,
                confidence=confidence,
                frames=frame,
                leaves=len(leaves),
                partial=partial,
                probabilities=plant.tolist(),
                margin=float(p1 - p2),
            )
    raise RuntimeError("Static replay failed to terminate")


class Models:
    def __init__(self, assets: Path, threads: int = 4):
        self.assets = assets
        self.manifest = json.loads((assets / "model_manifest.json").read_text())
        assert (self.manifest.get("schema") == 4) == bool(
            self.manifest["classifier"].get("reference_disease_priority", False)
        ), "Schema 4 requires reference disease priority"
        self.manifest["_energy_reject_below"] = (
            self.manifest["classifier"].get("energy_reject_below")
            if self.manifest["classifier"].get("energy_gate")
            else None
        )
        self.members = []
        for member in self.manifest["classifier"]["members"]:
            self.members.append((self.load(member, threads), member["input"][1]))
        self.detector = self.load(self.manifest["detector"], threads)

    def load(self, contract, threads):
        path = self.assets / "models" / contract["file"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != contract["sha256"]:
            raise RuntimeError(
                f"Model SHA mismatch: {path} expected {contract['sha256']} got {actual}"
            )
        interpreter = Interpreter(model_path=str(path), num_threads=threads)
        interpreter.allocate_tensors()
        if interpreter.get_input_details()[0]["shape"].tolist() != contract["input"]:
            raise RuntimeError(f"Input shape mismatch: {path}")
        return interpreter

    @staticmethod
    def invoke(interpreter, tensor):
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
        if not np.isfinite(output).all():
            raise RuntimeError("Non-finite model output")
        return output

    def classify(self, image, box=None):
        outputs = [
            self.invoke(interpreter, canon.classifier_input(image, size, box)).ravel()
            for interpreter, size in self.members
        ]
        self.last_reference_logits = (
            [outputs[0][len(LABELS) : 2 * len(LABELS)].tolist()]
            if self.manifest["classifier"].get("reference_disease_priority", False)
            else None
        )
        return self.select_view_logits(
            outputs, lambda size: classifier_letterbox_input(image, size, box)
        )

    def select_view_logits(self, outputs, alternative_input):
        confirmation = self.manifest["classifier"].get("view_confirmation")
        if confirmation is None:
            return [z.tolist() for z in outputs]
        from view_guard_policy import decision_key, needs_confirmation

        policy = confirmation.get("policy")
        assert (
            self.manifest["schema"] in (3, 4)
            and len(outputs) == 1
            and confirmation
            == dict(
                policy=policy,
                reference_logit_offset=len(LABELS),
                padding_value=114,
            )
            and policy in ("changed_positive_full_roi", "changed_decision_full_roi")
        )
        raw = outputs[0]
        assert raw.shape == (2 * len(LABELS),)
        candidate, reference = raw[: len(LABELS)], raw[len(LABELS) :]
        cp = posterior([candidate], self.manifest)
        rp = posterior([reference], self.manifest)
        if needs_confirmation(
            rp, cp, confirm_unknown=policy == "changed_decision_full_roi"
        ):
            interpreter, size = self.members[0]
            full = self.invoke(interpreter, alternative_input(size)).ravel()[
                : len(LABELS)
            ]
            if decision_key(cp) != decision_key(posterior([full], self.manifest)):
                candidate = reference
        return [candidate.tolist()]

    def detect(self, image):
        contract = self.manifest["detector"]
        tensor, lb = canon.detector_input(image, contract["input"][1])
        output = self.invoke(self.detector, tensor)
        return canon.decode_detections(
            output,
            lb,
            image.shape[1],
            image.shape[0],
            contract["score_threshold"],
            contract["nms_iou"],
            contract["max_detections"],
        )


def image_record(models, record):
    image = np.asarray(Image.open(record["path"]).convert("RGB"))
    info = {k: v for k, v in record.items()}
    info.update(quality(image))
    whole = models.classify(image)
    probs = posterior(whole, models.manifest)
    info["whole_member_logits"] = whole
    info["whole_probabilities"] = probs.tolist()
    info["whole_label"] = LABELS[int(probs.argmax())]
    detections = models.detect(image)
    leaves = []
    for box in detections[:6]:
        logits = models.classify(image, box[:4])
        p = posterior(logits, models.manifest)
        leaf = dict(
            box=box[:4],
            score=box[4],
            member_logits=logits,
            probabilities=p.tolist(),
            label=LABELS[int(p.argmax())],
        )
        if models.manifest["classifier"].get("reference_disease_priority", False):
            leaf["reference_member_logits"] = models.last_reference_logits
        leaves.append(leaf)
    info["detections"] = detections
    info["leaves"] = leaves
    info["app_replay"] = replay(leaves, models.manifest, info["quality_weight"])
    info["ungated_app_replay"] = info["app_replay"]
    # The hardened scan flow never accumulates classifier evidence from rejected pixels.
    if info["contrast"] < 0.015 and info["sharpness"] < 0.012:
        info["app_replay"] = dict(
            kind="RETRY_QUALITY",
            label=None,
            confidence=None,
            frames=0,
            leaves=0,
            partial=False,
        )
    info["host_elapsed_ms"] = None
    oracle = []
    for index, obj in enumerate(record.get("objects", [])):
        box = obj["box"]
        l, t, r, b = canon.crop_box(image.shape[1], image.shape[0], box, 0.0)
        annotation_size = record.get("annotation_size")
        dimension_match = annotation_size is None or annotation_size == [
            image.shape[1],
            image.shape[0],
        ]
        valid = (
            dimension_match
            and r - l >= 2
            and b - t >= 2
            and 0 <= box[0] < box[2] <= image.shape[1] + 1
            and 0 <= box[1] < box[3] <= image.shape[0] + 1
        )
        if not valid:
            oracle.append(
                dict(
                    index=index,
                    valid=False,
                    source_label=obj.get("source_label"),
                    label=obj.get("label"),
                    box=box,
                )
            )
            continue
        logits = models.classify(image, box)
        p = posterior(logits, models.manifest)
        oracle.append(
            dict(
                index=index,
                valid=True,
                source_label=obj.get("source_label"),
                label=obj.get("label"),
                box=box,
                member_logits=logits,
                prediction=LABELS[int(p.argmax())],
                probabilities=p.tolist(),
                best_detector_iou=max(
                    (canon.iou(box, d[:4]) for d in detections), default=0.0
                ),
                best_selected_iou=max(
                    (canon.iou(box, d[:4]) for d in detections[:6]), default=0.0
                ),
            )
        )
    info["oracle_objects"] = oracle
    return info


WORKER = None


def init_worker(assets, threads):
    global WORKER
    WORKER = Models(Path(assets), threads)


def evaluate_one(record):
    try:
        return image_record(WORKER, record)
    except Exception as exc:
        return dict(id=record.get("id"), error=repr(exc))


def scope_of(record):
    objects = record.get("objects", [])
    names = [o.get("source_label", "") for o in objects]
    if record["label"] == "Diseased_unspecified":
        return "coarse_diseased"
    if record["label"] not in ("Other", "Mixed"):
        return "supported_tomato"
    if record["label"] == "Mixed":
        # Several supported labels in a scene are not an unsupported disease.
        # Keep object diagnostics, but exclude ambiguous scene labels from scalar metrics.
        return "mixed"
    if record["source"] == "plantvillage" or record.get("species") == "tomato":
        return "unsupported_tomato"
    if names and all(n.startswith("Tomato") for n in names):
        return "unsupported_tomato"
    if names and all(not n.startswith("Tomato") for n in names):
        return "non_tomato"
    return "mixed_species_other"


def summarize(rows, eligible_only=True):
    rows = [
        r
        for r in rows
        if "error" not in r and (r.get("analysis_eligible", True) or not eligible_only)
    ]
    mixed = [r for r in rows if r["label"] == "Mixed"]
    rows = [r for r in rows if r["label"] != "Mixed"]
    out = {"excluded_mixed_photos": len(mixed), "scalar_eligible_photos": len(rows)}
    groups = {
        "all": lambda r: True,
        "supported": lambda r: scope_of(r) == "supported_tomato",
        "healthy": lambda r: r["label"] == "Healthy",
        "supported_disease": lambda r: r["label"] in LABELS
        and r["label"] not in ("Healthy", "Other"),
        "coarse_diseased": lambda r: r["label"] == "Diseased_unspecified",
        "unsupported_tomato": lambda r: scope_of(r) == "unsupported_tomato",
        "non_tomato": lambda r: scope_of(r) == "non_tomato",
    }
    for name, predicate in groups.items():
        subset = [r for r in rows if predicate(r)]
        kinds = Counter(r["app_replay"]["kind"] for r in subset)
        correct = sum(
            (r["app_replay"]["kind"] == "HEALTHY" and r["label"] == "Healthy")
            or (
                r["app_replay"]["kind"] == "DISEASE"
                and r["app_replay"]["label"] == r["label"]
            )
            for r in subset
        )
        false_healthy = sum(
            r["app_replay"]["kind"] == "HEALTHY" and r["label"] != "Healthy"
            for r in subset
        )
        false_other = sum(
            r["app_replay"]["kind"] == "UNKNOWN_CONDITION"
            for r in subset
            if r["label"] in LABELS and r["label"] != "Other"
        )
        false_known = (
            sum(r["app_replay"]["kind"] in ("HEALTHY", "DISEASE") for r in subset)
            if any(r["label"] == "Other" for r in subset)
            else None
        )
        entry = dict(
            n=len(subset),
            correct=correct,
            correct_rate=correct / len(subset) if subset else None,
            healthy_verdict=sum(r["app_replay"]["kind"] == "HEALTHY" for r in subset),
            healthy_recall=(
                sum(r["app_replay"]["kind"] == "HEALTHY" for r in subset) / len(subset)
                if name == "healthy" and subset
                else None
            ),
            false_healthy=false_healthy,
            false_healthy_rate=false_healthy / len(subset) if subset else None,
            false_other=false_other,
            false_other_rate=false_other / len(subset) if subset else None,
            false_known=false_known,
            false_known_rate=(
                false_known / len(subset)
                if false_known is not None and subset
                else None
            ),
            **{f"kind_{k}": v for k, v in kinds.items()},
        )
        out[name] = entry
    healthy_rows = [r for r in rows if r["label"] == "Healthy"]
    healthy_verdicts = sum(r["app_replay"]["kind"] == "HEALTHY" for r in healthy_rows)
    decided_healthy = sum(r["app_replay"]["kind"] == "HEALTHY" for r in rows)
    out["healthy_precision"] = (
        healthy_verdicts / decided_healthy if decided_healthy else None
    )
    out["healthy_precision_n"] = decided_healthy
    out["by_source"] = {}
    for source in sorted({r["source"] for r in rows}):
        subset = [r for r in rows if r["source"] == source]
        out["by_source"][source] = dict(
            n=len(subset),
            healthy_recall=(
                sum(
                    r["app_replay"]["kind"] == "HEALTHY"
                    for r in subset
                    if r["label"] == "Healthy"
                )
                / max(1, sum(r["label"] == "Healthy" for r in subset))
            ),
            false_healthy=sum(
                r["app_replay"]["kind"] == "HEALTHY"
                for r in subset
                if r["label"] != "Healthy"
            ),
        )
    oracle = [
        o for r in rows + mixed for o in r.get("oracle_objects", []) if o.get("valid")
    ]
    out["oracle_objects"] = dict(
        n=len(oracle),
        correct=sum(o["prediction"] == o["label"] for o in oracle),
        false_other=sum(
            o["prediction"] == "Other" and o["label"] != "Other" for o in oracle
        ),
        healthy_n=sum(o["label"] == "Healthy" for o in oracle),
        healthy_correct=sum(
            o["prediction"] == "Healthy" and o["label"] == "Healthy" for o in oracle
        ),
        false_healthy=sum(
            o["prediction"] == "Healthy" and o["label"] != "Healthy" for o in oracle
        ),
    )
    return out


def r_scope_is_other(subset):
    return any(r["label"] == "Other" for r in subset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assets",
        required=True,
        help="bundle directory containing model_manifest.json and models/",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = json.loads(Path(args.manifest).read_text())
    if args.limit:
        records = records[: args.limit]
    output = out / "predictions.jsonl"
    identity_path = out / "evaluation_identity.json"
    identity = evaluation_identity(args.assets, args.manifest, args.limit)
    if output.exists():
        if not args.resume:
            raise ValueError(
                "Output already exists: choose a new directory or --resume"
            )
        if (
            not identity_path.exists()
            or json.loads(identity_path.read_text()) != identity
        ):
            raise ValueError(
                "Resume model/data/implementation identity differs; use a new output directory"
            )
    identity_path.write_text(json.dumps(identity, indent=1))
    done = {}
    if args.resume and output.exists():
        snapshot = output.read_text()
        # Recover only a final incomplete write; malformed complete records remain errors.
        if snapshot and not snapshot.endswith("\n"):
            snapshot = snapshot.rsplit("\n", 1)[0] + "\n" if "\n" in snapshot else ""
            output.write_text(snapshot)
        for line in snapshot.splitlines():
            row = json.loads(line)
            done[row.get("id")] = row
    pending = [r for r in records if r["id"] not in done]
    models = Models(Path(args.assets), args.threads)
    errors = []
    started = time.time()
    if args.workers > 1:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=(args.assets, args.threads),
        )
        iterator = executor.map(evaluate_one, pending)
    else:
        init_worker(args.assets, args.threads)
        iterator = map(evaluate_one, pending)
    with output.open("a", encoding="utf-8") as stream:
        for index, row in enumerate(iterator, len(done) + 1):
            if "error" in row:
                errors.append(row)
                print("EVALUATION ERROR", row, flush=True)
            else:
                done[row["id"]] = row
                stream.write(json.dumps(row) + "\n")
                stream.flush()
            if index % 25 == 0:
                print(
                    f"{args.tag}: {index}/{len(records)} {time.time() - started:.0f}s",
                    flush=True,
                )
    if args.workers > 1:
        executor.shutdown()
    rows = list(done.values())
    metrics = summarize(rows)
    metrics.update(
        tag=args.tag,
        manifest=str(Path(args.manifest).resolve()),
        assets=str(Path(args.assets).resolve()),
        records=len(rows),
        errors=len(errors),
        elapsed_s=round(time.time() - started, 1),
        runtime=dict(
            python=sys.version,
            platform=platform.platform(),
            litert=__import__("importlib.metadata", fromlist=["version"]).version(
                "ai-edge-litert"
            ),
            numpy=np.__version__,
            workers=args.workers,
            threads=args.threads,
            note="Static repeated-frame app replay; not a camera trial.",
        ),
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (out / "errors.json").write_text(json.dumps(errors, indent=1))
    print(json.dumps(metrics, indent=1))
    if errors or len(rows) != len(records):
        raise RuntimeError(
            "Evaluation incomplete; inspect errors.json before interpreting metrics"
        )


if __name__ == "__main__":
    main()
