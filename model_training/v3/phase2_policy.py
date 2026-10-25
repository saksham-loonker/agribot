"""Evidence-gated nine-class adaptation with an immutable original comparator.

A trained nine-class expert supplies the proposed diagnosis. An independently
trained CNN must corroborate Healthy/knownness. Binary evidence alone never
creates a disease name. Old named diagnoses cannot switch to another disease.
"""

from pathlib import Path
import argparse, itertools, json, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from campaign_cnn_fusion import calibrate
from campaign_cnn_train import sampling
from campaign_heads import app_evaluate
from campaign_factorized import T
from campaign_verify import regression_retained
from app_eval import summarize


class DomainPolicy(nn.Module):
    def __init__(
        self,
        alpha=0.75,
        student_temperature=1.0,
        student_min=0.95,
        healthy_min=0.95,
        known_min=0.95,
        reference_disease_min=0.7,
        reference_disease_max=1.0,
        cnn_calibration=None,
        allow_unknown=True,
        healthy_reference_classes=None,
    ):
        super().__init__()
        self.alpha = alpha
        self.student_temperature = student_temperature
        self.student_min = student_min
        self.healthy_min = healthy_min
        self.known_min = known_min
        self.reference_disease_min = reference_disease_min
        self.reference_disease_max = reference_disease_max
        self.allow_unknown = allow_unknown
        self.register_buffer("bias", torch.tensor([0.0] * 8 + [0.75]))
        self.register_buffer(
            "calibration",
            torch.tensor(
                cnn_calibration or [[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32
            ),
        )
        self.register_buffer(
            "healthy_reference_mask",
            torch.tensor(
                [
                    healthy_reference_classes is None or i in healthy_reference_classes
                    for i in range(9)
                ]
            ),
        )

    def forward(self, reference, student, cnn):
        lp = F.log_softmax((reference + self.bias) / T, 1)
        base = lp.exp()
        proposed = F.softmax((student + self.bias) / (T * self.student_temperature), 1)
        oldp, oldclass = base.max(1)
        newp, newclass = proposed.max(1)
        cnn = cnn * self.calibration[:, 0] + self.calibration[:, 1]
        h = cnn[:, 0].sigmoid()
        k = cnn[:, 1].sigmoid()
        old_disease = (oldclass != 1) & (oldclass != 8)
        new_disease = (newclass != 1) & (newclass != 8)
        healthy_evidence = (
            (newclass == 1) & (h >= self.healthy_min) & (k >= self.known_min)
        )
        healthy_evidence = healthy_evidence & self.healthy_reference_mask[oldclass]
        competing = torch.cat([base[:, :1], base[:, 2:8]], 1).amax(1)
        # Keep unresolved original Healthy/disease ambiguity unresolved.
        healthy_evidence = healthy_evidence & (
            (oldclass != 1) | ((base[:, 1] - competing) >= 0.2)
        )
        healthy_evidence = healthy_evidence & (
            (~old_disease)
            | (
                (oldp >= self.reference_disease_min)
                & (oldp <= self.reference_disease_max)
            )
        )
        disease_evidence = (
            new_disease & (h <= 1 - self.healthy_min) & (k >= self.known_min)
        )
        # A different named disease needs a separate clinical/ontology review.
        disease_evidence = disease_evidence & ((~old_disease) | (oldclass == newclass))
        unknown_evidence = (
            (newclass == 8)
            & (k <= 1 - self.known_min)
            & (h <= 1 - self.healthy_min)
            & ((~old_disease) | (oldp < 0.5))
            & self.allow_unknown
        )
        eligible = (newp >= self.student_min) & (
            healthy_evidence | disease_evidence | unknown_evidence
        )
        # Preserve settled matching predictions exactly, including their confidence.
        settled = (oldclass == newclass) & (
            ((oldclass == 1) & (oldp >= 0.6))
            | ((oldclass == 8) & (oldp >= 0.5))
            | (old_disease & (oldp >= 0.5))
        )
        eligible = eligible & (~settled)
        mixed = (1 - self.alpha) * base + self.alpha * proposed
        winner = mixed.argmax(1)
        confidence = mixed.amax(1)
        decided = torch.where(winner == 1, confidence >= 0.6, confidence >= 0.5)
        # Contradictory experts do not erase an already decided original verdict.
        original_decided = torch.where(oldclass == 1, oldp >= 0.6, oldp >= 0.5)
        eligible = eligible & ((~original_decided) | decided)
        logits = reference + T * (mixed.clamp_min(1e-12).log() - lp)
        return torch.where(eligible[:, None], logits, reference)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--student", required=True)
    a = p.parse_args()
    root = Path(a.root)
    student_path = Path(a.student)
    out = student_path.parent / ("policy_v2_" + student_path.stem)
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    rows = json.loads((root / "cnn_inputs/rows.json").read_text())
    ev = json.loads((root / "eval_rows.json").read_text())
    reference = torch.from_numpy(np.load(root / "reference_logits.npy")).cuda()
    student = torch.from_numpy(
        np.load(student_path.parent / (student_path.stem + "_logits.npy"))
    ).cuda()
    manifest = json.loads(
        (
            Path(
                json.loads((root / "complete.json").read_text())[
                    "original_reference_assets"
                ]
            )
            / "model_manifest.json"
        ).read_text()
    )
    jobs = rows
    old_ev = {
        k: v for k, v in ev.items() if k not in ("selection_app", "phase2_selection")
    }

    def selection(z):
        result = app_evaluate(
            z,
            jobs,
            {k: ev[k] for k in ["selection_app", "phase2_selection"]},
            manifest,
            True,
        )
        old = result["selection_app"]["rows"]
        novel = result["phase2_selection"]["rows"]
        return dict(
            original=summarize(
                [r for r in old if r["source"] not in ("tlid", "pakistan")]
            ),
            expanded=summarize([r for r in old if r["source"] in ("tlid", "pakistan")]),
            novel=summarize(novel),
        )

    baseline = selection(reference.cpu().numpy())
    oldbaseline = app_evaluate(reference.cpu().numpy(), jobs, old_ev, manifest)

    def score(metrics):
        return float(
            np.mean(
                [
                    (m["healthy"]["correct_rate"] or 0)
                    + (m["supported_disease"]["correct_rate"] or 0)
                    - 3 * (m["all"]["false_healthy_rate"] or 0)
                    - (m["supported"]["false_other_rate"] or 0)
                    - 2 * (m["unsupported_tomato"]["false_known_rate"] or 0)
                    for m in metrics.values()
                ]
            )
        )

    def keep(metrics):
        return regression_retained(metrics, baseline)

    rank = []
    serial = 0
    started = time.time()
    with torch.inference_mode(), (out / "search.jsonl").open("w") as stream:
        for loss in ["bce", "focal"]:
            run = (
                root
                / "cnn_inputs/runs"
                / ("mobilenetv3_small_100.lamb_in1k_center_" + loss + "_n2")
            )
            raw = np.load(run / "all_logits.npy")
            cnn = torch.from_numpy(raw).cuda()
            parameters = []
            for column, key in enumerate(["health_y", "known_y"]):
                ids = np.array(
                    [
                        i
                        for i, r in enumerate(rows)
                        if r["split"] == "val" and r[key] >= 0
                    ]
                )
                truth = np.array([rows[i][key] for i in ids])
                parameters.append(
                    calibrate(raw[ids, column], truth, sampling(rows, ids, target=key))
                )
            for alpha, temp, smin, hmin, kmin, minimum in itertools.product(
                [1.0],
                [0.8],
                [0.8, 0.9, 0.97],
                [0.8, 0.9, 0.95],
                [0.5, 0.7, 0.85],
                [0.0, 0.7],
            ):
                spec = dict(
                    alpha=alpha,
                    student_temperature=temp,
                    student_min=smin,
                    healthy_min=hmin,
                    known_min=kmin,
                    reference_disease_min=minimum,
                    reference_disease_max=1.0,
                    cnn_calibration=parameters,
                )
                policy = DomainPolicy(**spec).cuda()
                final = policy(reference, student, cnn).cpu().numpy()
                metrics = selection(final)
                record = dict(
                    id=serial,
                    student_checkpoint=str(student_path),
                    cnn_checkpoint=str(run / "best.pt"),
                    policy=spec,
                    selection=metrics,
                    validation_score=score(metrics),
                    retention_passed=keep(metrics),
                )
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                rank.append(record)
                serial += 1
            print(
                json.dumps(
                    dict(
                        student=str(student_path),
                        cnn=loss,
                        trials=serial,
                        seconds=round(time.time() - started, 1),
                    )
                ),
                flush=True,
            )
        rank.sort(
            key=lambda r: (r["retention_passed"], r["validation_score"]), reverse=True
        )
        for r in rank[:35]:
            raw = torch.from_numpy(
                np.load(Path(r["cnn_checkpoint"]).parent / "all_logits.npy")
            ).cuda()
            final = (
                DomainPolicy(**r["policy"])
                .cuda()(reference, student, raw)
                .cpu()
                .numpy()
            )
            r["app"] = app_evaluate(final, jobs, old_ev, manifest)
            r["regression_passed"] = regression_retained(r["app"], oldbaseline)
            (out / f"candidate_{r['id']}.json").write_text(json.dumps(r, indent=2))
            print(
                json.dumps(
                    dict(
                        id=r["id"],
                        retained=r["retention_passed"],
                        regression_passed=r["regression_passed"],
                        score=r["validation_score"],
                        selection={
                            k: [
                                v["healthy"]["correct"],
                                v["healthy"]["n"],
                                v["supported_disease"]["correct"],
                                v["supported_disease"]["false_healthy"],
                                v["unsupported_tomato"]["false_known"],
                            ]
                            for k, v in r["selection"].items()
                        },
                        app={
                            k: [
                                v["healthy"]["correct"],
                                v["supported_disease"]["correct"],
                                v["all"]["false_healthy"],
                                v["supported"]["false_other"],
                            ]
                            for k, v in r["app"].items()
                        },
                    )
                ),
                flush=True,
            )
    (out / "ranked.json").write_text(json.dumps(rank, indent=2) + "\n")


if __name__ == "__main__":
    main()
