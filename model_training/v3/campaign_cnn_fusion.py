"""Conservative CNN/DINO composition and validation-only logit calibration."""

from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import minimize
from campaign_cnn_train import sampling
from campaign_factorized import (
    T,
    selection_evaluate,
    selection_score,
    selection_retained,
)
from campaign_heads import app_evaluate


class ExpertFusion(nn.Module):
    def __init__(
        self,
        positive=0.75,
        negative=0.5,
        known=0.5,
        calibration=None,
        positive_scope="healthy",
        protect_healthy=True,
        known_scope="healthy",
        healthy_min=0.9,
        known_min=0.95,
        require_known_for_healthy=True,
        unhealthy_max=0.01,
        negative_policy="redistribute",
        require_healthy_for_known=False,
        disease_max=1.0,
        healthy_disease_margin=0.0,
    ):
        super().__init__()
        self.positive = positive
        self.negative = negative
        self.known = known
        self.positive_scope = positive_scope
        self.protect_healthy = protect_healthy
        self.known_scope = known_scope
        self.healthy_min = healthy_min
        self.known_min = known_min
        self.require_known_for_healthy = require_known_for_healthy
        self.unhealthy_max = unhealthy_max
        self.negative_policy = negative_policy
        self.require_healthy_for_known = require_healthy_for_known
        self.disease_max = disease_max
        self.healthy_disease_margin = healthy_disease_margin
        self.register_buffer("bias", torch.tensor([0.0] * 8 + [0.75]))
        self.register_buffer(
            "calibration",
            torch.tensor(calibration or [[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
        )

    def forward(self, reference, expert):
        lp = F.log_softmax((reference + self.bias) / T, 1)
        original = lp.exp()
        order = original.topk(2, dim=1)
        winner = order.indices[:, 0]
        margin = order.values[:, 0] - order.values[:, 1]
        disease = (
            (winner != 1)
            & (winner != 8)
            & (order.values[:, 0] >= 0.5)
            & (margin >= 0.15)
        )
        healthy = (winner == 1) & (order.values[:, 0] >= 0.6) & (margin >= 0.2)
        cnn = expert * self.calibration[:, 0] + self.calibration[:, 1]
        known_logit = lp[:, :8].logsumexp(1) - lp[:, 8]
        proposed_known = known_logit + self.known * (cnn[:, 1] - known_logit)
        change_known = (
            (winner == 8)
            & (cnn[:, 1].sigmoid() >= self.known_min)
            & (proposed_known > known_logit)
            & (self.known > 0)
        )
        # A binary expert cannot name a disease. Restrict rejection recovery to
        # the Healthy class already leading the original supported classes.
        if self.known_scope == "healthy":
            change_known = change_known & (lp[:, :8].argmax(1) == 1)
            if self.require_healthy_for_known:
                change_known = change_known & (cnn[:, 0].sigmoid() >= self.healthy_min)
        elif self.known_scope != "all":
            raise ValueError(self.known_scope)
        shift = torch.where(
            change_known,
            F.logsigmoid(proposed_known) - lp[:, :8].logsumexp(1),
            torch.zeros_like(known_logit),
        )
        adjusted = torch.cat(
            [
                lp[:, :8] + shift[:, None],
                torch.where(change_known, F.logsigmoid(-proposed_known), lp[:, 8])[
                    :, None
                ],
            ],
            1,
        )
        nonhealthy = torch.cat([adjusted[:, :1], adjusted[:, 2:]], 1).logsumexp(1)
        health_logit = adjusted[:, 1] - nonhealthy
        # A local specialist must not replace a named disease with Healthy.
        # The default only resolves uncertainty when Healthy already leads.
        eligible = winner == 1
        if self.positive_scope == "confirmed_known":
            eligible = eligible | (
                (winner == 8) & change_known & (lp[:, :8].argmax(1) == 1)
            )
        elif self.positive_scope == "legacy":
            eligible = ~disease
        elif self.positive_scope == "aggregate_reference":
            # Requires an 18-output bundle and aggregate reference disease
            # priority. Unsafe as a standalone nine-logit inference policy.
            eligible = torch.ones_like(disease)
        elif self.positive_scope != "healthy":
            raise ValueError(self.positive_scope)
        positive = (
            (cnn[:, 0].sigmoid() >= self.healthy_min)
            & eligible
            & (~healthy)
            & (cnn[:, 0] > health_logit)
        )
        if self.require_known_for_healthy:
            positive = positive & (cnn[:, 1].sigmoid() >= self.known_min)
        disease_probability = torch.cat([original[:, :1], original[:, 2:8]], 1).amax(1)
        positive = positive & (disease_probability <= self.disease_max)
        positive = positive & (
            original[:, 1] - disease_probability >= self.healthy_disease_margin
        )
        negative = (cnn[:, 0].sigmoid() <= self.unhealthy_max) & (
            cnn[:, 0] < health_logit
        )
        if self.negative_policy == "abstain":
            negative = negative & ((winner == 1) | (winner == 8))
        elif self.negative_policy != "redistribute":
            raise ValueError(self.negative_policy)
        if self.protect_healthy:
            negative = negative & (~healthy)
        strength = torch.where(
            positive,
            torch.full_like(health_logit, self.positive),
            torch.where(
                negative,
                torch.full_like(health_logit, self.negative),
                torch.zeros_like(health_logit),
            ),
        )
        new_logit = health_logit + strength * (cnn[:, 0] - health_logit)
        nonhealthy_shift = F.logsigmoid(-new_logit) - nonhealthy
        columns = [
            F.logsigmoid(new_logit) if i == 1 else adjusted[:, i] + nonhealthy_shift
            for i in range(9)
        ]
        modified_lp = torch.stack(columns, 1)
        if self.negative_policy == "abstain":
            # Binary not-Healthy evidence cannot identify another condition.
            # Distribute removed mass neutrally without creating a new verdict.
            probability = adjusted.exp()
            nonhealthy_max = torch.cat(
                [probability[:, :1], probability[:, 2:]], 1
            ).amax(1)
            room = (0.45 - nonhealthy_max).clamp_min(0) * 8
            removed = torch.minimum(
                (probability[:, 1] - new_logit.sigmoid()).clamp_min(0), room
            )
            neutral = torch.stack(
                [
                    (
                        probability[:, 1] - removed
                        if i == 1
                        else probability[:, i] + removed / 8
                    )
                    for i in range(9)
                ],
                1,
            )
            modified_lp = torch.where(
                (negative & (strength != 0))[:, None],
                neutral.clamp_min(1e-12).log(),
                modified_lp,
            )
        modified = reference + T * (modified_lp - lp)
        changed = change_known | (strength != 0)
        if self.negative_policy == "abstain":
            changed = change_known | ((strength != 0) & ((~negative) | (removed > 0)))
        return torch.where(changed[:, None], modified, reference)


def calibrate(logits, truth, weights):
    def objective(theta):
        p = logits * np.exp(theta[0]) + theta[1]
        return float(
            np.sum(weights * (np.logaddexp(0, p) - truth * p))
            + 0.001 * np.sum(np.square(theta))
        )

    result = minimize(
        objective, [0.0, 0.0], bounds=[(-2.0, 2.0), (-5.0, 5.0)], method="L-BFGS-B"
    )
    return [float(np.exp(result.x[0])), float(result.x[1])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--tag", default="protected")
    p.add_argument(
        "--positive-scope",
        default="healthy",
        choices=["healthy", "confirmed_known", "legacy"],
    )
    p.add_argument("--known-scope", default="healthy", choices=["healthy", "all"])
    p.add_argument(
        "--allow-healthy-downgrade",
        action="store_true",
        help="Allow independently validated specialist negative evidence on original Healthy winners",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Require retention independently on original and new validation domains",
    )
    a = p.parse_args()
    root = Path(a.root)
    run = Path(a.run)
    out = run / ("fusion_" + a.tag)
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    work = root / "dense_aligned"
    cache = torch.load(work / "features.pt", weights_only=True)
    jobs = json.loads((work / "feature_rows.json").read_text())
    eval_rows = json.loads((work / "eval_rows.json").read_text())
    rows = json.loads((root / "cnn_inputs/rows.json").read_text())
    expert = np.load(run / "all_logits.npy")
    assert all(rows[i]["feature_index"] == i for i in range(len(jobs)))
    reference = cache["logits"].cuda()
    pred = torch.from_numpy(expert[: len(jobs)]).cuda()
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "agribot_android_app/android/app/src/main/assets/model_manifest.json"
        ).read_text()
    )
    baseline = selection_evaluate(reference.cpu().numpy(), jobs, eval_rows, manifest)
    calibration = []
    for column, key in enumerate(["health_y", "known_y"]):
        indices = np.array(
            [i for i, r in enumerate(rows) if r["split"] == "val" and r[key] >= 0]
        )
        truth = np.array([rows[i][key] for i in indices])
        weights = sampling(rows, indices, target=key)
        calibration.append(calibrate(expert[indices, column], truth, weights))
    (out / "calibration.json").write_text(json.dumps(calibration, indent=2))
    ranked = []
    serial = 0
    with torch.inference_mode(), (out / "search.jsonl").open("w") as stream:
        for kind, parameters in [
            ("identity", [[1.0, 0.0], [1.0, 0.0]]),
            ("platt", calibration),
        ]:
            import itertools

            settings = (
                itertools.product(
                    [0.5, 1.0],
                    [0.0, 0.25, 0.5, 1.0],
                    [0.0, 1.0],
                    [0.9, 0.95, 0.99, 0.999],
                    [0.95, 0.99, 0.999],
                )
                if a.strict
                else itertools.product(
                    [0.0, 0.5, 0.75, 1.0],
                    [0.0, 0.25, 0.5, 0.75, 1.0],
                    [0.0, 0.5, 1.0],
                    [0.9],
                    [0.95],
                )
            )
            for positive, negative, known, healthy_min, known_min in settings:
                fusion = ExpertFusion(
                    positive,
                    negative,
                    known,
                    parameters,
                    a.positive_scope,
                    protect_healthy=not a.allow_healthy_downgrade,
                    known_scope=a.known_scope,
                    healthy_min=healthy_min,
                    known_min=known_min,
                ).cuda()
                final = fusion(reference, pred).cpu().numpy()
                selection = selection_evaluate(final, jobs, eval_rows, manifest)
                eligible = selection_retained(selection, baseline)
                if a.strict:
                    for scope in ["original", "new"]:
                        v, b = selection[scope], baseline[scope]
                        eligible = eligible and (
                            v["supported_disease"]["correct"]
                            >= b["supported_disease"]["correct"]
                            and v["supported_disease"]["false_healthy"]
                            <= b["supported_disease"]["false_healthy"]
                            and v["all"]["false_healthy"] <= b["all"]["false_healthy"]
                            and v["supported"]["false_other"]
                            <= b["supported"]["false_other"]
                            and all(
                                (v[s]["false_known"] or 0) <= (b[s]["false_known"] or 0)
                                for s in ["unsupported_tomato", "non_tomato"]
                            )
                        )
                record = dict(
                    id=serial,
                    calibration_kind=kind,
                    calibration=parameters,
                    positive=positive,
                    negative=negative,
                    known=known,
                    positive_scope=a.positive_scope,
                    protect_healthy=not a.allow_healthy_downgrade,
                    known_scope=a.known_scope,
                    healthy_min=healthy_min,
                    known_min=known_min,
                    strict=a.strict,
                    require_known_for_healthy=True,
                    validation_score=selection_score(selection),
                    retention_passed=eligible,
                    selection=selection,
                )
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                ranked.append(record)
                serial += 1
        ranked.sort(
            key=lambda r: (r["retention_passed"], r["validation_score"]), reverse=True
        )
        for candidate in ranked[:20]:
            fusion = ExpertFusion(
                candidate["positive"],
                candidate["negative"],
                candidate["known"],
                candidate["calibration"],
                a.positive_scope,
                protect_healthy=not a.allow_healthy_downgrade,
                known_scope=a.known_scope,
                healthy_min=candidate["healthy_min"],
                known_min=candidate["known_min"],
            ).cuda()
            final = fusion(reference, pred).cpu().numpy()
            candidate["app"] = app_evaluate(final, jobs, eval_rows, manifest)
            candidate["cnn_checkpoint"] = str(run / "best.pt")
            candidate["cnn_policy"] = json.loads((run / "result.json").read_text()).get(
                "pixel_policy", "center"
            )
            (out / f"candidate_{candidate['id']}.json").write_text(
                json.dumps(candidate, indent=2)
            )
            print(
                json.dumps(
                    dict(
                        id=candidate["id"],
                        retention=candidate["retention_passed"],
                        score=candidate["validation_score"],
                        positive=candidate["positive"],
                        negative=candidate["negative"],
                        known=candidate["known"],
                        app={
                            n: dict(
                                healthy=v["healthy"]["correct"],
                                disease=v["supported_disease"]["correct"],
                                false_healthy=v["supported_disease"]["false_healthy"],
                                false_other=v["supported"]["false_other"],
                            )
                            for n, v in candidate["app"].items()
                        },
                    )
                ),
                flush=True,
            )
    (out / "ranked.json").write_text(json.dumps(ranked, indent=2))
    print("CNN fusion calibration completed", run.name, flush=True)


if __name__ == "__main__":
    main()
