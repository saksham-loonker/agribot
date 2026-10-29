"""Fit one small leaf repair head and select on cached actual app verdicts."""

from pathlib import Path
import argparse
import copy
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from app_eval import posterior
from campaign_cnn_fusion import calibrate
from campaign_heads import app_evaluate, fast_static_replay
from campaign_reference import safe_logits
from compact_healthy_model import calibrated_probabilities, propose
from leaf_feature_expert import LeafFeatureHead
from opened_context_data import load, read
from permissive_encoder import digest
from phase17_context_select import coarse_counts
from phase2_select import retain
from phase29_compact_prepare import region_key
from phase5_scene_ablation import grouped, photo_metrics
from phase7_species_train import balanced
from source_balance import source_class_group_weights
from training_support import atomic_save
from train import LABELS


def healthy_only(reference, proposal):
    """A binary repair cannot replace original diagnoses or name a disease."""
    v = reference["app_replay"]
    if v["kind"] in ("DISEASE", "HEALTHY"):
        return reference
    if v["kind"] == "UNSURE" and v.get("label") not in (None, "", "Healthy", "Other"):
        return reference
    return proposal if proposal["app_replay"]["kind"] == "HEALTHY" else reference


class AppScreen:
    def __init__(self, campaign, root, device):
        self.p, self.root, self.device = Path(campaign), Path(root), device
        self.data = load(campaign)
        leaf = self.p / "phase10_leaf"
        self.jobs = read(leaf / "cnn_inputs/rows.json")
        self.ev = read(leaf / "eval_rows.json")
        self.safe = safe_logits(leaf)
        self.old_indices = np.array(
            [i for i, r in enumerate(self.jobs) if r["set"] in self.ev]
        )
        self.center = read(self.root / "center_indices.json")
        self.full = read(self.root / "full_indices.json")
        self.old_center = np.array(
            [
                self.center[region_key(self.jobs[i]["path"], self.jobs[i]["box"])]
                for i in self.old_indices
            ]
        )
        self.old_full = np.array(
            [
                self.full[region_key(self.jobs[i]["path"], self.jobs[i]["box"])]
                for i in self.old_indices
            ]
        )
        self.reference = torch.from_numpy(
            np.load(self.root / "reference_logits.npy")
        ).to(device)
        self.baseline = grouped(self.data["base"])
        self.sources = sorted({r["source"] for r in self.data["internal"]})
        self.internal_baseline = {
            s: photo_metrics([r for r in self.data["internal"] if r["source"] == s])
            for s in self.sources
        }
        self.coarse_before = coarse_counts(self.data["base"])
        # Verify that exposing the deployed encoders' pre-logit features has
        # not changed a single pre-existing reference decision.
        raw = self.safe.copy()
        raw[self.old_indices] = self.reference[self.old_center].cpu().numpy()
        control = {
            k: v["rows"]
            for k, v in app_evaluate(
                raw, self.jobs, self.ev, self.data["manifest"], True
            ).items()
        }

        def key(row):
            r = row["app_replay"]
            return r["kind"], r.get("label"), r.get("partial", False)

        assert all(
            key(a) == key(b)
            for k in control
            for a, b in zip(control[k], self.data["base"][k])
        ), "Feature exposure changed the frozen reference verdict"

    def evaluate(self, probability, minimum):
        with torch.inference_mode():
            proposal = propose(self.reference, probability, minimum)
            full_probability = calibrated_probabilities(proposal)
            # Every changed proposal is Healthy by construction. Confirm the
            # same diagnosis on the actual padded full-ROI input, as Android.
            order = full_probability.topk(2, dim=1)
            h = (
                (order.indices[:, 0] == 1)
                & (order.values[:, 0] >= 0.6)
                & (order.values[:, 0] - order.values[:, 1] >= 0.2)
            )
            active = (proposal - self.reference).abs().amax(1) > 1e-7
            use = active & h
            z = proposal.cpu().numpy()
            accept = use.cpu().numpy()
            full_h = h.cpu().numpy()
        leaf_values = self.safe.copy()
        approved = accept[self.old_center] & full_h[self.old_full]
        leaf_values[self.old_indices[approved]] = z[self.old_center[approved]]
        candidate = {
            k: v["rows"]
            for k, v in app_evaluate(
                leaf_values, self.jobs, self.ev, self.data["manifest"], True
            ).items()
        }
        for name in candidate:
            candidate[name] = [
                healthy_only(reference, row)
                for reference, row in zip(self.data["base"][name], candidate[name])
            ]
        internal = []
        for reference in self.data["internal"]:
            row = copy.deepcopy(reference)
            for leaf in row["leaves"]:
                k = region_key(row["path"], leaf["box"])
                center = self.center[k]
                full = self.full[k]
                if accept[center] and full_h[full]:
                    leaf["member_logits"] = [z[center].tolist()]
            row["app_replay"] = fast_static_replay(
                row["leaves"], self.data["manifest"], row["quality_weight"]
            )
            internal.append(healthy_only(reference, row))
        observed = grouped(candidate)
        metrics = {
            s: photo_metrics([r for r in internal if r["source"] == s])
            for s in self.sources
        }
        coarse = coarse_counts(candidate)
        changed = sum(
            a["app_replay"] != b["app_replay"]
            for k in candidate
            for a, b in zip(self.data["base"][k], candidate[k])
            if a["app_replay"]["kind"] in ("DISEASE", "HEALTHY")
        )
        checks = dict(
            old_retention=retain(observed, self.baseline),
            old_field_healthy=all(
                observed["phase5_selection"]["by_source"][s]["healthy_recall"] >= 0.7
                for s in ["tom2024", "swtt5wy2pr"]
            ),
            coarse_retention=all(
                v[1] <= self.coarse_before[k][1] + int(0.005 * v[0])
                for k, v in coarse.items()
            ),
            internal_retention=retain(metrics, self.internal_baseline),
            internal_healthy_each_source=all(
                m["healthy"]["correct_rate"] >= 0.7
                for m in metrics.values()
                if m["healthy"]["n"]
            ),
            original_disease_and_healthy_unchanged=changed == 0,
        )
        recalls = [
            observed["phase5_selection"]["by_source"][s]["healthy_recall"]
            for s in ["tom2024", "swtt5wy2pr"]
        ] + [
            m["healthy"]["correct_rate"] for m in metrics.values() if m["healthy"]["n"]
        ]
        return dict(
            checks=checks,
            passed=all(checks.values()),
            healthy_min=minimum,
            metrics=observed,
            internal_metrics=metrics,
            worst_source_healthy=min(recalls),
            mean_source_healthy=float(np.mean(recalls)),
            false_healthy=sum(m["all"]["false_healthy"] for m in metrics.values()),
            approved_leaf_proposals=int(approved.sum()),
        )


def ranking(value):
    c = value["checks"]
    return (
        value["passed"],
        c["old_retention"] and c["coarse_retention"] and c["internal_retention"],
        value["worst_source_healthy"],
        value["mean_source_healthy"],
        -value["false_healthy"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument(
        "--source-balanced",
        action="store_true",
        help="Isolated publisher-source balancing control; same model, seed and BCE loss",
    )
    a = ap.parse_args()
    p = Path(a.campaign)
    root = p / "phase29_compact"
    out = root / ("head_bce_source_balanced" if a.source_balanced else "head_bce")
    out.mkdir(exist_ok=True)
    complete = read(root / "complete.json")
    assert complete["features_sha256"] == digest(root / "features.npy") and complete[
        "fitting_rows_sha256"
    ] == digest(root / "fitting_rows.json")
    source = read(root / "protocol.json")
    protocol = dict(
        data_protocol_sha256=digest(root / "protocol.json"),
        features_sha256=complete["features_sha256"],
        head="MLP 1408->128->1, weighted BCE",
        negative_cost=4.0,
        sampling=(
            "class/source/family"
            if a.source_balanced
            else "historical class/domain/family"
        ),
        seed=773,
        learning_rate=0.001,
        weight_decay=0.01,
        maximum_epochs=100,
        app_checkpoint_epochs=source["checkpoint_epochs"],
        healthy_thresholds=source["healthy_thresholds"],
        original_classifier_frozen=True,
        reference_disease_and_healthy_priority=True,
        healthy_only_recovery=True,
        mandatory_full_roi_confirmation=True,
        cold_olid_used=False,
        ad_independent=False,
        script_sha256=digest(__file__),
        model_source_sha256=digest(
            Path(__file__).with_name("compact_healthy_model.py")
        ),
    )
    identity = out / "protocol.json"
    if identity.exists():
        assert read(identity) == protocol
    else:
        identity.write_text(json.dumps(protocol, indent=2))
    if (out / "result.json").exists():
        print((out / "result.json").read_text())
        return
    torch.set_num_threads(4)
    torch.manual_seed(773)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    values = torch.from_numpy(np.load(root / "features.npy")).cuda()
    rows = read(root / "fitting_rows.json")
    for r in rows:
        r["species_y"] = r["y"]
    train = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    val = np.array([i for i, r in enumerate(rows) if r["split"] == "val"])
    table = np.array([r["table"] for r in rows])
    target = np.array([r["y"] for r in rows])
    truth = torch.tensor(target, device="cuda", dtype=torch.float32)
    weights = source_class_group_weights if a.source_balanced else balanced
    sampling = torch.tensor(weights(rows, train), dtype=torch.float64)
    val_weights = weights(rows, val)
    mass = {}
    for i, w in zip(train, sampling.numpy()):
        key = rows[i]["source"] + ":" + str(rows[i]["y"])
        mass[key] = mass.get(key, 0.0) + float(w)
    (out / "sampling_mass.json").write_text(json.dumps(mass, indent=2))
    model = LeafFeatureHead("mlp", 1408).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    screen = AppScreen(p, root, "cuda")
    history = []
    ranked = []
    best = None
    passing_checkpoints = 0
    started = time.monotonic()
    for epoch in range(100):
        model.train()
        order = torch.multinomial(
            sampling,
            len(train),
            replacement=True,
            generator=torch.Generator().manual_seed(77300 + epoch),
        )
        total = 0.0
        batches = 0
        for positions in order.split(1024):
            ids = train[positions.numpy()]
            z = model(values[table[ids]])
            y = truth[ids]
            loss = (
                F.binary_cross_entropy_with_logits(z, y, reduction="none")
                * torch.where(y > 0, 1.0, 4.0)
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        if epoch not in source["checkpoint_epochs"]:
            continue
        model.eval()
        with torch.inference_mode():
            logits = model(values).cpu().numpy()
        calibration = calibrate(logits[table[val]], target[val], val_weights)
        probabilities = torch.from_numpy(
            1
            / (1 + np.exp(-np.clip(logits * calibration[0] + calibration[1], -80, 80)))
        ).cuda()
        evaluated = []
        for threshold in source["healthy_thresholds"]:
            candidate = screen.evaluate(probabilities, threshold)
            candidate.update(epoch=epoch, calibration=calibration, protocol=protocol)
            evaluated.append(candidate)
            ranked.append(candidate)
        selected = max(evaluated, key=ranking)
        checkpoint_state = dict(
            kind="mlp",
            model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            calibration=calibration,
            epoch=epoch,
            protocol=protocol,
        )
        atomic_save(checkpoint_state, out / ("epoch_" + str(epoch) + ".pt"))
        np.save(out / ("epoch_" + str(epoch) + "_logits.npy"), logits)
        if best is None or ranking(selected) > ranking(best):
            best = selected
            atomic_save(checkpoint_state, out / "best.pt")
            (out / "candidate.json").write_text(json.dumps(best, indent=2))
        record = dict(
            epoch=epoch,
            loss=total / max(1, batches),
            elapsed_seconds=time.monotonic() - started,
            passed=selected["passed"],
            checks=selected["checks"],
            healthy={
                s: [m["healthy"]["correct"], m["healthy"]["n"]]
                for s, m in selected["internal_metrics"].items()
            },
            policy=selected["healthy_min"],
        )
        history.append(record)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(record), flush=True)
        passing_checkpoints = passing_checkpoints + 1 if selected["passed"] else 0
        if passing_checkpoints >= 2:
            break
    state = torch.load(out / "best.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    with torch.inference_mode():
        actual = model(values).cpu().numpy()
    restore = LeafFeatureHead("mlp", 1408).cuda().eval()
    restore.load_state_dict(state["model"], strict=True)
    with torch.inference_mode():
        error = float(
            (model(values[table[val]]) - restore(values[table[val]])).abs().max()
        )
    assert error == 0.0
    np.save(out / "all_logits.npy", actual)
    result = dict(
        protocol=protocol,
        checkpoint_sha256=digest(out / "best.pt"),
        calibration=state["calibration"],
        selected_epoch=state["epoch"],
        selected=best,
        passed_cheap_checks=best["passed"],
        restored_head_error=error,
        training_seconds=time.monotonic() - started,
        evaluated_policies=len(ranked),
        original_encoders_trained=False,
        new_large_encoder_added=False,
        cold_olid_used=False,
        next_step="Restore full shared model and verify only if cheap checks passed; otherwise diagnose this single targeted head",
    )
    (out / "ranked.json").write_text(json.dumps(ranked, indent=2))
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            dict(
                passed=result["passed_cheap_checks"],
                epoch=state["epoch"],
                training_seconds=result["training_seconds"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
