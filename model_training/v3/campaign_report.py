"""Build an auditable campaign inventory and paired final model comparison."""

from pathlib import Path
import argparse, collections, hashlib, importlib.metadata, json, re
import numpy as np
from campaign_verify import regression_retained


def read(path):
    return json.loads(path.read_text())


def predictions(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def binary_false_healthy(metrics):
    coarse = metrics.get("coarse_diseased", {})
    return (
        coarse["false_healthy"]
        if coarse.get("n", 0) > 0
        else metrics["supported_disease"]["false_healthy"]
    )


def paired_healthy(original, candidate):
    lookup = {r["id"]: r for r in original}
    groups = collections.defaultdict(list)
    gained = []
    lost = []
    for r in candidate:
        if r["label"] != "Healthy":
            continue
        b = lookup[r["id"]]
        before = b["app_replay"]["kind"] == "HEALTHY"
        after = r["app_replay"]["kind"] == "HEALTHY"
        group = r["group"]
        if r["source"] == "bangladesh_binary":
            stem = Path(r["path"]).stem
            group = (
                "binary:"
                + stem[0]
                + ":"
                + str(int(re.search(r"\d+", stem).group()) // 16)
            )
        groups[group].append((int(before), int(after)))
        if after and not before:
            gained.append(r["id"])
        if before and not after:
            lost.append(r["id"])
    counts = np.array(
        [
            [sum(x[0] for x in rows), sum(x[1] for x in rows), len(rows)]
            for rows in groups.values()
        ]
    )
    rng = np.random.default_rng(29)
    values = []
    for _ in range(2000):
        sampled = counts[rng.integers(0, len(counts), len(counts))].sum(0)
        values.append((sampled[1] - sampled[0]) / sampled[2])
    total = counts.sum(0)
    return dict(
        photos=int(total[2]),
        inferred_groups=len(counts),
        baseline_correct=int(total[0]),
        candidate_correct=int(total[1]),
        recall_delta=float((total[1] - total[0]) / total[2]),
        paired_group_bootstrap_95ci=np.quantile(values, [0.025, 0.975]).tolist(),
        gains=gained,
        losses=lost,
        limitation="Inferred photo/capture groups are not verified plant IDs; intervals do not establish live camera accuracy.",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--export-name", default="export_cnn_final")
    p.add_argument("--evaluation-name", default="final_evaluation")
    p.add_argument("--candidate-name", default="candidate_cnn_final.json")
    p.add_argument("--protocol-name", default="final_candidate_protocol.json")
    p.add_argument(
        "--binary-opened",
        action="store_true",
        help="Mark previously opened binary cohort as regression evidence",
    )
    a = p.parse_args()
    root = Path(a.root)
    export = root / a.export_name
    evaluation = root / a.evaluation_name
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    verification = read(export / "regressions/regression_verification.json")
    original_new = read(
        root / "acceptance_cnn_center_focal/original_new_sources/metrics.json"
    )
    final_new = read(evaluation / "new_sources/metrics.json")
    original_binary = read(
        root / "acceptance_cnn_center_focal/original_binary/metrics.json"
    )
    final_binary = read(evaluation / "binary/metrics.json")
    original_binary_rows = predictions(
        root / "acceptance_cnn_center_focal/original_binary/predictions.jsonl"
    )
    final_binary_rows = predictions(evaluation / "binary/predictions.jsonl")
    binary_gate = (
        final_binary["errors"] == original_binary["errors"] == 0
        and final_binary["healthy"]["correct"] >= original_binary["healthy"]["correct"]
        and binary_false_healthy(final_binary) <= binary_false_healthy(original_binary)
    )
    new_gate = final_new["errors"] == original_new[
        "errors"
    ] == 0 and regression_retained({"new": final_new}, {"new": original_new})
    inventory = []
    for phase in [
        "heads",
        "aligned/heads",
        "factorized",
        "aligned/factorized",
        "dense_aligned/factorized",
    ]:
        for file in sorted((root / phase).glob("*/result.json")):
            r = read(file)
            inventory.append(
                dict(
                    phase=phase,
                    name=file.parent.name,
                    method=r.get("method"),
                    architecture=r.get("architecture"),
                    validation=r.get("validation"),
                    validation_retention_passed=r.get("validation_retention_passed"),
                    checkpoint_sha256=hashlib.sha256(
                        (file.parent / "best.pt").read_bytes()
                    ).hexdigest(),
                )
            )
    cnn = [read(f) for f in sorted(root.glob("cnn_inputs/runs/*/result.json"))]
    data = read(root / "data_quality_summary.json")
    data["limitations"] = [
        x for x in data["limitations"] if "Binary external source" not in x
    ]
    data["limitations"].append(
        "Binary author training was prospectively admitted before any predictions; clean 139 author validation photos stayed unopened until final policy freeze. See cnn_protocol."
    )
    result = dict(
        schema=1,
        model=read(export / "export_verification.json"),
        final_policy=read(root / a.candidate_name),
        final_protocol=read(root / a.protocol_name),
        data_quality=data,
        cnn_protocol=read(root / "cnn_inputs/protocol.json"),
        experiment_counts=dict(
            collections.Counter(r["phase"] for r in inventory),
            cnn=len(cnn),
            boosted_tree_models=len(
                list((root / "dense_aligned/classical").glob("*.ubj"))
            ),
            backbone_partial_finetune=1,
            isolated_adapters=len(list((root / "adapters").glob("*/result.json"))),
        ),
        experiments=inventory,
        cnn_trials=cnn,
        opened_regressions=verification,
        new_source_regression=dict(
            baseline=original_new,
            candidate=final_new,
            passed=new_gate,
            evidence_status="First candidate opened this cohort and failed unsupported rejection by one photo. Final policy refinement followed; this is regression, not untouched acceptance.",
            paired_healthy=paired_healthy(
                predictions(
                    root
                    / "acceptance_cnn_center_focal/original_new_sources/predictions.jsonl"
                ),
                predictions(evaluation / "new_sources/predictions.jsonl"),
            ),
        ),
        binary_evaluation=dict(
            baseline=original_binary,
            candidate=final_binary,
            passed=binary_gate,
            independent=not a.binary_opened,
            evidence_status=(
                "Previously opened author-validation photos; regression only."
                if a.binary_opened
                else "Final candidate frozen before author-validation metrics were first read."
            )
            + " Binary labels cannot establish named disease accuracy.",
            paired_healthy=paired_healthy(original_binary_rows, final_binary_rows),
        ),
        environment={
            n: importlib.metadata.version(n)
            for n in [
                "torch",
                "timm",
                "numpy",
                "Pillow",
                "scipy",
                "scikit-learn",
                "litert-torch",
                "ai-edge-litert",
                "xgboost",
            ]
        },
        gates_passed=verification["passed"] and new_gate and binary_gate,
        outstanding=[
            "Phone export/preprocessing parity and latency",
            "Android rebuild and CI",
            "Source licence attribution review before public release",
        ],
    )
    if (evaluation / "fresh/metrics.json").exists():
        before = read(evaluation / "original_fresh/metrics.json")
        after = read(evaluation / "fresh/metrics.json")
        fresh_gate = before["errors"] == after["errors"] == 0 and regression_retained(
            {"fresh": after}, {"fresh": before}
        )
        result["independent_fresh_classification"] = dict(
            baseline=before,
            candidate=after,
            passed=fresh_gate,
            audit=read(root / "fresh_acceptance_audit.json"),
            paired_healthy=paired_healthy(
                predictions(evaluation / "original_fresh/predictions.jsonl"),
                predictions(evaluation / "fresh/predictions.jsonl"),
            ),
            evidence_status="Evaluation-only source membership and final model frozen before predictions. No detected campaign overlap; original historical source independence is not guaranteed.",
        )
        result["gates_passed"] = result["gates_passed"] and fresh_gate
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                gates_passed=result["gates_passed"],
                old=verification["passed"],
                new=new_gate,
                binary=binary_gate,
                new_healthy=[
                    original_new["healthy"]["correct"],
                    final_new["healthy"]["correct"],
                ],
                new_disease_false_healthy=[
                    original_new["supported_disease"]["false_healthy"],
                    final_new["supported_disease"]["false_healthy"],
                ],
                new_unsupported_false_known=[
                    original_new["unsupported_tomato"]["false_known"],
                    final_new["unsupported_tomato"]["false_known"],
                ],
                binary_healthy=[
                    original_binary["healthy"]["correct"],
                    final_binary["healthy"]["correct"],
                ],
                binary_disease_false_healthy=[
                    binary_false_healthy(original_binary),
                    binary_false_healthy(final_binary),
                ],
                experiment_counts=result["experiment_counts"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
