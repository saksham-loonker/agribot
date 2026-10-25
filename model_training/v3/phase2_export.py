"""Export the frozen reference, copied multiclass tail and CNN as one LiteRT graph."""

from pathlib import Path
import argparse
import copy
import hashlib
import json

import numpy as np
import timm
import torch
from torch import nn
from PIL import Image
from app_eval import canon

from campaign_adapter import prefix
from campaign_pixels import open_pixels
from phase2_multiclass import MulticlassExpert
from phase2_policy import DomainPolicy
from campaign_cnn_export import make_model as make_safe_model
from training_support import load_initial_weights
from train import LABELS
from campaign_reference import safe_logits


class DomainClassifier(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.depth = spec["depth"]
        self.reference_logits = spec.get("reference_logits", False)
        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=False,
            num_classes=9,
            img_size=224,
        )
        self.expert = MulticlassExpert(
            self.backbone, spec["mode"], spec["rank"], self.depth
        )
        self.cnn = timm.create_model(
            spec["cnn_architecture"], pretrained=False, num_classes=2
        )
        self.policy = DomainPolicy(**spec["policy"])
        self.safe = (
            make_safe_model(spec["safe_baseline"])
            if spec.get("safe_baseline")
            else None
        )
        if self.safe is not None:
            # The reference backbone is shared, not exported a second time.
            self.safe.backbone = self.backbone
        for name, values in [
            ("canonical_mean", [0.485, 0.456, 0.406]),
            ("canonical_std", [0.229, 0.224, 0.225]),
            ("cnn_mean", spec["cnn_mean"]),
            ("cnn_std", spec["cnn_std"]),
        ]:
            self.register_buffer(name, torch.tensor(values).view(1, 3, 1, 1))

    def components(self, nhwc, return_features=False):
        image = nhwc.permute(0, 3, 1, 2)
        tokens = prefix(self.backbone, image, self.depth)
        original = tokens
        for block in self.backbone.blocks[-self.depth :]:
            original = block(original)
        original_features = self.backbone.norm(original)[:, 0]
        reference = self.backbone.head(original_features)
        if self.safe is not None:
            safe_cnn = self.safe.cnn(
                (image * self.canonical_std + self.canonical_mean - self.safe.cnn_mean)
                / self.safe.cnn_std
            )
            reference = self.safe.fusion(reference, safe_cnn)
        cnn = self.cnn(
            (image * self.canonical_std + self.canonical_mean - self.cnn_mean)
            / self.cnn_std
        )
        adapted_features = self.expert.features(tokens)
        parts = (reference, self.expert.head(adapted_features), cnn)
        return (
            (*parts, original_features, adapted_features) if return_features else parts
        )

    def forward(self, nhwc):
        reference, student, cnn = self.components(nhwc)
        candidate = self.policy(reference, student, cnn)
        return (
            torch.cat([candidate, reference], 1) if self.reference_logits else candidate
        )


def restore(path):
    state = torch.load(path, weights_only=True, map_location="cpu")
    assert state["architecture"] == "DomainClassifier" and state["labels"] == LABELS
    model = DomainClassifier(state).eval()
    model.load_state_dict(state["model"], strict=True)
    return model


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import litert_torch
    from ai_edge_litert.interpreter import Interpreter

    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reference-logits", action="store_true")
    p.add_argument("--confirm-unknown", action="store_true")
    p.add_argument(
        "--encoder",
        default="/home/genesis/sakshyam-agribot/v2/runs/field_vits2/best.pt",
    )
    a = p.parse_args()
    assert not a.confirm_unknown or a.reference_logits
    root = Path(a.root)
    out = Path(a.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    candidate = json.loads(Path(a.candidate).read_text())
    student = torch.load(
        candidate["student_checkpoint"], weights_only=True, map_location="cpu"
    )
    cnn = torch.load(candidate["cnn_checkpoint"], weights_only=True, map_location="cpu")
    assert (
        student["rows_sha256"]
        == cnn["rows_sha256"]
        == digest(root / "cnn_inputs/rows.json")
    )
    assert student["encoder_sha256"] == digest(a.encoder) and cnn["policy"] == "center"
    torch.set_num_threads(4)
    torch.manual_seed(67)
    timm.layers.set_fused_attn(False)
    spec = dict(
        schema=1,
        architecture="DomainClassifier",
        labels=LABELS,
        mode=student["mode"],
        depth=student.get("depth", 1),
        rank=student["rank"],
        cnn_architecture=cnn["architecture"],
        cnn_mean=cnn["mean"],
        cnn_std=cnn["std"],
        policy=candidate["policy"],
        encoder_sha256=digest(a.encoder),
        student_sha256=digest(candidate["student_checkpoint"]),
        cnn_sha256=digest(candidate["cnn_checkpoint"]),
        training_rows_sha256=student["rows_sha256"],
        candidate_sha256=digest(a.candidate),
    )
    safe_state = torch.load(
        root.parent / "export_cnn_safe/pytorch_bundle.pt",
        weights_only=True,
        map_location="cpu",
    )
    spec["safe_baseline"] = {k: v for k, v in safe_state.items() if k != "model"}
    spec["safe_baseline_sha256"] = digest(
        root.parent / "export_cnn_safe/pytorch_bundle.pt"
    )
    spec["reference_logits"] = a.reference_logits
    model = DomainClassifier(spec).eval()
    load_initial_weights(
        model.backbone, a.encoder, LABELS, "vit_small_patch14_dinov2.lvd142m", 224
    )
    model.expert.load_state_dict(student["model"], strict=True)
    model.cnn.load_state_dict(cnn["model"], strict=True)
    model.safe.load_state_dict(safe_state["model"], strict=True)
    bundle = copy.deepcopy(spec)
    bundle["model"] = model.state_dict()
    torch.save(bundle, out / "pytorch_bundle.pt")
    x = (
        torch.rand(1, 224, 224, 3) - torch.tensor([0.485, 0.456, 0.406])
    ) / torch.tensor([0.229, 0.224, 0.225])
    with torch.no_grad():
        torch.testing.assert_close(
            restore(out / "pytorch_bundle.pt")(x), model(x), rtol=0, atol=0
        )
    path = out / "models/disease_classifier_0.tflite"
    litert_torch.convert(model, (x,)).export(str(path))
    runtime = Interpreter(model_path=str(path), num_threads=4)
    runtime.allocate_tensors()
    ops = sorted({o["op_name"] for o in runtime._get_ops_details()})
    assert not any(
        o == "CUSTOM" or o.startswith("Flex") or "STABLEHLO" in o for o in ops
    ), ops
    rows = json.loads((root / "cnn_inputs/rows.json").read_text())
    pixels = open_pixels(root / "cnn_inputs")
    reference = safe_logits(root)
    sz = np.load(Path(candidate["student_checkpoint"]).parent / "best_logits.npy")
    cz = np.load(Path(candidate["cnn_checkpoint"]).parent / "all_logits.npy")
    with torch.no_grad():
        expected_all = model.policy(
            torch.from_numpy(reference), torch.from_numpy(sz), torch.from_numpy(cz)
        ).numpy()
    val = np.array([i for i, r in enumerate(rows) if r["split"] == "val"])
    changed = val[np.max(np.abs(expected_all[val] - reference[val]), axis=1) > 0.001]
    if a.reference_logits:
        expected_all = np.concatenate([expected_all, reference], 1)
    # Exercise actual switched cases, unchanged cases, and probability boundaries.
    chosen = (
        set(
            changed[
                np.linspace(0, len(changed) - 1, min(30, len(changed))).astype(int)
            ].tolist()
        )
        if len(changed)
        else set()
    )
    probs = torch.softmax(
        (torch.from_numpy(sz) + torch.tensor([0.0] * 8 + [0.75]))
        / (0.8814700841903687 * candidate["policy"]["student_temperature"]),
        1,
    ).numpy()
    chosen.update(
        val[
            np.argsort(np.abs(probs[val].max(1) - candidate["policy"]["student_min"]))[
                :12
            ]
        ].tolist()
    )
    chosen.update(val[np.linspace(0, len(val) - 1, 12).astype(int)].tolist())
    parity = []
    with torch.inference_mode():
        for i in sorted(chosen):
            tensor = np.ascontiguousarray(pixels[i : i + 1])
            expected = model(torch.from_numpy(tensor)).numpy()
            runtime.set_tensor(runtime.get_input_details()[0]["index"], tensor)
            runtime.invoke()
            actual = runtime.get_tensor(runtime.get_output_details()[0]["index"])
            parity.append(
                dict(
                    id=rows[i]["id"],
                    max_abs_logit_error=float(np.max(np.abs(expected - actual))),
                    cached_error=float(np.max(np.abs(expected[0] - expected_all[i]))),
                    label_match=int(expected.argmax()) == int(actual.argmax()),
                )
            )
        for file in sorted(
            (Path(__file__).resolve().parents[1] / "v2/bundle/golden").glob("*.png")
        ):
            tensor = canon.classifier_input(
                np.asarray(Image.open(file).convert("RGB")), 224
            )
            expected = model(torch.from_numpy(tensor)).numpy()
            runtime.set_tensor(runtime.get_input_details()[0]["index"], tensor)
            runtime.invoke()
            actual = runtime.get_tensor(runtime.get_output_details()[0]["index"])
            parity.append(
                dict(
                    id=file.name,
                    max_abs_logit_error=float(np.max(np.abs(expected - actual))),
                    cached_error=0.0,
                    label_match=int(expected.argmax()) == int(actual.argmax()),
                )
            )
    result = dict(
        spec,
        sha256=digest(path),
        bytes=path.stat().st_size,
        ops=ops,
        parity_cases=parity,
        max_abs_logit_error=max(r["max_abs_logit_error"] for r in parity),
        passed=all(
            r["label_match"]
            and r["max_abs_logit_error"] < 0.001
            and r["cached_error"] < 0.002
            for r in parity
        ),
    )
    (out / "export_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    assert result["passed"], result
    assets = Path(
        json.loads((root / "complete.json").read_text())["original_reference_assets"]
    )
    manifest = json.loads((assets / "model_manifest.json").read_text())
    manifest["bundle_id"] = "agribot-v3-domain-" + result["sha256"][:8]
    manifest["status"] = (
        "EXPERIMENTAL: awaiting actual-byte regression and independent acceptance"
    )
    manifest["classifier"]["members"][0].update(
        sha256=result["sha256"],
        arch="DINOv2 shared prefix + adapted multiclass tail + corroborating MobileNetV3",
    )
    if a.reference_logits:
        manifest["schema"] = 3
        manifest["classifier"]["members"][0]["output"] = [1, 18]
        manifest["classifier"]["view_confirmation"] = dict(
            policy=(
                "changed_decision_full_roi"
                if a.confirm_unknown
                else "changed_positive_full_roi"
            ),
            reference_logit_offset=9,
            padding_value=114,
        )
    manifest["training"] = spec
    manifest.pop("heldout_results", None)
    (out / "model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    detector = out / "models" / manifest["detector"]["file"]
    if not detector.exists():
        detector.symlink_to(assets / "models" / manifest["detector"]["file"])
    print(
        json.dumps(
            dict(
                sha256=result["sha256"],
                bytes=result["bytes"],
                cases=len(parity),
                passed=result["passed"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
