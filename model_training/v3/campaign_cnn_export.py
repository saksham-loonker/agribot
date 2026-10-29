"""Export the unchanged DINO classifier and a conservatively gated tiny CNN.

Both branches consume the existing canonical 224 NHWC tensor. The specialist
can resolve uncertain Healthy leaders and independently verify knownness; it
cannot replace a named disease winner with Healthy. No Android thresholds or
normalization constants are changed by this exporter.
"""

from pathlib import Path
import argparse, copy, hashlib, json
import numpy as np
from PIL import Image
import timm
import torch
from torch import nn
from app_eval import canon
from train import LABELS
from training_support import load_initial_weights
from campaign_cnn_fusion import ExpertFusion
from campaign_heads import FeatureHead
from campaign_consensus import agreement


class CnnExpertClassifier(nn.Module):
    def __init__(
        self, backbone, cnn, fusion, mean, std, confirmation=None, cnn_calibration=None
    ):
        super().__init__()
        self.backbone = backbone
        self.cnn = cnn
        self.fusion = fusion
        self.confirmation = confirmation
        self.register_buffer(
            "cnn_calibration", torch.tensor(cnn_calibration or [[1.0, 0.0], [1.0, 0.0]])
        )
        self.register_buffer("cnn_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("cnn_std", torch.tensor(std).view(1, 3, 1, 1))
        self.register_buffer(
            "canonical_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "canonical_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, nhwc):
        x = nhwc.permute(0, 3, 1, 2)
        if self.confirmation is not None:
            tokens = self.backbone.forward_features(x)
            features = self.backbone.forward_head(tokens, pre_logits=True)
            reference = self.backbone.head(features)
        else:
            reference = self.backbone(x)
        expert = self.cnn(
            (x * self.canonical_std + self.canonical_mean - self.cnn_mean)
            / self.cnn_std
        )
        if self.confirmation is not None:
            cnn = expert * self.cnn_calibration[:, 0] + self.cnn_calibration[:, 1]
            expert = agreement(self.confirmation(features), cnn)
        return self.fusion(reference, expert)


def make_model(spec):
    backbone = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained=False,
        num_classes=9,
        img_size=224,
    )
    cnn = timm.create_model(spec["cnn_architecture"], pretrained=False, num_classes=2)
    f = spec["fusion"]
    fusion = ExpertFusion(
        f["positive"],
        f["negative"],
        f["known"],
        f["calibration"],
        f["positive_scope"],
        f["protect_healthy"],
        f.get("known_scope", "all"),
        f.get("healthy_min", 0.9),
        f.get("known_min", 0.95),
        f.get("require_known_for_healthy", False),
        f.get("unhealthy_max", 0.01),
        f.get("negative_policy", "redistribute"),
        f.get("require_healthy_for_known", False),
        f.get("disease_max", 1.0),
        f.get("healthy_disease_margin", 0.0),
    )
    confirmation = (
        FeatureHead(backbone.head.state_dict(), spec["confirmation_architecture"])
        if spec.get("confirmation_architecture")
        else None
    )
    return CnnExpertClassifier(
        backbone,
        cnn,
        fusion,
        spec["cnn_mean"],
        spec["cnn_std"],
        confirmation,
        spec.get("cnn_calibration"),
    ).eval()


def restore(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["architecture"] == "CnnExpertClassifier" and state["labels"] == LABELS
    model = make_model(state)
    state["model"].setdefault("cnn_calibration", model.cnn_calibration)
    model.load_state_dict(state["model"], strict=True)
    return model


def main():
    import litert_torch
    from ai_edge_litert.interpreter import Interpreter

    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--encoder",
        default="/home/genesis/sakshyam-agribot/v2/runs/field_vits2/best.pt",
    )
    parser.add_argument("--campaign-root", required=True)
    a = parser.parse_args()
    root = Path(a.campaign_root)
    out = Path(a.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    candidate = json.loads(Path(a.candidate).read_text())
    checkpoint = Path(candidate["cnn_checkpoint"])
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    assert (
        state["policy"] == "center"
    ), "Letterbox experts need a separately validated input contract"
    assert (
        state["rows_sha256"]
        == hashlib.sha256((root / "cnn_inputs/rows.json").read_bytes()).hexdigest()
    )
    torch.set_num_threads(4)
    timm.layers.set_fused_attn(False)
    torch.manual_seed(47)
    fusion = (
        candidate["fusion"]
        if "fusion" in candidate
        else {
            key: candidate[key]
            for key in [
                "positive",
                "negative",
                "known",
                "calibration",
                "positive_scope",
                "protect_healthy",
                "known_scope",
                "healthy_min",
                "known_min",
                "require_known_for_healthy",
                "negative_policy",
                "require_healthy_for_known",
                "disease_max",
                "healthy_disease_margin",
            ]
        }
    )
    spec = dict(
        architecture="CnnExpertClassifier",
        labels=LABELS,
        cnn_architecture=state["architecture"],
        cnn_mean=state["mean"],
        cnn_std=state["std"],
        fusion=fusion,
        input_policy="canonical_center_224",
        encoder_sha256=hashlib.sha256(Path(a.encoder).read_bytes()).hexdigest(),
        cnn_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        training_rows_sha256=state["rows_sha256"],
        candidate_sha256=hashlib.sha256(Path(a.candidate).read_bytes()).hexdigest(),
    )
    confirmation_state = None
    if candidate.get("head_checkpoint"):
        confirmation_state = torch.load(
            candidate["head_checkpoint"], weights_only=True, map_location="cpu"
        )
        spec.update(
            confirmation_architecture=confirmation_state["architecture"],
            confirmation_checkpoint_sha256=hashlib.sha256(
                Path(candidate["head_checkpoint"]).read_bytes()
            ).hexdigest(),
            cnn_calibration=candidate["cnn_calibration"],
        )
    model = make_model(spec)
    load_initial_weights(
        model.backbone, a.encoder, LABELS, "vit_small_patch14_dinov2.lvd142m", 224
    )
    model.cnn.load_state_dict(state["model"], strict=True)
    if confirmation_state is not None:
        model.confirmation.load_state_dict(confirmation_state["model"], strict=True)
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
    cases = []
    for seed in range(5):
        torch.manual_seed(seed)
        cases.append(
            (
                f"random:{seed}",
                (torch.rand(1, 224, 224, 3) - torch.tensor([0.485, 0.456, 0.406]))
                / torch.tensor([0.229, 0.224, 0.225]),
            )
        )
    for file in sorted(
        (Path(__file__).resolve().parents[1] / "v2/bundle/golden").glob("*.png")
    ):
        cases.append(
            (
                file.name,
                torch.from_numpy(
                    canon.classifier_input(
                        np.asarray(Image.open(file).convert("RGB")), 224
                    )
                ),
            )
        )
    # Include real development tensors closest to every decision boundary.
    rows = json.loads((root / "cnn_inputs/rows.json").read_text())
    logits = np.load(checkpoint.parent / "all_logits.npy")
    calibration = np.array(fusion["calibration"])
    cnn = logits * calibration[:, 0] + calibration[:, 1]
    boundary = set()
    development = np.array([i for i, r in enumerate(rows) if r["split"] == "val"])
    for column, target in [
        (0, np.log(9.0)),
        (0, np.log(0.01 / 0.99)),
        (1, np.log(0.95 / 0.05)),
    ]:
        boundary.update(
            development[
                np.argsort(np.abs(cnn[development, column] - target))[:12]
            ].tolist()
        )
    for i in sorted(boundary):
        r = rows[i]
        image = np.asarray(Image.open(r["path"]).convert("RGB"))
        cases.append(
            (r["id"], torch.from_numpy(canon.classifier_input(image, 224, r["box"])))
        )
    parity = []
    with torch.inference_mode():
        for name, tensor in cases:
            expected = model(tensor).numpy()
            runtime.set_tensor(runtime.get_input_details()[0]["index"], tensor.numpy())
            runtime.invoke()
            actual = runtime.get_tensor(runtime.get_output_details()[0]["index"])
            parity.append(
                dict(
                    id=name,
                    max_abs_logit_error=float(np.abs(expected - actual).max()),
                    label_match=int(expected.argmax()) == int(actual.argmax()),
                )
            )
    result = dict(
        spec,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        bytes=path.stat().st_size,
        bundle_sha256=hashlib.sha256(
            (out / "pytorch_bundle.pt").read_bytes()
        ).hexdigest(),
        parity_cases=parity,
        max_abs_logit_error=max(r["max_abs_logit_error"] for r in parity),
        ops=ops,
        passed=all(
            r["label_match"] and r["max_abs_logit_error"] < 0.001 for r in parity
        ),
    )
    (out / "export_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    assert result["passed"], result
    assets = (
        Path(__file__).resolve().parents[2]
        / "agribot_android_app/android/app/src/main/assets"
    )
    manifest = json.loads((assets / "model_manifest.json").read_text())
    manifest["bundle_id"] = "agribot-v3-cnn-" + result["sha256"][:8]
    manifest["status"] = (
        "EXPERIMENTAL: candidate awaiting regression, prospective acceptance, device and licence review"
    )
    manifest["classifier"]["members"][0].update(
        sha256=result["sha256"],
        arch="DINOv2 ViT-S/14 + MobileNetV3 Small guarded experts",
    )
    manifest["training"] = dict(
        spec,
        source_licences=[
            dict(dataset="TLID v2", doi="10.17632/kt64b2kh89.2", licence="CC BY 4.0"),
            dict(
                dataset="Pakistan v2", doi="10.17632/3mbnb82mxd.2", licence="CC BY 4.0"
            ),
            dict(
                dataset="Bangladesh binary v1",
                doi="10.17632/rnbsw72zb5.1",
                licence="CC BY 4.0",
            ),
        ],
    )
    manifest.pop(
        "heldout_results", None
    )  # Historical results do not describe this model.
    (out / "model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    detector = out / "models" / manifest["detector"]["file"]
    if not detector.exists():
        detector.symlink_to(assets / "models" / manifest["detector"]["file"])
    print(
        json.dumps(
            dict(
                sha256=result["sha256"],
                bytes=result["bytes"],
                parity_cases=len(parity),
                max_abs_logit_error=result["max_abs_logit_error"],
                passed=result["passed"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
