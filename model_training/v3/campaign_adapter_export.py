"""Export a shared-prefix original DINO plus an isolated last-block expert."""

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
from campaign_adapter import LastBlockExpert, prefix
from campaign_cnn_fusion import ExpertFusion
from campaign_binary_consensus import binary_agreement


class AdapterClassifier(nn.Module):
    def __init__(
        self,
        backbone,
        expert,
        fusion,
        cnn=None,
        cnn_mean=None,
        cnn_std=None,
        adapter_calibration=None,
        cnn_calibration=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.expert = expert
        self.fusion = fusion
        self.cnn = cnn
        self.register_buffer(
            "adapter_calibration",
            torch.tensor(adapter_calibration or [[1.0, 0.0], [1.0, 0.0]]),
        )
        self.register_buffer(
            "cnn_calibration", torch.tensor(cnn_calibration or [[1.0, 0.0], [1.0, 0.0]])
        )
        self.register_buffer(
            "canonical_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "canonical_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "cnn_mean", torch.tensor(cnn_mean or [0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "cnn_std", torch.tensor(cnn_std or [0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, nhwc):
        image = nhwc.permute(0, 3, 1, 2)
        tokens = prefix(self.backbone, image)
        original = self.backbone.head(
            self.backbone.norm(self.backbone.blocks[-1](tokens))[:, 0]
        )
        expert = self.expert(tokens)
        if self.cnn is not None:
            expert = (
                expert * self.adapter_calibration[:, 0] + self.adapter_calibration[:, 1]
            )
            cnn = self.cnn(
                (image * self.canonical_std + self.canonical_mean - self.cnn_mean)
                / self.cnn_std
            )
            cnn = cnn * self.cnn_calibration[:, 0] + self.cnn_calibration[:, 1]
            expert = binary_agreement(expert, cnn)
        return self.fusion(original, expert)


def make_model(spec):
    backbone = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained=False,
        num_classes=9,
        img_size=224,
    )
    expert = LastBlockExpert(backbone, spec["mode"], spec["rank"])
    cnn = (
        timm.create_model(spec["cnn_architecture"], pretrained=False, num_classes=2)
        if spec.get("cnn_architecture")
        else None
    )
    return AdapterClassifier(
        backbone,
        expert,
        ExpertFusion(**spec["fusion"]),
        cnn,
        spec.get("cnn_mean"),
        spec.get("cnn_std"),
        spec.get("adapter_calibration"),
        spec.get("cnn_calibration"),
    ).eval()


def restore(path):
    state = torch.load(path, weights_only=True, map_location="cpu")
    assert state["architecture"] == "AdapterClassifier" and state["labels"] == LABELS
    model = make_model(state)
    model.load_state_dict(state["model"], strict=True)
    return model


def main():
    import litert_torch
    from ai_edge_litert.interpreter import Interpreter

    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--encoder",
        default="/home/genesis/sakshyam-agribot/v2/runs/field_vits2/best.pt",
    )
    a = p.parse_args()
    root = Path(a.root)
    out = Path(a.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    candidate = json.loads(Path(a.candidate).read_text())
    joint = "adapter_checkpoint" in candidate
    checkpoint = Path(
        candidate["adapter_checkpoint"] if joint else candidate["cnn_checkpoint"]
    )
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    assert (
        state["architecture"] == "LastBlockExpert"
        and state["rows_sha256"]
        == hashlib.sha256((root / "cnn_inputs/rows.json").read_bytes()).hexdigest()
    )
    torch.set_num_threads(4)
    timm.layers.set_fused_attn(False)
    torch.manual_seed(47)
    fusion = (
        candidate["fusion"]
        if joint
        else {
            k: candidate[k]
            for k in [
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
            ]
        }
    )
    spec = dict(
        schema=1,
        architecture="AdapterClassifier",
        labels=LABELS,
        mode=state["mode"],
        rank=state["rank"],
        loss=state["loss"],
        fusion=fusion,
        encoder_sha256=hashlib.sha256(Path(a.encoder).read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        training_rows_sha256=state["rows_sha256"],
        candidate_sha256=hashlib.sha256(Path(a.candidate).read_bytes()).hexdigest(),
    )
    assert state["encoder_sha256"] == spec["encoder_sha256"]
    cnn_state = None
    if joint:
        cnn_path = Path(candidate["cnn_checkpoint"])
        cnn_state = torch.load(cnn_path, weights_only=True, map_location="cpu")
        assert (
            cnn_state["rows_sha256"] == state["rows_sha256"]
            and cnn_state["policy"] == "center"
        )
        spec.update(
            cnn_architecture=cnn_state["architecture"],
            cnn_mean=cnn_state["mean"],
            cnn_std=cnn_state["std"],
            cnn_checkpoint_sha256=hashlib.sha256(cnn_path.read_bytes()).hexdigest(),
            adapter_calibration=candidate["adapter_calibration"],
            cnn_calibration=candidate["cnn_calibration"],
        )
    model = make_model(spec)
    load_initial_weights(
        model.backbone, a.encoder, LABELS, "vit_small_patch14_dinov2.lvd142m", 224
    )
    model.expert.load_state_dict(state["model"], strict=True)
    torch.testing.assert_close(
        model.expert.original_head.weight, model.backbone.head.weight, rtol=0, atol=0
    )
    torch.testing.assert_close(
        model.expert.original_head.bias, model.backbone.head.bias, rtol=0, atol=0
    )
    if cnn_state is not None:
        model.cnn.load_state_dict(cnn_state["model"], strict=True)
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
    rows = json.loads((root / "cnn_inputs/rows.json").read_text())
    raw = np.load(checkpoint.parent / "all_logits.npy")
    cal = np.array(
        candidate["adapter_calibration"] if joint else candidate["calibration"]
    )
    z = raw * cal[:, 0] + cal[:, 1]
    if joint:
        cnn_raw = np.load(Path(candidate["cnn_checkpoint"]).parent / "all_logits.npy")
        ccal = np.array(candidate["cnn_calibration"])
        z = binary_agreement(
            torch.from_numpy(z), torch.from_numpy(cnn_raw * ccal[:, 0] + ccal[:, 1])
        ).numpy()
    val = np.array([i for i, r in enumerate(rows) if r["split"] == "val"])
    boundary = set()
    for column, target in [
        (0, np.log(fusion["healthy_min"] / (1 - fusion["healthy_min"]))),
        (0, np.log(0.01 / 0.99)),
        (1, np.log(fusion["known_min"] / (1 - fusion["known_min"]))),
    ]:
        boundary.update(val[np.argsort(np.abs(z[val, column] - target))[:12]].tolist())
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
    manifest["bundle_id"] = "agribot-v3-" + state["mode"] + "-" + result["sha256"][:8]
    manifest["status"] = (
        "EXPERIMENTAL: awaiting regression, untouched-source acceptance and device verification"
    )
    manifest["classifier"]["members"][0].update(
        sha256=result["sha256"],
        arch="DINOv2 ViT-S/14 shared original prefix + isolated "
        + state["mode"]
        + " binary expert",
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
                max_abs_logit_error=result["max_abs_logit_error"],
                passed=result["passed"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
