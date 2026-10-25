"""Export an experimental factorized classifier with one original DINO backbone."""

from pathlib import Path
import argparse, copy, hashlib, json, sys
import numpy as np
from PIL import Image
import timm
import torch
from torch import nn
from train import LABELS
from training_support import load_initial_weights
from campaign_factorized import FactorizedHead, T
from app_eval import canon


class FactorizedClassifier(nn.Module):
    def __init__(self, backbone, head, feature_kind):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.feature_kind = feature_kind

    def forward(self, nhwc):
        tokens = self.backbone.forward_features(nhwc.permute(0, 3, 1, 2))
        features = self.backbone.forward_head(tokens, pre_logits=True)
        if self.feature_kind == "cls_mean_max_std":
            patches = tokens[:, self.backbone.num_prefix_tokens :]
            mean = patches.mean(1)
            std = (patches - mean[:, None, :]).square().mean(1).clamp_min(1e-12).sqrt()
            features = torch.cat([features, mean, patches.amax(1), std], 1)
        return self.head(features)


def restore(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["architecture"] == "FactorizedClassifier" and state["labels"] == LABELS
    backbone = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained=False,
        num_classes=9,
        img_size=224,
    ).eval()
    head = FactorizedHead(
        backbone.head.state_dict(),
        state["mode"],
        state["head_architecture"],
        state["feature_dim"],
    )
    model = FactorizedClassifier(backbone, head, state["feature_kind"]).eval()
    model.load_state_dict(state["model"], strict=True)
    return model


def main():
    import litert_torch
    from ai_edge_litert.interpreter import Interpreter

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--encoder",
        default="/home/genesis/sakshyam-agribot/v2/runs/field_vits2/best.pt",
    )
    p.add_argument("--feature-cache")
    a = p.parse_args()
    out = Path(a.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(47)
    timm.layers.set_fused_attn(False)
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    assert state["labels"] == LABELS
    dim = state.get("feature_dim", state["model"]["pair.0.weight"].shape[1])
    kind = state.get("feature_kind", "cls_mean_max_std" if dim == 1536 else "cls")
    backbone = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained=False,
        num_classes=9,
        img_size=224,
    ).eval()
    load_initial_weights(
        backbone, a.encoder, LABELS, "vit_small_patch14_dinov2.lvd142m", 224
    )
    assert torch.equal(
        backbone.head.weight, state["model"]["base.weight"]
    ) and torch.equal(backbone.head.bias, state["model"]["base.bias"])
    head = FactorizedHead(
        backbone.head.state_dict(), state["mode"], state["architecture"], dim
    )
    head.load_state_dict(state["model"], strict=True)
    model = FactorizedClassifier(backbone, head, kind).eval()
    bundle = dict(
        architecture="FactorizedClassifier",
        model=model.state_dict(),
        labels=LABELS,
        mode=state["mode"],
        head_architecture=state["architecture"],
        feature_dim=dim,
        feature_kind=kind,
        reference_other_bias=0.75,
        reference_temperature=T,
        checkpoint=a.checkpoint,
        encoder_sha256=hashlib.sha256(Path(a.encoder).read_bytes()).hexdigest(),
    )
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
    golden = Path(__file__).resolve().parents[1] / "v2/bundle/golden"
    for file in sorted(golden.glob("*.png")):
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
    if a.feature_cache:
        cache_path = Path(a.feature_cache)
        cache = torch.load(cache_path / "features.pt", weights_only=True)
        jobs = json.loads((cache_path / "feature_rows.json").read_text())
        assert cache["features"].shape[1] == dim
        with torch.no_grad():
            z, lp, pair, known = model.head.raw(cache["features"])
        boundary = set(
            (known.sigmoid() - 0.9).abs().topk(12, largest=False).indices.tolist()
        )
        boundary.update(
            (lp[:, 3].exp() - 0.7).abs().topk(12, largest=False).indices.tolist()
        )
        for i in sorted(boundary):
            r = jobs[i]
            image = np.asarray(Image.open(r["path"]).convert("RGB"))
            cases.append(
                (
                    r["id"],
                    torch.from_numpy(canon.classifier_input(image, 224, r["box"])),
                )
            )
    rows = []
    with torch.inference_mode():
        for name, tensor in cases:
            expected = model(tensor).numpy()
            runtime.set_tensor(runtime.get_input_details()[0]["index"], tensor.numpy())
            runtime.invoke()
            actual = runtime.get_tensor(runtime.get_output_details()[0]["index"])
            row = dict(
                id=name,
                max_abs_logit_error=float(np.abs(expected - actual).max()),
                label_match=int(expected.argmax()) == int(actual.argmax()),
            )
            rows.append(row)
    result = dict(
        architecture="FactorizedClassifier",
        head_architecture=state["architecture"],
        mode=state["mode"],
        feature_kind=kind,
        checkpoint=a.checkpoint,
        checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
        encoder_sha256=bundle["encoder_sha256"],
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        bytes=path.stat().st_size,
        bundle_sha256=hashlib.sha256(
            (out / "pytorch_bundle.pt").read_bytes()
        ).hexdigest(),
        parity_cases=rows,
        max_abs_logit_error=max(r["max_abs_logit_error"] for r in rows),
        ops=ops,
        passed=all(r["label_match"] and r["max_abs_logit_error"] < 0.001 for r in rows),
    )
    (out / "export_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    assert result["passed"]
    assets = (
        Path(__file__).resolve().parents[2]
        / "agribot_android_app/android/app/src/main/assets"
    )
    manifest = json.loads((assets / "model_manifest.json").read_text())
    manifest["bundle_id"] = "agribot-v3-factorized-" + result["sha256"][:8]
    manifest["status"] = (
        "EXPERIMENTAL: exported candidate; regression and prospective acceptance must pass before promotion"
    )
    manifest["classifier"]["members"][0]["sha256"] = result["sha256"]
    manifest["training"] = dict(
        architecture="FactorizedClassifier",
        mode=state["mode"],
        feature_kind=kind,
        source_checkpoint=a.checkpoint,
        source_encoder_sha256=bundle["encoder_sha256"],
        reference_temperature=T,
        reference_other_bias=0.75,
    )
    (out / "model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    detector = out / "models" / manifest["detector"]["file"]
    if not detector.exists():
        detector.symlink_to(assets / "models" / manifest["detector"]["file"])


if __name__ == "__main__":
    main()
