"""Export one PyTorch checkpoint to the Android TFLite contract and prove parity."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import litert_torch
from ai_edge_litert.interpreter import Interpreter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import LABELS  # noqa: E402

timm.layers.set_fused_attn(False)


class NHWC(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x.permute(0, 3, 1, 2))


def load_checkpoint(path: Path, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=False)
    args = state["args"]
    model = timm.create_model(args["model"], pretrained=False, num_classes=len(LABELS),
                              img_size=args["img"]).eval()
    weights = state["model"]
    weights = {k.replace("_orig_mod.", ""): v for k, v in weights.items()}
    model.load_state_dict(weights)
    model.to(device)
    return model, state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--assets", default="agribot_android_app/android/app/src/main/assets")
    parser.add_argument("--real-inputs", default=str(Path.home() / "agribot-data/work/test_cache_256.npy"))
    parser.add_argument("--n-real", type=int, default=16)
    parser.add_argument("--max-logit-error", type=float, default=1e-3)
    args = parser.parse_args()
    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    model, state = load_checkpoint(Path(args.checkpoint))
    img = int(state["args"]["img"])
    wrapper = NHWC(model).eval()
    example = torch.rand(1, img, img, 3)
    edge = litert_torch.convert(wrapper, (example,))
    target = out / "models" / "disease_classifier_0.tflite"
    edge.export(str(target))
    size = target.stat().st_size
    digest = hashlib.sha256(target.read_bytes()).hexdigest()

    interpreter = Interpreter(model_path=str(target), num_threads=2)
    interpreter.allocate_tensors()
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    assert in_detail["shape"].tolist() == [1, img, img, 3], in_detail["shape"]
    assert out_detail["shape"].tolist() == [1, len(LABELS)], out_detail["shape"]

    def tflite_logits(x):
        interpreter.set_tensor(in_detail["index"], x.astype(np.float32))
        interpreter.invoke()
        return interpreter.get_tensor(out_detail["index"])

    cache = np.load(args.real_inputs, mmap_mode="r")
    rng = np.random.default_rng(20261015)
    indices = rng.choice(len(cache), size=min(args.n_real, len(cache)), replace=False)
    from PIL import Image
    sys.path.insert(0, "model_training/v2/final")
    import canon
    cases = [("random_uniform", np.random.default_rng(7).random((1, img, img, 3), dtype=np.float32))]
    for index in indices:
        cases.append((f"cache_{int(index)}", canon.classifier_input(np.asarray(cache[int(index)]), img)))
    rows = []
    for name, x in cases:
        with torch.no_grad():
            torch_out = wrapper(torch.from_numpy(x)).numpy()
        edge_out = tflite_logits(x)
        max_abs = float(np.abs(torch_out - edge_out).max())
        rows.append(dict(case=name, max_abs_logit_error=max_abs,
                         torch_label=LABELS[int(torch_out.argmax())],
                         tflite_label=LABELS[int(edge_out.argmax())]))
        print(name, max_abs, rows[-1]["torch_label"], rows[-1]["tflite_label"], flush=True)
    worst = max(row["max_abs_logit_error"] for row in rows)
    label_changes = [row for row in rows if row["torch_label"] != row["tflite_label"]]
    parity = dict(rows=rows, worst_max_abs_logit_error=worst, label_changes=label_changes,
                  threshold=args.max_logit_error, passed=worst <= args.max_logit_error and not label_changes,
                  input_shape=[1, img, img, 3], output_shape=[1, len(LABELS)],
                  sha256=digest, size_bytes=size,
                  checkpoint=args.checkpoint, checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                  source_run=state.get("args", {}).get("run_name"), trained_epoch=state.get("epoch"),
                  converter=f"litert-torch {getattr(litert_torch, '__version__', 'unknown')}",
                  torch=torch.__version__, timm=timm.__version__)
    (out / "parity.json").write_text(json.dumps(parity, indent=1))
    assert parity["passed"], f"TFLite parity failed: {parity}"


if __name__ == "__main__":
    main()
