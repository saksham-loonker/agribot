"""PyTorch -> TFLite via litert-torch for every ensemble member (NHWC float32 in, logits out); parity vs PyTorch."""
import json
from pathlib import Path
import numpy as np, torch, timm
import litert_torch
timm.layers.set_fused_attn(False)
F_ = Path(__file__).resolve().parent; cal = json.load(open(F_ / "calibration.json"))
class NHWC(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): return s.m(x.permute(0, 3, 1, 2))
for k, mem in enumerate(cal["members"]):
    kw = dict(img_size=mem["img"]) if "vit" in mem["model"] else {}
    m = timm.create_model(mem["model"], pretrained=False, num_classes=cal["num_classes"], **kw).eval()
    m.load_state_dict(torch.load(F_.parent / "runs" / mem["run"] / "best.pt", map_location="cpu")); w = NHWC(m).eval()
    x = torch.rand(1, mem["img"], mem["img"], 3); edge = litert_torch.convert(w, (x,)); out = F_ / f"classifier_{k}.tflite"; edge.export(str(out))
    with torch.no_grad(): t = w(x).numpy()
    d = float(np.abs(np.asarray(edge(x.numpy())) - t).max()); print(mem["run"], "litert vs torch", d, "MB", round(out.stat().st_size / 1e6, 1)); assert d < 1e-3
