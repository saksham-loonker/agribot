"""Stage A (env agribot2): ONNX export of the final classifier + calibration on VALIDATION only.
Calibration: additive Other-logit bias (balanced val score), temperature (NLL), energy reject threshold."""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, timm, onnx
from onnxsim import simplify
timm.layers.set_fused_attn(False)                       # plain matmul/softmax attention -> TFLite/GPU-delegate friendly
V2 = Path(__file__).resolve().parent.parent; run = sys.argv[1]; OUT = V2 / "final"; dev = "cuda"
meta = json.load(open(V2 / "runs" / run / "meta.json")); S = meta["img"]; NC = meta["num_classes"]
kw = dict(img_size=S) if "vit" in meta["model"] else {}
m = timm.create_model(meta["model"], pretrained=False, num_classes=NC, **kw).eval()
m.load_state_dict(torch.load(V2 / "runs" / run / "best.pt", map_location="cpu"))
x = torch.randn(1, 3, S, S); onnx_path = OUT / "classifier.onnx"
torch.onnx.export(m, x, onnx_path, input_names=["image"], output_names=["logits"], opset_version=17, dynamo=False)
mo, ok = simplify(onnx.load(onnx_path)); assert ok; onnx.save(mo, onnx_path)
# ---- calibration on validation
C = json.load(open(V2 / "cache/field_mix_meta.json")); SP = np.array(C["split"]); D = np.array(C["domain"]); Y = np.array(C["label"])
XM = np.load(V2 / "cache/field_mix_X.npy", mmap_mode="r"); m = m.to(dev)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1) * 255; STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1) * 255
@torch.no_grad()
def lg(ii):
    o = []
    for i in range(0, len(ii), 128):
        b = torch.from_numpy(np.ascontiguousarray(XM[ii[i:i + 128]])).to(dev).permute(0, 3, 1, 2).float()
        o.append(m(((F.interpolate(b, size=(S, S), mode="bilinear", antialias=True, align_corners=False) - MEAN) / STD)).float().cpu())
    return torch.cat(o)
vs = [np.where(np.isin(SP, ["val_clean", "val_leaky"]))[0], np.where((SP == "pd_val") & (D == 1))[0], np.where((SP == "pd_val") & (D == 3))[0]]
vl = [lg(v) for v in vs]
def rule_scores(b, T, gate):
    """Balanced val score of the full app decision rule: Other if argmax==Other or (gate and energy < thr)."""
    adj = [lo + torch.tensor([0.0] * 8 + [b]) for lo in vl]
    e_in = torch.cat([T * torch.logsumexp(adj[0][:, :8] / T, 1), T * torch.logsumexp(adj[1][:, :8] / T, 1)]).numpy()
    thr = float(np.percentile(e_in, 5)) if gate else -1e9; sc = []
    for k, (v, lo) in enumerate(zip(vs, adj)):
        p = lo.argmax(1).numpy(); e = (T * torch.logsumexp(lo[:, :8] / T, 1)).numpy(); lab = np.where((p == 8) | (e < thr), 8, p)
        sc.append(float((lab == Y[v]).mean()))
    return float(np.mean(sc)), thr, sc
# temperature first (bias-free NLL on all val), then grid the decision rule
L0 = torch.cat(vl); YY = torch.tensor(np.concatenate([Y[v] for v in vs]))
Tt = torch.ones(1, requires_grad=True); opt = torch.optim.LBFGS([Tt], lr=0.1, max_iter=200)
def clo():
    opt.zero_grad(); l = F.cross_entropy(L0 / Tt, YY); l.backward(); return l
opt.step(clo); T = float(Tt)
best = (-1,)
for gate in (False, True):
    for bb in np.arange(-3, 3.01, 0.25):
        sc, thr, parts = rule_scores(float(bb), T, gate)
        if sc > best[0]: best = (sc, float(bb), gate, thr, parts)
best_score, bias, gate, thr, parts = best
print("val rule: bias", bias, "energy_gate", gate, "balanced", round(best_score, 4), "parts [va, pd_map, pd_other]", [round(p, 4) for p in parts])
cal = dict(run=run, model=meta["model"], img=S, num_classes=NC, classes=meta["classes"], other_bias=bias, temperature=T,
           energy_gate=gate, energy_reject_below=(thr if gate else None), balanced_val_score=best_score, val_parts=parts)
json.dump(cal, open(OUT / "calibration.json", "w"), indent=1); print(json.dumps(cal))
