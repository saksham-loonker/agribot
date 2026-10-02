"""Ensemble evaluation: mean of softmax probs over runs; temperature + decision rule (Other bias, energy gate) chosen on
VALIDATION only; each held-out test set evaluated once. Usage: ens_eval.py RUN [RUN...]"""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, timm
HERE = Path(__file__).parent; dev = "cuda"
C = json.load(open(HERE / "cache/field_mix_meta.json")); SP = np.array(C["split"]); GR = np.array(C["group"]); Y = np.array(C["label"]); D = np.array(C["domain"])
XM = np.load(HERE / "cache/field_mix_X.npy", mmap_mode="r")
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1) * 255; STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1) * 255
def load(run):
    meta = json.load(open(HERE / "runs" / run / "meta.json")); kw = dict(img_size=meta["img"]) if "vit" in meta["model"] else {}
    m = timm.create_model(meta["model"], pretrained=False, num_classes=9, **kw).to(dev).eval()
    m.load_state_dict(torch.load(HERE / "runs" / run / "best.pt", map_location=dev)); return m, meta["img"]
@torch.no_grad()
def logp(models, ii):
    out = []
    for i in range(0, len(ii), 128):
        b = torch.from_numpy(np.ascontiguousarray(XM[ii[i:i + 128]])).to(dev).permute(0, 3, 1, 2).float(); ps = []
        for m, r in models:
            x = (F.interpolate(b, size=(r, r), mode="bilinear", antialias=True, align_corners=False) - MEAN) / STD
            with torch.autocast("cuda", dtype=torch.bfloat16): ps.append(F.softmax(m(x).float(), 1))
        out.append(torch.stack(ps).mean(0).clamp_min(1e-8).log().cpu())
    return torch.cat(out)
models = [load(r) for r in sys.argv[1:]]
vs = [np.where(np.isin(SP, ["val_clean", "val_leaky"]))[0], np.where((SP == "pd_val") & (D == 1))[0], np.where((SP == "pd_val") & (D == 3))[0]]
vl = [logp(models, v) for v in vs]
L0 = torch.cat(vl); YY = torch.tensor(np.concatenate([Y[v] for v in vs])); Tt = torch.ones(1, requires_grad=True); opt = torch.optim.LBFGS([Tt], lr=0.1, max_iter=200)
def clo():
    opt.zero_grad(); l = F.cross_entropy(L0 / Tt, YY); l.backward(); return l
opt.step(clo); T = float(Tt)
def lab(lo, b, thr):
    lo = lo + torch.tensor([0.0] * 8 + [b]); p = lo.argmax(1).numpy(); e = (T * torch.logsumexp(lo[:, :8] / T, 1)).numpy()
    return np.where((p == 8) | (e < thr), 8, p)
best = (-1,)
for gate in (False, True):
    for b in np.arange(-3, 3.01, 0.25):
        thr = -1e9
        if gate:
            adj = [lo + torch.tensor([0.0] * 8 + [b]) for lo in vl[:2]]; thr = float(np.percentile(torch.cat([T * torch.logsumexp(a[:, :8] / T, 1) for a in adj]).numpy(), 5))
        sc = float(np.mean([(lab(lo, b, thr) == Y[v]).mean() for v, lo in zip(vs, vl)]))
        if sc > best[0]: best = (sc, float(b), gate, thr)
sc, b, gate, thr = best; print("runs", sys.argv[1:], "val-selected rule: bias", b, "gate", gate, "T", round(T, 3), "balanced val", round(sc, 4))
sets = {"va_test_clean_full": (SP == "test_clean") & (GR == "va_full"), "va_test_clean_crop": (SP == "test_clean") & (GR == "va_crop"),
        "pd_test_mapped": (SP == "pd_test") & (D == 1), "pd_test_other": (SP == "pd_test") & (D == 3),
        "tw_test_mapped": (SP == "tw_test") & (Y != 8), "tw_test_other": (SP == "tw_test") & (Y == 8)}
acc = {}
for k, msk in sets.items():
    ii = np.where(msk)[0]; acc[k] = round(float((lab(logp(models, ii), b, thr) == Y[ii]).mean()), 4)
print(acc, "AVERAGE", round(float(np.mean(list(acc.values()))), 4))
import os
if os.environ.get("SAVE_CAL"):
    runs = []
    for r in sys.argv[1:]:
        meta = json.load(open(HERE / "runs" / r / "meta.json")); runs.append(dict(run=r, model=meta["model"], img=meta["img"]))
    json.dump(dict(members=runs, num_classes=9, classes=json.load(open(HERE / "runs" / sys.argv[1] / "meta.json"))["classes"], combine="log(mean(softmax(logits_i)))",
                   other_bias=b, temperature=T, energy_gate=gate, energy_reject_below=(thr if gate else None), balanced_val_score=sc, torch_eval_test=acc),
              open(os.environ["SAVE_CAL"], "w"), indent=1)
