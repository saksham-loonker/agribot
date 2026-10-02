"""Held-out evaluation for 8- or 9-class classifiers on the cached test splits (same preprocessing as the app).
Suite: Tomato-Village test (clean, full+crop), PlantDoc TEST (3 mapped classes; Other rejection), Taiwan test (2 mapped; Other).
For 8-class models 'Other' = energy-gate rejection. Wilson 95% CIs. Usage: eval_field.py RUN [RUN...]"""
import json, math, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, timm
HERE = Path(__file__).parent; dev = "cuda"
C = json.load(open(HERE / "cache/field_mix_meta.json")); SP = np.array(C["split"]); GR = np.array(C["group"]); Yall = np.array(C["label"]); Dall = np.array(C["domain"])
XM = np.load(HERE / "cache/field_mix_X.npy", mmap_mode="r")
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1) * 255; STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1) * 255
def wilson(k, n):
    if n == 0: return (float("nan"),) * 3
    p = k / n; z = 1.96; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(p, 4), round(c - h, 4), round(c + h, 4)
@torch.no_grad()
def logits(m, ii, r):
    out = []
    for i in range(0, len(ii), 128):
        x = torch.from_numpy(np.ascontiguousarray(XM[ii[i:i + 128]])).to(dev).permute(0, 3, 1, 2).float()
        x = ((F.interpolate(x, size=(r, r), mode="bilinear", antialias=True, align_corners=False) - MEAN) / STD).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16): out.append(m(x).float())
    return torch.cat(out).cpu()
def load(run):
    meta = json.load(open(HERE / "runs" / run / "meta.json")); nc = meta.get("num_classes", 8)
    kw = dict(img_size=meta["img"]) if "vit" in meta["model"] else {}
    m = timm.create_model(meta["model"], pretrained=False, num_classes=nc, **kw).to(dev).eval().to(memory_format=torch.channels_last)
    m.load_state_dict(torch.load(HERE / "runs" / run / "best.pt", map_location=dev)); return m, meta["img"], nc
sets = {"va_test_clean_full": (SP == "test_clean") & (GR == "va_full"), "va_test_clean_crop": (SP == "test_clean") & (GR == "va_crop"),
        "va_test_all_full": np.isin(SP, ["test_clean", "test_leaky"]) & (GR == "va_full"),
        "pd_test_mapped": (SP == "pd_test") & (Dall == 1), "pd_test_other": (SP == "pd_test") & (Dall == 3),
        "tw_test_mapped": (SP == "tw_test") & (Yall != 8), "tw_test_other": (SP == "tw_test") & (Yall == 8)}
val_in = (np.isin(SP, ["val_clean", "val_leaky"]) & (GR == "va_full")) | ((SP == "pd_val") & (Dall == 1))
R = {}
for run in sys.argv[1:]:
    m, r, nc = load(run); res = {}
    e_val = torch.logsumexp(logits(m, np.where(val_in)[0], r)[:, :8], 1).numpy(); thr = float(np.percentile(e_val, 5))   # keep 95% of in-label val
    bias = 0.0
    if nc == 9:   # tune one additive bias on the Other logit using VALIDATION only (balanced: VA val, PD-val mapped, PD-val other)
        vs = [np.where(np.isin(SP, ["val_clean", "val_leaky"]))[0], np.where((SP == "pd_val") & (Dall == 1))[0], np.where((SP == "pd_val") & (Dall == 3))[0]]
        vl = [logits(m, v, r) for v in vs]; best = -1
        for b in np.arange(-3, 3.01, 0.25):
            s = []
            for k, (v, lo) in enumerate(zip(vs, vl)):
                lo = lo.clone(); lo[:, 8] += b; p = lo.argmax(1).numpy(); s.append(float((p == Yall[v]).mean()))
            if np.mean(s) > best: best, bias = np.mean(s), float(b)
        res["other_bias_from_val"] = bias
    for k, mask in sets.items():
        ii = np.where(mask)[0]; lo = logits(m, ii, r); y = Yall[ii]
        if nc == 9: lo[:, 8] += bias
        p9 = lo.argmax(1).numpy(); p8 = lo[:, :8].argmax(1).numpy(); en = torch.logsumexp(lo[:, :8], 1).numpy()
        rej = ((p9 == 8) | (en < thr)) if nc == 9 else (en < thr)
        if k.endswith("other"): res[k] = dict(rejected=wilson(int(rej.sum()), len(y)), n=len(y))
        else: res[k] = dict(acc=wilson(int((p9 == y).sum()), len(y)), acc_8way=wilson(int((p8 == y).sum()), len(y)), coverage=round(float((~rej).mean()), 4), n=len(y))
    R[run] = res
    print("==", run); [print("  ", k, v) for k, v in res.items()]
    del m; torch.cuda.empty_cache()
json.dump(R, open(HERE / "eval" / ("field_eval_" + "_".join(sys.argv[1:]) + ".json"), "w"), indent=1)
