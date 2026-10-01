"""Field fine-tune of the distilled MobileNetV4-M: 9 classes (8 Tomato-Village + Other).
Whole dataset lives on the GPU as uint8; all augmentation is batched GPU kernels (roi_align RRC, affine, colour,
blur, erasing); torch.compile + bf16 + channels_last. Init from runs/mnv4m_384_kd1; KD from DINOv2-L only on
Tomato-Village samples (teacher is unreliable on field photos)."""
import argparse, json, math, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, timm, timm.optim
from torchvision.ops import roi_align
from sklearn.metrics import f1_score
HERE = Path(__file__).parent; dev = "cuda"
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--per-epoch", type=int, default=6144)
ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=4e-4); ap.add_argument("--img", type=int, default=384)
ap.add_argument("--w", default="0.4,0.25,0.12,0.13,0.1", help="sampling weight per domain: VA, PlantDoc, Taiwan, Other, PlantVillage")
ap.add_argument("--kd-alpha", type=float, default=0.5); ap.add_argument("--no-compile", action="store_true")
ap.add_argument("--model", default="mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k"); ap.add_argument("--init", default="mnv4m_384_kd1")
ap.add_argument("--teacher", default="dinov2l_t_224", help="run name or none"); ap.add_argument("--kd-scope", default="va", choices=["va", "all"])
ap.add_argument("--layer-decay", type=float, default=0.0); ap.add_argument("--grad-ckpt", action="store_true"); ap.add_argument("--drop-path", type=float, default=0.1); ap.add_argument("--wd", type=float, default=0.05)
a = ap.parse_args(); torch.manual_seed(0); np.random.seed(0)
OUT = HERE / "runs" / a.name; OUT.mkdir(parents=True, exist_ok=True)
torch.backends.cudnn.benchmark = True; torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
C = json.load(open(HERE / "cache/field_mix_meta.json")); CL = C["classes"]; NC = len(CL)
XM = np.load(HERE / "cache/field_mix_X.npy", mmap_mode="r")[: C["n"]]
X = torch.empty((C["n"], 3, XM.shape[1], XM.shape[2]), dtype=torch.uint8, device=dev)   # whole set resident on GPU
for i in range(0, C["n"], 2048): X[i:i + 2048] = torch.from_numpy(np.ascontiguousarray(XM[i:i + 2048])).to(dev).permute(0, 3, 1, 2)
Y = torch.tensor(C["label"], device=dev); D = torch.tensor(C["domain"], device=dev); SP = np.array(C["split"]); GR = np.array(C["group"])
idx = lambda *s: torch.tensor(np.where(np.isin(SP, s))[0], device=dev)
tr = idx("train")
# per-sample sampling weight = domain weight / domain count (domain absent -> its weight is dropped)
dw = torch.tensor([float(v) for v in a.w.split(",")], device=dev); cnt = torch.bincount(D[tr], minlength=len(dw)).float()[: len(dw)]
dw = torch.where(cnt > 0, dw, torch.zeros_like(dw)); sw = (dw / cnt.clamp(min=1))[D[tr]]
prior = torch.zeros(NC, device=dev).index_add_(0, Y[tr], sw); prior = prior / prior.sum()
log_prior = prior.clamp(min=1e-6).log()
print("train", len(tr), "domain counts", cnt.tolist(), "effective class prior", [round(float(p), 3) for p in prior], flush=True)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1) * 255; STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1) * 255
S0 = X.shape[-1]; R = a.img
def gpu_aug(xb):
    n = xb.shape[0]; x = xb.float()
    # random resized crop (scale 0.3-1, ratio 3/4-4/3) via one roi_align kernel
    area = torch.empty(n, device=dev).uniform_(0.3, 1.0) * S0 * S0; logr = torch.empty(n, device=dev).uniform_(math.log(3 / 4), math.log(4 / 3))
    w = (area * logr.exp()).sqrt().clamp(max=S0); h = (area / logr.exp()).sqrt().clamp(max=S0)
    x0 = torch.rand(n, device=dev) * (S0 - w); y0 = torch.rand(n, device=dev) * (S0 - h)
    boxes = torch.stack([torch.arange(n, device=dev, dtype=torch.float), x0, y0, x0 + w, y0 + h], 1)
    x = roi_align(x, boxes, (R, R), spatial_scale=1.0, sampling_ratio=2, aligned=True)
    # flips + rotation (p=.5, +-30deg) in one affine grid_sample
    ang = torch.where(torch.rand(n, device=dev) < 0.5, torch.empty(n, device=dev).uniform_(-math.pi / 6, math.pi / 6), torch.zeros(n, device=dev))
    fx = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0); fy = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    th = torch.zeros(n, 2, 3, device=dev); th[:, 0, 0] = ang.cos() * fx; th[:, 0, 1] = -ang.sin() * fy; th[:, 1, 0] = ang.sin() * fx; th[:, 1, 1] = ang.cos() * fy
    x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), mode="bilinear", padding_mode="reflection", align_corners=False)
    # colour: brightness, contrast, saturation, slight per-channel gain (white balance)
    u = lambda lo, hi: torch.empty(n, 1, 1, 1, device=dev).uniform_(lo, hi)
    x = x * u(0.65, 1.35); m = x.mean((1, 2, 3), keepdim=True); x = (x - m) * u(0.7, 1.3) + m
    g = x.mean(1, keepdim=True); x = (x - g) * u(0.6, 1.4) + g; x = x * torch.empty(n, 3, 1, 1, device=dev).uniform_(0.9, 1.1)
    # blur (p=.25, phone defocus) via depthwise 5x5 gaussian
    k = torch.tensor([1, 4, 6, 4, 1], device=dev, dtype=torch.float); k = (k[:, None] * k[None]) / 256; k = k.expand(3, 1, 5, 5)
    bl = torch.rand(n, device=dev) < 0.25; x = torch.where(bl.view(-1, 1, 1, 1), F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), k, groups=3), x)
    x = ((x.clamp(0, 255) - MEAN) / STD)
    # random erasing (p=.25)
    er = torch.rand(n, device=dev) < 0.25; ew = (torch.rand(n, device=dev) * 0.3 + 0.1) * R; eh = (torch.rand(n, device=dev) * 0.3 + 0.1) * R
    ex = torch.rand(n, device=dev) * (R - ew); ey = torch.rand(n, device=dev) * (R - eh); ar = torch.arange(R, device=dev, dtype=torch.float)
    mask = er.view(-1, 1, 1) & (ar.view(1, 1, -1) >= ex.view(-1, 1, 1)) & (ar.view(1, 1, -1) < (ex + ew).view(-1, 1, 1)) & (ar.view(1, -1, 1) >= ey.view(-1, 1, 1)) & (ar.view(1, -1, 1) < (ey + eh).view(-1, 1, 1))
    x = torch.where(mask.unsqueeze(1), torch.randn_like(x), x)
    return x.contiguous(memory_format=torch.channels_last)
def gpu_eval(xb, r=R):
    return ((F.interpolate(xb.float(), size=(r, r), mode="bilinear", antialias=True, align_corners=False) - MEAN) / STD).contiguous(memory_format=torch.channels_last)
# ---- models
vit = lambda name, img: dict(img_size=img) if "vit" in name else {}
model = timm.create_model(a.model, pretrained=(a.init == "pretrained"), num_classes=NC, drop_path_rate=a.drop_path, **vit(a.model, a.img)).to(dev)
hn = model.default_cfg["classifier"]; head = model.get_classifier()
with torch.no_grad(): head.weight.zero_(); head.bias.zero_()
if a.init != "pretrained":
    sd = torch.load(HERE / "runs" / a.init / "best.pt", map_location=dev); hw, hb = sd.pop(hn + ".weight"), sd.pop(hn + ".bias")
    print(model.load_state_dict(sd, strict=False))
    with torch.no_grad(): head.weight[: hw.shape[0]] = hw; head.bias[: hb.shape[0]] = hb
model.set_grad_checkpointing(a.grad_ckpt)
model = model.to(memory_format=torch.channels_last); ema = timm.utils.ModelEmaV3(model, decay=0.998, use_warmup=True)
teacher = None
if a.teacher != "none":
    tm = json.load(open(HERE / "runs" / a.teacher / "meta.json")); tnc = tm.get("num_classes", 8)
    teacher = timm.create_model(tm["model"], pretrained=False, num_classes=tnc, **vit(tm["model"], tm["img"])).to(dev).eval().requires_grad_(False)
    teacher.load_state_dict(torch.load(HERE / "runs" / a.teacher / "best.pt", map_location=dev))
    assert a.kd_scope == "va" or tnc == NC, "kd-scope all needs a teacher with the same classes"
fwd = model if a.no_compile else torch.compile(model); tfwd = (teacher if a.no_compile else torch.compile(teacher)) if teacher is not None else None
opt = (timm.optim.create_optimizer_v2(model, opt="adamw", lr=a.lr, weight_decay=a.wd, layer_decay=a.layer_decay) if a.layer_decay > 0
       else torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, fused=True))
spe = a.per_epoch // a.bs; steps = a.epochs * spe; warm = spe
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))
@torch.no_grad()
def predict(m, ii, bs=256):
    m.eval(); out = []
    for i in range(0, len(ii), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16): out.append(m(gpu_eval(X[ii[i:i + bs]])).float())
    return torch.cat(out) if out else torch.zeros(0, NC, device=dev)
def acc(lo, ii): return float((lo.argmax(1) == Y[ii]).float().mean()) if len(ii) else float("nan")
V = {k: idx(k) for k in ("val_clean", "val_leaky", "pd_val")}
va_val = torch.cat([V["val_clean"], V["val_leaky"]]); va_full = va_val[torch.tensor(GR[va_val.cpu().numpy()] == "va_full", device=dev)]; va_crop = va_val[torch.tensor(GR[va_val.cpu().numpy()] == "va_crop", device=dev)]
pd_map = V["pd_val"][D[V["pd_val"]] == 1]; pd_oth = V["pd_val"][D[V["pd_val"]] == 3]
best, t0, step = -1, time.time(), 0
for ep in range(a.epochs):
    model.train(); tl = 0
    order = tr[torch.multinomial(sw, spe * a.bs, replacement=True)]
    for it in range(spe):
        ii = order[it * a.bs:(it + 1) * a.bs]; x = gpu_aug(X[ii]); y = Y[ii]; isva = D[ii] == 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = fwd(x); loss = F.cross_entropy(lo.float() + log_prior, y, label_smoothing=0.05)
            km = isva if a.kd_scope == "va" else torch.ones_like(isva)
            if teacher is not None and a.kd_alpha > 0 and km.any():
                xt = x[km] if tm["img"] == R else F.interpolate(x[km], size=tm["img"], mode="bilinear", antialias=False, align_corners=False)
                with torch.no_grad(): tl_ = tfwd(xt).float()
                k = tl_.shape[1]
                kd = F.kl_div(F.log_softmax(lo[km, :k].float() / 2, 1), F.softmax(tl_ / 2, 1), reduction="batchmean") * 4
                loss = (1 - a.kd_alpha) * loss + a.kd_alpha * kd * km.float().mean() if a.kd_scope == "all" else loss + a.kd_alpha * kd * km.float().mean()
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        step += 1; ema.update(model, step=step); tl += loss.item() if it % 16 == 0 else 0
    if ep % 2 == 1 or ep == a.epochs - 1:
        e = ema.module; r = dict(va_full=acc(predict(e, va_full), va_full), va_crop=acc(predict(e, va_crop), va_crop), pd_map=acc(predict(e, pd_map), pd_map), pd_other=acc(predict(e, pd_oth), pd_oth))
        score = 0.25 * r["va_full"] + 0.25 * r["va_crop"] + 0.3 * r["pd_map"] + 0.2 * r["pd_other"]
        if score > best: best = score; best_ep = ep; torch.save(e.state_dict(), OUT / "best.pt")
        print(f"ep {ep:02d} loss {tl / (spe / 16):.3f} " + " ".join(f"{k} {v:.4f}" for k, v in r.items()) + f" score {score:.4f} [{time.time() - t0:.0f}s]", flush=True)
json.dump(dict(model=a.model, img=a.img, num_classes=NC, classes=CL, best_epoch=best_ep, train_minutes=(time.time() - t0) / 60, args=vars(a)), open(OUT / "meta.json", "w"), indent=1)
print("done", best_ep, f"{(time.time() - t0) / 60:.1f} min")
