"""Tomato-Village Variant-A classifier (timm). Balanced-softmax + EMA + optional online KD.
Reports val / test / test-clean (leak-free) accuracy + macro-F1, fits temperature on val."""
import argparse, json, math, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as T
from sklearn.metrics import f1_score, confusion_matrix

ROOT = Path.home() / "sakshyam-agribot/archive/Variant-a(Multiclass Classification)"
HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k")
ap.add_argument("--img", type=int, default=320)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--drop-path", type=float, default=0.1)
ap.add_argument("--teacher", default="")          # path to a run dir with best.pt
ap.add_argument("--kd-alpha", type=float, default=0.7)
ap.add_argument("--kd-t", type=float, default=2.0)
ap.add_argument("--name", required=True)
ap.add_argument("--ra-mag", type=int, default=9)
ap.add_argument("--grad-ckpt", action="store_true")
ap.add_argument("--smooth", type=float, default=0.1)
args = ap.parse_args()
torch.manual_seed(0); np.random.seed(0)
dev = "cuda"; OUT = HERE / "runs" / args.name; OUT.mkdir(parents=True, exist_ok=True)

CLASSES = sorted(d.name for d in (ROOT / "train").iterdir() if d.is_dir())
def items(split):
    return [(str(p), i) for i, c in enumerate(CLASSES) for p in sorted((ROOT / split / c).iterdir())
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
leaky = set(json.load(open(HERE / "leaky_T4.json")))

class DS(Dataset):
    def __init__(s, it, tf): s.it, s.tf = it, tf; s.cache = {}
    def __len__(s): return len(s.it)
    def __getitem__(s, i):
        p, y = s.it[i]
        return s.tf(Image.open(p).convert("RGB")), y

cfg = timm.data.resolve_data_config({}, model=args.model) if False else None
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
tf_train = T.Compose([T.ToImage(), T.RandomResizedCrop(args.img, scale=(0.3, 1.0), ratio=(0.75, 1.33), antialias=True),
    T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomApply([T.RandomRotation(30)], p=0.5),
    T.RandAugment(num_ops=2, magnitude=args.ra_mag), T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD),
    T.RandomErasing(p=0.25)])
tf_eval = T.Compose([T.ToImage(), T.Resize(args.img, antialias=True), T.CenterCrop(args.img),
    T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])

tr, va, te = items("train"), items("val"), items("test")
dl = lambda it, tf, sh: DataLoader(DS(it, tf), batch_size=args.bs, shuffle=sh, num_workers=args.workers,
                                   pin_memory=True, drop_last=sh, persistent_workers=True)
dl_tr, dl_va, dl_te = dl(tr, tf_train, True), dl(va, tf_eval, False), dl(te, tf_eval, False)

counts = np.bincount([y for _, y in tr], minlength=len(CLASSES))
log_prior = torch.tensor(np.log(counts / counts.sum()), dtype=torch.float32, device=dev)

kw = dict(img_size=args.img) if "vit" in args.model else {}
model = timm.create_model(args.model, pretrained=True, num_classes=len(CLASSES), drop_path_rate=args.drop_path, **kw).to(dev)
model.set_grad_checkpointing(args.grad_ckpt)
head = model.get_classifier(); nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
model = model.to(memory_format=torch.channels_last)
ema = timm.utils.ModelEmaV3(model, decay=0.998, use_warmup=True)
teacher = None
if args.teacher:
    tm = json.load(open(Path(args.teacher) / "meta.json"))
    teacher = timm.create_model(tm["model"], pretrained=False, num_classes=len(CLASSES), **(dict(img_size=tm["img"]) if "vit" in tm["model"] else {})).to(dev).eval()
    teacher.load_state_dict(torch.load(Path(args.teacher) / "best.pt", map_location=dev)); teacher.requires_grad_(False)
    t_img = tm["img"]
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
steps = args.epochs * len(dl_tr); warm = 2 * len(dl_tr)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))

@torch.no_grad()
def predict(m, loader):
    m.eval(); L, Y = [], []
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = m(x.to(dev, non_blocking=True).to(memory_format=torch.channels_last))
            lo = lo + m(torch.flip(x, dims=[3]).to(dev).to(memory_format=torch.channels_last)) if False else lo
        L.append(lo.float().cpu()); Y.append(y)
    return torch.cat(L), torch.cat(Y)

def metrics(L, Y, mask=None):
    if mask is not None: L, Y = L[mask], Y[mask]
    p = L.argmax(1).numpy(); y = Y.numpy()
    return dict(acc=float((p == y).mean()), f1=float(f1_score(y, p, average="macro")), n=int(len(y)))

best, t0, gstep = -1, time.time(), 0
for ep in range(args.epochs):
    model.train(); tl = 0
    for x, y in dl_tr:
        x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last); y = y.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = model(x)
            loss = F.cross_entropy(lo + log_prior, y, label_smoothing=args.smooth)   # balanced softmax
            if teacher is not None:
                with torch.no_grad():
                    xt = F.interpolate(x, size=t_img, mode="bilinear", align_corners=False) if t_img != args.img else x
                    tlo = teacher(xt).float()
                kd = F.kl_div(F.log_softmax(lo.float() / args.kd_t, 1), F.softmax(tlo / args.kd_t, 1), reduction="batchmean") * args.kd_t ** 2
                loss = (1 - args.kd_alpha) * loss + args.kd_alpha * kd
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); gstep += 1; ema.update(model, step=gstep)
        tl += loss.item()
    Lv, Yv = predict(ema.module, dl_va); mv = metrics(Lv, Yv)
    score = mv["f1"] + mv["acc"]
    if score > best:
        best = score; torch.save(ema.module.state_dict(), OUT / "best.pt"); best_ep = ep
    print(f"ep {ep:02d} loss {tl/len(dl_tr):.3f} val acc {mv['acc']:.4f} f1 {mv['f1']:.4f} [{time.time()-t0:.0f}s]", flush=True)

# final eval with best EMA weights
ema.module.load_state_dict(torch.load(OUT / "best.pt")); m = ema.module
Lv, Yv = predict(m, dl_va); Lt, Yt = predict(m, dl_te)
clean_v = torch.tensor([p not in leaky for p, _ in va]); clean_t = torch.tensor([p not in leaky for p, _ in te])
# temperature scaling on val (NLL)
Tt = torch.ones(1, requires_grad=True); o = torch.optim.LBFGS([Tt], lr=0.1, max_iter=100)
def closure():
    o.zero_grad(); l = F.cross_entropy(Lv / Tt, Yv); l.backward(); return l
o.step(closure)
def ece(L, Y, n=15):
    pr = F.softmax(L, 1); c, p = pr.max(1); acc = (p == Y).float(); e = 0
    for lo, hi in zip(torch.linspace(0, 1, n + 1)[:-1], torch.linspace(0, 1, n + 1)[1:]):
        b = (c > lo) & (c <= hi)
        if b.any(): e += b.float().mean() * (acc[b].mean() - c[b].mean()).abs()
    return float(e)
res = dict(model=args.model, img=args.img, params_M=sum(p.numel() for p in m.parameters()) / 1e6, best_epoch=best_ep,
           val=metrics(Lv, Yv), val_clean=metrics(Lv, Yv, clean_v), test=metrics(Lt, Yt), test_clean=metrics(Lt, Yt, clean_t),
           temperature=float(Tt), test_ece_raw=ece(Lt, Yt), test_ece_cal=ece(Lt / Tt.detach(), Yt),
           train_minutes=(time.time() - t0) / 60, classes=CLASSES,
           test_confusion=confusion_matrix(Yt.numpy(), Lt.argmax(1).numpy()).tolist())
json.dump(dict(model=args.model, img=args.img), open(OUT / "meta.json", "w"))
json.dump(res, open(OUT / "results.json", "w"), indent=1)
torch.save({"val": (Lv, Yv), "test": (Lt, Yt)}, OUT / "logits.pt")
print(json.dumps({k: v for k, v in res.items() if k not in ("classes", "test_confusion")}, indent=1))
