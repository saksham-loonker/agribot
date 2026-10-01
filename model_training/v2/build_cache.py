"""Decode every classifier sample ONCE into a uint8 448x448 'base' tensor (region -> short-side resize -> centre crop),
matching the app's crop -> resize -> centre-crop path. Output: v2/cache/field_mix.pt
Domains: 0 Tomato-Village (full + detector crop), 1 PlantDoc GT-box crops, 2 Taiwan (Mendeley), 3 Other (out-of-label leaves)."""
import json, random, re, xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch, imagehash
from PIL import Image
from libreyolo import LibreYOLO9
HERE = Path(__file__).parent; B = 448
VA = Path.home() / "sakshyam-agribot/archive/Variant-a(Multiclass Classification)"
CL = sorted(d.name for d in (VA / "train").iterdir() if d.is_dir()) + ["Other"]; OTHER = 8
PD_MAP = {"Tomato Early blight leaf": "Early_blight", "Tomato leaf late blight": "Late_blight", "Tomato leaf": "Healthy"}
TW_MAP = {"healthy": "Healthy", "late blight": "Late_blight", "late_blight": "Late_blight"}   # other Taiwan classes -> Other
leaky = set(json.load(open(HERE / "leaky_T4.json")))
def base(im, box=None, pad=0.1):
    if box is not None:
        x1, x2 = sorted(box[0::2]); y1, y2 = sorted(box[1::2]); w, h = x2 - x1, y2 - y1; W, H = im.size
        l, t, r, b = max(0, x1 - pad * w), max(0, y1 - pad * h), min(W, x2 + pad * w), min(H, y2 + pad * h)
        if r - l < 16 or b - t < 16: return None
        im = im.crop((l, t, r, b))
    w, h = im.size; k = B / min(w, h); im = im.resize((max(B, round(w * k)), max(B, round(h * k))), Image.BICUBIC)
    w, h = im.size; l, t = (w - B) // 2, (h - B) // 2
    return np.asarray(im.crop((l, t, l + B, t + B)), dtype=np.uint8)
(HERE / "cache").mkdir(exist_ok=True); NMAX = 30000
XM = np.lib.format.open_memmap(HERE / "cache/field_mix_X.npy", mode="w+", dtype=np.uint8, shape=(NMAX, B, B, 3))
S = []   # (label, domain, split, group); pixels go straight to the memmap
def add(a, y, d, s, g):
    if a is not None: XM[len(S)] = a; S.append((y, d, s, g))
# ---- Tomato-Village: full image + deployed detector's top crop
det = LibreYOLO9(str(HERE / "det_runs/y9t_416_comb/weights/best.pt"), size="t", nb_classes=1)
for split in ("train", "val", "test"):
    for ci, c in enumerate(CL[:8]):
        for p in sorted((VA / split / c).iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}: continue
            im = Image.open(p).convert("RGB"); sp = split if split == "train" else (split + ("_clean" if str(p) not in leaky else "_leaky"))
            add(base(im), ci, 0, sp, "va_full")
            b = det.predict(im, conf=0.25, imgsz=320).boxes
            add(base(im, b.xyxy[b.conf.argmax()].tolist()) if len(b) else base(im), ci, 0, sp, "va_crop")
print("VA done", len(S), flush=True)
# ---- PlantDoc (GitHub VOC): TRAIN -> 85% train / 15% field-val by photo; TEST -> test (dups of TRAIN dropped)
pd = HERE / "det_data/plantdoc_src"; hashes = {}
for split in ("TRAIN", "TEST"):
    for x in sorted((pd / split).glob("*.xml")):
        if x.with_suffix(".jpg").exists(): hashes[(split, x)] = None
def h(k):
    try: return k, imagehash.phash(Image.open(k[1].with_suffix(".jpg")).convert("RGB"))
    except Exception: return k, None
with ThreadPoolExecutor(24) as ex: hashes = dict(ex.map(h, hashes))
trH = [v for (s, _), v in hashes.items() if s == "TRAIN" and v is not None]
trH_arr = np.stack([v.hash.flatten() for v in trH])
random.seed(0); n_dup = 0; n_other_nt = 0
for (split, x), hv in sorted(hashes.items(), key=lambda kv: str(kv[0][1])):
    if hv is None: continue
    if split == "TEST":
        if (trH_arr != hv.hash.flatten()).sum(1).min() <= 4: n_dup += 1; continue
        sp = "pd_test"
    else: sp = "pd_val" if random.random() < 0.15 else "train"
    im = Image.open(x.with_suffix(".jpg")).convert("RGB")
    for o in ET.parse(x).getroot().iter("object"):
        n = o.find("name").text.strip(); bb = o.find("bndbox"); box = [float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
        if abs(box[2] - box[0]) < 24 or abs(box[3] - box[1]) < 24: continue
        if n in PD_MAP:
            y = CL.index(PD_MAP[n])
            for pad in ((0.0, 0.1, 0.25) if sp == "train" else (0.1,)): add(base(im, box, pad), y, 1, sp, x.stem)
        else:
            if not n.startswith("Tomato"):
                if sp == "train" and random.random() > 0.35: continue   # subsample non-tomato for train
                n_other_nt += 1
            add(base(im, box, 0.1), OTHER, 3, sp, x.stem)
print("PlantDoc done", len(S), "test photos dropped as TRAIN dups:", n_dup, flush=True)
# ---- Taiwan (Mendeley ngdgg79rzb, CC BY 4.0): original photos only, authors' Train/Test; Test near-dups of Train dropped
tw = HERE / "field_data/tw_raw/taiwan/Preprocessed data"; TW = {"health": "Healthy", "late blight": "Late_blight"}
trh = np.stack([imagehash.phash(Image.open(p).convert("RGB")).hash.flatten() for p in sorted((tw / "Train").glob("*/*"))])
n_tw_dup = 0
for split in ("Train", "Test"):
    for cdir in sorted(d for d in (tw / split).iterdir() if d.is_dir()):
        y = CL.index(TW[cdir.name.lower()]) if cdir.name.lower() in TW else OTHER
        for p in sorted(cdir.iterdir()):
            im = Image.open(p).convert("RGB")
            if split == "Test" and (trh != imagehash.phash(im).hash.flatten()).sum(1).min() <= 4: n_tw_dup += 1; continue
            add(base(im), y, 2, "train" if split == "Train" else "tw_test", "tw_" + cdir.name)
print("Taiwan done", len(S), "test dups dropped:", n_tw_dup, flush=True)
# ---- PlantVillage tomato (same Mendeley dataset, lab photos): train only, <=400 per class, domain 4
pvd = HERE / "field_data/pv_raw/Preprocessed data"; PV = {"healthy227": "Healthy", "early_blight227": "Early_blight", "late_blight227": "Late_blight"}
for cdir in sorted(d for d in pvd.iterdir() if d.is_dir()):
    y = CL.index(PV[cdir.name.lower()]) if cdir.name.lower() in PV else OTHER
    fs = sorted(p for p in cdir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}); random.Random(2).shuffle(fs)
    for p in fs[:400]: add(base(Image.open(p).convert("RGB")), y, 4, "train", "pv_" + cdir.name)
print("PlantVillage done", len(S), flush=True)
XM.flush(); del XM
meta = dict(n=len(S), classes=CL, label=[s[0] for s in S], domain=[s[1] for s in S], split=[s[2] for s in S], group=[s[3] for s in S])
json.dump(meta, open(HERE / "cache/field_mix_meta.json", "w"))
from collections import Counter
print("total", len(S))
for sp in sorted(set(meta["split"])):
    c = Counter((d, CL[y]) for y, d, s in zip(meta["label"], meta["domain"], meta["split"]) if s == sp); print(sp, dict(sorted(c.items())))
