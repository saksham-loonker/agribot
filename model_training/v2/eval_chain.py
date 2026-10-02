"""Classifier + chain evaluation.
 1. Variant-A test: per-class P/R/F1, hflip TTA, all student/teacher runs.
 2. Cross-dataset: PlantDoc tomato leaf crops (GT boxes) mapped to Early_blight / Late_blight / Healthy.
 3. OOD rejection: unmapped PlantDoc tomato diseases + non-tomato leaves vs Variant-A test (MSP, energy).
 4. Chain: detector -> top crop -> classifier on Variant-A test; detector -> crops -> vote on PlantDoc tomato photos."""
import json, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, timm
from PIL import Image
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from libreyolo import LibreYOLO9
HERE = Path(__file__).parent; E = HERE / "eval"; dev = "cuda"
VA = Path.home() / "sakshyam-agribot/archive/Variant-a(Multiclass Classification)"
leaky = set(json.load(open(HERE / "leaky_T4.json")))
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
def load_clf(run):
    meta = json.load(open(HERE / "runs" / run / "meta.json")); res = json.load(open(HERE / "runs" / run / "results.json"))
    kw = dict(img_size=meta["img"]) if "vit" in meta["model"] else {}
    m = timm.create_model(meta["model"], pretrained=False, num_classes=8, **kw).to(dev).eval()
    m.load_state_dict(torch.load(HERE / "runs" / run / "best.pt", map_location=dev)); return m, meta["img"], res["temperature"], res["classes"]
def prep(im, s):
    im = im.convert("RGB"); w, h = im.size; k = s / min(w, h)
    im = im.resize((max(s, round(w * k)), max(s, round(h * k))), Image.BILINEAR); w, h = im.size
    l, t = (w - s) // 2, (h - s) // 2; im = im.crop((l, t, l + s, t + s))
    return (torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255 - MEAN) / STD
@torch.no_grad()
def logits(m, ims, s, tta=False):
    out = []
    for i in range(0, len(ims), 64):
        x = torch.stack([prep(im, s) for im in ims[i:i + 64]]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = m(x).float()
            if tta: lo = (lo + m(torch.flip(x, dims=[3])).float()) / 2
        out.append(lo.cpu())
    return torch.cat(out)
def crop(im, b, pad=0.1):
    x1, x2 = sorted(b[0::2]); y1, y2 = sorted(b[1::2]); w, h = x2 - x1, y2 - y1; W, H = im.size
    l, t, r, b2 = max(0, x1 - pad * w), max(0, y1 - pad * h), min(W, x2 + pad * w), min(H, y2 + pad * h)
    return im.crop((l, t, r, b2)) if r - l >= 8 and b2 - t >= 8 else None
R = {}
# ---------- data
CL = sorted(d.name for d in (VA / "test").iterdir() if d.is_dir())
te = [(p, i) for i, c in enumerate(CL) for p in sorted((VA / "test" / c).iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
te_im = [Image.open(p).convert("RGB") for p, _ in te]; te_y = np.array([y for _, y in te]); clean = np.array([str(p) not in leaky for p, _ in te])
MAP = {"Tomato Early blight leaf": "Early_blight", "Tomato leaf late blight": "Late_blight", "Tomato leaf": "Healthy"}
pd_in, pd_in_y, pd_ood_tom, pd_ood_other, pd_photos = [], [], [], [], []
for split in ("TRAIN", "TEST"):
    for x in sorted((HERE / "det_data/plantdoc_src" / split).glob("*.xml")):
        img = x.with_suffix(".jpg")
        if not img.exists(): continue
        try: im = Image.open(img).convert("RGB")
        except Exception: continue
        objs = []
        for o in ET.parse(x).getroot().iter("object"):
            n = o.find("name").text.strip(); b = o.find("bndbox"); bb = [float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
            if abs(bb[2] - bb[0]) < 24 or abs(bb[3] - bb[1]) < 24: continue
            c = crop(im, bb, 0.0)
            if c is None: continue
            objs.append((n, bb))
            if n in MAP: pd_in.append(c); pd_in_y.append(CL.index(MAP[n]))
            elif n.startswith("Tomato"): pd_ood_tom.append(c)
            elif len(pd_ood_other) < 1500: pd_ood_other.append(c)
        labs = {n for n, _ in objs}
        if len(labs) == 1 and next(iter(labs)) in MAP: pd_photos.append((im, CL.index(MAP[next(iter(labs))])))
pd_in_y = np.array(pd_in_y)
R["data"] = dict(variantA_test=len(te), plantdoc_mapped_crops=len(pd_in), plantdoc_mapped_per_class={CL[k]: int((pd_in_y == k).sum()) for k in set(pd_in_y.tolist())},
                 ood_tomato_other_disease=len(pd_ood_tom), ood_other_plants=len(pd_ood_other), plantdoc_single_label_tomato_photos=len(pd_photos))
print(R["data"], flush=True)
def acc(lo, y, mask=None):
    p = lo.argmax(1).numpy()
    if mask is not None: p, y = p[mask], y[mask]
    return round(float((p == y).mean()), 4), round(float(f1_score(y, p, average="macro")), 4)
def ood(lo_in, lo_out, T):
    s_in, s_out = {}, {}
    s_in["msp"] = F.softmax(lo_in / T, 1).max(1).values.numpy(); s_out["msp"] = F.softmax(lo_out / T, 1).max(1).values.numpy()
    s_in["energy"] = (T * torch.logsumexp(lo_in / T, 1)).numpy(); s_out["energy"] = (T * torch.logsumexp(lo_out / T, 1)).numpy()
    o = {}
    for k in s_in:
        y = np.r_[np.ones(len(s_in[k])), np.zeros(len(s_out[k]))]; s = np.r_[s_in[k], s_out[k]]
        thr = np.percentile(s_in[k], 5)   # keep 95% of in-distribution
        o[k] = dict(auroc=round(float(roc_auc_score(y, s)), 4), fpr_at_tpr95=round(float((s_out[k] >= thr).mean()), 4), thr_tpr95=float(thr))
    return o
# ---------- 1-3 per classifier
for run in ("mnv4m_384", "mnv4m_384_kd1", "mnv4m_384_kd2", "dinov2l_t_448"):
    m, s, T, _ = load_clf(run)
    lo = logits(m, te_im, s); lo_t = logits(m, te_im, s, tta=True)
    lin = logits(m, pd_in, s); lin_t = logits(m, pd_in, s, tta=True)
    lot = logits(m, pd_ood_tom, s); loo = logits(m, pd_ood_other, s)
    r = dict(test=acc(lo, te_y), test_clean=acc(lo, te_y, clean), test_tta=acc(lo_t, te_y), test_clean_tta=acc(lo_t, te_y, clean),
             plantdoc_tomato3=acc(lin, pd_in_y), plantdoc_tomato3_tta=acc(lin_t, pd_in_y),
             plantdoc_pred_hist={CL[k]: int(v) for k, v in enumerate(np.bincount(lin.argmax(1).numpy(), minlength=8)) if v},
             ood_tomato_other_disease=ood(lo, lot, T), ood_other_plants=ood(lo, loo, T))
    if run == "mnv4m_384_kd1":
        r["per_class_test"] = classification_report(te_y, lo.argmax(1).numpy(), target_names=CL, output_dict=True, zero_division=0)
    R[run] = r; print(run, {k: v for k, v in r.items() if k != "per_class_test"}, flush=True)
    del m; torch.cuda.empty_cache()
# ---------- 4 chain
m, s, T, _ = load_clf("mnv4m_384_kd1")
for drun, size in (("y9t_416_comb", "t"), ("y9s_416_comb", "s")):
    det = LibreYOLO9(str(HERE / f"det_runs/{drun}/weights/best.pt"), size=size, nb_classes=1)
    for dsz in (320, 416):
        crops, found = [], 0
        for im in te_im:
            b = det.predict(im, conf=0.25, imgsz=dsz).boxes
            if len(b): found += 1; c = crop(im, b.xyxy[b.conf.argmax()].tolist()); crops.append(c if c is not None else im)
            else: crops.append(im)   # fallback: whole frame
        lo = logits(m, crops, s)
        # PlantDoc photos: classify all detections, confidence-weighted probability vote
        ph_p = []
        for im, y in pd_photos:
            b = det.predict(im, conf=0.25, imgsz=dsz).boxes
            keep = [(c, float(cf)) for bb, cf in zip(b.xyxy[:8], b.conf[:8]) if (c := crop(im, bb.tolist())) is not None] or [(im, 1.0)]
            cs = [c for c, _ in keep]; w = torch.tensor([cf for _, cf in keep])
            pr = F.softmax(logits(m, cs, s) / T, 1); ph_p.append(int((pr * w.view(-1, 1)).sum(0).argmax()))
        ph_y = np.array([y for _, y in pd_photos]); ph_p = np.array(ph_p)
        R[f"chain_{drun}@{dsz}"] = dict(variantA_detect_rate=round(found / len(te_im), 4), variantA_test=acc(lo, te_y), variantA_test_clean=acc(lo, te_y, clean),
                                       plantdoc_photo_acc=round(float((ph_p == ph_y).mean()), 4))
        print(f"chain_{drun}@{dsz}", R[f"chain_{drun}@{dsz}"], flush=True)
json.dump(R, open(E / "clf_chain_eval.json", "w"), indent=1)
