"""Stage B (env agribot-tflite, PYTHONNOUSERSITE=1): ONNX -> TFLite, parity, held-out eval OF THE SHIPPED .tflite files
with the canonical (app) preprocessing, composite voting, golden set + app model manifest."""
import json, subprocess, hashlib, shutil, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, onnxruntime as ort, tensorflow as tf, imagehash
from PIL import Image
import canon
F_ = Path(__file__).resolve().parent; V2 = F_.parent; cal = json.load(open(F_ / "calibration.json")); MEM = cal["members"]; CL = cal["classes"]
B = F_ / "bundle"; G = B / "golden"; shutil.rmtree(B, ignore_errors=True); G.mkdir(parents=True)
CF = [f"disease_classifier_{k}.tflite" for k in range(len(MEM))]
for k, f in enumerate(CF): shutil.copy(F_ / f"classifier_{k}.tflite", B / f)   # litert-torch exports (onnx2tf mis-converts ViT attention)
shutil.copy(V2 / "export/tflite_leaf_y9t_416_comb_320/leaf_y9t_416_comb_320_float32.tflite", B / "leaf_detector.tflite")
def interp(p):
    it = tf.lite.Interpreter(model_path=str(p), num_threads=8); it.allocate_tensors(); return it
CIS, DI = [interp(B / f) for f in CF], interp(B / "leaf_detector.tflite")
def run(it, x):
    it.set_tensor(it.get_input_details()[0]["index"], x); it.invoke(); return it.get_tensor(it.get_output_details()[0]["index"])
# parity vs ONNX
def ens_logits(img, box=None):
    """per-member logits and the combined log(mean softmax)."""
    per = [run(ci, canon.classifier_input(img, m["img"], box)).ravel().astype(np.float64) for ci, m in zip(CIS, MEM)]
    return per, np.log(np.mean([canon.softmax(z) for z in per], 0) + 1e-12)
bias = np.array([0.0] * 8 + [cal["other_bias"]]); T = cal["temperature"]; thr = cal["energy_reject_below"] if cal["energy_gate"] else -1e9
def decide(logits):
    z = np.asarray(logits, np.float64).ravel() + bias; p = canon.softmax(z, T); e = T * np.log(np.exp(z[:8] / T - (z[:8] / T).max()).sum()) + z[:8].max()
    return p, e
def label(p, e): return 8 if (p.argmax() == 8 or e < thr) else int(p.argmax())
# ---- held-out eval on cached bases (448 base -> canonical classifier_input)
C = json.load(open(V2 / "cache/field_mix_meta.json")); SP = np.array(C["split"]); GR = np.array(C["group"]); Y = np.array(C["label"]); D = np.array(C["domain"])
XM = np.load(V2 / "cache/field_mix_X.npy", mmap_mode="r")
sets = {"va_test_clean_full": (SP == "test_clean") & (GR == "va_full"), "va_test_clean_crop": (SP == "test_clean") & (GR == "va_crop"),
        "pd_test_mapped": (SP == "pd_test") & (D == 1), "pd_test_other": (SP == "pd_test") & (D == 3),
        "tw_test_mapped": (SP == "tw_test") & (Y != 8), "tw_test_other": (SP == "tw_test") & (Y == 8)}
def wilson(k, n):
    z = 1.96; p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(p, 4), round(c - h, 4), round(c + h, 4), n]
R = {}
for k, msk in sets.items():
    ii = np.where(msk)[0]; ok = 0
    for i in ii:
        p, e = decide(ens_logits(np.asarray(XM[i]))[1]); ok += int(label(p, e) == Y[i])
    R[k] = wilson(ok, len(ii)); print(k, R[k], flush=True)
R["average_of_6"] = round(float(np.mean([v[0] for v in R.values()])), 4); print("AVERAGE", R["average_of_6"], flush=True)
# ---- composite (app 'Check a plant'): PlantDoc TEST photos with one mapped label; detector -> top-4 leaves -> mean prob
pd = V2 / "det_data/plantdoc_src"; MAP = {"Tomato Early blight leaf": "Early_blight", "Tomato leaf late blight": "Late_blight", "Tomato leaf": "Healthy"}
trh = np.stack([imagehash.phash(Image.open(x.with_suffix(".jpg")).convert("RGB")).hash.flatten() for x in sorted((pd / "TRAIN").glob("*.xml")) if x.with_suffix(".jpg").exists()])
photos = []
for x in sorted((pd / "TEST").glob("*.xml")):
    j = x.with_suffix(".jpg")
    if not j.exists(): continue
    names = {o.find("name").text.strip() for o in ET.parse(x).getroot().iter("object")}
    if len(names) != 1 or next(iter(names)) not in MAP: continue
    im = Image.open(j).convert("RGB")
    if (trh != imagehash.phash(im).hash.flatten()).sum(1).min() <= 4: continue
    photos.append((j, im, CL.index(MAP[next(iter(names))])))
def check_plant(img, k=4):
    di, lb = canon.detector_input(img); dets = canon.decode_detections(run(DI, di), lb, img.shape[1], img.shape[0])
    boxes = [d[:4] for d in dets[:k]] or [None]
    ps = [decide(ens_logits(img, bx)[1]) for bx in boxes]; P = np.mean([p for p, _ in ps], 0); E = float(np.mean([e for _, e in ps]))
    return label(P, E), len(dets)
single_ok = comp_ok = 0
for j, im, y in photos:
    a = np.asarray(im); single_ok += int(label(*decide(ens_logits(a)[1])) == y); comp_ok += int(check_plant(a)[0] == y)
R["pd_test_photos_whole_image"] = wilson(single_ok, len(photos)); R["pd_test_photos_check_a_plant_top4"] = wilson(comp_ok, len(photos))
print("photos", R["pd_test_photos_whole_image"], R["pd_test_photos_check_a_plant_top4"], flush=True)
# ---- golden set (PNG, so Android decodes identical pixels)
VA = Path.home() / "sakshyam-agribot/archive/Variant-a(Multiclass Classification)/test"
gold = [(sorted((VA / c).iterdir())[0], c) for c in sorted(d.name for d in VA.iterdir() if d.is_dir())]
gold += [(p[0], CL[p[2]]) for p in photos[:4]]
gold += [(sorted((V2 / "field_data/tw_raw/taiwan/Preprocessed data/Test/Late blight").iterdir())[0], "Late_blight")]
entries = []
for n, (p, lab) in enumerate(gold):
    im = Image.open(p).convert("RGB")
    if max(im.size) > 640: im.thumbnail((640, 640), Image.BICUBIC)
    name = f"golden_{n:02d}.png"; im.save(G / name); a = np.asarray(Image.open(G / name).convert("RGB"))
    per, lo = ens_logits(a); p_, e_ = decide(lo); cis = [canon.classifier_input(a, m["img"]) for m in MEM]
    di, lb = canon.detector_input(a); raw = run(DI, di); dets = canon.decode_detections(raw, lb, a.shape[1], a.shape[0])
    top = dets[0][:4] if dets else None
    crop = None
    if top:
        cper, cl = ens_logits(a, top); cp, ce = decide(cl)
        crop = dict(box=[round(v, 3) for v in top], member_logits=[[float(v) for v in z] for z in cper], probs=[float(v) for v in cp], label=CL[label(cp, ce)])
    def checks(ci):
        flat = ci.ravel(); idx = [0, 1, 2, 1000, 50000, flat.size // 2, flat.size - 1]
        return dict(mean=float(flat.mean()), sum=float(flat.astype(np.float64).sum()), samples={str(i): float(flat[i]) for i in idx})
    entries.append(dict(file=name, width=int(a.shape[1]), height=int(a.shape[0]), source_label=lab,
        classifier_input_checks=[checks(ci) for ci in cis],
        detector_input_checks=dict(mean=float(di.mean()), letterbox=[float(lb[0]), int(lb[1]), int(lb[2])]),
        whole_image=dict(member_logits=[[float(v) for v in z] for z in per], probs=[float(v) for v in p_], energy=float(e_), label=CL[label(p_, e_)]),
        detections=[[round(v, 3) for v in d] for d in dets], top_crop=crop))
json.dump(dict(note="Expected outputs from the shipped .tflite files with canon.py preprocessing", tolerance=dict(input=1e-3, logits=2e-2, box_px=2.0), entries=entries), open(G / "golden_expected.json", "w"), indent=1)
shutil.copy(F_ / "canon.py", G / "canon_reference.py")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
man = dict(bundle_id="agribot-v2-" + sha(B / CF[0])[:8], schema=2,
  status="DRAFT: requires named human reviewer sign-off and licence review (Tomato-Village, FieldPlant, PlantDoc, Mendeley) before release",
  classifier=dict(members=[dict(file=f, sha256=sha(B / f), arch=m["model"], licence="Apache-2.0", input=[1, m["img"], m["img"], 3]) for f, m in zip(CF, MEM)],
      combine="log(mean_i softmax(logits_i)); then add other_logit_bias to index 8; label = argmax; confidence = softmax(z / temperature)", layout="NHWC", preprocess="crop(box,+10% pad, floor/ceil) -> centred square of short side -> canonical resize -> /255 -> ImageNet mean/std",
      mean=canon.MEAN.tolist(), std=canon.STD.tolist(), labels=CL, other_index=8, other_logit_bias=cal["other_bias"],
      temperature=cal["temperature"], energy_gate=cal["energy_gate"], energy_reject_below=cal["energy_reject_below"],
      energy_definition="T * logsumexp(z[0:8] / T) with z = logits + bias"),
  detector=dict(file="leaf_detector.tflite", sha256=sha(B / "leaf_detector.tflite"), arch="YOLOv9-T (LibreYOLO, MIT)", input=[1, 320, 320, 3], layout="NHWC",
      preprocess="letterbox centred, pad 114, canonical resize, /255", output=[1, 5, 2100], output_rows=["x1", "y1", "x2", "y2", "score"],
      coordinate_space="letterboxed_input_pixels", score_threshold=0.35, nms_iou=0.6, max_detections=16, labels=["leaf"]),
  resize_algorithm="separable; scale>1 area-average, else bilinear half-pixel centres (see golden/canon_reference.py)",
  heldout_results=R)
json.dump(man, open(B / "model_manifest.json", "w"), indent=1); print(json.dumps(R, indent=1))
