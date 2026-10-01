"""Roboflow FieldPlant + PlantDoc (yolov8 export) -> single-class 'leaf' YOLO set.
Splits: a source's own test split stays test; val (and test, if the source has none) are carved
10% each from train, grouped by original photo so Roboflow '.rf.' augment copies stay together.
Polygons -> bbox; images downscaled to max side 832 (labels are normalized, unchanged)."""
import random, re, shutil
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
RAW = Path(__file__).parent / "det_data/raw_roboflow"; OUT = Path(__file__).parent / "det_data/leaf_combined"
SPLITS = {"train": "train", "valid": "val", "test": "test"}; MAX = 832
def orig(p): return re.sub(r"(_jpg|_jpeg|_png)?\.rf\..*$", "", p.stem)
def one(job):
    src, lab, split, stem = job
    lines = []
    for ln in Path(lab).read_text().split("\n"):
        t = ln.split()
        if len(t) < 5: continue
        v = list(map(float, t[1:]))
        if len(v) > 4:   # polygon
            xs, ys = v[0::2], v[1::2]; x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            v = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
        if v[2] * v[3] <= 1e-5: continue
        lines.append("0 " + " ".join(f"{min(max(a, 0), 1):.6f}" for a in v[:4]))
    if not lines: return 0
    try: im = Image.open(src).convert("RGB")
    except Exception: return 0
    if max(im.size) > MAX: im.thumbnail((MAX, MAX))
    im.save(OUT / "images" / split / f"{stem}.jpg", quality=92)
    (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    return len(lines)
if __name__ == "__main__":
    if OUT.exists(): shutil.rmtree(OUT)
    for s in SPLITS.values(): (OUT / "images" / s).mkdir(parents=True); (OUT / "labels" / s).mkdir(parents=True)
    jobs = []
    for ds in sorted(p for p in RAW.iterdir() if p.is_dir()):
        have = {rs: sorted((ds / rs / "images").glob("*")) for rs in SPLITS if (ds / rs / "images").exists()}
        assign = {i: SPLITS[rs] for rs, imgs in have.items() if rs != "train" for i in imgs}
        groups = sorted({orig(i) for i in have.get("train", [])}); random.Random(0).shuffle(groups)
        n10 = len(groups) // 10
        gs = {g: "val" for g in groups[:n10]}
        if "test" not in have: gs.update({g: "test" for g in groups[n10:2 * n10]})
        for i in have.get("train", []): assign[i] = gs.get(orig(i), "train")
        for img, s in assign.items():
            lab = img.parent.parent / "labels" / (img.stem + ".txt")
            if lab.exists(): jobs.append((str(img), str(lab), s, f"{ds.name}_{len(jobs):06d}"))
    with ProcessPoolExecutor(16) as ex: n = list(ex.map(one, jobs, chunksize=32))
    stats = {}
    for j, k in zip(jobs, n):
        a = stats.setdefault((j[3].split("_")[0], j[2]), [0, 0]); a[0] += bool(k); a[1] += k
    for k, v in sorted(stats.items()): print(k, "images", v[0], "boxes", v[1])
    (OUT / "data.yaml").write_text(f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames: ['leaf']\n")
