"""PlantDoc VOC xml -> single-class 'leaf' YOLO dataset (train/val from TRAIN, test = TEST)."""
import random, shutil, xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
SRC = Path(__file__).parent / "det_data/plantdoc_src"; OUT = Path(__file__).parent / "det_data/leaf_yolo"
random.seed(0); stats = {}
def convert(xmlp, split):
    img = xmlp.with_suffix(".jpg")
    if not img.exists(): return 0
    try: W, H = Image.open(img).size
    except Exception: return 0
    lines = []
    for o in ET.parse(xmlp).getroot().iter("object"):
        b = o.find("bndbox"); x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = max(0, min(x1, x2)), min(W, max(x1, x2)); y1, y2 = max(0, min(y1, y2)), min(H, max(y1, y2))
        if x2 - x1 < 2 or y2 - y1 < 2: continue
        lines.append(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
    if not lines: return 0
    stem = f"{split}_{abs(hash(img.name)) % 10**8}_{img.stem[:40]}".replace(" ", "_")
    shutil.copy(img, OUT / "images" / split / f"{stem}.jpg")
    (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    return len(lines)
if OUT.exists(): shutil.rmtree(OUT)
for s in ("train", "val", "test"): (OUT / "images" / s).mkdir(parents=True); (OUT / "labels" / s).mkdir(parents=True)
tr = sorted((SRC / "TRAIN").glob("*.xml")); random.shuffle(tr); nv = len(tr) // 10
for split, xs in (("val", tr[:nv]), ("train", tr[nv:]), ("test", sorted((SRC / "TEST").glob("*.xml")))):
    n = [convert(x, split) for x in xs]; stats[split] = (sum(1 for k in n if k), sum(n))
(OUT / "data.yaml").write_text(f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames: ['leaf']\n")
print("images/boxes per split:", stats)
