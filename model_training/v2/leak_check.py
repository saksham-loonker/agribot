"""Near-duplicate leakage check across Variant-A splits (pHash, Hamming <= T)."""
import json, sys
from pathlib import Path
from collections import defaultdict
import imagehash, numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

ROOT = Path.home() / "sakshyam-agribot/archive/Variant-a(Multiclass Classification)"
T = int(sys.argv[1]) if len(sys.argv) > 1 else 4

def h(p):
    try:
        return str(p), imagehash.phash(Image.open(p).convert("RGB")).hash.flatten()
    except Exception:
        return str(p), None

paths = [p for p in ROOT.glob("*/*/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
with ProcessPoolExecutor(24) as ex:
    res = [r for r in ex.map(h, paths, chunksize=64) if r[1] is not None]
P = [r[0] for r in res]; H = np.stack([r[1] for r in res]).astype(np.uint8)
split = np.array([Path(p).parts[-3] for p in P]); cls = np.array([Path(p).parts[-2] for p in P])
D = (H[:, None, :] != H[None, :, :]).sum(-1)
np.fill_diagonal(D, 99)
out = defaultdict(int); conflicts = 0; flagged = set()
for i, j in zip(*np.where(np.triu(D <= T))):
    key = "-".join(sorted([split[i], split[j]]))
    out[key] += 1
    if cls[i] != cls[j]: conflicts += 1
    if split[i] != split[j]:
        for k in (i, j):
            if split[k] in ("val", "test"): flagged.add(P[k])
print("images", len(P), "threshold", T)
print("dup pairs by split:", dict(out))
print("label conflicts among dups:", conflicts)
print("val/test images with a near-dup in another split:", len(flagged))
for s in ("val", "test"):
    n = sum(1 for p in flagged if f"/{s}/" in p); tot = (split == s).sum()
    print(f"  {s}: {n}/{tot}")
json.dump(sorted(flagged), open(Path(__file__).parent / f"leaky_T{T}.json", "w"), indent=0)
