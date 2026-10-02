"""Test-split evaluation of leaf detectors: combined / per-source, at 416 and 320 input."""
import json, contextlib, io
from pathlib import Path
from libreyolo import LibreYOLO9
HERE = Path(__file__).parent; D = HERE / "det_data/leaf_combined"; E = HERE / "eval"
imgs = sorted((D / "images/test").glob("*.jpg"))
yamls = {"all": D / "data.yaml"}
for src in ("fieldplant", "plantdoc"):
    lst = E / f"test_{src}.txt"; lst.write_text("\n".join(str(p) for p in imgs if p.name.startswith(src)) + "\n")
    y = E / f"data_{src}.yaml"
    y.write_text(f"path: {D}\ntrain: images/train\nval: {lst}\ntest: {lst}\nnc: 1\nnames: ['leaf']\n"); yamls[src] = y
out = {}
for run, size in (("y9s_416_comb", "s"), ("y9t_416_comb", "t"), ("y9t_416", "t")):
    m = LibreYOLO9(str(HERE / f"det_runs/{run}/weights/best.pt"), size=size, nb_classes=1)
    for sz in (416, 320):
        for k, y in yamls.items():
            with contextlib.redirect_stdout(io.StringIO()):
                r = m.val(data=str(y), split="test", imgsz=sz, batch=32, verbose=False, project=str(E / "det_val"), name=f"{run}_{sz}_{k}", exist_ok=True)
            out[f"{run}@{sz}/{k}"] = {kk: round(float(r[kk]), 4) for kk in ("metrics/mAP50", "metrics/mAP50-95", "metrics/precision", "metrics/recall", "metrics/best_conf", "metrics/best_conf_f1") if kk in r}
            print(f"{run}@{sz}/{k}", out[f"{run}@{sz}/{k}"], flush=True)
json.dump(out, open(E / "det_test.json", "w"), indent=1)
