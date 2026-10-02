"""Single-class leaf detector: LibreYOLO YOLOv9 (MIT code + weights), COCO-pretrained transfer."""
import argparse, json, time
from pathlib import Path
from libreyolo import LibreYOLO9
HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--size", default="t"); ap.add_argument("--imgsz", type=int, default=416)
ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--workers", type=int, default=6); ap.add_argument("--data", default=str(HERE / "det_data/leaf_yolo/data.yaml"))
ap.add_argument("--name", required=True)
a = ap.parse_args(); t0 = time.time()
m = LibreYOLO9(None, size=a.size, nb_classes=1)
res = m.train(data=a.data, epochs=a.epochs, batch=a.batch, imgsz=a.imgsz, optimizer="AdamW", lr0=1e-3,
              workers=a.workers, project=str(HERE / "det_runs"), name=a.name, exist_ok=True, pretrained=True,
              amp_dtype="bfloat16", patience=30, eval_interval=5)
res = {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in res.items()}
res.update(size=a.size, imgsz=a.imgsz, train_minutes=(time.time() - t0) / 60)
json.dump(res, open(HERE / "det_runs" / a.name / "summary.json", "w"), indent=1); print(json.dumps(res, indent=1))
