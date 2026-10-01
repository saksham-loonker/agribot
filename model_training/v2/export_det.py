"""Export leaf detectors to ONNX (no NMS in graph; NMS done in-app) and time CPU 4-thread inference (proxy)."""
import json, time, shutil
from pathlib import Path
import numpy as np, onnxruntime as ort
from libreyolo import LibreYOLO9
HERE = Path(__file__).parent; OUT = HERE / "export"; info = {}
for run, size in (("y9t_416_comb", "t"), ("y9s_416_comb", "s")):
    for sz in (320, 416):
        m = LibreYOLO9(str(HERE / f"det_runs/{run}/weights/best.pt"), size=size, nb_classes=1)
        p = Path(m.export(format="onnx", imgsz=sz, simplify=True, output_path=str(OUT / f"leaf_{run}_{sz}.onnx")))
        so = ort.SessionOptions(); so.intra_op_num_threads = 4
        s = ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"]); i = s.get_inputs()[0]
        x = np.random.rand(1, 3, sz, sz).astype(np.float32)
        for _ in range(5): o = s.run(None, {i.name: x})
        t = time.perf_counter(); n = 50
        for _ in range(n): s.run(None, {i.name: x})
        info[f"{run}@{sz}"] = dict(onnx=str(p), size_MB=round(p.stat().st_size / 1e6, 2), input=[i.name, i.shape], outputs=[(a.name, a.shape) for a in s.get_outputs()],
                                   cpu4t_ms=round((time.perf_counter() - t) / n * 1000, 2))
        print(f"{run}@{sz}", info[f"{run}@{sz}"], flush=True)
json.dump(info, open(OUT / "det_export.json", "w"), indent=1)
