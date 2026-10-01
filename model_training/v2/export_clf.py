"""Export a trained classifier run to ONNX (fp32, simplified), check parity, CPU latency (4 threads, proxy only)."""
import json, sys, time
from pathlib import Path
import numpy as np, torch, timm, onnx, onnxruntime as ort
from onnxsim import simplify
run = Path(sys.argv[1]); bs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
meta = json.load(open(run / "meta.json")); res = json.load(open(run / "results.json"))
m = timm.create_model(meta["model"], pretrained=False, num_classes=len(res["classes"])).eval()
m.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
x = torch.randn(bs, 3, meta["img"], meta["img"]); out = run / f"classifier_b{bs}.onnx"
torch.onnx.export(m, x, out, input_names=["image"], output_names=["logits"], opset_version=17, dynamo=False)
mo, ok = simplify(onnx.load(out)); assert ok; onnx.save(mo, out)
so = ort.SessionOptions(); so.intra_op_num_threads = 4
s = ort.InferenceSession(str(out), so, providers=["CPUExecutionProvider"])
y = s.run(None, {"image": x.numpy()})[0]
with torch.no_grad(): yr = m(x).numpy()
for _ in range(5): s.run(None, {"image": x.numpy()})
t = time.perf_counter(); n = 30
for _ in range(n): s.run(None, {"image": x.numpy()})
info = dict(onnx=str(out), size_MB=out.stat().st_size / 1e6, max_abs_diff=float(np.abs(y - yr).max()),
            cpu4t_ms_per_batch=(time.perf_counter() - t) / n * 1000, batch=bs, img=meta["img"],
            temperature=res["temperature"], classes=res["classes"],
            preprocess="RGB, resize short side to img + center crop, /255, ImageNet mean/std, NCHW")
json.dump(info, open(run / f"export_b{bs}.json", "w"), indent=1); print(json.dumps(info, indent=1))
