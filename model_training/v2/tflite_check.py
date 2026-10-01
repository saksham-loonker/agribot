"""TFLite parity vs ONNX (random input) + fp16 classifier accuracy on Variant-A test; CPU 4-thread latency (proxy)."""
import json, time
from pathlib import Path
import numpy as np, onnxruntime as ort, tensorflow as tf
from PIL import Image
HERE = Path(__file__).parent; X = HERE / "export"; out = {}
VA = Path.home() / "sakshyam-agribot/archive/Variant-a(Multiclass Classification)/test"
def tfl(path):
    it = tf.lite.Interpreter(model_path=str(path), num_threads=4); it.allocate_tensors(); return it
def run(it, x):
    it.set_tensor(it.get_input_details()[0]["index"], x); it.invoke(); return it.get_tensor(it.get_output_details()[0]["index"])
for name, sz in (("clf_mnv4m_kd1_384", 384), ("leaf_y9t_416_comb_320", 320), ("leaf_y9s_416_comb_320", 320)):
    s = ort.InferenceSession(str(X / f"{name}.onnx"), providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, sz, sz).astype(np.float32); ref = s.run(None, {s.get_inputs()[0].name: x})[0]
    for prec in ("float32",):
        p = X / f"tflite_{name}" / f"{name}_{prec}.tflite"; it = tfl(p); xh = x.transpose(0, 2, 3, 1).copy()
        y = run(it, xh)
        if y.shape != ref.shape and y.ndim == 3: y = y.transpose(0, 2, 1)
        for _ in range(3): run(it, xh)
        t = time.perf_counter()
        for _ in range(20): run(it, xh)
        r = dict(size_MB=round(p.stat().st_size / 1e6, 2), out_shape=list(y.shape), max_abs_diff=float(np.abs(y - ref).max()) if y.shape == ref.shape else "shape mismatch",
                 rel_diff=float(np.abs(y - ref).max() / (np.abs(ref).max() + 1e-9)) if y.shape == ref.shape else None,
                 cpu4t_ms=round((time.perf_counter() - t) / 20 * 1000, 2))
        if name.startswith("clf"):
            M = np.array([0.485, 0.456, 0.406]); S = np.array([0.229, 0.224, 0.225]); classes = sorted(d.name for d in VA.iterdir() if d.is_dir()); ok = n = 0
            for ci, c in enumerate(classes):
                for f in sorted((VA / c).iterdir()):
                    im = Image.open(f).convert("RGB").resize((sz, sz), Image.BILINEAR)   # 256x256 square -> resize == short-side resize + center crop
                    a = ((np.asarray(im, dtype=np.float32) / 255 - M) / S).astype(np.float32)[None]
                    ok += int(run(it, a).argmax() == ci); n += 1
            r["variantA_test_acc"] = round(ok / n, 4)
        out[f"{name}/{prec}"] = r; print(f"{name}/{prec}", r, flush=True)
json.dump(out, open(X / "tflite_check.json", "w"), indent=1)
