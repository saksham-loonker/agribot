"""Canonical Agribot preprocessing / decoding. The Android app implements EXACTLY this maths (ml/preprocessing).
resize: separable; per axis, scale s = n_in / n_out. s > 1 -> area average (box filter over [i*s, (i+1)*s));
        s <= 1 -> bilinear with half-pixel centres (x = (i + 0.5) * s - 0.5, clamped to [0, n_in - 1]).
classifier: crop box (+pad, clamped, floor/ceil to ints) -> centred square of the short side -> resize to S x S
            -> /255 -> (x - mean) / std, NHWC float32.
detector:   letterbox to 320: k = min(320/w, 320/h), nw = round(w*k), nh = round(h*k), resize, pad 114 centred
            (px = (320 - nw) // 2, py = (320 - nh) // 2) -> /255, NHWC float32.
            output [1, 5, N]: x1, y1, x2, y2 (letterboxed px), score in [0,1]; greedy class-agnostic NMS."""
import numpy as np
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
def axis_weights(n_in, n_out):
    W = np.zeros((n_out, n_in), np.float64); s = n_in / n_out
    for i in range(n_out):
        if s > 1:
            a, b = i * s, (i + 1) * s
            for j in range(int(np.floor(a)), min(n_in, int(np.ceil(b)))):
                W[i, j] = (min(b, j + 1) - max(a, j)) / s
        else:
            x = min(max((i + 0.5) * s - 0.5, 0.0), n_in - 1); j0 = int(np.floor(x)); j1 = min(j0 + 1, n_in - 1); f = x - j0
            W[i, j0] += 1 - f; W[i, j1] += f
    return W.astype(np.float32)
def resize(img, out_w, out_h):
    """img HxWx3 uint8 -> out_h x out_w x 3 float32 (0..255)."""
    h, w = img.shape[:2]; Wy, Wx = axis_weights(h, out_h), axis_weights(w, out_w); x = img.astype(np.float32)
    return np.einsum("ow,hwc->hoc", Wx, np.einsum("oh,hwc->owc", Wy, x))
def crop_box(w, h, box, pad):
    x1, x2 = sorted((box[0], box[2])); y1, y2 = sorted((box[1], box[3])); bw, bh = x2 - x1, y2 - y1
    l = max(0, int(np.floor(x1 - pad * bw))); t = max(0, int(np.floor(y1 - pad * bh)))
    r = min(w, int(np.ceil(x2 + pad * bw))); b = min(h, int(np.ceil(y2 + pad * bh)))
    return l, t, r, b
def classifier_input(img, size, box=None, pad=0.10):
    h, w = img.shape[:2]
    if box is not None:
        l, t, r, b = crop_box(w, h, box, pad); img = img[t:b, l:r]; h, w = img.shape[:2]
    side = min(w, h); ox, oy = (w - side) // 2, (h - side) // 2
    x = resize(img[oy:oy + side, ox:ox + side], size, size) / 255.0
    return ((x - MEAN) / STD).astype(np.float32)[None]
def detector_input(img, size=320):
    h, w = img.shape[:2]; k = min(size / w, size / h); nw, nh = int(round(w * k)), int(round(h * k))
    px, py = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), 114.0, np.float32); out[py:py + nh, px:px + nw] = resize(img, nw, nh)
    return (out / 255.0).astype(np.float32)[None], (k, px, py)
def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0
def decode_detections(out, lb, w, h, conf=0.35, iou_thr=0.6, max_det=16):
    """out: [1, 5, N] -> list of (x1, y1, x2, y2, score) in original image px, sorted by score."""
    o = out[0]; k, px, py = lb; keep = np.where(o[4] >= conf)[0]; keep = keep[np.argsort(-o[4, keep], kind="stable")]
    sel = []
    for i in keep:
        b = [float(np.clip((o[0, i] - px) / k, 0, w)), float(np.clip((o[1, i] - py) / k, 0, h)),
             float(np.clip((o[2, i] - px) / k, 0, w)), float(np.clip((o[3, i] - py) / k, 0, h)), float(o[4, i])]
        if b[2] - b[0] < 2 or b[3] - b[1] < 2: continue
        if all(iou(b, s) <= iou_thr for s in sel): sel.append(b)
        if len(sel) >= max_det: break
    return sel
def softmax(z, T=1.0):
    z = np.asarray(z, np.float64) / T; z = z - z.max(); e = np.exp(z); return e / e.sum()
