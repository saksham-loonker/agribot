"""Select the full-ROI guard on opened development/regressions, never reservation."""

from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import time
import copy

import numpy as np
from PIL import Image
import timm
import torch

from app_eval import classifier_letterbox_input
from campaign_reference import safe_logits
from phase2_export import restore
from phase2_policy import DomainPolicy
from phase2_select import evaluate, retain


def keys(logits):
    """Vectorized float32 leaf decision keys, including stable winner ties."""
    values = np.asarray(logits, np.float64).copy()
    values -= values.max(1, keepdims=True)
    probs = np.exp(values)
    probs /= probs.sum(1, keepdims=True)
    values = np.log(probs + 1e-12)
    values[:, 8] += .75
    values /= .8814700841903687
    values -= values.max(1, keepdims=True)
    probs = np.exp(values)
    probs = (probs / probs.sum(1, keepdims=True)).astype(np.float32)
    order = np.argsort(-probs, axis=1, kind="stable")
    top = order[:, 0]
    confidence = np.take_along_axis(probs, top[:, None], 1).ravel()
    margin = confidence - np.take_along_axis(probs, order[:, 1:2], 1).ravel()
    healthy = (top == 1) & (confidence >= np.float32(.6)) & (margin >= np.float32(.2))
    disease = (top != 1) & (top != 8) & (confidence >= np.float32(.5)) & (margin >= np.float32(.15))
    other = (top == 8) & (confidence >= np.float32(.5))
    kind = np.where(healthy, 1, np.where(disease, 2, np.where(other, 3, 0)))
    return kind * 9 + top, healthy | disease


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra-healthy-thresholds", default="")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timm.layers.set_fused_attn(False)
    rows_path = root / "cnn_inputs/rows.json"
    rows = json.loads(rows_path.read_text())
    ev = json.loads((root / "eval_rows.json").read_text())
    manifest = json.loads((root.parent / "original_reference/model_manifest.json").read_text())
    candidates = json.loads((root / "selection/ranked.json").read_text())
    base_candidates = list(candidates)
    for threshold in filter(None, a.extra_healthy_thresholds.split(',')):
        value = float(threshold)
        assert .95 <= value < 1., 'Only stricter confirmation in this follow-up'
        for old in base_candidates:
            if old['policy']['healthy_min'] != .95:
                continue
            item = copy.deepcopy(old)
            item['id'] = len(candidates)
            item['policy']['healthy_min'] = value
            candidates.append(item)
    model = restore(a.bundle).cuda().eval()
    reference = safe_logits(root)
    student = np.load(Path(candidates[0]["student_checkpoint"]).parent / "best_logits.npy")
    cnn = np.load(Path(candidates[0]["cnn_checkpoint"]).parent / "all_logits.npy")
    baseline = evaluate(reference, rows, ev, manifest)
    eligible = np.array([r['set'] in ev for r in rows])
    ref_keys, _ = keys(reference)
    changed = np.zeros(len(rows), bool)
    tensors = [torch.from_numpy(z).cuda() for z in (reference, student, cnn)]
    with torch.inference_mode():
        for c in candidates:
            proposed = DomainPolicy(**c['policy']).cuda()(*tensors).cpu().numpy()
            decision, positive = keys(proposed)
            changed |= eligible & positive & (decision != ref_keys)
    indices = np.flatnonzero(changed)
    identity = dict(rows_sha256=hashlib.sha256(rows_path.read_bytes()).hexdigest(),
                    bundle_sha256=hashlib.sha256(Path(a.bundle).read_bytes()).hexdigest(),
                    indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest())
    cache = out / 'full_roi_components.npz'
    identity_path = out / 'components_identity.json'
    started = time.time()
    if cache.exists():
        assert json.loads(identity_path.read_text()) == identity
        saved = np.load(cache)
        full = [saved[k] for k in ('reference', 'student', 'cnn')]
    else:
        full = [np.zeros_like(z) for z in (reference, student, cnn)]
        def prepare(i):
            image = np.asarray(Image.open(rows[i]['path']).convert('RGB'))
            return classifier_letterbox_input(image, 224, rows[i]['box'])
        with torch.inference_mode(), concurrent.futures.ThreadPoolExecutor(8) as pool:
            for start in range(0, len(indices), 64):
                ids = indices[start:start+64]
                pixels = np.concatenate(list(pool.map(prepare, ids)), 0)
                parts = model.components(torch.from_numpy(pixels).cuda())
                for dest, values in zip(full, parts):
                    dest[ids] = values.cpu().numpy()
                if start % 256 == 0:
                    print(json.dumps(dict(full_roi_crops=start, total=len(indices), seconds=time.time()-started)), flush=True)
        np.savez(cache, reference=full[0], student=full[1], cnn=full[2])
        identity_path.write_text(json.dumps(identity, indent=2))
    full_tensors = [torch.from_numpy(z).cuda() for z in full]
    rank = []
    best_final = None
    with torch.inference_mode():
        for c in candidates:
            policy = DomainPolicy(**c['policy']).cuda()
            proposed = policy(*tensors).cpu().numpy()
            alternative = policy(*full_tensors).cpu().numpy()
            decision, positive = keys(proposed)
            alt_keys, _ = keys(alternative)
            fallback = eligible & positive & (decision != ref_keys) & (decision != alt_keys)
            final = proposed.copy()
            final[fallback] = reference[fallback]
            metrics = evaluate(final, rows, ev, manifest)
            score = float(np.mean([(m['healthy']['correct_rate'] or 0) +
                (m['supported_disease']['correct_rate'] or 0) -
                3*(m['all']['false_healthy_rate'] or 0) -
                2*(m['unsupported_tomato']['false_known_rate'] or 0) for m in metrics.values()]))
            record = dict(c, metrics=metrics, score=score, retained=retain(metrics, baseline),
                          view_confirmation=True, fallback_crops=int(fallback.sum()))
            rank.append(record)
            if best_final is None or (record['retained'], score) > best_final[0]:
                best_final = ((record['retained'], score), final)
            print(json.dumps(dict(id=c['id'], retained=record['retained'], score=score,
                healthy=metrics['phase5_selection']['healthy'], false_healthy=metrics['phase5_selection']['all']['false_healthy'])), flush=True)
    rank.sort(key=lambda c:(c['retained'],c['score']), reverse=True)
    (out/'baseline.json').write_text(json.dumps(baseline, indent=2))
    (out/'ranked.json').write_text(json.dumps(rank, indent=2))
    (out/'candidate.json').write_text(json.dumps(rank[0], indent=2))
    np.save(out/'guarded_logits.npy', best_final[1])
    print(json.dumps(dict(best_id=rank[0]['id'], passed=rank[0]['retained'], confirmations=len(indices))), flush=True)


if __name__ == '__main__':
    main()
