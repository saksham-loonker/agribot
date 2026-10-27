"""Independent LiteRT references for the existing, public Android golden PNGs."""

from pathlib import Path
import argparse
import json
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

from app_eval import Models, canon, classifier_letterbox_input, posterior, quality, replay
from permissive_encoder import digest
from qualified_recognition_policy import route


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', required=True)
    ap.add_argument('--goldens', required=True)
    ap.add_argument('--manifest', required=True)
    args = ap.parse_args()
    p, goldens = Path(args.campaign), Path(args.goldens)
    out = p / 'phase75_native_goldens'
    out.mkdir(exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    models = {}
    for name, contract in [('leaf', manifest['classifier']['members'][0]), ('whole', manifest['primary_recovery']['model'])]:
        path = p / 'phase72_mobile_export' / (name + '.tflite')
        assert digest(path) == contract['sha256']
        interpreter = Interpreter(model_path=str(path), num_threads=4)
        interpreter.allocate_tensors()
        models[name] = interpreter

    def run(name, pixels):
        return Models.invoke(models[name], pixels).ravel()

    entries = []
    original = json.loads((goldens / 'golden_expected.json').read_text())
    for entry in original['entries']:
        image_path = goldens / entry['file']
        image = np.asarray(Image.open(image_path).convert('RGB'))
        # Detector boxes are already frozen and tested independently in Android.
        # A different detection/order fails parity instead of substituting a ROI.
        boxes = entry['detections'][:6]
        leaves, views = [], []
        for box in boxes:
            center = run('leaf', canon.classifier_input(image, 224, box[:4]))
            full = run('leaf', classifier_letterbox_input(image, 224, box[:4]))
            leaves.append(dict(box=box[:4], score=box[4], member_logits=[center[:9].tolist()]))
            views.append([center[9:].tolist(), full[9:].tolist()])
        context = run('whole', classifier_letterbox_input(image, 224))
        whole_original = posterior([run('leaf', canon.classifier_input(image, 224))[:9]], manifest).tolist() if not leaves else None
        whole_adapted = None
        if context[0] >= .85 and context[2] >= .5 and context[3] >= .5 and sum(v[0][2] < .5 for v in views) >= 2:
            whole_adapted = float(run('leaf', classifier_letterbox_input(image, 224))[11])
        baseline = dict(quality(image), app_replay=replay(leaves, manifest), context_whole_reference_probabilities=whole_original)
        candidate = route(baseline, manifest, context, views, whole_adapted)
        entries.append(dict(file=entry['file'], file_sha256=digest(image_path), context=context.tolist(), leaves=views,
                            whole_adapted=whole_adapted, original_whole_probabilities=whole_original,
                            baseline=baseline['app_replay'], candidate=candidate['app_replay']))
    result = dict(schema=1, source_sha256=digest(__file__), manifest_sha256=digest(args.manifest),
                  tolerances=dict(auxiliary=0.002, logits=0.02, box_pixels=2),
                  note='Existing development PNGs; repeated frames test consensus and are never independent images.', entries=entries)
    target = out / 'primary_expected.json'
    if target.exists():
        assert json.loads(target.read_text()) == result, 'Golden identity changed; use a new frozen output directory'
    else:
        target.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(sha256=digest(target), verdicts=[(e['file'], e['baseline']['kind'], e['candidate']['kind']) for e in entries])))


if __name__ == '__main__':
    main()
