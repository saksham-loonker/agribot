"""Run the shipped TFLite models, canonical preprocessing, and static-frame app replay.

This evaluates public files, not camera performance. Replayed frames are correlated
copies used to exercise consensus rules, never counted as independent samples.
"""
from __future__ import annotations
import argparse, concurrent.futures, functools, hashlib, importlib.metadata, json, math, os, platform, sys, time
from pathlib import Path
import importlib.util
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from ai_edge_litert.interpreter import Interpreter
from canonical_runtime import install as install_sparse_resize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'dist/model-evaluation-20261015'
ASSETS = ROOT / 'agribot_android_app/android/app/src/main/assets'
spec = importlib.util.spec_from_file_location('agribot_canonical', ROOT / 'model_training/v2/final/canon.py')
canon = importlib.util.module_from_spec(spec); spec.loader.exec_module(canon)
install_sparse_resize(canon)
MANIFEST = json.loads((ASSETS / 'model_manifest.json').read_text())
LABELS = MANIFEST['classifier']['labels']
OTHER = MANIFEST['classifier']['other_index']
HEALTHY = LABELS.index('Healthy')


def posterior(member_logits, bias=None):
    c = MANIFEST['classifier']
    mean = np.mean([canon.softmax(z) for z in member_logits], axis=0)
    combined = np.log(mean + 1e-12)
    combined[OTHER] += c['other_logit_bias'] if bias is None else bias
    return canon.softmax(combined, c['temperature']).astype(np.float32)


def quality(image):
    h, w = image.shape[:2]; stride = max(1, int(math.sqrt((h*w)//24000)))
    rgb = image[::stride,::stride].astype(np.float64)
    sample = .2126*rgb[:,:,0]+.7152*rgb[:,:,1]+.0722*rgb[:,:,2]
    brightness=float(sample.mean()/255); contrast=float(sample.std()/255)
    grad=np.abs(np.diff(sample,axis=0)).sum()+np.abs(np.diff(sample,axis=1)).sum()
    pairs=sample.shape[0]*max(0,sample.shape[1]-1)+sample.shape[1]*max(0,sample.shape[0]-1)
    sharpness=float(grad/max(1,pairs)/255)
    accept = not (brightness<.08 or brightness>.95 or (contrast<.015 and sharpness<.012) or (sharpness<.012 and contrast<.08))
    return dict(brightness=brightness, contrast=contrast, sharpness=sharpness, quality_accept=accept,
                quality_weight=min(1.,max(.4,.4+12*sharpness)) if accept else .3)


def replay(leaves, weight=1., bias=None, other_min_prob=.5):
    """Same static boxes across up to 14 frames: PlantEvidence + CheckPlantViewModel.

    A fixed image has fixed tracks; this does not simulate motion, tracking failures,
    real viewpoint diversity, thermal pacing, or changing leaf illumination.
    """
    if not leaves:
        return dict(kind='NO_LEAVES', label=None, confidence=None, frames=0, leaves=0, partial=False)
    ps=np.asarray([posterior(leaf['member_logits'], bias) for leaf in leaves], np.float64)
    obs_weights=np.clip(np.asarray([leaf['score'] for leaf in leaves], np.float32)*np.float32(weight),.05,1).astype(np.float64)
    last=None; stable=0
    for frame in range(1,15):
        weights=np.minimum(frame*obs_weights,3.)
        plant=np.sum(ps*weights[:,None],axis=0)/weights.sum()
        order=np.argsort(-plant,kind='stable'); top, second=map(int,order[:2])
        p1,p2=np.float32(plant[top]),np.float32(plant[second]); partial=False
        kind='NEED_MORE_VIEWS'; label=None; confidence=float(p1)
        if frame>=3 and frame*len(leaves)>=5:
            candidates=[]
            if frame>=2:
                for d in range(len(LABELS)):
                    if d in (HEALTHY,OTHER): continue
                    strong=ps[:,d]>=np.float32(.7)
                    if strong.sum()>=2: candidates.append((int(strong.sum()),float(ps[strong,d].mean()),d))
            best=max(candidates,key=lambda candidate:candidate[:2]) if candidates else None
            if best and top!=best[2] and (top in (HEALTHY,OTHER) or p1-p2<np.float32(.15)):
                kind='DISEASE'; label=best[2]; confidence=best[1]; partial=True
            else:
                label=top
                if top==OTHER: kind='UNKNOWN_CONDITION' if p1>=np.float32(other_min_prob) else 'UNSURE'
                elif top==HEALTHY: kind='HEALTHY' if p1>=np.float32(.60) and p1-p2>=np.float32(.20) else 'UNSURE'
                else: kind='DISEASE' if p1>=np.float32(.50) and p1-p2>=np.float32(.15) else 'UNSURE'
        key=(kind,label); stable=stable+1 if key==last else 0; last=key
        if kind!='NEED_MORE_VIEWS' and ((kind!='UNSURE' and stable>=2) or frame>=14):
            return dict(kind=kind,label=LABELS[label] if label is not None else None,confidence=confidence,
                        frames=frame,leaves=len(leaves),partial=partial,probabilities=plant.tolist(), margin=float(p1-p2))
    raise RuntimeError('Static replay failed to terminate')


class Models:
    def __init__(self, threads=4):
        self.members=[]
        for member in MANIFEST['classifier']['members']:
            self.members.append((self.load(member,threads),member['input'][1]))
        self.detector=self.load(MANIFEST['detector'],threads)

    @staticmethod
    def load(contract,threads):
        path=ASSETS/'models'/contract['file']
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=contract['sha256']: raise RuntimeError('Model SHA mismatch: '+str(path))
        interpreter=Interpreter(model_path=str(path),num_threads=threads)
        interpreter.allocate_tensors()
        if interpreter.get_input_details()[0]['shape'].tolist()!=contract['input']: raise RuntimeError('Input shape mismatch')
        return interpreter

    @staticmethod
    def invoke(interpreter, tensor):
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'],tensor)
        interpreter.invoke()
        output=interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
        if not np.isfinite(output).all(): raise RuntimeError('Non-finite model output')
        return output

    def classify(self,image,box=None):
        return [self.invoke(interpreter,canon.classifier_input(image,size,box)).ravel().tolist() for interpreter,size in self.members]

    def detect(self,image):
        contract=MANIFEST['detector']; tensor,lb=canon.detector_input(image,contract['input'][1])
        output=self.invoke(self.detector,tensor)
        return canon.decode_detections(output,lb,image.shape[1],image.shape[0],contract['score_threshold'],contract['nms_iou'],contract['max_detections'])


def golden_check(models):
    directory=ROOT/'model_training/v2/bundle/golden'
    expected=json.loads((directory/'golden_expected.json').read_text())
    rows=[]
    reference_spec=importlib.util.spec_from_file_location('dense_canonical_reference',ROOT/'model_training/v2/final/canon.py')
    reference=importlib.util.module_from_spec(reference_spec); reference_spec.loader.exec_module(reference)
    for entry in expected['entries']:
        image=np.asarray(Image.open(directory/entry['file']).convert('RGB'))
        logits=models.classify(image); target_whole=entry['whole_image']
        p=posterior(logits); max_diff=float(np.max(np.abs(np.asarray(logits)-np.asarray(target_whole['member_logits']))))
        prob_diff=float(np.max(np.abs(p-np.asarray(target_whole['probs']))))
        actual=models.detect(image); target=entry['detections']
        box_diff=max((max(abs(a-b) for a,b in zip(actual[i][:4],target[i][:4])) for i in range(min(len(actual),len(target)))), default=0.)
        tensor=canon.classifier_input(image,MANIFEST['classifier']['members'][0]['input'][1])
        dense_tensor=reference.classifier_input(image,MANIFEST['classifier']['members'][0]['input'][1])
        full_input_diff=float(np.max(np.abs(tensor-dense_tensor)))
        detector_tensor,_=canon.detector_input(image)
        dense_detector,_=reference.detector_input(image)
        full_detector_input_diff=float(np.max(np.abs(detector_tensor-dense_detector)))
        checks=entry['classifier_input_checks'][0]
        input_diff=max(abs(float(tensor.ravel()[int(i)])-v) for i,v in checks['samples'].items())
        whole_ok=max_diff<=.02 and LABELS[int(p.argmax())]==target_whole['label'] and prob_diff<=.002
        detector_ok=abs(len(actual)-len(target))<=1 and box_diff<=2.
        rows.append(dict(file=entry['file'],max_abs_logit_diff=max_diff,max_abs_probability_diff=prob_diff,
                         max_input_sample_diff=input_diff,max_full_classifier_input_diff=full_input_diff,max_full_detector_input_diff=full_detector_input_diff,
                         box_px_diff=box_diff,actual_detections=len(actual),expected_detections=len(target),
                         whole_pass=whole_ok,detector_pass=detector_ok))
        print('Golden:',entry['file'],'logit difference:',round(max_diff,6),'detector:',detector_ok,flush=True)
    (OUT/'golden_checks.json').write_text(json.dumps(rows,indent=2))
    if not all(r['whole_pass'] and r['detector_pass'] and r['max_input_sample_diff']<=1e-3 and r['max_full_classifier_input_diff']<=1e-3 and r['max_full_detector_input_diff']<=1e-3 for r in rows):
        raise RuntimeError('Golden parity failed; inspect golden_checks.json before continuing')
    return rows


def image_record(models,record):
    image=np.asarray(Image.open(ROOT/record['path']).convert('RGB'))
    info={k:v for k,v in record.items() if k not in ('objects',)}
    info['preprocessing_backend']='sparse_canonical_weights'
    info.update(quality(image)); info['image_phash']=None
    import imagehash
    info['image_phash']=str(imagehash.phash(Image.fromarray(image)))
    t=time.perf_counter(); whole=models.classify(image); info['whole_member_logits']=whole
    probs=posterior(whole); info['whole_probabilities']=probs.tolist(); info['whole_label']=LABELS[int(probs.argmax())]
    dets=models.detect(image); leaves=[]
    for box in dets[:6]:
        logits=models.classify(image,box[:4]); p=posterior(logits)
        leaves.append(dict(box=box[:4],score=box[4],member_logits=logits,probabilities=p.tolist(),label=LABELS[int(p.argmax())]))
    info['detections']=dets; info['leaves']=leaves
    info['app_replay']=replay(leaves,info['quality_weight'])
    info['host_elapsed_ms']=(time.perf_counter()-t)*1000
    oracle=[]
    for index,obj in enumerate(record.get('objects',[])):
        box=obj['box']; l,t_,r,b=canon.crop_box(image.shape[1],image.shape[0],box,0.)
        annotation_size=record.get('annotation_size')
        dimension_match=annotation_size is None or annotation_size==[image.shape[1],image.shape[0]]
        valid=dimension_match and r-l>=2 and b-t_>=2 and 0<=box[0]<box[2]<=image.shape[1]+1 and 0<=box[1]<box[3]<=image.shape[0]+1
        if not valid:
            oracle.append(dict(index=index,valid=False,source_label=obj['source_label'],label=obj['label'],box=box)); continue
        logits=models.classify(image,box); p=posterior(logits)
        oracle.append(dict(index=index,valid=True,source_label=obj['source_label'],label=obj['label'],box=box,
                           member_logits=logits,prediction=LABELS[int(p.argmax())],probabilities=p.tolist(),
                           best_detector_iou=max((canon.iou(box,d[:4]) for d in dets),default=0.),
                           best_selected_iou=max((canon.iou(box,d[:4]) for d in dets[:6]),default=0.)))
    info['oracle_objects']=oracle
    return info


WORKER_MODELS=None


def initialize_worker(threads):
    global WORKER_MODELS
    WORKER_MODELS=Models(threads)


def evaluate_one(record):
    try:
        return image_record(WORKER_MODELS,record)
    except Exception as exc:
        return dict(id=record['id'],error=repr(exc))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--golden-only',action='store_true'); parser.add_argument('--limit',type=int); parser.add_argument('--stress',action='store_true')
    parser.add_argument('--workers',type=int,default=1); parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    models=Models(args.threads); golden_check(models)
    if args.golden_only: return
    if args.stress:
        records=[json.loads(line) for line in (OUT/'predictions.jsonl').read_text().splitlines()]
        groups={}
        for row in records:
            if row['source']=='tomatovillage': groups.setdefault(row['label'],[]).append(row)
        target=[]
        for label,rows in sorted(groups.items()): target.extend(sorted(rows,key=lambda r:r['id'])[:3])
        with (OUT/'stress_predictions.jsonl').open('w') as stream:
            for row in target:
                source=Image.open(ROOT/row['path']).convert('RGB')
                for name,im in [('dark_50pct',ImageEnhance.Brightness(source).enhance(.5)),('blur_radius2',source.filter(ImageFilter.GaussianBlur(2))),('rotate90',source.transpose(Image.Transpose.ROTATE_90))]:
                    logits=models.classify(np.asarray(im)); p=posterior(logits)
                    stream.write(json.dumps(dict(id=row['id'],label=row['label'],transform=name,baseline=row['whole_label'],prediction=LABELS[int(p.argmax())],member_logits=logits,probabilities=p.tolist()))+'\n'); stream.flush()
                print('Stress:',row['label'],row['id'],flush=True)
        return
    records=json.loads((OUT/'images_manifest.json').read_text(encoding='utf-8'))
    if args.limit: records=records[:args.limit]
    output=OUT/'predictions.jsonl'
    done={json.loads(line)['id'] for line in output.read_text().splitlines()} if output.exists() else set()
    errors=[]
    pending=[record for record in records if record['id'] not in done]
    if args.workers>1:
        executor=concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,initializer=initialize_worker,initargs=(args.threads,))
        iterator=executor.map(evaluate_one,pending)
    else:
        initialize_worker(args.threads)
        iterator=map(evaluate_one,pending)
    try:
        with output.open('a',encoding='utf-8') as stream:
            for index,row in enumerate(iterator,len(done)+1):
                if 'error' in row:
                    errors.append(row); print('EVALUATION ERROR:',row,flush=True)
                else:
                    row['runtime_threads']=args.threads; row['evaluation_workers']=args.workers
                    stream.write(json.dumps(row)+'\n'); stream.flush()
                if index%10==0: print(f'Evaluated {index}/{len(records)}',row.get('source','error'),flush=True)
    finally:
        if args.workers>1: executor.shutdown()
    (OUT/'evaluation_errors.json').write_text(json.dumps(errors,indent=2))
    runtime=dict(python=sys.version,platform=platform.platform(),runtime=importlib.metadata.version('ai-edge-litert'),numpy=np.__version__,model_manifest=MANIFEST,threads_per_worker=args.threads,workers=args.workers,
                 note='Host CPU inference, canonical original-image preprocessing; app verdicts are repeated-identical-frame simulations, not camera trials.')
    (OUT/'runtime.json').write_text(json.dumps(runtime,indent=2))
    print('Evaluation complete; errors:',len(errors),flush=True)


if __name__=='__main__': main()
