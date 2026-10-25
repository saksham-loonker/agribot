"""Ablate full-ROI confirmation of changed decisions on opened regressions only."""
from pathlib import Path
import argparse,copy,json
import numpy as np
import torch
from PIL import Image
from app_eval import Models,posterior,summarize,replay
from campaign_cnn_data import letterbox_classifier
from phase2_select import evaluate,retain
from phase2_accept import metrics


def leaf_decision(logits,manifest):
    p=posterior([np.asarray(logits)],manifest);order=np.argsort(-p,kind='stable');top,second=map(int,order[:2]);confidence=p[top];margin=confidence-p[second]
    if top==1:return ('HEALTHY',top) if confidence>=np.float32(.6) and margin>=np.float32(.2) else ('UNSURE',top)
    if top==8:return ('UNKNOWN_CONDITION',top) if confidence>=np.float32(.5) else ('UNSURE',top)
    return ('DISEASE',top) if confidence>=np.float32(.5) and margin>=np.float32(.15) else ('UNSURE',top)


def needs_confirmation(reference,candidate,manifest):
    old=leaf_decision(reference,manifest);new=leaf_decision(candidate,manifest)
    return new[0] in ('HEALTHY','DISEASE') and old!=new


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--assets',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();q=Path(a.root);assets=Path(a.assets);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);model=Models(assets,4);manifest=model.manifest;runtime=model.classifiers[0] if hasattr(model,'classifiers') else None
    # Use the interpreter directly so this ablation changes only the image view.
    from ai_edge_litert.interpreter import Interpreter
    runtime=Interpreter(model_path=str(assets/'models'/manifest['classifier']['members'][0]['file']),num_threads=4);runtime.allocate_tensors()
    def alternative(path,box):
        image=np.asarray(Image.open(path).convert('RGB'));tensor=np.ascontiguousarray(letterbox_classifier(image,box)[None]);runtime.set_tensor(runtime.get_input_details()[0]['index'],tensor);runtime.invoke();return runtime.get_tensor(runtime.get_output_details()[0]['index'])[0]
    rows=json.loads((q/'cnn_inputs/rows.json').read_text());ev=json.loads((q/'eval_rows.json').read_text());reference=torch.load(q/'healthy_verifier/features.pt',weights_only=True)['safe_logits'].numpy();candidate=np.load(assets/'regressions/actual_logits.npy');guarded=candidate.copy();checks=[]
    for i,r in enumerate(rows):
        if r['set'] not in ev or not needs_confirmation(reference[i],candidate[i],manifest):continue
        alt=alternative(r['path'],r['box']);confirmed=leaf_decision(candidate[i],manifest)==leaf_decision(alt,manifest)
        if not confirmed:guarded[i]=reference[i]
        checks.append(dict(index=i,id=r['id'],confirmed=confirmed,center=leaf_decision(candidate[i],manifest),full_roi=leaf_decision(alt,manifest)))
    baseline=evaluate(reference,rows,ev,manifest);observed=evaluate(guarded,rows,ev,manifest);np.save(out/'guarded_logits.npy',guarded)
    selected=assets.parent/'opened_slif';original=q/'slif_evaluation/baseline';base_rows={r['id']:r for r in [json.loads(l) for l in (original/'predictions.jsonl').read_text().splitlines()]};slif=[];slif_checks=[]
    for line in (selected/'predictions.jsonl').read_text().splitlines():
        row=json.loads(line);base=base_rows[row['id']];assert len(row['leaves'])==len(base['leaves'])
        for i,leaf in enumerate(row['leaves']):
            rz=np.asarray(base['leaves'][i]['member_logits'][0]);cz=np.asarray(leaf['member_logits'][0])
            if not needs_confirmation(rz,cz,manifest):continue
            alt=alternative(row['path'],leaf['box']);confirmed=leaf_decision(cz,manifest)==leaf_decision(alt,manifest)
            if not confirmed:leaf['member_logits']=[rz.tolist()]
            slif_checks.append(dict(id=row['id'],leaf=i,confirmed=confirmed))
        row['app_replay']=replay(row['leaves'],manifest,row['quality_weight'])
        if row['contrast']<.015 and row['sharpness']<.012:row['app_replay']=dict(kind='RETRY_QUALITY',label=None)
        slif.append(row)
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in slif));slif_metrics=metrics(out);slif_baseline=metrics(original)
    result=dict(assets=str(assets),validation=observed,baseline=baseline,validation_passed=retain(observed,baseline),slif=slif_metrics,slif_baseline=slif_baseline,slif_passed=retain(dict(slif=slif_metrics),dict(slif=slif_baseline)),checks=checks,slif_checks=slif_checks,note='Opened regressions only. Full-ROI classifier letterbox confirms changed positive decisions; otherwise original 2.0.2 logits are retained. No fresh acceptance photos evaluated.')
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(validation_passed=result['validation_passed'],slif_passed=result['slif_passed'],new_healthy=observed['phase2_selection']['healthy']['correct'],public_healthy=observed['public']['healthy']['correct'],slif_healthy=slif_metrics['healthy']['correct'],slif_false_healthy=slif_metrics['all']['false_healthy'],slif_false_known=slif_metrics['unsupported_tomato']['false_known'],confirmations=len(checks))),flush=True)


if __name__=='__main__':main()
