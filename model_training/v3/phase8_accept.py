"""Freeze the selected prototype and open the prospective reservation once.

This is a PyTorch prototype acceptance check, not permission to install it.
Passing still requires exported-byte replay and native integration/parity.
Failure consumes the reservation for further tuning and blocks promotion.
"""

from pathlib import Path
import argparse
import copy
import hashlib
import json
import math

import numpy as np
from PIL import Image
import timm
import torch

from app_eval import Models, classifier_letterbox_input, image_record, posterior, replay
from campaign_reference import safe_logits
from context_abstention import route_context
from phase2_export import restore as restore_classifier
from phase2_policy import DomainPolicy
from phase2_select import retain
from phase5_scene_ablation import photo_metrics
from phase7_feature_heads import BinaryHead
from phase7_verifier import restore as restore_verifier


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Prototype(Models):
    def __init__(self,root,candidate):
        super().__init__(root.parent/'export_cnn_safe',threads=2)
        self.domain=restore_classifier(root/'export_initial/pytorch_bundle.pt').cuda().eval()
        self.domain.policy=DomainPolicy(**candidate['leaf_candidate']['policy']).cuda().eval()
        self.scene=restore_verifier(root/candidate['fusion']['scene_run']).cuda().eval()
        self.species=restore_verifier(root/'species').cuda().eval()
        self.heads=[]
        for kind in ['linear_adapted','mlp_both']:
            directory=root/'feature_heads/healthy_scene'/kind
            state=torch.load(directory/'best.pt',weights_only=True,map_location='cpu')
            head=BinaryHead(kind).cuda().eval();head.load_state_dict(state['model'])
            self.heads.append((head,json.loads((directory/'result.json').read_text())['calibration']))
        self.candidate=candidate
        self.reference_by_box={}

    def classify(self,image,box=None):
        from view_guard_policy import decision_key,needs_confirmation
        from app_eval import canon
        tensor=torch.from_numpy(canon.classifier_input(image,224,box)).cuda()
        reference,student,cnn=self.domain.components(tensor)
        candidate=self.domain.policy(reference,student,cnn)[0].cpu().numpy()
        reference=reference[0].cpu().numpy()
        if needs_confirmation(posterior([reference],self.manifest),posterior([candidate],self.manifest),confirm_unknown=True):
            full=torch.from_numpy(classifier_letterbox_input(image,224,box)).cuda()
            r,s,c=self.domain.components(full)
            alternative=self.domain.policy(r,s,c)[0].cpu().numpy()
            if decision_key(posterior([candidate],self.manifest))!=decision_key(posterior([alternative],self.manifest)):
                candidate=reference
        self.reference_by_box[tuple(box) if box is not None else None]=[reference.tolist()]
        return [candidate.tolist()]

    def record(self,record):
        self.reference_by_box={}
        row=image_record(self,record)
        reference=dict(row);reference['leaves']=copy.deepcopy(row['leaves'])
        for leaf in reference['leaves']:
            leaf['member_logits']=self.reference_by_box[tuple(leaf['box'])]
            p=posterior(leaf['member_logits'],self.manifest)
            leaf['probabilities']=p.tolist();leaf['label']=self.manifest['classifier']['labels'][int(p.argmax())]
        reference['app_replay']=replay(reference['leaves'],self.manifest,row['quality_weight'])
        image=np.asarray(Image.open(record['path']).convert('RGB'))
        x=torch.from_numpy(classifier_letterbox_input(image,224)).cuda()
        *_,original,adapted=self.domain.components(x,return_features=True)
        features=torch.cat([original,adapted],1)
        z=[float((head(features)*cal[0]+cal[1]).item()) for head,cal in self.heads]
        z.extend([float(self.scene.logits(x).item()),float(self.species.logits(x).item())])
        fit=self.candidate['fusion']
        h=float(1/(1+np.exp(-np.clip(np.clip(z,-fit['input_clip'],fit['input_clip'])@np.array(fit['weights'])+fit['bias'],-80,80))))
        species=float(1/(1+np.exp(-np.clip(z[-1],-80,80))))
        def named(r):return any(max(posterior(leaf['member_logits'],self.manifest)[j] for j in range(9) if j not in (1,8))>=.7 for leaf in r['leaves'])
        revised=route_context(row,reference,self.manifest,h>=self.candidate['healthy_min'] and species>=self.candidate['species_min'],
            bool(self.candidate['negative_healthy_max'] and h<=self.candidate['negative_healthy_max']),h,named(row),named(reference))
        revised['context_audit']=dict(calibrated_logits=z,healthy_probability=h,species_probability=species)
        return revised


def wilson(k,n):
    if not n:return None
    p=k/n;z=1.959963984540054;d=1+z*z/n
    c=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.,c-half),min(1.,c+half)]


def verify_development(root,candidate,out):
    """Verify executable prototype against frozen cached selection before opening."""
    from campaign_heads import app_evaluate
    model=Prototype(root,candidate)
    rows=json.loads((root/'cnn_inputs/rows.json').read_text())
    ev=json.loads((root/'eval_rows.json').read_text())
    original={k:v['rows'] for k,v in app_evaluate(safe_logits(root),rows,ev,model.manifest,True).items()}
    adapted={k:v['rows'] for k,v in app_evaluate(np.load(root/'guard_selection/guarded_logits.npy'),rows,ev,model.manifest,True).items()}
    base={r['path']:r for values in original.values() for r in values}
    photos={r['path']:r for values in adapted.values() for r in values}
    paths=sorted(photos);lookup={p:i for i,p in enumerate(paths)}
    directory=root/'scene_focal_n4_selection'
    identity=json.loads((directory/'input_identity.json').read_text())
    assert identity['paths_sha256']==hashlib.sha256(json.dumps(paths).encode()).hexdigest()
    assert identity['input_identity']==candidate['fusion']['input_identity']
    logits=np.load(directory/'evaluation_logits.npy');fit=candidate['fusion']
    hp=1/(1+np.exp(-np.clip(np.clip(logits,-fit['input_clip'],fit['input_clip'])@np.array(fit['weights'])+fit['bias'],-80,80)))
    tp=1/(1+np.exp(-np.clip(logits[:,-1],-80,80)))
    ids=set(np.linspace(0,len(paths)-1,48).astype(int).tolist())
    for threshold in [candidate['healthy_min'],candidate['negative_healthy_max']]:ids.update(np.argsort(abs(hp-threshold))[:24].tolist())
    ids.update(np.argsort(abs(tp-candidate['species_min']))[:16].tolist())
    def named(r):return any(max(posterior(leaf['member_logits'],model.manifest)[j] for j in range(9) if j not in (1,8))>=.7 for leaf in r['leaves'])
    results=[]
    with torch.inference_mode():
        for i in sorted(ids):
            path=paths[i];row=photos[path];old=base[path]
            expected=route_context(row,old,model.manifest,bool(hp[i]>=candidate['healthy_min'] and tp[i]>=candidate['species_min']),bool(hp[i]<=candidate['negative_healthy_max']),float(hp[i]),named(row),named(old))
            observed=model.record(row)
            e=expected['app_replay'];a=observed['app_replay']
            error=float(np.max(abs(np.array(observed['context_audit']['calibrated_logits'])-logits[i])))
            results.append(dict(path=path,verdict_match=(e['kind'],e['label'],e.get('scope'))==(a['kind'],a['label'],a.get('scope')),context_logit_max_error=error,expected=e,actual=a))
    report=dict(passed=all(r['verdict_match'] and r['context_logit_max_error']<.002 for r in results),cases=results)
    (out/'development_prototype_parity.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(passed=report['passed'],cases=len(results),max_context_logit_error=max(r['context_logit_max_error'] for r in results))),flush=True)
    assert report['passed'],'Prototype differs from frozen selection; do not open reservation'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--selection-run',default='scene_focal_n4_selection');ap.add_argument('--freeze-only',action='store_true');ap.add_argument('--verify-development',action='store_true');a=ap.parse_args()
    root=Path(a.root);out=root/'phase8_acceptance';out.mkdir(exist_ok=True)
    path=root/a.selection_run/'candidate.json';candidate=json.loads(path.read_text())
    assert candidate['retained'] and candidate['field_healthy_passed'] and candidate.get('leaf_healthy_min',0.)==0.
    assert candidate['leaf_candidate']['confirm_unknown']
    fit=candidate['fusion'];identity=fit['input_identity'];head_paths=[root/'feature_heads/healthy_scene'/k for k in ['linear_adapted','mlp_both']]
    assert identity['feature_model_sha256']==digest(root/'export_initial/pytorch_bundle.pt')
    assert identity['scene_cnn_sha256']==digest(root/fit['scene_run']/'best.pt')
    assert identity['scene_cnn_calibration_sha256']==digest(root/fit['scene_run']/'result.json')
    assert identity['species_cnn_sha256']==digest(root/'species/best.pt')
    assert identity['species_cnn_calibration_sha256']==digest(root/'species/result.json')
    assert identity['head_hashes']==[digest(d/'best.pt') for d in head_paths]
    assert identity['head_calibration_hashes']==[digest(d/'result.json') for d in head_paths]
    manifest_path=root/'reserved_manifest.json';records=json.loads(manifest_path.read_text())
    assert len(records)==246 and digest(manifest_path)=='ca91cf80b05eab1c733a54746542e8fe37bee503c7c0fe03a97022bcdd0f5dd1'
    protocol=dict(stage='PyTorch prototype, before exported-byte/native verification',candidate_sha256=digest(path),candidate_id=candidate['id'],input_identity=identity,
        reserved_manifest_sha256=digest(manifest_path),reserved_n=len(records),healthy_recall_min=.7,healthy_gain_min=.05,paired_capture_group_healthy_gain_bootstrap_lower_95_min_exclusive=0.,bootstrap_seed=239,bootstrap_samples=5000,
        per_source_relative_retention='phase2_select.retain unchanged: integer-floor allowances, no Healthy decrease, <=1pp named disease loss, <=0.5pp false Healthy/Other/known increases',
        field_scope='Same-source prospectively reserved inferred capture families, not independently adjudicated whole-plant diagnoses',
        rule='One opening; failure blocks promotion. Never tune on these predictions and call the reservation independent again.',
        baseline_classifier_sha256=json.loads((root.parent/'export_cnn_safe/model_manifest.json').read_text())['classifier']['members'][0]['sha256'])
    frozen=out/'frozen_protocol.json'
    if frozen.exists():
        if json.loads(frozen.read_text())!=protocol:
            assert not (out/'opened_identity.json').exists(),'Acceptance identity changed after opening'
            frozen.write_text(json.dumps(protocol,indent=2))
    else:frozen.write_text(json.dumps(protocol,indent=2))
    (out/'frozen_candidate.json').write_text(json.dumps(candidate,indent=2))
    if a.freeze_only:print(json.dumps(protocol),flush=True);return
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;timm.layers.set_fused_attn(False)
    if a.verify_development:verify_development(root,candidate,out);return
    parity=json.loads((out/'development_prototype_parity.json').read_text());assert parity['passed']
    execution=dict(protocol_sha256=digest(frozen),script_sha256=digest(__file__),routing_sha256=digest(Path(__file__).with_name('context_abstention.py')),app_eval_sha256=digest(Path(__file__).with_name('app_eval.py')))
    started=out/'opened_identity.json'
    if started.exists():assert json.loads(started.read_text())==execution,'Opened acceptance implementation changed'
    else:started.write_text(json.dumps(execution,indent=2))
    if (out/'result.json').exists():print((out/'result.json').read_text());return
    baseline=Models(root.parent/'export_cnn_safe',threads=2);model=Prototype(root,candidate)
    predictions=out/'predictions.jsonl';done={}
    if predictions.exists():
        for line in predictions.read_text().splitlines():
            r=json.loads(line);done[r['id']]=r
    with predictions.open('a') as stream,torch.inference_mode():
        for index,record in enumerate(records):
            if record['id'] in done:continue
            assert digest(record['path'])==record['file_sha256']
            r=dict(id=record['id'],baseline=image_record(baseline,record),candidate=model.record(record))
            stream.write(json.dumps(r)+'\n');stream.flush();done[record['id']]=r
            if index%20==0:print(json.dumps(dict(evaluated=index+1,total=len(records))),flush=True)
    before=[done[r['id']]['baseline'] for r in records];after=[done[r['id']]['candidate'] for r in records]
    sources=sorted({r['source'] for r in records})
    b={s:photo_metrics([r for r in before if r['source']==s]) for s in sources};c={s:photo_metrics([r for r in after if r['source']==s]) for s in sources}
    healthy=[i for i,r in enumerate(records) if r['label']=='Healthy'];gains=np.array([int(after[i]['app_replay']['kind']=='HEALTHY')-int(before[i]['app_replay']['kind']=='HEALTHY') for i in healthy])
    groups=sorted({records[i]['group'] for i in healthy});group_values=[gains[[j for j,i in enumerate(healthy) if records[i]['group']==g]] for g in groups]
    rng=np.random.default_rng(239);bootstrap=[float(np.concatenate([group_values[i] for i in rng.integers(0,len(groups),len(groups))]).mean()) for _ in range(5000)]
    lower,upper=np.quantile(bootstrap,[.025,.975]).tolist()
    checks=dict(retained_per_source=retain(c,b),healthy_min_70_percent=all(m['healthy']['correct']/m['healthy']['n']>=.7 for m in c.values() if m['healthy']['n']),healthy_gain_min_5pp=float(gains.mean())>=.05,paired_capture_group_bootstrap_lower_positive=lower>0.)
    result=dict(protocol=protocol,baseline=b,candidate=c,checks=checks,passed=all(checks.values()),paired_healthy_gain=float(gains.mean()),paired_healthy_capture_group_bootstrap_95=[lower,upper],healthy_wilson_95={s:wilson(m['healthy']['correct'],m['healthy']['n']) for s,m in c.items()},reserved_now_opened=True,next_step='Exported-byte replay and native implementation still required if passed; otherwise reject and do not retune on this reservation.')
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(passed=result['passed'],checks=checks,counts={s:dict(Healthy=m['healthy']['correct'],Healthy_n=m['healthy']['n'],false_Healthy=m['all']['false_healthy']) for s,m in c.items()})),flush=True)


if __name__=='__main__':main()
