"""Bounded input-domain ablation: neutralize only border-connected neutral background.

The original app inputs, weights and verdicts stay unchanged. This is an
auxiliary proposal experiment on consumed training families, not a release.
"""
from pathlib import Path
import argparse,hashlib,json,math,sqlite3,time
import cv2
import numpy as np
from PIL import Image
import timm,torch
from app_eval import canon,classifier_letterbox_input
from compact_healthy_model import primary,calibrated_probabilities
from opened_context_data import load,read
from permissive_encoder import digest
from phase83_eligible_operating_points import eligible
from phase85_population_sft import evaluation_data
from phase87_diagnostic_bag_features import cached_mobile
from population_siglip_tail import freeze


def neutral_background(rgb,box):
    left,top,right,bottom=box;dx=(right-left)*.1;dy=(bottom-top)*.1
    x1=max(0,int(left-dx));x2=min(rgb.shape[1],int(math.ceil(right+dx)))
    y1=max(0,int(top-dy));y2=min(rgb.shape[0],int(math.ceil(bottom+dy)))
    roi=rgb[y1:y2,x1:x2];h,w=roi.shape[:2]
    if min(h,w)<32:return None
    lab=cv2.cvtColor(roi,cv2.COLOR_RGB2LAB).astype(np.float32)
    strip=max(1,min(h,w)//64)
    border=np.concatenate([lab[:strip].reshape(-1,3),lab[-strip:].reshape(-1,3),lab[:,:strip].reshape(-1,3),lab[:,-strip:].reshape(-1,3)])
    center=np.median(border,axis=0)
    if abs(center[1]-128)>7 or abs(center[2]-128)>7:return None
    difference=np.abs(lab-center)
    candidate=((difference[:,:,0]<=20)&(difference[:,:,1]<=6)&(difference[:,:,2]<=6)).astype(np.uint8)
    border_match=np.concatenate([candidate[:strip].ravel(),candidate[-strip:].ravel(),candidate[:,:strip].ravel(),candidate[:,-strip:].ravel()]).mean()
    if border_match<.65:return None
    count,components=cv2.connectedComponents(candidate,connectivity=8)
    connected=np.unique(np.concatenate([components[0],components[-1],components[:,0],components[:,-1]]))
    connected=connected[connected!=0]
    mask=np.isin(components,connected).astype(np.uint8)
    radius=max(3,int(math.ceil(max(h,w)/112)))
    mask=cv2.erode(mask,np.ones((2*radius+1,2*radius+1),np.uint8)).astype(bool)
    fraction=float(mask.mean())
    if not .2<=fraction<=.85:return None
    result=rgb.copy();replacement=roi.copy();replacement[mask]=114
    assert np.array_equal(replacement[~mask],roi[~mask]),'Foreground must remain byte-identical'
    result[y1:y2,x1:x2]=replacement
    return result,dict(background_fraction=fraction,border_match=float(border_match),median_lab=center.tolist(),collar_pixels=radius)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase108_background_pilot';out.mkdir(exist_ok=True)
    if (out/'result.json').exists():
        assert read(out/'result.json')['source_sha256']==digest(__file__);return
    started=time.monotonic();data=load(p);_,records,packets,before,_=evaluation_data(p,data)
    fitting=read(p/'phase84_population_head/fitting_rows.json');by_path={r['path']:r for r in fitting}
    candidates=[b for b in before if not by_path[b['path']]['validation'] and eligible(b,packets[b['id']]) and packets[b['id']]['leaves']]
    def cohort(row):
        if row['label']=='Healthy':return 'healthy'
        if row['label']=='Other':return 'unsupported'
        if row['label']=='Leaf Miner':return 'miner'
        return 'other_disease'
    chosen=[];groups=set()
    for kind,limit in [('healthy',8),('unsupported',8),('miner',4),('other_disease',4)]:
        values=sorted([r for r in candidates if cohort(r)==kind],key=lambda r:hashlib.sha256(('bg-pilot-2108:'+r['group']+r['id']).encode()).hexdigest())
        n=0
        for row in values:
            if row['group'] in groups:continue
            groups.add(row['group']);chosen.append(row);n+=1
            if n==limit:break
    assert sum(r['label']=='Healthy' for r in chosen)==8
    protocol=dict(source_sha256=digest(__file__),hypothesis='Uniform dark photographic backgrounds may cause domain rejection independently of normalization or classifier weights',
        cases=[dict(id=r['id'],path=r['path'],file_sha256=r['file_sha256'],group=r['group'],label=r['label']) for r in chosen],
        model_sha256=digest(p/'export_cnn_safe/pytorch_bundle.pt'),mobile_scores_sha256=digest(p/'phase76_fresh_acceptance/mobile_scores.jsonl'),
        background_rule='Border-connected, low-chroma Lab background only; fixed thresholds and erosion collar; leave all other RGB pixels byte-identical',
        positive_threshold=.95,original_model_calibration_unchanged=True,weights_changed=False,
        original_disease_and_healthy_verdicts_untouched=True,ad_independent=False,olid_independent=False,
        stopping='One fixed pilot only. Advance only if at least 7/8 rejected Healthy recover with zero new false-Healthy among pilot diseases; otherwise stop without a threshold/mask search or full inference.')
    freeze(out/'protocol.json',protocol)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;timm.layers.set_fused_attn(False)
    _,model=primary(p);model=model.cuda().eval()
    db=sqlite3.connect('file:'+str(p/'phase76_fresh_acceptance/cache/predictions.sqlite3')+'?mode=ro',uri=True)
    detector_sha=read(p/'phase76_fresh_acceptance/frozen_protocol.json')['models']['detector']
    cases=[];new_inputs={}
    for row in chosen:
        assert digest(row['path'])==row['file_sha256']
        with Image.open(row['path']) as image:rgb=np.asarray(image.convert('RGB'))
        tensor,letterbox=canon.detector_input(rgb,320)
        raw=cached_mobile(db,detector_sha,tensor).reshape(1,5,2100);d=data['manifest']['detector']
        boxes=canon.decode_detections(raw,letterbox,rgb.shape[1],rgb.shape[0],d['score_threshold'],d['nms_iou'],d['max_detections'])[:6]
        assert len(boxes)==len(packets[row['id']]['leaves'])
        pair=dict(center=[],full=[]);quality=[];valid=True
        for box in boxes:
            changed=neutral_background(rgb,box[:4])
            if changed is None:valid=False;break
            modified,metadata=changed;quality.append(metadata)
            for name,preprocess in [('center',canon.classifier_input),('full',classifier_letterbox_input)]:
                tensor=preprocess(modified,224,box[:4])
                key=hashlib.sha256(tensor.tobytes()).hexdigest();new_inputs.setdefault(key,tensor);pair[name].append(key)
        cases.append(dict(id=row['id'],label=row['label'],valid_mask=valid,inputs=pair,mask_audit=quality))
    db.close();predictions={}
    keys=sorted(new_inputs)
    with torch.inference_mode():
        for start in range(0,len(keys),64):
            part=keys[start:start+64];_,reference=model.components(torch.from_numpy(np.concatenate([new_inputs[k] for k in part])).cuda())
            for key,value in zip(part,reference.cpu().numpy()):predictions[key]=value
    for case in cases:
        case['healthy_probability']=0.
        if case['valid_mask']:
            means=np.stack([np.mean([predictions[key] for key in case['inputs'][name]],axis=0) for name in ('center','full')])
            probability=calibrated_probabilities(torch.from_numpy(means)).numpy()[:,1]
            case['healthy_probability']=float(probability.min())
        case['proposed_healthy']=case['healthy_probability']>=.95
    recovered=sum(r['proposed_healthy'] for r in cases if r['label']=='Healthy')
    false=sum(r['proposed_healthy'] for r in cases if r['label']!='Healthy')
    result=dict(advance=recovered>=7 and false==0,healthy_recovered=recovered,healthy_n=8,new_false_healthy=false,
                new_encoder_inputs=len(keys),cases=cases,protocol_sha256=digest(out/'protocol.json'),source_sha256=digest(__file__),
                elapsed_seconds=time.monotonic()-started,ad_independent=False,olid_independent=False,model_promoted=False)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({key:value for key,value in result.items() if key!='cases'}),flush=True)


if __name__=='__main__':main()
