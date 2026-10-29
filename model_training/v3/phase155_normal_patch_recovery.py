"""Normal-patch memory on already cached features; two predeclared resolutions.

Inspired by normal-patch memory/coresets, not an exact PatchCore replication:
https://arxiv.org/abs/2106.08265 . Clinical evidence still comes from image labels
and unchanged app-verdict gates, not industrial benchmark AUROC.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.model_selection import StratifiedGroupKFold
from conditional_recovery_policy import recover_only
from monotone_normality import fit_normality,predict_normality
from opened_context_data import load,read
from phase85_population_sft import evaluation_data
from phase114_new_supervision_features import original_features
from permissive_encoder import digest
from population_siglip_tail import freeze
from target_replay_sampling import target_replay_weights


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase155_normal_patch_recovery';out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    values,original_bags,descriptors,identity=original_features(p);del values
    rows=read(p/'phase84_population_head/fitting_rows.json');selection=read(p/'phase118_conditional_recovery/protocol.json');allowed=set(read(p/'phase127_clinical_anchors/protocol.json')['train_indices'])
    train=np.array([i for i in selection['train_indices'] if i in allowed]);val=np.array(selection['validation_indices']);y=np.array([rows[i]['y'] for i in train])
    # Join family and exact original-ROI identities for all inner calibration folds.
    parent=list(range(len(train)));owners={}
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for offset,i in enumerate(train):
        keys=['family:'+rows[i]['group']]+['pixel:'+descriptors[key]['input_sha256'] for view in original_bags[rows[i]['path']].values() for key in view]
        for key in keys:
            if key in owners:parent[root(offset)]=root(owners[key])
            else:owners[key]=offset
    groups=np.array([root(i) for i in range(len(train))]);folds=list(StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=2155).split(np.zeros((len(train),1)),y,groups))
    weights=target_replay_weights(rows,train,'olid_tomato')*len(train)
    for a,b in folds:assert not set(groups[a])&set(groups[b]) and set(y[b])=={0,1}
    data=load(p);entries,records,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({b['path']:(b,packets[b['id']],r) for b,r in zip(before,previous)})
    protocol=dict(source_sha256=digest(__file__),calibration_source_sha256=digest(Path(__file__).with_name('monotone_normality.py')),parents=identity,selection_sha256=digest(p/'phase118_conditional_recovery/protocol.json'),
        train_indices=train.tolist(),validation_indices=val.tolist(),folds=[dict(reference_fit=a.tolist(),calibration=b.tolist()) for a,b in folds],resolutions=[224,448],token_cache_complete_sha256={name:digest(p/name/'complete.json') for name in ('phase120_local_patch_cache','phase128_native_dino_patch_cache')},reference_patches_per_healthy_photo=64,memory_patches=512,
        distance='One minus maximum cosine similarity to a farthest-point normal-patch coreset; final normalized local tokens only, CLS excluded',statistics=['q50','q90','q95','q99','maximum'],
        normal_reference='Healthy training photos from reference folds only; no disease patch labels, no validation images in memories',
        calibration='Each held-out training fold fits a monotone nonpositive-coefficient logistic normality model; each photo has two equally weighted crop-view samples; regularization .01',
        combination='Mean of four held-out-fold calibrators per crop view, then minimum over center/full; admission remains .95',
        existing_healthy_and_named_disease_preserved=True,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False,
        stopping='One fixed memory size and calibration recipe per cached resolution. No threshold, coreset-size or quantile grid. Queue complete unchanged gates only after >=28/40 Healthy and zero new source false Healthy; never export from this probe.')
    freeze(out/'protocol.json',protocol);controls=[]
    for size,name in [(224,'phase120_local_patch_cache'),(448,'phase128_native_dino_patch_cache')]:
        folder=out/str(size);folder.mkdir(exist_ok=True);source=p/name;sp=read(source/'protocol.json');sc=read(source/'complete.json')
        assert sc['protocol_sha256']==digest(source/'protocol.json') and sc['tokens_sha256']==digest(source/'tokens.npy')
        if size==448:bags=sp['bags']
        else:
            lookup={sha:i for i,sha in enumerate(sp['input_sha256'])};bags={path:{view:[lookup[descriptors[key]['input_sha256']] for key in keys] for view,keys in pair.items()} for path,pair in original_bags.items() if pair['center'] and all(descriptors[key]['input_sha256'] in lookup for keys in pair.values() for key in keys)}
        assert all(rows[i]['path'] in bags for i in train.tolist()+val.tolist())
        assert not ({key for i in train for keys in bags[rows[i]['path']].values() for key in keys}&{key for i in val for keys in bags[rows[i]['path']].values() for key in keys})
        token_file=np.load(source/'tokens.npy',mmap_mode='r');patches=F.normalize(torch.from_numpy(np.array(token_file[:,1:],copy=True)).cuda(),dim=-1);predictions=[]
        for fold,(a,b) in enumerate(folds):
            dest=folder/str(fold);dest.mkdir(exist_ok=True);reference=[int(train[i]) for i in a if rows[train[i]]['y']==1];samples=[]
            for i in reference:
                keys=[key for view in ('center','full') for key in bags[rows[i]['path']][view]];raw=patches[keys].reshape(-1,384);sample_ids=torch.linspace(0,len(raw)-1,64,device='cuda').long();samples.append(raw[sample_ids])
            candidates=torch.cat(samples);nearest=torch.full((len(candidates),),float('inf'),device='cuda');chosen=[];current=0
            with torch.inference_mode():
                for _ in range(min(512,len(candidates))):
                    chosen.append(current);distance=(1-candidates@candidates[current]).clamp_min(0)
                    nearest=torch.minimum(nearest,distance);nearest[chosen]=-1.;current=int(nearest.argmax())
                memory=candidates[chosen];np.save(dest/'memory.npy',memory.cpu().numpy())
            requested=train[b].tolist()+val.tolist();features=[]
            with torch.inference_mode():
                for i in requested:
                    pair=[]
                    for view in ('center','full'):
                        raw=patches[bags[rows[i]['path']][view]].reshape(-1,384);distance=torch.cat([(1-part@memory.T).min(1).values.clamp_min(0) for part in raw.split(2048)])
                        pair.append(torch.cat([torch.quantile(distance,torch.tensor([.5,.9,.95,.99],device='cuda')),distance.max()[None]]).cpu().numpy())
                    features.append(pair)
            features=np.array(features);cal=features[:len(b)];evaluation=features[len(b):];model=fit_normality(cal.reshape(-1,5),np.repeat(y[b],2),np.repeat(weights[b]/2,2))
            probability=predict_normality(model,evaluation.reshape(-1,5)).reshape(len(val),2);predictions.append(probability)
            np.savez(dest/'scores.npz',indices=requested,features=features,protocol_sha256=digest(out/'protocol.json'),token_complete_sha256=digest(source/'complete.json'),memory_sha256=digest(dest/'memory.npy'))
            np.savez(dest/'calibrator.npz',**model);(dest/'reference_photos.json').write_text(json.dumps(reference))
            print(json.dumps(dict(stage='normal_patch_calibration',resolution=size,fold=fold,normal_reference_photos=len(reference),calibration_photos=len(b))),flush=True)
        probability=np.mean(predictions,axis=0).min(1);errors={};healthy=selection['fixed_healthy_validation_correct'];old_h=0
        for i,value in zip(val,probability):
            row=rows[i];original,packet,retained=items[row['path']];now=recover_only(original,retained,packet,float(value),data['manifest'])['app_replay']['kind']=='HEALTHY'
            if not row['y']:errors[row['source']]=errors.get(row['source'],0)+int(now)
            elif row['source']=='olid_tomato':healthy+=int(now)
            else:old_h+=int(now)
        result=dict(resolution=size,healthy_validation_correct=healthy,new_false_healthy_by_source=errors,old_healthy_correct=old_h,cheap_pass=healthy>=28 and sum(errors.values())==0,passed=False,full_screen_run=False,new_encoder_inputs=0)
        np.savez(folder/'validation_predictions.npz',indices=val,probability=probability,protocol_sha256=digest(out/'protocol.json'));(folder/'result.json').write_text(json.dumps(result,indent=2));controls.append(result);print(json.dumps(result),flush=True)
        del patches;torch.cuda.empty_cache()
    result=dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),controls=controls,model_promoted=False,new_encoder_inputs=0,elapsed_seconds=time.monotonic()-started,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    if any(c['cheap_pass'] for c in controls):result['next_action']='Freeze the cheap-passing normal-memory recipe and run complete unchanged development gates, extending only genuinely missing feature identities before restored-model verification'
    (out/'result.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
