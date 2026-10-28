"""One rotation-robust head probe with newly audited development supervision."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from torch.nn import functional as F
from leaf_feature_expert import LeafFeatureHead
from opened_context_data import load,read
from opened_guard_screen import OpenedGuardScreen
from permissive_encoder import digest
from phase2_select import retain
from phase5_scene_ablation import photo_metrics
from phase22_accept import paired_interval
from phase83_eligible_operating_points import safety
from phase85_population_sft import evaluation_data
from phase88_diagnostic_bag_head import route_paired
from phase114_new_supervision_features import original_features
from population_siglip_tail import freeze
from source_balance import source_class_group_weights
from training_support import atomic_save,capture_rng,restore_rng
from routed_validation_cohort import healthy_cohort


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase115_new_supervision_head';out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    torch.set_num_threads(4);torch.manual_seed(2115);torch.backends.cuda.matmul.allow_tf32=False
    values,bags,descriptors,parent=original_features(p);old_count=len(values)
    extra=p/'phase114_new_supervision_features';complete=read(extra/'complete.json');assert complete['protocol_sha256']==digest(extra/'protocol.json')
    for name,field in [('new_features.npy','new_features_sha256'),('bags.json','bags_sha256'),('new_input_sha256.json','new_keys_sha256')]:assert digest(extra/name)==complete[field]
    new_bags=read(extra/'bags.json');new_keys=read(extra/'new_input_sha256.json');values=np.concatenate([values,np.load(extra/'new_features.npy')])
    identities=[r['input_sha256'] for r in descriptors]+new_keys;assert len(identities)==len(values)
    base_rows=read(p/'phase84_population_head/fitting_rows.json');rows=list(base_rows);tables={}
    for i,row in enumerate(rows):
        if row['active'] and row['path'] in bags and bags[row['path']]['center']:tables[i]=[[bags[row['path']][view] for view in ('center','full')]]
    for path,bag in new_bags.items():
        if not bag['actual_leaves']:continue
        index=len(rows);record=bag['source_record']
        rows.append(dict(path=path,y=int(record['label']=='Healthy'),source='new107_development',group='new107:entire_publisher_collection',validation=False,active=True))
        tables[index]=bag['views']
    val=np.array([i for i in tables if rows[i]['validation']]);val_shas={identities[key] for i in val for pair in tables[i] for view in pair for key in view}
    excluded={rows[i]['group'] for i in tables if not rows[i]['validation'] and any(identities[key] in val_shas for pair in tables[i] for view in pair for key in view)}
    train=np.array([i for i in tables if not rows[i]['validation'] and rows[i]['group'] not in excluded])
    assert not ({rows[i]['group'] for i in train}&{rows[i]['group'] for i in val})
    assert not ({identities[key] for i in train for pair in tables[i] for view in pair for key in view}&val_shas)
    assert any(rows[i]['source']=='new107_development' for i in train),'New-source overlap invalidates this probe; do not silently train the old experiment again'
    warm_dir=p/'phase88_diagnostic_bag_head/mlp';warm_result=read(warm_dir/'result.json');assert warm_result['checkpoint_sha256']==digest(warm_dir/'best.pt')
    calibration=warm_result['calibration'];data=load(p);entries,records,packets,before,previous=evaluation_data(p,data)
    denominator,inactive_healthy=healthy_cohort(base_rows,{rows[i]['path'] for i in val},{row['path']:row for row in previous},'olid_tomato')
    assert denominator==40
    protected=p/'phase116_protected_reservation_overlap/result.json'
    assert protected.exists() and read(protected)['overlap_images']==0,'Preserve the provisional acceptance reservation before new-source training'
    protocol=dict(source_sha256=digest(__file__),parents=parent,new_feature_complete_sha256=digest(extra/'complete.json'),warm_checkpoint_sha256=digest(warm_dir/'best.pt'),
                  train_images=len(train),validation_images=len(val),new_source_training_images=sum(rows[i]['source']=='new107_development' for i in train),
                  healthy_validation_full_cohort=denominator,inactive_healthy_fixed_correct=inactive_healthy,protected_reservation_audit_sha256=digest(protected),
                  maximum_epochs=30,minimum_epochs=8,patience=6,learning_rate=.001,negative_loss_weight=8.,weight_decay=.01,seed=2115,
                  calibration=calibration,positive_threshold=.95,negative_threshold=.001,source_class_group_balanced=True,encoder_weights_changed=False,
                  robust_loss='Healthy minimum and diseased maximum over mean actual-ROI logits of both crop views and all four new-source rotations; one publisher image label per bag',
                  new_source_role='Entire collection training-only; camera/style confounding remains unresolved; never independent acceptance',
                  acceptance_data_used=False,ad_independent=False,olid_independent=False,
                  stopping='One MLP, at most 30 epochs and six stale actual-verdict epochs. Require >=28/40 validation Healthy with no source false-Healthy increase before full cached head screen. No threshold/calibration/architecture search, full restored model or export without all recognition gates.')
    freeze(out/'protocol.json',protocol)
    features=torch.from_numpy(values).cuda();truth=torch.tensor([r['y'] for r in rows],device='cuda',dtype=torch.float32)
    index=np.zeros((len(rows),4,2,6),np.int64);mask=np.zeros(index.shape,np.float32)
    for i,pairs in tables.items():
        for rotation,pair in enumerate(pairs):
            for view,keys in enumerate(pair):index[i,rotation,view,:len(keys)]=keys;mask[i,rotation,view,:len(keys)]=1.
    index=torch.from_numpy(index).cuda();mask=torch.from_numpy(mask).cuda();valid=mask.sum(-1)>0
    def forward(ids):
        z=model(features[index[ids]].reshape(-1,1408)).reshape(len(ids),4,2,6)*calibration[0]+calibration[1]
        means=(z*mask[ids]).sum(-1)/mask[ids].sum(-1).clamp_min(1)
        return means.masked_fill(~valid[ids],float('inf')).amin((1,2)),means.masked_fill(~valid[ids],-float('inf')).amax((1,2))
    model=LeafFeatureHead('mlp',1408).cuda();model.load_state_dict(torch.load(warm_dir/'best.pt',weights_only=True,map_location='cpu')['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);sampling=torch.from_numpy(source_class_group_weights(rows,train))
    weights=torch.tensor(source_class_group_weights(rows,val),device='cuda');history=[];best=None;stale=0;start=0
    items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({b['path']:(b,packets[b['id']],r) for b,r in zip(before,previous)})
    def decision(positive,negative,loss):
        false={};healthy=inactive_healthy;old_h=0
        for k,i in enumerate(val):
            row=rows[i];original,packet,retained=items[row['path']]
            after=route_paired(original,retained,packet,float(positive[k]),float(negative[k]),.95,data['manifest']);was=retained['app_replay']['kind']=='HEALTHY';now=after['app_replay']['kind']=='HEALTHY'
            if not row['y']:false[row['source']]=false.get(row['source'],0)+int(now)-int(was)
            elif row['source']=='olid_tomato':healthy+=int(now)
            else:old_h+=int(now)
        return (-sum(max(0,v) for v in false.values()),healthy,old_h,-loss),false
    if (out/'last.pt').exists():
        saved=torch.load(out/'last.pt',weights_only=True,map_location='cpu');assert saved['protocol_sha256']==digest(out/'protocol.json')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);history,best,stale,start=saved['history'],saved['best'],saved['stale'],saved['epoch']+1
    for epoch in range(start,30):
        if epoch>=8 and stale>=6:break
        model.train();order=torch.multinomial(sampling,len(train),replacement=True,generator=torch.Generator().manual_seed(211500+epoch))
        for batch in order.split(64):
            ids=train[batch.numpy()];positive,negative=forward(ids);z=torch.where(truth[ids]==1,positive,negative)
            loss=(F.binary_cross_entropy_with_logits(z,truth[ids],reduction='none')*torch.where(truth[ids]==0,8.,1.)).mean();assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        model.eval()
        with torch.inference_mode():
            pairs=[forward(ids) for ids in np.array_split(val,max(1,(len(val)+63)//64))];positive=torch.cat([pair[0] for pair in pairs]);negative=torch.cat([pair[1] for pair in pairs]);z=torch.where(truth[val]==1,positive,negative)
            loss=float((F.binary_cross_entropy_with_logits(z,truth[val],reduction='none')*torch.where(truth[val]==0,8.,1.)*weights).sum())
            score,false=decision(positive.sigmoid().cpu().numpy(),negative.sigmoid().cpu().numpy(),loss)
        state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
        if best is None or score>tuple(best):best=score;stale=0;atomic_save(dict(model=state,epoch=epoch,decision=score,protocol_sha256=digest(out/'protocol.json')),out/'best.pt')
        else:stale+=1
        history.append(dict(epoch=epoch,decision=score,false_healthy_delta_by_source=false));print(json.dumps(history[-1]),flush=True)
        atomic_save(dict(model=state,optimizer=optimizer.state_dict(),rng=capture_rng(),epoch=epoch,history=history,best=best,stale=stale,protocol_sha256=digest(out/'protocol.json')),out/'last.pt')
    selected=torch.load(out/'best.pt',weights_only=True,map_location='cpu');cheap_pass=best[0]==0 and best[1]>=28
    result=dict(cheap_pass=cheap_pass,selected_epoch=selected['epoch'],best_decision=best,history=history,full_cached_screen_run=False,passed=False,
                protocol_sha256=digest(out/'protocol.json'),source_sha256=digest(__file__),checkpoint_sha256=digest(out/'best.pt'),
                old_encoder_inputs_recomputed=0,new_source_training_images=protocol['new_source_training_images'],acceptance_data_used=False,ad_independent=False,olid_independent=False,model_promoted=False)
    if cheap_pass:
        model.load_state_dict(selected['model']);model.eval()
        with torch.inference_mode():raw=torch.cat([model(part)*calibration[0]+calibration[1] for part in features[:old_count].split(1024)]).cpu().numpy()
        probabilities={path:dict(positive=float(min(1/(1+np.exp(-np.clip(raw[keys].mean(),-80,80))) for keys in pair.values())),negative=float(max(1/(1+np.exp(-np.clip(raw[keys].mean(),-80,80))) for keys in pair.values()))) for path,pair in bags.items() if pair['center']}
        def revised(row,packet,retained):
            score=probabilities.get(row['path']);return retained if score is None else route_paired(row,retained,packet,score['positive'],score['negative'],.95,data['manifest'])
        def revise(row):return revised(row,*entries[id(row)])
        screen=OpenedGuardScreen(data).evaluate(revise);screen['checks'].pop('original_disease_and_healthy_exact')
        screen['checks']['original_disease_exact']=all(revise(r)['app_replay']==r['app_replay'] for group in list(data['base'].values())+[data['internal']] for r in group if r['app_replay']['kind']=='DISEASE')
        screen['passed']=all(screen['checks'].values());after=[revised(b,packets[b['id']],retained) for b,retained in zip(before,previous)]
        vi=[i for i,row in enumerate(base_rows[-len(records):]) if row['validation']];vr,vb,va=[[group[i] for i in vi] for group in (records,before,after)]
        vm=photo_metrics(va);metrics=photo_metrics(after);bm=photo_metrics(before)
        for group,metric in [(after,metrics),(before,bm)]:metric['unsupported_tomato']=photo_metrics([r for r in group if r['label']=='Other'])['all']
        vsafe,vs=safety(vr,vb,va);asafe,alls=safety(records,before,after);gain=paired_interval(vr,vb,va,lambda r:r['label']=='Healthy',lambda r:int(r['app_replay']['kind']=='HEALTHY'))
        checks=dict(screen['checks'],consumed_validation_healthy=vm['healthy']['correct_rate']>=.7,consumed_full_healthy=metrics['healthy']['correct_rate']>=.7,
                    consumed_gain=gain is not None and gain['point']>=.1 and gain['bootstrap_two_sided_95'][0]>0,consumed_validation_safety=vsafe,consumed_full_safety=asafe,consumed_full_retention=retain({'olid':metrics},{'olid':bm}))
        result.update(full_cached_screen_run=True,passed=all(checks.values()),checks=checks,old_screen=screen,consumed_validation=vm,consumed_all=metrics,validation_safety=vs,all_safety=alls,validation_gain=gain)
        (out/'probabilities.json').write_text(json.dumps(probabilities))
    result['elapsed_seconds']=time.monotonic()-started;(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)


if __name__=='__main__':main()
