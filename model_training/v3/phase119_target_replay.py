"""Test target-error exposure and decision-margin training from cached features."""
from pathlib import Path
import argparse,collections,hashlib,json,math,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from conditional_recovery_policy import recover_only
from conditional_recovery_losses import cross_source_contrastive,conditional_domain_loss
from leaf_feature_expert import LeafFeatureHead
from opened_context_data import load,read
from opened_guard_screen import OpenedGuardScreen
from permissive_encoder import digest
from phase2_select import retain
from phase5_scene_ablation import photo_metrics
from phase22_accept import paired_interval
from phase83_eligible_operating_points import eligible,safety
from phase85_population_sft import evaluation_data
from phase114_new_supervision_features import original_features
from population_siglip_tail import freeze
from routed_validation_cohort import healthy_cohort
from source_balance import source_class_group_weights
from training_support import atomic_save,capture_rng,restore_rng
from target_replay_sampling import target_replay_weights


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase119_target_replay';out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(2118)
    values,bags,descriptors,parent=original_features(p);rows=read(p/'phase84_population_head/fitting_rows.json')
    data=load(p);entries,records,packets,before,previous=evaluation_data(p,data)
    items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({b['path']:(b,packets[b['id']],r) for b,r in zip(before,previous)})
    active=[];photo_keys=[]
    for i,row in enumerate(rows):
        pair=bags.get(row['path']);photo_keys.append('' if not pair else hashlib.sha256(json.dumps([sorted(descriptors[key]['input_sha256'] for key in pair[view]) for view in ('center','full')]).encode()).hexdigest())
        if pair and pair['center']:
            original,packet,retained=items[row['path']]
            if retained['app_replay']['kind']!='HEALTHY' and eligible(original,packet):active.append(i)
    val=np.array([i for i in active if rows[i]['validation']]);val_shas={descriptors[key]['input_sha256'] for i in val for view in bags[rows[i]['path']].values() for key in view}
    excluded={rows[i]['group'] for i in active if not rows[i]['validation'] and any(descriptors[key]['input_sha256'] in val_shas for view in bags[rows[i]['path']].values() for key in view)}
    train=np.array([i for i in active if not rows[i]['validation'] and rows[i]['group'] not in excluded])
    assert not ({rows[i]['group'] for i in train}&{rows[i]['group'] for i in val})
    assert not ({descriptors[key]['input_sha256'] for i in train for view in bags[rows[i]['path']].values() for key in view}&val_shas)
    grouped=collections.defaultdict(set)
    for i in train:grouped[photo_keys[i]].add(rows[i]['y'])
    assert all(len(labels)==1 for labels in grouped.values()),'Contradictory image-bag training labels'
    denominator,fixed=healthy_cohort(rows,{rows[i]['path'] for i in val},{r['path']:r for r in previous},'olid_tomato');assert denominator==40
    source_names=sorted({r['source'] for r in rows});source_ids={name:i for i,name in enumerate(source_names)}
    source_counts=collections.Counter((rows[i]['y'],source_ids[rows[i]['source']]) for i in train)
    domain_maps={str(label):{source:index for index,source in enumerate(sorted(source for (y,source),n in source_counts.items() if y==label and n>=5))} for label in (0,1)}
    domain_maps={label:mapping for label,mapping in domain_maps.items() if len(mapping)>=2};assert domain_maps
    warm=p/'phase88_diagnostic_bag_head/mlp';wr=read(warm/'result.json');assert wr['checkpoint_sha256']==digest(warm/'best.pt')
    calibration=wr['calibration'];methods=['source_balanced_warmup','target_replay','target_replay_margin']
    protocol=dict(source_sha256=digest(__file__),loss_source_sha256=digest(Path(__file__).with_name('conditional_recovery_losses.py')),policy_source_sha256=digest(Path(__file__).with_name('conditional_recovery_policy.py')),
                  parents=parent,warm_checkpoint_sha256=digest(warm/'best.pt'),fitting_rows_sha256=digest(p/'phase84_population_head/fitting_rows.json'),methods=methods,
                  hypothesis='Selected safe checkpoints remain underconfident on training target errors. Test a bounded longer warm-up, then explicitly reserve half of training mass for consumed OLID errors and half for protective replay; isolate a decision-margin objective as the third control.',
                  train_indices=train.tolist(),validation_indices=val.tolist(),full_healthy_validation_denominator=denominator,fixed_healthy_validation_correct=fixed,
                  source_counts={f'{label}:{source_names[source]}':n for (label,source),n in source_counts.items()},domain_maps=domain_maps,
                  maximum_epochs=200,minimum_epochs=40,patience=20,batch_images=64,learning_rate=.001,weight_decay=.01,negative_loss_weight=1.,seed=2118,
                  target_replay_weight=.5,target_source='olid_tomato',sampler_source_sha256=digest(Path(__file__).with_name('target_replay_sampling.py')),margin_weight=.1,margin=.5,
                  calibration=calibration,admission_probability=.95,existing_healthy_and_disease_preserved=True,all_existing_evidence_guards_preserved=True,
                  supcon_weight=.1,supcon_temperature=.1,conditional_adversary_weight=.1,minimum_domain_class_training_photos=5,
                  reward_weight=.1,reward_false_healthy_cost=8.,reward_boundary_temperature=.25,
                  reward_scope='Differentiable expected admission reward from consumed training labels, not environmental RL or fresh acceptance feedback',
                  new_encoder_inputs=0,new_publisher_collection_used=False,ad_independent=False,olid_independent=False,acceptance_data_used=False,
                  stopping='Three predeclared exposure/objective controls on identical cached inputs and fixed policy. Minimum forty epochs, stop at twenty stale routed-validation epochs, two hundred maximum. Full cached screen only after >=28/40 Healthy with zero new source false-Healthy; no export or restored-model inference unless all old and consumed gates pass.')
    freeze(out/'protocol.json',protocol)
    features=torch.from_numpy(values).cuda();truth=torch.tensor([r['y'] for r in rows],dtype=torch.float32,device='cuda');sources=torch.tensor([source_ids[r['source']] for r in rows],device='cuda')
    photo_map={key:i for i,key in enumerate(sorted(set(photo_keys)))};photo_ids=torch.tensor([photo_map[key] for key in photo_keys],device='cuda')
    tables=np.zeros((len(rows),2,6),np.int64);mask=np.zeros(tables.shape,np.float32)
    for i in active:
        for view,name in enumerate(('center','full')):
            keys=bags[rows[i]['path']][name];tables[i,view,:len(keys)]=keys;mask[i,view,:len(keys)]=1.
    tables=torch.from_numpy(tables).cuda();mask=torch.from_numpy(mask).cuda();counts=mask.sum(-1).clamp_min(1)
    sampling=torch.from_numpy(source_class_group_weights(rows,train));vw=torch.tensor(source_class_group_weights(rows,val),device='cuda')
    def decision(probability,loss):
        errors={};healthy=fixed;old_healthy=0
        for offset,i in enumerate(val):
            row=rows[i];original,packet,retained=items[row['path']];after=recover_only(original,retained,packet,float(probability[offset]),data['manifest'])
            now=after['app_replay']['kind']=='HEALTHY'
            if not row['y']:errors[row['source']]=errors.get(row['source'],0)+int(now)
            elif row['source']=='olid_tomato':healthy+=int(now)
            else:old_healthy+=int(now)
        return (-sum(errors.values()),healthy,old_healthy,-loss),errors
    controls=[]
    for method in methods:
        sampling=torch.from_numpy(source_class_group_weights(rows,train) if method=='source_balanced_warmup' else target_replay_weights(rows,train,'olid_tomato'))
        folder=out/method;folder.mkdir(exist_ok=True)
        if (folder/'result.json').exists():
            result=read(folder/'result.json');assert result['protocol_sha256']==digest(out/'protocol.json');assert result['checkpoint_sha256']==digest(folder/'best.pt');controls.append(result);continue
        torch.manual_seed(2118)
        model=nn.ModuleDict(dict(expert=LeafFeatureHead('mlp',1408),adversaries=nn.ModuleDict({label:nn.Linear(128,len(mapping)) for label,mapping in domain_maps.items()}))).cuda()
        model['expert'].load_state_dict(torch.load(warm/'best.pt',weights_only=True,map_location='cpu')['model']);torch.manual_seed(2118)
        def forward(ids):
            h=model['expert'].network[:-1](features[tables[ids]].reshape(-1,1408))
            raw=(model['expert'].network[-1](h).ravel()*calibration[0]+calibration[1]).reshape(len(ids),2,6)
            means=(raw*mask[ids]).sum(-1)/counts[ids]
            hidden=(h.reshape(len(ids),2,6,128)*mask[ids][...,None]).sum(2)/counts[ids][...,None]
            return means,hidden.mean(1)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);history=[];best=None;stale=0;start=0
        if (folder/'last.pt').exists():
            saved=torch.load(folder/'last.pt',weights_only=True,map_location='cpu');assert saved['protocol_sha256']==digest(out/'protocol.json');model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);history,best,stale,start=saved['history'],saved['best'],saved['stale'],saved['epoch']+1
        for epoch in range(start,200):
            if epoch>=40 and stale>=20:break
            model.train();order=torch.multinomial(sampling,len(train),replacement=True,generator=torch.Generator().manual_seed(211800+epoch))
            for batch in order.split(64):
                ids=train[batch.numpy()];z,h=forward(ids);raw=torch.where(truth[ids]==1,z.min(1).values,z.max(1).values);loss=F.binary_cross_entropy_with_logits(raw,truth[ids])
                if method=='cross_source_supcon':loss=loss+.1*cross_source_contrastive(h,truth[ids],sources[ids],photo_ids[ids])
                elif method=='conditional_domain_adversarial':loss=loss+.1*conditional_domain_loss(h,truth[ids],sources[ids],model['adversaries'],domain_maps)
                elif method=='decision_reward':
                    action=((z.min(1).values-math.log(.95/.05))/.25).sigmoid();reward=truth[ids]*action-8.*(1-truth[ids])*action;loss=loss-.1*reward.mean()
                elif method=='target_replay_margin':
                    threshold=math.log(.95/.05)
                    margin_loss=truth[ids]*F.softplus(threshold+.5-z.min(1).values)+8.*(1-truth[ids])*F.softplus(z.max(1).values-threshold+.5)
                    loss=loss+.1*margin_loss.mean()
                assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            model.eval()
            with torch.inference_mode():
                z=torch.cat([forward(part)[0] for part in np.array_split(val,max(1,(len(val)+63)//64))]);raw=torch.where(truth[val]==1,z.min(1).values,z.max(1).values);loss=float((F.binary_cross_entropy_with_logits(raw,truth[val],reduction='none')*vw).sum());score,errors=decision(z.min(1).values.sigmoid().cpu().numpy(),loss)
            state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            if best is None or score>tuple(best):best=score;stale=0;atomic_save(dict(model=state,epoch=epoch,decision=score,protocol_sha256=digest(out/'protocol.json')),folder/'best.pt')
            else:stale+=1
            history.append(dict(epoch=epoch,decision=score,new_false_healthy_by_source=errors));print(json.dumps(dict(method=method,**history[-1])),flush=True)
            atomic_save(dict(model=state,epoch=epoch,optimizer=optimizer.state_dict(),rng=capture_rng(),history=history,best=best,stale=stale,protocol_sha256=digest(out/'protocol.json')),folder/'last.pt')
        chosen=torch.load(folder/'best.pt',weights_only=True,map_location='cpu');cheap=best[0]==0 and best[1]>=28
        result=dict(method=method,cheap_pass=cheap,best_decision=best,selected_epoch=chosen['epoch'],history=history,protocol_sha256=digest(out/'protocol.json'),checkpoint_sha256=digest(folder/'best.pt'),passed=False,full_cached_screen_run=False,new_encoder_inputs=0)
        if cheap:
            model.load_state_dict(chosen['model']);model.eval()
            with torch.inference_mode():raw=torch.cat([model['expert'](part)*calibration[0]+calibration[1] for part in features.split(1024)]).cpu().numpy()
            probabilities={path:float(min(1/(1+np.exp(-np.clip(raw[keys].mean(),-80,80))) for keys in pair.values())) for path,pair in bags.items() if pair['center']}
            def revised(row,packet,retained):return recover_only(row,retained,packet,probabilities.get(row['path'],0.),data['manifest'])
            def revise(row):return revised(row,*entries[id(row)])
            screen=OpenedGuardScreen(data).evaluate(revise);after=[revised(b,packets[b['id']],retained) for b,retained in zip(before,previous)]
            vi=[i for i,row in enumerate(rows[-len(records):]) if row['validation']];vr,vb,va=[[group[i] for i in vi] for group in (records,before,after)]
            vm=photo_metrics(va);metrics=photo_metrics(after);bm=photo_metrics(before)
            for group,metric in [(after,metrics),(before,bm)]:metric['unsupported_tomato']=photo_metrics([r for r in group if r['label']=='Other'])['all']
            vsafe,vs=safety(vr,vb,va);asafe,alls=safety(records,before,after);gain=paired_interval(vr,vb,va,lambda r:r['label']=='Healthy',lambda r:int(r['app_replay']['kind']=='HEALTHY'))
            checks=dict(screen['checks'],consumed_validation_healthy=vm['healthy']['correct_rate']>=.7,consumed_full_healthy=metrics['healthy']['correct_rate']>=.7,
                        consumed_gain=gain is not None and gain['point']>=.1 and gain['bootstrap_two_sided_95'][0]>0,consumed_validation_safety=vsafe,consumed_full_safety=asafe,consumed_full_retention=retain({'olid':metrics},{'olid':bm}))
            result.update(full_cached_screen_run=True,passed=all(checks.values()),checks=checks,old_screen=screen,consumed_validation=vm,consumed_all=metrics,validation_safety=vs,all_safety=alls,validation_gain=gain)
            (folder/'probabilities.json').write_text(json.dumps(probabilities))
        (folder/'result.json').write_text(json.dumps(result,indent=2));controls.append(result);print(json.dumps({key:value for key,value in result.items() if key not in ('history','old_screen','consumed_all','consumed_validation')}),flush=True)
    result=dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),controls=controls,passing=sum(r['passed'] for r in controls),new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False,elapsed_seconds=time.monotonic()-started,model_promoted=False)
    (out/'result.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
