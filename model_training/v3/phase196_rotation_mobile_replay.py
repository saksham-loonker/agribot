"""Two bounded recovery-only heads using the full audited development source."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from torch.nn import functional as F
from conditional_recovery_policy import recover_only
from hard_negative_ranking import hard_negative_rank_loss
from leaf_feature_expert import LeafFeatureHead
from opened_context_data import load,read
from phase85_population_sft import evaluation_data
from phase114_new_supervision_features import original_features
from permissive_encoder import digest
from population_siglip_tail import freeze
from source_balance import source_class_group_weights
from photo_rotation_replay import select_photo_rotation
from clinical_replay_balance import clinical_replay_weights
from training_support import atomic_save,capture_rng,restore_rng


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);root=p/'phase196_rotation_mobile_replay';root.mkdir(exist_ok=True);started=time.monotonic()
    if (root/'result.json').exists():assert read(root/'result.json')['source_sha256']==digest(__file__);return
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    values,bags,descriptors,parent=original_features(p);rows=read(p/'phase84_population_head/fitting_rows.json');selection=read(p/'phase118_conditional_recovery/protocol.json');anchors=read(p/'phase133_anchor_input_cache/protocol.json')
    extra=p/'phase137_full_source_features';complete=read(extra/'complete.json');assert complete['protocol_sha256']==digest(extra/'protocol.json')
    for name,field in [('new_features.npy','new_features_sha256'),('bags.json','bags_sha256'),('new_input_sha256.json','new_keys_sha256')]:assert digest(extra/name)==complete[field]
    protected=p/'phase138_full_source_reservation_overlap/result.json';assert read(protected)['overlap_images']==0 and not read(protected)['clinical_predictions_opened']
    identities=[r['input_sha256'] for r in descriptors]+read(extra/'new_input_sha256.json');values=np.concatenate([values,np.load(extra/'new_features.npy')]);assert len(values)==len(identities)
    field=p/'phase164_new_field_features';fc=read(field/'complete.json');assert fc['protocol_sha256']==digest(field/'protocol.json')
    for filename,key in [('features.npy','features_sha256'),('input_sha256.json','input_sha256_sha256'),('bags.json','bags_sha256')]:assert digest(field/filename)==fc[key]
    field_values=np.load(field/'features.npy');field_keys=read(field/'input_sha256.json');assert field_values.shape==(len(field_keys),3200)
    field_offset=len(identities);identities+=field_keys;values=np.concatenate([values,field_values[:,:1408]])
    target=anchors['target_indices'];omitted=read(p/'phase178_guard_coverage_audit/result.json')['training_coverage']['omitted_indices'];clinical=read(p/'phase127_clinical_anchors/protocol.json');replay=[i for i in clinical['train_indices'] if i not in set(target)];assert len(set(target+replay))==len(target+replay) and len(replay)>1000;val=np.array(selection['validation_indices']);old=target+replay+val.tolist()
    tables={i:[[bags[rows[i]['path']][view] for view in ('center','full')]] for i in old};new=[]
    for path,bag in read(extra/'bags.json').items():
        if not bag['actual_leaves']:continue
        new.append(len(rows));record=bag['source_record'];tables[len(rows)]=bag['views']
        rows.append(dict(path=path,y=int(record['label']=='Healthy'),source='new107_development',group='new107:entire_publisher_collection',validation=False,label=record['label']))
    train=np.array(target+replay+new);assert len(new)>1800
    field_indices=[]
    for path,bag in read(field/'bags.json').items():
        if not bag['actual_leaves']:continue
        i=len(rows);field_indices.append(i);tables[i]=[[[field_offset+key for key in view] for view in pair] for pair in bag['views']];record=bag['source_record']
        rows.append(dict(path=path,y=int(record['label']=='Healthy'),source='field93',group=record['group'],validation=False,label=record['label']))
    train=np.array(target+replay+field_indices);assert len(field_indices)>100 and sum(rows[i]['y'] for i in field_indices)>30
    rotation=p/'phase195_rotation_feature_extension';rotation_protocol=read(rotation/'protocol.json');rotation_complete=read(rotation/'complete.json')
    assert rotation_complete['protocol_sha256']==digest(rotation/'protocol.json') and rotation_complete['features_sha256']==digest(rotation/'features.npy') and rotation_complete['augmented_bags_sha256']==digest(rotation/'augmented_bags.json')
    assert rotation_protocol['base_count']==len(values) and rotation_protocol['train_indices']==train.tolist() and rotation_protocol['validation_indices']==val.tolist()
    identities += [row['input_sha256'] for row in rotation_protocol['new_inputs']];values=np.concatenate([values,np.load(rotation/'features.npy')]);assert len(values)==len(identities)
    tables.update({int(i):views for i,views in read(rotation/'augmented_bags.json').items()})
    val_keys={identities[key] for i in val for pair in tables[i] for keys in pair for key in keys}
    rotation_quarantined_groups={rows[i]['group'] for i in train if any(identities[key] in val_keys for pair in tables[i] for view in pair for key in view)}
    target=[i for i in target if rows[i]['group'] not in rotation_quarantined_groups];replay=[i for i in replay if rows[i]['group'] not in rotation_quarantined_groups];field_indices=[i for i in field_indices if rows[i]['group'] not in rotation_quarantined_groups];train=np.array(target+replay+field_indices)
    assert not ({identities[key] for i in train for pair in tables[i] for view in pair for key in view}&val_keys)

    assert not ({rows[i]['group'] for i in train}&{rows[i]['group'] for i in val})
    warm=p/'phase88_diagnostic_bag_head/mlp';wr=read(warm/'result.json');assert wr['checkpoint_sha256']==digest(warm/'best.pt');calibration=wr['calibration']
    methods=['field_source_sft','field_source_rank']
    protocol=dict(source_sha256=digest(__file__),rotation_complete_sha256=digest(rotation/'complete.json'),rotation_selector_sha256=digest(Path(__file__).with_name('photo_rotation_replay.py')),rotation_quarantined_groups=sorted(rotation_quarantined_groups),validation_augmented=False,training_rotation_shared_across_views_and_leaves=True,exhaustive_replay_protocol_sha256=digest(p/'phase127_clinical_anchors/protocol.json'),sampler_sha256=digest(Path(__file__).with_name('clinical_replay_balance.py')),all_available_quarantined_original_training_images=True,policy_source_sha256=digest(Path(__file__).with_name('conditional_recovery_policy.py')),rank_source_sha256=digest(Path(__file__).with_name('hard_negative_ranking.py')),
        parents=parent,extra_complete_sha256=digest(extra/'complete.json'),anchor_selection_sha256=digest(p/'phase133_anchor_input_cache/protocol.json'),protected_audit_sha256=digest(protected),warm_checkpoint_sha256=digest(warm/'best.pt'),
        field_complete_sha256=digest(field/'complete.json'),field_training_indices=field_indices,field_training_photos=len(field_indices),training_mass=dict(target=.5,expanded_old_anchor=.25,old107=0.,field93=.25),added_full_field_indices=omitted,added_field_healthy=sum(rows[i]['y'] for i in omitted),methods=methods,train_indices=train.tolist(),validation_indices=val.tolist(),target_indices=target,old_anchor_indices=replay,new_source_training_images=len(new),target_weight=.5,old_anchor_weight=.25,new_source_weight=.25,
        hypothesis='Test orientation shortcuts directly with right-angle pixel rotations for every available training source and class. All actual ROIs and both crop views of each sampled photo share one randomly selected rotation; validation remains the original input. Use the same full-replay mobile head, class balancing, loss and training-convergence budget. Newly matching rotated training/validation identities quarantine whole training families, never validation cases.',
        maximum_epochs=40,minimum_epochs=20,training_ema_patience=12,validation_staleness_stops_training=False,save_every_epoch=True,draws_per_epoch=2048,batch_images=64,learning_rate=.001,weight_decay=.01,negative_loss_weight=1.,ranking_weight=.2,ranking_margin=1.,maximum_hard_negatives=4,seed=2139,
        calibration=calibration,healthy_admission=.95,existing_healthy_and_disease_preserved=True,original_safety_and_recognition_gates_unchanged=True,
        supervision='One publisher image label for actual ROI bags, mean ROI logits per crop/rotation; Healthy uses minimum view and disease maximum view; no per-leaf clinical labels',
        new_source_role='Entire collection development-only; camera/style confounding unresolved; unused remainder never independent acceptance',
        ad_independent=False,olid_independent=False,acceptance_data_used=False,new_encoder_inputs=0,
        stopping='Exactly supervised and ranking controls, minimum twenty epochs then twelve stale training-loss EMA epochs at .001 improvement, forty maximum. Stop early only if unchanged cheap gates pass. Preserve every epoch; no failed export or validation-derived thresholds.')
    freeze(root/'protocol.json',protocol)
    features=torch.from_numpy(values).cuda();truth=torch.tensor([r['y'] for r in rows],dtype=torch.float32,device='cuda');index=np.zeros((len(rows),4,2,6),np.int64);mask=np.zeros(index.shape,np.float32)
    for i,pairs in tables.items():
        for rotation,pair in enumerate(pairs):
            for view,keys in enumerate(pair):index[i,rotation,view,:len(keys)]=keys;mask[i,rotation,view,:len(keys)]=1.
    index=torch.from_numpy(index).cuda();mask=torch.from_numpy(mask).cuda();valid=mask.sum(-1)>0
    data=load(p);entries,records,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({b['path']:(b,packets[b['id']],r) for b,r in zip(before,previous)})
    sampling=torch.from_numpy(np.concatenate([.5*clinical_replay_weights(rows,target),.25*clinical_replay_weights(rows,replay),.25*clinical_replay_weights(rows,field_indices)]));vw=torch.tensor(source_class_group_weights(rows,val),device='cuda')
    def decision(probability,loss):
        errors={};healthy=selection['fixed_healthy_validation_correct'];old_h=0
        for i,score in zip(val,probability):
            row=rows[i];original,packet,retained=items[row['path']];now=recover_only(original,retained,packet,float(score),data['manifest'])['app_replay']['kind']=='HEALTHY'
            if not row['y']:errors[row['source']]=errors.get(row['source'],0)+int(now)
            elif row['source']=='olid_tomato':healthy+=int(now)
            else:old_h+=int(now)
        return (-sum(errors.values()),healthy,old_h,-loss),errors
    controls=[]
    for method in methods:
        out=root/method;out.mkdir(exist_ok=True)
        if (out/'result.json').exists():
            result=read(out/'result.json');assert result['protocol_sha256']==digest(root/'protocol.json');controls.append(result);continue
        torch.manual_seed(2139);model=LeafFeatureHead('mlp',1408).cuda();model.load_state_dict(torch.load(warm/'best.pt',weights_only=True,map_location='cpu')['model'])
        def forward(ids):
            chosen,chosen_mask,_=select_photo_rotation(index[ids],mask[ids],model.training)
            z=model(features[chosen].reshape(-1,1408)).reshape(len(ids),2,6)*calibration[0]+calibration[1]
            means=(z*chosen_mask).sum(-1)/chosen_mask.sum(-1).clamp_min(1)
            return means.min(1).values,means.max(1).values
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);history=[];best=None;stale=0;start=0
        if (out/'last.pt').exists():
            saved=torch.load(out/'last.pt',weights_only=True,map_location='cpu');assert saved['protocol_sha256']==digest(root/'protocol.json');model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);history,best,stale,start=saved['history'],saved['best'],saved['stale'],saved['epoch']+1
        ema=history[-1]['training_loss_ema'] if history else None;training_best=min((r['training_loss_ema'] for r in history),default=float('inf'));training_stale=history[-1]['training_stale'] if history else 0
        for epoch in range(start,40):
            if epoch>=20 and training_stale>=12:break
            batch_losses=[]
            model.train();order=torch.multinomial(sampling,2048,replacement=True,generator=torch.Generator().manual_seed(213900+epoch))
            for batch in order.split(64):
                ids=train[batch.numpy()];positive,negative=forward(ids);raw=torch.where(truth[ids]==1,positive,negative);loss=F.binary_cross_entropy_with_logits(raw,truth[ids])
                if method=='field_source_rank':loss=loss+.2*hard_negative_rank_loss(torch.stack([positive,negative],1),truth[ids])
                assert torch.isfinite(loss);batch_losses.append(float(loss.detach()));optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            model.eval()
            with torch.inference_mode():
                pairs=[forward(part) for part in np.array_split(val,max(1,(len(val)+63)//64))];positive=torch.cat([pair[0] for pair in pairs]);negative=torch.cat([pair[1] for pair in pairs]);raw=torch.where(truth[val]==1,positive,negative)
                loss=float((F.binary_cross_entropy_with_logits(raw,truth[val],reduction='none')*vw).sum());score,errors=decision(positive.sigmoid().cpu().numpy(),loss)
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if best is None or score>tuple(best):
                best=score;stale=0;atomic_save(dict(model=state,epoch=epoch,protocol_sha256=digest(root/'protocol.json')),out/'best.pt')
                np.savez(out/'best_validation_predictions.npz',indices=val,probability=positive.sigmoid().cpu().numpy(),protocol_sha256=digest(root/'protocol.json'),checkpoint_sha256=digest(out/'best.pt'))
            else:stale+=1
            train_loss=float(np.mean(batch_losses));ema=train_loss if ema is None else .9*ema+.1*train_loss
            if ema<training_best-.001:training_best=ema;training_stale=0
            else:training_stale+=1
            checkpoint=out/f'epoch_{epoch:03d}.pt';atomic_save(dict(model=state,epoch=epoch,protocol_sha256=digest(root/'protocol.json')),checkpoint)
            np.savez(out/f'epoch_{epoch:03d}_predictions.npz',indices=val,probability=positive.sigmoid().cpu().numpy(),protocol_sha256=digest(root/'protocol.json'),checkpoint_sha256=digest(checkpoint))
            history.append(dict(epoch=epoch,decision=score,new_false_healthy_by_source=errors,training_loss=train_loss,training_loss_ema=ema,training_stale=training_stale,checkpoint_sha256=digest(checkpoint)));print(json.dumps(dict(method=method,**history[-1])),flush=True)
            atomic_save(dict(model=state,optimizer=optimizer.state_dict(),rng=capture_rng(),epoch=epoch,history=history,best=best,stale=stale,protocol_sha256=digest(root/'protocol.json')),out/'last.pt')
        chosen=torch.load(out/'best.pt',weights_only=True,map_location='cpu');cheap=best[0]==0 and best[1]>=28
        result=dict(method=method,protocol_sha256=digest(root/'protocol.json'),checkpoint_sha256=digest(out/'best.pt'),best_decision=best,selected_epoch=chosen['epoch'],history=history,cheap_pass=cheap,passed=False,model_promoted=False,new_encoder_inputs=0)
        if cheap:result['next_action']='Extend exact-identity head predictions to all applicable consumed cases, then run unchanged full recognition/safety gates before restored verification or export'
        (out/'result.json').write_text(json.dumps(result,indent=2));controls.append(result)
    result=dict(source_sha256=digest(__file__),protocol_sha256=digest(root/'protocol.json'),controls=controls,passing=0,model_promoted=False,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False,elapsed_seconds=time.monotonic()-started)
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(controls=[dict(method=r['method'],best_decision=r['best_decision'],cheap_pass=r['cheap_pass']) for r in controls])),flush=True)


if __name__=='__main__':main()
