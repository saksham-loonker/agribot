"""Teach damage distinctions using correctly recognized disease anchors too."""
from pathlib import Path
import argparse,collections,hashlib,json,time
import numpy as np
import torch
from torch.nn import functional as F
from clinical_anchor_head import ClinicalAnchorHead,image_condition_loss
from conditional_recovery_losses import cross_source_contrastive
from conditional_recovery_policy import recover_only
from opened_context_data import load,read
from opened_guard_screen import OpenedGuardScreen
from permissive_encoder import digest
from phase2_select import retain
from phase5_scene_ablation import photo_metrics
from phase22_accept import paired_interval
from phase81_consumed_olid_features import prepare as prepare_whole
from phase85_population_sft import evaluation_data
from phase83_eligible_operating_points import safety
from phase114_new_supervision_features import original_features
from population_siglip_tail import freeze
from source_balance import source_class_group_weights
from training_support import atomic_save,capture_rng,restore_rng


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase127_clinical_anchors';out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    torch.set_num_threads(4);torch.manual_seed(2127);torch.backends.cuda.matmul.allow_tf32=False
    leaf,bags,descriptors,parent=original_features(p);rows=read(p/'phase84_population_head/fitting_rows.json');selection=read(p/'phase118_conditional_recovery/protocol.json')
    data=load(p);entries,records,packets,before,previous=evaluation_data(p,data);whole,whole_metadata=prepare_whole(p,data)
    items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({b['path']:(b,packets[b['id']],r) for b,r in zip(before,previous)})
    labels=data['manifest']['classifier']['labels'];healthy_index=labels.index('Healthy');coarse={'Diseased_unspecified'}
    paths=sorted(path for path,pair in bags.items() if pair['center'] and path in whole);path_index={path:i for i,path in enumerate(paths)}
    paired=np.stack([[leaf[bags[path][view]].mean(0) for view in ('center','full')] for path in paths]);whole_values=np.stack([whole[path] for path in paths])
    source_files=read(p/'phase29_compact/source_files.json');source_files.update({r['path']:r['file_sha256'] for r in read(p/'phase24_olid_source/sealed_manifest.json')})
    validation=[i for i,r in enumerate(rows) if r['validation']]
    val_pixels={descriptors[key]['input_sha256'] for i in validation if rows[i]['path'] in bags for view in bags[rows[i]['path']].values() for key in view}
    val_files={source_files[r['path']] for r in rows if r['validation'] and r['path'] in source_files}
    candidates=[i for i,r in enumerate(rows) if not r['validation'] and r['path'] in path_index and r['label'] in set(labels)|coarse]
    excluded={rows[i]['group'] for i in candidates if source_files.get(rows[i]['path']) in val_files or any(descriptors[key]['input_sha256'] in val_pixels for view in bags[rows[i]['path']].values() for key in view)}
    candidates=[i for i in candidates if rows[i]['group'] not in excluded]
    assert not ({rows[i]['group'] for i in candidates}&{rows[i]['group'] for i in validation})
    for i in candidates:assert rows[i]['y']==int(rows[i]['label']=='Healthy'),'Binary and clinical labels disagree'
    target=[i for i in selection['train_indices'] if rows[i]['source']=='olid_tomato' and i in set(candidates)]
    anchors=[i for i in candidates if i not in set(target)];train=np.array(target+anchors);val=np.array(selection['validation_indices'])
    assert len(target)>100 and len(anchors)>100 and {rows[i]['y'] for i in target}=={0,1}
    draw_weights=np.concatenate([.5*source_class_group_weights(rows,target),.5*source_class_group_weights([dict(r,anchor_class=r['label']) for r in rows],anchors,target='anchor_class')])
    source_names=sorted({r['source'] for r in rows});sources=torch.tensor([source_names.index(r['source']) for r in rows],device='cuda')
    family_names=sorted({r['group'] for r in rows});families=torch.tensor([family_names.index(r['group']) for r in rows],device='cuda')
    raw_tags={i:('Other|'+str(items[rows[i]['path']][0].get('raw_label','Other')).strip().lower()) for i in candidates if rows[i]['label']=='Other'}
    tag_counts=collections.Counter(raw_tags.values());subtypes=sorted(tag for tag,n in tag_counts.items() if n>=10 and tag!='Other|other')
    warm=p/'phase88_diagnostic_bag_head/mlp';wr=read(warm/'result.json');assert wr['checkpoint_sha256']==digest(warm/'best.pt');calibration=wr['calibration']
    methods=['clinical_anchors','condition_subtypes','subtype_contrastive']
    protocol=dict(source_sha256=digest(__file__),head_source_sha256=digest(Path(__file__).with_name('clinical_anchor_head.py')),parents=parent,whole_metadata=whole_metadata,
                  selection_protocol_sha256=digest(p/'phase118_conditional_recovery/protocol.json'),warm_checkpoint_sha256=digest(warm/'best.pt'),train_indices=train.tolist(),validation_indices=val.tolist(),
                  target_images=len(target),anchor_images=len(anchors),recognized_disease_anchors=sum(items[rows[i]['path']][2]['app_replay']['kind']=='DISEASE' for i in anchors),
                  anchor_label_counts=dict(collections.Counter(rows[i]['label'] for i in anchors)),quarantined_overlap_groups=sorted(excluded),all_validation_raw_and_roi_identities_quarantined=True,
                  methods=methods,extended_subtypes=subtypes,minimum_subtype_training_photos=10,target_training_mass=.5,anchor_training_mass=.5,
                  feature_pooling='Mean original actual ROI feature vectors separately for center/full image views; no per-leaf clinical labels',
                  hypothesis='Routed-only training omits correctly recognized disease exemplars. Add clinical-class-balanced training anchors and a shared condition-classification objective to retain damage distinctions while learning rejected Healthy cases.',
                  maximum_epochs=120,minimum_epochs=25,patience=15,draws_per_epoch=1024,batch_images=64,learning_rate=.001,condition_loss_weight=.3,contrastive_weight=.05,seed=2127,
                  admission_probability=.95,auxiliary_agreement='Both crop-view fine-class winners must be Healthy',existing_healthy_and_named_disease_preserved=True,
                  new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False,
                  stopping='Three predeclared anchor-supervised controls; fixed routing/evidence guards. Full cached screen only after >=28/40 Healthy with zero new source false-Healthy; no restored inference or export unless all original and consumed gates pass.')
    freeze(out/'protocol.json',protocol)
    feature=torch.from_numpy(paired).cuda();context=torch.from_numpy(whole_values).cuda();row_paths=torch.tensor([path_index.get(r['path'],0) for r in rows],device='cuda')
    truth=torch.tensor([r['y'] for r in rows],dtype=torch.float32,device='cuda');sampling=torch.from_numpy(draw_weights);vw=torch.tensor(source_class_group_weights(rows,val),device='cuda')
    def score(probability,loss):
        errors={};healthy=selection['fixed_healthy_validation_correct'];old_h=0
        for offset,i in enumerate(val):
            row=rows[i];original,packet,retained=items[row['path']];after=recover_only(original,retained,packet,float(probability[offset]),data['manifest']);now=after['app_replay']['kind']=='HEALTHY'
            if not row['y']:errors[row['source']]=errors.get(row['source'],0)+int(now)
            elif row['source']=='olid_tomato':healthy+=int(now)
            else:old_h+=int(now)
        return (-sum(errors.values()),healthy,old_h,-loss),errors
    controls=[]
    for method in methods:
        folder=out/method;folder.mkdir(exist_ok=True)
        if (folder/'result.json').exists():
            cached=read(folder/'result.json');assert cached['protocol_sha256']==digest(out/'protocol.json');controls.append(cached);continue
        names=labels+(subtypes if method!='clinical_anchors' else []);fine_map={name:i for i,name in enumerate(names)};fine=[]
        for i,row in enumerate(rows):
            label=row['label'];tag='Other|'+str(items[row['path']][0].get('raw_label','Other')).strip().lower() if label=='Other' else label
            fine.append(fine_map.get(tag,fine_map.get(label,-1)))
        fine=torch.tensor(fine,dtype=torch.long,device='cuda');torch.manual_seed(2127)
        model=ClinicalAnchorHead(torch.load(warm/'best.pt',weights_only=True,map_location='cpu')['model'],calibration,len(names)).cuda();torch.manual_seed(2127)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);history=[];best=None;stale=0;start=0
        if (folder/'last.pt').exists():
            saved=torch.load(folder/'last.pt',weights_only=True,map_location='cpu');assert saved['protocol_sha256']==digest(out/'protocol.json');model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);history,best,stale,start=saved['history'],saved['best'],saved['stale'],saved['epoch']+1
        for epoch in range(start,120):
            if epoch>=25 and stale>=15:break
            model.train();order=torch.multinomial(sampling,1024,replacement=True,generator=torch.Generator().manual_seed(212700+epoch))
            for batch in order.split(64):
                ids=train[batch.numpy()];z,f,h=model(feature[row_paths[ids]],context[row_paths[ids]]);raw=torch.where(truth[ids]==1,z.min(1).values,z.max(1).values)
                loss=F.binary_cross_entropy_with_logits(raw,truth[ids])+.3*image_condition_loss(f,fine[ids],healthy_index)
                if method=='subtype_contrastive':
                    exact=fine[ids]>=0
                    if exact.any():loss=loss+.05*cross_source_contrastive(h[exact],fine[ids][exact],sources[ids][exact],families[ids][exact])
                assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            model.eval()
            with torch.inference_mode():
                z,f,h=model(feature[row_paths[val]],context[row_paths[val]]);raw=torch.where(truth[val]==1,z.min(1).values,z.max(1).values);loss=float((F.binary_cross_entropy_with_logits(raw,truth[val],reduction='none')*vw).sum())
                probability=z.min(1).values.sigmoid()*(f.argmax(-1)==healthy_index).all(1);decision,errors=score(probability.cpu().numpy(),loss)
            state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            if best is None or decision>tuple(best):
                best=decision;stale=0;atomic_save(dict(model=state,epoch=epoch,decision=decision,fine_labels=names,protocol_sha256=digest(out/'protocol.json')),folder/'best.pt')
                np.savez(folder/'best_validation_predictions.npz',indices=val,probability=probability.cpu().numpy(),paired_margin=z.cpu().numpy(),fine_logits=f.cpu().numpy())
            else:stale+=1
            history.append(dict(epoch=epoch,decision=decision,new_false_healthy_by_source=errors));print(json.dumps(dict(method=method,**history[-1])),flush=True)
            atomic_save(dict(model=state,optimizer=optimizer.state_dict(),rng=capture_rng(),epoch=epoch,history=history,best=best,stale=stale,protocol_sha256=digest(out/'protocol.json')),folder/'last.pt')
        chosen=torch.load(folder/'best.pt',weights_only=True,map_location='cpu');cheap=best[0]==0 and best[1]>=28
        result=dict(method=method,cheap_pass=cheap,best_decision=best,selected_epoch=chosen['epoch'],history=history,checkpoint_sha256=digest(folder/'best.pt'),protocol_sha256=digest(out/'protocol.json'),passed=False,full_cached_screen_run=False,new_encoder_inputs=0)
        if cheap:
            model.load_state_dict(chosen['model']);model.eval();applicable={path for path,(original,packet,retained) in items.items() if retained['app_replay']['kind'] not in ('HEALTHY','DISEASE') and recover_only(original,retained,packet,1.,data['manifest'])['app_replay']['kind']=='HEALTHY'}
            assert applicable<=set(path_index),'Full-screen evidence missing; never silently skip possible false Healthy'
            cached=np.load(folder/'best_validation_predictions.npz');probabilities={rows[i]['path']:float(v) for i,v in zip(cached['indices'],cached['probability'])}
            remaining=sorted(applicable-set(probabilities))
            with torch.inference_mode():
                for start in range(0,len(remaining),64):
                    part=remaining[start:start+64];ids=[path_index[path] for path in part];z,f,h=model(feature[ids],context[ids]);v=(z.min(1).values.sigmoid()*(f.argmax(-1)==healthy_index).all(1)).cpu().numpy();probabilities.update({path:float(value) for path,value in zip(part,v)})
            def revised(row,packet,retained):return recover_only(row,retained,packet,probabilities.get(row['path'],0.),data['manifest'])
            def revise(row):return revised(row,*entries[id(row)])
            screen=OpenedGuardScreen(data).evaluate(revise);after=[revised(b,packets[b['id']],r) for b,r in zip(before,previous)]
            vi=[i for i,row in enumerate(rows[-len(records):]) if row['validation']];vr,vb,va=[[group[i] for i in vi] for group in (records,before,after)]
            vm=photo_metrics(va);metrics=photo_metrics(after);bm=photo_metrics(before)
            for group,metric in [(after,metrics),(before,bm)]:metric['unsupported_tomato']=photo_metrics([r for r in group if r['label']=='Other'])['all']
            vsafe,vs=safety(vr,vb,va);asafe,alls=safety(records,before,after);gain=paired_interval(vr,vb,va,lambda r:r['label']=='Healthy',lambda r:int(r['app_replay']['kind']=='HEALTHY'))
            checks=dict(screen['checks'],consumed_validation_healthy=vm['healthy']['correct_rate']>=.7,consumed_full_healthy=metrics['healthy']['correct_rate']>=.7,consumed_gain=gain is not None and gain['point']>=.1 and gain['bootstrap_two_sided_95'][0]>0,
                        consumed_validation_safety=vsafe,consumed_full_safety=asafe,consumed_full_retention=retain({'olid':metrics},{'olid':bm}))
            result.update(passed=all(checks.values()),full_cached_screen_run=True,checks=checks,old_screen=screen,consumed_validation=vm,consumed_all=metrics,validation_safety=vs,all_safety=alls,validation_gain=gain)
            (folder/'probabilities.json').write_text(json.dumps(probabilities))
        (folder/'result.json').write_text(json.dumps(result,indent=2));controls.append(result)
    result=dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),controls=controls,passing=sum(r['passed'] for r in controls),new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False,elapsed_seconds=time.monotonic()-started,model_promoted=False)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(passing=result['passing'],controls=[dict(method=r['method'],decision=r['best_decision'],passed=r['passed']) for r in controls])),flush=True)


if __name__=='__main__':main()
