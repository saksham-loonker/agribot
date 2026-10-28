"""Two bounded recovery-only heads using the full audited development source."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from torch.nn import functional as F
from conditional_recovery_policy import recover_only
from hard_negative_ranking import hard_negative_rank_loss
from partial_cnn_cache import original_cnn
from partial_cnn_healthy import prefix_cut
from replay_pixel_cache import replay_pixels
from photometric_replay import lighting_variation
from large_photometric_dino import prepare as prepare_dino,trainable_state,restore_trainable
from balanced_clinical_loss import balanced_clinical_loss
from opened_context_data import load,read
from phase85_population_sft import evaluation_data
from phase114_new_supervision_features import original_features
from permissive_encoder import digest
from population_siglip_tail import freeze
from source_balance import source_class_group_weights
from training_support import atomic_save,capture_rng,restore_rng


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);root=p/'phase152_balanced_photo_teacher';root.mkdir(exist_ok=True);started=time.monotonic()
    if (root/'result.json').exists():assert read(root/'result.json')['source_sha256']==digest(__file__);return
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    values,bags,descriptors,parent=original_features(p);rows=read(p/'phase84_population_head/fitting_rows.json');selection=read(p/'phase118_conditional_recovery/protocol.json');anchors=read(p/'phase133_anchor_input_cache/protocol.json')
    extra=p/'phase114_new_supervision_features';complete=read(extra/'complete.json');assert complete['protocol_sha256']==digest(extra/'protocol.json')
    for name,field in [('new_features.npy','new_features_sha256'),('bags.json','bags_sha256'),('new_input_sha256.json','new_keys_sha256')]:assert digest(extra/name)==complete[field]
    protected=p/'phase138_full_source_reservation_overlap/result.json';assert read(protected)['overlap_images']==0 and not read(protected)['clinical_predictions_opened']
    identities=[r['input_sha256'] for r in descriptors]+read(extra/'new_input_sha256.json');values=np.concatenate([values,np.load(extra/'new_features.npy')]);assert len(values)==len(identities)
    target=anchors['target_indices'];replay=anchors['anchor_indices'];val=np.array(selection['validation_indices']);old=target+replay+val.tolist()
    tables={i:[[bags[rows[i]['path']][view] for view in ('center','full')]] for i in old};new=[]
    for path,bag in read(extra/'bags.json').items():
        if not bag['actual_leaves']:continue
        new.append(len(rows));record=bag['source_record'];tables[len(rows)]=bag['views']
        rows.append(dict(path=path,y=int(record['label']=='Healthy'),source='new107_development',group='new107:entire_publisher_collection',validation=False,label=record['label']))
    train=np.array(target+replay+new);assert len(new)>200
    val_keys={identities[key] for i in val for pair in tables[i] for keys in pair for key in keys}
    assert not ({identities[key] for i in train for pair in tables[i] for keys in pair for key in keys}&val_keys)
    assert not ({rows[i]['group'] for i in train}&{rows[i]['group'] for i in val})
    warm=p/'export_cnn_safe/pytorch_bundle.pt';calibration=[1.,0.]
    backbone,architecture,normalization=original_cnn(p);backbone=backbone.cuda().eval().requires_grad_(True);normalization={k:v.cuda() for k,v in normalization.items()}
    selected=sorted({key for pairs in tables.values() for pair in pairs for keys in pair for key in keys})
    pixel_values,pixel_lookup,pixel_audit=replay_pixels(p,selected,descriptors,identities,read(extra/'bags.json'),values,backbone,normalization)
    del backbone
    backbone,prefix_values,dino_metadata=prepare_dino(p,pixel_values,selected,identities)
    normalization['cnn_mean']=normalization['canonical_mean'];normalization['cnn_std']=normalization['canonical_std']
    methods=['balanced_photo_clinical_teacher']
    protocol=dict(source_sha256=digest(__file__),policy_source_sha256=digest(Path(__file__).with_name('conditional_recovery_policy.py')),
        parents=parent,extra_complete_sha256=digest(extra/'complete.json'),anchor_selection_sha256=digest(p/'phase133_anchor_input_cache/protocol.json'),protected_audit_sha256=digest(protected),warm_checkpoint_sha256=digest(warm),pixel_audit=pixel_audit,dino_metadata=dino_metadata,comparison_parent_protocol_sha256=digest(p/'phase147_large_clinical_teacher/protocol.json'),clinical_loss_unit='Average valid clinical views within each photo, then average photos; augmentation multiplicity never changes clinical photo weight',dino_wrapper_source_sha256=digest(Path(__file__).with_name('large_photometric_dino.py')),pixel_source_sha256=digest(Path(__file__).with_name('replay_pixel_cache.py')),photometric_source_sha256=digest(Path(__file__).with_name('photometric_replay.py')),
        methods=methods,train_indices=train.tolist(),validation_indices=val.tolist(),target_indices=target,old_anchor_indices=replay,new_source_training_images=len(new),target_weight=.5,old_anchor_weight=.25,new_source_weight=.25,
        hypothesis='Visual consumed-data triage suggests bright or underside-like Healthy leaves are a failure mode. Correct the clinical-view weighting bug: the earlier flattened-view loss overweighted four-rotation new-source images. Keep the larger teacher, inputs, sampler, seed and losses otherwise unchanged, but give each photo one averaged clinical loss with label-symmetric moderate photometric variation and balanced disease replay; preserve all legacy app verdicts and clinical labels. No leaf-side labels are invented.',
        maximum_epochs=24,minimum_epochs=8,patience=6,draws_per_epoch=512,batch_images=2,cached_head_warmup_epochs=10,cached_head_warmup_draws=1024,cached_head_warmup_batch=64,cached_head_warmup_learning_rate=.001,adapter_only_checkpoints=True,candidate_role="teacher_probe",learning_rate=.0001,weight_decay=.01,negative_loss_weight=1.,seed=2139,tf32=False,
        clinical_loss_weight=1.,binary_loss_weight=.3,clinical_loss_source_sha256=digest(Path(__file__).with_name('balanced_clinical_loss.py')),calibration=calibration,healthy_admission=.95,existing_healthy_and_disease_preserved=True,original_safety_and_recognition_gates_unchanged=True,
        supervision='One publisher image label for actual ROI bags, mean ROI logits per crop/rotation; Healthy uses minimum view and disease maximum view; no per-leaf clinical labels',
        augmentation=dict(identity_probability=.5,brightness=[.85,1.15],contrast=[.85,1.15],saturation=[.75,1.25],white_balance_per_channel=[.9,1.1],shared_across_views_of_each_training_photo=True,validation_augmented=False),
        new_source_role='Previously audited 256-photo development pilot; entire collection remains development-only, no independent acceptance' ,
        ad_independent=False,olid_independent=False,acceptance_data_used=False,new_cached_encoder_features=0,all_transformer_blocks_adapted_by_lora=True,
        stopping='One photometric LoRA supervised control, at most ten cached clinical-head warm-up epochs with early exit on a cheap pass; then minimum eight adapter epochs, six stale actual-verdict epochs, twenty-four maximum. Require >=28/40 Healthy and zero new source false Healthy before full cached screening; no restored inference or export before all unchanged gates pass.')
    freeze(root/'protocol.json',protocol)
    truth=torch.tensor([r['y'] for r in rows],dtype=torch.float32,device='cuda');index=np.zeros((len(rows),4,2,6),np.int64);mask=np.zeros(index.shape,np.float32)
    for i,pairs in tables.items():
        for rotation,pair in enumerate(pairs):
            for view,keys in enumerate(pair):index[i,rotation,view,:len(keys)]=[pixel_lookup[key] for key in keys];mask[i,rotation,view,:len(keys)]=1.
    index=torch.from_numpy(index).cuda();mask=torch.from_numpy(mask).cuda();valid=mask.sum(-1)>0
    data=load(p);entries,records,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({b['path']:(b,packets[b['id']],r) for b,r in zip(before,previous)})
    sampling=torch.from_numpy(np.concatenate([.5*source_class_group_weights(rows,target),.25*source_class_group_weights(rows,replay),.25*source_class_group_weights(rows,new)]));vw=torch.tensor(source_class_group_weights(rows,val),device='cuda')
    labels=data['manifest']['classifier']['labels'];assert labels[1]=='Healthy' and labels[8]=='Other'
    fine=torch.tensor([labels.index(r['label']) if r['label'] in labels else -1 for r in rows],device='cuda')
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
        torch.manual_seed(2139);model=backbone
        def forward(ids,augment=False,cached=False):
            actual=mask[ids]>0;flat=index[ids][actual]
            if cached:
                logits=model.classifier(model.frozen_features[flat]);margin=model.margin(logits)
            elif augment:
                owners=torch.arange(len(ids),device='cuda')[:,None,None,None].expand_as(index[ids])[actual]
                factors=torch.rand((len(ids),6),device='cuda');identity=torch.rand(len(ids),device='cuda')<.5
                b=.85+.3*factors[:,0];c=.85+.3*factors[:,1];sat=.75+.5*factors[:,2];wb=.9+.2*factors[:,3:]
                b[identity]=1.;c[identity]=1.;sat[identity]=1.;wb[identity]=1.
                tokens=prefix_values[flat].clone();changed=~identity[owners]
                if changed.any():
                    rgb=pixel_values[flat[changed]]*normalization['canonical_std']+normalization['canonical_mean'];who=owners[changed]
                    rgb=lighting_variation(rgb,b[who,None,None,None],c[who,None,None,None],sat[who,None,None,None],wb[who])
                    with torch.no_grad():tokens[changed]=model.stem((rgb-normalization['canonical_mean'])/normalization['canonical_std'])
                logits=model.from_tokens(tokens);margin=model.margin(logits)
            else:
                unique,inverse=flat.unique(return_inverse=True);raw_logits=model.from_tokens(prefix_values[unique]);logits=raw_logits[inverse];margin=model.margin(logits)
            z=pixel_values.new_zeros((len(ids),4,2,6));z[actual]=margin
            means=(z*mask[ids]).sum(-1)/mask[ids].sum(-1).clamp_min(1)
            clinic_groups=torch.arange(len(ids)*8,device='cuda').reshape(len(ids),4,2,1).expand_as(index[ids])[actual]
            totals=logits.new_zeros((len(ids)*8,9)).index_add(0,clinic_groups,logits);counts=mask[ids].sum(-1).reshape(-1);active_views=counts>0
            clinical=(totals/counts.clamp_min(1)[:,None]).reshape(len(ids),8,9)
            clinical_loss=balanced_clinical_loss(clinical,fine[ids],counts.reshape(len(ids),8)>0,1)
            return means.masked_fill(~valid[ids],float('inf')).amin((1,2)),means.masked_fill(~valid[ids],-float('inf')).amax((1,2)),clinical_loss
        model.eval()
        with torch.inference_mode():
            pixel_reference=pixel_values;pixel_shape=tuple(pixel_values.shape)
            first=forward(train[:2],augment=True);raw=torch.where(truth[train[:2]]==1,first[0],first[1]);second=forward(train[2:4],augment=True)
            assert pixel_values is pixel_reference and tuple(pixel_values.shape)==pixel_shape and torch.isfinite(raw).all() and all(torch.isfinite(value).all() for value in second)
        torch.manual_seed(2139);print(json.dumps(dict(stage='two_consecutive_batch_pixel_cache_smoke',passed=True,pixel_shape=pixel_shape)),flush=True)
        # Keep head-only fitting separate from all transformer inference.
        warm_result=out/'head_warm_result.json';warm_last=out/'head_warm_last.pt'
        if not (out/'last.pt').exists():
            warm_optimizer=torch.optim.AdamW(model.classifier.parameters(),lr=.001,weight_decay=.01);warm_history=[];warm_start=0
            if warm_last.exists():
                saved=torch.load(warm_last,weights_only=True,map_location='cpu');assert saved['protocol_sha256']==digest(root/'protocol.json')
                model.classifier.load_state_dict(saved['head']);warm_optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);warm_history=saved['history'];warm_start=saved['epoch']+1
            for warm_epoch in range(warm_start,10):
                model.train();order=torch.multinomial(sampling,1024,replacement=True,generator=torch.Generator().manual_seed(214700+warm_epoch))
                for batch in order.split(64):
                    ids=train[batch.numpy()];positive,negative,clinical_loss=forward(ids,cached=True);margin=torch.where(truth[ids]==1,positive,negative);loss=clinical_loss+.3*F.binary_cross_entropy_with_logits(margin,truth[ids])
                    assert torch.isfinite(loss);warm_optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.classifier.parameters(),1.);warm_optimizer.step()
                model.eval()
                with torch.inference_mode():
                    pairs=[forward(part,cached=True) for part in np.array_split(val,max(1,(len(val)+63)//64))];positive=torch.cat([pair[0] for pair in pairs]);negative=torch.cat([pair[1] for pair in pairs]);margin=torch.where(truth[val]==1,positive,negative)
                    loss=float((F.binary_cross_entropy_with_logits(margin,truth[val],reduction='none')*vw).sum());score,errors=decision(positive.sigmoid().cpu().numpy(),loss)
                warm_history.append(dict(epoch=warm_epoch,decision=score,new_false_healthy_by_source=errors));print(json.dumps(dict(stage='cached_clinical_head_warmup',**warm_history[-1])),flush=True)
                atomic_save(dict(head={k:v.detach().cpu().clone() for k,v in model.classifier.state_dict().items()},optimizer=warm_optimizer.state_dict(),rng=capture_rng(),epoch=warm_epoch,history=warm_history,protocol_sha256=digest(root/'protocol.json')),warm_last)
                if score[0]==0 and score[1]>=28:
                    atomic_save(dict(model=trainable_state(model),epoch=-1,stage='cached_clinical_head',protocol_sha256=digest(root/'protocol.json')),out/'best.pt')
                    np.savez(out/'best_validation_predictions.npz',indices=val,probability=positive.sigmoid().cpu().numpy(),protocol_sha256=digest(root/'protocol.json'),checkpoint_sha256=digest(out/'best.pt'))
                    candidate=dict(method=method,protocol_sha256=digest(root/'protocol.json'),checkpoint_sha256=digest(out/'best.pt'),best_decision=score,selected_epoch=-1,stage='cached_clinical_head',cheap_pass=True,passed=False,model_promoted=False,candidate_role='teacher_probe',next_action='Extend exact frozen-large-encoder features to all applicable consumed cases and run all unchanged gates; mobile student/distillation and prospective acceptance remain required')
                    (out/'result.json').write_text(json.dumps(candidate,indent=2));(root/'result.json').write_text(json.dumps(dict(source_sha256=digest(__file__),protocol_sha256=digest(root/'protocol.json'),controls=[candidate],passing=0,model_promoted=False,elapsed_seconds=time.monotonic()-started,ad_independent=False,olid_independent=False,acceptance_data_used=False),indent=2));return
            warm_result.write_text(json.dumps(dict(history=warm_history,head_checkpoint_sha256=digest(warm_last),protocol_sha256=digest(root/'protocol.json')),indent=2))
            print(json.dumps(dict(stage='cached_head_warmup_finished',adapter_training_next=True)),flush=True)
        optimizer=torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],lr=.0001,weight_decay=.01);history=[];best=None;stale=0;start=0
        if (out/'last.pt').exists():
            saved=torch.load(out/'last.pt',weights_only=True,map_location='cpu');assert saved['protocol_sha256']==digest(root/'protocol.json');restore_trainable(model,saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);history,best,stale,start=saved['history'],saved['best'],saved['stale'],saved['epoch']+1
        for epoch in range(start,24):
            if epoch>=8 and stale>=6:break
            model.train()
            for module in model.modules():
                if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
            order=torch.multinomial(sampling,512,replacement=True,generator=torch.Generator().manual_seed(213900+epoch))
            for batch in order.split(2):
                ids=train[batch.numpy()];positive,negative,clinical_loss=forward(ids,augment=True);raw=torch.where(truth[ids]==1,positive,negative);loss=clinical_loss+.3*F.binary_cross_entropy_with_logits(raw,truth[ids])
                assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            model.eval()
            with torch.inference_mode():
                pairs=[forward(part) for part in np.array_split(val,max(1,(len(val)+63)//64))];positive=torch.cat([pair[0] for pair in pairs]);negative=torch.cat([pair[1] for pair in pairs]);raw=torch.where(truth[val]==1,positive,negative)
                loss=float((F.binary_cross_entropy_with_logits(raw,truth[val],reduction='none')*vw).sum());score,errors=decision(positive.sigmoid().cpu().numpy(),loss)
            state=trainable_state(model)
            if best is None or score>tuple(best):
                best=score;stale=0;atomic_save(dict(model=state,epoch=epoch,protocol_sha256=digest(root/'protocol.json')),out/'best.pt')
                np.savez(out/'best_validation_predictions.npz',indices=val,probability=positive.sigmoid().cpu().numpy(),protocol_sha256=digest(root/'protocol.json'),checkpoint_sha256=digest(out/'best.pt'))
            else:stale+=1
            history.append(dict(epoch=epoch,decision=score,new_false_healthy_by_source=errors));print(json.dumps(dict(method=method,**history[-1])),flush=True)
            atomic_save(dict(model=state,optimizer=optimizer.state_dict(),rng=capture_rng(),epoch=epoch,history=history,best=best,stale=stale,protocol_sha256=digest(root/'protocol.json')),out/'last.pt')
        chosen=torch.load(out/'best.pt',weights_only=True,map_location='cpu');cheap=best[0]==0 and best[1]>=28
        result=dict(method=method,protocol_sha256=digest(root/'protocol.json'),checkpoint_sha256=digest(out/'best.pt'),best_decision=best,selected_epoch=chosen['epoch'],history=history,cheap_pass=cheap,passed=False,model_promoted=False,new_cached_encoder_features=0,all_transformer_blocks_adapted_by_lora=True)
        if cheap:result['next_action']='Extend exact-identity head predictions to all applicable consumed cases, then run unchanged full recognition/safety gates before restored verification or export'
        (out/'result.json').write_text(json.dumps(result,indent=2));controls.append(result)
    result=dict(source_sha256=digest(__file__),protocol_sha256=digest(root/'protocol.json'),controls=controls,passing=0,model_promoted=False,new_cached_encoder_features=0,all_transformer_blocks_adapted_by_lora=True,ad_independent=False,olid_independent=False,acceptance_data_used=False,elapsed_seconds=time.monotonic()-started)
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(controls=[dict(method=r['method'],best_decision=r['best_decision'],cheap_pass=r['cheap_pass']) for r in controls])),flush=True)


if __name__=='__main__':main()
