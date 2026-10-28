"""Exact-state continuation of full mobile adaptation, conditional on binary underfit."""
from pathlib import Path
from functools import lru_cache
import argparse,hashlib,json,math,shutil,time
import numpy as np
from PIL import Image
import torch,timm
from torch.nn import functional as F
from app_eval import canon,classifier_letterbox_input
from balanced_clinical_loss import balanced_clinical_loss
from clinical_replay_balance import clinical_replay_weights
from compact_healthy_model import primary
from conditional_recovery_policy import recover_only
from exhaustive_replay_data import exhaustive_replay_data
from full_mobile_clinical_expert import FullMobileClinicalExpert
from hard_negative_ranking import hard_negative_rank_loss
from large_photometric_dino import trainable_state,restore_trainable
from opened_context_data import load,read
from partial_dino_cache import CanonicalInputs
from permissive_encoder import digest
from phase85_population_sft import evaluation_data
from population_siglip_tail import freeze
from training_support import atomic_save,capture_rng,restore_rng


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);p=Path(parser.parse_args().campaign);out=p/'phase208_full_mobile_convergence';out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    values,identities,rows,tables,train,val,meta=exhaustive_replay_data(p);parent=p/'phase196_rotation_mobile_replay';pp=read(parent/'protocol.json');quarantine=set(pp['rotation_quarantined_groups']);train=[i for i in train if rows[i]['group'] not in quarantine];target=[i for i in meta['target'] if i in set(train)];replay=[i for i in meta['replay'] if i in set(train)];field=[i for i in meta['field_indices'] if i in set(train)];assert train==target+replay+field
    selected=sorted({key for i in train+val for view in tables[i][0] for key in view});pixel_lookup={key:i for i,key in enumerate(selected)};descriptors=meta['original_descriptors'];native=read(p/'phase172_native_large_field_features/protocol.json');by_native={r['input_sha256']:r for r in native['inputs']};field_metadata={r['parent_224_input_sha256']:by_native[r['input_sha256']] for r in native['links']}
    torch.set_num_threads(4);torch.manual_seed(2206);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;timm.layers.set_fused_attn(False)
    warm=parent/'field_source_sft/best.pt';warm_state=torch.load(warm,weights_only=True,map_location='cpu');assert warm_state['protocol_sha256']==digest(parent/'protocol.json') and digest(warm)==read(parent/'field_source_sft/result.json')['checkpoint_sha256'];_,original=primary(p);model=FullMobileClinicalExpert(original,warm_state['model'],pp['calibration']).cuda().eval();del original
    gradient_checkpointing=hasattr(model.features.primary.backbone,'set_grad_checkpointing')
    if gradient_checkpointing:model.features.primary.backbone.set_grad_checkpointing(True)
    state_bytes=sum(parameter.numel()*parameter.element_size() for parameter in model.parameters() if parameter.requires_grad);checkpoint_root=Path('/dev/shm/agribot-phase208-checkpoints');checkpoint_root.mkdir(exist_ok=True);assert shutil.disk_usage(checkpoint_root).free>6*state_bytes+(32<<20),'Preserve exact full optimizer checkpoints in dedicated RAM'
    snapshot_epochs=[79,119];assert shutil.disk_usage(checkpoint_root).free>6*state_bytes+(32<<20),'RAM reserve for atomic rolling optimizer, best and two additional full snapshots'
    parent_last=Path('/dev/shm/agribot-phase206-checkpoints/last.pt');parent_sha='701e2749a461989e08d0d517cacd88e9cd893c05c33d9b956192447cd695b1b7';assert digest(parent_last)==parent_sha
    protocol=dict(parent_checkpoint_sha256=parent_sha,parent_protocol_sha256_206=digest(p/'phase206_full_mobile_supervision/protocol.json'),source_sha256=digest(__file__),expert_source_sha256=digest(Path(__file__).with_name('full_mobile_clinical_expert.py')),data_source_sha256=digest(Path(__file__).with_name('exhaustive_replay_data.py')),clinical_source_sha256=digest(Path(__file__).with_name('balanced_clinical_loss.py')),sampler_source_sha256=digest(Path(__file__).with_name('clinical_replay_balance.py')),original_bundle_sha256=digest(p/'export_cnn_safe/pytorch_bundle.pt'),warm_head_sha256=digest(warm),parent_protocol_sha256=digest(parent/'protocol.json'),
        train_indices=train,validation_indices=val,selected_input_sha256=[identities[key] for key in selected],target_indices=target,old_replay_indices=replay,field_indices=field,full_original_replay=True,source107_continued_replay=False,
        hypothesis='Determine whether the complete mobile encoder is binary-underfit after forty epochs. Resume exact final model, optimizer and RNG only if fixed training-only binary audit BCE exceeds .03. Continue the unchanged objective with separate component logging and training binary EMA stopping. This is a convergence control, not another validation-selected objective.',
        training_mass=dict(target=.5,old_replay=.25,field93=.25),loss=dict(binary=1.,clinical=.3,ranking=.2),learning_rates=dict(dino=.000002,cnn=.00002,head=.0001),continuation_lr_multiplier_cosine=[.25,.05],weight_decay=.01,seed=2206,
        augmentation='One shared random exact right-angle rotation per training photo, applied to all its actual ROIs and both crops; no color changes, no per-ROI clinical labels',validation_augmented=False,maximum_epochs=120,minimum_epochs=24,additional_epochs=80,binary_underfit_audit_threshold=.03,training_ema_patience=12,training_ema_minimum_improvement=.001,draws_per_epoch=512,batch_photos=4,gradient_checkpointing=gradient_checkpointing,
        full_weight_snapshot_epochs=snapshot_epochs,full_weight_snapshot_storage=str(checkpoint_root),parent_checkpoint_backup_required=True,all_epoch_predictions_saved=True,model_state_digest_each_epoch=True,optimizer_and_rng_checkpoint_each_epoch=True,optimizer_checkpoint_storage=str(checkpoint_root),optimizer_checkpoint_volatile=True,
        new_detector_inputs=0,frozen_feature_cache_reused_only_for_initial_parity=True,adapted_features_never_reused_under_original_weight_identity=True,existing_healthy_and_named_disease_exact=True,original_gates_unchanged=True,ad_independent=False,olid_independent=False,acceptance_data_used=False,
        stopping='If training-only audit BCE <= .03, stop without training. Otherwise at least twenty-four additional epochs, then twelve stale binary-loss EMA epochs; at most eighty additional epochs. Exit early only for >=28/40 Healthy and zero new false Healthy. Original validation denominator, labels and recognition gates unchanged.')
    freeze(out/'protocol.json',protocol);required=len(selected)*3*224*224*4;free,_=torch.cuda.mem_get_info();assert free>required+(8<<30),'Preserve GPU activation/optimizer headroom';pixel_bank=torch.empty((len(selected),3,224,224),dtype=torch.float32,device='cuda').contiguous(memory_format=torch.channels_last);pixels=CanonicalInputs(p)
    @lru_cache(maxsize=4)
    def image(path,sha):
        assert digest(path)==sha
        with Image.open(path) as im:return np.asarray(im.convert('RGB'))
    with torch.inference_mode():
        for position,key in enumerate(selected):
            if key<len(descriptors):
                row=descriptors[key];value=pixels(row if 'kind' in row else dict(row,kind='new',policy=row['view']))
            else:
                assert key>=meta['field_offset'];row=field_metadata[identities[key]];value=(canon.classifier_input if row['view']=='center' else classifier_letterbox_input)(image(row['path'],row['file_sha256']),224,row['box'])
            assert value.shape==(1,224,224,3) and value.dtype==np.float32 and hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()==identities[key];pixel_bank[position]=torch.from_numpy(value[0]).cuda().permute(2,0,1)
            if position%512==0 or position+1==len(selected):print(json.dumps(dict(stage='reuse_exact_canonical_pixels',completed=position+1,total=len(selected))),flush=True)
    pixels.close();image.cache_clear();audit=selected[:8]+selected[-8:]
    with torch.inference_mode():actual=model.features.components(pixel_bank[[pixel_lookup[key] for key in audit]].contiguous(memory_format=torch.channels_last).permute(0,2,3,1))[0].cpu().numpy()
    feature_error=float(np.abs(actual-values[audit]).max());assert feature_error<.002,('Full mobile preprocessing/encoder differs before adaptation',feature_error)
    index=np.zeros((len(rows),2,6),np.int64);mask=np.zeros(index.shape,np.float32)
    for i in train+val:
        for view,keys in enumerate(tables[i][0]):index[i,view,:len(keys)]=[pixel_lookup[key] for key in keys];mask[i,view,:len(keys)]=1.
    index=torch.from_numpy(index).cuda();mask=torch.from_numpy(mask).cuda();valid=mask.sum(-1)>0;truth=torch.tensor([r['y'] for r in rows],dtype=torch.float32,device='cuda');data=load(p);labels=data['manifest']['classifier']['labels'];assert labels[1]=='Healthy';fine=torch.tensor([labels.index(r['label']) if r['label'] in labels else -1 for r in rows],device='cuda');entries,_,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({r['path']:(r,packets[r['id']],retained) for r,retained in zip(before,previous)});selection=read(p/'phase118_conditional_recovery/protocol.json')
    def forward(ids):
        actual=mask[ids]>0;owner=torch.arange(len(ids),device='cuda').reshape(-1,1,1).expand_as(index[ids])[actual];inputs=pixel_bank[index[ids][actual]]
        if model.training:
            turns=torch.randint(4,(len(ids),),device='cuda')[owner];options=torch.stack([torch.rot90(inputs,turn,(2,3)) for turn in range(4)],1);inputs=options[torch.arange(len(inputs),device='cuda'),turns]
        binary,reference=model(inputs.contiguous(memory_format=torch.channels_last).permute(0,2,3,1));z=binary.new_zeros((len(ids),2,6));z[actual]=binary;counts=mask[ids].sum(-1);means=(z*mask[ids]).sum(-1)/counts.clamp_min(1);crop_owner=torch.arange(len(ids)*2,device='cuda').reshape(len(ids),2,1).expand_as(index[ids])[actual];totals=reference.new_zeros((len(ids)*2,9)).index_add(0,crop_owner,reference);clinical=(totals/counts.reshape(-1).clamp_min(1)[:,None]).reshape(len(ids),2,9)
        return means.masked_fill(~valid[ids],float('inf')).min(1).values,means.masked_fill(~valid[ids],-float('inf')).max(1).values,balanced_clinical_loss(clinical,fine[ids],valid[ids],1)
    def decision(probability,loss):
        errors={};healthy=selection['fixed_healthy_validation_correct'];old_h=0;kinds=[]
        for i,score in zip(val,probability):
            row=rows[i];original,packet,retained=items[row['path']];kind=recover_only(original,retained,packet,float(score),data['manifest'])['app_replay']['kind'];now=kind=='HEALTHY';kinds.append(kind)
            if not row['y']:errors[row['source']]=errors.get(row['source'],0)+int(now)
            elif row['source']=='olid_tomato':healthy+=int(now)
            else:old_h+=int(now)
        return (-sum(errors.values()),healthy,old_h,-loss),errors,kinds
    with torch.inference_mode():
        parts=[forward(part) for part in np.array_split(np.array(val),max(1,(len(val)+3)//4))];positive=torch.cat([r[0] for r in parts]);negative=torch.cat([r[1] for r in parts]);warm_loss=float(F.binary_cross_entropy_with_logits(torch.where(truth[val]==1,positive,negative),truth[val]));warm_probability=positive.sigmoid().cpu().numpy();warm_decision,_,warm_kinds=decision(warm_probability,warm_loss)
    cached=np.load(parent/'field_source_sft/best_validation_predictions.npz');assert np.array_equal(cached['indices'],val) and str(cached['checkpoint_sha256'])==digest(warm);_,_,old_kinds=decision(cached['probability'],warm_loss);assert old_kinds==warm_kinds;probability_error=float(np.abs(cached['probability']-warm_probability).max());assert probability_error<.0005
    rng=capture_rng();buffers={name:value.detach().clone() for name,value in model.named_buffers()};model.train();test_ids=np.array(train[:2]);positive,negative,clinical=forward(test_ids);test_loss=F.binary_cross_entropy_with_logits(torch.where(truth[test_ids]==1,positive,negative),truth[test_ids])+.3*clinical;test_loss.backward();dino_grad=sum(float(parameter.grad.abs().sum()) for parameter in model.features.primary.backbone.parameters() if parameter.grad is not None);cnn_grad=sum(float(parameter.grad.abs().sum()) for parameter in model.features.primary.cnn.parameters() if parameter.grad is not None);assert dino_grad>0 and cnn_grad>0
    assert all(torch.equal(value,buffers[name]) for name,value in model.named_buffers());model.zero_grad(set_to_none=True);model.eval();restore_rng(rng);freeze(out/'warm_parity.json',dict(protocol_sha256=digest(out/'protocol.json'),maximum_original_feature_error=feature_error,maximum_probability_error=probability_error,all_validation_verdicts_equal=True,warm_decision=warm_decision,real_dino_gradient_nonzero=True,real_cnn_gradient_nonzero=True,all_running_buffers_unchanged=True))
    optimizer=torch.optim.AdamW([dict(params=[parameter for parameter in model.features.primary.backbone.parameters() if parameter.requires_grad],lr=.000002),dict(params=[parameter for parameter in model.features.primary.cnn.parameters() if parameter.requires_grad],lr=.00002),dict(params=model.head.parameters(),lr=.0001)],weight_decay=.01);sampling=torch.from_numpy(np.concatenate([.5*clinical_replay_weights(rows,target),.25*clinical_replay_weights(rows,replay),.25*clinical_replay_weights(rows,field)]));history=[];best=warm_decision;start=0;torch.manual_seed(2206)
    if (checkpoint_root/'last.pt').exists():
        saved=torch.load(checkpoint_root/'last.pt',weights_only=True,map_location='cpu');assert saved['protocol_sha256']==digest(out/'protocol.json');restore_trainable(model,saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);history,best,start=saved['history'],saved['best'],saved['epoch']+1
    else:
        atomic_save(dict(model=trainable_state(model),epoch=-1,protocol_sha256=digest(out/'protocol.json')),checkpoint_root/'best.pt');np.savez(out/'best_validation_predictions.npz',indices=np.array(val),probability=warm_probability,protocol_sha256=digest(out/'protocol.json'),checkpoint_sha256=digest(checkpoint_root/'best.pt'))
    assert start==0,'This standalone continuation must start once from its exact parent; do not silently rerun'
    parent_state=torch.load(parent_last,weights_only=True,map_location='cpu');assert parent_state['epoch']==39 and parent_state['protocol_sha256']==digest(p/'phase206_full_mobile_supervision/protocol.json')
    restore_trainable(model,parent_state['model']);optimizer.load_state_dict(parent_state['optimizer']);restore_rng(parent_state['rng']);start=40
    model.eval();audit_order=torch.multinomial(sampling,256,replacement=True,generator=torch.Generator().manual_seed(220800));audit_components=[]
    with torch.inference_mode():
        for batch in audit_order.split(4):
            ids=np.array(train)[batch.numpy()];pos,neg,clinical=forward(ids);binary=F.binary_cross_entropy_with_logits(torch.where(truth[ids]==1,pos,neg),truth[ids]);rank=hard_negative_rank_loss(torch.stack((pos,neg),1),truth[ids],margin=1.,maximum_negatives=4);audit_components.append([float(binary),float(clinical),float(rank)])
    audit_mean=np.mean(audit_components,axis=0).tolist();freeze(out/'training_component_audit.json',dict(binary_bce=audit_mean[0],clinical=audit_mean[1],ranking=audit_mean[2],draws=256,training_only=True,augmented=False,seed=220800,parent_checkpoint_sha256=parent_sha))
    print(json.dumps(dict(stage='training_component_audit',components=audit_mean)),flush=True)
    if audit_mean[0]<=.03:
        (out/'result.json').write_text(json.dumps(dict(protocol_sha256=digest(out/'protocol.json'),training_component_audit=audit_mean,continuation_started=False,reason='Training binary BCE already low; more convergence not justified',model_promoted=False,acceptance_data_used=False),indent=2));return
    # The laptop must verify both parent files before this explicit rolling-state relocation.
    receipt=read(p/'phase206_full_mobile_supervision/checkpoint_backup.json');assert receipt['verified'] and receipt['last_sha256']==parent_sha
    parent_last.replace(checkpoint_root/'last.pt');freeze(out/'continuation_restore.json',dict(parent_checkpoint_sha256=parent_sha,parent_epoch=39,parent_optimizer_and_rng_restored=True,parent_backup=receipt,rolling_state_relocated=True))
    ema=None;training_best=float('inf');training_stale=0
    for epoch in range(start,120):
        if epoch>=64 and training_stale>=12:break
        epoch_started=time.monotonic();factor=.05+.1*(1+math.cos(math.pi*(epoch-40)/79))
        for group,base_lr in zip(optimizer.param_groups,(.000002,.00002,.0001)):group['lr']=base_lr*factor
        model.train();order=torch.multinomial(sampling,512,replacement=True,generator=torch.Generator().manual_seed(220600+epoch));batch_losses=[];binary_losses=[];clinical_losses=[];rank_losses=[]
        for batch in order.split(4):
            batch_ids=np.array(train)[batch.numpy()];positive,negative,clinical=forward(batch_ids);raw=torch.where(truth[batch_ids]==1,positive,negative);rank=hard_negative_rank_loss(torch.stack((positive,negative),1),truth[batch_ids],margin=1.,maximum_negatives=4);binary=F.binary_cross_entropy_with_logits(raw,truth[batch_ids]);loss=binary+.3*clinical+.2*rank;assert torch.isfinite(loss);batch_losses.append(float(loss.detach()));binary_losses.append(float(binary.detach()));clinical_losses.append(float(clinical.detach()));rank_losses.append(float(rank.detach()));optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        model.eval()
        with torch.inference_mode():
            parts=[forward(part) for part in np.array_split(np.array(val),max(1,(len(val)+3)//4))];positive=torch.cat([r[0] for r in parts]);negative=torch.cat([r[1] for r in parts]);loss=float(F.binary_cross_entropy_with_logits(torch.where(truth[val]==1,positive,negative),truth[val]));probability=positive.sigmoid().cpu().numpy();score,errors,_=decision(probability,loss)
        state=trainable_state(model);state_digest=hashlib.sha256()
        for name,value in sorted(state.items()):
            a=value.contiguous().numpy();state_digest.update(str((name,str(a.dtype),a.shape)).encode());state_digest.update(a.tobytes())
        if score>tuple(best):
            best=score;atomic_save(dict(model=state,epoch=epoch,protocol_sha256=digest(out/'protocol.json')),checkpoint_root/'best.pt');np.savez(out/'best_validation_predictions.npz',indices=np.array(val),probability=probability,protocol_sha256=digest(out/'protocol.json'),checkpoint_sha256=digest(checkpoint_root/'best.pt'))
        snapshot=checkpoint_root/f'epoch_{epoch:03d}.pt'
        if epoch in snapshot_epochs or score[0]==0 and score[1]>=28:
            assert shutil.disk_usage(checkpoint_root).free>state_bytes+3*state_bytes+(32<<20),'Preserve RAM reserve for next atomic optimizer write';atomic_save(dict(model=state,epoch=epoch,protocol_sha256=digest(out/'protocol.json')),snapshot)
        snapshot_sha=digest(snapshot) if snapshot.exists() else None;np.savez(out/f'epoch_{epoch:03d}_predictions.npz',indices=np.array(val),probability=probability,protocol_sha256=digest(out/'protocol.json'),model_state_sha256=state_digest.hexdigest(),checkpoint_sha256=snapshot_sha or '')
        train_loss=float(np.mean(batch_losses));binary_loss=float(np.mean(binary_losses));ema=binary_loss if ema is None else .9*ema+.1*binary_loss
        if ema<training_best-.001:training_best=ema;training_stale=0
        else:training_stale+=1
        history.append(dict(epoch=epoch,decision=score,new_false_healthy_by_source=errors,training_loss=train_loss,training_binary_loss=binary_loss,training_clinical_loss=float(np.mean(clinical_losses)),training_rank_loss=float(np.mean(rank_losses)),training_loss_ema=ema,ema_scope="binary_bce",training_stale=training_stale,model_state_sha256=state_digest.hexdigest(),checkpoint_sha256=snapshot_sha,elapsed_seconds=time.monotonic()-epoch_started));print(json.dumps(history[-1]),flush=True)
        atomic_save(dict(model=state,optimizer=optimizer.state_dict(),rng=capture_rng(),epoch=epoch,history=history,best=best,protocol_sha256=digest(out/'protocol.json')),checkpoint_root/'last.pt')
        if best[0]==0 and best[1]>=28:break
    chosen=torch.load(checkpoint_root/'best.pt',weights_only=True,map_location='cpu');result=dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),checkpoint_sha256=digest(checkpoint_root/'best.pt'),selected_epoch=chosen['epoch'],best_decision=best,history=history,cheap_pass=best[0]==0 and best[1]>=28,model_promoted=False,checkpoint_root=str(checkpoint_root),checkpoint_volatile=True,ram_snapshot_epochs=snapshot_epochs,parent_training_component_audit=audit_mean,exact_gpu_pixel_bank_bytes=required,new_detector_inputs=0,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)


if __name__=='__main__':
    raise SystemExit('Retired restore preflight: phase209_full_mobile_convergence.py contains the strict restore repair. Read its completed result before any explicit continuation; the rolling parent was relocated. Original phase208 execution bytes are preserved in executed_sources.zip.')
