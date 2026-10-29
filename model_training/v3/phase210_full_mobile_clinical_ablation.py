"""End-to-end copied mobile feature encoders, exact pixels and full replay."""
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
from large_photometric_dino import trainable_state
from strict_trainable_restore import restore_trainable_exact as restore_trainable
from concurrent.futures import ThreadPoolExecutor
from opened_context_data import load,read
from partial_dino_cache import CanonicalInputs
from permissive_encoder import digest
from phase85_population_sft import evaluation_data
from population_siglip_tail import freeze
from training_support import atomic_save,capture_rng,restore_rng


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);p=Path(parser.parse_args().campaign);out=p/'phase210_full_mobile_clinical_ablation';out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    values,identities,rows,tables,train,val,meta=exhaustive_replay_data(p);parent=p/'phase196_rotation_mobile_replay';pp=read(parent/'protocol.json');quarantine=set(pp['rotation_quarantined_groups']);train=[i for i in train if rows[i]['group'] not in quarantine];target=[i for i in meta['target'] if i in set(train)];replay=[i for i in meta['replay'] if i in set(train)];field=[i for i in meta['field_indices'] if i in set(train)];assert train==target+replay+field
    selected=sorted({key for i in train+val for view in tables[i][0] for key in view});pixel_lookup={key:i for i,key in enumerate(selected)};descriptors=meta['original_descriptors'];native=read(p/'phase172_native_large_field_features/protocol.json');by_native={r['input_sha256']:r for r in native['inputs']};field_metadata={r['parent_224_input_sha256']:by_native[r['input_sha256']] for r in native['links']}
    torch.set_num_threads(4);torch.manual_seed(2206);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;timm.layers.set_fused_attn(False)
    warm=parent/'field_source_sft/best.pt';warm_state=torch.load(warm,weights_only=True,map_location='cpu');assert warm_state['protocol_sha256']==digest(parent/'protocol.json') and digest(warm)==read(parent/'field_source_sft/result.json')['checkpoint_sha256'];_,original=primary(p);model=FullMobileClinicalExpert(original,warm_state['model'],pp['calibration']).cuda().eval();del original
    gradient_checkpointing=hasattr(model.features.primary.backbone,'set_grad_checkpointing')
    if gradient_checkpointing:model.features.primary.backbone.set_grad_checkpointing(True)
    state_bytes=sum(parameter.numel()*parameter.element_size() for parameter in model.parameters() if parameter.requires_grad);checkpoint_root=Path('/dev/shm/agribot-phase210-checkpoints');checkpoint_root.mkdir(exist_ok=True);assert shutil.disk_usage(checkpoint_root).free>7*state_bytes+(32<<20),'Preserve exact full optimizer checkpoints in dedicated RAM'
    snapshot_epochs=[39]
    protocol=dict(matched_control_phase206_protocol_sha256=digest(p/'phase206_full_mobile_supervision/protocol.json'),restore_source_sha256=digest(Path(__file__).with_name('strict_trainable_restore.py')),canonical_pixel_io='Six read-only threads and batches of 32 SHA-verified inputs',source_sha256=digest(__file__),expert_source_sha256=digest(Path(__file__).with_name('full_mobile_clinical_expert.py')),data_source_sha256=digest(Path(__file__).with_name('exhaustive_replay_data.py')),clinical_source_sha256=digest(Path(__file__).with_name('balanced_clinical_loss.py')),sampler_source_sha256=digest(Path(__file__).with_name('clinical_replay_balance.py')),original_bundle_sha256=digest(p/'export_cnn_safe/pytorch_bundle.pt'),warm_head_sha256=digest(warm),parent_protocol_sha256=digest(parent/'protocol.json'),
        train_indices=train,validation_indices=val,selected_input_sha256=[identities[key] for key in selected],target_indices=target,old_replay_indices=replay,field_indices=field,full_original_replay=True,source107_continued_replay=False,
        hypothesis='Matched auxiliary-clinical-loss ablation: restart from exactly the same benchmark encoders and phase196 warm head as phase206, using the same data, seed, optimizer, learning rates, draws, augmentations and forty-epoch budget. Remove only the clinical-preservation term; retain BCE and ranking. The expert is conditionally routed and does not replace the original app named-disease classifier. Test whether preserving the old fine-label geometry impedes the new binary distinction; no safety or recognition gate changes',
        training_mass=dict(target=.5,old_replay=.25,field93=.25),loss=dict(binary=1.,clinical=0.,ranking=.2),learning_rates=dict(dino=.000002,cnn=.00002,head=.0001),weight_decay=.01,seed=2206,
        augmentation='One shared random exact right-angle rotation per training photo, applied to all its actual ROIs and both crops; no color changes, no per-ROI clinical labels',validation_augmented=False,maximum_epochs=40,minimum_epochs=24,training_ema_patience=12,training_ema_minimum_improvement=.001,draws_per_epoch=512,batch_photos=4,gradient_checkpointing=gradient_checkpointing,
        full_weight_snapshot_epochs=snapshot_epochs,all_epoch_predictions_saved=True,model_state_digest_each_epoch=True,optimizer_and_rng_checkpoint_each_epoch=True,optimizer_checkpoint_storage=str(checkpoint_root),optimizer_checkpoint_volatile=True,full_snapshot_storage=str(checkpoint_root),
        new_detector_inputs=0,frozen_feature_cache_reused_only_for_initial_parity=True,adapted_features_never_reused_under_original_weight_identity=True,existing_healthy_and_named_disease_exact=True,original_gates_unchanged=True,ad_independent=False,olid_independent=False,acceptance_data_used=False,
        stopping='Minimum twenty-four epochs, then twelve stale training-loss EMA epochs; forty maximum. Exit earlier only for >=28/40 Healthy and zero new false Healthy. Save every prediction, best, passing candidate and resource-planned full snapshots; no export or app replacement before complete unchanged qualification.')
    freeze(out/'protocol.json',protocol);required=len(selected)*3*224*224*4;free,_=torch.cuda.mem_get_info();assert free>required+(8<<30),'Preserve GPU activation/optimizer headroom';pixel_bank=torch.empty((len(selected),3,224,224),dtype=torch.float32,device='cuda').contiguous(memory_format=torch.channels_last);pixels=CanonicalInputs(p)
    @lru_cache(maxsize=4)
    def image(path,sha):
        assert digest(path)==sha
        with Image.open(path) as im:return np.asarray(im.convert('RGB'))
    def canonical_pixel(key):
        if key<len(descriptors):
            row=descriptors[key];value=pixels(row if 'kind' in row else dict(row,kind='new',policy=row['view']))
        else:
            assert key>=meta['field_offset'];row=field_metadata[identities[key]];value=(canon.classifier_input if row['view']=='center' else classifier_letterbox_input)(image(row['path'],row['file_sha256']),224,row['box'])
        assert value.shape==(1,224,224,3) and value.dtype==np.float32 and hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()==identities[key]
        return value[0]
    with torch.inference_mode(),ThreadPoolExecutor(max_workers=6) as pool:
        for position in range(0,len(selected),32):
            keys=selected[position:position+32];batch=np.stack(list(pool.map(canonical_pixel,keys)))
            pixel_bank[position:position+len(keys)]=torch.from_numpy(batch).cuda().permute(0,3,1,2)
            if position%512==0 or position+len(keys)==len(selected):print(json.dumps(dict(stage='reuse_exact_canonical_pixels',completed=position+len(keys),total=len(selected))),flush=True)
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
    rng=capture_rng();buffers={name:value.detach().clone() for name,value in model.named_buffers()};model.train();test_ids=np.array(train[:2]);positive,negative,clinical=forward(test_ids);test_loss=F.binary_cross_entropy_with_logits(torch.where(truth[test_ids]==1,positive,negative),truth[test_ids])+0.*clinical;test_loss.backward();dino_grad=sum(float(parameter.grad.abs().sum()) for parameter in model.features.primary.backbone.parameters() if parameter.grad is not None);cnn_grad=sum(float(parameter.grad.abs().sum()) for parameter in model.features.primary.cnn.parameters() if parameter.grad is not None);assert dino_grad>0 and cnn_grad>0
    assert all(torch.equal(value,buffers[name]) for name,value in model.named_buffers());model.zero_grad(set_to_none=True);model.eval();restore_rng(rng);freeze(out/'warm_parity.json',dict(protocol_sha256=digest(out/'protocol.json'),maximum_original_feature_error=feature_error,maximum_probability_error=probability_error,all_validation_verdicts_equal=True,warm_decision=warm_decision,real_dino_gradient_nonzero=True,real_cnn_gradient_nonzero=True,all_running_buffers_unchanged=True))
    optimizer=torch.optim.AdamW([dict(params=[parameter for parameter in model.features.primary.backbone.parameters() if parameter.requires_grad],lr=.000002),dict(params=[parameter for parameter in model.features.primary.cnn.parameters() if parameter.requires_grad],lr=.00002),dict(params=model.head.parameters(),lr=.0001)],weight_decay=.01);sampling=torch.from_numpy(np.concatenate([.5*clinical_replay_weights(rows,target),.25*clinical_replay_weights(rows,replay),.25*clinical_replay_weights(rows,field)]));history=[];best=warm_decision;start=0;torch.manual_seed(2206)
    if (checkpoint_root/'last.pt').exists():
        saved=torch.load(checkpoint_root/'last.pt',weights_only=True,map_location='cpu');assert saved['protocol_sha256']==digest(out/'protocol.json');restore_trainable(model,saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);history,best,start=saved['history'],saved['best'],saved['epoch']+1
    else:
        atomic_save(dict(model=trainable_state(model),epoch=-1,protocol_sha256=digest(out/'protocol.json')),checkpoint_root/'best.pt');np.savez(out/'best_validation_predictions.npz',indices=np.array(val),probability=warm_probability,protocol_sha256=digest(out/'protocol.json'),checkpoint_sha256=digest(checkpoint_root/'best.pt'))
    ema=history[-1]['training_loss_ema'] if history else None;training_best=min((r['training_loss_ema'] for r in history),default=float('inf'));training_stale=history[-1]['training_stale'] if history else 0
    for epoch in range(start,40):
        if epoch>=24 and training_stale>=12:break
        epoch_started=time.monotonic();factor=1. if epoch<16 else .1+.45*(1+math.cos(math.pi*(epoch-15)/24))
        for group,base_lr in zip(optimizer.param_groups,(.000002,.00002,.0001)):group['lr']=base_lr*factor
        model.train();order=torch.multinomial(sampling,512,replacement=True,generator=torch.Generator().manual_seed(220600+epoch));batch_losses=[]
        for batch in order.split(4):
            batch_ids=np.array(train)[batch.numpy()];positive,negative,clinical=forward(batch_ids);raw=torch.where(truth[batch_ids]==1,positive,negative);rank=hard_negative_rank_loss(torch.stack((positive,negative),1),truth[batch_ids],margin=1.,maximum_negatives=4);loss=F.binary_cross_entropy_with_logits(raw,truth[batch_ids])+0.*clinical+.2*rank;assert torch.isfinite(loss);batch_losses.append(float(loss.detach()));optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
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
            assert shutil.disk_usage(checkpoint_root).free>4*state_bytes+(32<<20),'Preserve RAM for next atomic rolling state';atomic_save(dict(model=state,epoch=epoch,protocol_sha256=digest(out/'protocol.json')),snapshot)
        snapshot_sha=digest(snapshot) if snapshot.exists() else None;np.savez(out/f'epoch_{epoch:03d}_predictions.npz',indices=np.array(val),probability=probability,protocol_sha256=digest(out/'protocol.json'),model_state_sha256=state_digest.hexdigest(),checkpoint_sha256=snapshot_sha or '')
        train_loss=float(np.mean(batch_losses));ema=train_loss if ema is None else .9*ema+.1*train_loss
        if ema<training_best-.001:training_best=ema;training_stale=0
        else:training_stale+=1
        history.append(dict(epoch=epoch,decision=score,new_false_healthy_by_source=errors,training_loss=train_loss,training_loss_ema=ema,training_stale=training_stale,model_state_sha256=state_digest.hexdigest(),checkpoint_sha256=snapshot_sha,elapsed_seconds=time.monotonic()-epoch_started));print(json.dumps(history[-1]),flush=True)
        atomic_save(dict(model=state,optimizer=optimizer.state_dict(),rng=capture_rng(),epoch=epoch,history=history,best=best,protocol_sha256=digest(out/'protocol.json')),checkpoint_root/'last.pt')
        if best[0]==0 and best[1]>=28:break
    chosen=torch.load(checkpoint_root/'best.pt',weights_only=True,map_location='cpu');result=dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),checkpoint_sha256=digest(checkpoint_root/'best.pt'),selected_epoch=chosen['epoch'],best_decision=best,history=history,cheap_pass=best[0]==0 and best[1]>=28,model_promoted=False,checkpoint_root=str(checkpoint_root),checkpoint_volatile=True,ram_snapshot_epochs=snapshot_epochs,exact_gpu_pixel_bank_bytes=required,new_detector_inputs=0,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)


if __name__=='__main__':main()
