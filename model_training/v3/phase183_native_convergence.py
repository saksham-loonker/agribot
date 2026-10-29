"""Guard-aware expected reward continuation of completed native supervised adapters."""
from pathlib import Path
from functools import lru_cache
import argparse,json,hashlib,shutil,time,math
import numpy as np
from PIL import Image
import torch,timm
from torch.nn import functional as F
from app_eval import canon,classifier_letterbox_input
from balanced_clinical_loss import balanced_clinical_loss
from conditional_recovery_policy import recover_only
from dino_large_encoder import SOURCE_URL,SOURCE_SHA256
from hard_negative_ranking import hard_negative_rank_loss
from joint_clinical_head import JointClinicalHead
from large_photometric_dino import trainable_state,restore_trainable
from native_lora_tail import NativeLoraTail
from opened_context_data import load,read
from partial_dino_healthy import healthy_margin
from permissive_encoder import digest
from phase85_population_sft import evaluation_data
from population_siglip_tail import freeze
from safe_feature_memmap import feature_memmap
from source_balance import source_class_group_weights
from training_support import atomic_save,capture_rng,restore_rng


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase183_native_convergence';out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    cache=p/'phase172_native_large_field_features';cp=read(cache/'protocol.json');cc=read(cache/'complete.json');assert cc['protocol_sha256']==digest(cache/'protocol.json') and cc['features_sha256']==digest(cache/'features.npy')
    parent=p/'phase173_native_large_field_head';pr=read(parent/'result.json');warm=parent/'best.pt';assert digest(warm)==pr['checkpoint_sha256']
    rows=read(p/'phase168_early_cnn_prefix/rows.json');assert digest(p/'phase168_early_cnn_prefix/rows.json')==cp['rows_sha256'];train=np.array(cp['train_indices']);val=np.array(cp['validation_indices']);selection=read(p/'phase118_conditional_recovery/protocol.json');assert val.tolist()==selection['validation_indices']
    earlier=read(p/'phase168_early_cnn_prefix/protocol.json');target=earlier['target_indices'];anchors=earlier['anchor_indices'];field_indices=earlier['field_indices'];assert train.tolist()==target+anchors+field_indices
    conv=p/'phase154_convnext_features';conv_proto=read(conv/'protocol.json');conv_complete=read(conv/'complete.json');assert digest(conv/'complete.json')==cp['convnext_complete_sha256'] and conv_complete['features_sha256']==digest(conv/'features.npy')
    textures={sha:value for sha,value in zip(conv_proto['input_sha256'],np.load(conv/'features.npy'))};field=p/'phase164_new_field_features';fc=read(field/'complete.json');assert digest(field/'complete.json')==cp['field_complete_sha256'] and fc['features_sha256']==digest(field/'features.npy') and fc['input_sha256_sha256']==digest(field/'input_sha256.json')
    textures.update({sha:value[2432:] for sha,value in zip(read(field/'input_sha256.json'),np.load(field/'features.npy'))})
    torch.set_num_threads(4);torch.manual_seed(2174);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;timm.layers.set_fused_attn(False)
    official=read(p/'phase23_dino/encoder_source.json');assert official['url']==SOURCE_URL and official['file_sha256']==SOURCE_SHA256 and digest(official['local_file'])==SOURCE_SHA256
    backbone=timm.create_model(official['model_name'],pretrained=True,img_size=518,dynamic_img_size=True,num_classes=0,pretrained_cfg_overlay=dict(file=official['local_file'],hf_hub_id=None,license='apache-2.0')).cuda().eval().requires_grad_(False)
    assert backbone.num_prefix_tokens==1 and backbone.num_features==1024 and isinstance(backbone.fc_norm,torch.nn.Identity)
    head=JointClinicalHead();head.load_state_dict(torch.load(warm,weights_only=True,map_location='cpu')['model'],strict=True);model=NativeLoraTail(backbone,head).cuda().eval();cut=len(backbone.blocks)-2
    prefix_digest=hashlib.sha256()
    for name,value in sorted(backbone.state_dict().items()):
        if name in ('cls_token','pos_embed') or name.startswith(('patch_embed.','norm_pre.')) or any(name.startswith(f'blocks.{i}.') for i in range(cut)):
            a=value.detach().cpu().contiguous().numpy();prefix_digest.update(str((name,str(a.dtype),a.shape)).encode());prefix_digest.update(a.tobytes())
    inputs=cp['inputs'];keys=[r['input_sha256'] for r in inputs];position={sha:i for i,sha in enumerate(keys)};assert len(position)==len(keys)
    continuation=p/'phase174_native_tail_adaptation';continuation_result=read(continuation/'result.json');parent_last=continuation/'last.pt';parent_state=torch.load(parent_last,weights_only=True,map_location='cpu');assert parent_state['protocol_sha256']==digest(continuation/'protocol.json')
    protocol=dict(source_sha256=digest(__file__),parent_continuation_protocol_sha256=digest(continuation/'protocol.json'),parent_last_sha256=digest(parent_last),safe_incumbent_sha256=digest(continuation/'best.pt'),reward_weight=0.,native_complete_sha256=digest(cache/'complete.json'),native_protocol_sha256=digest(cache/'protocol.json'),warm_head_sha256=digest(warm),
        tail_source_sha256=digest(Path(__file__).with_name('native_lora_tail.py')),lora_source_sha256=digest(Path(__file__).with_name('lora_dino.py')),policy_source_sha256=digest(Path(__file__).with_name('conditional_recovery_policy.py')),
        prefix_state_sha256=prefix_digest.hexdigest(),frozen_prefix_blocks=cut,adapted_blocks=2,rank=8,alpha=16.,adapter_dropout=.05,input_size=518,
        train_indices=train.tolist(),validation_indices=val.tolist(),training_mass=dict(target=.5,old_anchor=.25,field93=.25,old107=0.),old107_present_in_warm_history=True,
        maximum_epochs=40,minimum_epochs=24,training_ema_patience=12,training_ema_minimum_improvement=.001,training_ema_decay=.9,validation_staleness_stops_training=False,save_every_epoch=True,encoder_lr_schedule='Resume 1e-4 through absolute epoch 23, cosine decay to 1e-5 at epoch 47',draws_per_epoch=512,batch_photos=4,native_checkpoint_chunk=2,learning_rate=.0001,weight_decay=.01,seed=2174,clinical_weight=1.,binary_weight=.3,ranking_weight=.2,
        frozen_prefix_storage='GPU FP32 bank, reused every epoch. No room for a persistent 12GB copy; process restart regenerates this missing intermediate representation, never discards existing caches or optimizer state.',
        local_grid_storage='Persist a separate frozen-original 6x6 local-token grid while building the prefix, avoiding another encoder pass if a later local head is justified. It is not adapted-token evidence.',
        local_grid_shape=[len(inputs),36,1024],checkpointing='Non-reentrant PyTorch activation checkpointing with RNG preservation; frozen inputs still yield adapter gradients',
        hypothesis='Previous native adapter fitting stopped after only eight epochs because an unqualified but safe low-recall warm incumbent dominated validation selection. Test premature stopping directly: restore the completed supervised epoch-seven model/optimizer/RNG, retain the same objective/data/guards, continue up to forty epochs with training-loss convergence rather than validation-safety staleness, and preserve every epoch for later exact verification. Release gates remain unchanged.',
        existing_healthy_and_named_disease_preserved=True,healthy_admission=.95,ad_independent=False,olid_independent=False,acceptance_data_used=False,
        stopping='At least twenty-four additional epochs, then stop at twelve stale EMA training-loss epochs with absolute improvement .001; forty additional epochs maximum. Stop earlier only if the existing cheap recognition/safety gates pass. Preserve every checkpoint and probability vector. No failed export or acceptance feedback.')
    freeze(out/'protocol.json',protocol)
    required=len(inputs)*1370*1024*4;free,_=torch.cuda.mem_get_info();assert free>required+(10<<30),'Preserve GPU activation headroom'
    # Allocate outside inference_mode: normal frozen inputs must be usable by
    # non-reentrant autograd checkpointing during adapter training.
    prefix_bank=torch.empty((len(inputs),1370,1024),dtype=torch.float32,device='cuda')
    grid=continuation/'frozen_local_grid';grid_done=True
    assert read(grid/'complete.json')['features_sha256']==digest(grid/'features.npy') and read(grid/'complete.json')['protocol_sha256']==digest(grid/'protocol.json')
    local=None if grid_done else feature_memmap(grid/'features.npy',[len(inputs),36,1024]);reference=np.load(cache/'features.npy',mmap_mode='r');maximum=0.
    @lru_cache(maxsize=4)
    def rgb(path,sha):
        assert digest(path)==sha
        with Image.open(path) as image:return np.asarray(image.convert('RGB'))
    with torch.inference_mode():
        for start in range(0,len(inputs),2):
            batch=[]
            for row in inputs[start:start+2]:
                value=(canon.classifier_input if row['view']=='center' else classifier_letterbox_input)(rgb(row['path'],row['file_sha256']),518,row['box']);assert hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()==row['input_sha256'];batch.append(value)
            x=torch.from_numpy(np.concatenate(batch)).cuda().permute(0,3,1,2).contiguous(memory_format=torch.channels_last);tokens=backbone.norm_pre(backbone.patch_drop(backbone._pos_embed(backbone.patch_embed(x))));tokens=backbone.blocks[:cut](tokens);assert tokens.shape[1:]==(1370,1024);prefix_bank[start:start+len(batch)]=tokens
            if not grid_done or start<16:
                raw=backbone.norm(backbone.blocks[cut:](tokens));cls=F.normalize(raw[:,0],dim=1)*(1024**.5);error=float(np.abs(cls.cpu().numpy()-reference[start:start+len(batch)]).max());maximum=max(maximum,error);assert error<.002,('Native prefix differs from frozen cached CLS',error)
                if not grid_done:
                    patches=F.normalize(raw[:,1:],dim=-1)*(1024**.5);pooled=F.adaptive_avg_pool2d(patches.transpose(1,2).reshape(len(batch),1024,37,37),(6,6)).flatten(2).transpose(1,2);local[start:start+len(batch)]=pooled.cpu().numpy()
            if start%64==0 or start+len(batch)==len(inputs):
                if local is not None:local.flush()
                print(json.dumps(dict(stage='native_frozen_prefix',completed=start+len(batch),total=len(inputs),maximum_cls_error=maximum)),flush=True)
    if not grid_done:
        local.flush();(grid/'complete.json').write_text(json.dumps(dict(protocol_sha256=digest(grid/'protocol.json'),features_sha256=digest(grid/'features.npy'),maximum_cls_error=maximum,encoder_adapted=False),indent=2));del local
    del backbone,head;rgb.cache_clear();torch.cuda.empty_cache()
    link_native=torch.tensor([position[r['input_sha256']] for r in cp['links']],device='cuda');texture=torch.from_numpy(np.stack([textures[r['parent_224_input_sha256']] for r in cp['links']])).cuda()
    index=np.zeros((len(rows),2,6),np.int64);mask=np.zeros(index.shape,np.float32)
    for i in train.tolist()+val.tolist():
        for view,ids in enumerate(cp['bags'][rows[i]['path']]):index[i,view,:len(ids)]=ids;mask[i,view,:len(ids)]=1.
    index=torch.from_numpy(index).cuda();mask=torch.from_numpy(mask).cuda();valid=mask.sum(-1)>0;truth=torch.tensor([r['y'] for r in rows],dtype=torch.float32,device='cuda')
    data=load(p);labels=data['manifest']['classifier']['labels'];fine=torch.tensor([labels.index(r['label']) if r['label'] in labels else -1 for r in rows],device='cuda');entries,records,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({b['path']:(b,packets[b['id']],r) for b,r in zip(before,previous)})
    sampling=torch.from_numpy(np.concatenate([.5*source_class_group_weights(rows,target),.25*source_class_group_weights(rows,anchors),.25*source_class_group_weights(rows,field_indices)]))
    def forward(ids):
        actual=mask[ids]>0;links=index[ids][actual];needed,inverse=link_native[links].unique(return_inverse=True);parts=[]
        for chunk in needed.split(2 if model.training else 8):parts.append(model.features(prefix_bank[chunk],checkpointed=model.training))
        features=torch.cat(parts)[inverse];logits=model.head(torch.cat((features,texture[links]),1));raw=healthy_margin(logits);z=prefix_bank.new_zeros((len(ids),2,6));z[actual]=raw;means=(z*mask[ids]).sum(-1)/mask[ids].sum(-1).clamp_min(1)
        owner=torch.arange(len(ids)*2,device='cuda').reshape(len(ids),2,1).expand_as(index[ids])[actual];totals=logits.new_zeros((len(ids)*2,9)).index_add(0,owner,logits);counts=mask[ids].sum(-1);clinical=(totals/counts.reshape(-1).clamp_min(1)[:,None]).reshape(len(ids),2,9)
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
        warm_parts=[forward(part) for part in np.array_split(val,max(1,(len(val)+7)//8))];warm_positive=torch.cat([r[0] for r in warm_parts]);warm_negative=torch.cat([r[1] for r in warm_parts]);warm_loss=float(F.binary_cross_entropy_with_logits(torch.where(truth[val]==1,warm_positive,warm_negative),truth[val]));warm_probability=warm_positive.sigmoid().cpu().numpy()
    cached=np.load(parent/'best_validation_predictions.npz');assert np.array_equal(cached['indices'],val) and str(cached['checkpoint_sha256'])==digest(warm);warm_decision,_,warm_kinds=decision(warm_probability,warm_loss);_,_,cached_kinds=decision(cached['probability'],warm_loss);assert warm_kinds==cached_kinds,'Zero-adapter warm app verdicts changed'
    probability_error=float(np.abs(cached['probability']-warm_probability).max());assert probability_error<.0005
    freeze(out/'warm_parity.json',dict(protocol_sha256=digest(out/'protocol.json'),maximum_native_cls_error=maximum,maximum_probability_error=probability_error,all_71_app_verdicts_equal=True,warm_decision=warm_decision))
    optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=.0001,weight_decay=.01);history=[];best=warm_decision;stale=0;start=0;torch.manual_seed(2174)
    if (out/'last.pt').exists():
        saved=torch.load(out/'last.pt',weights_only=True,map_location='cpu');assert saved['protocol_sha256']==digest(out/'protocol.json');restore_trainable(model,saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);history,best,stale,start=saved['history'],saved['best'],saved['stale'],saved['epoch']+1
    else:
        atomic_save(dict(model=trainable_state(model),epoch=-1,protocol_sha256=digest(out/'protocol.json')),out/'best.pt');np.savez(out/'best_validation_predictions.npz',indices=val,probability=warm_probability,protocol_sha256=digest(out/'protocol.json'),checkpoint_sha256=digest(out/'best.pt'))
        restore_trainable(model,parent_state['model']);optimizer.load_state_dict(parent_state['optimizer']);restore_rng(parent_state['rng'])
        model.eval()
        with torch.inference_mode():
            parts=[forward(part) for part in np.array_split(val,max(1,(len(val)+7)//8))];parent_positive=torch.cat([r[0] for r in parts]);parent_negative=torch.cat([r[1] for r in parts]);parent_loss=float(F.binary_cross_entropy_with_logits(torch.where(truth[val]==1,parent_positive,parent_negative),truth[val]));parent_probability=parent_positive.sigmoid().cpu().numpy();parent_decision,parent_errors,_=decision(parent_probability,parent_loss)
        assert parent_decision[:3]==tuple(parent_state['history'][-1]['decision'][:3]),'Continuation does not restore parent actual-verdict counts'
        np.savez(out/'parent_last_validation_predictions.npz',indices=val,probability=parent_probability,checkpoint_sha256=digest(parent_last),protocol_sha256=digest(out/'protocol.json'))
        freeze(out/'continuation_restore.json',dict(parent_epoch=parent_state['epoch'],decision=parent_decision,new_false_healthy_by_source=parent_errors,optimizer_and_rng_restored=True))
        print(json.dumps(dict(stage='continue_native_parent',parent_epoch=parent_state['epoch'],decision=parent_decision)),flush=True)
    if start==0:assert shutil.disk_usage(out).free>(1<<30)+(70<<20),'Preserve disk reserve for all forty small adapter checkpoints'
    ema=history[-1]['training_loss_ema'] if history else None;training_best=min((r['training_loss_ema'] for r in history),default=float('inf'));training_stale=history[-1]['training_stale'] if history else 0
    for epoch in range(start,40):
        if epoch>=24 and training_stale>=12:break
        absolute_epoch=parent_state['epoch']+1+epoch
        lr=.0001 if absolute_epoch<24 else .00001+.5*(.0001-.00001)*(1+math.cos(math.pi*(absolute_epoch-23)/24))
        for group in optimizer.param_groups:group['lr']=lr
        batch_losses=[]
        epoch_started=time.monotonic();model.train();order=torch.multinomial(sampling,512,replacement=True,generator=torch.Generator().manual_seed(217400+absolute_epoch))
        for batch in order.split(4):
            ids=train[batch.numpy()];positive,negative,clinical=forward(ids);raw=torch.where(truth[ids]==1,positive,negative);rank=hard_negative_rank_loss(torch.stack((positive,negative),1),truth[ids],margin=1.,maximum_negatives=4);loss=clinical+.3*F.binary_cross_entropy_with_logits(raw,truth[ids])+.2*rank
            assert torch.isfinite(loss);batch_losses.append(float(loss.detach()));optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        model.eval()
        with torch.inference_mode():
            parts=[forward(part) for part in np.array_split(val,max(1,(len(val)+7)//8))];positive=torch.cat([r[0] for r in parts]);negative=torch.cat([r[1] for r in parts]);loss=float(F.binary_cross_entropy_with_logits(torch.where(truth[val]==1,positive,negative),truth[val]));probability=positive.sigmoid().cpu().numpy();score,errors,_=decision(probability,loss)
        state=trainable_state(model)
        if score>tuple(best):
            best=score;stale=0;atomic_save(dict(model=state,epoch=epoch,protocol_sha256=digest(out/'protocol.json')),out/'best.pt');np.savez(out/'best_validation_predictions.npz',indices=val,probability=probability,protocol_sha256=digest(out/'protocol.json'),checkpoint_sha256=digest(out/'best.pt'))
        else:stale+=1
        train_loss=float(np.mean(batch_losses));ema=train_loss if ema is None else .9*ema+.1*train_loss
        if ema<training_best-.001:training_best=ema;training_stale=0
        else:training_stale+=1
        checkpoint=out/f'epoch_{absolute_epoch:03d}.pt';atomic_save(dict(model=state,epoch=absolute_epoch,protocol_sha256=digest(out/'protocol.json')),checkpoint)
        np.savez(out/f'epoch_{absolute_epoch:03d}_predictions.npz',indices=val,probability=probability,protocol_sha256=digest(out/'protocol.json'),checkpoint_sha256=digest(checkpoint))
        history.append(dict(epoch=epoch,absolute_epoch=absolute_epoch,decision=score,new_false_healthy_by_source=errors,training_loss=train_loss,training_loss_ema=ema,training_stale=training_stale,learning_rate=lr,checkpoint_sha256=digest(checkpoint),elapsed_seconds=time.monotonic()-epoch_started));print(json.dumps(history[-1]),flush=True)
        atomic_save(dict(model=state,optimizer=optimizer.state_dict(),rng=capture_rng(),epoch=epoch,history=history,best=best,stale=stale,protocol_sha256=digest(out/'protocol.json')),out/'last.pt')
        if best[0]==0 and best[1]>=28:break
    selected=torch.load(out/'best.pt',weights_only=True,map_location='cpu');result=dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),checkpoint_sha256=digest(out/'best.pt'),selected_epoch=selected['epoch'],best_decision=best,history=history,cheap_pass=best[0]==0 and best[1]>=28,passed=False,model_promoted=False,
        frozen_prefix_cache_bytes=required,frozen_prefix_persisted=False,frozen_local_grid_complete_sha256=digest(grid/'complete.json'),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    if result['cheap_pass']:result['next_action']='Freeze candidate and extend only missing frozen-prefix identities for all applicable consumed cases; run every unchanged full recognition/safety gate before restored full-model verification/export; mobile student and prospective acceptance remain necessary'
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)


if __name__=='__main__':main()
