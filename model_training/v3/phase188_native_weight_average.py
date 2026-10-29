"""Guard-aware expected reward continuation of completed native supervised adapters."""
from pathlib import Path
from functools import lru_cache
import argparse,json,hashlib,shutil,time
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
    p=Path(parser.parse_args().campaign);out=p/'phase188_native_weight_average';out.mkdir(exist_ok=True);started=time.monotonic()
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
    trajectory=p/'phase183_native_convergence';assert read(trajectory/'result.json')['history'][-1]['absolute_epoch']==47
    average_inputs={f'epoch_{epoch:03d}.pt':digest(trajectory/f'epoch_{epoch:03d}.pt') for epoch in range(38,48)}
    protocol=dict(source_sha256=digest(__file__),parent_continuation_protocol_sha256=digest(continuation/'protocol.json'),parent_last_sha256=digest(parent_last),safe_incumbent_sha256=digest(continuation/'best.pt'),weight_average_inputs=average_inputs,weight_average_rule='Arithmetic mean of all trainable parameters from fixed final ten supervised epochs 38..47, FP64 accumulation then FP32, no validation selection',native_complete_sha256=digest(cache/'complete.json'),native_protocol_sha256=digest(cache/'protocol.json'),warm_head_sha256=digest(warm),
        tail_source_sha256=digest(Path(__file__).with_name('native_lora_tail.py')),lora_source_sha256=digest(Path(__file__).with_name('lora_dino.py')),policy_source_sha256=digest(Path(__file__).with_name('conditional_recovery_policy.py')),
        prefix_state_sha256=prefix_digest.hexdigest(),frozen_prefix_blocks=cut,adapted_blocks=2,rank=8,alpha=16.,adapter_dropout=.05,input_size=518,
        train_indices=train.tolist(),validation_indices=val.tolist(),training_mass=dict(target=.5,old_anchor=.25,field93=.25,old107=0.),old107_present_in_warm_history=True,
        maximum_epochs=16,minimum_epochs=8,patience=6,draws_per_epoch=512,batch_photos=4,native_checkpoint_chunk=2,learning_rate=.0001,weight_decay=.01,seed=2174,clinical_weight=1.,binary_weight=.3,ranking_weight=.2,
        frozen_prefix_storage='GPU FP32 bank, reused every epoch. No room for a persistent 12GB copy; process restart regenerates this missing intermediate representation, never discards existing caches or optimizer state.',
        local_grid_storage='Persist a separate frozen-original 6x6 local-token grid while building the prefix, avoiding another encoder pass if a later local head is justified. It is not adapted-token evidence.',
        local_grid_shape=[len(inputs),36,1024],checkpointing='Non-reentrant PyTorch activation checkpointing with RNG preservation; frozen inputs still yield adapter gradients',
        hypothesis='The longer supervised trajectory fit training labels but still made unsafe validation admissions. A fixed final-ten-epoch parameter average tests optimization variance without fitting a committee or threshold on validation. Restore only the averaged trainable suffix against its unchanged official prefix, and reuse all matching inputs.',
        existing_healthy_and_named_disease_preserved=True,healthy_admission=.95,ad_independent=False,olid_independent=False,acceptance_data_used=False,
        stopping='One fixed weight-average candidate only, no optimizer or validation-derived weights. Require >=28/40 Healthy and zero new false Healthy before any complete restored-model verification or export.')
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
    average=None
    for name,sha in average_inputs.items():
        checkpoint=trajectory/name;assert digest(checkpoint)==sha
        saved=torch.load(checkpoint,weights_only=True,map_location='cpu');assert saved['protocol_sha256']==digest(trajectory/'protocol.json')
        state=saved['model']
        if average is None:average={k:v.double()/len(average_inputs) for k,v in state.items()}
        else:
            assert set(average)==set(state)
            for key in average:average[key]+=state[key].double()/len(average_inputs)
    average={k:v.float() for k,v in average.items()};restore_trainable(model,average);model.eval()
    with torch.inference_mode():
        parts=[forward(part) for part in np.array_split(val,max(1,(len(val)+7)//8))];positive=torch.cat([r[0] for r in parts]);negative=torch.cat([r[1] for r in parts]);loss=float(F.binary_cross_entropy_with_logits(torch.where(truth[val]==1,positive,negative),truth[val]));probability=positive.sigmoid().cpu().numpy();score,errors,_=decision(probability,loss)
    atomic_save(dict(model=average,protocol_sha256=digest(out/'protocol.json'),average_inputs=average_inputs),out/'best.pt')
    np.savez(out/'best_validation_predictions.npz',indices=val,probability=probability,protocol_sha256=digest(out/'protocol.json'),checkpoint_sha256=digest(out/'best.pt'))
    result=dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),checkpoint_sha256=digest(out/'best.pt'),best_decision=score,new_false_healthy_by_source=errors,cheap_pass=score[0]==0 and score[1]>=28,passed=False,model_promoted=False,weight_average_inputs=average_inputs,elapsed_seconds=time.monotonic()-started,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
