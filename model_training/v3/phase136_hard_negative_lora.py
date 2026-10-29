"""Class-supervised LoRA with explicit recognized-disease anchor replay."""
from pathlib import Path
import argparse,copy,json,time
import numpy as np
import torch,timm
from torch.nn import functional as F
from conditional_recovery_policy import recover_only
from continued_diagnostic_expert import ContinuedDiagnosticExpert
from lora_dino import inject_qv_lora
from opened_context_data import load,read
from opened_guard_screen import OpenedGuardScreen
from partial_dino_cache import original_dino
from partial_dino_healthy import healthy_margin
from permissive_encoder import digest
from phase2_select import retain
from phase5_scene_ablation import photo_metrics
from phase22_accept import paired_interval
from phase83_eligible_operating_points import eligible,safety
from phase85_population_sft import evaluation_data
from phase114_new_supervision_features import original_features
from population_siglip_tail import freeze
from source_balance import source_class_group_weights
from clinical_anchor_head import image_condition_loss
from hard_negative_ranking import hard_negative_rank_loss
from training_support import atomic_save,capture_rng,restore_rng


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase136_hard_negative_lora';out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    torch.set_num_threads(4);torch.manual_seed(2124);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;timm.layers.set_fused_attn(False)
    values,bags,descriptors,parent=original_features(p);rows=read(p/'phase84_population_head/fitting_rows.json');selection=read(p/'phase118_conditional_recovery/protocol.json')
    anchor_selection=read(p/'phase133_anchor_input_cache/protocol.json');train=np.array(anchor_selection['train_indices']);val=np.array(anchor_selection['validation_indices']);active=sorted(set(train.tolist()+val.tolist()))
    cache=p/'phase133_anchor_input_cache';cache_protocol=read(cache/'protocol.json');complete=read(cache/'complete.json')
    assert complete['protocol_sha256']==digest(cache/'protocol.json') and complete['tokens_sha256']==digest(cache/'tokens.npy')
    lookup={sha:i for i,sha in enumerate(cache_protocol['input_sha256'])}
    tables={i:[[lookup[descriptors[key]['input_sha256']] for key in bags[rows[i]['path']][view]] for view in ('center','full')] for i in active}
    train_shas={descriptors[key]['input_sha256'] for i in train for view in bags[rows[i]['path']].values() for key in view};val_shas={descriptors[key]['input_sha256'] for i in val for view in bags[rows[i]['path']].values() for key in view}
    assert not (train_shas&val_shas) and not ({rows[i]['group'] for i in train}&{rows[i]['group'] for i in val})
    data=load(p);entries,records,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({b['path']:(b,packets[b['id']],r) for b,r in zip(before,previous)})
    initial=torch.load(p/'phase55_dino_continuation/head/candidate.pt',weights_only=True,map_location='cpu');warm=p/'phase97_continued_verified_expert/best.pt';wr=read(p/'phase97_continued_verified_expert/result.json');assert wr['checkpoint_sha256']==digest(warm)
    original=original_dino(p);expert=ContinuedDiagnosticExpert(original,initial);expert.load_state_dict(torch.load(warm,weights_only=True,map_location='cpu')['model'])
    backbone=original_dino(p)
    for index,module in enumerate(expert.blocks,start=len(backbone.blocks)-2):backbone.blocks[index]=module
    backbone.norm=expert.norm;backbone.fc_norm=expert.head[0];backbone.head=expert.head[1]
    teacher=copy.deepcopy(backbone).cuda().eval();backbone=inject_qv_lora(backbone).cuda().eval();tokens=torch.from_numpy(np.array(np.load(cache/'tokens.npy',mmap_mode='r'),copy=True)).cuda()
    def margins(model,x):return healthy_margin(model.forward_head(model.norm(model.blocks(x))))*initial['calibration'][0]+initial['calibration'][1]
    with torch.inference_mode():error=float((margins(backbone,tokens[:16])-margins(teacher,tokens[:16])).abs().max())
    assert error<.0001,('Zero LoRA injection changes restored expert',error)
    del teacher,expert,original;torch.cuda.empty_cache()
    protocol=dict(source_sha256=digest(__file__),lora_source_sha256=digest(Path(__file__).with_name('lora_dino.py')),parents=parent,input_cache_complete_sha256=digest(cache/'complete.json'),
                  selected_population_protocol_sha256=digest(p/'phase118_conditional_recovery/protocol.json'),anchor_selection_protocol_sha256=digest(p/'phase133_anchor_input_cache/protocol.json'),condition_loss_source_sha256=digest(Path(__file__).with_name('clinical_anchor_head.py')),rank_loss_source_sha256=digest(Path(__file__).with_name('hard_negative_ranking.py')),ranking_loss_weight=.2,ranking_margin=1.,maximum_hard_negatives_per_batch=4,warm_checkpoint_sha256=digest(warm),
                  train_images=len(train),validation_images=len(val),rank=8,alpha=16.,dropout=.05,lora_blocks=12,lora_components=['query','value'],
                  trainable_parameters=sum(parameter.numel() for parameter in backbone.parameters() if parameter.requires_grad),frozen_base_parameters=sum(parameter.numel() for parameter in backbone.parameters() if not parameter.requires_grad),
                  learning_rate=.0001,weight_decay=.01,batch_images=8,maximum_epochs=80,minimum_epochs=25,patience=15,seed=2124,target_replay_weight=.5,anchor_replay_weight=.5,condition_loss_weight=.3,draws_per_epoch=1024,
                  zero_adapter_maximum_margin_error=error,calibration=initial['calibration'],admission_probability=.95,existing_healthy_and_named_disease_preserved=True,
                  hypothesis='Anchor-supervised LoRA raised Healthy recall while admitting diseases. Add a training-only pairwise logit separation loss against the four hardest non-Healthy examples per batch; keep the same input, warm model, sampler, BCE/clinical loss, seed, admission and safety rules.',
                  ad_independent=False,olid_independent=False,acceptance_data_used=False,new_publisher_source_used=False,
                  stopping='One hard-negative ranking clinical-anchor rank-eight Q/V control, minimum twenty-five epochs, fifteen stale actual-verdict epochs, eighty maximum. Extend cached input coverage and run full gates only after >=28/40 Healthy and zero new source false-Healthy. No export unless complete old and consumed recognition gates pass.')
    freeze(out/'protocol.json',protocol)
    truth=torch.tensor([r['y'] for r in rows],dtype=torch.float32,device='cuda');target=anchor_selection['target_indices'];anchors=anchor_selection['anchor_indices'];assert train.tolist()==target+anchors
    sampling=torch.from_numpy(np.concatenate([.5*source_class_group_weights(rows,target),.5*source_class_group_weights([dict(r,condition_class=r['label']) for r in rows],anchors,target='condition_class')]))
    labels=data['manifest']['classifier']['labels'];healthy_index=labels.index('Healthy');fine=torch.tensor([labels.index(r['label']) if r['label'] in labels else -1 for r in rows],device='cuda')
    vw=torch.tensor(source_class_group_weights(rows,val),device='cuda')
    def forward(ids):
        flat=[];groups=[]
        for offset,i in enumerate(ids):
            for view,keys in enumerate(tables[i]):flat.extend(keys);groups.extend([2*offset+view]*len(keys))
        unique,inverse=torch.tensor(flat,device='cuda').unique(return_inverse=True)
        logits=backbone.forward_head(backbone.norm(backbone.blocks(tokens[unique])))[inverse]
        z=healthy_margin(logits)*initial['calibration'][0]+initial['calibration'][1];groups=torch.tensor(groups,device='cuda');counts=torch.bincount(groups,minlength=2*len(ids))
        sums=z.new_zeros(2*len(ids)).index_add(0,groups,z)
        clinical=logits.new_zeros((2*len(ids),len(labels))).index_add(0,groups,logits)/counts[:,None]
        return (sums/counts).reshape(len(ids),2),clinical.reshape(len(ids),2,len(labels))
    def decision(probability,loss):
        errors={};healthy=selection['fixed_healthy_validation_correct'];old_h=0
        for offset,i in enumerate(val):
            row=rows[i];original,packet,retained=items[row['path']];after=recover_only(original,retained,packet,float(probability[offset]),data['manifest']);now=after['app_replay']['kind']=='HEALTHY'
            if not row['y']:errors[row['source']]=errors.get(row['source'],0)+int(now)
            elif row['source']=='olid_tomato':healthy+=int(now)
            else:old_h+=int(now)
        return (-sum(errors.values()),healthy,old_h,-loss),errors
    optimizer=torch.optim.AdamW([parameter for parameter in backbone.parameters() if parameter.requires_grad],lr=.0001,weight_decay=.01);history=[];best=None;stale=0;start=0
    if (out/'last.pt').exists():
        saved=torch.load(out/'last.pt',weights_only=True,map_location='cpu');assert saved['protocol_sha256']==digest(out/'protocol.json');backbone.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);history,best,stale,start=saved['history'],saved['best'],saved['stale'],saved['epoch']+1
    for epoch in range(start,80):
        if epoch>=25 and stale>=15:break
        backbone.train();order=torch.multinomial(sampling,1024,replacement=True,generator=torch.Generator().manual_seed(212400+epoch))
        for batch in order.split(8):
            ids=train[batch.numpy()];z,clinical=forward(ids);raw=torch.where(truth[ids]==1,z.min(1).values,z.max(1).values);loss=F.binary_cross_entropy_with_logits(raw,truth[ids])+.3*image_condition_loss(clinical,fine[ids],healthy_index)+.2*hard_negative_rank_loss(z,truth[ids]);assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(backbone.parameters(),1.);optimizer.step()
        backbone.eval()
        with torch.inference_mode():
            z=torch.cat([forward(part)[0] for part in np.array_split(val,max(1,(len(val)+7)//8))]);raw=torch.where(truth[val]==1,z.min(1).values,z.max(1).values);loss=float((F.binary_cross_entropy_with_logits(raw,truth[val],reduction='none')*vw).sum());score,errors=decision(z.min(1).values.sigmoid().cpu().numpy(),loss)
        state={key:value.detach().cpu().clone() for key,value in backbone.state_dict().items()}
        if best is None or score>tuple(best):
            best=score;stale=0;atomic_save(dict(model=state,epoch=epoch,decision=score,protocol_sha256=digest(out/'protocol.json')),out/'best.pt')
            np.savez(out/'best_validation_predictions.npz',indices=val,paired_margin=z.cpu().numpy(),protocol_sha256=digest(out/'protocol.json'),checkpoint_sha256=digest(out/'best.pt'))
        else:stale+=1
        history.append(dict(epoch=epoch,decision=score,new_false_healthy_by_source=errors));print(json.dumps(history[-1]),flush=True)
        atomic_save(dict(model=state,epoch=epoch,optimizer=optimizer.state_dict(),rng=capture_rng(),history=history,best=best,stale=stale,protocol_sha256=digest(out/'protocol.json')),out/'last.pt')
    chosen=torch.load(out/'best.pt',weights_only=True,map_location='cpu');cheap=best[0]==0 and best[1]>=28
    result=dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),checkpoint_sha256=digest(out/'best.pt'),best_decision=best,selected_epoch=chosen['epoch'],history=history,
                cheap_pass=cheap,passed=False,full_screen_run=False,model_promoted=False,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    if cheap:
        result['next_action']='Extend exact-identity adapted-model prediction coverage to all applicable consumed-development cases, then run unchanged full recognition/safety gates before restored verification or export'
    result['elapsed_seconds']=time.monotonic()-started;(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('history','old_screen','consumed_all','consumed_validation')}),flush=True)


if __name__=='__main__':main()
