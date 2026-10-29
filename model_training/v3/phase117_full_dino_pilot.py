"""Bound one full-DINO-prefix adaptation after its suffix plateaued at 21/40."""
from pathlib import Path
import argparse,json,time
import numpy as np
import timm,torch
from torch.nn import functional as F
from continued_diagnostic_expert import ContinuedDiagnosticExpert
from opened_context_data import load,read
from partial_dino_cache import CanonicalInputs,original_dino
from partial_dino_healthy import prefix_cut,prefix_features,healthy_margin
from permissive_encoder import digest
from phase85_population_sft import evaluation_data
from phase88_diagnostic_bag_head import route_paired
from phase114_new_supervision_features import original_features
from population_siglip_tail import freeze
from routed_validation_cohort import healthy_cohort
from source_balance import source_class_group_weights
from training_support import atomic_save,capture_rng,restore_rng


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase117_full_dino_pilot';out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    torch.set_num_threads(4);torch.manual_seed(2117);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;timm.layers.set_fused_attn(False)
    values,bags,descriptors,parent=original_features(p);rows=read(p/'phase84_population_head/fitting_rows.json')
    active=[i for i,r in enumerate(rows) if r['active'] and r['path'] in bags and bags[r['path']]['center']]
    val=np.array([i for i in active if rows[i]['validation']]);val_shas={descriptors[key]['input_sha256'] for i in val for view in bags[rows[i]['path']].values() for key in view}
    excluded={rows[i]['group'] for i in active if not rows[i]['validation'] and any(descriptors[key]['input_sha256'] in val_shas for view in bags[rows[i]['path']].values() for key in view)}
    train=np.array([i for i in active if not rows[i]['validation'] and rows[i]['group'] not in excluded])
    assert not ({rows[i]['group'] for i in train}&{rows[i]['group'] for i in val})
    assert not ({descriptors[key]['input_sha256'] for i in train for view in bags[rows[i]['path']].values() for key in view}&val_shas)
    selected=sorted({key for i in active for view in bags[rows[i]['path']].values() for key in view});lookup={key:i for i,key in enumerate(selected)}
    tables={i:[[lookup[key] for key in bags[rows[i]['path']][view]] for view in ('center','full')] for i in active}
    data=load(p);entries,records,packets,before,previous=evaluation_data(p,data)
    denominator,fixed=healthy_cohort(rows,{rows[i]['path'] for i in val},{r['path']:r for r in previous},'olid_tomato');assert denominator==40
    initial=torch.load(p/'phase55_dino_continuation/head/candidate.pt',weights_only=True,map_location='cpu')
    warm_path=p/'phase97_continued_verified_expert/best.pt';warm_result=read(p/'phase97_continued_verified_expert/result.json');assert warm_result['checkpoint_sha256']==digest(warm_path)
    original=original_dino(p);expert=ContinuedDiagnosticExpert(original,initial);expert.load_state_dict(torch.load(warm_path,weights_only=True,map_location='cpu')['model'])
    backbone=original_dino(p);cut=prefix_cut(backbone)
    for index,module in enumerate(expert.blocks,start=cut):backbone.blocks[index]=module
    backbone.norm=expert.norm;backbone.fc_norm=expert.head[0];backbone.head=expert.head[1];backbone=backbone.cuda().eval().requires_grad_(True);expert=expert.cuda().eval();original=original.cuda().eval()
    pixels=CanonicalInputs(p);rgb=[]
    for key in selected:
        row=descriptors[key];rgb.append(pixels(row if 'kind' in row else dict(row,kind='new',policy=row['view'])))
    pixels.close();inputs=torch.from_numpy(np.concatenate(rgb)).cuda().permute(0,3,1,2);del rgb
    with torch.inference_mode():
        expected=expert(prefix_features(original,inputs[:16]),None);actual=healthy_margin(backbone(inputs[:16]));parity=float((actual-expected).abs().max())
    assert parity<.0001,('Warm full-DINO restoration mismatch',parity)
    del original,expert;torch.cuda.empty_cache();calibration=initial['calibration']
    protocol=dict(source_sha256=digest(__file__),parents=parent,warm_checkpoint_sha256=digest(warm_path),fitting_rows_sha256=digest(p/'phase84_population_head/fitting_rows.json'),
                  selected_input_sha256=[descriptors[key]['input_sha256'] for key in selected],train_images=len(train),validation_images=len(val),healthy_validation_denominator=denominator,inactive_fixed_healthy=fixed,
                  hypothesis='The corrected continued DINO suffix reached 21/40, above the full CNN. Permit one low-rate full-prefix adaptation while retaining its selected suffix and original auxiliary margin calibration.',
                  prefix_learning_rate=1e-5,suffix_learning_rate=1e-5,head_learning_rate=1e-4,maximum_epochs=12,minimum_epochs=4,patience=4,batch_images=8,negative_loss_weight=1.,seed=2117,
                  warm_suffix_restore_max_error=parity,calibration=calibration,healthy_min=.95,negative_healthy_max=.001,
                  pixel_cache_reused=True,feature_cache_reuse_invalid_for_unfrozen_prefix=True,encoder_weights_changed_in_copied_expert_only=True,
                  original_deployed_model_changed=False,new_publisher_source_used=False,ad_independent=False,olid_independent=False,acceptance_data_used=False,
                  stopping='One run, at most twelve epochs and four stale actual-verdict epochs. Require complete-cohort validation Healthy >=28/40 and no source false-Healthy increase before full-cohort inference. No export or acceptance opening from this pilot.')
    freeze(out/'protocol.json',protocol)
    truth=torch.tensor([r['y'] for r in rows],dtype=torch.float32,device='cuda');sampling=torch.from_numpy(source_class_group_weights(rows,train));vw=torch.tensor(source_class_group_weights(rows,val),device='cuda')
    def forward(ids):
        flat=[];groups=[]
        for offset,i in enumerate(ids):
            for view,keys in enumerate(tables[i]):flat.extend(keys);groups.extend([2*offset+view]*len(keys))
        unique,inverse=torch.tensor(flat,device='cuda').unique(return_inverse=True)
        z=(healthy_margin(backbone(inputs[unique]))*calibration[0]+calibration[1])[inverse];groups=torch.tensor(groups,device='cuda')
        sums=z.new_zeros(2*len(ids)).index_add(0,groups,z);return (sums/torch.bincount(groups,minlength=2*len(ids))).reshape(len(ids),2)
    items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({b['path']:(b,packets[b['id']],r) for b,r in zip(before,previous)})
    def score(z,loss):
        probability=z.sigmoid().cpu().numpy();false={};healthy=fixed;old_h=0
        for offset,i in enumerate(val):
            row=rows[i];original,packet,retained=items[row['path']];pair=probability[offset];after=route_paired(original,retained,packet,float(pair.min()),float(pair.max()),.95,data['manifest'])
            was=retained['app_replay']['kind']=='HEALTHY';now=after['app_replay']['kind']=='HEALTHY'
            if not row['y']:false[row['source']]=false.get(row['source'],0)+int(now)-int(was)
            elif row['source']=='olid_tomato':healthy+=int(now)
            else:old_h+=int(now)
        return (-sum(max(0,v) for v in false.values()),healthy,old_h,-loss),false
    prefix=[];suffix=[];head=[]
    for name,param in backbone.named_parameters():
        (head if name.startswith(('head.','fc_norm.')) else suffix if name.startswith('norm.') or any(name.startswith(f'blocks.{i}.') for i in range(cut,len(backbone.blocks))) else prefix).append(param)
    optimizer=torch.optim.AdamW([dict(params=prefix,lr=1e-5),dict(params=suffix,lr=1e-5),dict(params=head,lr=1e-4)],weight_decay=.01)
    history=[];best=None;stale=0;start=0
    if (out/'last.pt').exists():
        saved=torch.load(out/'last.pt',weights_only=True,map_location='cpu');assert saved['protocol_sha256']==digest(out/'protocol.json');backbone.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);history,best,stale,start=saved['history'],saved['best'],saved['stale'],saved['epoch']+1
    for epoch in range(start,12):
        if epoch>=4 and stale>=4:break
        backbone.train();order=torch.multinomial(sampling,len(train),replacement=True,generator=torch.Generator().manual_seed(211700+epoch))
        for batch in order.split(8):
            ids=train[batch.numpy()];z=forward(ids);raw=torch.where(truth[ids]==1,z.min(1).values,z.max(1).values);loss=F.binary_cross_entropy_with_logits(raw,truth[ids]);assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(backbone.parameters(),1.);optimizer.step()
        backbone.eval()
        with torch.inference_mode():
            z=torch.cat([forward(ids) for ids in np.array_split(val,max(1,(len(val)+7)//8))]);raw=torch.where(truth[val]==1,z.min(1).values,z.max(1).values);loss=float((F.binary_cross_entropy_with_logits(raw,truth[val],reduction='none')*vw).sum());decision,false=score(z,loss)
        state={key:value.detach().cpu().clone() for key,value in backbone.state_dict().items()}
        if best is None or decision>tuple(best):best=decision;stale=0;atomic_save(dict(model=state,epoch=epoch,decision=decision,protocol_sha256=digest(out/'protocol.json')),out/'best.pt')
        else:stale+=1
        history.append(dict(epoch=epoch,decision=decision,false_healthy_delta_by_source=false));print(json.dumps(history[-1]),flush=True)
        atomic_save(dict(model=state,epoch=epoch,optimizer=optimizer.state_dict(),rng=capture_rng(),history=history,best=best,stale=stale,protocol_sha256=digest(out/'protocol.json')),out/'last.pt')
    chosen=torch.load(out/'best.pt',weights_only=True,map_location='cpu')
    result=dict(cheap_pass=best[0]==0 and best[1]>=28,best_decision=best,selected_epoch=chosen['epoch'],history=history,source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),checkpoint_sha256=digest(out/'best.pt'),
                elapsed_seconds=time.monotonic()-started,full_cohort_inference_run=False,model_promoted=False,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)


if __name__=='__main__':main()
