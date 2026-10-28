"""One bounded full-CNN adaptation after frozen-prefix controls plateaued.

Uses only consumed development data and existing exact canonical RGB identities.
This pilot cannot promote/export a model; full gates follow only a cheap pass.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
import timm,torch
from torch.nn import functional as F
from cnn_bag_cache import CopiedCnnBagTail
from opened_context_data import load,read
from partial_cnn_cache import original_cnn
from partial_cnn_healthy import prefix_cut,prefix_features
from partial_dino_cache import CanonicalInputs,input_rows
from permissive_encoder import digest
from phase85_population_sft import evaluation_data
from phase87_diagnostic_bag_features import prepare
from phase88_diagnostic_bag_head import route_paired
from population_siglip_tail import freeze
from source_balance import source_class_group_weights
from training_support import atomic_save,capture_rng,restore_rng


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase110_full_cnn_pilot';out.mkdir(exist_ok=True)
    if (out/'result.json').exists():
        assert read(out/'result.json')['source_sha256']==digest(__file__);return
    started=time.monotonic();torch.set_num_threads(4);torch.manual_seed(2110)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;timm.layers.set_fused_attn(False)
    data=load(p);values,bags,metadata=prepare(p,data)
    entries,records,packets,before,previous=evaluation_data(p,data)
    rows=read(p/'phase84_population_head/fitting_rows.json')
    active=[i for i,row in enumerate(rows) if row['active'] and row['path'] in bags and bags[row['path']]['center']]
    val=np.array([i for i in active if rows[i]['validation']])
    val_inputs={key for i in val for view in bags[rows[i]['path']].values() for key in view}
    excluded={rows[i]['group'] for i in active if not rows[i]['validation'] and any(key in val_inputs for view in bags[rows[i]['path']].values() for key in view)}
    train=np.array([i for i in active if not rows[i]['validation'] and rows[i]['group'] not in excluded])
    assert not ({rows[i]['group'] for i in train}&{rows[i]['group'] for i in val})
    assert not ({key for i in train for view in bags[rows[i]['path']].values() for key in view}&val_inputs)
    selected=sorted({key for i in active for view in bags[rows[i]['path']].values() for key in view})
    descriptors=input_rows(p)[:len(read(p/'phase29_compact/inputs.json'))]+[dict(row,kind='new',policy=row['view']) for row in read(p/'phase87_diagnostic_bag_features/new_inputs.json')]
    protocol=dict(source_sha256=digest(__file__),warm_checkpoint_sha256=digest(p/'phase101_cnn_bag_adaptation/best.pt'),feature_cache=metadata,
                  canonical_input_sha256=[descriptors[i]['input_sha256'] for i in selected],fitting_rows_sha256=digest(p/'phase84_population_head/fitting_rows.json'),
                  train_images=len(train),validation_images=len(val),seed=2110,maximum_epochs=12,patience=4,
                  prefix_learning_rate=1e-5,suffix_learning_rate=1e-4,weight_decay=.01,batch_images=16,
                  hypothesis='Frozen spatial prefix may prevent Healthy domain adaptation. Warm-start the selected copied CNN suffix, then allow one low-rate full-backbone adaptation. Preserve original nine-class app model and freeze BatchNorm statistics.',
                  selection='Actual routed validation verdicts: no source false-Healthy increase, then Healthy recovery, then earlier Healthy and weighted BCE',
                  stopping='Stop after four stale routed validation epochs or twelve total; full-cohort inference only if >=28/40 consumed validation Healthy and no source false-Healthy increase; no export or acceptance opening from this pilot.',
                  thresholds=dict(admit=.95,veto=.001),ad_independent=False,olid_independent=False,fresh_data_used=False,app_model_changed=False)
    cnn,architecture,normalization=original_cnn(p)
    warm=torch.load(p/'phase101_cnn_bag_adaptation/best.pt',weights_only=True,map_location='cpu')
    tail=CopiedCnnBagTail(cnn);tail.load_state_dict(warm['model'])
    cut=prefix_cut(cnn)
    for index,module in enumerate(tail.blocks,start=cut):cnn.blocks[index]=module
    for name,module in zip(('global_pool','conv_head','norm_head','act2','flatten'),tail.norm):setattr(cnn,name,module)
    cnn.classifier=tail.head;cnn=cnn.cuda().requires_grad_(True)
    pixels=CanonicalInputs(p)
    rgb=np.concatenate([pixels(descriptors[i]) for i in selected]);pixels.close()
    raw=torch.from_numpy(rgb).cuda().permute(0,3,1,2);del rgb
    normal={name:value.cuda() for name,value in normalization.items()}
    raw=(raw*normal['canonical_std']+normal['canonical_mean']-normal['cnn_mean'])/normal['cnn_std']
    cnn.eval();tail=tail.cuda().eval()
    with torch.inference_mode():
        expected=tail(prefix_features(cnn,raw[:16]),None);actual=cnn(raw[:16]);actual=actual[:,0]-actual[:,1]
        parity=float((actual-expected).abs().max())
    assert parity<1e-4,('Full CNN warm suffix restoration mismatch',parity)
    protocol['warm_suffix_restore_max_error']=parity
    freeze(out/'protocol.json',protocol)
    del tail
    lookup={key:offset for offset,key in enumerate(selected)}
    tables={i:[[lookup[key] for key in bags[rows[i]['path']][view]] for view in ('center','full')] for i in active}
    truth=torch.tensor([row['y'] for row in rows],dtype=torch.float32,device='cuda')
    sampling=torch.from_numpy(source_class_group_weights(rows,train));vw=torch.tensor(source_class_group_weights(rows,val),device='cuda')
    path_items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group}
    path_items.update({b['path']:(b,packets[b['id']],r) for b,r in zip(before,previous)})
    def forward(ids):
        flat=[];groups=[]
        for offset,index in enumerate(ids):
            for view,indices in enumerate(tables[index]):flat.extend(indices);groups.extend([2*offset+view]*len(indices))
        unique,inverse=torch.tensor(flat,device='cuda').unique(return_inverse=True)
        logits=cnn(raw[unique]);margin=(logits[:,0]-logits[:,1])[inverse]
        groups=torch.tensor(groups,device='cuda');totals=margin.new_zeros(2*len(ids)).index_add(0,groups,margin)
        return (totals/torch.bincount(groups,minlength=2*len(ids))).reshape(len(ids),2)
    def evaluate(z,loss):
        probability=z.sigmoid().cpu().numpy();false={};healthy=0;old_healthy=0
        for offset,index in enumerate(val):
            row=rows[index];original,packet,retained=path_items[row['path']];pair=probability[offset]
            revised=route_paired(original,retained,packet,float(pair.min()),float(pair.max()),.95,data['manifest'])
            was_h=retained['app_replay']['kind']=='HEALTHY';is_h=revised['app_replay']['kind']=='HEALTHY'
            if row['y']==0:false[row['source']]=false.get(row['source'],0)+int(is_h)-int(was_h)
            elif row['source']=='olid_tomato':healthy+=int(is_h)
            else:old_healthy+=int(is_h)
        return (-sum(max(0,n) for n in false.values()),healthy,old_healthy,-loss),false
    prefix=[];suffix=[]
    for name,param in cnn.named_parameters():
        (suffix if name.startswith(('classifier.','conv_head.','norm_head.')) or any(name.startswith(f'blocks.{i}.') for i in range(cut,len(cnn.blocks))) else prefix).append(param)
    optimizer=torch.optim.AdamW([dict(params=prefix,lr=1e-5),dict(params=suffix,lr=1e-4)],weight_decay=.01)
    history=[];best=None;stale=0;start=0
    if (out/'last.pt').exists():
        saved=torch.load(out/'last.pt',weights_only=True,map_location='cpu');assert saved['protocol_sha256']==digest(out/'protocol.json')
        cnn.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
        history,best,stale,start=saved['history'],saved['best'],saved['stale'],saved['epoch']+1
    for epoch in range(start,12):
        if epoch>=4 and stale>=4:break
        cnn.train()
        for module in cnn.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
        order=torch.multinomial(sampling,len(train),replacement=True,generator=torch.Generator().manual_seed(211000+epoch))
        for batch in order.split(16):
            ids=train[batch.numpy()];z=forward(ids);margin=torch.where(truth[ids]==1,z.min(1).values,z.max(1).values)
            loss=F.binary_cross_entropy_with_logits(margin,truth[ids]);assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(cnn.parameters(),1.);optimizer.step()
        cnn.eval()
        with torch.inference_mode():
            z=torch.cat([forward(ids) for ids in np.array_split(val,max(1,(len(val)+15)//16))]);margin=torch.where(truth[val]==1,z.min(1).values,z.max(1).values)
            loss=float((F.binary_cross_entropy_with_logits(margin,truth[val],reduction='none')*vw).sum());score,false=evaluate(z,loss)
        weights={name:value.detach().cpu().clone() for name,value in cnn.state_dict().items()}
        if best is None or score>tuple(best):
            best=score;stale=0;atomic_save(dict(model=weights,epoch=epoch,decision=score,protocol_sha256=digest(out/'protocol.json')),out/'best.pt')
        else:stale+=1
        history.append(dict(epoch=epoch,decision=score,false_healthy_delta_by_source=false));print(json.dumps(history[-1]),flush=True)
        atomic_save(dict(model=weights,epoch=epoch,optimizer=optimizer.state_dict(),rng=capture_rng(),history=history,best=best,stale=stale,protocol_sha256=digest(out/'protocol.json')),out/'last.pt')
    saved=torch.load(out/'best.pt',weights_only=True,map_location='cpu')
    result=dict(cheap_pass=best[0]==0 and best[1]>=28,selected_epoch=saved['epoch'],best_decision=best,history=history,
                checkpoint_sha256=digest(out/'best.pt'),protocol_sha256=digest(out/'protocol.json'),source_sha256=digest(__file__),
                elapsed_seconds=time.monotonic()-started,ad_independent=False,olid_independent=False,model_promoted=False,full_cohort_inference_run=False)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
