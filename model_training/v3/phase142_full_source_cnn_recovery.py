"""Two bounded recovery-only heads using the full audited development source."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from torch.nn import functional as F
from conditional_recovery_policy import recover_only
from hard_negative_ranking import hard_negative_rank_loss
from cnn_bag_cache import CopiedCnnBagTail
from partial_cnn_cache import original_cnn
from partial_dino_cache import input_rows
from opened_context_data import load,read
from phase85_population_sft import evaluation_data
from phase114_new_supervision_features import original_features
from permissive_encoder import digest
from population_siglip_tail import freeze
from source_balance import source_class_group_weights
from training_support import atomic_save,capture_rng,restore_rng


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);root=p/'phase142_full_source_cnn_recovery';root.mkdir(exist_ok=True);started=time.monotonic()
    if (root/'result.json').exists():assert read(root/'result.json')['source_sha256']==digest(__file__);return
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    values,bags,descriptors,parent=original_features(p);rows=read(p/'phase84_population_head/fitting_rows.json');selection=read(p/'phase118_conditional_recovery/protocol.json');anchors=read(p/'phase133_anchor_input_cache/protocol.json')
    extra=p/'phase137_full_source_features';complete=read(extra/'complete.json');assert complete['protocol_sha256']==digest(extra/'protocol.json')
    for name,field in [('new_features.npy','new_features_sha256'),('bags.json','bags_sha256'),('new_input_sha256.json','new_keys_sha256')]:assert digest(extra/name)==complete[field]
    protected=p/'phase138_full_source_reservation_overlap/result.json';assert read(protected)['overlap_images']==0 and not read(protected)['clinical_predictions_opened']
    identities=[r['input_sha256'] for r in descriptors]+read(extra/'new_input_sha256.json');values=np.concatenate([values,np.load(extra/'new_features.npy')]);assert len(values)==len(identities)
    target=anchors['target_indices'];replay=anchors['anchor_indices'];val=np.array(selection['validation_indices']);old=target+replay+val.tolist()
    tables={i:[[bags[rows[i]['path']][view] for view in ('center','full')]] for i in old};new=[]
    for path,bag in read(extra/'bags.json').items():
        if not bag['actual_leaves']:continue
        new.append(len(rows));record=bag['source_record'];tables[len(rows)]=bag['views']
        rows.append(dict(path=path,y=int(record['label']=='Healthy'),source='new107_development',group='new107:entire_publisher_collection',validation=False,label=record['label']))
    train=np.array(target+replay+new);assert len(new)>1800
    val_keys={identities[key] for i in val for pair in tables[i] for keys in pair for key in keys}
    assert not ({identities[key] for i in train for pair in tables[i] for keys in pair for key in keys}&val_keys)
    assert not ({rows[i]['group'] for i in train}&{rows[i]['group'] for i in val})
    warm=p/'export_cnn_safe/pytorch_bundle.pt';calibration=[1.,0.]
    cache=p/'phase141_full_source_cnn_prefix';cp=read(cache/'protocol.json');cc=read(cache/'complete.json');assert cc['protocol_sha256']==digest(cache/'protocol.json') and cc['missing_prefix_sha256']==digest(cache/'missing_prefix.npy')
    cnn,_,_=original_cnn(p);backbone=cnn.cuda().eval();donor=p/'phase49_partial_cnn/prefix';dv=np.load(donor/'features.npy',mmap_mode='r');available={r['input_sha256']:(dv,i) for i,r in enumerate(input_rows(p))}
    extension=p/'phase101_cnn_bag_adaptation/prefix';ep=read(extension/'protocol.json');ev=np.load(extension/'missing_prefix.npy',mmap_mode='r')
    for offset,key in enumerate(ep['missing_tables']):available[descriptors[key]['input_sha256']]=(ev,offset)
    all_keys=[r['input_sha256'] for r in descriptors]+read(extra/'new_input_sha256.json');missing=np.load(cache/'missing_prefix.npy',mmap_mode='r');missing_lookup={key:i for i,key in enumerate(cp['missing_tables'])};selected=cp['selected_tables'];prefix_lookup={key:i for i,key in enumerate(selected)}
    prefix=torch.empty((len(selected),*dv.shape[1:]),dtype=torch.float32,device='cuda')
    for start in range(0,len(selected),128):
        part=selected[start:start+128];array=np.stack([missing[missing_lookup[key]] if key in missing_lookup else available[all_keys[key]][0][available[all_keys[key]][1]] for key in part]);prefix[start:start+len(part)]=torch.from_numpy(array).cuda()
    methods=['full_source_cnn_sft','full_source_cnn_rank']
    protocol=dict(source_sha256=digest(__file__),policy_source_sha256=digest(Path(__file__).with_name('conditional_recovery_policy.py')),rank_source_sha256=digest(Path(__file__).with_name('hard_negative_ranking.py')),
        parents=parent,extra_complete_sha256=digest(extra/'complete.json'),anchor_selection_sha256=digest(p/'phase133_anchor_input_cache/protocol.json'),protected_audit_sha256=digest(protected),warm_checkpoint_sha256=digest(warm),spatial_prefix_complete_sha256=digest(cache/'complete.json'),cnn_wrapper_source_sha256=digest(Path(__file__).with_name('cnn_bag_cache.py')),
        methods=methods,train_indices=train.tolist(),validation_indices=val.tolist(),target_indices=target,old_anchor_indices=replay,new_source_training_images=len(new),target_weight=.5,old_anchor_weight=.25,new_source_weight=.25,
        hypothesis='Frozen features and full-source heads lacked transfer. Adapt copied spatial CNN final stages and binary head on all new-source images with fixed BatchNorm statistics, target-error replay and protected clinical anchors; compare supervised BCE and a separately tested ranking loss.',
        maximum_epochs=60,minimum_epochs=20,patience=12,draws_per_epoch=1024,batch_images=16,learning_rate=.0001,weight_decay=.01,negative_loss_weight=1.,ranking_weight=.2,ranking_margin=1.,maximum_hard_negatives=4,seed=2139,
        calibration=calibration,healthy_admission=.95,existing_healthy_and_disease_preserved=True,original_safety_and_recognition_gates_unchanged=True,
        supervision='One publisher image label for actual ROI bags, mean ROI logits per crop/rotation; Healthy uses minimum view and disease maximum view; no per-leaf clinical labels',
        new_source_role='Entire collection development-only; camera/style confounding unresolved; unused remainder never independent acceptance',
        ad_independent=False,olid_independent=False,acceptance_data_used=False,new_encoder_inputs=0,
        stopping='Two predeclared objectives, minimum twenty epochs, twelve stale actual-verdict epochs, sixty maximum. Require >=28/40 Healthy and zero new source false Healthy before full cached screening; no restored inference or export before all unchanged gates pass.')
    freeze(root/'protocol.json',protocol)
    truth=torch.tensor([r['y'] for r in rows],dtype=torch.float32,device='cuda');index=np.zeros((len(rows),4,2,6),np.int64);mask=np.zeros(index.shape,np.float32)
    for i,pairs in tables.items():
        for rotation,pair in enumerate(pairs):
            for view,keys in enumerate(pair):index[i,rotation,view,:len(keys)]=[prefix_lookup[key] for key in keys];mask[i,rotation,view,:len(keys)]=1.
    index=torch.from_numpy(index).cuda();mask=torch.from_numpy(mask).cuda();valid=mask.sum(-1)>0
    data=load(p);entries,records,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({b['path']:(b,packets[b['id']],r) for b,r in zip(before,previous)})
    sampling=torch.from_numpy(np.concatenate([.5*source_class_group_weights(rows,target),.25*source_class_group_weights(rows,replay),.25*source_class_group_weights(rows,new)]));vw=torch.tensor(source_class_group_weights(rows,val),device='cuda')
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
        torch.manual_seed(2139);model=CopiedCnnBagTail(backbone).cuda()
        def forward(ids):
            actual=mask[ids]>0;unique,inverse=index[ids][actual].unique(return_inverse=True);z=prefix.new_zeros((len(ids),4,2,6));z[actual]=model(prefix[unique],prefix.new_empty((len(unique),0)))[inverse]
            means=(z*mask[ids]).sum(-1)/mask[ids].sum(-1).clamp_min(1)
            return means.masked_fill(~valid[ids],float('inf')).amin((1,2)),means.masked_fill(~valid[ids],-float('inf')).amax((1,2))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.01);history=[];best=None;stale=0;start=0
        if (out/'last.pt').exists():
            saved=torch.load(out/'last.pt',weights_only=True,map_location='cpu');assert saved['protocol_sha256']==digest(root/'protocol.json');model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);history,best,stale,start=saved['history'],saved['best'],saved['stale'],saved['epoch']+1
        for epoch in range(start,60):
            if epoch>=20 and stale>=12:break
            model.train();order=torch.multinomial(sampling,1024,replacement=True,generator=torch.Generator().manual_seed(213900+epoch))
            for batch in order.split(16):
                ids=train[batch.numpy()];positive,negative=forward(ids);raw=torch.where(truth[ids]==1,positive,negative);loss=F.binary_cross_entropy_with_logits(raw,truth[ids])
                if method=='full_source_cnn_rank':loss=loss+.2*hard_negative_rank_loss(torch.stack([positive,negative],1),truth[ids])
                assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            model.eval()
            with torch.inference_mode():
                pairs=[forward(part) for part in np.array_split(val,max(1,(len(val)+63)//64))];positive=torch.cat([pair[0] for pair in pairs]);negative=torch.cat([pair[1] for pair in pairs]);raw=torch.where(truth[val]==1,positive,negative)
                loss=float((F.binary_cross_entropy_with_logits(raw,truth[val],reduction='none')*vw).sum());score,errors=decision(positive.sigmoid().cpu().numpy(),loss)
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if best is None or score>tuple(best):
                best=score;stale=0;atomic_save(dict(model=state,epoch=epoch,protocol_sha256=digest(root/'protocol.json')),out/'best.pt')
                np.savez(out/'best_validation_predictions.npz',indices=val,probability=positive.sigmoid().cpu().numpy(),protocol_sha256=digest(root/'protocol.json'),checkpoint_sha256=digest(out/'best.pt'))
            else:stale+=1
            history.append(dict(epoch=epoch,decision=score,new_false_healthy_by_source=errors));print(json.dumps(dict(method=method,**history[-1])),flush=True)
            atomic_save(dict(model=state,optimizer=optimizer.state_dict(),rng=capture_rng(),epoch=epoch,history=history,best=best,stale=stale,protocol_sha256=digest(root/'protocol.json')),out/'last.pt')
        chosen=torch.load(out/'best.pt',weights_only=True,map_location='cpu');cheap=best[0]==0 and best[1]>=28
        result=dict(method=method,protocol_sha256=digest(root/'protocol.json'),checkpoint_sha256=digest(out/'best.pt'),best_decision=best,selected_epoch=chosen['epoch'],history=history,cheap_pass=cheap,passed=False,model_promoted=False,new_encoder_inputs=0)
        if cheap:result['next_action']='Extend exact-identity head predictions to all applicable consumed cases, then run unchanged full recognition/safety gates before restored verification or export'
        (out/'result.json').write_text(json.dumps(result,indent=2));controls.append(result)
    result=dict(source_sha256=digest(__file__),protocol_sha256=digest(root/'protocol.json'),controls=controls,passing=0,model_promoted=False,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False,elapsed_seconds=time.monotonic()-started)
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(controls=[dict(method=r['method'],best_decision=r['best_decision'],cheap_pass=r['cheap_pass']) for r in controls])),flush=True)


if __name__=='__main__':main()
