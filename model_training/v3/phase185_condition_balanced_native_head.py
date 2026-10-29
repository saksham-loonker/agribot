"""Bounded cached clinical head on native-518 DINO and existing CNN texture."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from torch.nn import functional as F
from balanced_clinical_loss import balanced_clinical_loss
from conditional_recovery_policy import recover_only
from hard_negative_ranking import hard_negative_rank_loss
from joint_clinical_head import JointClinicalHead
from native_joint_features import join_native_cnn
from opened_context_data import load,read
from partial_dino_healthy import healthy_margin
from permissive_encoder import digest
from phase85_population_sft import evaluation_data
from population_siglip_tail import freeze
from clinical_replay_balance import clinical_replay_weights
from training_support import atomic_save,capture_rng,restore_rng


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase185_condition_balanced_native_head';out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    source=p/'phase172_native_large_field_features';cp=read(source/'protocol.json');cc=read(source/'complete.json');assert cc['protocol_sha256']==digest(source/'protocol.json') and cc['features_sha256']==digest(source/'features.npy') and cc['native_positional_embeddings_exact']
    rows=read(p/'phase168_early_cnn_prefix/rows.json');assert digest(p/'phase168_early_cnn_prefix/rows.json')==cp['rows_sha256'];population=read(p/'phase171_field_replay_ablation/protocol.json');assert digest(p/'phase171_field_replay_ablation/protocol.json')==cp['population_protocol_sha256']
    conv=p/'phase154_convnext_features';proto=read(conv/'protocol.json');complete=read(conv/'complete.json');assert complete['features_sha256']==digest(conv/'features.npy') and digest(conv/'complete.json')==cp['convnext_complete_sha256']
    cnn={sha:value for sha,value in zip(proto['input_sha256'],np.load(conv/'features.npy'))};field=p/'phase164_new_field_features';fc=read(field/'complete.json');assert digest(field/'complete.json')==cp['field_complete_sha256'] and fc['features_sha256']==digest(field/'features.npy') and fc['input_sha256_sha256']==digest(field/'input_sha256.json')
    cnn.update({sha:value[2432:] for sha,value in zip(read(field/'input_sha256.json'),np.load(field/'features.npy'))})
    keys=[row['input_sha256'] for row in cp['inputs']];features=join_native_cnn(np.load(source/'features.npy'),keys,cp['links'],cnn)
    train=np.array(cp['train_indices']);val=np.array(cp['validation_indices']);selection=read(p/'phase118_conditional_recovery/protocol.json');assert val.tolist()==selection['validation_indices'] and train.tolist()==population['train_indices']
    earlier=read(p/'phase168_early_cnn_prefix/protocol.json');target=earlier['target_indices'];anchors=earlier['anchor_indices'];field_indices=earlier['field_indices'];assert train.tolist()==target+anchors+field_indices
    torch.set_num_threads(4);torch.manual_seed(2158);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    features=torch.from_numpy(features).cuda();index=np.zeros((len(rows),8,6),np.int64);mask=np.zeros(index.shape,np.float32)
    for i in train.tolist()+val.tolist():
        for view,ids in enumerate(cp['bags'][rows[i]['path']]):index[i,view,:len(ids)]=ids;mask[i,view,:len(ids)]=1
    index=torch.from_numpy(index).cuda();mask=torch.from_numpy(mask).cuda();active=mask.sum(-1)>0;truth=torch.tensor([r['y'] for r in rows],dtype=torch.float32,device='cuda')
    data=load(p);labels=data['manifest']['classifier']['labels'];fine=torch.tensor([labels.index(r['label']) if r['label'] in labels else -1 for r in rows],device='cuda');entries,records,packets,before,previous=evaluation_data(p,data)
    items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({b['path']:(b,packets[b['id']],r) for b,r in zip(before,previous)})
    warm=p/'phase152_balanced_photo_teacher/balanced_photo_clinical_teacher/head_warm_last.pt';saved=torch.load(warm,weights_only=True,map_location='cpu');assert saved['protocol_sha256']==digest(p/'phase152_balanced_photo_teacher/protocol.json')
    model=JointClinicalHead().cuda();model.load_warm_head(saved['head']);sampling=torch.from_numpy(np.concatenate([.5*clinical_replay_weights(rows,target),.25*clinical_replay_weights(rows,anchors),.25*clinical_replay_weights(rows,field_indices)]))
    protocol=dict(source_sha256=digest(__file__),sampler_sha256=digest(Path(__file__).with_name('clinical_replay_balance.py')),sampler='Equal binary/source mass, equal clinical condition inside each binary/source, equal family inside each clinical condition',native_complete_sha256=digest(source/'complete.json'),native_protocol_sha256=digest(source/'protocol.json'),replay_ablation_protocol_sha256=digest(p/'phase171_field_replay_ablation/protocol.json'),
        join_source_sha256=digest(Path(__file__).with_name('native_joint_features.py')),head_source_sha256=digest(Path(__file__).with_name('joint_clinical_head.py')),clinical_source_sha256=digest(Path(__file__).with_name('balanced_clinical_loss.py')),warm_head_sha256=digest(warm),
        train_indices=train.tolist(),validation_indices=val.tolist(),training_mass=dict(target=.5,old_anchor=.25,field93=.25,old107=0.),old107_remains_in_warm_head_history=True,
        hypothesis='Two high-confidence false Healthy cases are Nitrogen Deficiency. The previous binary/source sampler gives common Other examples more disease replay than sparse named conditions. Test equal conditional clinical-class mass without changing Healthy mass, data, features, objective, calibration or guards. This cached head is a cheap test before changing encoder adaptation.',
        maximum_epochs=30,minimum_epochs=6,patience=6,draws_per_epoch=1024,batch_photos=64,learning_rate=.0003,weight_decay=.01,seed=2158,clinical_weight=1.,binary_weight=.3,ranking_weight=.2,ranking_margin=1.,maximum_hard_negatives=4,
        existing_healthy_and_named_disease_preserved=True,admission=.95,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False,candidate_role='Native-resolution teacher probe; mobile student qualification remains necessary',
        stopping='One supervised native-resolution control, six-stale-epoch stop after minimum six and maximum thirty; exit immediately at >=28/40 Healthy and zero new false Healthy. Only then extend missing native identities and run complete unchanged gates before restored inference or export.')
    freeze(out/'protocol.json',protocol)
    def forward(ids):
        logits=model(features[index[ids]]);clinical=(logits*mask[ids][...,None]).sum(2)/mask[ids].sum(2).clamp_min(1)[...,None]
        raw=healthy_margin(logits.reshape(-1,9)).reshape(logits.shape[:-1]);means=(raw*mask[ids]).sum(2)/mask[ids].sum(2).clamp_min(1)
        return means.masked_fill(~active[ids],float('inf')).min(1).values,means.masked_fill(~active[ids],-float('inf')).max(1).values,balanced_clinical_loss(clinical,fine[ids],active[ids],1)
    def decision(probability,loss):
        errors={};healthy=selection['fixed_healthy_validation_correct'];old_h=0
        for i,score in zip(val,probability):
            row=rows[i];original,packet,retained=items[row['path']];now=recover_only(original,retained,packet,float(score),data['manifest'])['app_replay']['kind']=='HEALTHY'
            if not row['y']:errors[row['source']]=errors.get(row['source'],0)+int(now)
            elif row['source']=='olid_tomato':healthy+=int(now)
            else:old_h+=int(now)
        return (-sum(errors.values()),healthy,old_h,-loss),errors
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01);history=[];best=None;stale=0;start=0
    if (out/'last.pt').exists():
        state=torch.load(out/'last.pt',weights_only=True,map_location='cpu');assert state['protocol_sha256']==digest(out/'protocol.json');model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng']);history,best,stale,start=state['history'],state['best'],state['stale'],state['epoch']+1
    for epoch in range(start,30):
        if epoch>=6 and stale>=6:break
        model.train();order=torch.multinomial(sampling,1024,replacement=True,generator=torch.Generator().manual_seed(215800+epoch))
        for batch in order.split(64):
            ids=train[batch.numpy()];positive,negative,clinical=forward(ids);raw=torch.where(truth[ids]==1,positive,negative);rank=hard_negative_rank_loss(torch.stack((positive,negative),1),truth[ids],margin=1.,maximum_negatives=4);loss=clinical+.3*F.binary_cross_entropy_with_logits(raw,truth[ids])+.2*rank
            assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        model.eval()
        with torch.inference_mode():positive,negative,_=forward(val);loss=float(F.binary_cross_entropy_with_logits(torch.where(truth[val]==1,positive,negative),truth[val]));probability=positive.sigmoid().cpu().numpy();score,errors=decision(probability,loss)
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if best is None or score>tuple(best):
            best=score;stale=0;atomic_save(dict(model=state,epoch=epoch,protocol_sha256=digest(out/'protocol.json')),out/'best.pt');np.savez(out/'best_validation_predictions.npz',indices=val,probability=probability,protocol_sha256=digest(out/'protocol.json'),checkpoint_sha256=digest(out/'best.pt'))
        else:stale+=1
        history.append(dict(epoch=epoch,decision=score,new_false_healthy_by_source=errors));print(json.dumps(history[-1]),flush=True)
        atomic_save(dict(model=state,optimizer=optimizer.state_dict(),rng=capture_rng(),epoch=epoch,history=history,best=best,stale=stale,protocol_sha256=digest(out/'protocol.json')),out/'last.pt')
        if best[0]==0 and best[1]>=28:break
    result=dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),checkpoint_sha256=digest(out/'best.pt'),selected_epoch=torch.load(out/'best.pt',weights_only=True,map_location='cpu')['epoch'],best_decision=best,history=history,cheap_pass=best[0]==0 and best[1]>=28,passed=False,model_promoted=False,new_encoder_inputs=0,elapsed_seconds=time.monotonic()-started,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    if result['cheap_pass']:result['next_action']='Freeze candidate and extend exact native features/predictions only for missing guard-applicable old/consumed photos, then run every unchanged full development gate before restored-model verification or export'
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)


if __name__=='__main__':main()
