"""Bounded cached clinical head on native-518 DINO and existing CNN texture."""
from pathlib import Path
import argparse,json,time,collections
from clinical_replay_balance import clinical_replay_weights
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
from source_balance import source_class_group_weights
from training_support import atomic_save,capture_rng,restore_rng


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);parser.add_argument('--balance',choices=('binary','clinical'),required=True);args=parser.parse_args()
    p=Path(args.campaign);out=p/'phase190_full_field_native_head'/args.balance;out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    source=p/'phase189_full_field_native_extension';cp=read(source/'protocol.json');cc=read(source/'complete.json');assert cc['protocol_sha256']==digest(source/'protocol.json') and cc['features_sha256']==digest(source/'features.npy')
    rows=read(p/'phase168_early_cnn_prefix/rows.json');assert digest(p/'phase168_early_cnn_prefix/rows.json')==cp['rows_sha256'];features=np.load(source/'features.npy');assert features.shape==tuple(cp['shape'])
    train=np.array(cp['train_indices']);val=np.array(cp['validation_indices']);selection=read(p/'phase118_conditional_recovery/protocol.json');assert val.tolist()==selection['validation_indices'];target=cp['target_indices'];anchors=cp['anchor_indices'];field_indices=cp['field_indices'];assert train.tolist()==target+anchors+field_indices
    balance=clinical_replay_weights if args.balance=='clinical' else source_class_group_weights
    torch.set_num_threads(4);torch.manual_seed(2158);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    features=torch.from_numpy(features).cuda();index=np.zeros((len(rows),8,6),np.int64);mask=np.zeros(index.shape,np.float32)
    for i in train.tolist()+val.tolist():
        for view,ids in enumerate(cp['bags'][rows[i]['path']]):index[i,view,:len(ids)]=ids;mask[i,view,:len(ids)]=1
    index=torch.from_numpy(index).cuda();mask=torch.from_numpy(mask).cuda();active=mask.sum(-1)>0;truth=torch.tensor([r['y'] for r in rows],dtype=torch.float32,device='cuda')
    data=load(p);labels=data['manifest']['classifier']['labels'];fine=torch.tensor([labels.index(r['label']) if r['label'] in labels else -1 for r in rows],device='cuda');entries,records,packets,before,previous=evaluation_data(p,data)
    items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({b['path']:(b,packets[b['id']],r) for b,r in zip(before,previous)})
    warm=p/'phase152_balanced_photo_teacher/balanced_photo_clinical_teacher/head_warm_last.pt';saved=torch.load(warm,weights_only=True,map_location='cpu');assert saved['protocol_sha256']==digest(p/'phase152_balanced_photo_teacher/protocol.json')
    model=JointClinicalHead().cuda();model.load_warm_head(saved['head']);sampling=torch.from_numpy(np.concatenate([.5*balance(rows,target),.25*balance(rows,anchors),.25*balance(rows,field_indices)]))
    protocol=dict(source_sha256=digest(__file__),native_complete_sha256=digest(source/'complete.json'),native_protocol_sha256=digest(source/'protocol.json'),sampling=args.balance,sampler_sha256=digest(Path(__file__).with_name('clinical_replay_balance.py')),all_field_training_restored=True,
        join_source_sha256=digest(Path(__file__).with_name('native_joint_features.py')),head_source_sha256=digest(Path(__file__).with_name('joint_clinical_head.py')),clinical_source_sha256=digest(Path(__file__).with_name('balanced_clinical_loss.py')),warm_head_sha256=digest(warm),
        train_indices=train.tolist(),validation_indices=val.tolist(),training_mass=dict(target=.5,old_anchor=.25,field93=.25,old107=0.),old107_remains_in_warm_head_history=True,
        hypothesis='The 71 omitted quarantined field images, including 66 Healthy, were restored only in the mobile representation. Restore them to the native semantic/texture teacher as well. Compare binary-balanced and condition-balanced replay with all other data, optimizer and gates fixed. Use training-loss convergence rather than low-recall-incumbent validation staleness, and preserve every epoch.',
        maximum_epochs=60,minimum_epochs=20,training_ema_patience=12,save_every_epoch=True,validation_staleness_stops_training=False,draws_per_epoch=1024,batch_photos=64,learning_rate=.0003,weight_decay=.01,seed=2158,clinical_weight=1.,binary_weight=.3,ranking_weight=.2,ranking_margin=1.,maximum_hard_negatives=4,
        existing_healthy_and_named_disease_preserved=True,admission=.95,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False,candidate_role='Native-resolution teacher probe; mobile student qualification remains necessary',
        stopping='At least twenty epochs, then twelve stale training-EMA epochs with .001 minimum improvement, sixty epochs maximum. Stop earlier only at unchanged cheap gate success. Preserve every checkpoint/prediction vector. No export before complete original gates.')
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
    ema=history[-1]['training_loss_ema'] if history else None;training_best=min((r['training_loss_ema'] for r in history),default=float('inf'));training_stale=history[-1]['training_stale'] if history else 0
    for epoch in range(start,60):
        if epoch>=20 and training_stale>=12:break
        batch_losses=[]
        model.train();order=torch.multinomial(sampling,1024,replacement=True,generator=torch.Generator().manual_seed(215800+epoch))
        for batch in order.split(64):
            ids=train[batch.numpy()];positive,negative,clinical=forward(ids);raw=torch.where(truth[ids]==1,positive,negative);rank=hard_negative_rank_loss(torch.stack((positive,negative),1),truth[ids],margin=1.,maximum_negatives=4);loss=clinical+.3*F.binary_cross_entropy_with_logits(raw,truth[ids])+.2*rank
            assert torch.isfinite(loss);batch_losses.append(float(loss.detach()));optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        model.eval()
        with torch.inference_mode():positive,negative,_=forward(val);loss=float(F.binary_cross_entropy_with_logits(torch.where(truth[val]==1,positive,negative),truth[val]));probability=positive.sigmoid().cpu().numpy();score,errors=decision(probability,loss)
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if best is None or score>tuple(best):
            best=score;stale=0;atomic_save(dict(model=state,epoch=epoch,protocol_sha256=digest(out/'protocol.json')),out/'best.pt');np.savez(out/'best_validation_predictions.npz',indices=val,probability=probability,protocol_sha256=digest(out/'protocol.json'),checkpoint_sha256=digest(out/'best.pt'))
        else:stale+=1
        train_loss=float(np.mean(batch_losses));ema=train_loss if ema is None else .9*ema+.1*train_loss
        if ema<training_best-.001:training_best=ema;training_stale=0
        else:training_stale+=1
        checkpoint=out/f'epoch_{epoch:03d}.pt';atomic_save(dict(model=state,epoch=epoch,protocol_sha256=digest(out/'protocol.json')),checkpoint)
        np.savez(out/f'epoch_{epoch:03d}_predictions.npz',indices=val,probability=probability,protocol_sha256=digest(out/'protocol.json'),checkpoint_sha256=digest(checkpoint))
        history.append(dict(epoch=epoch,decision=score,new_false_healthy_by_source=errors,training_loss=train_loss,training_loss_ema=ema,training_stale=training_stale,checkpoint_sha256=digest(checkpoint)));print(json.dumps(history[-1]),flush=True)
        atomic_save(dict(model=state,optimizer=optimizer.state_dict(),rng=capture_rng(),epoch=epoch,history=history,best=best,stale=stale,protocol_sha256=digest(out/'protocol.json')),out/'last.pt')
        if best[0]==0 and best[1]>=28:break
    result=dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),checkpoint_sha256=digest(out/'best.pt'),selected_epoch=torch.load(out/'best.pt',weights_only=True,map_location='cpu')['epoch'],best_decision=best,history=history,cheap_pass=best[0]==0 and best[1]>=28,passed=False,model_promoted=False,new_encoder_inputs=0,elapsed_seconds=time.monotonic()-started,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    if result['cheap_pass']:result['next_action']='Freeze candidate and extend exact native features/predictions only for missing guard-applicable old/consumed photos, then run every unchanged full development gate before restored-model verification or export'
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)


if __name__=='__main__':main()
