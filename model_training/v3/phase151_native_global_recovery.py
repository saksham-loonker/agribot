"""Test the unused native-448 global CLS cache before any new image encoding."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from torch.nn import functional as F
from conditional_recovery_policy import recover_only
from native_global_head import NativeGlobalHead
from opened_context_data import load,read
from partial_dino_cache import original_dino
from partial_dino_healthy import healthy_margin
from phase85_population_sft import evaluation_data
from phase114_new_supervision_features import original_features
from permissive_encoder import digest
from population_siglip_tail import freeze
from source_balance import source_class_group_weights
from target_replay_sampling import target_replay_weights
from training_support import atomic_save,capture_rng,restore_rng


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase151_native_global_recovery';out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    values,bags,descriptors,parent=original_features(p);rows=read(p/'phase84_population_head/fitting_rows.json');selection=read(p/'phase118_conditional_recovery/protocol.json');train=np.array(selection['train_indices']);val=np.array(selection['validation_indices']);active=train.tolist()+val.tolist()
    source=p/'phase128_native_dino_patch_cache';sp=read(source/'protocol.json');sc=read(source/'complete.json');assert sc['protocol_sha256']==digest(source/'protocol.json') and sc['native_cls_sha256']==digest(source/'native_cls.npy') and sp['parents']==parent
    native=np.load(source/'native_cls.npy');assert native.shape==(len(sp['inputs']),384) and np.isfinite(native).all()
    native=F.normalize(torch.from_numpy(native),dim=1)*(384**.5);features=torch.cat([torch.from_numpy(values[sp['parent_224_feature_indices']]),native],1).cuda();high_bags=sp['bags']
    assert all(rows[i]['path'] in high_bags for i in active)
    assert not ({key for i in train for keys in high_bags[rows[i]['path']].values() for key in keys}&{key for i in val for keys in high_bags[rows[i]['path']].values() for key in keys})
    data=load(p);entries,records,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({b['path']:(b,packets[b['id']],r) for b,r in zip(before,previous)})
    table=np.zeros((len(rows),2,6),np.int64);mask=np.zeros(table.shape,np.float32)
    for i in active:
        for view,name in enumerate(('center','full')):
            keys=high_bags[rows[i]['path']][name];table[i,view,:len(keys)]=keys;mask[i,view,:len(keys)]=1.
    table=torch.from_numpy(table).cuda();mask=torch.from_numpy(mask).cuda();counts=mask.sum(-1).clamp_min(1);truth=torch.tensor([r['y'] for r in rows],dtype=torch.float32,device='cuda')
    warm=p/'phase88_diagnostic_bag_head/mlp';wr=read(warm/'result.json');assert wr['checkpoint_sha256']==digest(warm/'best.pt');calibration=wr['calibration']
    protocol=dict(source_sha256=digest(__file__),head_source_sha256=digest(Path(__file__).with_name('native_global_head.py')),parents=parent,native_complete_sha256=digest(source/'complete.json'),
        warm_checkpoint_sha256=digest(warm/'best.pt'),train_indices=train.tolist(),validation_indices=val.tolist(),methods=['original_native_class_head','linear','mlp'],
        hypothesis='Local-patch experiments retained the old 224px CLS. Test the separately cached true native-448 global CLS, first with the original nine-class head, then as a supervised residual next to original DINO/CNN features.',
        maximum_epochs=120,minimum_epochs=40,patience=20,draws_per_epoch=1024,batch_images=64,residual_learning_rate=.001,base_learning_rate=.0001,weight_decay=.01,seed=2151,
        calibration=calibration,admission_probability=.95,target_replay_mass=.5,legacy_healthy_and_named_disease_preserved=True,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False,
        stopping='One unchanged native class-head baseline and two declared cached residual heads. No cutoff/encoder grid. Full cached population checks only after >=28/40 Healthy and zero new source false Healthy; no export.')
    freeze(out/'protocol.json',protocol)
    sampling=torch.from_numpy(target_replay_weights(rows,train,'olid_tomato'));vw=torch.tensor(source_class_group_weights(rows,val),device='cuda')
    def decision(probability,loss):
        errors={};healthy=selection['fixed_healthy_validation_correct'];old_h=0
        for i,value in zip(val,probability):
            row=rows[i];original,packet,retained=items[row['path']];now=recover_only(original,retained,packet,float(value),data['manifest'])['app_replay']['kind']=='HEALTHY'
            if not row['y']:errors[row['source']]=errors.get(row['source'],0)+int(now)
            elif row['source']=='olid_tomato':healthy+=int(now)
            else:old_h+=int(now)
        return (-sum(errors.values()),healthy,old_h,-loss),errors
    def aggregate(raw,ids):return (raw[table[ids]]*mask[ids]).sum(-1)/counts[ids]
    original=original_dino(p).cuda().eval()
    with torch.inference_mode():
        raw=healthy_margin(original.head(original.fc_norm(torch.from_numpy(np.load(source/'native_cls.npy')).cuda())));z=aggregate(raw,val);score,errors=decision(z.min(1).values.sigmoid().cpu().numpy(),0.)
    controls=[dict(method='original_native_class_head',best_decision=score,new_false_healthy_by_source=errors,cheap_pass=score[0]==0 and score[1]>=28,passed=False)]
    np.savez(out/'original_native_class_predictions.npz',indices=val,probability=z.min(1).values.sigmoid().cpu().numpy());del original
    for kind in ('linear','mlp'):
        folder=out/kind;folder.mkdir(exist_ok=True);torch.manual_seed(2151);model=NativeGlobalHead(kind,torch.load(warm/'best.pt',weights_only=True,map_location='cpu')['model'],calibration).cuda()
        optimizer=torch.optim.AdamW([dict(params=model.base.parameters(),lr=.0001),dict(params=model.residual.parameters(),lr=.001)],weight_decay=.01);history=[];best=None;stale=0;start=0
        if (folder/'last.pt').exists():
            saved=torch.load(folder/'last.pt',weights_only=True,map_location='cpu');assert saved['protocol_sha256']==digest(out/'protocol.json');model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);history,best,stale,start=saved['history'],saved['best'],saved['stale'],saved['epoch']+1
        def forward(ids):
            valid=mask[ids]>0;unique,inverse=table[ids][valid].unique(return_inverse=True);raw=features.new_zeros((len(ids),2,6));raw[valid]=model(features[unique])[inverse]
            return (raw*mask[ids]).sum(-1)/counts[ids]
        for epoch in range(start,120):
            if epoch>=40 and stale>=20:break
            model.train();order=torch.multinomial(sampling,1024,replacement=True,generator=torch.Generator().manual_seed(215100+epoch))
            for part in order.split(64):
                ids=train[part.numpy()];z=forward(ids);raw=torch.where(truth[ids]==1,z.min(1).values,z.max(1).values);loss=F.binary_cross_entropy_with_logits(raw,truth[ids]);assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            model.eval()
            with torch.inference_mode():
                z=torch.cat([forward(part) for part in np.array_split(val,max(1,(len(val)+63)//64))]);raw=torch.where(truth[val]==1,z.min(1).values,z.max(1).values);loss=float((F.binary_cross_entropy_with_logits(raw,truth[val],reduction='none')*vw).sum());score,errors=decision(z.min(1).values.sigmoid().cpu().numpy(),loss)
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if best is None or score>tuple(best):
                best=score;stale=0;atomic_save(dict(model=state,epoch=epoch,protocol_sha256=digest(out/'protocol.json')),folder/'best.pt');np.savez(folder/'validation_predictions.npz',indices=val,probability=z.min(1).values.sigmoid().cpu().numpy())
            else:stale+=1
            history.append(dict(epoch=epoch,decision=score,new_false_healthy_by_source=errors));atomic_save(dict(model=state,optimizer=optimizer.state_dict(),rng=capture_rng(),epoch=epoch,history=history,best=best,stale=stale,protocol_sha256=digest(out/'protocol.json')),folder/'last.pt')
        controls.append(dict(method=kind,best_decision=best,selected_epoch=torch.load(folder/'best.pt',weights_only=True,map_location='cpu')['epoch'],checkpoint_sha256=digest(folder/'best.pt'),cheap_pass=best[0]==0 and best[1]>=28,passed=False,history=history))
        print(json.dumps({k:v for k,v in controls[-1].items() if k!='history'}),flush=True)
    result=dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),controls=controls,new_encoder_inputs=0,model_promoted=False,ad_independent=False,olid_independent=False,acceptance_data_used=False,elapsed_seconds=time.monotonic()-started)
    if any(r['cheap_pass'] for r in controls):result['next_action']='Freeze the cheap-passing native head and run every unchanged complete-population gate, extending only missing native input identities before restored verification'
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(controls=[{k:v for k,v in r.items() if k!='history'} for r in controls])),flush=True)


if __name__=='__main__':main()
