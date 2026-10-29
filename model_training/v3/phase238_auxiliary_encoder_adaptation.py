"""Matched frozen/LoRA/full-suffix cross-crop supervision from exact prefix caches."""
from pathlib import Path
import argparse,json,shutil,time
import numpy as np,torch
from torch.nn import functional as F
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from auxiliary_suffix_expert import AuxiliarySuffixExpert
from partial_dino_cache import original_dino
from clinical_replay_balance import clinical_replay_weights
from hard_negative_ranking import hard_negative_rank_loss
from large_photometric_dino import trainable_state,restore_trainable
from opened_context_data import load,read
from phase85_population_sft import evaluation_data
from conditional_recovery_policy import recover_only
from permissive_encoder import digest
from population_siglip_tail import freeze
from training_support import atomic_save,capture_rng,restore_rng

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);p=Path(ap.parse_args().campaign);out=p/'phase238_auxiliary_encoder_adaptation';out.mkdir(exist_ok=True);started=time.monotonic();cache=p/'phase237_auxiliary_prefix_cache';cp=read(cache/'protocol.json');cc=read(cache/'complete.json');assert cc['protocol_sha256']==digest(cache/'protocol.json') and cc['tokens_sha256']==digest(cp['storage']);assert all(cc[field]==digest(cache/name) for field,name in [('tables_sha256','tables.npz'),('rows_sha256','rows.json'),('reference_features_sha256','reference_features.npy'),('locations_sha256','prefix_locations.json')]);rows=read(cache/'rows.json');train=cp['train_indices'];val=cp['validation_indices'];aux=cp['auxiliary_indices'];all_train=train+aux;roles=cp['role_indices']+[aux];weights=np.zeros(len(rows))
    for mass,ids in zip((.375,.1875,.1875,.25),roles):weights[ids]=mass*clinical_replay_weights(rows,ids)
    folds=[(np.array(f['fit']+aux),np.array(f['test'])) for f in cp['calibration_folds']];methods=['frozen','lora','full_suffix'];protocol=dict(source_sha256=digest(__file__),expert_source_sha256=digest(Path(__file__).with_name('auxiliary_suffix_expert.py')),cache_complete_sha256=digest(cache/'complete.json'),original_bundle_sha256=cp['original_bundle_sha256'],train_indices=all_train,original_training_indices=train,auxiliary_indices=aux,validation_indices=val,validation_original_indices=cp['validation_original_indices'],training_weights=weights.tolist(),calibration_folds=[dict(fit=a.tolist(),test=b.tolist()) for a,b in folds],methods=methods,epochs=40,draws_per_epoch=1024,batch_images=16,head_learning_rate=.0001,lora_learning_rate=.0001,full_suffix_learning_rate=.00001,weight_decay=.01,ranking_weight=.2,rank=8,alpha=16.,lora_dropout=.05,seed=2380,save_full_snapshots_every_epochs=5,save_optimizer_rng_each_epoch=True,calibration='Original977-photo domain grouped OOF Platt C1 only. Entire auxiliary collection stays on training side of each fit; no auxiliary row enters OOF calibration or validation. Each fit starts from original frozen prefix/suffix and cold head; no held-fold healthy labels in warm state. Final endpoint40epochs, no validation snapshot selection.',supervision='Primary-image labels over real-ROI means per crop. Frozen CNN features reused; original prefix immutable; copied last2 DINO blocks adapted. No per-ROI clinical labels. Native-cohort bounded encoder replay977photos is explicit, not full original replay.',hypothesis='New expert nutrient supervision did not help frozen heads; test whether adapting copied visual suffixes learns transferable stress cues instead of fitting another fixed-embedding boundary. Compare matched frozen, rank8 Q/V LoRA and full final2block supervision.',healthy_admission=.95,original_gates_unchanged=True,existing_healthy_and_named_disease_exact=True,new_source_role='Cucumber auxiliary training only, never eligible as Tomato without unchanged genus/evidence checks',ad_independent=False,olid_independent=False,acceptance_data_used=False,stopping='Exactly3matched parameterizations, four original-domain grouped fits plus full fit each, fixed40epochs and rates. No threshold/rank/rate/epoch grid. >=28/40 Healthy and zero new falseHealthy before full original qualification. No export on failure.')
    freeze(out/'protocol.json',protocol)
    if (out/'result.json').exists():assert read(out/'result.json')['protocol_sha256']==digest(out/'protocol.json');return
    assert shutil.disk_usage(out).free>(700<<20),'Preserve checkpoints and disk reserve';torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;reference=torch.tensor(np.load(cache/'reference_features.npy'),device='cuda');tables=np.load(cache/'tables.npz');index=torch.tensor(tables['indices'],device='cuda');mask=torch.tensor(tables['mask'],device='cuda');truth=torch.tensor([r['y'] for r in rows],dtype=torch.float32,device='cuda');locations=read(cache/'prefix_locations.json');old=np.load(p/'phase51_dino_adaptation/prefix/features.npy',mmap_mode='r');new=np.load(cp['storage'],mmap_mode='r');tokens=torch.empty((len(locations),257,384),dtype=torch.float32,device='cuda')
    for start in range(0,len(locations),64):tokens[start:start+64]=torch.from_numpy(np.stack([old[r['index']] if r['bank']=='original' else new[r['index']] for r in locations[start:start+64]])).cuda()
    backbone=original_dino(p).eval();data=load(p);entries,_,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({r['path']:(r,packets[r['id']],retained) for r,retained in zip(before,previous)});selection=read(p/'phase118_conditional_recovery/protocol.json');controls=[]
    def predict(model,ids):
        model.eval()
        with torch.inference_mode():return np.concatenate([model(tokens,reference,index[part],mask[part]).cpu().numpy() for part in np.array_split(ids,max(1,(len(ids)+15)//16))])
    for mode in methods:
        folder=out/mode;folder.mkdir(exist_ok=True)
        def fit(ids,name,save_epochs=False):
            dst=folder/name;dst.mkdir(exist_ok=True);torch.manual_seed(2380);model=AuxiliarySuffixExpert(backbone,mode).cuda();head_params=list(model.head.parameters());suffix_params=[v for k,v in model.named_parameters() if v.requires_grad and not k.startswith('head.')];groups=[dict(params=head_params,lr=.0001)]
            if suffix_params:groups.append(dict(params=suffix_params,lr=.0001 if mode=='lora' else .00001))
            optimizer=torch.optim.AdamW(groups,weight_decay=.01);start=0;history=[]
            if (dst/'last.pt').exists():
                saved=torch.load(dst/'last.pt',weights_only=True,map_location='cpu');assert saved['protocol_sha256']==digest(out/'protocol.json');restore_trainable(model,saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);start=saved['epoch']+1;history=saved['history']
            if start==0:
                # Zero adapters/copies must reconstruct the matched frozen feature head.
                clone=AuxiliarySuffixExpert(backbone,'frozen').cuda();clone.head.load_state_dict(model.head.state_dict());a=predict(model,np.array(val));b=predict(clone,np.array(val));error=float(abs(a-b).max());assert error<.002;freeze(dst/'initial_parity.json',dict(maximum_logit_error=error,protocol_sha256=digest(out/'protocol.json')));del clone
            sampler=torch.tensor(weights[ids],dtype=torch.float64)
            for epoch in range(start,40):
                model.train();order=torch.multinomial(sampler,1024,replacement=True,generator=torch.Generator().manual_seed(238000+epoch));losses=[]
                for batch in order.split(16):
                    samples=ids[batch.numpy()];y=truth[samples];z=model(tokens,reference,index[samples],mask[samples]);lo=z.min(1).values;hi=z.max(1).values;raw=torch.where(y==1,lo,hi);loss=F.binary_cross_entropy_with_logits(raw,y)+.2*hard_negative_rank_loss(torch.stack([lo,hi],1),y);assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad],1.);optimizer.step();losses.append(float(loss.detach()))
                history.append(dict(epoch=epoch,training_loss=float(np.mean(losses))));state=trainable_state(model)
                if save_epochs and (epoch+1)%5==0:atomic_save(dict(model=state,epoch=epoch,protocol_sha256=digest(out/'protocol.json')),dst/f'epoch_{epoch:03d}.pt')
                atomic_save(dict(model=state,optimizer=optimizer.state_dict(),rng=capture_rng(),epoch=epoch,history=history,protocol_sha256=digest(out/'protocol.json')),dst/'last.pt')
                if (epoch+1)%5==0:print(json.dumps(dict(mode=mode,fit=name,**history[-1])),flush=True)
            (dst/'complete.json').write_text(json.dumps(dict(protocol_sha256=digest(out/'protocol.json'),model_sha256=digest(dst/'last.pt'),history=history),indent=2));return model
        oof=np.full((len(rows),2),np.nan)
        for n,(a,b) in enumerate(folds):oof[b]=predict(fit(a,f'fold_{n}'),b)
        assert np.isfinite(oof[train]).all();cal_weights=weights[train]/weights[train].sum()*len(train);cal=LogisticRegression(C=1.,max_iter=2000).fit(oof[train].reshape(-1,1),np.repeat([rows[i]['y'] for i in train],2),sample_weight=np.repeat(cal_weights/2,2));a=float(cal.coef_[0,0]);b=float(cal.intercept_[0])
        if a<=0:controls.append(dict(mode=mode,rejected='Non-positive OOF Healthy calibration slope',cheap_pass=False));continue
        model=fit(np.array(all_train),'full',True);raw=predict(model,np.array(val)).astype(np.float64);prob=expit(a*raw+b).min(1);healthy=selection['fixed_healthy_validation_correct'];old_h=0;errors={}
        for i,score in zip(val,prob):
            row=rows[i];original,packet,retained=items[row['path']];now=recover_only(original,retained,packet,float(score),data['manifest'])['app_replay']['kind']=='HEALTHY'
            if not row['y']:errors[row['source']]=errors.get(row['source'],0)+int(now)
            elif row['source']=='olid_tomato':healthy+=int(now)
            else:old_h+=int(now)
        np.savez(folder/'predictions.npz',indices=np.array(cp['validation_original_indices']),probability=prob,validation_margin=raw,oof_training_margin=oof,calibration=np.array([a,b]),model_sha256=digest(folder/'full/last.pt'),protocol_sha256=digest(out/'protocol.json'));r=dict(mode=mode,healthy_validation_correct=healthy,new_false_healthy_by_source=errors,old_healthy_correct=old_h,calibration=[a,b],model_sha256=digest(folder/'full/last.pt'),cheap_pass=healthy>=28 and sum(errors.values())==0);controls.append(r);print(json.dumps(r),flush=True)
        if r['cheap_pass']:freeze(folder/'candidate.json',dict(**r,full_qualification_required=True,model_promoted=False))
    (out/'result.json').write_text(json.dumps(dict(protocol_sha256=digest(out/'protocol.json'),controls=controls,elapsed_seconds=time.monotonic()-started,model_promoted=False,acceptance_data_used=False,ad_independent=False,olid_independent=False),indent=2))

if __name__=='__main__':main()
