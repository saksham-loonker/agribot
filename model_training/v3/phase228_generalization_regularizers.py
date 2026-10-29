"""Matched cold-start heads with training-only fold calibration and fixed endpoints."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from source_interpolation import CrossSourcePartners
from leaf_feature_expert import LeafFeatureHead
from hard_negative_ranking import hard_negative_rank_loss
from phase215_shrinkage_density import inputs
from opened_context_data import load,read
from phase85_population_sft import evaluation_data
from conditional_recovery_policy import recover_only
from permissive_encoder import digest
from population_siglip_tail import freeze
from training_support import atomic_save,capture_rng

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);p=Path(parser.parse_args().campaign);out=p/'phase228_generalization_regularizers';out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    rows,train,val,x,v,w,keys,parents=inputs(p,'mobile');prior_path=p/'phase215_shrinkage_density/mobile/protocol.json';prior=read(prior_path);assert train==prior['train_indices'] and val==prior['validation_indices'] and np.array_equal(w,prior['training_weights']);folds=[(np.array(f['fit']),np.array(f['test'])) for f in prior['calibration_folds']];methods=['matched_erm','cross_source_same_label_mixup','spectral_decoupling'];protocol=dict(source_sha256=digest(__file__),partner_source_sha256=digest(Path(__file__).with_name('source_interpolation.py')),parent_protocol_sha256=digest(prior_path),parents=parents,train_indices=train,validation_indices=val,training_weights=w.tolist(),calibration_folds=prior['calibration_folds'],cv_groups=prior['cv_groups'],methods=methods,seed=2280,epochs=40,draws_per_epoch=2048,batch=64,learning_rate=.001,weight_decay=.01,ranking_weight=.2,mixup_weight=.5,mixup_beta=[2.,2.],spectral_logit_square_weight=.01,calibration='Every fold starts from scratch; grouped OOF Platt C1. No warm checkpoint exposure to held-fold labels. Final endpoint always epoch39; no validation snapshot selection.',input='Exact mobile1408 cached photo ROI means, independently unit-normalized384/1024 blocks, actual two views. Interpolation mixes only training feature means from same primary clinical label across different source/family. Not pixel augmentation or per-ROI diagnoses.',hypothesis='Strong training-fit/target-ranking gap suggests capture-style dependence or feature starvation. Test class-conditional cross-source interpolation and logit-square regularization separately against matched supervised ERM, with identical cached inputs and fixed40epoch endpoints.',healthy_admission=.95,original_gates_unchanged=True,existing_healthy_and_named_disease_exact=True,new_encoder_inputs=0,acceptance_data_used=False,ad_independent=False,olid_independent=False,stopping='Exactly three objectives with four grouped cold-start calibration fits and one full training fit each. Forty epochs, no rate/weight/endpoint/threshold grid. >=28/40 Healthy and zero new falseHealthy required before unchanged full qualification.')
    freeze(out/'protocol.json',protocol);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;device='cuda';xx=torch.tensor(x,dtype=torch.float32,device=device);vv=torch.tensor(v,dtype=torch.float32,device=device);y=np.array([rows[i]['y'] for i in train]);yy=torch.tensor(y,dtype=torch.float32,device=device);train_rows=[rows[i] for i in train];data=load(p);entries,_,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({r['path']:(r,packets[r['id']],retained) for r,retained in zip(before,previous)});selection=read(p/'phase118_conditional_recovery/protocol.json');controls=[]
    def predict(model,values):
        model.eval()
        with torch.inference_mode():return model(values.reshape(-1,1408)).reshape(len(values),2).cpu().numpy()
    for method in methods:
        folder=out/method;folder.mkdir(exist_ok=True)
        def fit(ids,name,save_epochs=False):
            dst=folder/name;dst.mkdir(exist_ok=True);torch.manual_seed(2280);model=LeafFeatureHead('mlp',1408).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);sampler=torch.tensor(w[ids],dtype=torch.float64);sampler/=sampler.sum();partners=CrossSourcePartners(train_rows,ids);history=[];mixable=0;draws=0
            if (dst/'complete.json').exists():
                c=read(dst/'complete.json');assert c['protocol_sha256']==digest(out/'protocol.json') and c['model_sha256']==digest(dst/'last.pt');model.load_state_dict(torch.load(dst/'last.pt',weights_only=True,map_location=device)['model']);return model
            for epoch in range(40):
                model.train();order=torch.multinomial(sampler,2048,replacement=True,generator=torch.Generator().manual_seed(228000+epoch));rng=np.random.default_rng(228000+epoch);losses=[]
                for batch in order.split(64):
                    selected=ids[batch.numpy()];labels=yy[selected];logits=model(xx[selected].reshape(-1,1408)).reshape(len(selected),2);raw=torch.where(labels==1,logits.min(1).values,logits.max(1).values);loss=F.binary_cross_entropy_with_logits(raw,labels)+.2*hard_negative_rank_loss(torch.stack([logits.min(1).values,logits.max(1).values],1),labels)
                    if method=='spectral_decoupling':loss=loss+.01*logits.square().mean()
                    if method=='cross_source_same_label_mixup':
                        other,valid=partners.sample(selected,rng);mixable+=int(valid.sum());draws+=len(selected)
                        if valid.any():
                            own=selected[valid];other=other[valid];lam=torch.tensor(rng.beta(2.,2.,size=len(own)),dtype=torch.float32,device=device)[:,None,None];mixed=lam*xx[own]+(1-lam)*xx[other];z=model(mixed.reshape(-1,1408)).reshape(len(own),2);truth=yy[own];worst=torch.where(truth==1,z.min(1).values,z.max(1).values);loss=loss+.5*F.binary_cross_entropy_with_logits(worst,truth)
                    assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
                history.append(dict(epoch=epoch,training_loss=float(np.mean(losses))))
                if save_epochs:atomic_save(dict(model={k:z.detach().cpu().clone() for k,z in model.state_dict().items()},epoch=epoch,protocol_sha256=digest(out/'protocol.json')),dst/f'epoch_{epoch:03d}.pt')
            atomic_save(dict(model={k:z.detach().cpu().clone() for k,z in model.state_dict().items()},optimizer=optimizer.state_dict(),rng=capture_rng(),epoch=39,protocol_sha256=digest(out/'protocol.json')),dst/'last.pt');(dst/'complete.json').write_text(json.dumps(dict(protocol_sha256=digest(out/'protocol.json'),model_sha256=digest(dst/'last.pt'),history=history,valid_mixup_draws=mixable,total_mixup_draws=draws),indent=2));print(json.dumps(dict(method=method,fit=name,final_training_loss=history[-1]['training_loss'])),flush=True);return model
        oof=np.full((len(train),2),np.nan)
        for n,(fit_ids,test) in enumerate(folds):oof[test]=predict(fit(fit_ids,f'fold_{n}'),xx[test])
        assert np.isfinite(oof).all();cal=LogisticRegression(C=1.,max_iter=2000).fit(oof.reshape(-1,1),np.repeat(y,2),sample_weight=np.repeat(w/2,2));a=float(cal.coef_[0,0]);b=float(cal.intercept_[0]);assert a>0;model=fit(np.arange(len(train)),'full',True);raw=predict(model,vv);prob=expit(a*raw+b).min(1);healthy=selection['fixed_healthy_validation_correct'];old_h=0;errors={}
        for i,score in zip(val,prob):
            row=rows[i];original,packet,retained=items[row['path']];now=recover_only(original,retained,packet,float(score),data['manifest'])['app_replay']['kind']=='HEALTHY'
            if not row['y']:errors[row['source']]=errors.get(row['source'],0)+int(now)
            elif row['source']=='olid_tomato':healthy+=int(now)
            else:old_h+=int(now)
        restored=LeafFeatureHead('mlp',1408).to(device);restored.load_state_dict(torch.load(folder/'full/last.pt',weights_only=True,map_location=device)['model']);replay=predict(restored,vv);error=float(abs(raw-replay).max());assert error<1e-6 and np.array_equal(expit(a*replay+b).min(1)>=.95,prob>=.95);np.savez(folder/'predictions.npz',indices=np.array(val),probability=prob,validation_margin=raw,oof_training_margin=oof,calibration=np.array([a,b]),model_sha256=digest(folder/'full/last.pt'),protocol_sha256=digest(out/'protocol.json'));r=dict(method=method,healthy_validation_correct=healthy,new_false_healthy_by_source=errors,old_healthy_correct=old_h,calibration=[a,b],model_sha256=digest(folder/'full/last.pt'),restored_head_max_error=error,cheap_pass=healthy>=28 and sum(errors.values())==0);controls.append(r);print(json.dumps(r),flush=True)
        if r['cheap_pass']:freeze(folder/'candidate.json',dict(**r,full_qualification_required=True,model_promoted=False))
    (out/'result.json').write_text(json.dumps(dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),controls=controls,elapsed_seconds=time.monotonic()-started,model_promoted=False,new_encoder_inputs=0,acceptance_data_used=False,ad_independent=False,olid_independent=False),indent=2))

if __name__=='__main__':main()
