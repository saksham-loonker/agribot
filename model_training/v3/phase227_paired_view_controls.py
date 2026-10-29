"""Conditional cross-crop comparison; preserve all guard and two-view criteria."""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.special import expit
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from cached_rbf import training_gamma,restored_margin
from paired_view_features import paired_view_features
from phase215_shrinkage_density import inputs
from opened_context_data import load,read
from phase85_population_sft import evaluation_data
from conditional_recovery_policy import recover_only
from permissive_encoder import digest
from population_siglip_tail import freeze

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);p=Path(parser.parse_args().campaign);out=Path('/dev/shm/agribot-phase227-paired');out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    rows,train,val,x,v,w,keys,parents=inputs(p,'native');x=paired_view_features(x);v=paired_view_features(v);prior_path=p/'phase215_shrinkage_density/native/protocol.json';prior=read(prior_path);assert train==prior['train_indices'] and val==prior['validation_indices'] and np.array_equal(w,prior['training_weights']);folds=[(np.array(f['fit']),np.array(f['test'])) for f in prior['calibration_folds']];y=np.array([rows[i]['y'] for i in train]);methods=['conditional_linear','conditional_rbf']
    protocol=dict(source_sha256=digest(__file__),basis_source_sha256=digest(Path(__file__).with_name('paired_view_features.py')),rbf_source_sha256=digest(Path(__file__).with_name('cached_rbf.py')),parent_protocol_sha256=digest(prior_path),parents=parents,train_indices=train,validation_indices=val,training_weights=w.tolist(),calibration_folds=prior['calibration_folds'],cv_groups=prior['cv_groups'],methods=methods,C=1.,gamma='Training-photo median inside each fit fold separately; final full training fit separately',input='Per-view own,other,absolute difference and product, four unit-normalized blocks. The same model evaluates both view orderings; minimum calibrated view probability still required.',calibration='Grouped training-only OOF Platt C1. All source93/family/shared-input groups and views stay together.',hypothesis='Target Healthy crop discrepancy exceeds disease median; existing independent-view heads cannot condition a crop decision on its counterpart. Test fixed explicit paired interactions, not another encoder or threshold.',healthy_admission=.95,original_gates_unchanged=True,existing_healthy_and_named_disease_exact=True,new_encoder_inputs=0,acceptance_data_used=False,ad_independent=False,olid_independent=False,stopping='Exactly two C1 conditional controls on native cache. No feature/C/gamma grid. Require >=28/40 Healthy and zero new falseHealthy before all full original gates, otherwise stop.')
    freeze(out/'protocol.json',protocol);data=load(p);entries,_,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({r['path']:(r,packets[r['id']],retained) for r,retained in zip(before,previous)});selection=read(p/'phase118_conditional_recovery/protocol.json');controls=[]
    with threadpool_limits(limits=4):
        for method in methods:
            def fit(ids):
                gamma=training_gamma(x[ids]) if method.endswith('rbf') else None;model=SVC(C=1.,gamma=gamma,cache_size=256) if gamma is not None else LogisticRegression(C=1.,max_iter=2000);model.fit(x[ids].reshape(-1,x.shape[-1]),np.repeat(y[ids],2),sample_weight=np.repeat(w[ids]/2,2));return model,gamma
            oof=np.full((len(train),2),np.nan);gammas=[]
            for n,(fit_ids,test) in enumerate(folds):
                model,gamma=fit(fit_ids);oof[test]=model.decision_function(x[test].reshape(-1,x.shape[-1])).reshape(len(test),2);gammas.append(gamma);print(json.dumps(dict(method=method,fold=n)),flush=True)
            assert np.isfinite(oof).all();cal=LogisticRegression(C=1.,max_iter=2000).fit(oof.reshape(-1,1),np.repeat(y,2),sample_weight=np.repeat(w/2,2));a=float(cal.coef_[0,0]);b=float(cal.intercept_[0]);assert a>0;model,gamma=fit(np.arange(len(train)));raw=model.decision_function(v.reshape(-1,v.shape[-1])).reshape(len(val),2);prob=expit(a*raw+b).min(1);folder=out/method;folder.mkdir(exist_ok=True)
            state=dict(classes=model.classes_,calibration=np.array([a,b]));state.update(dict(support_vectors=model.support_vectors_,dual_coef=model.dual_coef_,intercept=model.intercept_,gamma=np.array(gamma)) if gamma is not None else dict(coef=model.coef_,intercept=model.intercept_));np.savez_compressed(folder/'model.npz',**state);restored=dict(np.load(folder/'model.npz',allow_pickle=False));replay=restored_margin(v.reshape(-1,v.shape[-1]),restored) if gamma is not None else (v.reshape(-1,v.shape[-1])@restored['coef'].T+restored['intercept']).ravel();replay=replay.reshape(len(val),2);error=float(abs(raw-replay).max());assert error<1e-10 and np.array_equal(expit(a*replay+b).min(1)>=.95,prob>=.95)
            healthy=selection['fixed_healthy_validation_correct'];old_h=0;errors={}
            for i,score in zip(val,prob):
                row=rows[i];original,packet,retained=items[row['path']];now=recover_only(original,retained,packet,float(score),data['manifest'])['app_replay']['kind']=='HEALTHY'
                if not row['y']:errors[row['source']]=errors.get(row['source'],0)+int(now)
                elif row['source']=='olid_tomato':healthy+=int(now)
                else:old_h+=int(now)
            np.savez(folder/'predictions.npz',indices=np.array(val),probability=prob,validation_margin=raw,oof_training_margin=oof,model_sha256=digest(folder/'model.npz'),protocol_sha256=digest(out/'protocol.json'));r=dict(method=method,healthy_validation_correct=healthy,new_false_healthy_by_source=errors,old_healthy_correct=old_h,fold_gammas=gammas,full_training_gamma=gamma,calibration=[a,b],model_sha256=digest(folder/'model.npz'),restore_maximum_margin_error=error,cheap_pass=healthy>=28 and sum(errors.values())==0);controls.append(r);print(json.dumps(r),flush=True)
            if r['cheap_pass']:freeze(folder/'candidate.json',dict(**r,full_qualification_required=True,model_promoted=False))
    result=dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),controls=controls,elapsed_seconds=time.monotonic()-started,model_promoted=False,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False);(out/'result.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
