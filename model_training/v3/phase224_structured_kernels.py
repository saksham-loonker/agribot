"""Two fixed source-calibrated spatial-colour kernels; unchanged app gates."""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.special import expit
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from cached_rbf import training_gamma,restored_margin
from phase215_shrinkage_density import inputs
from opened_context_data import load,read
from phase85_population_sft import evaluation_data
from conditional_recovery_policy import recover_only
from permissive_encoder import digest
from population_siglip_tail import freeze

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);p=Path(parser.parse_args().campaign);started=time.monotonic();out=Path('/dev/shm/agribot-phase224-structured-kernels');out.mkdir(exist_ok=True);link=p/'phase224_structured_kernels'
    if not link.exists():link.symlink_to(out,target_is_directory=True)
    assert link.resolve()==out
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    rows,train,val,native_x,native_v,w,keys,parents=inputs(p,'native');prior=read(p/'phase215_shrinkage_density/native/protocol.json');assert train==prior['train_indices'] and val==prior['validation_indices'] and np.array_equal(w,prior['training_weights']);folds=[(np.array(f['fit']),np.array(f['test'])) for f in prior['calibration_folds']];native=read(p/'phase189_full_field_native_extension/protocol.json');cache=p/'phase223_structured_leaf_cache';complete=read(cache/'complete.json');assert complete['protocol_sha256']==digest(cache/'protocol.json') and complete['features_sha256']==digest(cache/'features.npy');values=np.load(cache/'features.npy');assert values.shape==(len(native['links']),144)
    def photos(ids):
        views=[];quality=[]
        for i in ids:
            v=[];ok=True
            for bag in native['bags'][rows[i]['path']]:
                blocks=[values[bag,:75].mean(0),values[bag,75:].mean(0)];v.append(np.concatenate([b/max(float(np.linalg.norm(b)),1e-8) for b in blocks])/2**.5);ok=ok and bool((values[bag,-3]>0).all())
            views.append(v);quality.append(ok)
        return np.asarray(views),np.array(quality)
    descriptor_x,train_quality=photos(train);descriptor_v,val_quality=photos(val);y=np.array([rows[i]['y'] for i in train]);assert np.isfinite(descriptor_x).all()
    protocol=dict(source_sha256=digest(__file__),rbf_source_sha256=digest(Path(__file__).with_name('cached_rbf.py')),descriptor_complete_sha256=digest(cache/'complete.json'),parent_protocol_sha256=digest(p/'phase215_shrinkage_density/native/protocol.json'),parents=parents,train_indices=train,validation_indices=val,training_weights=w.tolist(),calibration_folds=prior['calibration_folds'],cv_groups=prior['cv_groups'],methods=['structured_only','native_plus_structured'],C=1.,gamma='Inverse median squared photo-mean training distance, computed separately inside each calibration fit fold and final full training fit.',calibration='Training-only grouped OOF weighted Platt C1. Whole publisher93, families and exact input identities held together.',healthy_admission=.95,quality='Invalid heuristic foreground in any actual ROI prevents new Healthy admission. No rows removed from training or validation denominators.',invalid_training_photos=int((~train_quality).sum()),invalid_validation_photos=int((~val_quality).sum()),hypothesis='Fixed ridge/margin colour contrasts may recover deficiency-related morphology absent from pooled embeddings. Compare descriptor alone and equal four-block joint representation; masks are not clinical annotations.',original_gates_unchanged=True,existing_healthy_and_named_disease_exact=True,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False,stopping='Exactly two C1 controls, no C/gamma/feature/threshold grid. >=28/40 Healthy with zero new false Healthy required before all original full qualification; no export for failures.')
    freeze(out/'protocol.json',protocol);data=load(p);entries,_,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({r['path']:(r,packets[r['id']],retained) for r,retained in zip(before,previous)});selection=read(p/'phase118_conditional_recovery/protocol.json');controls=[]
    with threadpool_limits(limits=4):
        for method in protocol['methods']:
            x=descriptor_x if method=='structured_only' else np.concatenate([native_x,descriptor_x],axis=2)/2**.5;v=descriptor_v if method=='structured_only' else np.concatenate([native_v,descriptor_v],axis=2)/2**.5;oof=np.full((len(train),2),np.nan);gammas=[]
            def fit(ids):
                gamma=training_gamma(x[ids]);return SVC(C=1.,gamma=gamma,cache_size=256).fit(x[ids].reshape(-1,x.shape[-1]),np.repeat(y[ids],2),sample_weight=np.repeat(w[ids]/2,2)),gamma
            for n,(a,b) in enumerate(folds):
                model,gamma=fit(a);gammas.append(gamma);oof[b]=model.decision_function(x[b].reshape(-1,x.shape[-1])).reshape(len(b),2);print(json.dumps(dict(method=method,fold=n,training_gamma=gamma)),flush=True)
            assert np.isfinite(oof).all();cal=LogisticRegression(C=1.,max_iter=2000).fit(oof.reshape(-1,1),np.repeat(y,2),sample_weight=np.repeat(w/2,2));a=float(cal.coef_[0,0]);b=float(cal.intercept_[0]);assert a>0,'Non-positive OOF calibration slope';model,gamma=fit(np.arange(len(train)));raw=model.decision_function(v.reshape(-1,v.shape[-1])).reshape(len(val),2);prob=expit(a*raw+b).min(1);prob[~val_quality]=0.;folder=out/method;folder.mkdir(exist_ok=True);state=dict(support_vectors=model.support_vectors_,dual_coef=model.dual_coef_,intercept=model.intercept_,classes=model.classes_,gamma=np.array(gamma),calibration=np.array([a,b]));np.savez(folder/'model.npz',**state);restored=dict(np.load(folder/'model.npz',allow_pickle=False));replay=restored_margin(v.reshape(-1,v.shape[-1]),restored).reshape(len(val),2);error=float(abs(replay-raw).max());assert error<1e-10 and np.array_equal((expit(a*replay+b).min(1)>=.95)&val_quality,prob>=.95)
            healthy=selection['fixed_healthy_validation_correct'];old_h=0;errors={}
            for i,score in zip(val,prob):
                row=rows[i];original,packet,retained=items[row['path']];now=recover_only(original,retained,packet,float(score),data['manifest'])['app_replay']['kind']=='HEALTHY'
                if not row['y']:errors[row['source']]=errors.get(row['source'],0)+int(now)
                elif row['source']=='olid_tomato':healthy+=int(now)
                else:old_h+=int(now)
            np.savez(folder/'predictions.npz',indices=np.array(val),probability=prob,validation_margin=raw,oof_training_margin=oof,quality=val_quality,model_sha256=digest(folder/'model.npz'),protocol_sha256=digest(out/'protocol.json'));r=dict(method=method,healthy_validation_correct=healthy,new_false_healthy_by_source=errors,old_healthy_correct=old_h,fold_gammas=gammas,full_training_gamma=gamma,calibration=[a,b],model_sha256=digest(folder/'model.npz'),restore_maximum_margin_error=error,cheap_pass=healthy>=28 and sum(errors.values())==0);controls.append(r);print(json.dumps(r),flush=True)
            if r['cheap_pass']:freeze(folder/'candidate.json',dict(**r,full_qualification_required=True,model_promoted=False))
    result=dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),controls=controls,elapsed_seconds=time.monotonic()-started,model_promoted=False,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False);(out/'result.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
