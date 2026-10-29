"""Full-feature supervised discriminants: no unsupervised PCA truncation."""
from pathlib import Path
import argparse,json,shutil,time
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from conditional_recovery_policy import recover_only
from full_fisher import fit_fisher,fisher_margin
from opened_context_data import load,read
from phase215_shrinkage_density import inputs
from phase85_population_sft import evaluation_data
from permissive_encoder import digest
from population_siglip_tail import freeze

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);parser.add_argument('--features',choices=['mobile','native'],required=True);args=parser.parse_args();p=Path(args.campaign);base=Path('/dev/shm/agribot-phase220-fisher');base.mkdir(exist_ok=True);link=p/'phase220_full_fisher'
    if not link.exists():link.symlink_to(base,target_is_directory=True)
    assert link.resolve()==base and shutil.disk_usage(base).free>16<<20
    out=base/args.features;out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    rows,train,val,x,v,w,keys,parents=inputs(p,args.features);previous=p/'phase215_shrinkage_density'/args.features/'protocol.json';prior=read(previous);assert train==prior['train_indices'] and val==prior['validation_indices'] and np.array_equal(w,prior['training_weights']);folds=[(np.array(f['fit']),np.array(f['test'])) for f in prior['calibration_folds']];parent=list(range(len(train)));owner={}
    def find(j):
        while parent[j]!=j:parent[j]=parent[parent[j]];j=parent[j]
        return j
    for j,i in enumerate(train):
        for key in [('family',rows[i]['group'])]+[('input',sha) for sha in keys[i]]:
            if key in owner:parent[find(j)]=find(owner[key])
            else:owner[key]=j
    families=np.array([find(j) for j in range(len(train))]);labels=np.array([rows[i]['label'] for i in train]);y=np.array([rows[i]['y'] for i in train])
    for fit,test in folds:assert not set(families[fit])&set(families[test])
    protocol=dict(source_sha256=digest(__file__),fisher_source_sha256=digest(Path(__file__).with_name('full_fisher.py')),input_source_sha256=digest(Path(__file__).with_name('phase215_shrinkage_density.py')),parent_protocol_sha256=digest(previous),features=args.features,parents=parents,train_indices=train,validation_indices=val,training_weights=w.tolist(),family_ids=families.tolist(),calibration_folds=prior['calibration_folds'],cv_groups=prior['cv_groups'],methods=['binary','clinical_mixture'],input_dimensions=x.shape[-1],projection='No PCA truncation; all original cached feature dimensions retained',shrinkage='OAS-style trace formula with effective number of inferred families 1/sum(family_mass^2), plus1e-6 trace-scaled numerical floor. Weighted/family version is a declared heuristic, not a claimed OAS statistical guarantee.',supervision='Primary-image clinical labels on actual ROI means per original crop; no per-ROI diagnoses',calibration=prior['calibration'],healthy_admission=.95,original_gates_unchanged=True,existing_healthy_and_named_disease_exact=True,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False,output_volatile=True,hypothesis='PCA32 may discard low-variance clinical directions. Fit supervised full-feature binary and clinical-mixture discriminants with training-only analytic covariance regularization, rather than expanding the failed unsupervised PCA dimension grid.',stopping='Exactly binary and clinical-mixture learners on mobile/native cached banks, same existing training folds, no covariance/threshold grid. Only >=28/40 Healthy and zero new false Healthy permits full original qualification; otherwise stop this family.')
    freeze(out/'protocol.json',protocol);oof={method:np.full((len(train),2),np.nan) for method in protocol['methods']}
    with threadpool_limits(limits=4):
        for n,(fit,test) in enumerate(folds):
            for method in protocol['methods']:
                state=fit_fisher(x[fit].reshape(-1,x.shape[-1]),np.repeat(labels[fit],2),np.repeat(w[fit]/2,2),np.repeat(families[fit],2),method);oof[method][test]=fisher_margin(x[test].reshape(-1,x.shape[-1]),state).reshape(len(test),2)
                print(json.dumps(dict(stage='training_only_fold',features=args.features,fold=n,method=method,shrinkage=float(state['shrinkage']))),flush=True)
        data=load(p);entries,_,packets,before,previous_=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({r['path']:(r,packets[r['id']],retained) for r,retained in zip(before,previous_)});selection=read(p/'phase118_conditional_recovery/protocol.json');controls=[]
        for method in protocol['methods']:
            assert np.isfinite(oof[method]).all();cal=LogisticRegression(C=1.,max_iter=2000).fit(oof[method].reshape(-1,1),np.repeat(y,2),sample_weight=np.repeat(w/2,2));a=float(cal.coef_[0,0]);b=float(cal.intercept_[0]);folder=out/method;folder.mkdir(exist_ok=True)
            if a<=0:controls.append(dict(method=method,cheap_pass=False,rejected='Non-positive training-only OOF calibration slope'));continue
            state=fit_fisher(x.reshape(-1,x.shape[-1]),np.repeat(labels,2),np.repeat(w/2,2),np.repeat(families,2),method);raw=fisher_margin(v.reshape(-1,v.shape[-1]),state).reshape(len(val),2);probability=expit(a*raw+b).min(1);model=folder/'model.npz';np.savez(model,**state,calibration=np.array([a,b]));restored=dict(np.load(model,allow_pickle=False));replay=fisher_margin(v.reshape(-1,v.shape[-1]),restored).reshape(len(val),2);error=float(np.abs(raw-replay).max());assert error<1e-10 and np.array_equal(expit(a*replay+b).min(1)>=.95,probability>=.95)
            healthy=selection['fixed_healthy_validation_correct'];errors={};old_h=0
            for i,score in zip(val,probability):
                row=rows[i];original,packet,retained=items[row['path']];now=recover_only(original,retained,packet,float(score),data['manifest'])['app_replay']['kind']=='HEALTHY'
                if not row['y']:errors[row['source']]=errors.get(row['source'],0)+int(now)
                elif row['source']=='olid_tomato':healthy+=int(now)
                else:old_h+=int(now)
            np.savez(folder/'predictions.npz',indices=np.array(val),probability=probability,validation_margin=raw,oof_training_margin=oof[method],model_sha256=digest(model),protocol_sha256=digest(out/'protocol.json'))
            r=dict(method=method,healthy_validation_correct=healthy,new_false_healthy_by_source=errors,old_healthy_correct=old_h,shrinkage=float(state['shrinkage']),effective_families=float(state['effective_families']),model_sha256=digest(model),restore_maximum_margin_error=error,cheap_pass=healthy>=28 and sum(errors.values())==0);controls.append(r);print(json.dumps(r),flush=True)
            if r['cheap_pass']:freeze(folder/'candidate.json',dict(**r,protocol_sha256=digest(out/'protocol.json'),full_qualification_required=True,model_promoted=False))
    result=dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),features=args.features,controls=controls,new_encoder_inputs=0,model_promoted=False,ad_independent=False,olid_independent=False,acceptance_data_used=False,output_volatile=True,elapsed_seconds=time.monotonic()-started);(out/'result.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
