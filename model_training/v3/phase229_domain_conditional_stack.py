"""Training-only source conditioning over diverse cross-fitted model evidence."""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from domain_stack_basis import domain_stack_basis
from phase215_shrinkage_density import inputs
from opened_context_data import load,read
from phase85_population_sft import evaluation_data
from conditional_recovery_policy import recover_only
from permissive_encoder import digest
from population_siglip_tail import freeze

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);p=Path(ap.parse_args().campaign);out=p/'phase229_domain_conditional_stack';out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    rows,tr,val,x,v,w,keys,parents=inputs(p,'native');mr,mt,mv,mx,mm,mw,mkeys,mp=inputs(p,'mobile');assert mv==val;lookup={mr[i]['path']:j for j,i in enumerate(mt)};joined=[]
    for i in tr:
        j=lookup[rows[i]['path']];r=mr[mt[j]];assert (r['y'],r['label'],r['group'])==(rows[i]['y'],rows[i]['label'],rows[i]['group']) and set(mkeys[mt[j]])<=set(keys[i]);joined.append(j)
    prior_path=p/'phase215_shrinkage_density/native/protocol.json';prior=read(prior_path);assert tr==prior['train_indices'] and val==prior['validation_indices'];folds=[(np.array(f['fit']),np.array(f['test'])) for f in prior['calibration_folds']];source=np.array([rows[i]['source']=='olid_tomato' for i in tr],dtype=int);y=np.array([rows[i]['y'] for i in tr]);paths={name:Path('/dev/shm/agribot-phase227-paired')/name/'predictions.npz' for name in ('conditional_linear','conditional_rbf')};paths.update({name:p/'phase228_generalization_regularizers'/name/'predictions.npz' for name in ('matched_erm','cross_source_same_label_mixup','spectral_decoupling')});training=[];validation=[]
    for name,path in paths.items():
        z=np.load(path);assert np.array_equal(z['indices'],val);a=z['oof_training_margin'];assert a.shape==(len(tr) if name.startswith('conditional_') else len(mt),2);training.append(a if name.startswith('conditional_') else a[joined]);validation.append(z['validation_margin'])
    train_scores=np.stack(training,axis=2);val_scores=np.stack(validation,axis=2);methods=['single_rbf_domain','diverse_unconditional','diverse_domain'];protocol=dict(source_sha256=digest(__file__),basis_source_sha256=digest(Path(__file__).with_name('domain_stack_basis.py')),parent_protocol_sha256=digest(prior_path),prediction_sha256={name:digest(path) for name,path in paths.items()},train_indices=tr,validation_indices=val,mobile_training_positions=joined,training_weights=w.tolist(),calibration_folds=prior['calibration_folds'],cv_groups=prior['cv_groups'],expert_names=list(paths),methods=methods,C=1.,domain='Cross-fitted logistic C1 source-membership probe using only native cached photo means and TRAINING source IDs; no validation source IDs used for predictions. Source/style proxy, not camera ground truth.',stack='Weighted logistic C1 trained on OOF base margins, own/counterpart/absolute difference/products, optionally source-posterior interactions. Center/scale from training OOF only. No additional independent meta-calibration claim.',hypothesis='Global OOF calibration may suppress target-source scores. Test a training-inferred domain interaction and complementary paired/mixup/logit-regularized evidence instead of validation-fitted thresholds.',healthy_admission=.95,original_gates_unchanged=True,existing_healthy_and_named_disease_exact=True,new_encoder_inputs=0,acceptance_data_used=False,ad_independent=False,olid_independent=False,stopping='Exactly one domain-conditional paired-RBF calibrator and two matched diverse stacks with/without domain interaction. No C/source/feature/threshold grid; >=28/40 Healthy and zero new falseHealthy before unchanged full qualification.')
    freeze(out/'protocol.json',protocol);domain_oof=np.full(len(tr),np.nan)
    with threadpool_limits(limits=4):
        for fit,test in folds:
            assert set(source[fit])=={0,1};probe=LogisticRegression(C=1.,class_weight='balanced',max_iter=2000).fit(x[fit].mean(1),source[fit]);domain_oof[test]=probe.predict_proba(x[test].mean(1))[:,1]
        assert np.isfinite(domain_oof).all();probe=LogisticRegression(C=1.,class_weight='balanced',max_iter=2000).fit(x.mean(1),source);domain_val=probe.predict_proba(v.mean(1))[:,1];data=load(p);entries,_,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({r['path']:(r,packets[r['id']],retained) for r,retained in zip(before,previous)});selection=read(p/'phase118_conditional_recovery/protocol.json');controls=[]
        for method in methods:
            columns=[1] if method=='single_rbf_domain' else list(range(len(paths)));a=train_scores[:,:,columns];b=val_scores[:,:,columns];weight=np.repeat(w/2,2);flat=a.reshape(-1,len(columns));center=np.average(flat,axis=0,weights=weight);scale=np.sqrt(np.average((flat-center)**2,axis=0,weights=weight)).clip(1e-6);conditional=method!='diverse_unconditional';basis=domain_stack_basis(a,center,scale,domain_oof if conditional else None);q=domain_stack_basis(b,center,scale,domain_val if conditional else None);model=LogisticRegression(C=1.,max_iter=2000).fit(basis.reshape(-1,basis.shape[-1]),np.repeat(y,2),sample_weight=weight);raw=model.decision_function(q.reshape(-1,q.shape[-1])).reshape(len(val),2);prob=expit(raw).min(1);folder=out/method;folder.mkdir(exist_ok=True);np.savez(folder/'model.npz',coef=model.coef_,intercept=model.intercept_,center=center,scale=scale,columns=np.array(columns),domain_coef=probe.coef_,domain_intercept=probe.intercept_,conditional=conditional);state=dict(np.load(folder/'model.npz',allow_pickle=False));restored_domain=expit((v.mean(1)@state['domain_coef'].T+state['domain_intercept']).ravel());restored_basis=domain_stack_basis(b,state['center'],state['scale'],restored_domain if conditional else None);replay=(restored_basis.reshape(-1,q.shape[-1])@state['coef'].T+state['intercept']).reshape(len(val),2);error=float(abs(raw-replay).max());assert error<1e-10 and np.array_equal(expit(replay).min(1)>=.95,prob>=.95);healthy=selection['fixed_healthy_validation_correct'];old_h=0;errors={}
            for i,score in zip(val,prob):
                row=rows[i];original,packet,retained=items[row['path']];now=recover_only(original,retained,packet,float(score),data['manifest'])['app_replay']['kind']=='HEALTHY'
                if not row['y']:errors[row['source']]=errors.get(row['source'],0)+int(now)
                elif row['source']=='olid_tomato':healthy+=int(now)
                else:old_h+=int(now)
            np.savez(folder/'predictions.npz',indices=np.array(val),probability=prob,validation_margin=raw,domain_probability=domain_val,model_sha256=digest(folder/'model.npz'),protocol_sha256=digest(out/'protocol.json'));r=dict(method=method,healthy_validation_correct=healthy,new_false_healthy_by_source=errors,old_healthy_correct=old_h,model_sha256=digest(folder/'model.npz'),restore_max_error=error,cheap_pass=healthy>=28 and sum(errors.values())==0);controls.append(r);print(json.dumps(r),flush=True)
            if r['cheap_pass']:freeze(folder/'candidate.json',dict(**r,full_qualification_required=True,model_promoted=False))
    (out/'result.json').write_text(json.dumps(dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),controls=controls,elapsed_seconds=time.monotonic()-started,model_promoted=False,new_encoder_inputs=0,acceptance_data_used=False,ad_independent=False,olid_independent=False),indent=2))

if __name__=='__main__':main()
