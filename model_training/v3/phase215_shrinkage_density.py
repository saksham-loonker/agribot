"""Four train-calibrated Gaussian-mixture controls, using exact cached features."""
from pathlib import Path
import argparse,json,shutil,time
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from threadpoolctl import threadpool_limits
from clinical_replay_balance import clinical_replay_weights
from conditional_recovery_policy import recover_only
from exhaustive_replay_data import exhaustive_replay_data
from opened_context_data import load,read
from phase85_population_sft import evaluation_data
from permissive_encoder import digest
from population_siglip_tail import freeze
from shrinkage_density import fit_projection,project,fit_density,density_margin

def inputs(p,kind):
    if kind=='mobile':
        values,identities,rows,tables,train,val,meta=exhaustive_replay_data(p)
        bags={i:tables[i][0] for i in train+val};split=384
        keys={i:[identities[k] for view in bags[i] for k in view] for i in train+val}
        parts=[meta['target'],meta['replay'],meta['field_indices']]
        parents=dict(full_replay_protocol_sha256=meta['full_replay_protocol_sha256'],original_parents=meta['original_parents'],field_complete_sha256=digest(p/'phase164_new_field_features/complete.json'))
    else:
        root=p/'phase189_full_field_native_extension';cp=read(root/'protocol.json');cc=read(root/'complete.json')
        assert cc['protocol_sha256']==digest(root/'protocol.json') and cc['features_sha256']==digest(root/'features.npy')
        assert digest(p/'phase168_early_cnn_prefix/rows.json')==cp['rows_sha256']
        rows=read(p/'phase168_early_cnn_prefix/rows.json');values=np.load(root/'features.npy');assert list(values.shape)==cp['shape']
        train=cp['train_indices'];val=cp['validation_indices'];bags={i:cp['bags'][rows[i]['path']] for i in train+val};split=1024
        keys={i:[sha for view in bags[i] for k in view for sha in (cp['links'][k]['input_sha256'],cp['links'][k]['parent_224_input_sha256'])] for i in train+val}
        parts=[cp['target_indices'],cp['anchor_indices'],cp['field_indices']]
        parents=dict(native_complete_sha256=digest(root/'complete.json'),native_protocol_sha256=digest(root/'protocol.json'))
    assert train==sum(parts,[]) and val==read(p/'phase118_conditional_recovery/protocol.json')['validation_indices']
    assert not ({key for i in train for key in keys[i]}&{key for i in val for key in keys[i]})
    assert not {rows[i]['group'] for i in train}&{rows[i]['group'] for i in val}
    def photos(indices):
        result=[]
        for i in indices:
            assert len(bags[i])==2 and all(bags[i])
            views=[]
            for view in bags[i]:
                value=values[view].mean(0).astype(np.float64)
                views.append(np.concatenate([block/max(float(np.linalg.norm(block)),1e-8) for block in (value[:split],value[split:])])/(2**.5))
            result.append(views)
        return np.asarray(result)
    x=photos(train);v=photos(val);weights=np.concatenate([mass*clinical_replay_weights(rows,ids) for mass,ids in zip((.5,.25,.25),parts)])*len(train)
    return rows,train,val,x,v,weights,keys,parents

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);parser.add_argument('--features',choices=['mobile','native'],required=True);args=parser.parse_args()
    p=Path(args.campaign);base=Path('/dev/shm/agribot-phase215-shrinkage-density');base.mkdir(exist_ok=True)
    link=p/'phase215_shrinkage_density'
    if not link.exists():link.symlink_to(base,target_is_directory=True)
    assert link.resolve()==base and shutil.disk_usage(base).free>32<<20
    out=base/args.features;out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    rows,train,val,x,v,w,keys,parents=inputs(p,args.features);labels=np.array([rows[i]['label'] for i in train]);y=np.array([rows[i]['y'] for i in train]);parent=list(range(len(train)));owner={}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for j,i in enumerate(train):
        group_keys=[('family',rows[i]['group'])]+[('input',key) for key in keys[i]]
        if rows[i]['source']=='field93':group_keys.append(('collection','field93'))
        for key in group_keys:
            if key in owner:parent[find(j)]=find(owner[key])
            else:owner[key]=j
    groups=np.array([find(j) for j in range(len(train))]);folds=list(StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=2215).split(x,y,groups))
    for fit,test in folds:assert not set(groups[fit])&set(groups[test]) and set(y[fit])=={0,1}
    protocol=dict(source_sha256=digest(__file__),density_source_sha256=digest(Path(__file__).with_name('shrinkage_density.py')),data_source_sha256=digest(Path(__file__).with_name('exhaustive_replay_data.py')),features=args.features,parents=parents,train_indices=train,validation_indices=val,training_weights=w.tolist(),cv_groups=groups.tolist(),calibration_folds=[dict(fit=a.tolist(),test=b.tolist()) for a,b in folds],methods=['shared','class_shrinkage'],projection_dimension=32,projection='Weighted PCA whitening fitted separately inside each training fold; two crop means each get half a photo weight',components='One Gaussian per primary-image clinical label; Healthy prior .5 and disease mixture prior .5',covariance='Shared pooled within-condition covariance shrunk halfway toward trace-scaled identity; alternate half class covariance and half that shared base; .001 trace-scaled diagonal floor',calibration='Training-only grouped four-fold OOF per-view margins, weighted Platt C=1; all views/families/shared inputs stay together, whole field93 collection held together',hypothesis='Discriminative heads can overfit capture cues. Test regularized generative class geometry, accounting for disease mixture/covariance rather than another margin head. Original and larger/native cached representations are tested separately without new encoding.',healthy_admission=.95,supervision='Primary-image labels on actual ROI means per crop, never individual ROI clinical labels',original_gates_unchanged=True,existing_healthy_and_named_disease_exact=True,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False,output_volatile=True,stopping='Exactly two covariance settings on each declared cached representation; no dimension/shrinkage/threshold grid. Require >=28/40 Healthy and zero new false Healthy before full restored qualification. Stop unsuccessful families and preserve every model/calibration.')
    freeze(out/'protocol.json',protocol);oof={m:np.full((len(train),2),np.nan) for m in protocol['methods']}
    with threadpool_limits(limits=4):
        for n,(fit,test) in enumerate(folds):
            projection=fit_projection(x[fit].reshape(-1,x.shape[-1]),np.repeat(w[fit]/2,2),32)
            z=project(x[fit].reshape(-1,x.shape[-1]),projection);q=project(x[test].reshape(-1,x.shape[-1]),projection)
            for method in protocol['methods']:
                state=fit_density(z,np.repeat(labels[fit],2),np.repeat(w[fit]/2,2),method);oof[method][test]=density_margin(q,state).reshape(len(test),2)
            print(json.dumps(dict(stage='training_only_fold',features=args.features,fold=n)),flush=True)
        projection=fit_projection(x.reshape(-1,x.shape[-1]),np.repeat(w/2,2),32);z=project(x.reshape(-1,x.shape[-1]),projection);q=project(v.reshape(-1,v.shape[-1]),projection)
        data=load(p);entries,_,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({r['path']:(r,packets[r['id']],retained) for r,retained in zip(before,previous)});selection=read(p/'phase118_conditional_recovery/protocol.json');controls=[]
        for method in protocol['methods']:
            assert np.isfinite(oof[method]).all();cal=LogisticRegression(C=1.,max_iter=2000).fit(oof[method].reshape(-1,1),np.repeat(y,2),sample_weight=np.repeat(w/2,2));a=float(cal.coef_[0,0]);b=float(cal.intercept_[0]);folder=out/method;folder.mkdir(exist_ok=True)
            if a<=0:
                controls.append(dict(method=method,rejected='Non-positive training-only OOF calibration slope',cheap_pass=False));continue
            state=fit_density(z,np.repeat(labels,2),np.repeat(w/2,2),method);raw=density_margin(q,state).reshape(len(val),2);probability=expit(a*raw+b).min(1)
            model=folder/'model.npz';np.savez(model,**projection,**state,calibration=np.array([a,b]));restored=dict(np.load(model,allow_pickle=False));restored_raw=density_margin(project(v.reshape(-1,v.shape[-1]),restored),restored).reshape(len(val),2);assert np.array_equal(raw,restored_raw)
            healthy=selection['fixed_healthy_validation_correct'];errors={};old_h=0
            for i,score in zip(val,probability):
                row=rows[i];original,packet,retained=items[row['path']];now=recover_only(original,retained,packet,float(score),data['manifest'])['app_replay']['kind']=='HEALTHY'
                if not row['y']:errors[row['source']]=errors.get(row['source'],0)+int(now)
                elif row['source']=='olid_tomato':healthy+=int(now)
                else:old_h+=int(now)
            np.savez(folder/'predictions.npz',indices=np.array(val),probability=probability,validation_margin=raw,oof_training_margin=oof[method],protocol_sha256=digest(out/'protocol.json'),model_sha256=digest(model))
            control=dict(method=method,healthy_validation_correct=healthy,new_false_healthy_by_source=errors,old_healthy_correct=old_h,model_sha256=digest(model),restored_margin_exact=True,calibration=[a,b],cheap_pass=healthy>=28 and sum(errors.values())==0);controls.append(control);print(json.dumps(control),flush=True)
    result=dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),features=args.features,controls=controls,model_promoted=False,new_encoder_inputs=0,output_volatile=True,ad_independent=False,olid_independent=False,acceptance_data_used=False,elapsed_seconds=time.monotonic()-started);(out/'result.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
