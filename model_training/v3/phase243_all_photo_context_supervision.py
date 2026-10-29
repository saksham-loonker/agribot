"""Use cached whole-photo supervision, including photos without detected ROIs."""
from pathlib import Path
import argparse, collections, json, time
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC, SVC
from threadpoolctl import threadpool_limits
from cached_rbf import restored_margin, training_gamma
from clinical_replay_balance import clinical_replay_weights
from conditional_recovery_policy import recover_only
from opened_context_data import load, read
from permissive_encoder import digest
from phase85_population_sft import evaluation_data
from population_siglip_tail import freeze


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True)
    p=Path(ap.parse_args().campaign);out=p/'phase243_all_photo_context_supervision';out.mkdir(exist_ok=True)
    if (out/'result.json').exists():
        assert read(out/'result.json')['source_sha256']==digest(__file__);return
    started=time.monotonic();rows=read(p/'phase84_population_head/fitting_rows.json')
    whole={};identities={};parents={}
    for name in ('opened_siglip_feature_cache','phase81_consumed_olid_features'):
        root=p/name;complete=read(root/'complete.json')
        assert complete['features_sha256']==digest(root/'features.npy')
        assert complete['inputs_sha256']==digest(root/'inputs.json')
        assert complete['protocol_sha256']==digest(root/'protocol.json')
        values=np.load(root/'features.npy',mmap_mode='r');descriptors=read(root/'inputs.json')
        assert len(values)==len(descriptors)
        for r,v in zip(descriptors,values):
            whole[r['path']]=v;identities[r['path']]=r
        parents[name]=digest(root/'complete.json')
    validation=[r for r in rows if r['validation']]
    families={r['group'] for r in validation}
    files={identities[r['path']]['file_sha256'] for r in validation if r['path'] in identities}
    pixels={identities[r['path']]['input_sha256'] for r in validation if r['path'] in identities}
    excluded=set(read(p/'phase127_clinical_anchors/protocol.json')['quarantined_overlap_groups'])
    # Any member sharing a whole-photo identity quarantines the entire family.
    for r in rows:
        d=identities.get(r['path'])
        if d and (d['file_sha256'] in files or d['input_sha256'] in pixels):excluded.add(r['group'])
    train=[i for i,r in enumerate(rows) if not r['validation'] and r['path'] in whole and r['group'] not in families|excluded]
    val=read(p/'phase118_conditional_recovery/protocol.json')['validation_indices']
    assert all(rows[i]['path'] in whole for i in val)
    original=set(read(p/'phase127_clinical_anchors/protocol.json')['train_indices'])
    added=[i for i in train if i not in original and rows[i]['source']=='olid_tomato']
    assert sum(rows[i]['label']=='Healthy' for i in added)==20
    def photos(ids):
        x=np.asarray([whole[rows[i]['path']] for i in ids],dtype=np.float64)
        return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
    x=photos(train);v=photos(val);y=np.array([rows[i]['y'] for i in train])
    parent=list(range(len(train)));owner={}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for j,i in enumerate(train):
        for key in [('family',rows[i]['group']),('input',identities[rows[i]['path']]['input_sha256'])]:
            if key in owner:parent[find(j)]=find(owner[key])
            else:owner[key]=j
    groups=np.array([find(i) for i in range(len(train))])
    folds=list(StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=2430).split(x,y,groups))
    protocol=dict(source_sha256=digest(__file__),parents=parents,train_indices=train,validation_indices=val,
        added_olid_indices=added,added_olid_labels=dict(collections.Counter(rows[i]['label'] for i in added)),
        calibration_folds=[dict(fit=a.tolist(),test=b.tolist()) for a,b in folds],
        methods=['linear','rbf'],C=1.,calibration='Training-only grouped OOF Platt C1; family and exact whole-photo inputs joined. Per-fit clinical/source/family weights and training-only RBF bandwidth.',
        hypothesis='Earlier full-field context/ROI fusion still required detected ROIs. Include genuine whole-photo labels from no-ROI training images, without inventing ROI labels or changing runtime guards.',
        preprocessing='Exact original frozen whole SigLIP cached features, unit norm; no new model inference',
        healthy_admission=.95,original_gates_unchanged=True,existing_healthy_and_named_disease_exact=True,
        new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False,
        stopping='Exactly one linear and one RBF C1 control; no hyperparameter/threshold sweep. Require28/40 Healthy and zero new falseHealthy before full qualification. No export on failure.')
    freeze(out/'protocol.json',protocol);data=load(p);entries,_,packets,before,previous=evaluation_data(p,data)
    items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group}
    items.update({r['path']:(r,packets[r['id']],old) for r,old in zip(before,previous)})
    controls=[]
    with threadpool_limits(limits=4):
        for method in protocol['methods']:
            folder=out/method;folder.mkdir(exist_ok=True)
            def fit(ids):
                w=clinical_replay_weights(rows,[train[j] for j in ids])*len(ids)
                if method=='linear':model=LinearSVC(C=1.,dual='auto',max_iter=10000,random_state=2430)
                else:
                    # Bounded deterministic bandwidth sample; fit side only.
                    subset=np.random.default_rng(2430).choice(ids,min(len(ids),1024),replace=False)
                    model=SVC(C=1.,kernel='rbf',gamma=training_gamma(x[subset,None,:]),cache_size=256)
                return model.fit(x[ids],y[ids],sample_weight=w)
            oof=np.full(len(train),np.nan)
            for n,(a,b) in enumerate(folds):
                assert not set(groups[a])&set(groups[b]);model=fit(a);oof[b]=model.decision_function(x[b])
                print(json.dumps(dict(method=method,completed_fold=n)),flush=True)
            weights=clinical_replay_weights(rows,train)*len(train)
            cal=LogisticRegression(C=1.,max_iter=2000).fit(oof[:,None],y,sample_weight=weights)
            model=fit(np.arange(len(train)));raw=model.decision_function(v);a=float(cal.coef_[0,0]);b=float(cal.intercept_[0])
            state=dict(classes=model.classes_,calibration=np.array([a,b]))
            if method=='linear':state.update(coef=model.coef_,intercept=model.intercept_)
            else:state.update(support_vectors=model.support_vectors_,dual_coef=model.dual_coef_,intercept=model.intercept_,gamma=np.array(model._gamma))
            np.savez(folder/'model.npz',**state);restored=dict(np.load(folder/'model.npz',allow_pickle=False))
            replay=(v@restored['coef'].ravel()+restored['intercept'][0]) if method=='linear' else restored_margin(v,restored)
            assert np.allclose(raw,replay,rtol=1e-10,atol=1e-10)
            probability=expit(a*raw+b) if a>0 else np.zeros(len(val))
            healthy=13;errors={};old_h=0
            for i,score in zip(val,probability):
                row=rows[i];original,packet,retained=items[row['path']]
                now=recover_only(original,retained,packet,float(score),data['manifest'])['app_replay']['kind']=='HEALTHY'
                if not row['y']:errors[row['source']]=errors.get(row['source'],0)+int(now)
                elif row['source']=='olid_tomato':healthy+=int(now)
                else:old_h+=int(now)
            np.savez(folder/'predictions.npz',indices=val,probability=probability,validation_margin=raw,oof_training_margin=oof)
            control=dict(method=method,healthy_validation_correct=healthy,new_false_healthy_by_source=errors,old_healthy_correct=old_h,
                         calibration_slope=a,cheap_pass=healthy>=28 and sum(errors.values())==0,model_sha256=digest(folder/'model.npz'))
            controls.append(control);print(json.dumps(control),flush=True)
    result=dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),controls=controls,
                elapsed_seconds=time.monotonic()-started,new_encoder_inputs=0,model_promoted=False,acceptance_data_used=False)
    (out/'result.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
