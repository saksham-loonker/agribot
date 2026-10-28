"""Family-calibrated nonlinear recovery on already cached larger-encoder ROIs."""
from pathlib import Path
import argparse,json,time
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import pairwise_distances
from conditional_recovery_policy import recover_only
from opened_context_data import load,read
from phase85_population_sft import evaluation_data
from phase114_new_supervision_features import original_features
from permissive_encoder import digest
from population_siglip_tail import freeze
from target_replay_sampling import target_replay_weights


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase182_full_field_context_kernel';out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    values,bags,descriptors,parent=original_features(p);rows=read(p/'phase84_population_head/fitting_rows.json');selection=read(p/'phase118_conditional_recovery/protocol.json')
    replay=read(p/'phase179_full_field_replay/protocol.json');train=np.array(replay['target_indices']+replay['old_anchor_indices']);val=np.array(selection['validation_indices'])
    assert len(set(train))==len(train) and not any(rows[i]['validation'] for i in train)
    data=load(p)
    from phase81_consumed_olid_features import prepare as prepare_whole
    whole,metadata=prepare_whole(p,data)
    def unit(value):return value/max(float(np.linalg.norm(value)),1e-8)
    def photos(indices):
        return np.array([[np.concatenate([unit(values[bags[rows[i]['path']][view]].mean(0)),unit(whole[rows[i]['path']])])/(2**.5) for view in ('center','full')] for i in indices])
    x=photos(train);v=photos(val);y=np.array([rows[i]['y'] for i in train]);groups=np.array([rows[i]['group'] for i in train]);weights=target_replay_weights(rows,train,'olid_tomato')*len(train)
    assert not set(groups)&{rows[i]['group'] for i in val}
    def identities(indices):return {descriptors[key]['input_sha256'] for i in indices for view in ('center','full') for key in bags[rows[i]['path']][view]}
    assert not identities(train)&identities(val)
    # Calibration components merge families that share even one exact ROI.
    parent_group=list(range(len(train)));owner={}
    def find(j):
        while parent_group[j]!=j:parent_group[j]=parent_group[parent_group[j]];j=parent_group[j]
        return j
    for j,i in enumerate(train):
        keys=[('family',str(groups[j]))]+[('roi',descriptors[key]['input_sha256']) for view in ('center','full') for key in bags[rows[i]['path']][view]]
        for key in keys:
            if key in owner:parent_group[find(j)]=find(owner[key])
            else:owner[key]=j
    groups=np.array([find(j) for j in range(len(train))])
    squared=pairwise_distances(x.mean(1),metric='sqeuclidean');median=float(np.median(squared[np.triu_indices(len(x),1)]));assert median>0;gamma=1/median
    cv=list(StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=2140).split(x,y,groups))
    for fit,test in cv:assert not set(groups[fit])&set(groups[test]) and set(y[fit])=={0,1}
    protocol=dict(source_sha256=digest(__file__),parents=parent,whole_metadata=metadata,coverage_selection_sha256=digest(p/'phase179_full_field_replay/protocol.json'),
        train_indices=train.tolist(),validation_indices=val.tolist(),regularization_C=[1.,10.],gamma=gamma,median_training_squared_distance=median,training_calibration_folds=[dict(fit=a.tolist(),test=b.tolist()) for a,b in cv],training_component_groups=groups.tolist(),
        hypothesis='Test nonlinear whole-scene/leaf interactions with the full quarantined field cohort rather than bounded rejected-case replay. All original mobile ROI and SigLIP whole-photo features are reused exactly. Earlier bilinear context heads used only eligible rejected training images; this control includes 66 omitted Healthy field examples. No upstream guard is changed.',
        input='Equal unit-norm whole SigLIP and mean original DINO/CNN per crop, two views remain in the same calibration fold',
        calibration='Four training-only group/ROI-component folds, weighted Platt logistic C=1; no validation calibration or threshold fitting',target_replay_weight=.5,healthy_admission=.95,seed=2140,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False,
        stopping='Exactly C=1 and C=10 with training-median kernel width. Stop either failing family without encoder inference. Full gates only after >=28/40 Healthy and zero new false Healthy; no export.')
    freeze(out/'protocol.json',protocol)
    data=load(p);entries,records,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({b['path']:(b,packets[b['id']],r) for b,r in zip(before,previous)})
    controls=[]
    for c in (1.,10.):
        def fit(ids):
            return SVC(C=c,kernel='rbf',gamma=gamma,cache_size=256).fit(x[ids].reshape(-1,x.shape[-1]),np.repeat(y[ids],2),sample_weight=np.repeat(weights[ids],2))
        oof=np.full((len(train),2),np.nan)
        for a,b in cv:oof[b]=fit(a).decision_function(x[b].reshape(-1,x.shape[-1])).reshape(len(b),2)
        assert np.isfinite(oof).all()
        calibrator=LogisticRegression(C=1.,max_iter=2000).fit(oof.reshape(-1,1),np.repeat(y,2),sample_weight=np.repeat(weights,2))
        model=fit(np.arange(len(train)));raw=model.decision_function(v.reshape(-1,v.shape[-1])).reshape(len(val),2)
        probability=calibrator.predict_proba(raw.reshape(-1,1))[:,list(calibrator.classes_).index(1)].reshape(len(val),2).min(1)
        errors={};healthy=selection['fixed_healthy_validation_correct'];old_h=0
        for i,score in zip(val,probability):
            row=rows[i];original,packet,retained=items[row['path']];now=recover_only(original,retained,packet,float(score),data['manifest'])['app_replay']['kind']=='HEALTHY'
            if not row['y']:errors[row['source']]=errors.get(row['source'],0)+int(now)
            elif row['source']=='olid_tomato':healthy+=int(now)
            else:old_h+=int(now)
        cheap=sum(errors.values())==0 and healthy>=28
        result=dict(C=c,healthy_validation_correct=healthy,new_false_healthy_by_source=errors,old_healthy_correct=old_h,cheap_pass=cheap,passed=False)
        folder=out/('C'+str(int(c)));folder.mkdir(exist_ok=True)
        np.savez(folder/'predictions.npz',indices=val,probability=probability,oof_training_margin=oof,validation_margin=raw,protocol_sha256=digest(out/'protocol.json'))
        # Explicit arrays preserve model bytes without executing arbitrary pickle on restore.
        np.savez(folder/'model.npz',support_vectors=model.support_vectors_,dual_coef=model.dual_coef_,intercept=model.intercept_,classes=model.classes_,gamma=gamma,platt_coef=calibrator.coef_,platt_intercept=calibrator.intercept_,platt_classes=calibrator.classes_)
        result['model_sha256']=digest(folder/'model.npz');controls.append(result);print(json.dumps(result),flush=True)
    result=dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),controls=controls,model_promoted=False,new_encoder_inputs=0,elapsed_seconds=time.monotonic()-started,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    (out/'result.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
