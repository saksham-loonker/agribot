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
    p=Path(parser.parse_args().campaign);out=p/'phase180_native_field_kernel';out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    source=p/'phase172_native_large_field_features';lp=read(source/'protocol.json');lc=read(source/'complete.json')
    assert lc['protocol_sha256']==digest(source/'protocol.json') and lc['features_sha256']==digest(source/'features.npy')
    rows=read(p/'phase168_early_cnn_prefix/rows.json');assert digest(p/'phase168_early_cnn_prefix/rows.json')==lp['rows_sha256']
    selection=read(p/'phase118_conditional_recovery/protocol.json');train=np.array(lp['train_indices']);val=np.array(lp['validation_indices']);assert val.tolist()==selection['validation_indices']
    conv=p/'phase154_convnext_features';cp=read(conv/'protocol.json');cc=read(conv/'complete.json');assert cc['features_sha256']==digest(conv/'features.npy') and digest(conv/'complete.json')==lp['convnext_complete_sha256']
    cnn={sha:value for sha,value in zip(cp['input_sha256'],np.load(conv/'features.npy'))}
    field=p/'phase164_new_field_features';fc=read(field/'complete.json');assert digest(field/'complete.json')==lp['field_complete_sha256'] and fc['features_sha256']==digest(field/'features.npy') and fc['input_sha256_sha256']==digest(field/'input_sha256.json')
    cnn.update({sha:value[2432:] for sha,value in zip(read(field/'input_sha256.json'),np.load(field/'features.npy'))})
    from native_joint_features import join_native_cnn
    features=join_native_cnn(np.load(source/'features.npy'),[r['input_sha256'] for r in lp['inputs']],lp['links'],cnn)
    features=np.concatenate([features[:,:1024]/np.maximum(np.linalg.norm(features[:,:1024],axis=1,keepdims=True),1e-8),features[:,1024:]/np.maximum(np.linalg.norm(features[:,1024:],axis=1,keepdims=True),1e-8)],axis=1)/(2**.5)
    def photos(indices):
        return np.array([[features[keys].mean(0) for keys in lp['bags'][rows[i]['path']]] for i in indices])
    x=photos(train);v=photos(val);y=np.array([rows[i]['y'] for i in train])
    # A complete new publisher collection is one calibration group. Merge any
    # older groups sharing an exact native input, keeping every view together.
    group=[('field93:entire_publisher_collection' if rows[i]['source']=='field93' else rows[i]['group']) for i in train]
    parent=list(range(len(train)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    owner={}
    for j,i in enumerate(train):
        keys=[('family',group[j])]+[('input',lp['inputs'][key]['input_sha256']) for view in lp['bags'][rows[i]['path']] for key in view]
        for key in keys:
            if key in owner:parent[find(j)]=find(owner[key])
            else:owner[key]=j
    groups=np.array([find(i) for i in range(len(train))])
    assert not ({lp['inputs'][key]['input_sha256'] for i in train for view in lp['bags'][rows[i]['path']] for key in view}&{lp['inputs'][key]['input_sha256'] for i in val for view in lp['bags'][rows[i]['path']] for key in view})
    population=read(p/'phase168_early_cnn_prefix/protocol.json');target=population['target_indices'];anchors=population['anchor_indices'];field_indices=population['field_indices'];assert train.tolist()==target+anchors+field_indices
    from source_balance import source_class_group_weights
    weights=np.concatenate([.5*source_class_group_weights(rows,target),.25*source_class_group_weights(rows,anchors),.25*source_class_group_weights(rows,field_indices)])*len(train)
    squared=pairwise_distances(x.mean(1),metric='sqeuclidean');median=float(np.median(squared[np.triu_indices(len(x),1)]));assert median>0;gamma=1/median
    cv=list(StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=2140).split(x,y,groups))
    for fit,test in cv:assert not set(groups[fit])&set(groups[test]) and set(y[fit])=={0,1}
    protocol=dict(source_sha256=digest(__file__),native_complete_sha256=digest(source/'complete.json'),convnext_complete_sha256=digest(conv/'complete.json'),field_complete_sha256=digest(field/'complete.json'),
        train_indices=train.tolist(),validation_indices=val.tolist(),regularization_C=[1.,10.],gamma=gamma,median_training_squared_distance=median,training_calibration_folds=[dict(fit=a.tolist(),test=b.tolist()) for a,b in cv],training_component_groups=groups.tolist(),
        hypothesis='Nonlinear kernel separation of genuine native-518 semantic evidence plus cached ConvNeXt texture, with audited field supervision. Earlier large-kernel control used 224px evidence and no new field collection. Two fixed C values only; no validation-derived calibration, cutoff or reward.',
        input='Per-branch L2 normalization, actual ROI mean per center/full crop; image-level supervision only',
        calibration='Training-only four-fold grouped OOF Platt C=1. Whole field93 publisher source, family groups, shared exact ROI inputs and all views remain together.',
        training_mass=dict(target=.5,old_anchor=.25,field93=.25),healthy_admission=.95,seed=2140,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False,
        stopping='Exactly two cheap kernels, no validation grid or encoder runs. Require >=28/40 Healthy and zero new false Healthy before unchanged full gates; no export.')
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
