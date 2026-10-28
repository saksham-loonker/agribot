"""Family-calibrated recovery using semantic priors on existing image features."""
from pathlib import Path
import argparse,collections,json,time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from conditional_recovery_policy import recover_only
from opened_context_data import load,read
from opened_guard_screen import OpenedGuardScreen
from phase81_consumed_olid_features import prepare as prepare_whole
from phase85_population_sft import evaluation_data
from phase83_eligible_operating_points import safety
from phase22_accept import paired_interval
from phase2_select import retain
from phase5_scene_ablation import photo_metrics
from source_balance import source_class_group_weights
from permissive_encoder import digest
from population_siglip_tail import freeze


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase150_semantic_recovery';out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    text=p/'phase149_semantic_text_features';tc=read(text/'complete.json');assert tc['protocol_sha256']==digest(text/'protocol.json') and tc['text_features_sha256']==digest(text/'text_features.npy') and tc['vision_weight_proof']['all_vision_weights_exact']
    data=load(p);whole,whole_metadata=prepare_whole(p,data);rows=read(p/'phase84_population_head/fitting_rows.json');clinical=read(p/'phase127_clinical_anchors/protocol.json');selection=read(p/'phase118_conditional_recovery/protocol.json');target_selection=read(p/'phase133_anchor_input_cache/protocol.json')
    entries,records,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({b['path']:(b,packets[b['id']],r) for b,r in zip(before,previous)})
    identities={r['path']:r['input_sha256'] for r in read(p/'opened_siglip_feature_cache/inputs.json')+read(p/'phase81_consumed_olid_features/inputs.json')}
    validation=[i for i,r in enumerate(rows) if r['validation']];val_sha={identities[rows[i]['path']] for i in validation if rows[i]['path'] in identities}
    candidates=[i for i in clinical['train_indices'] if rows[i]['path'] in whole];by_sha=collections.defaultdict(set)
    for i in candidates:by_sha[identities[rows[i]['path']]].add(rows[i]['y'])
    conflicting={sha for sha,labels in by_sha.items() if len(labels)>1}
    excluded={rows[i]['group'] for i in candidates if identities[rows[i]['path']] in val_sha|conflicting}
    candidates=[i for i in candidates if rows[i]['group'] not in excluded]
    target_set=set(target_selection['target_indices']);one={}
    # Prefer the predeclared target identity over a coherent old-source duplicate.
    for i in sorted(candidates,key=lambda i:(i not in target_set,i)):one.setdefault(identities[rows[i]['path']],i)
    unique=set(one.values());target=[i for i in target_selection['target_indices'] if i in unique];anchors=sorted(unique-set(target));train=np.array(target+anchors);val=np.array(selection['validation_indices'])
    assert len(target)>100 and len(anchors)>100 and {rows[i]['y'] for i in target}=={0,1}
    assert not ({rows[i]['group'] for i in train}&{rows[i]['group'] for i in validation}) and not ({identities[rows[i]['path']] for i in train}&val_sha)
    prototypes=np.load(text/'text_features.npy');assert prototypes.shape[1]==768 and prototypes.dtype==np.float32
    def semantic_features(paths):
        images=np.stack([whole[path] for path in paths]);images=images/np.linalg.norm(images,axis=1,keepdims=True)
        cosine=images@prototypes.T;return np.concatenate([cosine,(cosine[:,:3].max(1)-cosine[:,3:].max(1))[:,None]],1)
    paths=sorted({rows[i]['path'] for i in train.tolist()+val.tolist()});position={path:i for i,path in enumerate(paths)};features=semantic_features(paths)
    x=features[[position[rows[i]['path']] for i in train]];v=features[[position[rows[i]['path']] for i in val]];y=np.array([rows[i]['y'] for i in train]);groups=np.array([rows[i]['group'] for i in train])
    weights=np.concatenate([.5*source_class_group_weights(rows,target),.5*source_class_group_weights([dict(r,condition_class=r['label']) for r in rows],anchors,target='condition_class')])*len(train)
    folds=list(StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=2150).split(x,y,groups))
    for a,b in folds:assert not set(groups[a])&set(groups[b]) and set(y[a])=={0,1}
    protocol=dict(source_sha256=digest(__file__),semantic_complete_sha256=digest(text/'complete.json'),whole_feature_metadata=whole_metadata,clinical_protocol_sha256=digest(p/'phase127_clinical_anchors/protocol.json'),
        target_selection_sha256=digest(p/'phase133_anchor_input_cache/protocol.json'),train_indices=train.tolist(),validation_indices=val.tolist(),target_photos=len(target),anchor_photos=len(anchors),
        whole_pixel_overlap_groups_quarantined=sorted(excluded),contradictory_training_whole_pixel_identities=len(conflicting),coherent_exact_whole_duplicates_deduplicated=len(candidates)-len(unique),
        feature='Cosine similarity to each frozen clinical prompt plus best Healthy minus best non-Healthy prompt. No source/label metadata enters model inputs.',
        healthy_prompt_columns=[0,1,2],regularization_C=[.1,1.],target_mass=.5,clinical_anchor_mass=.5,
        calibration='Four-fold training-only stratified family OOF decision margins -> weighted Platt logistic C=1; scaler fit within each training fold; no validation cutoff tuning',
        folds=[dict(fit=a.tolist(),test=b.tolist()) for a,b in folds],maximum_solver_iterations=2000,admission_probability=.95,legacy_healthy_and_named_disease_preserved=True,
        original_safety_and_recognition_gates_unchanged=True,new_image_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False,
        stopping='Two fixed regularization controls on the cached semantic basis. Full cached screen only after >=28/40 Healthy and zero new false Healthy. No restored candidate verification or export before every original and consumed gate passes.')
    freeze(out/'protocol.json',protocol)
    controls=[]
    for c in (.1,1.):
        folder=out/('C'+str(c));folder.mkdir(exist_ok=True)
        def fit(ids):
            scaler=StandardScaler().fit(x[ids],sample_weight=weights[ids]);model=LogisticRegression(C=c,max_iter=2000).fit(scaler.transform(x[ids]),y[ids],sample_weight=weights[ids]);assert model.classes_.tolist()==[0,1]
            return scaler,model
        oof=np.full(len(train),np.nan)
        for a,b in folds:
            scaler,model=fit(a);oof[b]=model.decision_function(scaler.transform(x[b]))
        assert np.isfinite(oof).all();calibrator=LogisticRegression(C=1.,max_iter=2000).fit(oof[:,None],y,sample_weight=weights);assert calibrator.classes_.tolist()==[0,1]
        scaler,model=fit(np.arange(len(train)));raw=model.decision_function(scaler.transform(v));probability=calibrator.predict_proba(raw[:,None])[:,1]
        errors={};healthy=selection['fixed_healthy_validation_correct'];old_h=0
        for i,score in zip(val,probability):
            row=rows[i];original,packet,retained=items[row['path']];now=recover_only(original,retained,packet,float(score),data['manifest'])['app_replay']['kind']=='HEALTHY'
            if not row['y']:errors[row['source']]=errors.get(row['source'],0)+int(now)
            elif row['source']=='olid_tomato':healthy+=int(now)
            else:old_h+=int(now)
        cheap=sum(errors.values())==0 and healthy>=28
        np.savez(folder/'model.npz',mean=scaler.mean_,scale=scaler.scale_,coef=model.coef_,intercept=model.intercept_,calibration_coef=calibrator.coef_,calibration_intercept=calibrator.intercept_)
        np.savez(folder/'validation_predictions.npz',indices=val,probability=probability,oof_training_margin=oof,protocol_sha256=digest(out/'protocol.json'))
        result=dict(C=c,healthy_validation_correct=healthy,new_false_healthy_by_source=errors,old_healthy_correct=old_h,cheap_pass=cheap,full_cached_screen_run=False,passed=False,model_sha256=digest(folder/'model.npz'))
        if cheap:
            applicable={path for path,(original,packet,retained) in items.items() if retained['app_replay']['kind'] not in ('HEALTHY','DISEASE') and recover_only(original,retained,packet,1.,data['manifest'])['app_replay']['kind']=='HEALTHY'}
            assert applicable<=set(whole),'No silently missing full-screen cases'
            probabilities={rows[i]['path']:float(score) for i,score in zip(val,probability)};remaining=sorted(applicable-set(probabilities))
            if remaining:
                scores=calibrator.predict_proba(model.decision_function(scaler.transform(semantic_features(remaining)))[:,None])[:,1];probabilities.update({path:float(score) for path,score in zip(remaining,scores)})
            def revised(row,packet,retained):return recover_only(row,retained,packet,probabilities.get(row['path'],0.),data['manifest'])
            def revise(row):return revised(row,*entries[id(row)])
            screen=OpenedGuardScreen(data).evaluate(revise);after=[revised(b,packets[b['id']],r) for b,r in zip(before,previous)]
            vi=[i for i,row in enumerate(rows[-len(records):]) if row['validation']];vr,vb,va=[[group[i] for i in vi] for group in (records,before,after)]
            vm=photo_metrics(va);metrics=photo_metrics(after);bm=photo_metrics(before)
            for group,metric in [(after,metrics),(before,bm)]:metric['unsupported_tomato']=photo_metrics([r for r in group if r['label']=='Other'])['all']
            vsafe,vs=safety(vr,vb,va);asafe,alls=safety(records,before,after);gain=paired_interval(vr,vb,va,lambda r:r['label']=='Healthy',lambda r:int(r['app_replay']['kind']=='HEALTHY'))
            checks=dict(screen['checks'],consumed_validation_healthy=vm['healthy']['correct_rate']>=.7,consumed_full_healthy=metrics['healthy']['correct_rate']>=.7,
                consumed_gain=gain is not None and gain['point']>=.1 and gain['bootstrap_two_sided_95'][0]>0,consumed_validation_safety=vsafe,consumed_full_safety=asafe,consumed_full_retention=retain({'olid':metrics},{'olid':bm}))
            result.update(full_cached_screen_run=True,passed=all(checks.values()),checks=checks,old_screen=screen,consumed_validation=vm,consumed_all=metrics,validation_safety=vs,all_safety=alls,validation_gain=gain)
            (folder/'probabilities.json').write_text(json.dumps(probabilities))
        (folder/'result.json').write_text(json.dumps(result,indent=2));controls.append(result);print(json.dumps({k:v for k,v in result.items() if k not in ('old_screen','consumed_validation','consumed_all')}),flush=True)
    result=dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),controls=controls,passing=sum(r['passed'] for r in controls),model_promoted=False,new_image_encoder_inputs=0,elapsed_seconds=time.monotonic()-started,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    (out/'result.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
