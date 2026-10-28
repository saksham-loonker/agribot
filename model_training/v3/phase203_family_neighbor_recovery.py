"""Two local family-neighbor controls on exact cached frozen mobile features."""
from pathlib import Path
from collections import defaultdict
import argparse,json,time
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from clinical_replay_balance import clinical_replay_weights
from conditional_recovery_policy import recover_only
from exhaustive_replay_data import exhaustive_replay_data
from family_knn import family_knn_margin
from opened_context_data import load,read
from phase85_population_sft import evaluation_data
from permissive_encoder import digest
from population_siglip_tail import freeze


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);p=Path(parser.parse_args().campaign);out=p/'phase203_family_neighbor_recovery';out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    values,identities,rows,tables,train,val,meta=exhaustive_replay_data(p);rotation=p/'phase195_rotation_feature_extension';rp=read(rotation/'protocol.json');rc=read(rotation/'complete.json');assert rc['protocol_sha256']==digest(rotation/'protocol.json') and rc['features_sha256']==digest(rotation/'features.npy') and rc['augmented_bags_sha256']==digest(rotation/'augmented_bags.json')
    assert rp['base_count']==len(values);identities += [r['input_sha256'] for r in rp['new_inputs']];values=np.concatenate([values,np.load(rotation/'features.npy')]);tables.update({int(i):views for i,views in read(rotation/'augmented_bags.json').items()})
    quarantine=set(read(p/'phase196_rotation_mobile_replay/protocol.json')['rotation_quarantined_groups']);train=[i for i in train if rows[i]['group'] not in quarantine];target=[i for i in meta['target'] if i in set(train)];replay=[i for i in meta['replay'] if i in set(train)];field=[i for i in meta['field_indices'] if i in set(train)];assert train==target+replay+field
    weights=np.concatenate([.5*clinical_replay_weights(rows,target),.25*clinical_replay_weights(rows,replay),.25*clinical_replay_weights(rows,field)])*len(train);y=np.array([rows[i]['y'] for i in train]);parent=list(range(len(train)));owner={}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for j,i in enumerate(train):
        keys=[('family',rows[i]['group'])]+[('input',identities[key]) for pair in tables[i] for view in pair for key in view]
        for key in keys:
            if key in owner:parent[find(j)]=find(owner[key])
            else:owner[key]=j
    families=np.array([find(i) for i in range(len(train))]);field_roots={families[j] for j,i in enumerate(train) if rows[i]['source']=='field93'};cv_groups=np.array(['field93_entire_collection' if family in field_roots else str(family) for family in families]);folds=list(StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=2203).split(train,y,cv_groups))
    for fit,test in folds:assert not set(cv_groups[fit])&set(cv_groups[test]) and not set(families[fit])&set(families[test]) and set(y[fit])=={0,1}
    family_mass=defaultdict(float)
    for family,weight in zip(families,weights):family_mass[int(family)]+=float(weight)
    mass=np.array([family_mass[int(family)] for family in families])
    def unit(value):return value/max(float(np.linalg.norm(value)),1e-8)
    def photo(i):
        result=[]
        for pair in tables[i]:
            for keys in pair:
                value=values[keys].mean(0);result.append(np.concatenate([unit(value[:384]),unit(value[384:])])/(2**.5))
        return np.stack(result)
    x=np.stack([photo(i) for i in train]);v=np.stack([photo(i) for i in val]);assert x.shape[1:]==(8,1408) and v.shape[1:]==(2,1408)
    protocol=dict(source_sha256=digest(__file__),neighbor_source_sha256=digest(Path(__file__).with_name('family_knn.py')),data_source_sha256=digest(Path(__file__).with_name('exhaustive_replay_data.py')),full_replay_protocol_sha256=meta['full_replay_protocol_sha256'],rotation_complete_sha256=digest(rotation/'complete.json'),
        train_indices=train,validation_indices=val,methods=['unrotated','right_angle_replay'],neighbors=5,distance='Equal unit-norm original DINO384 and CNN1024 photo/view means; squared Euclidean',voting='Inverse squared-distance weighted family mass, at most one vote per inferred family regardless of augmentation count',
        family_ids=families.tolist(),cv_group_ids=cv_groups.tolist(),training_weights=weights.tolist(),calibration_folds=[dict(fit=a.tolist(),test=b.tolist()) for a,b in folds],
        calibration='Training-only four-fold family/input-component OOF margins, with whole field93 collection in one fold; weighted Platt logistic C=1, two query views get half a photo weight each',
        hypothesis='Global MLP and fixed-width RBF fits can overfit sparse field distinctions. Test local supervised family neighbors, first unrotated and then with already cached exact right-angle training views. No new image encoding, no per-leaf clinical labels, no validation cutoff fitting.',
        healthy_admission=.95,original_gates_unchanged=True,existing_healthy_and_named_disease_exact=True,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False,
        stopping='Exactly k=5 and the two declared augmentation settings; no neighbor/metric grid. Require >=28/40 Healthy and zero new false Healthy before full unchanged gates. Save calibration and exact bank references rather than duplicating cached feature arrays.')
    freeze(out/'protocol.json',protocol);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    data=load(p);entries,_,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({r['path']:(r,packets[r['id']],retained) for r,retained in zip(before,previous)});selection=read(p/'phase118_conditional_recovery/protocol.json');controls=[]
    for method in protocol['methods']:
        count=2 if method=='unrotated' else 8;folder=out/method;folder.mkdir(exist_ok=True)
        def predict(fit,query):return family_knn_margin(x[fit,:count].reshape(-1,1408),np.repeat(y[fit],count),np.repeat(families[fit],count),np.repeat(mass[fit],count),query.reshape(-1,1408),k=5,device='cuda').reshape(len(query),2)
        oof=np.full((len(train),2),np.nan)
        for fit,test in folds:oof[test]=predict(fit,x[test,:2])
        assert np.isfinite(oof).all();calibrator=LogisticRegression(C=1.,max_iter=2000).fit(oof.reshape(-1,1),np.repeat(y,2),sample_weight=np.repeat(weights/2,2));assert calibrator.coef_[0,0]>0,'OOF evidence is not positively ordered'
        raw=predict(np.arange(len(train)),v);probability=calibrator.predict_proba(raw.reshape(-1,1))[:,list(calibrator.classes_).index(1)].reshape(len(val),2).min(1);errors={};healthy=selection['fixed_healthy_validation_correct'];old_h=0
        for i,score in zip(val,probability):
            row=rows[i];original,packet,retained=items[row['path']];now=recover_only(original,retained,packet,float(score),data['manifest'])['app_replay']['kind']=='HEALTHY'
            if not row['y']:errors[row['source']]=errors.get(row['source'],0)+int(now)
            elif row['source']=='olid_tomato':healthy+=int(now)
            else:old_h+=int(now)
        model=dict(protocol_sha256=digest(out/'protocol.json'),method=method,neighbors=5,train_indices=train,family_ids=families.tolist(),family_mass=mass.tolist(),calibration_coef=calibrator.coef_.tolist(),calibration_intercept=calibrator.intercept_.tolist(),classes=calibrator.classes_.tolist(),bank_reconstruction='exhaustive_replay_data + phase195 augmentation, photo ROI mean and per-branch unit normalization; vectors derive from existing frozen caches')
        (folder/'model.json').write_text(json.dumps(model,indent=2));np.savez(folder/'predictions.npz',indices=np.array(val),probability=probability,oof_training_margin=oof,validation_margin=raw,model_sha256=digest(folder/'model.json'),protocol_sha256=digest(out/'protocol.json'))
        result=dict(method=method,healthy_validation_correct=healthy,new_false_healthy_by_source=errors,old_healthy_correct=old_h,model_sha256=digest(folder/'model.json'),cheap_pass=healthy>=28 and sum(errors.values())==0,model_promoted=False);controls.append(result);print(json.dumps(result),flush=True)
    result=dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),controls=controls,model_promoted=False,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False,elapsed_seconds=time.monotonic()-started);(out/'result.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
