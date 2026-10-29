"""Consumed-case diagnosis, never a fitted recovery rule or fresh acceptance."""
from pathlib import Path
import argparse,collections,json
import numpy as np
from sklearn.metrics import roc_auc_score
from conditional_recovery_policy import recover_only
from opened_context_data import load,read
from phase85_population_sft import evaluation_data
from phase215_shrinkage_density import inputs
from permissive_encoder import digest

def main():
    a=argparse.ArgumentParser();a.add_argument('--campaign',required=True);p=Path(a.parse_args().campaign);out=Path('/dev/shm/agribot-phase226-attribution');out.mkdir(exist_ok=True);data=load(p);entries,_,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({r['path']:(r,packets[r['id']],retained) for r,retained in zip(before,previous)});rows=read(p/'phase84_population_head/fitting_rows.json');selection=read(p/'phase118_conditional_recovery/protocol.json');ids=selection['validation_indices'];bank_path=p/'phase201_two_snapshot_coverage/candidate_bank.npz';bank=np.load(bank_path);assert ids==bank['indices'].tolist();cases=[]
    extra=[i for i,r in enumerate(rows) if r['validation'] and r['y'] and r['source']=='olid_tomato' and i not in ids]
    for j,i in enumerate(ids+extra):
        r=rows[i];original,packet,retained=items[r['path']];allowed=recover_only(original,retained,packet,1.,data['manifest'])['app_replay']['kind']=='HEALTHY'
        if j<len(ids):assert allowed==bool(bank['admissible'][j])
        v=retained['app_replay'];reference=v.get('probabilities') or retained.get('context_whole_reference_probabilities');cp=min((a[0] for leaf in packet['leaves'] for a in leaf),default=1.);cn=min((max(a[1] for a in leaf) for leaf in packet['leaves']),default=1.);reasons=[]
        if v['kind']=='DISEASE':reasons.append('existing_named_disease_preserved')
        if not packet['leaves']:reasons.append('no_actual_leaf_ROI')
        if not packet['usable']:reasons.append('unusable_evidence')
        if packet['whole'][0]<.85:reasons.append('whole_healthy_support_below_.85')
        if packet['whole'][2]<.5:reasons.append('genus_support_below_.5')
        if cp<.1:reasons.append('compact_positive_below_.1')
        if cn<=.001:reasons.append('compact_negative_at_or_below_.001')
        if retained['contrast']<.015 and retained['sharpness']<.012:reasons.append('low_image_quality')
        if v['kind']=='UNSURE' and v.get('label') not in (None,'','Healthy','Other'):reasons.append('protected_suspected_disease')
        if reference is None:reasons.append('no_original_class_probabilities')
        elif reference[8]<=.1 and sum(reference[k] for k in (0,2,3,4,5,6,7))>=.6:reasons.append('original_high_disease_mass')
        cases.append(dict(index=i,path=r['path'],label=r['label'],source=r['source'],healthy=bool(r['y']),guard_admissible=allowed,guard_diagnostic_reasons=reasons,retained_kind=v['kind'],retained_label=v.get('label'),original_probability=reference))
    screen=cases[:len(ids)];target=np.array([r['healthy'] and r['source']=='olid_tomato' for r in screen]);negative=np.array([not r['healthy'] for r in screen]);allowed=bank['admissible'];results=[]
    for kind in ('mobile','native'):
        rr,tr,va,x,v,w,keys,parents=inputs(p,kind);assert va==ids;y=np.array([rr[i]['y'] for i in tr]);view_distance=np.linalg.norm(v[:,0]-v[:,1],axis=1);training_distance=np.linalg.norm(x[:,0]-x[:,1],axis=1)
        for case,d in zip(cases,view_distance):case[kind+'_cross_crop_distance']=float(d)
        results.append(dict(features=kind,target_healthy_view_distance_quantiles=np.quantile(view_distance[target],[0,.25,.5,.75,1]).tolist(),diseased_view_distance_quantiles=np.quantile(view_distance[negative],[0,.25,.5,.75,1]).tolist(),training_view_distance_quantiles=np.quantile(training_distance,[0,.25,.5,.75,1]).tolist(),training_counts=dict(collections.Counter(f"{rr[i]['source']}:{rr[i]['label']}" for i in tr))))
    controls=[]
    for name in ('structured_only','native_plus_structured'):
        path=p/f'phase224_structured_kernels/{name}/predictions.npz';z=np.load(path);assert np.array_equal(z['indices'],ids);raw=z['validation_margin'].min(1);oof=z['oof_training_margin'].min(1);rp=read(p/'phase224_structured_kernels/protocol.json');y=np.array([rr[i]['y'] for i in rp['train_indices']]);valid=allowed&(target|negative);controls.append(dict(name=name,prediction_sha256=digest(path),training_group_oof_auc=float(roc_auc_score(y,oof)),consumed_eligible_target_vs_disease_auc=float(roc_auc_score(target[valid],raw[valid])),target_at_95=int(((z['probability']>=.95)&target&allowed).sum()),oracle_zero_fp_recoveries=int((target&allowed&(raw>raw[negative&allowed].max())).sum()),oracle_diagnostic_only=True))
    blocked=[r for r in cases if r['healthy'] and r['source']=='olid_tomato' and not r['guard_admissible']];result=dict(source_sha256=digest(__file__),bank_sha256=digest(bank_path),new_encoder_inputs=0,acceptance_data_used=False,ad_independent=False,olid_independent=False,healthy_denominator=40,retained_healthy=selection['fixed_healthy_validation_correct'],eligible_target_healthy=int((target&allowed).sum()),blocked_target_healthy=len(blocked),maximum_possible_under_unchanged_guards=selection['fixed_healthy_validation_correct']+int((target&allowed).sum()),blocked_reason_counts=dict(collections.Counter(reason for r in blocked for reason in r['guard_diagnostic_reasons'])),cases=cases,representation_diagnostics=results,model_diagnostics=controls,policy_fitted=False,labels_changed=False)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('cases','representation_diagnostics')}))

if __name__=='__main__':main()
