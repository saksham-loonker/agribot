"""Determine whether cached expert committees warrant new training predictions.

Consumed validation labels enter an optimistic diagnostic only. The resulting
weights must never be treated as a trained candidate or independent evidence.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.special import expit,logit
from committee_outer_bound import relaxed_committee_upper_bound
from conditional_recovery_policy import recover_only
from opened_context_data import load,read
from permissive_encoder import digest
from phase85_population_sft import evaluation_data
from population_siglip_tail import freeze


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase170_committee_outer_audit';out.mkdir(exist_ok=True);started=time.monotonic()
    old=p/'phase157_cached_separation_audit';previous=read(old/'protocol.json');files={p/name:sha for name,sha in previous['prediction_sha256'].items()}
    for name in ('phase158_joint_cached_teacher','phase159_image_bag_clinical_teacher'):
        path=p/name/'best_validation_predictions.npz';files[path]=digest(path)
    original_files=list(files)
    for name in ('phase165_new_field_joint_teacher','phase166_new_field_bag_teacher','phase167_field_student_recovery','phase169_field_style_cnn'):
        for path in sorted((p/name).rglob('best_validation_predictions.npz')):files[path]=digest(path)
    selection=read(p/'phase118_conditional_recovery/protocol.json');ids=np.array(selection['validation_indices']);rows=read(p/'phase84_population_head/fitting_rows.json');bank=[]
    for path,sha in files.items():
        assert digest(path)==sha;z=np.load(path);assert np.array_equal(z['indices'],ids)
        if 'probability' in z.files:probability=z['probability']
        else:
            raw=z['paired_margin'] if 'paired_margin' in z.files else z['logits'];probability=expit(raw.min(1))
        assert probability.shape==ids.shape and np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all();bank.append(probability)
    bank=np.stack(bank,1)
    protocol=dict(source_sha256=digest(__file__),solver_source_sha256=digest(Path(__file__).with_name('committee_outer_bound.py')),selection_sha256=digest(p/'phase118_conditional_recovery/protocol.json'),
        guard_source_sha256=digest(Path(__file__).with_name('conditional_recovery_policy.py')),prediction_sha256={str(path.relative_to(p)):sha for path,sha in files.items()},
        aggregation=['convex_probability','convex_clipped_logit'],clip_logit=12.,fixed_admission=.95,solver_time_limit_per_family_seconds=30.,numerical_outer_relaxation=1e-6,scopes=dict(previous_selected=len(original_files),expanded_with_field=len(files)),
        diagnostic_only=True,uses_consumed_validation_labels=True,weights_must_not_become_candidate=True,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False,
        stopping='Two aggregation families in each of two explicit cached expert scopes; positive and negative inequalities are relaxed outwards by 1e-6 so exact-boundary cases are not incorrectly excluded. No learned mixture is selected from these validation labels. If no committee can reach 28/40 at zero new false Healthy, stop these convex-committee families on the selected checkpoints. Otherwise acquire identity-checked training/calibration predictions under a separately frozen protocol before fitting any real committee.')
    freeze(out/'protocol.json',protocol)
    if (out/'result.json').exists():assert read(out/'result.json')['protocol_sha256']==digest(out/'protocol.json');return
    data=load(p);entries,records,packets,before,previous=evaluation_data(p,data)
    items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({b['path']:(b,packets[b['id']],r) for b,r in zip(before,previous)})
    eligible=[]
    for i in ids:
        original,packet,retained=items[rows[i]['path']];eligible.append(recover_only(original,retained,packet,1.,data['manifest'])['app_replay']['kind']=='HEALTHY')
    eligible=np.array(eligible);positive=eligible&np.array([rows[i]['y']==1 and rows[i]['source']=='olid_tomato' for i in ids]);negative=eligible&np.array([rows[i]['y']==0 for i in ids]);controls=[]
    for scope,count in [('previous_selected',len(original_files)),('expanded_with_field',len(files))]:
        subset=bank[:,:count]
        for name,values,boundary in [('convex_probability',subset,.95),('convex_clipped_logit',logit(np.clip(subset,expit(-12.),expit(12.))),logit(.95))]:
            result=relaxed_committee_upper_bound(values[positive],values[negative],boundary)
            fixed=selection['fixed_healthy_validation_correct'];result.update(scope=scope,experts=count,aggregation=name,fixed_healthy=fixed,relaxed_total_healthy_upper_bound=None if result['relaxed_healthy_upper_bound'] is None else fixed+result['relaxed_healthy_upper_bound'])
            controls.append(result);print(json.dumps({k:v for k,v in result.items() if k!='weights'}),flush=True)
    result=dict(protocol_sha256=digest(out/'protocol.json'),experts=len(files),controls=controls,diagnostic_only=True,candidate_selected=False,model_promoted=False,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False,elapsed_seconds=time.monotonic()-started)
    (out/'result.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
