"""Consumed-data oracle diagnostic; never a threshold or routing selector.

An optimistic union of per-expert, validation-label-derived zero-FP cutoffs
tests whether simple monotone threshold/OR combinations warrant more work.
It does not bound learned multivariate classifiers or unseen model families.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
from conditional_recovery_policy import recover_only
from opened_context_data import load,read
from permissive_encoder import digest
from phase85_population_sft import evaluation_data
from population_siglip_tail import freeze


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase157_cached_separation_audit';out.mkdir(exist_ok=True);started=time.monotonic()
    selection=read(p/'phase118_conditional_recovery/protocol.json');ids=np.array(selection['validation_indices']);rows=read(p/'phase84_population_head/fitting_rows.json')
    names=['phase124_lora_recovery','phase127_clinical_anchors','phase129_native_dino_mil','phase132_large_roi_recovery','phase134_anchor_lora','phase135_condition_split_lora','phase136_hard_negative_lora',
        'phase139_full_source_recovery','phase140_large_kernel_recovery','phase142_full_source_cnn_recovery','phase144_photometric_cnn_recovery','phase145_photometric_lora_recovery',
        'phase146_original_clinical_lora','phase147_large_clinical_teacher','phase148_guard_reward_teacher','phase150_semantic_recovery','phase151_native_global_recovery',
        'phase152_balanced_photo_teacher','phase153_healthy_axis_cnn','phase154_convnext_clinical_teacher','phase155_normal_patch_recovery','phase156_cached_teacher_ranking']
    files=sorted(path for name in names for path in (p/name).rglob('*.npz') if path.name in ('best_validation_predictions.npz','validation_predictions.npz','original_native_class_predictions.npz','predictions.npz'))
    assert len(files)>20
    protocol=dict(source_sha256=digest(__file__),selection_protocol_sha256=digest(p/'phase118_conditional_recovery/protocol.json'),
        routing_source_sha256=digest(Path(__file__).with_name('conditional_recovery_policy.py')),prediction_sha256={str(path.relative_to(p)):digest(path) for path in files},
        validation_indices=ids.tolist(),diagnostic_only=True,uses_consumed_validation_labels=True,ad_independent=False,olid_independent=False,acceptance_data_used=False,new_encoder_inputs=0,
        interpretation='Optimistic per-expert zero-new-false-Healthy cutoff union on guard-admissible consumed validation photos. These cutoffs must never be used as a candidate policy. A failed union stops simple threshold/OR work on these selected checkpoints; it does not rule out learned interactions, other epochs, architectures, supervision or better data.')
    freeze(out/'protocol.json',protocol)
    if (out/'result.json').exists():assert read(out/'result.json')['protocol_sha256']==digest(out/'protocol.json');return
    data=load(p);entries,records,packets,before,previous=evaluation_data(p,data)
    items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({b['path']:(b,packets[b['id']],r) for b,r in zip(before,previous)})
    admissible=[]
    for i in ids:
        original,packet,retained=items[rows[i]['path']];admissible.append(recover_only(original,retained,packet,1.,data['manifest'])['app_replay']['kind']=='HEALTHY')
    admissible=np.array(admissible);negative=admissible&np.array([rows[i]['y']==0 for i in ids]);positive=admissible&np.array([rows[i]['y']==1 and rows[i]['source']=='olid_tomato' for i in ids])
    assert negative.any() and positive.any();union=np.zeros(len(ids),bool);reports=[]
    for path in files:
        value=np.load(path);assert np.array_equal(value['indices'],ids),('Prediction population mismatch',path)
        if 'probability' in value.files:probability=value['probability']
        else:
            margin=value['paired_margin'] if 'paired_margin' in value.files else value['logits'];probability=1/(1+np.exp(-np.clip(margin.min(1),-80,80)))
        assert probability.shape==ids.shape and np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all()
        maximum_negative=float(probability[negative].max());recovery=admissible&(probability>maximum_negative);union|=recovery
        reports.append(dict(name=str(path.relative_to(p)),oracle_zero_new_fp_healthy=selection['fixed_healthy_validation_correct']+int((positive&recovery).sum()),healthy_at_fixed_admission=selection['fixed_healthy_validation_correct']+int((positive&(probability>=.95)).sum()),maximum_admissible_negative_probability=maximum_negative))
    reports.sort(key=lambda r:r['oracle_zero_new_fp_healthy'],reverse=True)
    result=dict(protocol_sha256=digest(out/'protocol.json'),selected_experts=len(files),controls=reports,
        fixed_healthy_correct=selection['fixed_healthy_validation_correct'],guard_reachable_rejected_healthy=int(positive.sum()),admissible_negative_images=int(negative.sum()),
        oracle_zero_fp_union_healthy=selection['fixed_healthy_validation_correct']+int((union&positive).sum()),required_healthy=28,
        simple_monotone_threshold_or_still_possible=selection['fixed_healthy_validation_correct']+int((union&positive).sum())>=28,
        diagnostic_only=True,candidate_selected=False,model_promoted=False,acceptance_data_used=False,ad_independent=False,olid_independent=False,new_encoder_inputs=0,elapsed_seconds=time.monotonic()-started)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='controls'}),flush=True)


if __name__=='__main__':main()
