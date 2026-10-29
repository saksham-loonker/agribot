"""Audit original-versus-retained admission eligibility without new predictions."""
from pathlib import Path
import argparse,collections,json,time
from actual_admission_target import actual_admission_target
from conditional_recovery_policy import recover_only
from opened_context_data import load,read
from permissive_encoder import digest
from phase83_eligible_operating_points import eligible
from phase85_population_sft import evaluation_data
from population_siglip_tail import freeze


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase177_actionability_audit';out.mkdir(exist_ok=True);started=time.monotonic()
    selection=read(p/'phase118_conditional_recovery/protocol.json');anchors=read(p/'phase133_anchor_input_cache/protocol.json');rows=read(p/'phase84_population_head/fitting_rows.json');data=load(p);entries,records,packets,before,previous=evaluation_data(p,data)
    items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({b['path']:(b,packets[b['id']],r) for b,r in zip(before,previous)})
    protocol=dict(source_sha256=digest(__file__),target_helper_sha256=digest(Path(__file__).with_name('actual_admission_target.py')),selection_protocol_sha256=digest(p/'phase118_conditional_recovery/protocol.json'),
        anchor_protocol_sha256=digest(p/'phase133_anchor_input_cache/protocol.json'),diagnostic_only=True,policy_unchanged=True,full_validation_population_unchanged=True,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    freeze(out/'protocol.json',protocol)
    if (out/'result.json').exists():assert read(out/'result.json')['protocol_sha256']==digest(out/'protocol.json');return
    def inspect(ids):
        counts=collections.Counter();cases=[];retained_ids=[]
        for i in ids:
            row=rows[i];original,packet,retained=items[row['path']];old=eligible(original,packet);actual=actual_admission_target(original,retained,packet,data['manifest'])
            counts[(row['label'],old,actual)]+=1
            if actual:retained_ids.append(i)
            elif old:
                verdict=retained['app_replay'];p0=verdict.get('probabilities') or packet.get('fallback');reasons=[]
                if verdict['kind'] in ('HEALTHY','DISEASE'):reasons.append('protected_retained_'+verdict['kind'].lower())
                if verdict['kind']=='UNSURE' and verdict.get('label') not in (None,'','Healthy','Other'):reasons.append('retained_named_suspicion')
                if retained['contrast']<.015 and retained['sharpness']<.012:reasons.append('final_low_visibility')
                if p0 is None:reasons.append('missing_retained_probabilities')
                elif p0[8]<=.1 and sum(p0[j] for j in (0,2,3,4,5,6,7))>=.6:reasons.append('retained_in_domain_disease_mass')
                cases.append(dict(index=i,id=original['id'],source=row['source'],label=row['label'],retained_kind=verdict['kind'],retained_label=verdict.get('label'),reasons=reasons or ['other_final_route_guard']))
        return dict(images=len(ids),counts=[dict(label=label,original_eligible=old,actual_admissible=actual,images=n) for (label,old,actual),n in sorted(counts.items())],eligible_indices=retained_ids,mismatches=cases)
    target=inspect(anchors['target_indices']);training=inspect(selection['train_indices']);validation=inspect(selection['validation_indices'])
    result=dict(protocol_sha256=digest(out/'protocol.json'),target_training=target,all_conditional_training=training,validation=validation,
        fixed_healthy_validation_correct=selection['fixed_healthy_validation_correct'],full_healthy_validation_denominator=selection['full_healthy_validation_denominator'],
        diagnostic_only=True,model_promoted=False,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False,elapsed_seconds=time.monotonic()-started)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(target_training_images=target['images'],actual_target_training_images=len(target['eligible_indices']),target_mismatch_images=len(target['mismatches']),validation_mismatch_images=len(validation['mismatches']),new_encoder_inputs=0)),flush=True)


if __name__=='__main__':main()
