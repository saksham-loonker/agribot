"""Full consumed-cohort guard ceiling and labeled-training coverage, no fitting."""
from pathlib import Path
import argparse,collections,json,time
from actual_admission_target import actual_admission_target
from conditional_recovery_policy import recover_only
from opened_context_data import load,read
from permissive_encoder import digest
from phase85_population_sft import evaluation_data
from population_siglip_tail import freeze


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase178_guard_coverage_audit';out.mkdir(exist_ok=True);started=time.monotonic()
    data=load(p);entries,records,packets,before,previous=evaluation_data(p,data);rows=read(p/'phase84_population_head/fitting_rows.json');by_path={r['path']:r for r in rows};clinical=read(p/'phase127_clinical_anchors/protocol.json');bounded=read(p/'phase133_anchor_input_cache/protocol.json')
    protocol=dict(source_sha256=digest(__file__),route_source_sha256=digest(Path(__file__).with_name('conditional_recovery_policy.py')),fitting_rows_sha256=digest(p/'phase84_population_head/fitting_rows.json'),
        full_clinical_protocol_sha256=digest(p/'phase127_clinical_anchors/protocol.json'),bounded_replay_protocol_sha256=digest(p/'phase133_anchor_input_cache/protocol.json'),diagnostic_only=True,uses_consumed_labels=True,
        counterfactual='Healthy-labelled cases get admission probability one only to measure a routing ceiling; these label-oracle choices are never a candidate policy',original_recognition_and_safety_gates_unchanged=True,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    freeze(out/'protocol.json',protocol)
    if (out/'result.json').exists():assert read(out/'result.json')['protocol_sha256']==digest(out/'protocol.json');return
    cases=[]
    for record,original,retained in zip(records,before,previous):
        if record['label']!='Healthy':continue
        packet=packets[record['id']];possible=recover_only(original,retained,packet,1.,data['manifest'])['app_replay']['kind']=='HEALTHY';fixed=retained['app_replay']['kind']=='HEALTHY';reasons=[]
        if not possible:
            v=retained['app_replay'];probability=v.get('probabilities') or packet.get('fallback')
            if v['kind']=='DISEASE':reasons.append('retained_named_disease')
            if v['kind']=='UNSURE' and v.get('label') not in (None,'','Healthy','Other'):reasons.append('retained_named_suspicion')
            if not packet['usable']:reasons.append('packet_unusable')
            if not packet['leaves']:reasons.append('no_detected_leaf')
            if packet['whole'][3]>=.5:reasons.append('whole_unusable')
            if packet['whole'][0]<.85:reasons.append('whole_healthy_below_fixed_guard')
            if packet['whole'][2]<.5:reasons.append('genus_below_fixed_guard')
            if min((view[0] for leaf in packet['leaves'] for view in leaf),default=1.)<.1:reasons.append('compact_healthy_below_fixed_guard')
            if min((max(view[1] for view in leaf) for leaf in packet['leaves']),default=1.)<=.001:reasons.append('compact_knownness_below_fixed_guard')
            if retained['contrast']<.015 and retained['sharpness']<.012:reasons.append('final_low_visibility')
            if probability is None:reasons.append('missing_retained_probabilities')
            elif probability[8]<=.1 and sum(probability[j] for j in (0,2,3,4,5,6,7))>=.6:reasons.append('retained_in_domain_disease_mass')
        cases.append(dict(id=record['id'],path=record['path'],validation=by_path[record['path']]['validation'],fixed_healthy=fixed,oracle_healthy=possible,blocked_reasons=reasons))
    def summarize(part):
        return dict(healthy_images=len(part),fixed_healthy=sum(r['fixed_healthy'] for r in part),guard_recoverable_rejected=sum(r['oracle_healthy'] and not r['fixed_healthy'] for r in part),oracle_maximum_healthy=sum(r['oracle_healthy'] for r in part),blocked_images=sum(not r['oracle_healthy'] for r in part),blocked_reason_counts=dict(collections.Counter(reason for r in part for reason in r['blocked_reasons'])))
    full=summarize(cases);validation=summarize([r for r in cases if r['validation']]);assert full['healthy_images']==223 and validation['healthy_images']==40
    full_pool=[i for i in clinical['train_indices'] if rows[i]['source']=='olid_tomato'];selected=set(bounded['target_indices']+bounded['anchor_indices']);omitted=[i for i in full_pool if i not in selected]
    training=dict(all_available_quarantined_full_field_images=len(full_pool),all_available_label_counts=dict(collections.Counter(rows[i]['label'] for i in full_pool)),selected_bounded_field_images=sum(i in selected for i in full_pool),
        omitted_field_images=len(omitted),omitted_label_counts=dict(collections.Counter(rows[i]['label'] for i in omitted)),omitted_indices=omitted,
        source_selection='All source/family/raw/ROI validation quarantines from phase127 retained; omission means bounded fitting replay, not independent or unused dataset evidence')
    result=dict(protocol_sha256=digest(out/'protocol.json'),full_consumed_healthy=full,validation_healthy=validation,training_coverage=training,full_gate_threshold=157,validation_gate_threshold=28,
        routing_ceiling_below_full_gate=full['oracle_maximum_healthy']<157,cases=cases,diagnostic_only=True,model_promoted=False,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False,elapsed_seconds=time.monotonic()-started)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k in ('full_consumed_healthy','validation_healthy','routing_ceiling_below_full_gate')}),flush=True);print(json.dumps({k:v for k,v in training.items() if k!='omitted_indices'}),flush=True)


if __name__=='__main__':main()
