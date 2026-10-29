"""Audit no-leaf fallback availability; missing scores are never model failures."""
from pathlib import Path
import argparse,collections,json
import numpy as np
from conditional_recovery_policy import recover_only
from opened_context_data import load,read
from phase85_population_sft import evaluation_data
from permissive_encoder import digest
from population_siglip_tail import freeze

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);p=Path(ap.parse_args().campaign);out=p/'phase235_scope_attribution';out.mkdir(exist_ok=True);data=load(p);entries,records,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({r['path']:(r,packets[r['id']],retained) for r,retained in zip(before,previous)});rows=read(p/'phase84_population_head/fitting_rows.json');selection=read(p/'phase118_conditional_recovery/protocol.json');ids=selection['validation_indices'];protocol=dict(source_sha256=digest(__file__),selection_sha256=digest(p/'phase118_conditional_recovery/protocol.json'),source_predictions_sha256=digest(p/'phase76_fresh_acceptance/mobile_scores.jsonl'),hypothesis='A strong whole-leaf benchmark need not imply strong detected-ROI app verdicts. Compare cached whole fallback9-class probabilities with actual ROI-pooled probabilities; evaluate exactly frozen whole Healthy probability at.95, preserving all original guards and output retention.',healthy_admission=.95,new_encoder_inputs=0,original_gates_unchanged=True,acceptance_data_used=False,ad_independent=False,olid_independent=False,stopping='One fixed original whole-primary score only. No ROC cutoff fitting, new training or inference. >=28/40 and zero new falseHealthy required before unchanged full gates.')
    freeze(out/'protocol.json',protocol);healthy=selection['fixed_healthy_validation_correct'];errors={};case_reports=[]
    for i in ids:
        r=rows[i];original,packet,retained=items[r['path']];fallback=packet['fallback'];prob=float(fallback[1]) if fallback is not None else 0.;now=recover_only(original,retained,packet,prob,data['manifest'])['app_replay']['kind']=='HEALTHY'
        if not r['y']:errors[r['source']]=errors.get(r['source'],0)+int(now)
        elif r['source']=='olid_tomato':healthy+=int(now)
        case_reports.append(dict(index=i,path=r['path'],source=r['source'],label=r['label'],fallback_score_present=fallback is not None,whole_healthy_probability=prob if fallback is not None else None,whole_top_class=int(np.argmax(fallback)) if fallback is not None else None,crop_healthy_probability=float(original['app_replay'].get('probabilities',[0]*9)[1]),routed_healthy=now))
    full=[]
    for r,retained in zip(before,previous):
        packet=packets[r['id']];fallback=packet['fallback'];crop=r['app_replay']['probabilities'];full.append(dict(id=r['id'],label=r['label'],whole_top_healthy=bool(fallback is not None and np.argmax(fallback)==1),whole_healthy_at95=bool(fallback is not None and fallback[1]>=.95),roi_top_healthy=bool(crop is not None and np.argmax(crop)==1),baseline_app_healthy=r['app_replay']['kind']=='HEALTHY',whole_healthy_probability=float(fallback[1]) if fallback is not None else None,roi_healthy_probability=float(crop[1]) if crop is not None else None,detected_rois=len(packet['leaves'])))
    aggregate={}
    for label in ('Healthy','Diseased'):
        subset=[r for r in full if (r['label']=='Healthy')==(label=='Healthy')];available=sum(r['whole_healthy_probability'] is not None for r in subset);aggregate[label]=dict(n=len(subset),available_fallback_scores=available,whole_classifier_accuracy_evaluable=available==len(subset),**{k:sum(r[k] for r in subset) for k in ('whole_top_healthy','whole_healthy_at95','roi_top_healthy','baseline_app_healthy')})
    result=dict(protocol_sha256=digest(out/'protocol.json'),healthy_validation_correct=healthy,new_false_healthy_by_source=errors,cheap_pass=False,invalid_as_whole_primary_model_test=True,reason='phase76 computes fallback only when no ROI exists. Missing fallback values cannot be counted as wrong whole-primary classifications. Actual whole-primary measurement is phase236.',full_consumed_olid_diagnostics=aggregate,validation_cases=case_reports,full_cases=full,model_promoted=False,new_encoder_inputs=0,acceptance_data_used=False,ad_independent=False,olid_independent=False);(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('validation_cases','full_cases')}))
    if result['cheap_pass']:freeze(out/'candidate.json',dict(protocol_sha256=digest(out/'protocol.json'),full_qualification_required=True,model_promoted=False))

if __name__=='__main__':main()
