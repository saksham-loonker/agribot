"""A fixed high-confidence recovery branch after density/condition vetoes."""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.special import expit
from opened_context_data import read
from permissive_encoder import digest
from population_siglip_tail import freeze

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);p=Path(parser.parse_args().campaign);out=p/'phase219_fixed_recovery_policy';out.mkdir(exist_ok=True);started=time.monotonic()
    base=p/'phase201_two_snapshot_coverage';cp=read(base/'result.json');assert digest(base/'candidate_bank.npz')==cp['candidate_bank_sha256'];paths={name:p/f'phase218_meta_domain_generalization/{name}/best_validation_predictions.npz' for name in ('episodic_erm','mldg')};paths.update({name:p/f'phase213_targeted_negative_heads/{name}/best_validation_predictions.npz' for name in ('nitrogen_vs_healthy','other_vs_healthy')});density=p/'phase215_shrinkage_density/mobile/shared/predictions.npz'
    protocol=dict(source_sha256=digest(__file__),candidate_bank_sha256=cp['candidate_bank_sha256'],prediction_sha256={name:digest(path) for name,path in paths.items()},density_prediction_sha256=digest(density),density_model_sha256=digest(density.with_name('model.npz')),main_admission=.95,density_negative=.5,condition_negative=.5,condition_modes=['none','nitrogen','other','both'],recovery_modes=['off','selected_training_heads'],recovery_experts=['episodic_erm','mldg','other_vs_healthy'],recovery_admission=.95,hypothesis='The fixed density veto reduces false Healthy but removes some Healthy cases. Permit independently trained, already selected heads to recover only high-confidence new Healthy, while testing the two targeted negative heads as standard-MAP vetoes. This is a fixed branch architecture, not validation-fitted coefficients or cutoffs.',original_gates_unchanged=True,existing_healthy_and_named_disease_exact=True,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False,stopping='Exactly eighteen existing two-native parents, four declared condition veto subsets and recovery off/on. No new snapshots, thresholds, coefficients or additional stages. Stop this architecture if none passes the original >=28/40 and zero-new-false-Healthy screen.')
    freeze(out/'protocol.json',protocol)
    if (out/'result.json').exists():assert read(out/'result.json')['protocol_sha256']==digest(out/'protocol.json');return
    selection=read(p/'phase118_conditional_recovery/protocol.json');rows=read(p/'phase84_population_head/fitting_rows.json');bank=np.load(base/'candidate_bank.npz');ids=bank['indices'];assert np.array_equal(ids,selection['validation_indices']);lookup={name:i for i,name in enumerate(bank['native_names'].tolist())};admissible=bank['admissible'];scores={}
    for name,path in paths.items():
        z=np.load(path);assert np.array_equal(z['indices'],ids) and str(z['checkpoint_sha256'])==read(path.parent/'result.json')['checkpoint_sha256'];scores[name]=z['probability']
    z=np.load(density);assert np.array_equal(z['indices'],ids) and str(z['model_sha256'])==digest(density.with_name('model.npz'));state=np.load(density.with_name('model.npz'),allow_pickle=False);a,b=state['calibration'];density_veto=expit(a*z['validation_margin']+b).max(1)<=.5;nitrogen=scores['nitrogen_vs_healthy']<.5;other=scores['other_vs_healthy']<.5;conditions={'none':np.zeros(len(ids),bool),'nitrogen':nitrogen,'other':other,'both':nitrogen|other};recovery=np.maximum.reduce([scores[name] for name in protocol['recovery_experts']])>=.95;positive=admissible&np.array([rows[i]['y'] and rows[i]['source']=='olid_tomato' for i in ids]);negative=admissible&np.array([not rows[i]['y'] for i in ids]);old=admissible&np.array([rows[i]['y'] and rows[i]['source']!='olid_tomato' for i in ids]);reports=[]
    for mode,condition_veto in conditions.items():
        for rescue in protocol['recovery_modes']:
            for pair in cp['controls']:
                parent=(bank['probability'][lookup[pair['first']]]>=.95)|(bank['probability'][lookup[pair['second']]]>=.95);admitted=(parent&~density_veto&~condition_veto)|(recovery if rescue!='off' else False);admitted &= admissible;healthy=selection['fixed_healthy_validation_correct']+int((admitted&positive).sum());errors={source:int(sum(admitted[k] and negative[k] for k,i in enumerate(ids) if rows[i]['source']==source)) for source in sorted({rows[i]['source'] for i in ids if not rows[i]['y']})}
                reports.append(dict(condition_mode=mode,recovery_mode=rescue,first=pair['first'],second=pair['second'],healthy_validation_correct=healthy,new_false_healthy_by_source=errors,old_healthy_correct=int((admitted&old).sum()),cheap_pass=healthy>=28 and sum(errors.values())==0))
    best=max(reports,key=lambda r:(r['cheap_pass'],-sum(r['new_false_healthy_by_source'].values()),r['healthy_validation_correct'],r['old_healthy_correct']))
    if best['cheap_pass']:freeze(out/'candidate.json',dict(**best,protocol_sha256=digest(out/'protocol.json'),full_qualification_required=True,model_promoted=False))
    result=dict(protocol_sha256=digest(out/'protocol.json'),best=best,controls=reports,policies_evaluated=len(reports),passing_policies=sum(r['cheap_pass'] for r in reports),new_encoder_inputs=0,model_promoted=False,acceptance_data_used=False,elapsed_seconds=time.monotonic()-started);(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='controls'}),flush=True)

if __name__=='__main__':main()
