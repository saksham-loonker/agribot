"""Fixed negative audits from new learners, with the prior recovery rule retained."""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.special import expit
from opened_context_data import read
from permissive_encoder import digest
from population_siglip_tail import freeze

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);p=Path(parser.parse_args().campaign);out=p/'phase239_auxiliary_encoder_policy';out.mkdir(exist_ok=True);started=time.monotonic();coverage=p/'phase201_two_snapshot_coverage';cp=read(coverage/'result.json');assert digest(coverage/'candidate_bank.npz')==cp['candidate_bank_sha256'];old=p/'phase219_fixed_recovery_policy';oldp=read(old/'protocol.json');oldr=read(old/'result.json')
    expert_paths={'phase238_auxiliary_encoder_adaptation/'+mode:p/f'phase238_auxiliary_encoder_adaptation/{mode}/predictions.npz' for mode in ('frozen','lora','full_suffix')}
    source_paths={name:p/f'phase218_meta_domain_generalization/{name}/best_validation_predictions.npz' for name in ('episodic_erm','mldg')};source_paths.update({name:p/f'phase213_targeted_negative_heads/{name}/best_validation_predictions.npz' for name in ('nitrogen_vs_healthy','other_vs_healthy')})
    protocol=dict(source_sha256=digest(__file__),candidate_bank_sha256=cp['candidate_bank_sha256'],prior_policy_protocol_sha256=digest(old/'protocol.json'),prior_policy_result_sha256=digest(old/'result.json'),expert_prediction_sha256={name:digest(path) for name,path in expert_paths.items()},main_admission=.95,negative_thresholds=[.05,.5],targeted_modes=['off','both_MAP'],fixed_recovery='Exact prior phase219 best rule OR-ed after the new parent veto; no new positive-threshold fit',auxiliary_scope='Exactly three matched frozen/LoRA/full-suffix auxiliary-aware models. Frozen final40epoch endpoints and training-only deployment-scope calibration; no extra snapshots.',diagnostics='Optimistic min-crop zero-FP ranking analysis only; oracle cutoffs cannot become candidates',new_encoder_inputs=0,original_gates_unchanged=True,existing_healthy_and_named_disease_exact=True,ad_independent=False,olid_independent=False,acceptance_data_used=False,stopping='Exactly three matched encoder parameterizations, two standard negative thresholds, targeted off/on and eighteen existing parents (216 policies). No additional source/snapshot/threshold grid. Full original qualification only after >=28/40 Healthy and zero new false Healthy.')
    freeze(out/'protocol.json',protocol)
    if (out/'result.json').exists():assert read(out/'result.json')['protocol_sha256']==digest(out/'protocol.json');return
    selection=read(p/'phase118_conditional_recovery/protocol.json');rows=read(p/'phase84_population_head/fitting_rows.json');bank=np.load(coverage/'candidate_bank.npz');ids=bank['indices'];assert np.array_equal(ids,selection['validation_indices']);lookup={name:i for i,name in enumerate(bank['native_names'].tolist())};admissible=bank['admissible'];scores={}
    for name,path in source_paths.items():
        assert digest(path)==oldp['prediction_sha256'][name];z=np.load(path);assert np.array_equal(z['indices'],ids);scores[name]=z['probability']
    targeted=(scores['nitrogen_vs_healthy']<.5)|(scores['other_vs_healthy']<.5);density=p/'phase215_shrinkage_density/mobile/shared/predictions.npz';assert digest(density)==oldp['density_prediction_sha256'];z=np.load(density);assert np.array_equal(z['indices'],ids);state=np.load(density.with_name('model.npz'),allow_pickle=False);a,b=state['calibration'];density_veto=expit(a*z['validation_margin']+b).max(1)<=.5;best=oldr['best'];parent=(bank['probability'][lookup[best['first']]]>=.95)|(bank['probability'][lookup[best['second']]]>=.95);conditions={'none':np.zeros(len(ids),bool),'nitrogen':scores['nitrogen_vs_healthy']<.5,'other':scores['other_vs_healthy']<.5,'both':targeted};rescue=(np.maximum.reduce([scores[name] for name in oldp['recovery_experts']])>=.95) if best['recovery_mode']!='off' else np.zeros(len(ids),bool);fallback=((parent&~density_veto&~conditions[best['condition_mode']])|rescue)&admissible
    positive=admissible&np.array([rows[i]['y'] and rows[i]['source']=='olid_tomato' for i in ids]);negative=admissible&np.array([not rows[i]['y'] for i in ids]);other=admissible&np.array([rows[i]['y'] and rows[i]['source']!='olid_tomato' for i in ids])
    def metrics(admitted):
        healthy=selection['fixed_healthy_validation_correct']+int((admitted&positive).sum());errors={s:int(sum(admitted[k] and negative[k] for k,i in enumerate(ids) if rows[i]['source']==s)) for s in sorted({rows[i]['source'] for i in ids if not rows[i]['y']})}
        return dict(healthy_validation_correct=healthy,new_false_healthy_by_source=errors,old_healthy_correct=int((admitted&other).sum()),cheap_pass=healthy>=28 and sum(errors.values())==0)
    baseline=metrics(fallback);assert baseline['healthy_validation_correct']==best['healthy_validation_correct'] and baseline['new_false_healthy_by_source']==best['new_false_healthy_by_source'];reports=[];diagnostics=[];union=np.zeros(len(ids),bool)
    for name,path in expert_paths.items():
        z=np.load(path);assert np.array_equal(z['indices'],ids)
        raw=z['validation_margin'].astype(np.float64)
        checkpoint=path.parent/'full/last.pt';assert digest(checkpoint)==str(z['model_sha256']);a,b=z['calibration']
        view_probability=expit(a*raw+b);ranking=raw.min(1)
        assert np.max(np.abs(view_probability.min(1)-z['probability']))<1e-6;oracle=admissible&(ranking>ranking[negative].max());union|=oracle;diagnostics.append(dict(expert=name,positive_branch_prohibited='nutrition_veto_only' in name,oracle_zero_fp_healthy=selection['fixed_healthy_validation_correct']+int((oracle&positive).sum())))
        for threshold in protocol['negative_thresholds']:
            veto=view_probability.max(1)<=threshold
            for mode in protocol['targeted_modes']:
                for pair in cp['controls']:
                    parent=(bank['probability'][lookup[pair['first']]]>=.95)|(bank['probability'][lookup[pair['second']]]>=.95);admitted=((parent&~veto&~(targeted if mode=='both_MAP' else np.zeros_like(targeted)))|fallback)&admissible
                    reports.append(dict(expert=name,threshold=threshold,targeted_mode=mode,first=pair['first'],second=pair['second'],**metrics(admitted)))
    best_new=max(reports,key=lambda r:(r['cheap_pass'],-sum(r['new_false_healthy_by_source'].values()),r['healthy_validation_correct'],r['old_healthy_correct']))
    if best_new['cheap_pass']:freeze(out/'candidate.json',dict(**best_new,protocol_sha256=digest(out/'protocol.json'),full_qualification_required=True,model_promoted=False))
    result=dict(protocol_sha256=digest(out/'protocol.json'),baseline=baseline,best=best_new,controls=reports,policies_evaluated=len(reports),passing_policies=sum(r['cheap_pass'] for r in reports),ranking_diagnostics=diagnostics,oracle_zero_fp_union_healthy=selection['fixed_healthy_validation_correct']+int((union&positive).sum()),oracle_diagnostic_only=True,new_encoder_inputs=0,model_promoted=False,acceptance_data_used=False,elapsed_seconds=time.monotonic()-started);(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='controls'}),flush=True)

if __name__=='__main__':main()
