"""Fixed negative audits from new learners, with the prior recovery rule retained."""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.special import expit
from opened_context_data import read
from permissive_encoder import digest
from population_siglip_tail import freeze

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);p=Path(parser.parse_args().campaign);out=p/'phase244_whole_context_recovery';out.mkdir(exist_ok=True);started=time.monotonic();coverage=p/'phase201_two_snapshot_coverage';cp=read(coverage/'result.json');assert digest(coverage/'candidate_bank.npz')==cp['candidate_bank_sha256'];old=p/'phase219_fixed_recovery_policy';oldp=read(old/'protocol.json');oldr=read(old/'result.json')
    visual=p/'phase240_local_visual_veto';visual_complete=read(visual/'complete.json');assert visual_complete['protocol_sha256']==digest(visual/'protocol.json') and visual_complete['predictions_sha256']==digest(visual/'predictions.json');predictions=read(visual/'predictions.json')
    source_paths={name:p/f'phase218_meta_domain_generalization/{name}/best_validation_predictions.npz' for name in ('episodic_erm','mldg')};source_paths.update({name:p/f'phase213_targeted_negative_heads/{name}/best_validation_predictions.npz' for name in ('nitrogen_vs_healthy','other_vs_healthy')})
    protocol=dict(source_sha256=digest(__file__),candidate_bank_sha256=cp['candidate_bank_sha256'],prior_policy_protocol_sha256=digest(old/'protocol.json'),prior_policy_result_sha256=digest(old/'result.json'),visual_complete_sha256=digest(visual/'complete.json'),visual_predictions_sha256=digest(visual/'predictions.json'),whole_context_result_sha256=digest(p/'phase243_all_photo_context_supervision/result.json'),main_admission=.95,methods=['linear','rbf'],vetoes=['none','not_clearly_healthy','visible_stress_only'],targeted_modes=['off','both_MAP'],fixed_recovery='Exact prior phase219 best rule OR-ed after the new parent veto; no new positive-threshold fit',visual_scope='Fixed local Gemma4 predictions are negative-only and not clinical labels. No confidence scores or new positive Healthy branch.',diagnostics='Whole-only supervision after adding omitted no-ROI photos. Fixed .95 probabilities plus negative-only visual/condition vetoes; no new confidence thresholds.',new_encoder_inputs=0,original_gates_unchanged=True,existing_healthy_and_named_disease_exact=True,ad_independent=False,olid_independent=False,acceptance_data_used=False,stopping='Exactly two whole-photo experts, three fixed veto options, targeted off/on (12policies). No parent/snapshot/threshold sweep. Full qualification only after >=28/40 Healthy and zero new false Healthy.')
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
    assert [r['index'] for r in predictions]==ids.tolist()
    labels=np.array([r['prediction']['label'] for r in predictions]);details=np.array([r['prediction']['leaf_detail_sufficient'] for r in predictions])
    for method in protocol['methods']:
        root=p/'phase243_all_photo_context_supervision'/method
        z=np.load(root/'predictions.npz');assert np.array_equal(z['indices'],ids)
        parent=z['probability']>=.95
        for name in protocol['vetoes']:
            veto=np.zeros(len(ids),bool) if name=='none' else (((labels!='Healthy')|~details) if name=='not_clearly_healthy' else labels=='VisibleStress')
            for mode in protocol['targeted_modes']:
                admitted=((parent&~veto&~(targeted if mode=='both_MAP' else np.zeros_like(targeted)))|fallback)&admissible
                reports.append(dict(method=method,visual_veto=name,targeted_mode=mode,**metrics(admitted)))
    best_new=max(reports,key=lambda r:(r['cheap_pass'],-sum(r['new_false_healthy_by_source'].values()),r['healthy_validation_correct'],r['old_healthy_correct']))
    if best_new['cheap_pass']:freeze(out/'candidate.json',dict(**best_new,protocol_sha256=digest(out/'protocol.json'),full_qualification_required=True,model_promoted=False))
    result=dict(protocol_sha256=digest(out/'protocol.json'),baseline=baseline,best=best_new,controls=reports,policies_evaluated=len(reports),passing_policies=sum(r['cheap_pass'] for r in reports),visual_label_counts={str(k):int((labels==k).sum()) for k in np.unique(labels)},positive_visual_branch=False,new_encoder_inputs=0,model_promoted=False,acceptance_data_used=False,elapsed_seconds=time.monotonic()-started);(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='controls'}),flush=True)

if __name__=='__main__':main()
