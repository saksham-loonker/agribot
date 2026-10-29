"""Fixed negative audits from new learners, with the prior recovery rule retained."""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from clinical_replay_balance import clinical_replay_weights
from opened_context_data import read
from permissive_encoder import digest
from population_siglip_tail import freeze

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);p=Path(parser.parse_args().campaign);out=p/'phase248_target_scope_calibration';out.mkdir(exist_ok=True);started=time.monotonic();coverage=p/'phase201_two_snapshot_coverage';cp=read(coverage/'result.json');assert digest(coverage/'candidate_bank.npz')==cp['candidate_bank_sha256'];old=p/'phase219_fixed_recovery_policy';oldp=read(old/'protocol.json');oldr=read(old/'result.json')
    visual=p/'phase240_local_visual_veto';visual_complete=read(visual/'complete.json');assert visual_complete['protocol_sha256']==digest(visual/'protocol.json') and visual_complete['predictions_sha256']==digest(visual/'predictions.json');predictions=read(visual/'predictions.json')
    source_paths={name:p/f'phase218_meta_domain_generalization/{name}/best_validation_predictions.npz' for name in ('episodic_erm','mldg')};source_paths.update({name:p/f'phase213_targeted_negative_heads/{name}/best_validation_predictions.npz' for name in ('nitrogen_vs_healthy','other_vs_healthy')})
    protocol=dict(source_sha256=digest(__file__),candidate_bank_sha256=cp['candidate_bank_sha256'],prior_policy_protocol_sha256=digest(old/'protocol.json'),prior_policy_result_sha256=digest(old/'result.json'),visual_complete_sha256=digest(visual/'complete.json'),visual_predictions_sha256=digest(visual/'predictions.json'),parent_result_sha256={name:digest(p/name/'result.json') for name in ['phase243_all_photo_context_supervision','phase245_all_photo_nonlinear_supervision','phase247_all_photo_suffix_adaptation']},main_admission=.95,methods=['linear','rbf','binary','clinical_auxiliary','frozen','full_suffix'],vetoes=['none','not_clearly_healthy','visible_stress_only'],targeted_modes=['off','both_MAP'],fixed_recovery='Exact prior phase219 best rule OR-ed after the new parent veto; no new positive-threshold fit',visual_scope='Fixed local Gemma4 predictions are negative-only and not clinical labels. No confidence scores or new positive Healthy branch.',diagnostics='One target-scope calibration per fixed expert: fit Platt C1 only on original OLID training grouped-OOF margins, clinical/family balanced. No validation calibration. Other sources retain veto/falseHealthy checks. Fixed .95 and existing categorical/clinical negative vetoes. Oracle ranking ceilings are diagnostic only, never candidate thresholds.',new_encoder_inputs=0,original_gates_unchanged=True,existing_healthy_and_named_disease_exact=True,ad_independent=False,olid_independent=False,acceptance_data_used=False,stopping='Exactly six fixed experts, one target-scope OOF calibration each, three vetoes and targeted off/on (36policies). No thresholds/epochs/rates grid. Require28/40 and zero new falseHealthy. Stop this whole-photo family on failure; no full inference or export.')
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
    diagnostics=[]
    for method in protocol['methods']:
        phase='phase243_all_photo_context_supervision' if method in ('linear','rbf') else ('phase245_all_photo_nonlinear_supervision' if method in ('binary','clinical_auxiliary') else 'phase247_all_photo_suffix_adaptation')
        root=p/phase/method;z=np.load(root/'predictions.npz');assert np.array_equal(z['indices'],ids);parent_protocol=read(p/phase/'protocol.json')
        parent_result=read(p/phase/'result.json');entry=next(r for r in parent_result['controls'] if r['method']==method)
        model_path=root/('model.npz' if method in ('linear','rbf') else 'full/last.pt');assert digest(model_path)==entry['model_sha256']
        train=z['oof_indices'].tolist() if 'oof_indices' in z.files else parent_protocol['train_indices'];oof=z['oof_training_margin'];assert len(oof)==len(train)
        selected=[j for j,i in enumerate(train) if rows[i]['source']=='olid_tomato'];training_ids=[train[j] for j in selected];assert not any(rows[i]['validation'] for i in training_ids)
        weights=clinical_replay_weights(rows,training_ids)*len(selected);cal=LogisticRegression(C=1.,max_iter=2000).fit(oof[selected,None],np.array([rows[i]['y'] for i in training_ids]),sample_weight=weights);a=float(cal.coef_[0,0]);b=float(cal.intercept_[0]);raw=z['validation_margin'].astype(np.float64);probability=expit(a*raw+b) if a>0 else np.zeros(len(ids));parent=probability>=.95
        folder=out/method;folder.mkdir(exist_ok=True);np.savez(folder/'calibration.npz',coef=np.array([a,b]),indices=ids,probability=probability,training_indices=training_ids,source_predictions_sha256=digest(root/'predictions.npz'),source_model_sha256=entry['model_sha256'])
        ceiling=admissible&(raw>raw[negative].max());diagnostics.append(dict(method=method,calibration_slope=a,oracle_zero_fp_healthy=selection['fixed_healthy_validation_correct']+int((ceiling&positive).sum()),oracle_zero_fp_with_incumbent=metrics(ceiling|fallback)['healthy_validation_correct'],oracle_diagnostic_only=True))
        for name in protocol['vetoes']:
            veto=np.zeros(len(ids),bool) if name=='none' else (((labels!='Healthy')|~details) if name=='not_clearly_healthy' else labels=='VisibleStress')
            for mode in protocol['targeted_modes']:
                admitted=((parent&~veto&~(targeted if mode=='both_MAP' else np.zeros_like(targeted)))|fallback)&admissible
                reports.append(dict(method=method,visual_veto=name,targeted_mode=mode,**metrics(admitted)))
    best_new=max(reports,key=lambda r:(r['cheap_pass'],-sum(r['new_false_healthy_by_source'].values()),r['healthy_validation_correct'],r['old_healthy_correct']))
    if best_new['cheap_pass']:freeze(out/'candidate.json',dict(**best_new,protocol_sha256=digest(out/'protocol.json'),full_qualification_required=True,model_promoted=False))
    result=dict(protocol_sha256=digest(out/'protocol.json'),baseline=baseline,best=best_new,controls=reports,policies_evaluated=len(reports),passing_policies=sum(r['cheap_pass'] for r in reports),visual_label_counts={str(k):int((labels==k).sum()) for k in np.unique(labels)},positive_visual_branch=False,ranking_diagnostics=diagnostics,new_encoder_inputs=0,model_promoted=False,acceptance_data_used=False,elapsed_seconds=time.monotonic()-started);(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='controls'}),flush=True)

if __name__=='__main__':main()
