"""Bounded new-admission density vetoes and diagnostic ranking coverage."""
from pathlib import Path
import argparse,json,time
import numpy as np
from opened_context_data import read
from permissive_encoder import digest
from population_siglip_tail import freeze

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);p=Path(parser.parse_args().campaign);out=p/'phase216_density_policy';out.mkdir(exist_ok=True);started=time.monotonic()
    coverage=p/'phase201_two_snapshot_coverage';cp=read(coverage/'result.json');assert digest(coverage/'candidate_bank.npz')==cp['candidate_bank_sha256'];paths={f'{kind}/{mode}':p/'phase215_shrinkage_density'/kind/mode/'predictions.npz' for kind in ('mobile','native') for mode in ('shared','class_shrinkage')}
    protocol=dict(source_sha256=digest(__file__),candidate_bank_sha256=cp['candidate_bank_sha256'],prediction_sha256={name:digest(path) for name,path in paths.items()},density_protocol_sha256={kind:digest(p/'phase215_shrinkage_density'/kind/'protocol.json') for kind in ('mobile','native')},main_admission=.95,negative_veto_thresholds=[.05,.5],negative_evidence='Both original crop means must have Healthy probability at or below the fixed negative threshold; veto only new admissions',diagnostic='Unclipped min-crop density ranking oracle is optimistic consumed-data analysis only; its cutoffs cannot become candidates',original_gates_unchanged=True,ad_independent=False,olid_independent=False,acceptance_data_used=False,new_encoder_inputs=0,stopping='Exactly four cached density experts, two standard negative thresholds and eighteen existing coverage parents. No validation threshold fitting or new encoder work; full qualification only after original cheap screen passes.')
    freeze(out/'protocol.json',protocol)
    if (out/'result.json').exists():assert read(out/'result.json')['protocol_sha256']==digest(out/'protocol.json');return
    selection=read(p/'phase118_conditional_recovery/protocol.json');rows=read(p/'phase84_population_head/fitting_rows.json');bank=np.load(coverage/'candidate_bank.npz');ids=bank['indices'];assert np.array_equal(ids,selection['validation_indices']);lookup={name:i for i,name in enumerate(bank['native_names'].tolist())};admissible=bank['admissible'];positive=admissible&np.array([rows[i]['y'] and rows[i]['source']=='olid_tomato' for i in ids]);negative=admissible&np.array([not rows[i]['y'] for i in ids]);other=admissible&np.array([rows[i]['y'] and rows[i]['source']!='olid_tomato' for i in ids]);reports=[];oracle_union=np.zeros(len(ids),bool);diagnostics=[]
    for name,path in paths.items():
        z=np.load(path);assert np.array_equal(z['indices'],ids);state=np.load(path.with_name('model.npz'),allow_pickle=False);assert digest(path.with_name('model.npz'))==str(z['model_sha256']);a,b=state['calibration'];raw=z['validation_margin'];probability=1/(1+np.exp(-np.clip(a*raw+b,-80,80)));assert np.max(np.abs(probability.min(1)-z['probability']))<1e-12
        ranking=raw.min(1);cutoff=float(ranking[negative].max());safe_oracle=admissible&(ranking>cutoff);oracle_union|=safe_oracle;diagnostics.append(dict(expert=name,oracle_zero_fp_healthy=selection['fixed_healthy_validation_correct']+int((safe_oracle&positive).sum())))
        for threshold in protocol['negative_veto_thresholds']:
            veto=probability.max(1)<=threshold
            for pair in cp['controls']:
                admitted=((bank['probability'][lookup[pair['first']]]>=.95)|(bank['probability'][lookup[pair['second']]]>=.95))&admissible&~veto;healthy=selection['fixed_healthy_validation_correct']+int((admitted&positive).sum());errors={source:int(sum(admitted[k] and negative[k] for k,i in enumerate(ids) if rows[i]['source']==source)) for source in sorted({rows[i]['source'] for i in ids if not rows[i]['y']})}
                reports.append(dict(expert=name,threshold=threshold,first=pair['first'],second=pair['second'],healthy_validation_correct=healthy,new_false_healthy_by_source=errors,old_healthy_correct=int((admitted&other).sum()),cheap_pass=healthy>=28 and sum(errors.values())==0))
    best=max(reports,key=lambda r:(r['cheap_pass'],-sum(r['new_false_healthy_by_source'].values()),r['healthy_validation_correct'],r['old_healthy_correct']))
    if best['cheap_pass']:freeze(out/'candidate.json',dict(**best,protocol_sha256=digest(out/'protocol.json'),full_qualification_required=True,model_promoted=False))
    result=dict(protocol_sha256=digest(out/'protocol.json'),best=best,controls=reports,policies_evaluated=len(reports),passing_policies=sum(r['cheap_pass'] for r in reports),ranking_diagnostics=diagnostics,oracle_zero_fp_union_healthy=selection['fixed_healthy_validation_correct']+int((oracle_union&positive).sum()),oracle_diagnostic_only=True,new_encoder_inputs=0,model_promoted=False,acceptance_data_used=False,elapsed_seconds=time.monotonic()-started)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='controls'}),flush=True)

if __name__=='__main__':main()
