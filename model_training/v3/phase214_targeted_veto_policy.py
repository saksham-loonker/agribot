"""Fixed condition-specific vetoes on previously declared new admissions."""
from pathlib import Path
import argparse,json,time
import numpy as np
from opened_context_data import read
from permissive_encoder import digest
from population_siglip_tail import freeze

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase214_targeted_veto_policy';out.mkdir(exist_ok=True);started=time.monotonic()
    coverage=p/'phase201_two_snapshot_coverage';specialists=p/'phase213_targeted_negative_heads'
    cp=read(coverage/'result.json');assert digest(coverage/'candidate_bank.npz')==cp['candidate_bank_sha256']
    selection=read(p/'phase118_conditional_recovery/protocol.json');rows=read(p/'phase84_population_head/fitting_rows.json')
    methods=['nitrogen_vs_healthy','other_vs_healthy'];predictions={m:specialists/m/'best_validation_predictions.npz' for m in methods}
    protocol=dict(source_sha256=digest(__file__),coverage_protocol_sha256=digest(coverage/'protocol.json'),candidate_bank_sha256=cp['candidate_bank_sha256'],specialist_protocol_sha256=digest(specialists/'protocol.json'),specialist_predictions_sha256={m:digest(path) for m,path in predictions.items()},modes=['nitrogen_only','other_only','both'],thresholds=[.5,.95],main_admission=.95,
        hypothesis='Target only the observed nitrogen-deficiency and Other new-admission errors using independently trained condition-specific binary heads. New admissions must still pass the unchanged parent confidence and guards. Specialists can only veto, never admit or name a disease; preserve all original Healthy/named outputs.',
        stopping='Exactly eighteen existing coverage-capable parents, three specialist subsets and two fixed standard thresholds. No fitted validation cutoffs or new feature inference. Only a passing unchanged actual-verdict screen permits complete restored-model qualification.',original_gates_unchanged=True,ad_independent=False,olid_independent=False,acceptance_data_used=False,new_encoder_inputs=0)
    freeze(out/'protocol.json',protocol)
    if (out/'result.json').exists():assert read(out/'result.json')['protocol_sha256']==digest(out/'protocol.json');return
    bank=np.load(coverage/'candidate_bank.npz');ids=bank['indices'];assert np.array_equal(ids,selection['validation_indices']);names=bank['native_names'].tolist();lookup={name:i for i,name in enumerate(names)};admissible=bank['admissible'];scores={}
    for method,path in predictions.items():
        z=np.load(path);r=read(path.parent/'result.json');assert np.array_equal(z['indices'],ids) and str(z['checkpoint_sha256'])==r['checkpoint_sha256'];scores[method]=z['probability']
    positive=admissible&np.array([rows[i]['y'] and rows[i]['source']=='olid_tomato' for i in ids]);negative=admissible&np.array([not rows[i]['y'] for i in ids]);other_positive=admissible&np.array([rows[i]['y'] and rows[i]['source']!='olid_tomato' for i in ids]);reports=[]
    modes={'nitrogen_only':scores[methods[0]],'other_only':scores[methods[1]],'both':np.minimum(scores[methods[0]],scores[methods[1]])}
    for mode,specialist_score in modes.items():
        for threshold in protocol['thresholds']:
            for pair in cp['controls']:
                admitted=((bank['probability'][lookup[pair['first']]]>=.95)|(bank['probability'][lookup[pair['second']]]>=.95))&admissible&(specialist_score>=threshold)
                healthy=selection['fixed_healthy_validation_correct']+int((admitted&positive).sum());errors={source:int(sum(admitted[k] and negative[k] for k,i in enumerate(ids) if rows[i]['source']==source)) for source in sorted({rows[i]['source'] for i in ids if not rows[i]['y']})}
                reports.append(dict(mode=mode,threshold=threshold,first=pair['first'],second=pair['second'],healthy_validation_correct=healthy,new_false_healthy_by_source=errors,old_rejected_healthy_recovered=int((admitted&other_positive).sum()),cheap_pass=healthy>=28 and sum(errors.values())==0))
    best=max(reports,key=lambda r:(r['cheap_pass'],-sum(r['new_false_healthy_by_source'].values()),r['healthy_validation_correct'],r['old_rejected_healthy_recovered']))
    if best['cheap_pass']:freeze(out/'candidate.json',dict(**best,protocol_sha256=digest(out/'protocol.json'),model_promoted=False,full_restored_qualification_required=True))
    result=dict(protocol_sha256=digest(out/'protocol.json'),best=best,controls=reports,policies_evaluated=len(reports),passing_policies=sum(r['cheap_pass'] for r in reports),model_promoted=False,new_encoder_inputs=0,acceptance_data_used=False,elapsed_seconds=time.monotonic()-started)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='controls'}),flush=True)

if __name__=='__main__':main()
