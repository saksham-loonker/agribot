"""Cheap coverage preflight for a fixed OR-admission / negative-veto family."""
from pathlib import Path
import argparse,json,time
import numpy as np
from conditional_recovery_policy import recover_only
from opened_context_data import load,read
from phase85_population_sft import evaluation_data
from permissive_encoder import digest
from population_siglip_tail import freeze


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);p=Path(parser.parse_args().campaign);out=p/'phase200_cached_admission_policy_preflight';out.mkdir(exist_ok=True);started=time.monotonic()
    selection=read(p/'phase118_conditional_recovery/protocol.json');ids=np.array(selection['validation_indices']);rows=read(p/'phase84_population_head/fitting_rows.json');mobile_files=[p/phase/method/'best_validation_predictions.npz' for phase in ('phase193_exhaustive_mobile_replay','phase196_rotation_mobile_replay') for method in ('field_source_sft','field_source_rank')]
    def prediction(path,checkpoint=None):
        z=np.load(path);assert np.array_equal(z['indices'],ids);prob=z['probability'];assert prob.shape==ids.shape and np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
        checkpoint=checkpoint or path.with_name('best.pt');assert checkpoint.is_file() and str(z['checkpoint_sha256'])==digest(checkpoint)
        return prob
    mobile=np.stack([prediction(path) for path in mobile_files]).max(0);files=[]
    for phase in ('phase183_native_convergence','phase191_adapted_patch_supervision','phase199_full_patch_supervision'):
        assert (p/phase/'result.json').exists()
        for path in sorted((p/phase).glob('epoch_*_predictions.npz')):
            checkpoint=path.with_name(path.name.replace('_predictions.npz','.pt'))
            if checkpoint.is_file():files.append((path,checkpoint))
    protocol=dict(source_sha256=digest(__file__),selection_sha256=digest(p/'phase118_conditional_recovery/protocol.json'),mobile_prediction_sha256={str(path.relative_to(p)):digest(path) for path in mobile_files},native_prediction_sha256={str(path.relative_to(p)):digest(path) for path,_ in files},
        family='Fixed max of four selected full-replay mobile heads and one saved native checkpoint; optional unanimous .05 negative veto across both crop views and all four orientations from the full-replay mobile SFT/ranking experts applies ONLY to new Healthy admissions',healthy_admission=.95,negative_veto=.05,existing_healthy_and_named_disease_exact=True,original_gates_unchanged=True,
        stopping='If no saved native/mobile OR combination can reach 28/40 even before veto, stop this declared family without new encoder inference. Otherwise evaluate the fixed negative-veto alternatives on cached/missing exact identities before any full restored-model check.',new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    freeze(out/'protocol.json',protocol)
    if (out/'result.json').exists():assert read(out/'result.json')['protocol_sha256']==digest(out/'protocol.json');return
    data=load(p);entries,_,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({r['path']:(r,packets[r['id']],retained) for r,retained in zip(before,previous)})
    reports=[]
    for path,checkpoint in files:
        probability=np.maximum(mobile,prediction(path,checkpoint));healthy=selection['fixed_healthy_validation_correct'];errors={}
        for i,score in zip(ids,probability):
            row=rows[i];original,packet,retained=items[row['path']];now=recover_only(original,retained,packet,float(score),data['manifest'])['app_replay']['kind']=='HEALTHY'
            if row['y'] and row['source']=='olid_tomato':healthy+=int(now)
            elif not row['y']:errors[row['source']]=errors.get(row['source'],0)+int(now)
        reports.append(dict(prediction=str(path.relative_to(p)),checkpoint=str(checkpoint.relative_to(p)),checkpoint_sha256=digest(checkpoint),healthy_before_veto=healthy,new_false_healthy_before_veto=errors,coverage_can_pass=healthy>=28))
    possible=[r for r in reports if r['coverage_can_pass']];result=dict(protocol_sha256=digest(out/'protocol.json'),saved_native_branches=len(files),mobile_branches=len(mobile_files),maximum_or_healthy=max(r['healthy_before_veto'] for r in reports),branches_with_sufficient_coverage=len(possible),controls=reports,
        next_action='Evaluate fixed rotation-consistent veto on these coverage-capable branches' if possible else 'Stop this OR/veto family: a veto can only lower already insufficient Healthy coverage',model_promoted=False,new_encoder_inputs=0,acceptance_data_used=False,ad_independent=False,olid_independent=False,elapsed_seconds=time.monotonic()-started)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='controls'}),flush=True)


if __name__=='__main__':main()
