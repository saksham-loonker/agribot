"""Replay valid cached features after phase204's validation-index shadowing error."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from leaf_feature_expert import LeafFeatureHead
from opened_context_data import read
from permissive_encoder import digest
from population_siglip_tail import freeze


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);p=Path(parser.parse_args().campaign);out=p/'phase205_map_policy_replay';out.mkdir(exist_ok=True);started=time.monotonic();source=p/'phase204_map_admission_veto';sp=read(source/'protocol.json');coverage=p/'phase201_two_snapshot_coverage';cr=read(coverage/'result.json');bank_file=coverage/'candidate_bank.npz';assert digest(bank_file)==cr['candidate_bank_sha256']
    cache=np.load(bank_file);validation_ids=cache['indices'];bank=cache['probability'];admissible=cache['admissible'];names=cache['native_names'].tolist();name_index={name:i for i,name in enumerate(names)};selection=read(p/'phase118_conditional_recovery/protocol.json');assert np.array_equal(validation_ids,selection['validation_indices']);assert bank.shape==(len(names),len(validation_ids))
    local=np.load(source/'features.npy');assert local.shape==(len(sp['input_metadata']),1408) and np.isfinite(local).all();rows=read(p/'phase84_population_head/fitting_rows.json');parent=p/'phase193_exhaustive_mobile_replay';pp=read(parent/'protocol.json');assert digest(parent/'protocol.json')==sp['parent_protocol_sha256']
    protocol=dict(source_sha256=digest(__file__),feature_protocol_sha256=digest(source/'protocol.json'),features_sha256=digest(source/'features.npy'),coverage_protocol_sha256=digest(coverage/'protocol.json'),candidate_bank_sha256=digest(bank_file),validation_ids=validation_ids.tolist(),parent_protocol_sha256=digest(parent/'protocol.json'),checkpoints=sp['checkpoint_sha256'],
        repairs='Phase204 overwrote validation IDs with encoding batch indices; its prediction NPZ is invalid and must never be used. The independently identity-checked pixel features remain valid and are reused here. Canonical selection IDs are restored from phase201 and asserted against phase118.',
        inference_rule='Exactly two native snapshots OR fixed four-head mobile OR, veto only NEW Healthy admissions when all crop/orientation predictions are non-Healthy at the fixed binary MAP boundary; test SFT alone and unanimous SFT/ranking only',healthy_admission=.95,negative_veto=.5,original_healthy_and_named_disease_exact=True,original_gates_unchanged=True,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    freeze(out/'protocol.json',protocol)
    if (out/'result.json').exists():assert read(out/'result.json')['protocol_sha256']==digest(out/'protocol.json');return
    torch.set_num_threads(4);tensor=torch.from_numpy(local).cuda();scores={}
    for method in pp['methods']:
        checkpoint=parent/method/'best.pt';assert digest(checkpoint)==sp['checkpoint_sha256'][method];saved=torch.load(checkpoint,weights_only=True,map_location='cpu');assert saved['protocol_sha256']==digest(parent/'protocol.json');model=LeafFeatureHead('mlp',1408).cuda().eval();model.load_state_dict(saved['model'],strict=True)
        with torch.inference_mode():logits=(torch.cat([model(part) for part in tensor.split(256)]).ravel().cpu().numpy()*pp['calibration'][0]+pp['calibration'][1]) if len(tensor) else np.empty(0)
        scores[method]={path:max(float(logits[keys].mean()) for keys in views) for path,views in sp['bags'].items()}
    raw_sft=np.array([scores['field_source_sft'].get(rows[i]['path'],float('inf')) for i in validation_ids]);raw_rank=np.array([scores['field_source_rank'].get(rows[i]['path'],float('inf')) for i in validation_ids]);assert raw_sft.shape==raw_rank.shape==validation_ids.shape
    veto_modes={'sft_only':raw_sft<=0.,'unanimous_sft_rank':np.maximum(raw_sft,raw_rank)<=0.};positive=admissible&np.array([rows[i]['y'] and rows[i]['source']=='olid_tomato' for i in validation_ids]);negative=admissible&np.array([not rows[i]['y'] for i in validation_ids]);other_positive=admissible&np.array([rows[i]['y'] and rows[i]['source']!='olid_tomato' for i in validation_ids]);reports=[]
    np.savez(out/'veto_predictions.npz',indices=validation_ids,sft_max_all_orientation_healthy_logit=raw_sft,rank_max_all_orientation_healthy_logit=raw_rank,protocol_sha256=digest(out/'protocol.json'))
    for mode,veto in veto_modes.items():
        for pair in cr['controls']:
            admitted=((bank[name_index[pair['first']]]>=.95)|(bank[name_index[pair['second']]]>=.95))&admissible&~veto;healthy=selection['fixed_healthy_validation_correct']+int((admitted&positive).sum());errors={origin:int(sum(admitted[k] and negative[k] for k,i in enumerate(validation_ids) if rows[i]['source']==origin)) for origin in sorted({rows[i]['source'] for i in validation_ids if not rows[i]['y']})}
            reports.append(dict(mode=mode,first=pair['first'],second=pair['second'],healthy_validation_correct=healthy,new_false_healthy_by_source=errors,old_rejected_healthy_recovered=int((admitted&other_positive).sum()),cheap_pass=healthy>=28 and sum(errors.values())==0))
    best=max(reports,key=lambda r:(r['cheap_pass'],-sum(r['new_false_healthy_by_source'].values()),r['healthy_validation_correct'],r['old_rejected_healthy_recovered']))
    if best['cheap_pass']:
        native={}
        for key in ('first','second'):
            prediction=p/best[key];checkpoint=prediction.with_name(prediction.name.replace('_predictions.npz','.pt'));assert digest(checkpoint)==str(np.load(prediction)['checkpoint_sha256']);native[key]=dict(prediction=best[key],checkpoint=str(checkpoint.relative_to(p)),checkpoint_sha256=digest(checkpoint))
        freeze(out/'candidate.json',dict(**best,native_models=native,mobile_prediction_sha256=read(p/'phase200_cached_admission_policy_preflight/protocol.json')['mobile_prediction_sha256'],negative_checkpoints=sp['checkpoint_sha256'],protocol_sha256=digest(out/'protocol.json'),original_gates_unchanged=True,model_promoted=False,acceptance_data_used=False))
    result=dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),best=best,controls=reports,policies_evaluated=len(reports),passing_policies=sum(r['cheap_pass'] for r in reports),new_encoder_inputs=0,cached_encoder_inputs_reused=len(local),invalid_phase204_prediction_file_excluded=True,model_promoted=False,ad_independent=False,olid_independent=False,acceptance_data_used=False,elapsed_seconds=time.monotonic()-started)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='controls'}),flush=True)


if __name__=='__main__':main()
