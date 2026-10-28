"""Unclipped frozen-head score audit; oracle cutoffs are diagnostics only."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from leaf_feature_expert import LeafFeatureHead
from conditional_recovery_policy import recover_only
from opened_context_data import load,read
from phase114_new_supervision_features import original_features
from phase85_population_sft import evaluation_data
from permissive_encoder import digest
from population_siglip_tail import freeze


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);p=Path(parser.parse_args().campaign);out=p/'phase207_unclipped_score_audit';out.mkdir(exist_ok=True);started=time.monotonic();selection=read(p/'phase118_conditional_recovery/protocol.json');validation_ids=np.array(selection['validation_indices']);rows=read(p/'phase84_population_head/fitting_rows.json');files=[]
    for phase in ('phase179_full_field_replay','phase193_exhaustive_mobile_replay','phase196_rotation_mobile_replay'):
        root=p/phase;assert read(root/'protocol.json')['training_mass']['old107']==0.
        for path in sorted(root.rglob('*.pt')):
            if path.name not in ('last.pt',):files.append((path,root,read(root/'protocol.json')['calibration']))
    protocol=dict(source_sha256=digest(__file__),selection_sha256=digest(p/'phase118_conditional_recovery/protocol.json'),checkpoints={str(path.relative_to(p)):digest(path) for path,_,_ in files},
        hypothesis='Stored float32 sigmoid probabilities can lose ordering at saturation. Recompute only the inexpensive classifier heads on existing exact frozen features, retain unclipped view-mean logits, and measure whether any monotonic calibration/OR family warrants held-model calibration work.',
        diagnostic_only=True,oracle_thresholds_must_not_become_candidate=True,validation_cutoffs_not_fitted_for_deployment=True,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    freeze(out/'protocol.json',protocol)
    if (out/'result.json').exists():assert read(out/'result.json')['protocol_sha256']==digest(out/'protocol.json');return
    values,bags,_,parents=original_features(p);needed=sorted({key for i in validation_ids for view in bags[rows[i]['path']].values() for key in view});lookup={key:i for i,key in enumerate(needed)};features=torch.from_numpy(values[needed]).cuda();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    data=load(p);entries,_,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({r['path']:(r,packets[r['id']],retained) for r,retained in zip(before,previous)})
    admissible=[]
    for i in validation_ids:
        original,packet,retained=items[rows[i]['path']];admissible.append(recover_only(original,retained,packet,1.,data['manifest'])['app_replay']['kind']=='HEALTHY')
    admissible=np.array(admissible);positive=admissible&np.array([rows[i]['y'] and rows[i]['source']=='olid_tomato' for i in validation_ids]);negative=admissible&np.array([not rows[i]['y'] for i in validation_ids]);union=np.zeros(len(validation_ids),bool);reports=[]
    for checkpoint,root,calibration in files:
        saved=torch.load(checkpoint,weights_only=True,map_location='cpu');assert saved['protocol_sha256']==digest(root/'protocol.json');model=LeafFeatureHead('mlp',1408).cuda().eval();model.load_state_dict(saved['model'],strict=True)
        with torch.inference_mode():raw=(model(features).ravel()*calibration[0]+calibration[1]).cpu().numpy()
        margin=np.array([min(float(raw[[lookup[key] for key in bags[rows[i]['path']][view]]].mean()) for view in ('center','full')) for i in validation_ids]);probability=1/(1+np.exp(-np.clip(margin,-80,80)));prediction=checkpoint.with_name('best_validation_predictions.npz' if checkpoint.name=='best.pt' else checkpoint.stem+'_predictions.npz')
        if prediction.exists():
            old=np.load(prediction);assert np.array_equal(old['indices'],validation_ids) and str(old['checkpoint_sha256'])==digest(checkpoint);assert np.max(np.abs(old['probability']-probability))<.00001;assert np.array_equal(old['probability']>=.95,probability>=.95)
        maximum_negative=float(margin[negative].max());oracle=admissible&(margin>maximum_negative);union|=oracle;name=str(checkpoint.relative_to(p));filename=out/(name.replace('/','__').replace('.pt','.npz'));np.savez(filename,indices=validation_ids,margin=margin,probability=probability,checkpoint_sha256=digest(checkpoint),protocol_sha256=digest(out/'protocol.json'))
        reports.append(dict(checkpoint=name,checkpoint_sha256=digest(checkpoint),cache_file=filename.name,oracle_zero_fp_healthy=selection['fixed_healthy_validation_correct']+int((oracle&positive).sum()),oracle_maximum_negative_logit=maximum_negative,healthy_at_fixed_admission=selection['fixed_healthy_validation_correct']+int(((probability>=.95)&positive).sum())))
    reports.sort(key=lambda r:r['oracle_zero_fp_healthy'],reverse=True);result=dict(protocol_sha256=digest(out/'protocol.json'),heads_evaluated=len(files),maximum_single_oracle_healthy=reports[0]['oracle_zero_fp_healthy'],oracle_zero_fp_union_healthy=selection['fixed_healthy_validation_correct']+int((union&positive).sum()),controls=reports,diagnostic_only=True,candidate_selected=False,model_promoted=False,new_encoder_inputs=0,acceptance_data_used=False,elapsed_seconds=time.monotonic()-started)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='controls'}),flush=True)


if __name__=='__main__':main()
