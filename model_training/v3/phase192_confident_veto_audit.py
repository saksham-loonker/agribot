"""Test whether a learned veto can correct false Healthy without relaxing gates.

This audits an additional routing family, not a change to recover_only. A
strong negative must agree in BOTH existing crop views. No threshold fitting,
source-membership routing, clinical relabeling or promotion is performed.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from leaf_feature_expert import LeafFeatureHead
from opened_context_data import load,read
from phase85_population_sft import evaluation_data
from phase114_new_supervision_features import original_features
from permissive_encoder import digest
from population_siglip_tail import freeze


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);p=Path(parser.parse_args().campaign);out=p/'phase192_confident_veto_audit';out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    parent=p/'phase179_full_field_replay';pp=read(parent/'protocol.json');methods=pp['methods'];protocol=dict(source_sha256=digest(__file__),parent_protocol_sha256=digest(parent/'protocol.json'),checkpoints={name:digest(parent/name/'best.pt') for name in methods},
        hypothesis='Recovery-only routing freezes existing false Healthy as well as correct Healthy. Audit a separate data-independent veto family: reject an existing Healthy only when both crop views agree with <=.05 model probability of Healthy. All original exact-retention gates stay unchanged; no source or ground-truth label enters the veto score.',
        threshold=.05,both_views_must_agree=True,existing_named_disease_preserved=True,original_gates_unchanged=True,diagnostic_only=True,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    freeze(out/'protocol.json',protocol)
    values,bags,_,_=original_features(p);data=load(p);entries,records,packets,before,previous=evaluation_data(p,data)
    protected=[(r,entries[id(r)][1]) for group in list(data['base'].values())+[data['internal']] for r in group];olid=list(zip(before,previous));all_rows=protected+olid
    paths=list(dict.fromkeys(row['path'] for row,retained in all_rows if retained['app_replay']['kind']=='HEALTHY' and row['path'] in bags and bags[row['path']]['center']))
    torch.set_num_threads(4);features=torch.from_numpy(values).cuda();controls=[]
    for name in methods:
        model=LeafFeatureHead('mlp',1408).cuda().eval();saved=torch.load(parent/name/'best.pt',weights_only=True,map_location='cpu');assert saved['protocol_sha256']==digest(parent/'protocol.json');model.load_state_dict(saved['model'],strict=True)
        with torch.inference_mode():
            logits=torch.cat([model(part) for part in features.split(256)]).ravel()*pp['calibration'][0]+pp['calibration'][1];logits=logits.cpu().numpy()
        probability={path:float(1/(1+np.exp(-np.clip(max(float(logits[bags[path][view]].mean()) for view in ('center','full')),-80,80)))) for path in paths}
        veto=lambda row,retained:retained['app_replay']['kind']=='HEALTHY' and probability.get(row['path'],1.)<=.05
        protected_changes=[r['path'] for r,retained in protected if veto(r,retained)]
        corrected=[r['path'] for r,retained in olid if r['label']!='Healthy' and veto(r,retained)]
        damaged=[r['path'] for r,retained in olid if r['label']=='Healthy' and veto(r,retained)]
        folder=out/name;folder.mkdir(exist_ok=True);np.savez(folder/'predictions.npz',paths=np.array(paths),probability=np.array([probability[path] for path in paths]),checkpoint_sha256=digest(parent/name/'best.pt'),protocol_sha256=digest(out/'protocol.json'))
        result=dict(method=name,protected_original_or_internal_changes=len(protected_changes),corrected_consumed_false_healthy=len(corrected),damaged_consumed_correct_healthy=len(damaged),protected_changes=protected_changes,corrected=corrected,damaged=damaged,
            passes_preflight=not protected_changes and not damaged and bool(corrected),model_promoted=False,missing_features_keep_existing_healthy=True)
        controls.append(result);print(json.dumps({k:v for k,v in result.items() if k not in ('protected_changes','corrected','damaged')}),flush=True)
    result=dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),controls=controls,diagnostic_only=True,model_promoted=False,new_encoder_inputs=0,elapsed_seconds=time.monotonic()-started,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    (out/'result.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
