"""Conditional four-orientation negative agreement, with exact-retention gates."""
from pathlib import Path
from functools import lru_cache
import argparse,hashlib,json,time
import numpy as np
import torch,timm
from compact_healthy_model import primary
from leaf_feature_expert import LeafFeatureHead
from opened_context_data import load,read
from partial_dino_cache import CanonicalInputs
from phase85_population_sft import evaluation_data
from phase114_new_supervision_features import original_features
from permissive_encoder import digest
from population_siglip_tail import freeze


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);p=Path(parser.parse_args().campaign);out=p/'phase198_veto_rotation_consistency';out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    parent=p/'phase193_exhaustive_mobile_replay';pp=read(parent/'protocol.json');methods=pp['methods'];rotation=p/'phase195_rotation_feature_extension';rp=read(rotation/'protocol.json');rc=read(rotation/'complete.json');assert rc['protocol_sha256']==digest(rotation/'protocol.json') and rc['features_sha256']==digest(rotation/'features.npy')
    purpose=dict(source_sha256=digest(__file__),parent_protocol_sha256=digest(parent/'protocol.json'),rotation_complete_sha256=digest(rotation/'complete.json'),checkpoint_sha256={name:digest(parent/name/'best.pt') for name in methods},
        rule='Veto an existing Healthy only when both crop views in all four right-angle orientations have model Healthy probability <=.05; apply no rule to named disease; never use labels or source identity in scoring',threshold=.05,orientations=[0,90,180,270],original_gates_unchanged=True,clinical_labels_only_for_diagnostic_counts=True,independent_frames_claimed=False,acceptance_data_used=False)
    freeze(out/'purpose.json',purpose)
    values,bags,descriptors,_=original_features(p);data=load(p);entries,_,_,before,previous=evaluation_data(p,data);protected=[(r,entries[id(r)][1]) for group in list(data['base'].values())+[data['internal']] for r in group];olid=list(zip(before,previous));cases=protected+olid
    paths=list(dict.fromkeys(r['path'] for r,retained in cases if retained['app_replay']['kind']=='HEALTHY' and r['path'] in bags and bags[r['path']]['center']))
    torch.set_num_threads(4);features=torch.from_numpy(values).cuda();models={};potential=set()
    for name in methods:
        model=LeafFeatureHead('mlp',1408).cuda().eval();saved=torch.load(parent/name/'best.pt',weights_only=True,map_location='cpu');assert saved['protocol_sha256']==digest(parent/'protocol.json');model.load_state_dict(saved['model'],strict=True);models[name]=model
        with torch.inference_mode():logits=torch.cat([model(part) for part in features.split(256)]).ravel().cpu().numpy()*pp['calibration'][0]+pp['calibration'][1]
        potential.update(path for path in paths if max(float(logits[bags[path][view]].mean()) for view in ('center','full'))<=float(np.log(.05/.95)))
    potential=sorted(potential);donor={row['input_sha256']:values[i] for i,row in enumerate(descriptors)};rotated=np.load(rotation/'features.npy',mmap_mode='r');rotated_lookup={row['input_sha256']:i for i,row in enumerate(rp['new_inputs'])};pixels=CanonicalInputs(p)
    @lru_cache(maxsize=4)
    def raw(key):
        row=descriptors[key];value=np.ascontiguousarray(pixels(row if 'kind' in row else dict(row,kind='new',policy=row['view'])));assert hashlib.sha256(value.tobytes()).hexdigest()==row['input_sha256'];return value
    local=[];lookup={};inputs=[];work=[];pairs={};reused=0
    for path in potential:
        pairs[path]=[]
        for turn in range(4):
            for view in ('center','full'):
                ids=[]
                for key in bags[path][view]:
                    value=np.ascontiguousarray(np.rot90(raw(key),turn,axes=(1,2)));sha=hashlib.sha256(value.tobytes()).hexdigest()
                    if sha not in lookup:
                        lookup[sha]=len(local);inputs.append(dict(input_sha256=sha,parent_input_sha256=descriptors[key]['input_sha256'],parent_index=key,rotation_quarters=turn))
                        if sha in donor:local.append(donor[sha]);reused+=1
                        elif sha in rotated_lookup:local.append(rotated[rotated_lookup[sha]]);reused+=1
                        else:local.append(None);work.append(lookup[sha])
                    ids.append(lookup[sha])
                pairs[path].append(ids)
    protocol=dict(**purpose,purpose_sha256=digest(out/'purpose.json'),potential_healthy_paths=potential,input_metadata=inputs,bags=pairs,new_encoder_inputs=len(work),cached_encoder_inputs_reused=reused)
    freeze(out/'protocol.json',protocol);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;timm.layers.set_fused_attn(False)
    if work:
        _,encoder=primary(p);encoder=encoder.cuda().eval()
        with torch.inference_mode():
            audit=[row['parent_index'] for row in inputs[:16]];actual=encoder.components(torch.from_numpy(np.concatenate([raw(key) for key in audit])).cuda())[0].cpu().numpy();error=float(np.abs(actual-values[audit]).max());assert error<.002
            for start in range(0,len(work),32):
                batch=[];ids=work[start:start+32]
                for i in ids:
                    row=inputs[i];value=np.ascontiguousarray(np.rot90(raw(row['parent_index']),row['rotation_quarters'],axes=(1,2)));assert hashlib.sha256(value.tobytes()).hexdigest()==row['input_sha256'];batch.append(value)
                actual=encoder.components(torch.from_numpy(np.concatenate(batch)).cuda())[0].cpu().numpy()
                for i,value in zip(ids,actual):local[i]=value
        del encoder
    pixels.close();local=np.stack(local);np.save(out/'features.npy',local);tensor=torch.from_numpy(local).cuda();controls=[]
    for name,model in models.items():
        with torch.inference_mode():logits=torch.cat([model(part) for part in tensor.split(256)]).ravel().cpu().numpy()*pp['calibration'][0]+pp['calibration'][1]
        score={path:max(float(logits[keys].mean()) for keys in pairs[path]) for path in potential};veto=lambda r,retained:retained['app_replay']['kind']=='HEALTHY' and score.get(r['path'],float('inf'))<=float(np.log(.05/.95))
        changed=[r for r,retained in protected if veto(r,retained)];corrected=[r for r,retained in olid if r['label']!='Healthy' and veto(r,retained)];damaged=[r for r,retained in olid if r['label']=='Healthy' and veto(r,retained)]
        result=dict(method=name,protected_original_or_internal_changes=len(changed),protected_publisher_healthy_changes=sum(r['label']=='Healthy' for r in changed),corrected_consumed_false_healthy=len(corrected),damaged_consumed_correct_healthy=len(damaged),passes_preflight=not changed and not damaged and bool(corrected),model_promoted=False)
        controls.append(result);print(json.dumps(result),flush=True)
        np.savez(out/(name+'_predictions.npz'),paths=np.array(potential),maximum_all_orientation_healthy_logit=np.array([score[path] for path in potential]),checkpoint_sha256=digest(parent/name/'best.pt'),protocol_sha256=digest(out/'protocol.json'))
    result=dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),features_sha256=digest(out/'features.npy'),controls=controls,new_encoder_inputs=len(work),cached_encoder_inputs_reused=reused,diagnostic_only=True,model_promoted=False,ad_independent=False,olid_independent=False,acceptance_data_used=False,elapsed_seconds=time.monotonic()-started)
    (out/'result.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
