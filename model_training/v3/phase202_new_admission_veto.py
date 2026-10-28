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
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);p=Path(parser.parse_args().campaign);out=p/'phase202_new_admission_veto';out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    parent=p/'phase193_exhaustive_mobile_replay';pp=read(parent/'protocol.json');methods=pp['methods'];rotation=p/'phase195_rotation_feature_extension';rp=read(rotation/'protocol.json');rc=read(rotation/'complete.json');assert rc['protocol_sha256']==digest(rotation/'protocol.json') and rc['features_sha256']==digest(rotation/'features.npy')
    coverage=p/'phase201_two_snapshot_coverage';coverage_result=read(coverage/'result.json');assert coverage_result['coverage_capable_pairs']>0;bank_file=coverage/'candidate_bank.npz';assert digest(bank_file)==coverage_result['candidate_bank_sha256'];bank_cache=np.load(bank_file);bank=bank_cache['probability'];ids=bank_cache['indices'];admissible=bank_cache['admissible'];native_names=bank_cache['native_names'].tolist();name_index={name:i for i,name in enumerate(native_names)};selection=read(p/'phase118_conditional_recovery/protocol.json');assert np.array_equal(ids,selection['validation_indices'])
    purpose=dict(source_sha256=digest(__file__),coverage_protocol_sha256=digest(coverage/'protocol.json'),coverage_result_sha256=digest(coverage/'result.json'),candidate_bank_sha256=digest(bank_file),parent_protocol_sha256=digest(parent/'protocol.json'),rotation_complete_sha256=digest(rotation/'complete.json'),checkpoint_sha256={name:digest(parent/name/'best.pt') for name in methods},
        rule='Two fixed negative-veto alternatives: full-replay SFT alone, or unanimous SFT/ranking. Both require <=.05 Healthy in every crop/orientation and affect only NEW Healthy admissions from the frozen two-native/four-mobile OR. Preserve every existing Healthy and named disease output exactly; no source or labels enter scoring',threshold=.05,orientations=[0,90,180,270],original_gates_unchanged=True,clinical_labels_only_for_diagnostic_counts=True,independent_frames_claimed=False,acceptance_data_used=False)
    freeze(out/'purpose.json',purpose)
    values,bags,descriptors,_=original_features(p);data=load(p);entries,_,_,before,previous=evaluation_data(p,data);protected=[(r,entries[id(r)][1]) for group in list(data['base'].values())+[data['internal']] for r in group];olid=list(zip(before,previous));cases=protected+olid
    rows=read(p/'phase84_population_head/fitting_rows.json');planned=np.zeros(len(ids),bool)
    for pair in coverage_result['controls']:planned |= ((bank[name_index[pair['first']]]>=.95)|(bank[name_index[pair['second']]]>=.95))&admissible
    paths=[rows[i]['path'] for k,i in enumerate(ids) if planned[k]]
    torch.set_num_threads(4);features=torch.from_numpy(values).cuda();models={};potential=set()
    for name in methods:
        model=LeafFeatureHead('mlp',1408).cuda().eval();saved=torch.load(parent/name/'best.pt',weights_only=True,map_location='cpu');assert saved['protocol_sha256']==digest(parent/'protocol.json');model.load_state_dict(saved['model'],strict=True);models[name]=model
        with torch.inference_mode():logits=torch.cat([model(part) for part in features.split(256)]).ravel().cpu().numpy()*pp['calibration'][0]+pp['calibration'][1]
        potential.update(path for path in paths if max(float(logits[bags[path][view]].mean()) for view in ('center','full'))<=float(np.log(.05/.95)))
    potential=sorted(potential);donor={row['input_sha256']:values[i] for i,row in enumerate(descriptors)};rotated=np.load(rotation/'features.npy',mmap_mode='r');rotated_lookup={row['input_sha256']:i for i,row in enumerate(rp['new_inputs'])};pixels=CanonicalInputs(p)
    @lru_cache(maxsize=4)
    def raw(key):
        row=descriptors[key];value=np.ascontiguousarray(pixels(row if 'kind' in row else dict(row,kind='new',policy=row['view'])));assert hashlib.sha256(value.tobytes()).hexdigest()==row['input_sha256'];return value
    previous=p/'phase198_veto_rotation_consistency';previous_protocol=read(previous/'protocol.json');previous_result=read(previous/'result.json');assert previous_result['features_sha256']==digest(previous/'features.npy');previous_values=np.load(previous/'features.npy',mmap_mode='r');previous_lookup={row['input_sha256']:i for i,row in enumerate(previous_protocol['input_metadata'])};assert len(previous_values)==len(previous_lookup)
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
                        elif sha in previous_lookup:local.append(previous_values[previous_lookup[sha]]);reused+=1
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
    pixels.close();local=np.stack(local) if local else np.empty((0,1408),np.float32);np.save(out/'features.npy',local);tensor=torch.from_numpy(local).cuda();scores={}
    for name,model in models.items():
        with torch.inference_mode():logits=(torch.cat([model(part) for part in tensor.split(256)]).ravel().cpu().numpy()*pp['calibration'][0]+pp['calibration'][1]) if len(tensor) else np.empty(0)
        scores[name]={path:max(float(logits[keys].mean()) for keys in pairs[path]) for path in potential}
    raw_sft=np.array([scores['field_source_sft'].get(rows[i]['path'],float('inf')) for i in ids]);raw_rank=np.array([scores['field_source_rank'].get(rows[i]['path'],float('inf')) for i in ids]);boundary=float(np.log(.05/.95))
    veto_modes={'sft_only':raw_sft<=boundary,'unanimous_sft_rank':np.maximum(raw_sft,raw_rank)<=boundary};positive=admissible&np.array([rows[i]['y'] and rows[i]['source']=='olid_tomato' for i in ids]);negative=admissible&np.array([not rows[i]['y'] for i in ids]);other_positive=admissible&np.array([rows[i]['y'] and rows[i]['source']!='olid_tomato' for i in ids]);reports=[]
    np.savez(out/'veto_predictions.npz',indices=ids,sft_max_all_orientation_healthy_logit=raw_sft,rank_max_all_orientation_healthy_logit=raw_rank,protocol_sha256=digest(out/'protocol.json'))
    for mode,veto in veto_modes.items():
        for pair in coverage_result['controls']:
            admitted=((bank[name_index[pair['first']]]>=.95)|(bank[name_index[pair['second']]]>=.95))&admissible&~veto;healthy=selection['fixed_healthy_validation_correct']+int((admitted&positive).sum());errors={source:int(sum(admitted[k] and negative[k] for k,i in enumerate(ids) if rows[i]['source']==source)) for source in sorted({rows[i]['source'] for i in ids if not rows[i]['y']})}
            reports.append(dict(mode=mode,first=pair['first'],second=pair['second'],healthy_validation_correct=healthy,new_false_healthy_by_source=errors,old_rejected_healthy_recovered=int((admitted&other_positive).sum()),cheap_pass=healthy>=28 and sum(errors.values())==0))
    best=max(reports,key=lambda r:(r['cheap_pass'],-sum(r['new_false_healthy_by_source'].values()),r['healthy_validation_correct'],r['old_rejected_healthy_recovered']))
    if best['cheap_pass']:
        models_={}
        for key in ('first','second'):
            prediction=p/best[key];checkpoint=prediction.with_name(prediction.name.replace('_predictions.npz','.pt'));assert digest(checkpoint)==str(np.load(prediction)['checkpoint_sha256']);models_[key]=dict(prediction=best[key],checkpoint=str(checkpoint.relative_to(p)),checkpoint_sha256=digest(checkpoint))
        candidate=dict(**best,native_models=models_,mobile_prediction_sha256=read(p/'phase200_cached_admission_policy_preflight/protocol.json')['mobile_prediction_sha256'],negative_checkpoints=purpose['checkpoint_sha256'],protocol_sha256=digest(out/'protocol.json'),original_healthy_and_named_disease_exact=True,original_gates_unchanged=True,acceptance_data_used=False,model_promoted=False)
        freeze(out/'candidate.json',candidate)
    result=dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),features_sha256=digest(out/'features.npy'),veto_predictions_sha256=digest(out/'veto_predictions.npz'),best=best,controls=reports,policies_evaluated=len(reports),passing_policies=sum(r['cheap_pass'] for r in reports),new_encoder_inputs=len(work),cached_encoder_inputs_reused=reused,model_promoted=False,ad_independent=False,olid_independent=False,acceptance_data_used=False,elapsed_seconds=time.monotonic()-started)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='controls'}),flush=True)


if __name__=='__main__':main()
