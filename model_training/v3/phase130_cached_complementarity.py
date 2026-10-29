"""Consumed-development diagnostic: can already selected safe experts complement each other?

This computes an optimistic OR coverage bound, not independent acceptance and
not an exported ensemble. It never chooses thresholds using validation labels.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from leaf_feature_expert import LeafFeatureHead
from local_damage_heads import LocalRecoveryHead
from context_recovery_heads import ContextRecoveryHead
from conditional_recovery_policy import recover_only
from opened_context_data import load,read
from phase85_population_sft import evaluation_data
from phase114_new_supervision_features import original_features
from phase81_consumed_olid_features import prepare as prepare_whole
from permissive_encoder import digest
from population_siglip_tail import freeze


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase130_cached_complementarity';out.mkdir(exist_ok=True);started=time.monotonic()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    values,bags,descriptors,parent=original_features(p);rows=read(p/'phase84_population_head/fitting_rows.json');selection=read(p/'phase118_conditional_recovery/protocol.json');indices=np.array(selection['validation_indices'])
    data=load(p);entries,records,packets,before,previous=evaluation_data(p,data)
    items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({b['path']:(b,packets[b['id']],r) for b,r in zip(before,previous)})
    features=torch.from_numpy(values).cuda();whole=None;local=None;experts={};identities={};new_predictions=0
    def register(folder,probability):
        r=read(folder/'result.json');assert r['checkpoint_sha256']==digest(folder/'best.pt') and r['protocol_sha256']==digest(folder/'protocol.json' if (folder/'protocol.json').exists() else folder.parent/'protocol.json')
        key=str(folder.relative_to(p));identity=dict(checkpoint_sha256=digest(folder/'best.pt'),protocol_sha256=r['protocol_sha256'])
        if (folder/'probabilities.json').exists():identity['existing_predictions_sha256']=digest(folder/'probabilities.json')
        identities[key]=identity;experts[key]=np.asarray(probability,dtype=np.float64);assert experts[key].shape==(len(indices),) and np.isfinite(experts[key]).all()
    # Existing complete old-expert predictions: do not invoke their encoders again.
    for name in ('phase95_corrected_continued_expert','phase97_continued_verified_expert','phase101_cnn_bag_adaptation'):
        folder=p/name;prob=read(folder/'probabilities.json');assert all(rows[i]['path'] in prob for i in indices)
        register(folder,[prob[rows[i]['path']]['positive'] for i in indices])
    # New cheap controls: cache minimal validation predictions, tied to checkpoint and input identities.
    warm=torch.load(p/'phase88_diagnostic_bag_head/mlp/best.pt',weights_only=True,map_location='cpu')['model']
    for phase in ('phase118_conditional_recovery','phase119_target_replay','phase121_local_damage_mil','phase122_context_fusion'):
        root=p/phase;protocol=read(root/'protocol.json');assert protocol['validation_indices']==indices.tolist()
        for method in protocol['methods']:
            folder=root/method;result=read(folder/'result.json');assert result['checkpoint_sha256']==digest(folder/'best.pt')
            cache=out/(phase+'_'+method+'.npz');identity=dict(checkpoint_sha256=digest(folder/'best.pt'),protocol_sha256=digest(root/'protocol.json'),indices=indices.tolist(),parent=parent)
            if cache.exists():
                saved=np.load(cache);assert json.loads(str(saved['identity']))==identity;register(folder,saved['probability']);continue
            saved=torch.load(folder/'best.pt',weights_only=True,map_location='cpu');calibration=protocol['calibration']
            with torch.inference_mode():
                if phase in ('phase118_conditional_recovery','phase119_target_replay'):
                    model=LeafFeatureHead('mlp',1408).cuda().eval();model.load_state_dict({key[len('expert.'):]:v for key,v in saved['model'].items() if key.startswith('expert.')})
                    needed=sorted({key for i in indices for view in bags[rows[i]['path']].values() for key in view});lookup={key:j for j,key in enumerate(needed)}
                    raw=(model(features[needed])*calibration[0]+calibration[1]).cpu().numpy()
                    margin=np.array([min(raw[[lookup[key] for key in bags[rows[i]['path']][view]]].mean() for view in ('center','full')) for i in indices])
                elif phase=='phase121_local_damage_mil':
                    if local is None:
                        lc=p/'phase120_local_patch_cache';lp=read(lc/'protocol.json');complete=read(lc/'complete.json');assert complete['tokens_sha256']==digest(lc/'tokens.npy') and complete['protocol_sha256']==digest(lc/'protocol.json')
                        local=torch.from_numpy(np.array(np.load(lc/'tokens.npy',mmap_mode='r'),copy=True)).cuda();cnn=torch.from_numpy(values[lp['selected_tables'],384:].copy()).cuda();lookup={sha:j for j,sha in enumerate(lp['input_sha256'])}
                    model=LocalRecoveryHead(method,warm,calibration).cuda().eval();model.load_state_dict(saved['model'])
                    raw=torch.cat([model(local[start:start+32],cnn[start:start+32]) for start in range(0,len(local),32)]).cpu().numpy()
                    margin=np.array([min(raw[[lookup[descriptors[key]['input_sha256']] for key in bags[rows[i]['path']][view]]].mean() for view in ('center','full')) for i in indices])
                else:
                    if whole is None:whole,_=prepare_whole(p,data)
                    model=ContextRecoveryHead(method,warm,calibration).cuda().eval();model.load_state_dict(saved['model']);resulting=[]
                    for i in indices:
                        pair=bags[rows[i]['path']];n=max(len(keys) for keys in pair.values());leaf=features.new_zeros((1,2,n,1408));mask=features.new_zeros((1,2,n))
                        for v,view in enumerate(('center','full')):
                            leaf[0,v,:len(pair[view])]=features[pair[view]];mask[0,v,:len(pair[view])]=1.
                        raw=model(leaf,torch.from_numpy(whole[rows[i]['path']][None]).cuda());resulting.append(float(((raw*mask).sum(-1)/mask.sum(-1)).min()))
                    margin=np.array(resulting)
                probability=1/(1+np.exp(-np.clip(margin,-80,80)));new_predictions+=len(indices)
            np.savez(cache,identity=json.dumps(identity,sort_keys=True),probability=probability);register(folder,probability)
    for phase in ('phase124_lora_recovery','phase127_clinical_anchors','phase129_native_dino_mil'):
        root=p/phase;protocol=read(root/'protocol.json');folders=[root/method for method in protocol['methods']] if 'methods' in protocol else [root]
        for folder in folders:
            cached=np.load(folder/'best_validation_predictions.npz');assert np.array_equal(cached['indices'],indices)
            if 'probability' in cached.files:probability=cached['probability']
            else:
                z=cached['paired_margin'] if 'paired_margin' in cached.files else cached['logits'];probability=1/(1+np.exp(-np.clip(z.min(1),-80,80)))
            register(folder,probability)
    routed={};summaries={};labels=np.array([rows[i]['y'] for i in indices]);target=np.array([rows[i]['source']=='olid_tomato' for i in indices]);fixed=selection['fixed_healthy_validation_correct']
    for name,probability in experts.items():
        outcome=[]
        for i,value in zip(indices,probability):
            original,packet,retained=items[rows[i]['path']];outcome.append(recover_only(original,retained,packet,float(value),data['manifest'])['app_replay']['kind']=='HEALTHY')
        routed[name]=np.array(outcome);summaries[name]=dict(new_false_healthy=int((routed[name]&(labels==0)).sum()),healthy_validation_correct=fixed+int((routed[name]&(labels==1)&target).sum()))
    safe=[name for name,s in summaries.items() if s['new_false_healthy']==0];union=np.logical_or.reduce([routed[name] for name in safe]) if safe else np.zeros(len(indices),bool)
    protocol=dict(source_sha256=digest(__file__),experts=identities,selection_protocol_sha256=digest(p/'phase118_conditional_recovery/protocol.json'),policy_source_sha256=digest(Path(__file__).with_name('conditional_recovery_policy.py')),
        purpose='Optimistic complementarity bound of already selected zero-new-FP experts on consumed development; not a frozen selected ensemble or independent acceptance',ad_independent=False,olid_independent=False,acceptance_data_used=False)
    freeze(out/'protocol.json',protocol)
    report=dict(protocol_sha256=digest(out/'protocol.json'),summaries=summaries,safe_experts=safe,optimistic_safe_union_healthy_validation_correct=fixed+int((union&(labels==1)&target).sum()),healthy_denominator=40,required=28,
        full_model_inference_inputs=0,new_encoder_inputs=0,new_cached_head_validation_predictions=new_predictions,elapsed_seconds=time.monotonic()-started,model_promoted=False)
    (out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
