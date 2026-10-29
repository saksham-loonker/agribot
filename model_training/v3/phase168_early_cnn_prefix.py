"""Cache an earlier frozen CNN stage for controlled style-generalization SFT.

The existing stage-five cache cannot supply stage-three activations. Reuse
canonical pixels and cached detections; never recompute old DINO representations.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,sqlite3,time
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from app_eval import canon,classifier_letterbox_input
from opened_context_data import read
from partial_cnn_cache import original_cnn
from partial_dino_cache import CanonicalInputs
from permissive_encoder import digest
from phase114_new_supervision_features import original_features
from population_siglip_tail import freeze
from safe_feature_memmap import feature_memmap


def prefix_state_identity(cnn,cut):
    h=hashlib.sha256(str(cut).encode())
    for name,value in sorted(cnn.state_dict().items()):
        if name.startswith(('conv_stem.','bn1.')) or any(name.startswith(f'blocks.{i}.') for i in range(cut)):
            a=value.detach().cpu().contiguous().numpy();h.update(str((name,str(a.dtype),a.shape)).encode());h.update(a.tobytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase168_early_cnn_prefix';out.mkdir(exist_ok=True);started=time.monotonic()
    dataset=p/'phase165_new_field_joint_teacher';ds=read(dataset/'protocol.json');values,bags,descriptors,parent=original_features(p);rows=read(p/'phase84_population_head/fitting_rows.json')
    extra=p/'phase114_new_supervision_features';ec=read(extra/'complete.json');field=p/'phase164_new_field_features';fc=read(field/'complete.json')
    assert digest(extra/'complete.json')==read(p/'phase152_balanced_photo_teacher/protocol.json')['extra_complete_sha256'] and ec['new_features_sha256']==digest(extra/'new_features.npy') and ec['new_keys_sha256']==digest(extra/'new_input_sha256.json') and ec['bags_sha256']==digest(extra/'bags.json')
    for filename,key in [('features.npy','features_sha256'),('input_sha256.json','input_sha256_sha256'),('bags.json','bags_sha256')]:assert digest(field/filename)==fc[key]
    identities=[r['input_sha256'] for r in descriptors]+read(extra/'new_input_sha256.json');values=np.concatenate([values,np.load(extra/'new_features.npy')]);assert len(values)==len(identities)
    old=read(p/'phase152_balanced_photo_teacher/protocol.json');target=old['target_indices'];anchors=old['old_anchor_indices'];val=ds['validation_indices'];tables={i:[[bags[rows[i]['path']][view] for view in ('center','full')]] for i in target+anchors+val};sources=[];new=[]
    for path,bag in read(extra/'bags.json').items():
        if not bag['actual_leaves']:continue
        i=len(rows);new.append(i);tables[i]=bag['views'];record=bag['source_record'];rows.append(dict(path=path,y=int(record['label']=='Healthy'),source='new107_development',group='new107:entire_publisher_collection',validation=False,label=record['label']))
        sources.append((path,bag,extra,0))
    offset=len(identities);field_keys=read(field/'input_sha256.json');identities+=field_keys;values=np.concatenate([values,np.load(field/'features.npy')[:,:1408]]);field_indices=[]
    for path,bag in read(field/'bags.json').items():
        if not bag['actual_leaves']:continue
        i=len(rows);field_indices.append(i);tables[i]=[[[offset+key for key in view] for view in pair] for pair in bag['views']];record=bag['source_record'];rows.append(dict(path=path,y=int(record['label']=='Healthy'),source='field93',group=record['group'],validation=False,label=record['label']))
        sources.append((path,bag,field,offset))
    train=target+anchors+new+field_indices;assert train==ds['train_indices'] and field_indices==ds['field_training_indices']
    selected=sorted({key for i in train+val for pair in tables[i] for view in pair for key in view});keys=sorted({identities[key] for key in selected});position={sha:i for i,sha in enumerate(keys)}
    local_tables={str(i):[[[position[identities[key]] for key in view] for view in pair] for pair in tables[i]] for i in train+val}
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cnn,architecture,normalization=original_cnn(p);cnn=cnn.cuda().eval();normalization={k:v.cuda() for k,v in normalization.items()};cut=3;assert cut<len(cnn.blocks)-1
    def encode(value):
        x=torch.from_numpy(np.ascontiguousarray(value)).cuda().permute(0,3,1,2)
        x=(x*normalization['canonical_std']+normalization['canonical_mean']-normalization['cnn_mean'])/normalization['cnn_std']
        return cnn.blocks[:cut](cnn.bn1(cnn.conv_stem(x.contiguous(memory_format=torch.channels_last))))
    pixels=CanonicalInputs(p);audit_keys=[key for key in selected if key<len(descriptors)][:16];assert len(audit_keys)==16
    def canonical(key):
        row=descriptors[key];value=pixels(row if 'kind' in row else dict(row,kind='new',policy=row['view']));assert hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()==identities[key];return value
    with torch.inference_mode():
        sample=encode(np.concatenate([canonical(key) for key in audit_keys]));actual=cnn.forward_head(cnn.blocks[cut:](sample),pre_logits=True);actual=(F.normalize(actual,dim=1)*(1024**.5)).cpu().numpy()
    error=float(np.abs(actual-values[audit_keys,384:]).max());assert error<.002,('Earlier prefix restore/normalization differs from verified original CNN',error)
    shape=[len(keys),*sample.shape[1:]];detector=read(p/'export_cnn_safe/model_manifest.json')['detector']
    protocol=dict(source_sha256=digest(__file__),dataset_protocol_sha256=digest(dataset/'protocol.json'),field_complete_sha256=digest(field/'complete.json'),old_extra_complete_sha256=digest(extra/'complete.json'),
        model_bundle_sha256=digest(p/'export_cnn_safe/pytorch_bundle.pt'),architecture=architecture,cut=cut,prefix_state_sha256=prefix_state_identity(cnn,cut),input_sha256=keys,shape=shape,
        train_indices=train,validation_indices=val,target_indices=target,anchor_indices=anchors,old107_indices=new,field_indices=field_indices,
        preprocessing='Exact canonical FP32 NHWC inputs; original CNN native normalization and channels_last stem; earlier prefix after third EfficientNet stage',
        existing_cache_not_reusable='Existing final-two-stage prefix lies downstream of this representation; do not invert it or relabel it as an earlier-stage cache',
        raw_inputs_verified_by_sha256=True,detector_source_sha256=detector['sha256'],new_detector_inputs=0,new_dino_inputs=0,tf32=False,
        maximum_original_cnn_restore_error=error,source_role='development_only',ad_independent=False,olid_independent=False,acceptance_data_used=False)
    freeze(out/'protocol.json',protocol)
    for filename,value in [('rows.json',rows),('bags.json',local_tables)]:freeze(out/filename,value)
    if (out/'complete.json').exists():
        result=read(out/'complete.json');assert result['protocol_sha256']==digest(out/'protocol.json') and result['features_sha256']==digest(out/'features.npy');pixels.close();return
    if not (out/'features.npy').exists():assert shutil.disk_usage(out).free>int(np.prod(shape))*4+(1<<30),'Preserve prior caches and one GiB disk reserve'
    prefix=feature_memmap(out/'features.npy',shape);progress=read(out/'progress.json') if (out/'progress.json').exists() else dict(protocol_sha256=digest(out/'protocol.json'),completed=[]);assert progress['protocol_sha256']==digest(out/'protocol.json')
    done=set(progress['completed']);assert done<=set(range(len(keys)));pending={};new_count=0
    def flush():
        nonlocal new_count
        if not pending:return
        shas=list(pending)
        with torch.inference_mode():part=encode(np.concatenate([pending[sha] for sha in shas])).cpu().numpy()
        for sha,value in zip(shas,part):prefix[position[sha]]=value;done.add(position[sha])
        prefix.flush();new_count+=len(shas);temporary=out/'progress.tmp';temporary.write_text(json.dumps(dict(protocol_sha256=digest(out/'protocol.json'),completed=sorted(done))));temporary.replace(out/'progress.json');pending.clear()
    def put(sha,value):
        assert value.shape==(1,224,224,3) and value.dtype==np.float32 and hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()==sha
        if position[sha] not in done and sha not in pending:pending[sha]=value
        if len(pending)>=32:flush()
    banks={}
    try:
        for key in selected:
            if key<len(descriptors) and position[identities[key]] not in done and identities[key] not in pending:put(identities[key],canonical(key))
        for path,bag,root,base in sources:
            required={identities[base+key] for pair in bag['views'] for view in pair for key in view}
            if all(position[sha] in done or sha in pending for sha in required):continue
            assert digest(path)==bag['source_record']['file_sha256']
            with Image.open(path) as image:rgb=np.asarray(image.convert('RGB'))
            tensor,letterbox=canon.detector_input(rgb,320);identity=hashlib.sha256(detector['sha256'].encode()+str(tensor.shape).encode()+tensor.tobytes()).hexdigest()
            if root not in banks:banks[root]=sqlite3.connect('file:'+str(root/'features.sqlite3')+'?mode=ro',uri=True)
            saved=banks[root].execute('SELECT payload,sha FROM detections WHERE identity=?',(identity,)).fetchone();assert saved is not None and hashlib.sha256(saved[0]).hexdigest()==saved[1]
            boxes=canon.decode_detections(np.frombuffer(saved[0],np.float32).reshape(1,5,2100),letterbox,rgb.shape[1],rgb.shape[0],detector['score_threshold'],detector['nms_iou'],detector['max_detections'])[:6];assert len(boxes)==bag['actual_leaves']
            for rotation,pair in enumerate(bag['views']):
                for preprocess,view in zip((canon.classifier_input,classifier_letterbox_input),pair):
                    assert len(view)==len(boxes)
                    for box,key in zip(boxes,view):
                        sha=identities[base+key]
                        if position[sha] in done or sha in pending:continue
                        put(sha,np.ascontiguousarray(np.rot90(preprocess(rgb,224,box[:4]),rotation,axes=(1,2))))
        flush()
    finally:
        pixels.close()
        for bank in banks.values():bank.close()
    assert len(done)==len(keys),'No missing early-stage feature can be treated as zero training memory'
    result=dict(protocol_sha256=digest(out/'protocol.json'),features_sha256=digest(out/'features.npy'),rows_sha256=digest(out/'rows.json'),bags_sha256=digest(out/'bags.json'),
        newly_computed_earlier_prefixes=new_count,reused_completed_prefixes=len(keys)-new_count,canonical_inputs=len(keys),new_detector_inputs=0,new_dino_inputs=0,
        all_raw_inputs_sha256_checked=True,maximum_original_cnn_restore_error=error,source_role='development_only',acceptance_data_used=False,elapsed_seconds=time.monotonic()-started)
    (out/'complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
