"""Official DINOv2-L native-518 features from original photos, not 224 upscales.

Uses the already verified Apache-2.0 Meta checkpoint and actual cached detector
geometry. CNN texture features remain their existing exact 224-input identities.
"""
from pathlib import Path
from functools import lru_cache
import argparse,hashlib,json,shutil,sqlite3,time
import numpy as np
from PIL import Image
import torch,timm
from torch.nn import functional as F
from app_eval import canon,classifier_letterbox_input
from dino_large_encoder import SOURCE_URL,SOURCE_SHA256
from feature_cache_reuse import runtime_identity
from opened_context_data import load,read
from partial_dino_cache import CanonicalInputs
from permissive_encoder import digest
from phase85_population_sft import evaluation_data
from phase87_diagnostic_bag_features import cached_mobile
from phase114_new_supervision_features import original_features
from population_siglip_tail import freeze
from safe_feature_memmap import feature_memmap


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase172_native_large_field_features';out.mkdir(exist_ok=True);started=time.monotonic()
    population=p/'phase171_field_replay_ablation';pp=read(population/'protocol.json');dataset=p/'phase168_early_cnn_prefix';dc=read(dataset/'complete.json');assert dc['rows_sha256']==digest(dataset/'rows.json')
    rows=read(dataset/'rows.json');train=pp['train_indices'];val=pp['validation_indices'];active=train+val;assert val==read(p/'phase118_conditional_recovery/protocol.json')['validation_indices']
    original,bags,descriptors,parent=original_features(p);field=p/'phase164_new_field_features';fc=read(field/'complete.json')
    for filename,key in [('features.npy','features_sha256'),('input_sha256.json','input_sha256_sha256'),('bags.json','bags_sha256')]:assert digest(field/filename)==fc[key]
    field_bags=read(field/'bags.json');field_keys=read(field/'input_sha256.json');field_values=np.load(field/'features.npy');assert field_values.shape==(len(field_keys),3200)
    conv=p/'phase154_convnext_features';conv_protocol=read(conv/'protocol.json');conv_complete=read(conv/'complete.json');assert conv_complete['protocol_sha256']==digest(conv/'protocol.json') and conv_complete['features_sha256']==digest(conv/'features.npy')
    conv_bank={sha:value for sha,value in zip(conv_protocol['input_sha256'],np.load(conv/'features.npy'))};conv_bank.update({sha:value[2432:] for sha,value in zip(field_keys,field_values)})
    data=load(p);entries,records,packets,before,previous=evaluation_data(p,data)
    old_sha=read(p/'phase29_compact/source_files.json');old_sha.update({r['path']:r['file_sha256'] for r in read(p/'phase24_olid_source/sealed_manifest.json')})
    old_detector=sqlite3.connect('file:'+str(p/'phase76_fresh_acceptance/cache/predictions.sqlite3')+'?mode=ro',uri=True)
    field_detector=sqlite3.connect('file:'+str(field/'features.sqlite3')+'?mode=ro',uri=True);detector=data['manifest']['detector'];work={};pairs={};links=[];link_lookup={};short_sides=[]
    @lru_cache(maxsize=4)
    def rgb(path,sha):
        assert digest(path)==sha
        with Image.open(path) as image:return np.asarray(image.convert('RGB'))
    try:
        for i in active:
            row=rows[i];path=row['path']
            if row['source']=='field93':
                bag=field_bags[path];sha=bag['source_record']['file_sha256'];low_keys=[[field_keys[key] for key in view] for view in bag['views'][0]]
                image=rgb(path,sha);tensor,letterbox=canon.detector_input(image,320);identity=hashlib.sha256(detector['sha256'].encode()+str(tensor.shape).encode()+tensor.tobytes()).hexdigest();saved=field_detector.execute('SELECT payload,sha FROM detections WHERE identity=?',(identity,)).fetchone();assert saved is not None and hashlib.sha256(saved[0]).hexdigest()==saved[1];raw=np.frombuffer(saved[0],np.float32).reshape(1,5,2100)
                boxes=[box[:4] for box in canon.decode_detections(raw,letterbox,image.shape[1],image.shape[0],detector['score_threshold'],detector['nms_iou'],detector['max_detections'])[:6]]
            else:
                sha=old_sha[path];image=rgb(path,sha);low_keys=[[descriptors[key]['input_sha256'] for key in bags[path][view]] for view in ('center','full')]
                if path in data['photos']:boxes=[leaf['box'] for leaf in data['photos'][path]['leaves']]
                else:
                    tensor,letterbox=canon.detector_input(image,320);raw=cached_mobile(old_detector,detector['sha256'],tensor).reshape(1,5,2100)
                    boxes=[box[:4] for box in canon.decode_detections(raw,letterbox,image.shape[1],image.shape[0],detector['score_threshold'],detector['nms_iou'],detector['max_detections'])[:6]]
            assert len(boxes)==len(low_keys[0])==len(low_keys[1])>0;pairs[path]=[[],[]]
            for leaf,box in enumerate(boxes):
                short_sides.append(min(box[2]-box[0],box[3]-box[1]))
                for view,preprocess in enumerate((canon.classifier_input,classifier_letterbox_input)):
                    low=preprocess(image,224,box);parent_sha=hashlib.sha256(np.ascontiguousarray(low).tobytes()).hexdigest();assert parent_sha==low_keys[view][leaf] and parent_sha in conv_bank,'Canonical parent/CNN identity or actual geometry mismatch'
                    native=preprocess(image,518,box);assert native.shape==(1,518,518,3) and native.dtype==np.float32;key=hashlib.sha256(np.ascontiguousarray(native).tobytes()).hexdigest()
                    work.setdefault(key,dict(path=path,file_sha256=sha,box=list(box),view='center' if view==0 else 'full',input_sha256=key))
                    link=(key,parent_sha)
                    if link not in link_lookup:link_lookup[link]=len(links);links.append(dict(input_sha256=key,parent_224_input_sha256=parent_sha))
                    pairs[path][view].append(link_lookup[link])
    finally:old_detector.close();field_detector.close()
    keys=sorted(work);position={sha:i for i,sha in enumerate(keys)}
    train_keys={links[key]['input_sha256'] for i in train for view in pairs[rows[i]['path']] for key in view};val_keys={links[key]['input_sha256'] for i in val for view in pairs[rows[i]['path']] for key in view}
    assert not train_keys&val_keys and not ({rows[i]['group'] for i in train}&{rows[i]['group'] for i in val})
    official=read(p/'phase23_dino/encoder_source.json');assert official['url']==SOURCE_URL and official['file_sha256']==SOURCE_SHA256 and digest(official['local_file'])==SOURCE_SHA256 and official['license']=='apache-2.0'
    torch.set_num_threads(4);torch.manual_seed(2172);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;timm.layers.set_fused_attn(False)
    model=timm.create_model(official['model_name'],pretrained=True,img_size=518,dynamic_img_size=True,num_classes=0,pretrained_cfg_overlay=dict(file=official['local_file'],hf_hub_id=None,license='apache-2.0')).cuda().eval().requires_grad_(False)
    raw=torch.load(official['local_file'],weights_only=True,map_location='cpu');assert model.pos_embed.shape==raw['pos_embed'].shape==(1,1370,1024) and torch.equal(model.pos_embed.detach().cpu(),raw['pos_embed']);del raw
    donor=p/'phase131_large_roi_cache';dp=read(donor/'protocol.json');done=read(donor/'complete.json');assert done['protocol_sha256']==digest(donor/'protocol.json') and done['features_sha256']==digest(donor/'features.npy');old_features=np.load(donor/'features.npy');old_lookup={sha:i for i,sha in enumerate(dp['input_sha256'])}
    audit_rows=[r for r in descriptors if r['input_sha256'] in old_lookup][:16];assert len(audit_rows)==16;pixels=CanonicalInputs(p)
    try:audit=np.concatenate([pixels(r if 'kind' in r else dict(r,kind='new',policy=r['view'])) for r in audit_rows])
    finally:pixels.close()
    with torch.inference_mode():
        actual=F.normalize(model(torch.from_numpy(audit).cuda().permute(0,3,1,2).contiguous(memory_format=torch.channels_last)),dim=1)*(1024**.5)
    error=float(np.abs(actual.cpu().numpy()-old_features[[old_lookup[r['input_sha256']] for r in audit_rows]]).max());assert error<.002,('Native/dynamic graph fails original 224 encoder restore',error)
    protocol=dict(source_sha256=digest(__file__),population_protocol_sha256=digest(population/'protocol.json'),rows_sha256=digest(dataset/'rows.json'),field_complete_sha256=digest(field/'complete.json'),convnext_complete_sha256=digest(conv/'complete.json'),
        original_features=parent,official_source_url=SOURCE_URL,official_source_sha256=SOURCE_SHA256,licence='Apache-2.0',timm_model=official['model_name'],runtime=runtime_identity(),vit_implementation_sha256=digest(Path(timm.models.vision_transformer.__file__)),
        input_size=518,shape=[len(keys),1024],inputs=[work[key] for key in keys],links=links,bags=pairs,train_indices=train,validation_indices=val,
        feature='Original official DINOv2-L native 37x37 positional grid, normalized CLS L2/RMS; retain matching cached 224 ConvNeXt texture features by canonical parent identity',
        maximum_dynamic_224_restore_error=error,native_positional_embeddings_exact=True,tf32=False,fused_attention=False,input_layout='channels_last NCHW from canonical normalized NHWC518',
        new_224_encoder_inputs=0,new_detector_inputs=0,no_cached_224_pixels_upscaled=True,source_role='development_only',ad_independent=False,olid_independent=False,acceptance_data_used=False)
    freeze(out/'protocol.json',protocol)
    if (out/'complete.json').exists():
        result=read(out/'complete.json');assert result['protocol_sha256']==digest(out/'protocol.json') and result['features_sha256']==digest(out/'features.npy');return
    assert (out/'features.npy').exists() or shutil.disk_usage(out).free>len(keys)*1024*4+(1<<30),'Preserve previous caches and disk reserve'
    values=feature_memmap(out/'features.npy',[len(keys),1024]);progress=read(out/'progress.json') if (out/'progress.json').exists() else dict(completed=0,protocol_sha256=digest(out/'protocol.json'));assert progress['protocol_sha256']==digest(out/'protocol.json');new=0
    print(json.dumps(dict(stage='native_large_restore_preflight',maximum_224_feature_error=error,native_position_exact=True,inputs=len(keys))),flush=True)
    with torch.inference_mode():
        for start in range(progress['completed'],len(keys),2):
            part=keys[start:start+2];batch=[]
            for key in part:
                row=work[key];value=(canon.classifier_input if row['view']=='center' else classifier_letterbox_input)(rgb(row['path'],row['file_sha256']),518,row['box']);assert hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()==key;batch.append(value)
            x=torch.from_numpy(np.concatenate(batch)).cuda().permute(0,3,1,2).contiguous(memory_format=torch.channels_last);features=F.normalize(model(x),dim=1)*(1024**.5);values[start:start+len(part)]=features.cpu().numpy();values.flush();new+=len(part)
            progress=dict(completed=start+len(part),protocol_sha256=digest(out/'protocol.json'));temporary=out/'progress.tmp';temporary.write_text(json.dumps(progress));temporary.replace(out/'progress.json')
            if start%64==0 or start+len(part)==len(keys):print(json.dumps(dict(stage='native_large_features',completed=progress['completed'],total=len(keys))),flush=True)
    result=dict(protocol_sha256=digest(out/'protocol.json'),features_sha256=digest(out/'features.npy'),new_native_518_inputs=new,cached_native_inputs_reused=len(keys)-new,canonical_native_inputs=len(keys),
        maximum_dynamic_224_restore_error=error,native_positional_embeddings_exact=True,native_roi_short_side_quantiles=np.quantile(short_sides,[0,.25,.5,.75,1]).tolist(),fraction_native_roi_short_side_above_224=float(np.mean(np.array(short_sides)>224)),fraction_native_roi_short_side_above_518=float(np.mean(np.array(short_sides)>518)),
        new_detector_inputs=0,old_224_encoders_recomputed=0,source_role='development_only',acceptance_data_used=False,model_promoted=False,elapsed_seconds=time.monotonic()-started)
    (out/'complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
