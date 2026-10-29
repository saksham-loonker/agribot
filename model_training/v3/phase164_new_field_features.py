"""Exact-identity cached field features; inference only for newly added inputs."""
from pathlib import Path
import argparse,collections,hashlib,json,shutil,sqlite3,time
import numpy as np
from PIL import Image
import timm,torch
from torchvision.models import convnext_tiny
from ai_edge_litert.interpreter import Interpreter
from app_eval import canon,classifier_letterbox_input,Models
from compact_healthy_model import primary
from dino_large_encoder import DinoLargeEncoder
from feature_cache_reuse import validate_completed_contract,runtime_identity
from opened_context_data import read
from partial_dino_cache import CanonicalInputs
from permissive_encoder import digest
from phase114_new_supervision_features import original_features
from population_siglip_tail import freeze


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase164_new_field_features';out.mkdir(exist_ok=True);started=time.monotonic()
    source=p/'phase163_field_source_overlap';status=read(source/'result.json');assert status['retained_manifest_sha256']==digest(source/'retained_development_manifest.json') and not status['model_predictions_opened'] and not status['protected_training_used']
    candidates=read(source/'retained_development_manifest.json');selected=[]
    for category in sorted({r['publisher_class'] for r in candidates}):
        seen=set();count=0
        for row in sorted([r for r in candidates if r['publisher_class']==category],key=lambda r:hashlib.sha256(('phase164:'+r['id']).encode()).hexdigest()):
            if row['group'] in seen:continue
            seen.add(row['group']);selected.append(row);count+=1
            if category!='Healthy' and count==16:break
    assert sum(row['label']=='Healthy' for row in selected)>30
    old_values,old_bags,descriptors,identity=original_features(p);donors=[{r['input_sha256']:value for r,value in zip(descriptors,old_values)},{},{}];contracts=dict(identity)
    for name in ('phase114_new_supervision_features','phase137_full_source_features'):
        root=p/name;complete=read(root/'complete.json');assert complete['protocol_sha256']==digest(root/'protocol.json') and complete['new_features_sha256']==digest(root/'new_features.npy') and complete['new_keys_sha256']==digest(root/'new_input_sha256.json')
        for sha,value in zip(read(root/'new_input_sha256.json'),np.load(root/'new_features.npy',mmap_mode='r')):
            if sha in donors[0]:assert np.array_equal(donors[0][sha],value)
            else:donors[0][sha]=value
        contracts[name]=digest(root/'complete.json')
    large_root=p/'phase23_dino/dinov2_large_feature_heads';validate_completed_contract(p/'phase23_dino',large_root,'dinov2_large');large_sha=digest(large_root/'encoder.pt')
    for name in ('phase131_large_roi_cache','phase147_large_teacher_features'):
        root=p/name;cp=read(root/'protocol.json');cc=read(root/'complete.json');assert cc['protocol_sha256']==digest(root/'protocol.json') and cc['features_sha256']==digest(root/'features.npy') and cp['encoder_sha256']==large_sha
        for sha,value in zip(cp['input_sha256'],np.load(root/'features.npy',mmap_mode='r')):
            if sha in donors[1]:assert np.array_equal(donors[1][sha],value)
            else:donors[1][sha]=value
        contracts[name]=digest(root/'complete.json')
    conv_root=p/'phase154_convnext_features';cp=read(conv_root/'protocol.json');cc=read(conv_root/'complete.json');assert cc['protocol_sha256']==digest(conv_root/'protocol.json') and cc['features_sha256']==digest(conv_root/'features.npy')
    donors[2]={sha:value for sha,value in zip(cp['input_sha256'],np.load(conv_root/'features.npy',mmap_mode='r'))};contracts['phase154_convnext_features']=digest(conv_root/'complete.json')
    model_sha=digest(p/'export_cnn_safe/pytorch_bundle.pt');detector=read(p/'export_cnn_safe/model_manifest.json')['detector'];detector_path=p/'export_cnn_safe/models'/detector['file'];assert digest(detector_path)==detector['sha256']
    conv_path=Path(torch.hub.get_dir())/'checkpoints'/cp['pretrained_url'].rsplit('/',1)[1];assert digest(conv_path)==cp['pretrained_sha256']
    tower_identity=dict(original_1408=model_sha,large_1024=large_sha,convnext_768=cp['pretrained_sha256'],runtime=runtime_identity(),tf32=False,fused_attention=False,feature_dimensions=[1408,1024,768])
    namespace=hashlib.sha256(json.dumps(tower_identity,sort_keys=True).encode()).hexdigest()
    protocol=dict(source_sha256=digest(__file__),source_overlap_result_sha256=digest(source/'result.json'),source_manifest_sha256=digest(source/'retained_development_manifest.json'),
        selected=[dict(id=r['id'],sha256=r['file_sha256'],label=r['label'],publisher_class=r['publisher_class'],group=r['group']) for r in selected],
        selection='All retained approximate Healthy photo families; at most sixteen families per other publisher class; one image per inferred family/class; entire collection remains development-only',
        donor_complete_sha256=contracts,towers=tower_identity,feature_namespace=namespace,detector=detector,preprocessing_source_sha256=digest(canon.__file__),app_eval_source_sha256=digest(Path(__file__).with_name('app_eval.py')),
        preprocessing='Original actual detector ROIs; canonical 224 FP32 normalized NHWC bytes; unaugmented center/full views; no model/normalization changes',
        large_layout='NHWC permutation to channels_last NCHW',convnext_layout='contiguous NCHW stem; channels_last frozen stage prefix, pooled before clinical norm',
        source_role='development_only',ad_independent=False,olid_independent=False,acceptance_data_used=False,new_training_used=False,model_promoted=False)
    freeze(out/'protocol.json',protocol)
    if (out/'complete.json').exists():
        complete=read(out/'complete.json');assert complete['protocol_sha256']==digest(out/'protocol.json')
        for name,field in [('features.npy','features_sha256'),('input_sha256.json','input_sha256_sha256'),('bags.json','bags_sha256')]:assert complete[field]==digest(out/name)
        return
    torch.set_num_threads(4);torch.manual_seed(2164);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;timm.layers.set_fused_attn(False)
    _,original=primary(p);original=original.cuda().eval();state=torch.load(large_root/'encoder.pt',weights_only=True,map_location='cpu');large=DinoLargeEncoder(state['spec']).cuda().eval();large.load_state_dict(state['model'],strict=True);del state
    conv=convnext_tiny(weights=None).cuda().eval();conv.load_state_dict(torch.load(conv_path,weights_only=True,map_location='cpu'),strict=True)
    def encode(which,value):
        x=torch.from_numpy(np.ascontiguousarray(value)).cuda()
        if which==0:return original.components(x)[0].cpu().numpy()
        if which==1:return large(x).cpu().numpy()
        prefix=conv.features[0](x.permute(0,3,1,2).contiguous()).contiguous(memory_format=torch.channels_last)
        return conv.avgpool(conv.features[1:](prefix)).flatten(1).cpu().numpy()
    audit_rows=[r for r in descriptors if all(r['input_sha256'] in donor for donor in donors)][:16];assert len(audit_rows)==16
    pixels=CanonicalInputs(p)
    try:audit=np.concatenate([pixels(r if 'kind' in r else dict(r,kind='new',policy=r['view'])) for r in audit_rows])
    finally:pixels.close()
    assert all(hashlib.sha256(np.ascontiguousarray(value[None]).tobytes()).hexdigest()==row['input_sha256'] for value,row in zip(audit,audit_rows))
    with torch.inference_mode():errors=[float(np.abs(encode(i,audit)-np.stack([donors[i][r['input_sha256']] for r in audit_rows])).max()) for i in range(3)]
    assert max(errors)<.002,('Frozen encoder restore/preprocessing does not match donor caches',errors)
    freeze(out/'restore_preflight.json',dict(protocol_sha256=digest(out/'protocol.json'),input_sha256=[r['input_sha256'] for r in audit_rows],maximum_feature_errors=errors,passed=True))
    print(json.dumps(dict(stage='field_feature_restore_preflight',maximum_feature_errors=errors,passed=True)),flush=True)
    bank=sqlite3.connect(out/'features.sqlite3');bank.execute('CREATE TABLE IF NOT EXISTS features(identity TEXT PRIMARY KEY,payload BLOB NOT NULL,sha TEXT NOT NULL)');bank.execute('CREATE TABLE IF NOT EXISTS detections(identity TEXT PRIMARY KEY,payload BLOB NOT NULL,sha TEXT NOT NULL)')
    old_banks=[sqlite3.connect('file:'+str(p/name/'features.sqlite3')+'?mode=ro',uri=True) for name in ('phase114_new_supervision_features','phase137_full_source_features')]
    def get(connection,table,key):
        value=connection.execute('SELECT payload,sha FROM '+table+' WHERE identity=?',(key,)).fetchone()
        if value is None:return None
        assert hashlib.sha256(value[0]).hexdigest()==value[1];array=np.frombuffer(value[0],np.float32).copy();assert np.isfinite(array).all();return array
    def put(table,key,array):
        payload=np.ascontiguousarray(array,dtype=np.float32).tobytes();bank.execute('INSERT OR REPLACE INTO '+table+' VALUES(?,?,?)',(key,payload,hashlib.sha256(payload).hexdigest()))
    key_for=lambda sha:hashlib.sha256((namespace+':'+sha).encode()).hexdigest()
    interpreter=Interpreter(model_path=str(detector_path),num_threads=4);interpreter.allocate_tensors();assert interpreter.get_input_details()[0]['shape'].tolist()==detector['input']
    pending={};needed=set();bags={};new=[0,0,0];reused=[0,0,0];detector_calls=0;detector_reused=0
    def flush():
        if not pending:return
        keys=list(pending);chunks={}
        with torch.inference_mode():
            for which,donor in enumerate(donors):
                missing=[sha for sha in keys if sha not in donor];chunks[which]={sha:donor[sha] for sha in keys if sha in donor};reused[which]+=len(keys)-len(missing)
                for start in range(0,len(missing),16):
                    part=missing[start:start+16];value=encode(which,np.concatenate([pending[sha] for sha in part]));chunks[which].update(zip(part,value));new[which]+=len(part)
        assert shutil.disk_usage(out).free>len(keys)*3200*4*3+(1<<30),'Preserve feature-cache disk reserve'
        for sha in keys:put('features',key_for(sha),np.concatenate([chunks[i][sha] for i in range(3)]))
        bank.commit();pending.clear()
    for number,row in enumerate(selected):
        assert digest(row['path'])==row['file_sha256']
        with Image.open(row['path']) as image:rgb=np.asarray(image.convert('RGB'))
        tensor,letterbox=canon.detector_input(rgb,320);key=hashlib.sha256(detector['sha256'].encode()+str(tensor.shape).encode()+tensor.tobytes()).hexdigest();raw=get(bank,'detections',key)
        if raw is None:
            for donor in old_banks:
                raw=get(donor,'detections',key)
                if raw is not None:detector_reused+=1;break
        if raw is None:raw=Models.invoke(interpreter,tensor).ravel();detector_calls+=1
        put('detections',key,raw)
        boxes=canon.decode_detections(raw.reshape(1,5,2100),letterbox,rgb.shape[1],rgb.shape[0],detector['score_threshold'],detector['nms_iou'],detector['max_detections'])[:6];pair=[]
        for preprocess in (canon.classifier_input,classifier_letterbox_input):
            keys=[]
            for box in boxes:
                value=np.ascontiguousarray(preprocess(rgb,224,box[:4]));sha=hashlib.sha256(value.tobytes()).hexdigest();keys.append(sha);needed.add(sha)
                if sha not in pending and get(bank,'features',key_for(sha)) is None:pending[sha]=value
                if len(pending)>=32:flush()
            pair.append(keys)
        bags[row['path']]=dict(views=[pair],source_record=row,actual_leaves=len(boxes));bank.commit()
        if number%16==0:print(json.dumps(dict(stage='field_features',completed_images=number+1,total_images=len(selected),new_encoder_inputs=new,new_detector_inputs=detector_calls)),flush=True)
    flush();keys=sorted(needed);assert keys;assert shutil.disk_usage(out).free>len(keys)*3200*4+(1<<30),'Preserve final feature-array reserve'
    values=np.stack([get(bank,'features',key_for(sha)) for sha in keys]);assert values.shape==(len(keys),3200)
    np.save(out/'features.npy',values);positions={sha:i for i,sha in enumerate(keys)}
    for bag in bags.values():bag['views']=[[[positions[sha] for sha in view] for view in pair] for pair in bag['views']]
    (out/'input_sha256.json').write_text(json.dumps(keys));(out/'bags.json').write_text(json.dumps(bags))
    complete=dict(protocol_sha256=digest(out/'protocol.json'),features_sha256=digest(out/'features.npy'),input_sha256_sha256=digest(out/'input_sha256.json'),bags_sha256=digest(out/'bags.json'),
        restore_preflight_sha256=digest(out/'restore_preflight.json'),source_images=len(selected),detected_images=sum(b['actual_leaves']>0 for b in bags.values()),canonical_input_identities=len(keys),
        new_encoder_inputs_by_tower=dict(zip(['original_1408','large_1024','convnext_768'],new)),donor_encoder_inputs_reused_by_tower=dict(zip(['original_1408','large_1024','convnext_768'],reused)),
        new_detector_inputs=detector_calls,old_detector_inputs_reused=detector_reused,elapsed_seconds=time.monotonic()-started,source_role='development_only',acceptance_data_used=False,model_promoted=False)
    (out/'complete.json').write_text(json.dumps(complete,indent=2));print(json.dumps(complete),flush=True)


if __name__=='__main__':main()
