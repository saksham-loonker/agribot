"""Cache only new frozen-encoder inputs for a bounded development-only probe."""
from pathlib import Path
import argparse,hashlib,json,sqlite3,time
import numpy as np
from PIL import Image
import timm,torch
from ai_edge_litert.interpreter import Interpreter
from app_eval import canon,classifier_letterbox_input,Models
from compact_healthy_model import primary
from opened_context_data import read
from permissive_encoder import digest
from population_siglip_tail import freeze


def original_features(p):
    base=p/'phase29_compact';extra=p/'phase87_diagnostic_bag_features'
    bc=read(base/'complete.json');ec=read(extra/'complete.json')
    assert bc['protocol_sha256']==digest(base/'protocol.json') and ec['protocol_sha256']==digest(extra/'protocol.json')
    assert bc['features_sha256']==digest(base/'features.npy') and ec['new_features_sha256']==digest(extra/'new_features.npy')
    assert read(base/'protocol.json')['primary_bundle_sha256']==digest(p/'export_cnn_safe/pytorch_bundle.pt')
    values=np.concatenate([np.load(base/'features.npy',mmap_mode='r'),np.load(extra/'new_features.npy',mmap_mode='r')])
    rows=read(base/'inputs.json')+read(extra/'new_inputs.json');assert len(rows)==len(values)
    return values,read(extra/'bags.json'),rows,dict(base_complete_sha256=digest(base/'complete.json'),extra_complete_sha256=digest(extra/'complete.json'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase114_new_supervision_features';out.mkdir(exist_ok=True);started=time.monotonic()
    values,old_bags,old_rows,identity=original_features(p)
    old_lookup={row['input_sha256']:i for i,row in enumerate(old_rows)}
    source=p/'phase113_development_overlap';ready=read(source/'result.json')
    assert not ready['training_used'] and not ready['diagnostic_predictions_opened'] and ready['source_role']=='development_only'
    candidates=read(source/'retained_development_manifest.json');selected=[];groups=set()
    for label in ('Healthy','Diseased_unspecified'):
        choices=sorted([r for r in candidates if r['label']==label],key=lambda r:hashlib.sha256(('2114:'+r['group']+r['id']).encode()).hexdigest())
        count=0
        for row in choices:
            if row['group'] in groups:continue
            groups.add(row['group']);selected.append(row);count+=1
            if count==128:break
        assert count==128
    detector=read(p/'export_cnn_safe/model_manifest.json')['detector'];model_sha=digest(p/'export_cnn_safe/pytorch_bundle.pt')
    protocol=dict(source_sha256=digest(__file__),parents=identity,retained_manifest_sha256=digest(source/'retained_development_manifest.json'),
                  selected=[dict(id=r['id'],sha256=r['file_sha256'],group=r['group'],label=r['label']) for r in selected],model_bundle_sha256=model_sha,
                  detector_sha256=detector['sha256'],seed=2114,photographs_per_class=128,rotations=[0,90,180,270],
                  quality_admission='Exploratory development training only. All new-source photos belong to one collection, never validation/acceptance. Rotate every selected input equally to break orientation shortcuts; camera/style confounding remains unresolved.',
                  supervision='Publisher primary-image Healthy versus broadly diseased; actual ROI bags, no per-leaf clinical labels or named-disease/Other relabeling',
                  old_encoder_weights_changed=False,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    freeze(out/'protocol.json',protocol)
    if (out/'complete.json').exists():
        saved=read(out/'complete.json');assert saved['protocol_sha256']==digest(out/'protocol.json')
        assert saved['new_features_sha256']==digest(out/'new_features.npy')
        assert saved['bags_sha256']==digest(out/'bags.json');return
    bank=sqlite3.connect(out/'features.sqlite3');bank.execute('CREATE TABLE IF NOT EXISTS features(identity TEXT PRIMARY KEY,payload BLOB NOT NULL,sha TEXT NOT NULL)')
    bank.execute('CREATE TABLE IF NOT EXISTS detections(identity TEXT PRIMARY KEY,payload BLOB NOT NULL,sha TEXT NOT NULL)')
    detector_path=p/'export_cnn_safe/models'/detector['file'];assert digest(detector_path)==detector['sha256']
    interpreter=Interpreter(model_path=str(detector_path),num_threads=4);interpreter.allocate_tensors()
    assert interpreter.get_input_details()[0]['shape'].tolist()==detector['input']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;timm.layers.set_fused_attn(False)
    _,model=primary(p);model=model.cuda().eval();pending={};needed=set();bags={};detector_calls=0;encoder_inputs=0
    def saved(table,key):
        row=bank.execute('SELECT payload,sha FROM '+table+' WHERE identity=?',(key,)).fetchone()
        if row is None:return None
        assert hashlib.sha256(row[0]).hexdigest()==row[1],'Changed cached payload';return np.frombuffer(row[0],np.float32).copy()
    def put(table,key,value):
        payload=np.ascontiguousarray(value,dtype=np.float32).tobytes();bank.execute('INSERT OR REPLACE INTO '+table+' VALUES(?,?,?)',(key,payload,hashlib.sha256(payload).hexdigest()))
    def feature_key(sha):return hashlib.sha256((model_sha+':original-1408:'+sha).encode()).hexdigest()
    def flush():
        nonlocal encoder_inputs
        if not pending:return
        keys=list(pending)
        with torch.inference_mode():features,_=model.components(torch.from_numpy(np.concatenate([pending[k] for k in keys])).cuda())
        for key,value in zip(keys,features.cpu().numpy()):put('features',feature_key(key),value)
        bank.commit();encoder_inputs+=len(keys);pending.clear()
    for number,row in enumerate(selected):
        assert digest(row['path'])==row['file_sha256']
        with Image.open(row['path']) as image:rgb=np.asarray(image.convert('RGB'))
        tensor,letterbox=canon.detector_input(rgb,320)
        key=hashlib.sha256(detector['sha256'].encode()+str(tensor.shape).encode()+tensor.tobytes()).hexdigest()
        raw=saved('detections',key)
        if raw is None:
            raw=Models.invoke(interpreter,tensor).ravel();put('detections',key,raw);detector_calls+=1
        boxes=canon.decode_detections(raw.reshape(1,5,2100),letterbox,rgb.shape[1],rgb.shape[0],detector['score_threshold'],detector['nms_iou'],detector['max_detections'])[:6]
        pairs=[]
        for rotation in range(4):
            pair=[]
            for preprocess in (canon.classifier_input,classifier_letterbox_input):
                keys=[]
                for box in boxes:
                    tensor=np.ascontiguousarray(np.rot90(preprocess(rgb,224,box[:4]),rotation,axes=(1,2)))
                    sha=hashlib.sha256(tensor.tobytes()).hexdigest();keys.append(sha);needed.add(sha)
                    if sha not in old_lookup and sha not in pending and saved('features',feature_key(sha)) is None:pending[sha]=tensor
                    if len(pending)>=64:flush()
                pair.append(keys)
            pairs.append(pair)
        bags[row['path']]=dict(views=pairs,source_record=row,actual_leaves=len(boxes))
        bank.commit()
        if number%16==0:print(json.dumps(dict(completed_images=number+1,total=len(selected),new_encoder_inputs=encoder_inputs)),flush=True)
    flush()
    new_keys=sorted(needed-set(old_lookup));new=np.stack([saved('features',feature_key(sha)) for sha in new_keys]);assert new.shape==(len(new_keys),1408)
    np.save(out/'new_features.npy',new);lookup=dict(old_lookup);lookup.update({key:len(values)+i for i,key in enumerate(new_keys)})
    for bag in bags.values():bag['views']=[[[lookup[key] for key in view] for view in pair] for pair in bag['views']]
    (out/'new_input_sha256.json').write_text(json.dumps(new_keys));(out/'bags.json').write_text(json.dumps(bags))
    complete=dict(protocol_sha256=digest(out/'protocol.json'),new_features_sha256=digest(out/'new_features.npy'),bags_sha256=digest(out/'bags.json'),
                  new_keys_sha256=digest(out/'new_input_sha256.json'),new_detector_inputs=detector_calls,new_encoder_inputs=encoder_inputs,old_encoder_inputs_reused=sum(key in old_lookup for key in needed),
                  source_images=len(selected),detected_images=sum(b['actual_leaves']>0 for b in bags.values()),elapsed_seconds=time.monotonic()-started,
                  role='development_only',model_promoted=False,acceptance_data_used=False)
    (out/'complete.json').write_text(json.dumps(complete,indent=2));print(json.dumps(complete),flush=True)


if __name__=='__main__':main()
