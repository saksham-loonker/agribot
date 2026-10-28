"""Restore exact development replay pixels into bounded GPU memory, without detection."""
import hashlib,math,sqlite3
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from app_eval import canon,classifier_letterbox_input
from opened_context_data import read
from partial_dino_cache import CanonicalInputs
from permissive_encoder import digest


def replay_pixels(p,selected,descriptors,keys,bags,expected,cnn,normalization):
    p=Path(p);lookup={key:i for i,key in enumerate(selected)};raw=torch.empty((len(selected),3,224,224),dtype=torch.float32,device='cuda');done=set();pixels=CanonicalInputs(p)
    detector=read(p/'export_cnn_safe/model_manifest.json')['detector'];source=p/'phase114_new_supervision_features';sp=read(source/'protocol.json');assert sp['detector_sha256']==detector['sha256']
    bank=sqlite3.connect('file:'+str(source/'features.sqlite3')+'?mode=ro',uri=True)
    def put(key,tensor):
        assert hashlib.sha256(np.ascontiguousarray(tensor).tobytes()).hexdigest()==keys[key]
        raw[lookup[key]]=torch.from_numpy(tensor[0]).cuda().permute(2,0,1);done.add(key)
    try:
        for key in selected:
            if key>=len(descriptors):continue
            row=descriptors[key];put(key,pixels(row if 'kind' in row else dict(row,kind='new',policy=row['view'])))
        for path,bag in bags.items():
            if not bag['actual_leaves']:continue
            record=bag['source_record'];assert digest(path)==record['file_sha256']
            with Image.open(path) as image:rgb=np.asarray(image.convert('RGB'))
            tensor,letterbox=canon.detector_input(rgb,320);identity=hashlib.sha256(detector['sha256'].encode()+str(tensor.shape).encode()+tensor.tobytes()).hexdigest()
            saved=bank.execute('SELECT payload,sha FROM detections WHERE identity=?',(identity,)).fetchone();assert saved is not None and hashlib.sha256(saved[0]).hexdigest()==saved[1]
            boxes=canon.decode_detections(np.frombuffer(saved[0],np.float32).reshape(1,5,2100),letterbox,rgb.shape[1],rgb.shape[0],detector['score_threshold'],detector['nms_iou'],detector['max_detections'])[:6]
            assert len(boxes)==bag['actual_leaves']
            for rotation,pair in enumerate(bag['views']):
                for preprocess,view in zip((canon.classifier_input,classifier_letterbox_input),pair):
                    for box,key in zip(boxes,view):
                        if key in lookup and key not in done:put(key,np.ascontiguousarray(np.rot90(preprocess(rgb,224,box[:4]),rotation,axes=(1,2))))
    finally:pixels.close();bank.close()
    assert done==set(selected),'Never leave a missing raw image as uninitialized training memory'
    maximum=0.;cnn.eval()
    with torch.inference_mode():
        for start in range(0,len(selected),64):
            x=raw[start:start+64];x=(x*normalization['canonical_std']+normalization['canonical_mean']-normalization['cnn_mean'])/normalization['cnn_std']
            actual=cnn.forward_head(cnn.forward_features(x),pre_logits=True);actual=(F.normalize(actual,dim=1)*math.sqrt(1024)).cpu().numpy()
            error=float(np.abs(actual-expected[selected[start:start+64],384:]).max());maximum=max(maximum,error);assert error<.002,('Raw pixel/CNN normalization cache mismatch',error)
    return raw,lookup,dict(inputs=len(selected),all_raw_inputs_audited=len(selected),maximum_cnn_feature_error=maximum,new_detector_inputs=0,new_dino_inputs=0)
