"""Reuse old CNN prefixes; compute missing spatial representations for new supervision."""
from pathlib import Path
import argparse,hashlib,json,math,shutil,sqlite3,time
import numpy as np
from PIL import Image
import torch,timm
from torch.nn import functional as F
from app_eval import canon,classifier_letterbox_input
from opened_context_data import read
from partial_cnn_cache import original_cnn
from partial_cnn_healthy import prefix_features,prefix_identity,prefix_cut
from partial_dino_cache import CanonicalInputs,input_rows
from phase114_new_supervision_features import original_features
from permissive_encoder import digest
from population_siglip_tail import freeze
from safe_feature_memmap import feature_memmap


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase141_full_source_cnn_prefix';out.mkdir(exist_ok=True);started=time.monotonic()
    values,bags,descriptors,parent=original_features(p);rows=read(p/'phase84_population_head/fitting_rows.json');selection=read(p/'phase133_anchor_input_cache/protocol.json')
    source=p/'phase137_full_source_features';sc=read(source/'complete.json');assert sc['protocol_sha256']==digest(source/'protocol.json')
    for name,key in [('bags.json','bags_sha256'),('new_features.npy','new_features_sha256'),('new_input_sha256.json','new_keys_sha256')]:assert digest(source/name)==sc[key]
    nb=read(source/'bags.json');new_keys=read(source/'new_input_sha256.json');combined=np.concatenate([values,np.load(source/'new_features.npy')]);keys=[r['input_sha256'] for r in descriptors]+new_keys;global_lookup={sha:i for i,sha in enumerate(keys)}
    selected=sorted({key for i in selection['train_indices']+selection['validation_indices'] for view in bags[rows[i]['path']].values() for key in view}|{key for bag in nb.values() for pair in bag['views'] for view in pair for key in view})
    donor=p/'phase49_partial_cnn/prefix';dc=read(donor/'complete.json');assert dc['protocol_sha256']==digest(donor/'protocol.json') and dc['features_sha256']==digest(donor/'features.npy')
    dr=input_rows(p);dv=np.load(donor/'features.npy',mmap_mode='r');assert len(dr)==len(dv);available={r['input_sha256']:(dv,i) for i,r in enumerate(dr)}
    extension=p/'phase101_cnn_bag_adaptation/prefix';ep=read(extension/'protocol.json');ec=read(extension/'complete.json');assert ec['protocol_sha256']==digest(extension/'protocol.json') and ec['missing_prefix_sha256']==digest(extension/'missing_prefix.npy')
    ev=np.load(extension/'missing_prefix.npy',mmap_mode='r')
    for offset,key in enumerate(ep['missing_tables']):available[descriptors[key]['input_sha256']]=(ev,offset)
    old=[key for key in selected if keys[key] in available];missing=[key for key in selected if keys[key] not in available];missing_lookup={key:i for i,key in enumerate(missing)}
    cnn,architecture,normalization=original_cnn(p);assert prefix_identity(cnn)==dc['prefix_state_sha256']==ep['prefix_state_sha256']
    detector=read(p/'export_cnn_safe/model_manifest.json')['detector'];shape=[len(missing),*dv.shape[1:]]
    protocol=dict(source_sha256=digest(__file__),parents=parent,new_source_complete_sha256=digest(source/'complete.json'),donor_complete_sha256=digest(donor/'complete.json'),extension_complete_sha256=digest(extension/'complete.json'),
        model_bundle_sha256=digest(p/'export_cnn_safe/pytorch_bundle.pt'),prefix_state_sha256=prefix_identity(cnn),architecture=architecture,normalization_source_sha256=digest(Path(__file__).with_name('partial_cnn_cache.py')),
        selected_tables=selected,selected_input_sha256=[keys[key] for key in selected],reused_tables=old,missing_tables=missing,shape=shape,
        representation='Original spatial CNN before final two stages; missing intermediate representation only, no DINO rerun and no detector rerun',tf32=False,dtype='float32',
        ad_independent=False,olid_independent=False,acceptance_data_used=False)
    freeze(out/'protocol.json',protocol)
    if (out/'complete.json').exists():
        result=read(out/'complete.json');assert result['protocol_sha256']==digest(out/'protocol.json') and result['missing_prefix_sha256']==digest(out/'missing_prefix.npy');return
    assert (out/'missing_prefix.npy').exists() or shutil.disk_usage(out).free>int(np.prod(shape))*4+(1<<30)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;timm.layers.set_fused_attn(False)
    cnn=cnn.cuda().eval().requires_grad_(False);normalization={k:v.cuda() for k,v in normalization.items()};prefix=feature_memmap(out/'missing_prefix.npy',shape)
    progress=read(out/'progress.json') if (out/'progress.json').exists() else dict(computed=[]);done=set(progress['computed'])
    if (out/'progress.json').exists():assert progress['protocol_sha256']==digest(out/'protocol.json')
    pending={}
    def flush():
        if not pending:return
        part=list(pending);x=torch.from_numpy(np.concatenate([pending[key] for key in part])).cuda().permute(0,3,1,2)
        x=(x*normalization['canonical_std']+normalization['canonical_mean']-normalization['cnn_mean'])/normalization['cnn_std']
        prefix[[missing_lookup[key] for key in part]]=prefix_features(cnn,x).cpu().numpy();prefix.flush();done.update(part);pending.clear()
        tmp=out/'progress.tmp';tmp.write_text(json.dumps(dict(computed=sorted(done),protocol_sha256=digest(out/'protocol.json'))));tmp.replace(out/'progress.json')
    pixels=CanonicalInputs(p);bank=sqlite3.connect('file:'+str(source/'features.sqlite3')+'?mode=ro',uri=True)
    try:
        with torch.inference_mode():
            for key in missing:
                if key>=len(descriptors) or key in done:continue
                row=descriptors[key];pending[key]=pixels(row if 'kind' in row else dict(row,kind='new',policy=row['view']))
                if len(pending)>=64:flush()
            for number,(path,bag) in enumerate(nb.items()):
                if not bag['actual_leaves']:continue
                record=bag['source_record'];assert digest(path)==record['file_sha256']
                with Image.open(path) as image:rgb=np.asarray(image.convert('RGB'))
                tensor,letterbox=canon.detector_input(rgb,320);identity=hashlib.sha256(detector['sha256'].encode()+str(tensor.shape).encode()+tensor.tobytes()).hexdigest()
                saved=bank.execute('SELECT payload,sha FROM detections WHERE identity=?',(identity,)).fetchone()
                if saved is None:
                    # Pilot detector inputs are deliberately retained in its read-only donor.
                    donor_bank=sqlite3.connect('file:'+str(p/'phase114_new_supervision_features/features.sqlite3')+'?mode=ro',uri=True)
                    try:saved=donor_bank.execute('SELECT payload,sha FROM detections WHERE identity=?',(identity,)).fetchone()
                    finally:donor_bank.close()
                assert saved is not None and hashlib.sha256(saved[0]).hexdigest()==saved[1]
                raw=np.frombuffer(saved[0],np.float32).reshape(1,5,2100);boxes=canon.decode_detections(raw,letterbox,rgb.shape[1],rgb.shape[0],detector['score_threshold'],detector['nms_iou'],detector['max_detections'])[:6]
                assert len(boxes)==bag['actual_leaves']
                for rotation,pair in enumerate(bag['views']):
                    for preprocess,view in zip((canon.classifier_input,classifier_letterbox_input),pair):
                        for box,key in zip(boxes,view):
                            if key not in missing_lookup or key in done or key in pending:continue
                            value=np.ascontiguousarray(np.rot90(preprocess(rgb,224,box[:4]),rotation,axes=(1,2)))
                            assert hashlib.sha256(value.tobytes()).hexdigest()==keys[key];pending[key]=value
                            if len(pending)>=64:flush()
                if number%128==0:print(json.dumps(dict(stage='missing_cnn_prefix',source_images=number+1,computed=len(done),needed=len(missing))),flush=True)
            flush();assert done==set(missing)
            maximum=0.;old_set=set(old)
            for start in range(0,len(selected),64):
                part=selected[start:start+64];x=np.stack([available[keys[key]][0][available[keys[key]][1]] if key in old_set else prefix[missing_lookup[key]] for key in part])
                restored=cnn.forward_head(cnn.blocks[prefix_cut(cnn):](torch.from_numpy(x).cuda()),pre_logits=True)
                error=float(np.abs((F.normalize(restored,dim=1)*math.sqrt(1024)).cpu().numpy()-combined[part,384:]).max());maximum=max(maximum,error);assert error<.002,('Spatial CNN prefix/cache mismatch',error)
    finally:pixels.close();bank.close()
    result=dict(protocol_sha256=digest(out/'protocol.json'),missing_prefix_sha256=digest(out/'missing_prefix.npy'),old_inputs_reused=len(old),new_spatial_prefix_inputs=len(missing),new_detector_inputs=0,new_dino_inputs=0,
        all_selected_inputs_audited=len(selected),maximum_cnn_feature_error=maximum,elapsed_seconds=time.monotonic()-started,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    (out/'complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
