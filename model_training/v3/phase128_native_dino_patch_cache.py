"""Frozen DINO patch evidence from native 448px ROI pixels, never enlarged 224px caches."""
from pathlib import Path
import argparse, hashlib, json, math, shutil, time
import numpy as np
from PIL import Image
import torch, timm
from torch.nn import functional as F
from app_eval import canon, classifier_letterbox_input
from feature_cache_reuse import runtime_identity
from high_resolution_cnn_cache import high_resolution_key
from opened_context_data import read
from partial_dino_cache import original_dino, CanonicalInputs
from permissive_encoder import digest
from phase114_new_supervision_features import original_features
from population_siglip_tail import freeze
from safe_feature_memmap import feature_memmap


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase128_native_dino_patch_cache';out.mkdir(exist_ok=True);started=time.monotonic()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;timm.layers.set_fused_attn(False)
    values,bags224,descriptors,parent=original_features(p);rows=read(p/'phase84_population_head/fitting_rows.json')
    conditional=read(p/'phase118_conditional_recovery/protocol.json');indices=conditional['train_indices']+conditional['validation_indices']
    high=p/'phase104_high_resolution_cnn/prefix';complete=read(high/'complete.json');hp=read(high/'protocol.json');hb=read(high/'bags.json')
    assert complete['protocol_sha256']==digest(high/'protocol.json') and complete['bags_sha256']==digest(high/'bags.json')
    paths={rows[i]['path'] for i in indices};assert paths<=set(hb),'Missing native geometry; never substitute enlarged cached inputs'
    selected=sorted({key for path in paths for view in hb[path].values() for key in view});lookup={key:i for i,key in enumerate(selected)}
    canonical={row['input_sha256']:i for i,row in enumerate(descriptors)}
    inputs=[hp['inputs'][key] for key in selected];parents=[canonical[row['parent_224_input_sha256']] for row in inputs]
    # Verify corresponding view/leaf identity, not just a collision-prone local row number.
    for path in paths:
        for view in ('center','full'):
            assert len(hb[path][view])==len(bags224[path][view])
            for hi,lo in zip(hb[path][view],bags224[path][view]):
                assert hp['inputs'][hi]['parent_224_input_sha256']==descriptors[lo]['input_sha256']
    protocol=dict(source_sha256=digest(__file__),parents=parent,geometry_protocol_sha256=digest(high/'protocol.json'),geometry_bags_sha256=digest(high/'bags.json'),
        conditional_protocol_sha256=digest(p/'phase118_conditional_recovery/protocol.json'),runtime=runtime_identity(),timm_version=timm.__version__,
        vit_implementation_sha256=digest(Path(timm.models.vision_transformer.__file__)),original_model_sha256=digest(p/'export_cnn_safe/pytorch_bundle.pt'),
        inputs=inputs,parent_224_feature_indices=parents,bags={path:{view:[lookup[key] for key in keys] for view,keys in hb[path].items()} for path in sorted(paths)},
        shape=[len(inputs),1025,384],representation='Original normalized 448px DINO patch tokens; CLS replaced by matching original normalized 224px CLS to preserve warm classifier. True 448px CLS stored separately.',
        position_embedding='timm dynamic_img_size, pretrained 16x16 positional grid interpolated to 32x32, implementation pinned by source hash',
        tf32=False,fused_attention=False,dtype='float32',input_size=448,detector_inputs_recomputed=0,old_224_encoder_inputs_recomputed=0,
        ad_independent=False,olid_independent=False,acceptance_data_used=False)
    if (out/'complete.json').exists():
        saved=read(out/'complete.json');assert read(out/'protocol.json')==protocol and saved['protocol_sha256']==digest(out/'protocol.json')
        assert saved['tokens_sha256']==digest(out/'tokens.npy') and saved['native_cls_sha256']==digest(out/'native_cls.npy');return
    freeze(out/'protocol.json',protocol)
    original=original_dino(p)
    model=timm.create_model('vit_small_patch14_dinov2.lvd142m',pretrained=False,num_classes=9,img_size=224,dynamic_img_size=True)
    model.load_state_dict(original.state_dict(),strict=True);del original;model=model.cuda().eval().requires_grad_(False)
    # Audit dynamic 224 graph against existing canonical CLS features; never compare different input identities.
    pixels=CanonicalInputs(p);maximum=0.
    try:
        with torch.inference_mode():
            audit=np.unique(np.linspace(0,len(parents)-1,16,dtype=int));batch=[]
            for offset in audit:
                row=descriptors[parents[offset]];row=row if 'kind' in row else dict(row,kind='new',policy=row['view'])
                batch.append(pixels(row))
            raw=model.forward_features(torch.from_numpy(np.concatenate(batch)).cuda().permute(0,3,1,2))
            maximum=float(np.abs((F.normalize(raw[:,0],dim=1)*math.sqrt(384)).cpu().numpy()-values[np.array(parents)[audit],:384]).max())
            assert maximum<.002,('Dynamic 224 graph differs from original',maximum)
    finally:pixels.close()
    needed=len(inputs)*1025*384*4
    assert (out/'tokens.npy').exists() or shutil.disk_usage(out).free>needed+(1<<30),'Preserve existing caches and disk reserve'
    tokens=feature_memmap(out/'tokens.npy',protocol['shape']);cls=feature_memmap(out/'native_cls.npy',[len(inputs),384])
    progress=read(out/'progress.json') if (out/'progress.json').exists() else dict(completed=0)
    if (out/'progress.json').exists():assert progress['protocol_sha256']==digest(out/'protocol.json')
    with torch.inference_mode():
        for start in range(progress['completed'],len(inputs),8):
            batch=[]
            for row in inputs[start:start+8]:
                assert digest(row['path'])==row['file_sha256']
                with Image.open(row['path']) as image:rgb=np.asarray(image.convert('RGB'))
                value=(canon.classifier_input if row['view']=='center' else classifier_letterbox_input)(rgb,448,row['box'])
                assert high_resolution_key(value)==row['input_sha256'];batch.append(value)
            raw=model.forward_features(torch.from_numpy(np.concatenate(batch)).cuda().permute(0,3,1,2)).cpu().numpy()
            assert raw.shape[1:]==(1025,384) and np.isfinite(raw).all()
            cls[start:start+len(raw)]=raw[:,0]
            # This warm-start branch consumes the same old CLS direction and CNN feature as every 224px control.
            raw[:,0]=values[parents[start:start+len(raw)],:384]
            tokens[start:start+len(raw)]=raw;tokens.flush();cls.flush()
            progress=dict(completed=start+len(raw),protocol_sha256=digest(out/'protocol.json'))
            tmp=out/'progress.tmp';tmp.write_text(json.dumps(progress));tmp.replace(out/'progress.json')
            if start%128==0:print(json.dumps(dict(stage='native_dino_patch_cache',**progress,total=len(inputs))),flush=True)
    result=dict(protocol_sha256=digest(out/'protocol.json'),tokens_sha256=digest(out/'tokens.npy'),native_cls_sha256=digest(out/'native_cls.npy'),
        new_448_encoder_inputs=len(inputs),old_224_encoder_inputs_recomputed=0,detector_inputs_recomputed=0,dynamic_224_audit_inputs=len(audit),maximum_224_cls_error=maximum,
        elapsed_seconds=time.monotonic()-started,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    (out/'complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
