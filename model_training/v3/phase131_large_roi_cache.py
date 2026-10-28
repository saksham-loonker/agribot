"""Extend the licensed DINOv2-L cache by exact canonical ROI identity only."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch,timm
from campaign_pixels import open_pixels
from dino_large_encoder import DinoLargeEncoder
from feature_cache_reuse import validate_completed_contract,runtime_identity,preprocessing_contract
from opened_context_data import read
from partial_dino_cache import CanonicalInputs
from permissive_encoder import digest
from phase114_new_supervision_features import original_features
from population_siglip_tail import freeze
from safe_feature_memmap import feature_memmap


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase131_large_roi_cache';out.mkdir(exist_ok=True);started=time.monotonic()
    values,bags,descriptors,parent=original_features(p);rows=read(p/'phase84_population_head/fitting_rows.json');selection=read(p/'phase118_conditional_recovery/protocol.json')
    selected=sorted({key for i in selection['train_indices']+selection['validation_indices'] for view in bags[rows[i]['path']].values() for key in view})
    source=p/'phase23_dino';directory=source/'dinov2_large_feature_heads';validate_completed_contract(source,directory,'dinov2_large')
    protocol=dict(source_sha256=digest(__file__),parents=parent,selection_protocol_sha256=digest(p/'phase118_conditional_recovery/protocol.json'),
        source_cache_contract_sha256=digest(directory/'cache_contract.json'),encoder_sha256=digest(directory/'encoder.pt'),runtime=runtime_identity(),
        preprocessing=preprocessing_contract(Path(__file__).parents[2],'dinov2_large'),selected_tables=selected,input_sha256=[descriptors[key]['input_sha256'] for key in selected],shape=[len(selected),1024],
        tf32=False,fused_attention=False,ad_independent=False,olid_independent=False,acceptance_data_used=False,
        scope='Canonical actual detector ROIs, center and full views; publisher image labels only, no clinical per-leaf labels')
    if (out/'complete.json').exists():
        result=read(out/'complete.json');assert read(out/'protocol.json')==protocol and result['protocol_sha256']==digest(out/'protocol.json') and result['features_sha256']==digest(out/'features.npy');return
    freeze(out/'protocol.json',protocol)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;timm.layers.set_fused_attn(False)
    old_rows=read(source/'cnn_inputs/rows.json');old_lookup={r['path']:i for i,r in enumerate(old_rows) if r.get('box') is None and r.get('input_variant')=='full_roi'}
    old_pixels=open_pixels(source/'cnn_inputs');old_features=np.load(directory/'features.npy',mmap_mode='r');pixels=CanonicalInputs(p)
    features=feature_memmap(out/'features.npy',protocol['shape']);progress=read(out/'progress.json') if (out/'progress.json').exists() else dict(completed=0,reused=0,new_encoder_inputs=0)
    if (out/'progress.json').exists():assert progress['protocol_sha256']==digest(out/'protocol.json')
    state=torch.load(directory/'encoder.pt',weights_only=True,map_location='cpu');model=DinoLargeEncoder(state['spec']).cuda().eval().requires_grad_(False);model.load_state_dict(state['model'],strict=True);del state
    audit_errors=[]
    try:
        with torch.inference_mode():
            for start in range(progress['completed'],len(selected),16):
                batch=[];missing=[];reuse=[]
                for offset,key in enumerate(selected[start:start+16]):
                    row=descriptors[key];row=row if 'kind' in row else dict(row,kind='new',policy=row['view']);value=pixels(row);old=old_lookup.get(row.get('path'))
                    if old is not None and np.array_equal(value[0],old_pixels[old]):
                        features[start+offset]=old_features[old];reuse.append((offset,value,old))
                    else:batch.append(value);missing.append(offset)
                if missing:
                    raw=model(torch.from_numpy(np.concatenate(batch)).cuda()).cpu().numpy();assert np.isfinite(raw).all();features[np.array(missing)+start]=raw
                if reuse and not audit_errors:
                    raw=model(torch.from_numpy(np.concatenate([r[1] for r in reuse[:4]])).cuda()).cpu().numpy()
                    error=float(np.abs(raw-old_features[[r[2] for r in reuse[:4]]]).max());assert error<.002,('Reused encoder graph/cache differs',error);audit_errors.append(error)
                features.flush();progress.update(completed=start+len(missing)+len(reuse),reused=progress['reused']+len(reuse),new_encoder_inputs=progress['new_encoder_inputs']+len(missing),protocol_sha256=digest(out/'protocol.json'))
                tmp=out/'progress.tmp';tmp.write_text(json.dumps(progress));tmp.replace(out/'progress.json')
                print(json.dumps(dict(stage='large_roi_features',**progress,total=len(selected))),flush=True)
    finally:
        pixels.close()
        if hasattr(old_pixels,'close'):old_pixels.close()
    result=dict(protocol_sha256=digest(out/'protocol.json'),features_sha256=digest(out/'features.npy'),**{k:v for k,v in progress.items() if k!='protocol_sha256'},
        reused_graph_audit_errors=audit_errors,elapsed_seconds=time.monotonic()-started,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    (out/'complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
